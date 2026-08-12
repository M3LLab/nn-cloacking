"""Generate 50×50 microstructures for each cloak cell with the 2D conditional
diffusion model (``v1_ortho``) instead of neural-field inverse design.

This is the diffusion-model counterpart of ``cell_inverse_design_sweep.py``.
For every cloak cell of a cloaking optimisation run it:

  1. reads the per-cell target stiffness [C11, C22, C12, C66] and density rho
     from ``optimized_params.npz`` (same source as the inverse-design sweep),
  2. conditions the diffusion model on (C11, C22, C12, C66, vol) — where
     vol = rho / rho_solid — and samples ``--num-seeds`` candidate cells
     (best-of-N: each batch element is an independent noise seed),
  3. homogenises every candidate with the SAME periodic-FEM setup used by the
     inverse-design sweep (``build_homog_setup`` + ``compute_flat4`` /
     ``compute_rho_eff``) so the realised stiffness is directly comparable,
  4. keeps the candidate with the lowest average-component relative error over
     [C11, C22, C12, C66, rho],
  5. writes each chosen cell to ``cell_XXX/{canvas.npy, weights.npz, cell.png}``
     in the SAME layout the inverse-design sweep uses, so
     ``frequency_sweep_validated.py --cell-designs`` can tile them directly.

The ``weights.npz`` only carries ``target_flat4 / pred_flat4 / target_rho /
pred_rho`` (there is no neural field) — exactly the keys
``build_canvas_from_cell_designs`` and ``compare_nn_vs_inverse.py`` consume.

Two phases keep the GPU uncontended: all diffusion sampling (torch/CUDA) runs
first, then the model is freed and the jax-fem homogenisation runs.

Usage
-----
    python scripts/cell_diffusion_design_sweep.py output/cell20_phys_flat4_materialreg4 \\
        --ckpt output/diffusion_ca_squared/v1_ortho/best.ckpt --num-seeds 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# jax-fem homogenisation runs alongside torch on the same box; don't let jax
# preallocate the whole GPU and starve the diffusion sampler.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
# Phase-2 hifi homogenisation is scipy/BLAS; we parallelise across cells with a
# process pool, so keep each worker single-threaded to avoid oversubscription.
# Must be set before numpy/scipy import (below) so BLAS honours it.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.stiffness.calc_fem import RHO_CEMENT
from dataset.cellular_chiral.inverse_design import (
    build_homog_setup,
    compute_flat4,
    compute_rho_eff,
    save_design,
)
# Static hi-fi homogeniser (TRI6 @ N=100 by default) — the SAME one used to
# label the ca_squared_2m dataset (stiffness.h5 attrs: homog_method='hifi',
# ele_type='TRI6', mesh_N=100, elem_per_pixel=2). Using it here keeps the
# diffusion cells' realised (C11,C22,C12,C66,rho) on the dataset's footing.
from dataset.stiffness.calc_fem_hifi import compute_stiffness_hifi
from dataset.cellular_chiral.bulk_stiffness import _ortho_params_from_C

from scripts.cell_inverse_design_sweep import (
    _FLAT4_LABELS,
    get_cloak_mask,
    opt_flat4_to_inv_flat4,
)

import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── diffusion sampling (phase 1) ─────────────────────────────────────


def sample_all_cells(
    ckpt: Path,
    scaler_dir: Path,
    targets_flat4: np.ndarray,   # (n, 4) [C11, C22, C12, C66]
    targets_vol: np.ndarray,     # (n,)
    num_seeds: int,
    steps: int,
    tensor_w: float,
    seed: int,
    device: str,
    compressed: bool | None,
) -> np.ndarray:
    """Sample ``num_seeds`` candidate 50×50 cells per target with the diffusion
    model. Returns (n_targets, num_seeds, 50, 50) uint8. Releases the model and
    CUDA cache before returning so the jax-fem phase has the GPU to itself.
    """
    import torch

    from microstructure_generation_2d.generate import _scale_targets, _decode_to_50
    from microstructure_generation_2d.network.model_trainer import DiffusionModel

    device = device if (torch.cuda.is_available() or device == "cpu") else "cpu"
    torch.manual_seed(seed)

    print(f"Loading diffusion checkpoint {ckpt} on {device} ...")
    module = DiffusionModel.load_from_checkpoint(str(ckpt), map_location=device).to(device)
    module.eval()
    generator = module.ema_model
    generator.eval()

    compressed = (
        compressed if compressed is not None
        else bool(module.hparams.get("compressed", False))
    )
    print(f"  compressed={compressed}  num_seeds={num_seeds}  steps={steps}  "
          f"tensor_w={tensor_w}")

    n = len(targets_vol)
    out = np.empty((n, num_seeds, 50, 50), dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(n):
        row = {
            "C11": float(targets_flat4[i, 0]),
            "C22": float(targets_flat4[i, 1]),
            "C12": float(targets_flat4[i, 2]),
            "C66": float(targets_flat4[i, 3]),
            "vol": float(targets_vol[i]),
        }
        tensor_c = _scale_targets(row, scaler_dir)
        img = generator.sample_with_tensor(
            tensor_c=tensor_c,
            batch_size=num_seeds,
            steps=steps,
            tensor_w=tensor_w,
            verbose=False,
        )
        arr = img.detach().cpu().numpy().squeeze(1)        # (num_seeds, H, H)
        bin_raw = (arr > 0).astype(np.uint8)
        out[i] = _decode_to_50(bin_raw, compressed=compressed)   # (num_seeds, 50, 50)
        if (i + 1) % 10 == 0 or i == n - 1:
            dt = time.perf_counter() - t0
            print(f"  sampled {i + 1}/{n} targets  ({dt:.1f}s, {dt / (i + 1):.2f}s/target)")

    del module, generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ── candidate evaluation / selection (phase 2) ───────────────────────


def _avg_comp_err(pf: np.ndarray, pr: float, tgt_flat4: np.ndarray, tgt_rho: float) -> float:
    """Average relative error over [C11, C22, C12, C66, rho] (selection metric,
    mirrors the best-of-NN guard in cell_inverse_design_sweep)."""
    e4 = np.abs((pf - tgt_flat4) / (np.abs(tgt_flat4) + 1e-30))
    er = abs((pr - tgt_rho) / (abs(tgt_rho) + 1e-30))
    return float((e4.sum() + er) / 5.0)


def select_best_candidate(
    candidates: np.ndarray,        # (num_seeds, 50, 50) uint8
    target_flat4: np.ndarray,
    target_rho: float,
    setup,
) -> dict:
    """Homogenise each candidate and return the best one's canvas + realised
    stiffness/density + per-seed errors."""
    import jax.numpy as jnp

    preds_flat4, preds_rho, errs = [], [], []
    for cand in candidates:
        cj = jnp.asarray(cand, dtype=jnp.float32)
        pf = np.asarray(compute_flat4(cj, setup))
        pr = float(compute_rho_eff(cj, setup))
        preds_flat4.append(pf)
        preds_rho.append(pr)
        errs.append(_avg_comp_err(pf, pr, target_flat4, target_rho))

    best = int(np.argmin(errs))
    return {
        "best_idx": best,
        "canvas": candidates[best].astype(np.uint8),
        "pred_flat4": preds_flat4[best],
        "pred_rho": preds_rho[best],
        "seed_errs": np.array(errs),
    }


# ── hi-fi (TRI6) homogenisation, parallel best-of-N (phase 2, default) ─


def _homog_one_hifi(canvas: np.ndarray, N: int, ele_type: str) -> np.ndarray:
    """Static hi-fi homogenise one 50×50 binary cell → [C11, C22, C12, C66, rho].

    Module-level (picklable) so it runs in a multiprocessing worker. Uses the
    exact dataset homogeniser (compute_stiffness_hifi + _ortho_params_from_C)."""
    C_eff, rho_eff, _vf = compute_stiffness_hifi(
        np.asarray(canvas, dtype=np.uint8), N=N, ele_type=ele_type,
    )
    c11, c22, c12, c66 = _ortho_params_from_C(C_eff)
    return np.array([c11, c22, c12, c66, rho_eff], dtype=np.float64)


def select_best_candidates_hifi(
    candidates: np.ndarray,        # (n_cells, num_seeds, 50, 50) uint8
    targets_flat4: np.ndarray,     # (n_cells, 4)
    targets_rho: np.ndarray,       # (n_cells,)
    N: int,
    ele_type: str,
    workers: int,
) -> list[dict]:
    """Homogenise EVERY candidate (all cells × all seeds) with the static hi-fi
    TRI6 homogeniser in a process pool, then keep the lowest-error seed per cell.

    Returns a list (len n_cells) of the same dicts ``select_best_candidate``
    produces, so the save/plot path is unchanged.
    """
    import multiprocessing as mp
    from functools import partial

    m, ns = candidates.shape[:2]
    flat = candidates.reshape(m * ns, *candidates.shape[2:])   # (m*ns, 50, 50)
    fn = partial(_homog_one_hifi, N=N, ele_type=ele_type)

    print(f"  homogenising {m * ns} candidates "
          f"({ele_type} N={N}) across {workers} workers ...")
    t0 = time.perf_counter()
    ctx = mp.get_context("spawn")   # avoid CUDA-in-forked-process issues
    preds = [None] * (m * ns)
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap(fn, list(flat), chunksize=4)):
            preds[i] = r
            if (i + 1) % 100 == 0 or i == m * ns - 1:
                dt = time.perf_counter() - t0
                print(f"    {i + 1}/{m * ns}  ({dt:.0f}s, {dt / (i + 1):.2f}s/solve)")
    preds = np.asarray(preds).reshape(m, ns, 5)                # [...,:4]=flat4, [...,4]=rho

    sels: list[dict] = []
    for j in range(m):
        pf, pr = preds[j, :, :4], preds[j, :, 4]
        errs = np.array([
            _avg_comp_err(pf[s], float(pr[s]), targets_flat4[j], float(targets_rho[j]))
            for s in range(ns)
        ])
        best = int(np.argmin(errs))
        sels.append({
            "best_idx": best,
            "canvas": candidates[j, best].astype(np.uint8),
            "pred_flat4": pf[best],
            "pred_rho": float(pr[best]),
            "seed_errs": errs,
        })
    return sels


# ── visualization ────────────────────────────────────────────────────


def _make_cell_figure(
    canvas: np.ndarray,
    target_flat4: np.ndarray,
    pred_flat4: np.ndarray,
    target_rho: float,
    pred_rho: float,
    cell_idx: int,
    seed_errs: np.ndarray,
    best_idx: int,
) -> plt.Figure:
    """3-panel figure: chosen cell | per-seed selection error | per-component error."""
    fig = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1.4], wspace=0.35)

    ax_cell = fig.add_subplot(gs[0])
    ax_cell.imshow(1.0 - canvas.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest")
    vf = float(canvas.mean())
    ax_cell.set_title(f"Cell {cell_idx}  (vf = {vf:.3f}, seed {best_idx})", fontsize=9)
    ax_cell.axis("off")

    ax_seed = fig.add_subplot(gs[1])
    colors = ["#2ecc71" if i == best_idx else "#95a5a6" for i in range(len(seed_errs))]
    ax_seed.bar(range(len(seed_errs)), seed_errs * 100.0, color=colors,
                edgecolor="k", linewidth=0.4)
    ax_seed.set_xlabel("Seed", fontsize=8)
    ax_seed.set_ylabel("Avg-comp rel err (%)", fontsize=8)
    ax_seed.set_title(f"Best-of-{len(seed_errs)} selection "
                      f"(chosen {seed_errs[best_idx] * 100:.2f}%)", fontsize=9)
    ax_seed.tick_params(labelsize=7)
    ax_seed.grid(True, axis="y", alpha=0.3)

    ax_err = fig.add_subplot(gs[2])
    labels = _FLAT4_LABELS + ["rho"]
    targets = np.append(target_flat4, target_rho)
    preds = np.append(pred_flat4, pred_rho)
    rel_err = (preds - targets) / (np.abs(targets) + 1e-30) * 100.0
    bar_colors = ["#e74c3c" if abs(e) > 10 else "#f39c12" if abs(e) > 1 else "#2ecc71"
                  for e in rel_err]
    bars = ax_err.barh(labels, rel_err, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax_err.axvline(0, color="k", lw=0.8)
    ax_err.set_xlabel("Relative error  (%)", fontsize=8)
    ax_err.set_title("Predicted vs target", fontsize=9)
    ax_err.tick_params(labelsize=7)
    ax_err.grid(True, axis="x", alpha=0.3)
    for bar, err in zip(bars, rel_err):
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        offset = max(abs(rel_err)) * 0.03 * (1 if w >= 0 else -1)
        ax_err.text(w + offset, bar.get_y() + bar.get_height() / 2,
                    f"{err:+.2f}%", va="center", ha=ha, fontsize=6.5)

    tgt_str = "  ".join(f"{l}={t:.2e}" for l, t in zip(labels, targets))
    fig.text(0.5, -0.03, f"Target:  {tgt_str}", ha="center", fontsize=6.5, style="italic")
    fig.suptitle(f"Cell {cell_idx} diffusion design", fontsize=10, y=1.01)
    return fig


def _make_summary_figure(summary: list[dict], out_path: Path) -> None:
    if not summary:
        return
    n = len(summary)
    labels = _FLAT4_LABELS + ["rho"]
    rel_errs = np.array([
        np.append(
            (s["pred_flat4"] - s["target_flat4"]) / (np.abs(s["target_flat4"]) + 1e-30),
            (s["pred_rho"] - s["target_rho"]) / (abs(s["target_rho"]) + 1e-30),
        ) * 100
        for s in summary
    ])
    fig, axes = plt.subplots(1, 5, figsize=(14, max(3, n * 0.18 + 1)), sharey=True)
    cmap = plt.cm.RdYlGn
    lim = max(30, float(np.abs(rel_errs).max()) * 1.1)
    for ax, lbl, col in zip(axes, labels, rel_errs.T):
        norm = plt.Normalize(-lim, lim)
        colors = [cmap(norm(v)) for v in col]
        ax.barh(range(n), col, color=colors, edgecolor="none", height=0.8)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_title(lbl, fontsize=9)
        ax.set_xlabel("Rel err (%)", fontsize=7)
        ax.set_xlim(-lim, lim)
        ax.tick_params(labelsize=6)
        ax.grid(True, axis="x", alpha=0.3)
    axes[0].set_yticks(range(n))
    axes[0].set_yticklabels([str(s["cell_idx"]) for s in summary], fontsize=5)
    axes[0].set_ylabel("Cell index", fontsize=8)
    fig.suptitle("Per-cell relative errors: diffusion predicted vs target (%)", fontsize=10)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary → {out_path}")


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("opt_dir", type=Path,
                        help="Cloaking optimization output directory")
    parser.add_argument("--ckpt", type=Path,
                        default=Path("output/diffusion_ca_squared/v1_ortho/best.ckpt"),
                        help="Diffusion model checkpoint (default v1_ortho/best.ckpt)")
    parser.add_argument("--scaler-dir", type=Path,
                        default=Path("microstructure_generation_2d"),
                        help="Directory with scaler_C11/.../scaler_C66 joblib files")
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: <opt_dir>/cell_designs_diffusion)")
    parser.add_argument("--start", type=int, default=0,
                        help="Index into the cloak-cell list to start from")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of cloak cells to process (default: all)")
    parser.add_argument("--num-seeds", type=int, default=16,
                        help="Candidate cells sampled per target; the closest one "
                             "to the target stiffness/density is kept (default 16)")
    parser.add_argument("--steps", type=int, default=50,
                        help="Diffusion sampling steps (default 50)")
    parser.add_argument("--tensor-w", type=float, default=2.0,
                        help="Classifier-free guidance scale (default 2.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base torch RNG seed for sampling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=None,
                        help="Override checkpoint's compressed flag (v1_ortho=False).")
    parser.add_argument("--simp-p", type=float, default=3.0,
                        help="SIMP exponent for homogenisation eval (default 3; inert "
                             "on a binary {0,1} canvas, matches the inverse-design sweep)")
    parser.add_argument("--homog", choices=("hifi", "legacy"), default="hifi",
                        help="Phase-2 homogeniser for selection & realised (C,rho). "
                             "'hifi' = static TRI6 @ N=100 (default) — the SAME "
                             "homogeniser that labelled ca_squared_2m, so pred is on "
                             "the dataset footing. 'legacy' = the old jax-fem TRI3 "
                             "dynamic homogenisation at f_star.")
    parser.add_argument("--homog-N", type=int, default=100,
                        help="hifi mesh resolution N (N=100 on a 50-px cell = 2 el/pix).")
    parser.add_argument("--homog-ele", choices=("TRI6", "TRI3"), default="TRI6",
                        help="hifi element type (default TRI6, matches the dataset).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="Process-pool size for parallel hifi homogenisation "
                             "(default: n_cpus-1).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose output already contains canvas.npy")
    args = parser.parse_args()

    opt_dir: Path = args.opt_dir
    params_path = opt_dir / "optimized_params.npz"
    config_path = opt_dir / "config.yaml"
    for p in (params_path, config_path, args.ckpt):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    # ── load optimization results ─────────────────────────────────────
    data = np.load(str(params_path))
    cell_C_flat_raw = data["cell_C_flat"]    # (n_cells, 4): optimizer ordering
    cell_rho = data["cell_rho"]               # (n_cells,)
    n_cells_total = len(cell_C_flat_raw)

    # Permute to [C11, C22, C12, C66] (compute_flat4 / diffusion ordering)
    cell_flat4 = opt_flat4_to_inv_flat4(cell_C_flat_raw)

    import yaml
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    f_star = float(cfg_raw["domain"]["f_star"])

    # ── cloak mask & slice selection ──────────────────────────────────
    print("Building cloak mask from geometry...")
    cloak_mask = get_cloak_mask(opt_dir)
    cloak_indices = np.where(cloak_mask)[0]
    print(f"  {len(cloak_indices)} / {n_cells_total} cells are inside the cloak")

    start = args.start
    end = len(cloak_indices) if args.num is None else min(len(cloak_indices), start + args.num)
    indices_to_process = cloak_indices[start:end]
    print(f"Processing cells {start}:{end} of cloak list → {len(indices_to_process)} cells")

    out_dir = args.out_dir or (opt_dir / "cell_designs_diffusion")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── resume filtering ──────────────────────────────────────────────
    rho_solid = RHO_CEMENT
    pending = []
    for ci in indices_to_process:
        canvas_path = out_dir / f"cell_{int(ci):03d}" / "canvas.npy"
        if args.resume and canvas_path.exists():
            print(f"  [cell {int(ci):03d}] already done – skipping.")
            continue
        pending.append(int(ci))

    if not pending:
        print("Nothing to do (all cells already done).")
    else:
        targets_flat4 = np.stack([cell_flat4[ci] for ci in pending])         # (m, 4)
        targets_rho = np.array([float(cell_rho[ci]) for ci in pending])      # (m,)
        targets_vol = targets_rho / rho_solid                               # vf ≈ rho/rho_solid

        # ── PHASE 1: diffusion sampling (torch / CUDA) ────────────────
        print(f"\n=== Phase 1: diffusion sampling ({len(pending)} cells × "
              f"{args.num_seeds} seeds) ===")
        candidates = sample_all_cells(
            ckpt=args.ckpt, scaler_dir=args.scaler_dir,
            targets_flat4=targets_flat4, targets_vol=targets_vol,
            num_seeds=args.num_seeds, steps=args.steps, tensor_w=args.tensor_w,
            seed=args.seed, device=args.device, compressed=args.compressed,
        )

        # ── PHASE 2: homogenise + select ──────────────────────────────
        print(f"\n=== Phase 2: {args.homog} homogenisation & "
              f"best-of-{args.num_seeds} selection ===")
        if args.homog == "hifi":
            # Static TRI6 @ N=100 — dataset-compatible; parallel over candidates.
            sels = select_best_candidates_hifi(
                candidates, targets_flat4, targets_rho,
                N=args.homog_N, ele_type=args.homog_ele, workers=args.workers,
            )
            setup = None
        else:
            print(f"Building FEM setup (f_star={f_star} Hz, simp_p={args.simp_p})...")
            setup = build_homog_setup(canvas_N=50, f_star=f_star, simp_p=args.simp_p,
                                      rho_solid=rho_solid)
            print("FEM setup ready.\n")
            sels = None

        for j, cell_idx in enumerate(pending):
            t0 = time.perf_counter()
            tgt4 = targets_flat4[j]
            tgt_rho = targets_rho[j]
            sel = sels[j] if sels is not None else \
                select_best_candidate(candidates[j], tgt4, tgt_rho, setup)
            pred_flat4, pred_rho = sel["pred_flat4"], sel["pred_rho"]

            cell_dir = out_dir / f"cell_{cell_idx:03d}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            np.save(str(cell_dir / "canvas.npy"), sel["canvas"])
            save_design(
                str(cell_dir / "weights"),
                [], None,
                target_flat4=tgt4,
                pred_flat4=pred_flat4,
                target_rho=np.array(tgt_rho),
                pred_rho=np.array(pred_rho),
                seed_errs=sel["seed_errs"],
            )
            fig = _make_cell_figure(
                canvas=sel["canvas"], target_flat4=tgt4, pred_flat4=pred_flat4,
                target_rho=tgt_rho, pred_rho=pred_rho, cell_idx=cell_idx,
                seed_errs=sel["seed_errs"], best_idx=sel["best_idx"],
            )
            fig.savefig(str(cell_dir / "cell.png"), dpi=120, bbox_inches="tight")
            plt.close(fig)

            rel_err = (pred_flat4 - tgt4) / (np.abs(tgt4) + 1e-30) * 100
            rho_err = (pred_rho - tgt_rho) / (abs(tgt_rho) + 1e-30) * 100
            print(f"  [cell {cell_idx:03d}] {j + 1}/{len(pending)}  "
                  f"best seed {sel['best_idx']:2d}  "
                  f"C11={rel_err[0]:+.1f} C22={rel_err[1]:+.1f} "
                  f"C12={rel_err[2]:+.1f} C66={rel_err[3]:+.1f} rho={rho_err:+.1f}  "
                  f"({time.perf_counter() - t0:.1f}s)")

    # ── aggregate summary across ALL completed cells ──────────────────
    summary: list[dict] = []
    for ci in indices_to_process:
        wpath = out_dir / f"cell_{int(ci):03d}" / "weights.npz"
        if not wpath.exists():
            continue
        d = np.load(str(wpath))
        if "pred_flat4" not in d:
            continue
        summary.append({
            "cell_idx": int(ci),
            "target_flat4": np.asarray(d["target_flat4"]),
            "pred_flat4": np.asarray(d["pred_flat4"]),
            "target_rho": float(d["target_rho"]),
            "pred_rho": float(d["pred_rho"]),
        })

    if summary:
        _make_summary_figure(summary, out_dir / "summary_errors.png")
        avg = np.mean([_avg_comp_err(s["pred_flat4"], s["pred_rho"],
                                     s["target_flat4"], s["target_rho"])
                       for s in summary])
        print(f"\nMean avg-component rel err across {len(summary)} cells: {avg * 100:.2f}%")

    canvas_files = sorted(out_dir.glob("cell_*/canvas.npy"))
    if canvas_files:
        canvases = np.stack([np.load(str(p)) for p in canvas_files])
        np.save(str(out_dir / "all_canvases.npy"), canvases)
        print(f"Aggregated {len(canvases)} canvases → {out_dir / 'all_canvases.npy'}")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
