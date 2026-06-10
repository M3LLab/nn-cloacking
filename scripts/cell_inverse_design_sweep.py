"""Generate 50×50 CA microstructures matching per-cell stiffness targets from a
cloaking optimization run.

Only cells inside the cloak region are processed (determined from the geometry
in config.yaml via CellDecomposition).

For each cloak cell a neural field is optimized so that the periodic-FEM effective
stiffness [C11, C22, C12, C66] and density rho of the assembled 50×50
microstructure match the target.  Optimization stops when every component's
relative error is below ``--tol`` (default 0.001) or after ``--n-iters`` steps
(default 350), whichever comes first.

Defaults encode the tuned config: dataset NN init, connectivity penalty OFF
(--weight-conn 0), beta projection 1→32 with a straight-through estimator in the
hardened tail (--straight-through, on by default), lr 2e-3→3e-4.  This took
cell_014 from ~10–24% to ~3% binary-matching error; see the per-cell summary.

Flat4 index mapping (n_C_params=4, C_to_flatC in materials.py):
    optimized_params index 0 → C_1111 = C11
    optimized_params index 1 → C_2222 = C22
    optimized_params index 2 → C_1212 = C66 (shear)
    optimized_params index 3 → C_1122 = C12 (lateral coupling)
Permuted to [C11, C22, C12, C66] = indices [0, 1, 3, 2] for compute_flat4.

Usage
-----
    python scripts/cell_inverse_design_sweep.py output/cell20_flat4_materialreg_pt2

    # Process a slice for parallel runs
    python scripts/cell_inverse_design_sweep.py output/cell20_flat4_materialreg_pt2 \\
        --start 0 --num 20

    # Resume interrupted run
    python scripts/cell_inverse_design_sweep.py output/cell20_flat4_materialreg_pt2 --resume
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.stiffness.calc_fem import RHO_CEMENT
from dataset.cellular_chiral.inverse_design import (
    CellDesignResult,
    HomogSetup,
    build_homog_setup,
    compute_flat4,
    compute_rho_eff,
    load_design,
    make_cell_neural_field,
    run_cell_design,
    save_design,
)

import logging
from dataclasses import dataclass

import h5py

logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── flat4 component labels ────────────────────────────────────────────

_FLAT4_LABELS = ["C11", "C22", "C12", "C66"]


# ── dataset nearest-neighbour matching ───────────────────────────────

@dataclass
class DatasetCache:
    """Pre-loaded dataset features for fast nearest-neighbour lookup."""
    X: np.ndarray     # (n, 5) raw [C11, C22, C12, C66, rho]
    h5_path: Path
    n: int


def load_dataset_for_matching(h5_path: Path) -> DatasetCache:
    """Load scalar features from stiffness.h5.

    Features: [C11, C22, C12, C66, rho] — the same quantities we optimise,
    kept raw (the NN metric is relative, applied per query).
    """
    print(f"Loading dataset features from {h5_path} ...")
    with h5py.File(str(h5_path), "r") as f:
        C11 = f["C11"][:]
        C22 = f["C22"][:]
        C12 = f["C12"][:]
        C66 = f["C66"][:]
        rho = f["rho"][:]
    X = np.column_stack([C11, C22, C12, C66, rho]).astype(np.float64)
    print(f"  {len(C11)} dataset cells loaded.")
    return DatasetCache(X=X, h5_path=h5_path, n=len(C11))


def find_nn_quadrant(
    target_flat4: np.ndarray,
    target_rho: float,
    ds: DatasetCache,
    rho_weight: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find the dataset cell nearest to (target_flat4, target_rho).

    Returns (quadrant_25x25, nn_flat4, nn_rho) where nn_flat4 and nn_rho are
    the dataset-stored values for the matched cell (computed with dataset FEM
    settings, which may differ from the inverse-design FEM settings).

    Matching uses a RELATIVE metric — sum_j w_j ((X_j - q_j)/|q_j|)^2 over
    [C11, C22, C12, C66, rho] — mirroring the inverse-design loss.  A global
    z-score metric is wrong here: the dataset std is dominated by stiff cells
    (~1e9), so soft-region (~1e8) stiffness differences become negligible and
    C22/C66 silently drop out of the match.  ``rho_weight`` rescales the rho
    term (>1 → more rho-sensitive).
    """
    q = np.array([target_flat4[0], target_flat4[1],
                  target_flat4[2], target_flat4[3], float(target_rho)],
                 dtype=np.float64)

    weights = np.array([1.0, 1.0, 1.0, 1.0, rho_weight])
    rel = (ds.X - q[None, :]) / (np.abs(q)[None, :] + 1e-30)   # relative error
    d2  = np.sum(rel ** 2 * weights[None, :], axis=1)
    nn_idx = int(np.argmin(d2))

    nn_params = ds.X[nn_idx]                       # [C11, C22, C12, C66, rho]
    nn_flat4 = nn_params[:4].astype(np.float64)
    nn_rho   = float(nn_params[4])

    with h5py.File(str(ds.h5_path), "r") as f:
        cell_50x50 = np.asarray(f["cells"][nn_idx])   # (50, 50) uint8

    # Top-left 25×25 is the original quadrant before squared assembly
    return cell_50x50[:25, :25], nn_flat4, nn_rho


# ── cloak mask from optimization config ──────────────────────────────

def get_cloak_mask(opt_dir: Path) -> np.ndarray:
    """Return (n_cells,) bool mask of cloak cells from the config in opt_dir.

    Builds the full CellDecomposition from the saved config.yaml so the
    mask exactly matches what the optimizer used.
    """
    from rayleigh_cloak import load_config
    from rayleigh_cloak.config import DerivedParams
    from rayleigh_cloak.cells import CellDecomposition
    from rayleigh_cloak.solver import _create_geometry

    cfg = load_config(str(opt_dir / "config.yaml"))
    dp = DerivedParams.from_config(cfg)
    geo = _create_geometry(cfg, dp)
    decomp = CellDecomposition(geo, int(cfg.cells.n_x), int(cfg.cells.n_y))
    return decomp.cloak_mask.astype(bool)


# ── flat4 permutation ─────────────────────────────────────────────────

def opt_flat4_to_inv_flat4(cell_C_flat: np.ndarray) -> np.ndarray:
    """Permute n_C_params=4 format [C1111, C2222, C1212, C1122]
    → compute_flat4 format [C11, C22, C12, C66] (indices [0, 1, 3, 2])."""
    return cell_C_flat[:, [0, 1, 3, 2]]


# ── visualization ─────────────────────────────────────────────────────

def _make_cell_figure(
    canvas: np.ndarray,
    target_flat4: np.ndarray,
    pred_flat4: np.ndarray,
    target_rho: float,
    pred_rho: float,
    cell_idx: int,
    loss_history: list[float],
) -> plt.Figure:
    """3-panel figure: cell image | loss curve | per-component relative error bars."""
    fig = plt.figure(figsize=(13, 4))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.2, 1.4], wspace=0.35)

    # ── panel 1: 50×50 cell image ──────────────────────────────────────
    ax_cell = fig.add_subplot(gs[0])
    # gray cmap: 0→black, 1→white. Invert so material(1)→black, void(0)→white.
    ax_cell.imshow(1.0 - canvas.astype(np.float32), cmap="gray", vmin=0, vmax=1,
                   interpolation="nearest")
    vf = float(canvas.mean())
    ax_cell.set_title(f"Cell {cell_idx}  (vf = {vf:.3f})", fontsize=9)
    ax_cell.axis("off")

    # ── panel 2: loss history ──────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[1])
    ax_loss.semilogy(loss_history, color="steelblue", lw=1.2)
    ax_loss.set_xlabel("Step", fontsize=8)
    ax_loss.set_ylabel("Loss", fontsize=8)
    ax_loss.set_title(f"Loss  (final {loss_history[-1]:.3e})", fontsize=9)
    ax_loss.tick_params(labelsize=7)
    ax_loss.grid(True, which="both", alpha=0.3)

    # ── panel 3: relative errors ──────────────────────────────────────
    ax_err = fig.add_subplot(gs[2])
    labels = _FLAT4_LABELS + ["rho"]
    targets = np.append(target_flat4, target_rho)
    preds = np.append(pred_flat4, pred_rho)
    rel_err = (preds - targets) / (np.abs(targets) + 1e-30) * 100.0

    colors = ["#e74c3c" if abs(e) > 10 else "#f39c12" if abs(e) > 1 else "#2ecc71"
              for e in rel_err]
    bars = ax_err.barh(labels, rel_err, color=colors, edgecolor="k", linewidth=0.5)
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
    fig.suptitle(f"Cell {cell_idx} inverse design", fontsize=10, y=1.01)
    return fig


# ── per-cell inverse design ───────────────────────────────────────────

def design_one_cell(
    cell_idx: int,
    target_flat4: np.ndarray,
    target_rho: float,
    setup: HomogSetup,
    out_dir: Path,
    n_iters: int,
    lr: float,
    lr_end: float,
    n_fourier: int,
    hidden_size: int,
    n_layers: int,
    seed: int,
    weight_rho: float,
    tol: float,
    resume: bool,
    weight_conn: float = 0.0,
    conn_steps: int = 200,
    beta_init: float = 1.0,
    beta_final: float = 32.0,
    beta_warmup_frac: float = 0.15,
    beta_ramp_frac: float = 0.25,
    straight_through: bool = True,
    ds_cache: DatasetCache | None = None,
    nn_rho_weight: float = 1.0,
) -> dict | None:
    """Run inverse design for one cell; return summary dict or None on skip."""
    import jax.numpy as jnp

    cell_dir = out_dir / f"cell_{cell_idx:03d}"
    canvas_path = cell_dir / "canvas.npy"
    weights_path = cell_dir / "weights.npz"
    img_path = cell_dir / "cell.png"

    if resume and canvas_path.exists() and weights_path.exists():
        print(f"  [cell {cell_idx:03d}] already done – skipping.")
        data = np.load(str(weights_path))
        return {
            "cell_idx": cell_idx,
            "target_flat4": target_flat4,
            "pred_flat4": data.get("pred_flat4", np.zeros(4)),
            "target_rho": target_rho,
            "pred_rho": float(data.get("pred_rho", 0.0)),
        }

    cell_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Cell {cell_idx:03d}  target=[{', '.join(f'{v:.3e}' for v in target_flat4)}]"
          f"  rho={target_rho:.1f}")

    # Nearest-neighbour initialisation
    initial_quadrant = None
    if ds_cache is not None:
        initial_quadrant, nn_flat4, nn_rho = find_nn_quadrant(
            target_flat4, target_rho, ds_cache, rho_weight=nn_rho_weight,
        )
        nn_vf = float(initial_quadrant.mean())
        nn_f4_str = "[" + ", ".join(f"{v:.3e}" for v in nn_flat4) + "]"
        nn_rel = np.abs((nn_flat4 - target_flat4) / (np.abs(target_flat4) + 1e-30))
        print(f"  NN init: quadrant vf={nn_vf:.3f}  rho={nn_rho:.1f}"
              f"  flat4={nn_f4_str}"
              f"  (max rel err vs target: {nn_rel.max():.2%})")

    # Build neural field (deterministic seed per cell)
    theta_init, nf = make_cell_neural_field(
        n_fourier=n_fourier, hidden_size=hidden_size,
        n_layers=n_layers, seed=seed + cell_idx,
        initial_quadrant=initial_quadrant,
    )

    t0 = time.perf_counter()
    result = run_cell_design(
        neural_field=nf,
        setup=setup,
        target_flat4=target_flat4,
        theta_init=theta_init,
        target_rho=target_rho,
        weight_rho=weight_rho,
        weight_conn=weight_conn,
        conn_steps=conn_steps,
        beta_init=beta_init,
        beta_final=beta_final,
        beta_warmup_frac=beta_warmup_frac,
        beta_ramp_frac=beta_ramp_frac,
        straight_through=straight_through,
        n_iters=n_iters,
        lr=lr,
        lr_end=lr_end,
        lr_schedule="cosine",
        tol=tol,
    )
    elapsed = time.perf_counter() - t0
    print(f"  Completed {len(result.loss_history)} steps in {elapsed:.1f}s")

    # Evaluate best weights.  The deliverable is the BINARIZED canvas, so the
    # reported prediction is computed on it (simp_p is irrelevant for a {0,1}
    # field).  The hardened soft canvas (beta=beta_final) is kept only as a
    # diagnostic; its gap to the binary prediction tells us how well the
    # beta-projection actually drove the field to binary.
    canvas_soft = nf.decode_canvas(result.best_theta, beta=beta_final)
    canvas_bin = nf.binarize(result.best_theta)
    canvas_bin_jnp = jnp.asarray(canvas_bin, dtype=jnp.float32)
    pred_flat4 = np.asarray(compute_flat4(canvas_bin_jnp, setup))
    pred_rho = float(compute_rho_eff(canvas_bin_jnp, setup))

    soft_flat4 = np.asarray(compute_flat4(jnp.asarray(canvas_soft), setup))
    soft_gap = np.abs((pred_flat4 - soft_flat4) / (np.abs(soft_flat4) + 1e-30))
    print(f"  soft→binary flat4 gap (max rel): {soft_gap.max():.2%}")

    # Save arrays
    np.save(str(canvas_path), canvas_bin)
    np.save(str(cell_dir / "canvas_soft.npy"), np.asarray(canvas_soft))
    save_design(
        str(cell_dir / "weights"),
        result.best_theta,
        result.opt_state,
        target_flat4=target_flat4,
        pred_flat4=pred_flat4,
        target_rho=np.array(target_rho),
        pred_rho=np.array(pred_rho),
        loss_history=np.array(result.loss_history),
    )

    # Save image
    fig = _make_cell_figure(
        canvas=canvas_bin,
        target_flat4=target_flat4,
        pred_flat4=pred_flat4,
        target_rho=target_rho,
        pred_rho=pred_rho,
        cell_idx=cell_idx,
        loss_history=result.loss_history,
    )
    fig.savefig(str(img_path), dpi=120, bbox_inches="tight")
    plt.close(fig)

    rel_err = (pred_flat4 - target_flat4) / (np.abs(target_flat4) + 1e-30) * 100
    rho_err = (pred_rho - target_rho) / (abs(target_rho) + 1e-30) * 100
    print(f"  Rel errors (%)  "
          f"C11={rel_err[0]:+.1f}  C22={rel_err[1]:+.1f}  "
          f"C12={rel_err[2]:+.1f}  C66={rel_err[3]:+.1f}  "
          f"rho={rho_err:+.1f}")

    return {
        "cell_idx": cell_idx,
        "target_flat4": target_flat4,
        "pred_flat4": pred_flat4,
        "target_rho": target_rho,
        "pred_rho": pred_rho,
    }


# ── summary figure ────────────────────────────────────────────────────

def _make_summary_figure(summary: list[dict], out_path: Path) -> None:
    """One subplot per component: horizontal bar of relative errors across cells."""
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
    ])   # (n_cells, 5)

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
    fig.suptitle("Per-cell relative errors: predicted vs target (%)", fontsize=10)
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
    parser.add_argument("-o", "--out-dir", type=Path, default=None,
                        help="Output directory (default: <opt_dir>/cell_designs)")
    parser.add_argument("--start", type=int, default=0,
                        help="Index into the cloak-cell list to start from")
    parser.add_argument("--num", type=int, default=None,
                        help="Number of cloak cells to process (default: all)")
    parser.add_argument("--n-iters", type=int, default=350,
                        help="Max Adam steps per cell (default 350, tuned config)")
    parser.add_argument("--tol", type=float, default=0.001,
                        help="Early-stop threshold: max relative error (default 0.001)")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--lr-end", type=float, default=3e-4)
    parser.add_argument("--n-fourier", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-rho", type=float, default=10.0)
    parser.add_argument("--weight-conn", type=float, default=0.0,
                        help="Weight for gate connectivity penalty (default 0 = off). "
                             "Nonzero values dominate the stiffness/rho matching terms and "
                             "wreck the match (conn loss ~0.8×weight vs ~0.13 matching); "
                             "the dataset NN init already supplies a connected topology.")
    parser.add_argument("--conn-steps", type=int, default=200,
                        help="Flood iterations for connectivity loss (default 200)")
    parser.add_argument("--beta-init", type=float, default=1.0,
                        help="Initial Heaviside-projection sharpness (default 1 = gray start, "
                             "allows soft refinement). Higher values (e.g. 8) saturate the "
                             "sigmoid → ~zero gradient → the field is frozen at the init and "
                             "cannot refine; only useful for diagnostics.")
    parser.add_argument("--beta-final", type=float, default=32.0,
                        help="Final Heaviside-projection sharpness (default 32). Higher → "
                             "soft field closer to binary, shrinking the soft→binary gap.")
    parser.add_argument("--beta-warmup-frac", type=float, default=0.15,
                        help="Fraction of iters held at beta_init before ramping (default 0.15)")
    parser.add_argument("--beta-ramp-frac", type=float, default=0.25,
                        help="Fraction of iters ramping beta_init→beta_final (default 0.25); "
                             "remaining tail held at beta_final (the straight-through phase)")
    parser.add_argument("--straight-through", action=argparse.BooleanOptionalAction, default=True,
                        help="Use a straight-through estimator in the hardened tail: optimize "
                             "the BINARIZED effective stiffness directly (closes soft→binary gap). "
                             "best_theta/early-stop then track the actual binary deliverable. "
                             "Default on; disable with --no-straight-through.")
    parser.add_argument("--simp-p", type=float, default=3.0,
                        help="SIMP exponent (default 3; penalizes gray during the soft phase "
                             "so the target stays binary-reachable. Inert once beta-projection "
                             "has driven the field to {0,1}.)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells whose output already contains canvas.npy")
    parser.add_argument("--dataset-path", type=Path,
                        default=Path("output/ca_bulk_squared/stiffness.h5"),
                        help="HDF5 dataset for nearest-neighbour initialisation")
    parser.add_argument("--no-nn-init", action="store_true",
                        help="Disable nearest-neighbour initialisation (start from sigmoid=0.5)")
    parser.add_argument("--nn-rho-weight", type=float, default=1.0,
                        help="Weight for rho in NN matching distance (default 1.0)")
    args = parser.parse_args()

    opt_dir: Path = args.opt_dir
    params_path = opt_dir / "optimized_params.npz"
    config_path = opt_dir / "config.yaml"

    for p in (params_path, config_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} not found")

    # ── load optimization results ─────────────────────────────────────
    data = np.load(str(params_path))
    cell_C_flat_raw = data["cell_C_flat"]   # (n_cells, 4): optimizer ordering
    cell_rho = data["cell_rho"]              # (n_cells,)
    n_cells_total = len(cell_C_flat_raw)

    import yaml
    with open(config_path) as f:
        cfg_raw = yaml.safe_load(f)
    f_star = float(cfg_raw["domain"]["f_star"])

    # Permute to [C11, C22, C12, C66]
    cell_flat4 = opt_flat4_to_inv_flat4(cell_C_flat_raw)

    # ── cloak mask (true cell decomposition from geometry) ────────────
    print("Building cloak mask from geometry...")
    cloak_mask = get_cloak_mask(opt_dir)
    cloak_indices = np.where(cloak_mask)[0]
    print(f"  {len(cloak_indices)} / {n_cells_total} cells are inside the cloak")

    # ── select slice of cloak cells ───────────────────────────────────
    start = args.start
    end = len(cloak_indices) if args.num is None else min(len(cloak_indices), start + args.num)
    indices_to_process = cloak_indices[start:end]
    print(f"Processing cells {start}:{end} of cloak list → {len(indices_to_process)} cells")

    # ── output directory ──────────────────────────────────────────────
    out_dir = args.out_dir or (opt_dir / "cell_designs")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load dataset for nearest-neighbour init ───────────────────────
    ds_cache: DatasetCache | None = None
    if not args.no_nn_init:
        if not args.dataset_path.exists():
            print(f"WARNING: dataset not found at {args.dataset_path}; "
                  f"falling back to random init (pass --no-nn-init to suppress).")
        else:
            ds_cache = load_dataset_for_matching(args.dataset_path)

    # ── build shared FEM setup ────────────────────────────────────────
    print(f"\nBuilding FEM setup (f_star={f_star} Hz, simp_p={args.simp_p})...")
    setup = build_homog_setup(
        canvas_N=50,
        f_star=f_star,
        simp_p=args.simp_p,
        rho_solid=RHO_CEMENT,
    )
    print("FEM setup ready.\n")

    # ── process each cloak cell ───────────────────────────────────────
    summary: list[dict] = []
    for cell_idx in indices_to_process:
        result = design_one_cell(
            cell_idx=int(cell_idx),
            target_flat4=cell_flat4[cell_idx],
            target_rho=float(cell_rho[cell_idx]),
            setup=setup,
            out_dir=out_dir,
            n_iters=args.n_iters,
            lr=args.lr,
            lr_end=args.lr_end,
            n_fourier=args.n_fourier,
            hidden_size=args.hidden_size,
            n_layers=args.n_layers,
            seed=args.seed,
            weight_rho=args.weight_rho,
            weight_conn=args.weight_conn,
            conn_steps=args.conn_steps,
            beta_init=args.beta_init,
            beta_final=args.beta_final,
            beta_warmup_frac=args.beta_warmup_frac,
            beta_ramp_frac=args.beta_ramp_frac,
            straight_through=args.straight_through,
            tol=args.tol,
            resume=args.resume,
            ds_cache=ds_cache,
            nn_rho_weight=args.nn_rho_weight,
        )
        if result is not None:
            summary.append(result)

    # ── summary outputs ───────────────────────────────────────────────
    if summary:
        _make_summary_figure(summary, out_dir / "summary_errors.png")

    canvas_files = sorted(out_dir.glob("cell_*/canvas.npy"))
    if canvas_files:
        canvases = np.stack([np.load(str(p)) for p in canvas_files])
        np.save(str(out_dir / "all_canvases.npy"), canvases)
        print(f"Aggregated {len(canvases)} canvases → {out_dir / 'all_canvases.npy'}")

    print(f"\nDone. Results in {out_dir}")


if __name__ == "__main__":
    main()
