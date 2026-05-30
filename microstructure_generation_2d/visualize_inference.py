"""Run the trained 2D conditional diffusion model on N dataset conditions and
plot the generated cells side-by-side with the dataset originals.

Conditions and ground-truth cells are pulled from ``stiffness.h5`` (defaults to
the validation indices in ``split.json``). For each condition the scaled 4-dim
vector (C11, C12, C66, vol) is reconstructed exactly as the training dataset
does, then fed to ``sample_with_tensor``.

Without ``--fem``: saves a single inference grid (current behaviour).
With ``--fem``: saves one figure per condition showing side-by-side images and
a bar-chart comparing dataset vs FEM-realised (C11, C12, C66, vol).

Usage:

    # grid only (fast)
    python -m microstructure_generation_2d.visualize_inference \\
        --ckpt output/diffusion_ca_squared/v1/last.ckpt \\
        --output output/diffusion_ca_squared/v1/inference_grid.png

    # per-condition FEM plots
    python -m microstructure_generation_2d.visualize_inference \\
        --ckpt output/diffusion_ca_squared/v2/last.ckpt \\
        --output output/diffusion_ca_squared/v2/inference_grid.png \\
        --fem --num 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import joblib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset.cellular_chiral.diffusion_dataset import (
    CELL_SIZE, PAD_TO, load_or_make_split, unfold_mirror,
)
from .network.model_trainer import DiffusionModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_64_to_50(arr: np.ndarray) -> np.ndarray:
    off = (PAD_TO - CELL_SIZE) // 2
    return arr[..., off : off + CELL_SIZE, off : off + CELL_SIZE]


def _decode_to_50(arr: np.ndarray, compressed: bool) -> np.ndarray:
    if compressed:
        return unfold_mirror(arr).astype(np.uint8)
    return _crop_64_to_50(arr)


def _pick_indices(split: dict, n: int, source: str, seed: int) -> list[int]:
    pool = split.get(source) or split["val"]
    pool = list(pool)
    if not pool:
        raise RuntimeError(f"No indices in split[{source!r}]")
    rng = np.random.default_rng(seed)
    if len(pool) <= n:
        return pool[:n]
    return rng.choice(pool, size=n, replace=False).tolist()


def _load_conditions(h5_path: str, scaler_dir: Path, indices: list[int]):
    scaler_C11 = joblib.load(scaler_dir / "scaler_C11")
    scaler_C12 = joblib.load(scaler_dir / "scaler_C12")
    scaler_C66 = joblib.load(scaler_dir / "scaler_C66")

    rows = []
    with h5py.File(h5_path, "r") as f:
        cells = f["cells"]
        C11 = f["C11"]
        C12 = f["C12"]
        C66 = f["C66"]
        vol = f["vol"]
        for h_idx in indices:
            c11 = float(C11[h_idx])
            c12 = float(C12[h_idx])
            c66 = float(C66[h_idx])
            v = float(vol[h_idx])
            tf = np.array([
                float(scaler_C11.transform([[c11]])[0, 0]),
                float(scaler_C12.transform([[c12]])[0, 0]),
                float(scaler_C66.transform([[c66]])[0, 0]),
                v,
            ], dtype=np.float32)
            rows.append({
                "idx": int(h_idx),
                "cell": np.asarray(cells[h_idx], dtype=np.uint8),  # (50, 50)
                "C11": c11, "C12": c12, "C66": c66, "vol": v,
                "tensor_feature": tf,
            })
    return rows


def _format_label(row: dict) -> str:
    return (
        f"idx={row['idx']}\n"
        f"C11={row['C11']:.2e}\n"
        f"C12={row['C12']:.2e}\n"
        f"C66={row['C66']:.2e}\n"
        f"vol={row['vol']:.3f}"
    )


# ---------------------------------------------------------------------------
# FEM
# ---------------------------------------------------------------------------

def _run_fem(cell_uint8: np.ndarray) -> dict:
    """FEM homogenization on a single (50, 50) uint8 cell. Lazy import."""
    from dataset.cellular_chiral.bulk_stiffness import (
        _compute_stiffness, _d4_params_from_C,
    )
    C, _rho, vf = _compute_stiffness(cell_uint8.astype(np.int8))
    C11, C12, C66 = _d4_params_from_C(C)
    return {"C11": C11, "C12": C12, "C66": C66, "vf": vf}


# ---------------------------------------------------------------------------
# Per-condition FEM figure
# ---------------------------------------------------------------------------

def _bar_pair(ax, ds_val: float, gen_val: float, label: str, fmt: str) -> None:
    """Draw a 2-bar (Dataset / Generated) chart on ax with value annotations."""
    colors = ["steelblue", "coral"]
    vals = [ds_val, gen_val]
    bars = ax.bar([0, 1], vals, color=colors, width=0.55, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DS", "Gen"], fontsize=8)
    ax.set_title(label, fontsize=9, pad=3)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(axis="y", linewidth=0.4, alpha=0.5, zorder=0)

    for bar, val in zip(bars, vals):
        x_pos = bar.get_x() + bar.get_width() / 2
        va = "bottom" if val >= 0 else "top"
        offset = ax.get_ylim()[1] * 0.02 if val >= 0 else ax.get_ylim()[0] * 0.02
        ax.text(x_pos, val, fmt % val, ha="center", va=va,
                fontsize=6.5, rotation=0)

    # Ensure y-axis has breathing room for annotations
    lo, hi = ax.get_ylim()
    span = hi - lo or 1.0
    ax.set_ylim(lo - 0.15 * span, hi + 0.15 * span)


def _plot_fem_figure(row: dict, i: int, out_dir: Path) -> Path:
    """One figure per condition: images (top) + 4 param bars (bottom)."""
    fig = plt.figure(figsize=(11, 6))
    gs_top = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.03, right=0.45, top=0.88, bottom=0.38,
        wspace=0.08,
    )
    gs_bot = gridspec.GridSpec(
        1, 4, figure=fig,
        left=0.06, right=0.97, top=0.32, bottom=0.08,
        wspace=0.45,
    )

    ax_ds  = fig.add_subplot(gs_top[0, 0])
    ax_gen = fig.add_subplot(gs_top[0, 1])
    ax_c11 = fig.add_subplot(gs_bot[0, 0])
    ax_c12 = fig.add_subplot(gs_bot[0, 1])
    ax_c66 = fig.add_subplot(gs_bot[0, 2])
    ax_vol = fig.add_subplot(gs_bot[0, 3])

    # --- images ---
    for ax, cell, title in (
        (ax_ds,  row["cell"],     "Dataset"),
        (ax_gen, row["gen_cell"], "Generated"),
    ):
        ax.imshow(cell, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # --- parameter bars ---
    fem = row["fem"]
    _bar_pair(ax_c11, row["C11"],  fem["C11"], "C11 (Pa)", "%.2e")
    _bar_pair(ax_c12, row["C12"],  fem["C12"], "C12 (Pa)", "%.2e")
    _bar_pair(ax_c66, row["C66"],  fem["C66"], "C66 (Pa)", "%.2e")
    _bar_pair(ax_vol, row["vol"],  fem["vf"],  "vol",      "%.3f")

    fig.suptitle(
        f"idx={row['idx']}   "
        f"C11={row['C11']:.2e}  C12={row['C12']:.2e}  "
        f"C66={row['C66']:.2e}  vol={row['vol']:.3f}",
        fontsize=9, y=0.97,
    )

    # legend (shared, placed between images and bars)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="steelblue", label="Dataset"),
        plt.Rectangle((0, 0), 1, 1, color="coral",     label="Generated"),
    ]
    fig.legend(handles=handles, loc="lower right",
               bbox_to_anchor=(0.97, 0.35), fontsize=9, framealpha=0.8)

    out_path = out_dir / f"inference_fem_{i:02d}_idx{row['idx']}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Grid figure (original flow)
# ---------------------------------------------------------------------------

def _plot_grid(rows: list[dict], out_path: Path, steps: int, tensor_w: float) -> None:
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5.4),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.1})
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, row in enumerate(rows):
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        ax_top.imshow(row["cell"], cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
        ax_bot.imshow(row["gen_cell"], cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
        for ax in (ax_top, ax_bot):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_top.set_title(_format_label(row), fontsize=8)

    axes[0, 0].set_ylabel("Dataset", fontsize=11)
    axes[1, 0].set_ylabel("Generated", fontsize=11)
    fig.suptitle(
        f"Conditional diffusion samples (steps={steps}, w={tensor_w})",
        fontsize=12,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True, help="Lightning .ckpt path")
    parser.add_argument("--h5", default="output/ca_bulk_squared/stiffness.h5")
    parser.add_argument("--scaler-dir", default="microstructure_generation_2d")
    parser.add_argument("--split-path", default="microstructure_generation_2d/split.json")
    parser.add_argument("--source", choices=("val", "train"), default="val",
                        help="Which split partition to draw conditions from")
    parser.add_argument("--num", type=int, default=6, help="Number of conditions")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--tensor-w", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the checkpoint's compressed flag (defaults to "
                             "checkpoint hparam).")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--indices", type=int, nargs="*", default=None,
                        help="Optional explicit dataset indices to use")
    parser.add_argument("--output", required=True, help="Grid figure path (.png/.pdf)")
    parser.add_argument("--fem", action="store_true", default=False,
                        help="Run FEM round-trip and save per-condition comparison plots")
    args = parser.parse_args()

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)

    with h5py.File(args.h5, "r") as f:
        n_total = f["cells"].shape[0]
    split = load_or_make_split(
        n=n_total, val_frac=0.02, seed=777, split_path=Path(args.split_path),
    )
    indices = args.indices if args.indices else _pick_indices(
        split, args.num, args.source, args.seed,
    )
    if len(indices) < args.num:
        raise RuntimeError(f"Need {args.num} indices, got {len(indices)}")
    indices = indices[: args.num]

    rows = _load_conditions(args.h5, Path(args.scaler_dir), indices)

    module = DiffusionModel.load_from_checkpoint(args.ckpt, map_location=device).to(device)
    module.eval()
    generator = module.ema_model if args.ema else module.model
    generator.eval()

    compressed = (
        args.compressed
        if args.compressed is not None
        else bool(module.hparams.get("compressed", False))
    )

    for row in tqdm(rows, desc="sampling"):
        img = generator.sample_with_tensor(
            tensor_c=row["tensor_feature"],
            batch_size=1,
            steps=args.steps,
            tensor_w=args.tensor_w,
            verbose=False,
        )
        arr = img.detach().cpu().numpy().squeeze()  # (H, H) with H in {64, 25}
        row["gen_cell"] = _decode_to_50((arr > 0).astype(np.uint8), compressed=compressed)

    # Grid (always saved)
    _plot_grid(rows, Path(args.output), args.steps, args.tensor_w)

    # Per-condition FEM plots
    if args.fem:
        print("Running FEM homogenization on generated cells...")
        out_dir = Path(args.output).parent
        for row in tqdm(rows, desc="FEM"):
            row["fem"] = _run_fem(row["gen_cell"])
        for i, row in enumerate(rows):
            out_path = _plot_fem_figure(row, i, out_dir)
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
