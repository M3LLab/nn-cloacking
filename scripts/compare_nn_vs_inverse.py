"""Compare nearest-neighbour dataset matching vs inverse design, per cloak cell.

The inverse-design sweep initialises each cell from its nearest dataset cell
(NN matching) and then optimises a neural field on top.  This script quantifies
how much the optimisation actually improves over just keeping the NN cell:

  * NN baseline error  : |nn_flat4/nn_rho − target| / |target|   (dataset cell,
                         the inverse-design starting point)
  * Inverse-design error: |pred_flat4/pred_rho − target| / |target|  (optimised
                         binary, read from cell_designs/cell_XXX/weights.npz)

Both are relative errors over [C11, C22, C12, C66, rho].  A figure and CSV are
written to the cell_designs directory.

Usage
-----
    python scripts/compare_nn_vs_inverse.py output/cell20_phys_flat4_materialreg4
    python scripts/compare_nn_vs_inverse.py <opt_dir> --cell-designs <dir> -o <png>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.cell_inverse_design_sweep import (
    load_dataset_for_matching, opt_flat4_to_inv_flat4,
)

_LABELS = ["C11", "C22", "C12", "C66", "rho"]


def nn_match(target5: np.ndarray, X: np.ndarray, rho_weight: float = 1.0) -> np.ndarray:
    """Return the dataset row [C11,C22,C12,C66,rho] nearest to target5.

    Same relative metric as find_nn_quadrant (no h5 cell read needed)."""
    w = np.array([1.0, 1.0, 1.0, 1.0, rho_weight])
    rel = (X - target5[None, :]) / (np.abs(target5)[None, :] + 1e-30)
    d2 = np.sum(rel ** 2 * w[None, :], axis=1)
    return X[int(np.argmin(d2))]


def compute_rows(cd: Path, ds, rho_weight: float = 1.0) -> list[dict]:
    """Read every cell_*/weights.npz under ``cd`` → per-cell NN vs design errors.

    Returns a list of dicts (sorted by NN avg-component error) with keys
    cell, nn, inv (per-component % arrays), and nn_max/inv_max/nn_avg/inv_avg.
    """
    rows = []
    for wpath in sorted(cd.glob("cell_*/weights.npz")):
        cell_idx = int(wpath.parent.name.split("_")[1])
        d = np.load(str(wpath))
        if "pred_flat4" not in d or "target_flat4" not in d:
            continue
        tgt4 = np.asarray(d["target_flat4"], dtype=np.float64)
        prd4 = np.asarray(d["pred_flat4"], dtype=np.float64)
        trho = float(d["target_rho"]); prho = float(d["pred_rho"])
        tgt5 = np.append(tgt4, trho)
        nn5 = nn_match(tgt5, ds.X, rho_weight)
        prd5 = np.append(prd4, prho)

        nn_rel = np.abs((nn5 - tgt5) / (np.abs(tgt5) + 1e-30)) * 100.0
        inv_rel = np.abs((prd5 - tgt5) / (np.abs(tgt5) + 1e-30)) * 100.0
        rows.append(dict(cell=cell_idx, nn=nn_rel, inv=inv_rel,
                         nn_max=float(nn_rel.max()), inv_max=float(inv_rel.max()),
                         nn_avg=float(nn_rel.mean()), inv_avg=float(inv_rel.mean())))
    rows.sort(key=lambda r: r["nn_avg"])
    return rows


def draw_dumbbell(ax, rows: list[dict], title: str = "",
                  design_label: str = "inverse design",
                  max_labels: int = 40, show_ylabel: bool = True) -> None:
    """Draw the NN → design per-cell dumbbell into ``ax`` (fixed size, thinned labels).

    Rows must already be sorted (see compute_rows).  Only ~``max_labels`` evenly
    spaced cell-index tick labels are shown so they never overlap regardless of n.
    """
    n = len(rows)
    nn_avg = np.array([r["nn_avg"] for r in rows])
    inv_avg = np.array([r["inv_avg"] for r in rows])
    improved = inv_avg < nn_avg
    y = np.arange(n)
    for i in range(n):
        c = "#2ecc71" if improved[i] else "#e74c3c"
        ax.plot([nn_avg[i], inv_avg[i]], [y[i], y[i]], "-", color=c, lw=1.0, alpha=0.7, zorder=1)
    ax.scatter(nn_avg, y, c="#7f8c8d", s=16, label="NN match", zorder=2)
    ax.scatter(inv_avg, y, c="#2980b9", s=16, label=design_label, zorder=3)

    step = max(1, int(np.ceil(n / max_labels)))
    ticks = y[::step]
    ax.set_yticks(ticks)
    ax.set_yticklabels([rows[i]["cell"] for i in ticks], fontsize=7)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlabel("avg-component relative error (%)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("cell", fontsize=10)
    n_imp = int(improved.sum())
    sub = (f"{title}\nNN {nn_avg.mean():.1f}% → design {inv_avg.mean():.1f}%   "
           f"improved {n_imp}/{n} ({100 * n_imp / n:.0f}%)")
    ax.set_title(sub, fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, axis="x")
    ax.tick_params(labelsize=8)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("opt_dir", type=Path)
    ap.add_argument("--cell-designs", type=Path, default=None,
                    help="cell_designs dir (default <opt_dir>/cell_designs)")
    ap.add_argument("--dataset-path", type=Path,
                    default=Path("output/ca_bulk_squared/stiffness.h5"))
    ap.add_argument("--nn-rho-weight", type=float, default=1.0)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output PNG (default <cell_designs>/nn_vs_inverse.png)")
    args = ap.parse_args()

    cd = args.cell_designs or (args.opt_dir / "cell_designs")
    out_png = args.out or (cd / "nn_vs_inverse.png")

    ds = load_dataset_for_matching(args.dataset_path)

    rows = compute_rows(cd, ds, args.nn_rho_weight)
    if not rows:
        sys.exit(f"No completed cells with predictions found in {cd}")
    n = len(rows)
    nn_max = np.array([r["nn_max"] for r in rows])
    inv_max = np.array([r["inv_max"] for r in rows])
    nn_avg = np.array([r["nn_avg"] for r in rows])   # user's proxy: mean over components
    inv_avg = np.array([r["inv_avg"] for r in rows])
    nn_comp = np.array([r["nn"] for r in rows])      # (n,5)
    inv_comp = np.array([r["inv"] for r in rows])
    improved = inv_avg < nn_avg                       # primary metric = avg-component
    impr = nn_avg - inv_avg                            # >0 = inverse better

    # ── CSV ──────────────────────────────────────────────────────────
    csv = cd / "nn_vs_inverse.csv"
    with open(csv, "w") as f:
        f.write("cell,nn_max,inv_max,improvement," +
                ",".join(f"nn_{l}" for l in _LABELS) + "," +
                ",".join(f"inv_{l}" for l in _LABELS) + "\n")
        for r in rows:
            f.write(f"{r['cell']},{r['nn_max']:.3f},{r['inv_max']:.3f},"
                    f"{r['nn_max']-r['inv_max']:.3f}," +
                    ",".join(f"{v:.3f}" for v in r["nn"]) + "," +
                    ",".join(f"{v:.3f}" for v in r["inv"]) + "\n")

    # ── figure ───────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.0], wspace=0.32)

    # (1) scatter NN vs inverse (per-cell AVG component error = proxy metric)
    ax = fig.add_subplot(gs[0])
    lim = max(nn_avg.max(), inv_avg.max()) * 1.15 + 1
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="no change")
    ax.scatter(nn_avg[improved], inv_avg[improved], c="#2ecc71", s=28,
               edgecolor="k", lw=0.3, label=f"improved ({int(improved.sum())})")
    ax.scatter(nn_avg[~improved], inv_avg[~improved], c="#e74c3c", s=28,
               edgecolor="k", lw=0.3, label=f"worse ({int((~improved).sum())})")
    ax.set_xlabel("NN-match avg-component err (%)", fontsize=8)
    ax.set_ylabel("Inverse-design avg-component err (%)", fontsize=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Per-cell AVG-component error (proxy)\n(below dashed = inverse better)", fontsize=9)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    ax.tick_params(labelsize=7)

    # (2) dumbbell per cell: NN -> inverse (sorted by NN avg)
    ax = fig.add_subplot(gs[1])
    y = np.arange(n)
    for i in range(n):
        c = "#2ecc71" if improved[i] else "#e74c3c"
        ax.plot([nn_avg[i], inv_avg[i]], [y[i], y[i]], "-", color=c, lw=1.0, alpha=0.7, zorder=1)
    ax.scatter(nn_avg, y, c="#7f8c8d", s=16, label="NN match", zorder=2)
    ax.scatter(inv_avg, y, c="#2980b9", s=16, label="inverse design", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([r["cell"] for r in rows], fontsize=5)
    ax.set_xlabel("avg-component rel err (%)", fontsize=8)
    ax.set_ylabel("cell idx (sorted by NN avg err)", fontsize=8)
    ax.set_title("NN → inverse design, per cell (avg-component)", fontsize=9)
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=0.3, axis="x")
    ax.tick_params(labelsize=7)

    # (3) per-component mean |err|
    ax = fig.add_subplot(gs[2])
    x = np.arange(5); bw = 0.38
    ax.bar(x - bw/2, nn_comp.mean(0), bw, label="NN match", color="#7f8c8d")
    ax.bar(x + bw/2, inv_comp.mean(0), bw, label="inverse design", color="#2980b9")
    ax.set_xticks(x); ax.set_xticklabels(_LABELS, fontsize=8)
    ax.set_ylabel("mean |rel err| (%)", fontsize=8)
    ax.set_title("Mean error per component", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3, axis="y"); ax.tick_params(labelsize=7)

    fig.suptitle(
        f"{args.opt_dir.name}: NN matching vs inverse design  ({n} cells)   "
        f"mean AVG-component err: NN={nn_avg.mean():.1f}%  →  inverse={inv_avg.mean():.1f}%   "
        f"(worst-comp: {nn_max.mean():.1f}% → {inv_max.mean():.1f}%)   "
        f"|  improved {int(improved.sum())}/{n} ({100*improved.mean():.0f}%)",
        fontsize=9.5, y=1.02)
    fig.savefig(str(out_png), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # ── standalone dumbbell figure (the middle panel, on its own) ────────
    db_png = out_png.with_name(out_png.stem + "_dumbbell" + out_png.suffix)
    fig2, ax2 = plt.subplots(figsize=(6.5, 8.0))
    draw_dumbbell(ax2, rows, title="NN → inverse design, per cell (sorted by NN error)",
                  design_label="inverse design")
    fig2.savefig(str(db_png), dpi=120, bbox_inches="tight")
    plt.close(fig2)

    # ── console summary ──────────────────────────────────────────────
    print(f"\n{n} cells.")
    print(f"  AVG-component rel err (proxy) %:  NN mean {nn_avg.mean():5.2f} median {np.median(nn_avg):5.2f}"
          f"  ->  inverse mean {inv_avg.mean():5.2f} median {np.median(inv_avg):5.2f}")
    print(f"  worst-component  rel err      %:  NN mean {nn_max.mean():5.2f} median {np.median(nn_max):5.2f}"
          f"  ->  inverse mean {inv_max.mean():5.2f} median {np.median(inv_max):5.2f}")
    print(f"  improved on AVG-proxy: {int(improved.sum())}/{n} ({100*improved.mean():.0f}%)   "
          f"mean improvement {impr.mean():+.2f} pp")
    print("  per-component mean |err| %  NN -> inverse:")
    for j, l in enumerate(_LABELS):
        print(f"    {l:4s}: {nn_comp[:,j].mean():5.2f} -> {inv_comp[:,j].mean():5.2f}")
    print(f"\nFigure   → {out_png}")
    print(f"Dumbbell → {db_png}")
    print(f"CSV      → {csv}")


if __name__ == "__main__":
    main()
