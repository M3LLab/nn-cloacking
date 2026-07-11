"""Overlay NN-matched vs inverse-designed cloaking frequency sweeps.

Companion to ``plot_frequency_sweep_comparison.py`` (single-dir). This overlays
two cloak realisations of the SAME optimised target on one axes:

  * NN  (snapped to nearest dataset cell)     — from <opt_dir>/val_nn/
  * INV (inverse-designed microstructures)    — from <opt_dir>/val_inverse/

for both the matched-homogenised and the pixel-validated sweeps (whichever CSVs
exist), plus any optimized/obstacle/ideal baselines in <opt_dir>. Saved to
<opt_dir>/nn_vs_inverse_sweep.png.

Usage:
    python scripts/vis/plot_nn_vs_inverse_sweep.py output/cell20_phys_flat4_materialreg4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(p: Path):
    d = np.genfromtxt(p, delimiter=",", names=True)
    return np.atleast_1d(d["f_star"]), np.atleast_1d(d["u_ratio"])


# (subdir, csv, style, label)
SERIES = [
    (".", "frequency_sweep_obstacle.csv",
     dict(color="black", marker="s", ls="--"), "Obstacle (no cloak)"),
    (".", "frequency_sweep_ideal.csv",
     dict(color="C3", marker="o", ls="-"), "Ideal (analytic)"),
    (".", "frequency_sweep_optimized.csv",
     dict(color="C0", marker="D", ls="-"), "Optimized (continuous homog.)"),
    ("val_nn", "frequency_sweep_matched.csv",
     dict(color="C1", marker="v", ls="--"), "NN matched (homogenised)"),
    ("val_inverse", "frequency_sweep_matched.csv",
     dict(color="C4", marker="v", ls="-"), "Inverse matched (homogenised)"),
    ("val_nn", "frequency_sweep_validated.csv",
     dict(color="C2", marker="^", ls="--"), "NN validated (pixel)"),
    ("val_inverse", "frequency_sweep_validated.csv",
     dict(color="C5", marker="^", ls="-"), "Inverse validated (pixel)"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("opt_dir", type=Path)
    ap.add_argument("--train-fstar", type=float, default=2.0)
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (args.opt_dir / "nn_vs_inverse_sweep.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    f_lo, f_hi, y_hi = np.inf, -np.inf, 0.0
    n = 0
    summary = []
    for sub, csv, style, label in SERIES:
        p = args.opt_dir / sub / csv
        if not p.exists():
            continue
        fs, r = _load(p)
        err = float(np.abs(1.0 - r).mean())   # cloaking error: distance from the perfect-cloak ratio 1.0
        ax.plot(fs, r, lw=1.7, markersize=6,
                label=f"{label}  (mean |1-ratio| {err:.3f})", **style)
        for x, y in zip(fs, r):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=6.5,
                        color=style["color"], alpha=0.8)
        f_lo, f_hi = min(f_lo, fs.min()), max(f_hi, fs.max())
        y_hi = max(y_hi, r.max())
        summary.append((label, float(r.mean()), err))
        n += 1

    if n == 0:
        raise SystemExit(f"no frequency_sweep_*.csv found under {args.opt_dir}")

    ax.axhline(1.0, color="green", ls="--", lw=1.0, alpha=0.7, label="Perfect cloak (ratio = 1.0)")
    if f_lo <= args.train_fstar <= f_hi:
        ax.axvline(args.train_fstar, color="gray", ls=":", lw=0.8, alpha=0.5,
                   label=rf"Training $f^*={args.train_fstar:g}$")
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$  (=1.0 is perfect cloak)")
    ax.set_title(f"{args.opt_dir.name} — NN vs inverse-designed cloak")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    pad = 0.05 * (f_hi - f_lo) if f_hi > f_lo else 0.05
    ax.set_xlim(f_lo - pad, f_hi + pad)
    ax.set_ylim(0, max(1.1, y_hi * 1.1))
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}\n  series plotted ({n}):")
    for lab, m, e in summary:
        print(f"    {lab:38s} mean ratio {m:.4f}   mean |1-ratio| {e:.4f}")


if __name__ == "__main__":
    main()
