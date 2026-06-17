#!/usr/bin/env python
"""Scatter plot of f_star vs loss for all samples in the surrogate dataset.

Shows:
  - All raw samples coloured by type (init / random / smooth / opt)
  - The "best achievable frontier" (minimum loss per frequency)
  - The ideal-cloak target (loss = 0)

Usage::

    python scripts/plot_surrogate_dataset.py output/surrogate_dataset.h5
    python scripts/plot_surrogate_dataset.py output/surrogate_dataset.h5 -o figs/dataset_scatter.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── colour / marker scheme ────────────────────────────────────────────────────

STYLE = {
    "init":   dict(color="#aaaaaa", marker="x", s=18, alpha=0.6, zorder=2, label="init (push-forward)"),
    "random": dict(color="#4e79a7", marker=".",  s=6,  alpha=0.25, zorder=2, label="random perturbation"),
    "smooth": dict(color="#76b7b2", marker=".",  s=6,  alpha=0.25, zorder=2, label="smooth random field"),
    "opt":    dict(color="#f28e2b", marker=".",  s=6,  alpha=0.35, zorder=3, label="optimisation trajectory"),
}


def categorise(sample_type: str) -> str:
    if sample_type == "init":
        return "init"
    if sample_type.startswith("random"):
        return "random"
    if sample_type.startswith("smooth"):
        return "smooth"
    if sample_type.startswith("opt"):
        return "opt"
    return "other"


def load(path: str):
    with h5py.File(path, "r") as f:
        fstars = f["f_star"][:]
        losses = f["loss"][:]
        stypes = f["sample_type"][:].astype(str)
    cats = np.array([categorise(s) for s in stypes])
    return fstars, losses, cats


def best_frontier(fstars, losses):
    """Min loss per unique frequency."""
    ufreqs = np.sort(np.unique(fstars))
    best = np.array([losses[fstars == f].min() for f in ufreqs])
    return ufreqs, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="HDF5 surrogate dataset path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output PNG path (default: alongside dataset)")
    parser.add_argument("--log", action="store_true", default=False,
                        help="Log-scale y-axis (default: False)")
    parser.add_argument("--no-log", dest="log", action="store_false")
    args = parser.parse_args()

    fstars, losses, cats = load(args.dataset)
    ufreqs, best = best_frontier(fstars, losses)

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        2, 1,
        figsize=(12, 8),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35},
    )
    ax, ax_bot = axes

    # --- scatter by category (draw cheap ones first so opt dots are on top)
    for cat in ("init", "random", "smooth", "opt"):
        mask = cats == cat
        if mask.sum() == 0:
            continue
        kw = STYLE[cat].copy()
        label = kw.pop("label")
        ax.scatter(fstars[mask], losses[mask], label=label, **kw)

    # --- best-achievable frontier
    ax.plot(ufreqs, best, color="#e15759", lw=1.8, zorder=5,
            label="best achieved per freq")

    # --- ideal cloak target
    ax.axhline(0, color="black", lw=1.2, ls="--", zorder=6, label="ideal cloak (loss = 0)")

    ax.set_xlabel("Normalised frequency  $f^*$", fontsize=12)
    ax.set_ylabel("Transmission loss", fontsize=12)
    ax.set_title("Surrogate dataset: cloaking loss vs. frequency", fontsize=13)
    if args.log:
        ax.set_yscale("log")
        ax.set_ylabel("Transmission loss  (log scale)", fontsize=12)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.set_xlim(fstars.min() - 0.05, fstars.max() + 0.05)

    # ── bottom panel: best frontier only, linear scale ────────────────────────
    ax_bot.plot(ufreqs, best, color="#e15759", lw=2.0)
    ax_bot.fill_between(ufreqs, 0, best, alpha=0.18, color="#e15759")
    ax_bot.axhline(0, color="black", lw=1.0, ls="--")
    ax_bot.set_xlabel("Normalised frequency  $f^*$", fontsize=11)
    ax_bot.set_ylabel("Best loss", fontsize=11)
    ax_bot.set_title("Frontier: minimum achievable loss per frequency", fontsize=11)
    ax_bot.set_xlim(fstars.min() - 0.05, fstars.max() + 0.05)

    # annotate worst frequency
    worst_idx = np.argmax(best)
    ax_bot.annotate(
        f"hardest\nf*={ufreqs[worst_idx]:.2f}",
        xy=(ufreqs[worst_idx], best[worst_idx]),
        xytext=(ufreqs[worst_idx] + 0.2, best[worst_idx] * 1.05),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
    )

    # ── stats box ─────────────────────────────────────────────────────────────
    n_total = len(fstars)
    n_opt   = (cats == "opt").sum()
    stats = (
        f"N = {n_total:,}  |  opt: {n_opt:,} ({100*n_opt/n_total:.0f}%)\n"
        f"Frequencies: {len(ufreqs)}  ({ufreqs[0]:.2f} → {ufreqs[-1]:.2f})\n"
        f"Frontier loss: median {np.median(best):.3e}, "
        f"max {best.max():.3e} (f*={ufreqs[worst_idx]:.2f}), "
        f"min {best.min():.3e}"
    )
    fig.text(0.5, 0.01, stats, ha="center", va="bottom", fontsize=8.5,
             family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7))

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    # ── save ──────────────────────────────────────────────────────────────────
    if args.output:
        out = Path(args.output)
    else:
        out = Path(args.dataset).with_suffix(".png")
        out = out.parent / ("scatter_" + out.name.replace(".h5", ".png"))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
