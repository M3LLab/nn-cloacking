#!/usr/bin/env python
"""Generate presentation figures for the squared-cell dataset slide.

Produces two PNGs from output/ca_squared_2m/stiffness.h5:
  * squared_cell_samples.png  -- gallery of real 50x50 cells + homogenised C tensor
  * squared_dataset_coverage.png -- design-space distribution in (C11, C12) coloured by C66

Run:
    /home/david/miniconda3/envs/jax-fem-env/bin/python \
        docs/metamat2026/figures/make_squared_dataset_figs.py
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

# ASTRA theme colours
NAVY = "#0D2B5E"
ORANGE = "#E97132"
GOLD = "#C8A84B"
BLUE = "#156082"

H5 = Path("output/ca_squared_2m/stiffness.h5")
OUT = Path("docs/metamat2026/figures")
GPA = 1e9
N_SUB = 60_000  # subset loaded for speed / scatter

rng = np.random.default_rng(0)


def load():
    # scalar columns are tiny (~7 MB each) -> read fully, subsample in numpy.
    # (h5py point-selection with a 60k fancy index is pathologically slow.)
    with h5py.File(H5, "r") as f:
        full = {k: f[k][:] for k in ("C11", "C22", "C12", "C66", "vol", "rho")}
    n = full["C11"].shape[0]
    idx = np.sort(rng.choice(n, size=min(N_SUB, n), replace=False))
    d = {k: v[idx] for k, v in full.items()}
    d["full"] = full
    d["n_total"] = n
    d["idx"] = idx
    return d


def pick_gallery(d, ncols=6, nrows=2):
    """Pick a diverse, physically valid set of cells spanning the design space."""
    r = d["C12"] / d["C11"]          # Poisson-like ratio (can be < 0: auxetic)
    # keep well-resolved cells (avoid near-empty / near-full extremes)
    ok = (d["vol"] > 0.30) & (d["vol"] < 0.80) & (d["C11"] > 1e9)
    order = np.argsort(r[ok])
    sub = np.flatnonzero(ok)[order]
    # evenly spaced across the ratio ordering -> auxetic ... stiff-shear variety
    sel = sub[np.linspace(0, len(sub) - 1, ncols * nrows).astype(int)]
    return sel


def fig_gallery(d):
    ncols, nrows = 6, 2
    sel = pick_gallery(d, ncols, nrows)
    gidx = d["idx"][sel]
    order = np.argsort(gidx)          # h5py requires increasing index order
    with h5py.File(H5, "r") as f:
        cells_sorted = f["cells"][gidx[order]]
    cells = np.empty_like(cells_sorted)
    cells[order] = cells_sorted       # restore the design-space ordering

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 5.4),
                             gridspec_kw={"hspace": 0.85, "wspace": 0.12})
    for ax, s, cell in zip(axes.ravel(), sel, cells):
        ax.imshow(cell, cmap="binary", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(NAVY); sp.set_linewidth(1.1)
        c11, c22, c12, c66 = (d[k][s] / GPA for k in ("C11", "C22", "C12", "C66"))
        # homogenised in-plane stiffness (Voigt), values in GPa
        txt = (rf"$C_{{11}}{{=}}{c11:.1f}\ \ C_{{22}}{{=}}{c22:.1f}$" + "\n"
               rf"$C_{{12}}{{=}}{c12:.1f}\ \ C_{{66}}{{=}}{c66:.1f}$")
        ax.set_title(txt, fontsize=7.5, color=NAVY, pad=3)
        nu = d["C12"][s] / d["C11"][s]
        tag = "  auxetic" if nu < 0 else ""
        ax.set_xlabel(rf"$\rho={d['vol'][s]:.2f}${tag}",
                      fontsize=8, color=ORANGE if tag else "0.35")
    fig.suptitle("Squared unit cells sampled from the catalogue  "
                 r"(homogenised $\mathbf{C}$ in GPa)",
                 fontsize=12, color=NAVY, y=1.0)
    p = OUT / "squared_cell_samples.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print("saved", p)


def fig_coverage(d):
    c11 = d["C11"] / GPA
    c12 = d["C12"] / GPA
    c66 = d["C66"] / GPA

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sc = ax.scatter(c11, c12, c=c66, s=3, alpha=0.45, cmap="viridis",
                    rasterized=True, linewidths=0)
    ax.axhline(0, color=ORANGE, lw=1.2, ls="--", zorder=0)
    ax.text(ax.get_xlim()[1] * 0.98, -0.05, "auxetic  ($C_{12}<0$)",
            ha="right", va="top", color=ORANGE, fontsize=9)
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label(r"shear stiffness  $C_{66}$  (GPa)", color=NAVY)
    ax.set_xlabel(r"axial stiffness  $C_{11}$  (GPa)", color=NAVY, fontsize=11)
    ax.set_ylabel(r"coupling  $C_{12}$  (GPa)", color=NAVY, fontsize=11)
    ax.set_title(f"Design-space coverage  (N = {d['n_total']:,} cells)",
                 color=NAVY, fontsize=12)
    ax.tick_params(colors="0.3")
    for sp in ax.spines.values():
        sp.set_edgecolor("0.6")
    fig.tight_layout()
    p = OUT / "squared_dataset_coverage.png"
    fig.savefig(p, dpi=180, bbox_inches="tight")
    print("saved", p)


if __name__ == "__main__":
    d = load()
    fig_gallery(d)
    fig_coverage(d)
