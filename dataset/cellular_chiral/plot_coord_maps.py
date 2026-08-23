"""Two-panel density maps of any two datasets in a choice of coordinate systems.

Generalises ``plot_aniso_shear`` to a registry of coordinate systems, so the same
validated styling and shared-colour-scale comparison can be pointed at whichever
projection of the 5-D property space is being asked about.

Coordinates available (``--coords``):

  aniso_shear  log10(C11/C22)          vs log10(C66)      the default elsewhere
  vol_c11      volume fraction         vs log10(C11)      density-stiffness (Ashby)
  k_g          log10(K)                vs log10(G)        2-D elastic map
  c12_c66      C12 [GPa]               vs log10(C66)      coupling vs shear; C12<0 is the auxetic side
  vol_gk       volume fraction         vs log10(G/K)      pentamode-ness: G/K -> 0 is fluid-like

with 2-D plane moduli K = (C11 + C22 + 2 C12)/4 and G = C66.  Both are strictly
positive on every dataset here, so the logs are safe; the script asserts it rather
than assuming.

Usage
-----

    python -m dataset.cellular_chiral.plot_coord_maps \
        -a output/ca_bulk_squared/stiffness.h5 \
        -b output/ca_bulk_squared/stiffness_tri6_uniform_v3.h5 \
        --coords vol_gk --label-a "reference pool" --label-b v3 \
        -o output/ca_bulk_squared/map_vol_gk.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from dataset.cellular_chiral.bounds import hs_upper_2d, voigt_upper
from dataset.cellular_chiral.plot_aniso_shear import (
    RAMP, _GRID, _INK, _INK2, _NEW, _SPINE, _SURFACE, _style,
)

_BOUND = "#c2410c"                    # bound curve: distinct from the blue ramp

FEATS = ("C11", "C22", "C12", "C66", "vol")


def derived(d: dict) -> dict:
    """2-D plane bulk/shear moduli, alongside the raw components."""
    out = dict(d)
    out["K"] = (d["C11"] + d["C22"] + 2 * d["C12"]) / 4.0
    out["G"] = d["C66"]
    return out


def _hs_K(v):
    return np.log10(hs_upper_2d(v)[0])


def _hs_G(v):
    return np.log10(hs_upper_2d(v)[1])


def _voigt_C11(v):
    return np.log10(np.maximum(voigt_upper(v)[0], 1e-30))


# name -> dict(x, y, xlabel, ylabel, [vline], [hline], [curve], [curve_label])
# ``curve`` is f(x) -> y, drawn on both panels: a bound in the same coordinates.
COORDS = {
    "aniso_shear": dict(
        x=lambda d: np.log10(d["C11"] / d["C22"]),
        y=lambda d: np.log10(d["C66"]),
        xlabel="anisotropy   log$_{10}$(C11 / C22)",
        ylabel="shear   log$_{10}$(C66)  [Pa]", vline=0.0),
    "vol_c11": dict(
        x=lambda d: d["vol"], y=lambda d: np.log10(d["C11"]),
        xlabel="volume fraction", ylabel="log$_{10}$(C11)  [Pa]",
        curve=_voigt_C11, curve_label="Voigt upper bound (C11)"),
    "k_g": dict(
        x=lambda d: np.log10(d["K"]), y=lambda d: np.log10(d["G"]),
        xlabel="bulk   log$_{10}$(K)  [Pa]",
        ylabel="shear   log$_{10}$(G = C66)  [Pa]"),
    "c12_c66": dict(
        x=lambda d: d["C12"] / 1e9, y=lambda d: np.log10(d["C66"]),
        xlabel="C12  [GPa]        (< 0 = auxetic side)",
        ylabel="shear   log$_{10}$(C66)  [Pa]", vline=0.0),
    "vol_gk": dict(
        x=lambda d: d["vol"], y=lambda d: np.log10(d["G"] / d["K"]),
        xlabel="volume fraction",
        ylabel="log$_{10}$(G / K)      (down = pentamode-like)", hline=0.0),
    # Hashin-Shtrikman. The K bound is sharp for ANY two-phase composite,
    # anisotropic included, so nothing may cross it.  The G bound is the
    # isotropic-composite bound and a strongly anisotropic cell may legitimately
    # beat it -- it is drawn as a diagnostic, not a hard ceiling (see bounds.py).
    "vol_K": dict(
        x=lambda d: d["vol"], y=lambda d: np.log10(d["K"]),
        xlabel="volume fraction",
        ylabel="bulk   log$_{10}$(K)  [Pa]",
        curve=_hs_K, curve_label="Hashin-Shtrikman upper bound (sharp)"),
    "vol_G": dict(
        x=lambda d: d["vol"], y=lambda d: np.log10(d["G"]),
        xlabel="volume fraction",
        ylabel="shear   log$_{10}$(G = C66)  [Pa]",
        curve=_hs_G, curve_label="Hashin-Shtrikman upper bound (isotropic; diagnostic)"),
}


def load(path: Path):
    with h5py.File(path, "r") as f:
        d = {k: f[k][:].astype(np.float64) for k in FEATS}
        prov = f["provenance"][:] if "provenance" in f else None
    return derived(d), prov


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-a", "--before", type=Path, required=True)
    p.add_argument("-b", "--after", type=Path, required=True)
    p.add_argument("-c", "--coords", choices=sorted(COORDS), required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--gridsize", type=int, default=120)
    p.add_argument("--highlight-new", action="store_true",
                   help="mark provenance==2 (inverse-designed) rows; off by "
                        "default — they are ordinary members of the dataset")
    p.add_argument("--report-above", action="store_true",
                   help="count rows above the bound curve (sanity check)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D

    spec = COORDS[args.coords]
    fx, fy = spec["x"], spec["y"]
    xlab, ylab = spec["xlabel"], spec["ylabel"]
    vline, hline = spec.get("vline"), spec.get("hline")
    curve, curve_label = spec.get("curve"), spec.get("curve_label")
    da, prova = load(args.before)
    db, provb = load(args.after)
    if args.coords in ("k_g", "vol_gk"):
        for nm, d in (("before", da), ("after", db)):
            bad = int((d["K"] <= 0).sum())
            assert bad == 0, f"{nm}: {bad} rows with K <= 0, log undefined"

    xa, ya = fx(da), fy(da)
    xb, yb = fx(db), fy(db)
    ok_a, ok_b = np.isfinite(xa) & np.isfinite(ya), np.isfinite(xb) & np.isfinite(yb)
    if provb is not None:
        provb = provb[ok_b]
    xa, ya, xb, yb = xa[ok_a], ya[ok_a], xb[ok_b], yb[ok_b]

    na = args.label_a or args.before.stem.split("_")[-1]
    nb = args.label_b or args.after.stem.split("_")[-1]

    xlo, xhi = np.percentile(np.concatenate([xa, xb]), [0.1, 99.9])
    ylo, yhi = np.percentile(np.concatenate([ya, yb]), [0.1, 99.9])
    px, py = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
    ext = (xlo - px, xhi + px, ylo - py, yhi + py)
    if curve is not None:
        # a bound the data never reaches still has to be fully visible -- the gap
        # between cloud and ceiling is the whole point of drawing it
        top = float(np.nanmax(curve(np.linspace(ext[0], ext[1], 400))))
        ext = (ext[0], ext[1], ext[2], max(ext[3], top + 0.04 * (yhi - ylo)))
    cmap = LinearSegmentedColormap.from_list("repo_blue", RAMP)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor=_SURFACE,
                             sharex=True, sharey=True)
    probe = axes[0].hexbin(xa, ya, gridsize=args.gridsize, extent=ext, mincnt=1)
    vmax = float(probe.get_array().max())
    probe.remove()

    hb = None
    for ax, (x, y, lab, n) in zip(axes, [(xa, ya, na, len(xa)), (xb, yb, nb, len(xb))]):
        _style(ax)
        hb = ax.hexbin(x, y, gridsize=args.gridsize, extent=ext, mincnt=1,
                       cmap=cmap, linewidths=0, norm=LogNorm(vmin=1, vmax=vmax))
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        if vline is not None:
            ax.axvline(vline, color=_INK2, lw=0.9, ls="--", zorder=2)
        if hline is not None:
            ax.axhline(hline, color=_INK2, lw=0.9, ls="--", zorder=2)
        if curve is not None:
            gx = np.linspace(ext[0], ext[1], 400)
            ax.plot(gx, curve(gx), color=_BOUND, lw=2.0, zorder=6,
                    solid_capstyle="round")
        ax.set_title(f"{lab}   ({n:,} cells)", fontsize=12, color=_INK, pad=8)
        ax.set_xlabel(xlab, fontsize=10.5, color=_INK2)
    axes[0].set_ylabel(ylab, fontsize=10.5, color=_INK2)

    handles = []
    if curve is not None:
        handles.append(Line2D([], [], color=_BOUND, lw=2.0, label=curve_label))
    if args.highlight_new and provb is not None and (provb == 2).any():
        m = provb == 2
        axes[1].scatter(xb[m], yb[m], s=42, marker="D", c=_NEW,
                        edgecolors=_SURFACE, linewidths=1.1, zorder=5)
        handles.append(Line2D([], [], marker="D", ls="none", color=_NEW, ms=8,
                              markeredgecolor=_SURFACE, markeredgewidth=1.1,
                              label=f"inverse-designed ({int(m.sum())})"))
    if handles:
        axes[1].legend(handles=handles, fontsize=9.5, labelcolor=_INK2, loc="best",
                       frameon=True, facecolor=_SURFACE, edgecolor="none",
                       framealpha=0.88)

    cb = fig.colorbar(hb, ax=axes, fraction=0.032, pad=0.015)
    cb.set_label("cells per hex bin", fontsize=10, color=_INK2)
    cb.ax.tick_params(colors=_INK2, labelsize=9)
    cb.outline.set_visible(False)

    n_out = int(((xb < ext[0]) | (xb > ext[1]) | (yb < ext[2]) | (yb > ext[3])).sum())
    fig.suptitle(f"{na}  vs  {nb}   [{args.coords}]   "
                 f"{len(xa):,} vs {len(xb):,} cells, shared log colour scale",
                 fontsize=11.5, color=_INK, y=0.975)
    fig.savefig(args.output, dpi=140, facecolor=_SURFACE, bbox_inches="tight")
    print(f"wrote {args.output}   ({n_out} of {len(xb):,} {nb} cells outside the window)")
    if curve is not None and args.report_above:
        for lab, x, y in ((na, xa, ya), (nb, xb, yb)):
            above = int((y > curve(x) + 1e-12).sum())
            print(f"    {lab:16s} {above:6d} of {len(y):,} above the bound "
                  f"({100 * above / len(y):.3f}%)")


if __name__ == "__main__":
    main()
