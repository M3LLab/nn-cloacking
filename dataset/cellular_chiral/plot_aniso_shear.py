"""v2 vs v3 in anisotropy/shear coordinates.

    x = log10(C11 / C22)   anisotropy, 0 = transversely isotropic in-plane
    y = log10(C66)         shear stiffness [Pa]

218k points cannot be a scatter plot, so the pool is a hexbin density on a log
count scale (sequential, single hue).  v3 is v2 plus 225 rows, so the two density
panels are identical to the eye *by construction* -- that is the honest result,
and the panels are drawn at a shared colour scale so the reader can confirm it
rather than take it on faith.  The 225 inverse-designed cells are marked
individually on the v3 panel, since that is the entire difference between them.

Usage
-----

    python -m dataset.cellular_chiral.plot_aniso_shear \
        -a output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5 \
        -b output/ca_bulk_squared/stiffness_tri6_uniform_v3.h5 \
        -o output/ca_bulk_squared/aniso_shear_v2_v3.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

# Single-hue sequential ramp at the hue of the repo blue #2a78d6, stepped so that
# adjacent dL >= 0.06 and the light end still clears 2.0:1 on the surface.
RAMP = ["#90b2df", "#6b9cdd", "#4686d7", "#1c6ecd", "#0057b7", "#004495"]
_NEW = "#d4276e"                      # inverse-designed (provenance 2)
_INK, _INK2, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
_GRID, _SPINE = "#ebeae6", "#d8d7d3"


def load(path: Path):
    with h5py.File(path, "r") as f:
        C11, C22, C66 = f["C11"][:], f["C22"][:], f["C66"][:]
        prov = f["provenance"][:] if "provenance" in f else None
    ok = (C11 > 0) & (C22 > 0) & (C66 > 0)
    return np.log10(C11[ok] / C22[ok]), np.log10(C66[ok]), (prov[ok] if prov is not None else None)


def _style(ax):
    ax.set_facecolor(_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_SPINE)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK2, labelsize=9)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-a", "--before", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("-b", "--after", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v3.h5"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/aniso_shear_v2_v3.png"))
    p.add_argument("--gridsize", type=int, default=120)
    p.add_argument("--label-a", default=None, help="override the left panel label")
    p.add_argument("--label-b", default=None, help="override the right panel label")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    from matplotlib.lines import Line2D

    xa, ya, _ = load(args.before)
    xb, yb, provb = load(args.after)
    na = args.label_a or args.before.stem.split("_")[-1]
    nb = args.label_b or args.after.stem.split("_")[-1]

    # shared extent, clipped to the bulk so a handful of outliers do not set the
    # scale for 218k points
    xlo, xhi = np.percentile(np.concatenate([xa, xb]), [0.1, 99.9])
    ylo, yhi = np.percentile(np.concatenate([ya, yb]), [0.1, 99.9])
    pad_x, pad_y = 0.06 * (xhi - xlo), 0.06 * (yhi - ylo)
    ext = (xlo - pad_x, xhi + pad_x, ylo - pad_y, yhi + pad_y)
    cmap = LinearSegmentedColormap.from_list("repo_blue", RAMP)

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.9), facecolor=_SURFACE,
                             sharex=True, sharey=True)

    # one norm for both panels, so "they look the same" is a real comparison
    hb0 = axes[0].hexbin(xa, ya, gridsize=args.gridsize, extent=ext, mincnt=1,
                         cmap=cmap, linewidths=0)
    vmax = max(hb0.get_array().max(),
               np.histogram2d(xb, yb, bins=1)[0].max())
    hbs = []
    for ax, (x, y, lab, n) in zip(axes, [(xa, ya, na, len(xa)), (xb, yb, nb, len(xb))]):
        ax.clear()
        _style(ax)
        hb = ax.hexbin(x, y, gridsize=args.gridsize, extent=ext, mincnt=1,
                       cmap=cmap, linewidths=0, norm=LogNorm(vmin=1, vmax=vmax))
        hbs.append(hb)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.axvline(0.0, color=_INK2, lw=0.9, ls="--", zorder=2)
        ax.set_title(f"{lab}   ({n:,} cells)", fontsize=12, color=_INK, pad=8)
        ax.set_xlabel("anisotropy   log$_{10}$(C11 / C22)", fontsize=10.5, color=_INK2)
    axes[0].set_ylabel("shear   log$_{10}$(C66)  [Pa]", fontsize=10.5, color=_INK2)
    axes[0].text(0.0, 1.004, "isotropic", transform=axes[0].get_xaxis_transform(),
                 ha="center", va="bottom", fontsize=8.5, color=_INK2)

    # the whole difference between the panels
    handles = [Line2D([], [], marker="h", ls="none", color=RAMP[3], ms=9,
                      label="cell density (log count)")]
    if provb is not None and (provb == 2).any():
        m = provb == 2
        axes[1].scatter(xb[m], yb[m], s=42, marker="D", c=_NEW,
                        edgecolors=_SURFACE, linewidths=1.1, zorder=5,
                        label=f"inverse-designed ({int(m.sum())})")
        handles.append(Line2D([], [], marker="D", ls="none", color=_NEW, ms=8,
                              markeredgecolor=_SURFACE, markeredgewidth=1.1,
                              label=f"inverse-designed ({int(m.sum())})"))
    axes[1].legend(handles=handles[1:], frameon=False, fontsize=9.5,
                   labelcolor=_INK2, loc="lower left")

    cb = fig.colorbar(hbs[1], ax=axes, fraction=0.032, pad=0.015)
    cb.set_label("cells per hex bin", fontsize=10, color=_INK2)
    cb.ax.tick_params(colors=_INK2, labelsize=9)
    cb.outline.set_visible(False)

    # only claim "identical by construction" when the marked cells really are the
    # entire difference; against the unthinned pool the panels differ genuinely
    n_new = int((provb == 2).sum()) if provb is not None else 0
    if n_new and len(xb) - len(xa) == n_new:
        sub = (f"{nb} adds {n_new} cells to {len(xa):,} — the densities are identical "
               "by construction; the diamonds are the difference.")
    else:
        sub = (f"{len(xa):,} vs {len(xb):,} cells. Different point sets, so the "
               "densities differ genuinely; note the shared log colour scale.")
    fig.suptitle(f"Anisotropy vs shear: {na} -> {nb}.  {sub}",
                 fontsize=11.5, color=_INK, y=0.975)
    fig.savefig(args.output, dpi=140, facecolor=_SURFACE, bbox_inches="tight")
    print(f"wrote {args.output}")
    print(f"  x log10(C11/C22) range shown {ext[0]:.2f}..{ext[1]:.2f}  "
          f"(full {min(xa.min(), xb.min()):.2f}..{max(xa.max(), xb.max()):.2f})")
    print(f"  y log10(C66)     range shown {ext[2]:.2f}..{ext[3]:.2f}  "
          f"(full {min(ya.min(), yb.min()):.2f}..{max(ya.max(), yb.max()):.2f})")
    n_out = int(((xb < ext[0]) | (xb > ext[1]) | (yb < ext[2]) | (yb > ext[3])).sum())
    print(f"  {n_out} of {len(xb):,} {nb} cells fall outside the shown window")


if __name__ == "__main__":
    main()
