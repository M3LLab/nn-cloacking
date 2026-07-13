"""Visualize the 10x10 cell decomposition over the triangular cloak geometry,
contrasting legacy cell-center masking vs. confine_to_cloak=True.

The triangular cloak (Chatzopoulos et al. 2023) is the ANNULUS between two
nested triangles sharing the free-surface base of half-width c:

    depth = y_top - y ,   r = |x - x_c| / c
    inner (defect) apex depth = a ,   outer apex depth = b
    in_cloak  <=>  r <= 1  AND  a*(1-r) <= depth <= b*(1-r)

A regular n_x x n_y grid of rectangular cells covers the bounding box
[x_c-c, x_c+c] x [y_top-b, y_top]. The MLP is queried at each cell CENTER.

  * confine_to_cloak = False : a cloak cell (center inside annulus) fills its
    WHOLE rectangle with MLP material.
  * confine_to_cloak = True  : only the quadrature points that actually fall
    inside the triangular annulus get MLP material; the rest of the cell's
    rectangular footprint reverts to fixed background. This trims the material
    field exactly to the triangle -- essential on coarse grids where a cell
    rectangle spills out of the annulus.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.lines import Line2D

# ── geometry (params from configs/A_1x1_ceiling/basin_s88.yaml) ──────────
lam = 1.0
H = 4.305 * lam
a = 0.0774 * H        # inner (defect) apex depth  ~0.333
b = 3.0 * a           # outer apex depth           ~1.000
c = 0.1545 * H        # surface half-width         ~0.665
x_c, y_top = 0.0, 0.0 # place cloak centre at origin for plotting

N_X, N_Y = 10, 10

x_min, x_max = x_c - c, x_c + c
y_min, y_max = y_top - b, y_top
dx = (x_max - x_min) / N_X
dy = (y_max - y_min) / N_Y


def in_cloak(x, y):
    """Vectorised annulus membership test (matches TriangularCloakGeometry)."""
    depth = y_top - y
    r = np.abs(x - x_c) / c
    return (r <= 1.0) & (depth >= a * (1.0 - r)) & (depth <= b * (1.0 - r))


# ── cell centres + mask ──────────────────────────────────────────────────
cx = x_min + (np.arange(N_X) + 0.5) * dx
cy = y_min + (np.arange(N_Y) + 0.5) * dy
gx, gy = np.meshgrid(cx, cy, indexing="ij")          # (n_x, n_y)
cell_mask = in_cloak(gx, gy)                          # (n_x, n_y) bool

# ── high-res pixel grid, tagged with owning cell + annulus membership ────
res = 900
px = np.linspace(x_min, x_max, res)
py = np.linspace(y_min, y_max, res)
PX, PY = np.meshgrid(px, py)
ix = np.clip(((PX - x_min) / dx).astype(int), 0, N_X - 1)
iy = np.clip(((PY - y_min) / dy).astype(int), 0, N_Y - 1)
pix_in_annulus = in_cloak(PX, PY)
pix_center_in = cell_mask[ix, iy]                    # cell's center in annulus?
checker = (ix + iy) % 2                              # 2-tone to reveal cells

# triangle outlines
outer_tri = np.array([[x_c - c, y_top], [x_c + c, y_top], [x_c, y_top - b]])
inner_tri = np.array([[x_c - c, y_top], [x_c + c, y_top], [x_c, y_top - a]])

GREEN_A, GREEN_B = "#2e8b57", "#6fc296"   # two tones for the material field
BG = "#e9ecef"

fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), constrained_layout=True)

GRAY = np.array([0.62, 0.62, 0.62])

for ax, confine in zip(axes, (False, True)):
    fill = pix_in_annulus if confine else pix_center_in
    # paint material field as a single flat gray region
    canvas = np.ones((*checker.shape, 3)) * np.array([0.945, 0.949, 0.953])
    canvas[fill] = GRAY
    ax.imshow(canvas, origin="lower",
              extent=[x_min, x_max, y_min, y_max], aspect="equal",
              interpolation="nearest", zorder=0)

    # cell-decomposition borders for cloak cells (clip to annulus when confine)
    clip = Polygon(outer_tri, closed=True, transform=ax.transData) if confine else None
    for i in range(N_X):
        for j in range(N_Y):
            if not cell_mask[i, j]:
                continue
            rect = Rectangle((x_min + i * dx, y_min + j * dy), dx, dy,
                             fill=False, edgecolor="0.35", lw=0.7, zorder=1)
            ax.add_patch(rect)
            if clip is not None:
                rect.set_clip_path(clip)
    if confine:
        # cover borders that fall in the inner defect void with background colour
        ax.add_patch(Polygon(inner_tri, closed=True, facecolor="#f1f2f3",
                             edgecolor="none", zorder=2))

    # triangle outlines: outer annulus boundary + inner defect void
    ax.add_patch(Polygon(outer_tri, closed=True, fill=False,
                         edgecolor="#c0392b", lw=2.4, zorder=3))
    ax.add_patch(Polygon(inner_tri, closed=True, fill=False,
                         edgecolor="#8e44ad", lw=2.0, ls="--", zorder=3))

    # cell centres: green = MLP-driven (mask True), grey = background
    ax.scatter(gx[cell_mask], gy[cell_mask], s=42, c="#154734",
               edgecolors="white", linewidths=0.8, zorder=4)
    ax.scatter(gx[~cell_mask], gy[~cell_mask], s=26, c="#adb5bd",
               edgecolors="white", linewidths=0.6, zorder=4)

    n_cloak = int(cell_mask.sum())
    title = ("confine_to_cloak = True\n"
             "material trimmed to the triangular annulus"
             if confine else
             "confine_to_cloak = False (legacy)\n"
             "cloak cell fills its whole rectangle")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("depth  (free surface at top)")
    ax.text(0.5, -b * 1.02, f"{N_X}x{N_Y} grid   |   {n_cloak} cloak cells",
            ha="center", va="top", transform=ax.transData, fontsize=10,
            color="0.3")
    ax.set_xlim(x_min - 0.05, x_max + 0.05)
    ax.set_ylim(y_min - 0.12, y_top + 0.05)

# shared legend
handles = [
    Line2D([0], [0], color="#c0392b", lw=2.4, label="outer annulus (cloak) boundary"),
    Line2D([0], [0], color="#8e44ad", lw=2.0, ls="--", label="inner defect void (excluded)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#154734",
           markersize=9, label="cell centre in cloak  → MLP params"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#adb5bd",
           markersize=8, label="cell centre in background → fixed C_init"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="#9e9e9e",
           markersize=11, label="material assigned by MLP (cloak region)"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           frameon=False, bbox_to_anchor=(0.5, -0.03))

fig.suptitle(
    "Cell decomposition over the triangular Rayleigh-wave cloak "
    f"(a={a:.3f}, b={b:.3f}, c={c:.3f})",
    fontsize=15, fontweight="bold")

out = "cell_layout_triangular_confine.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
