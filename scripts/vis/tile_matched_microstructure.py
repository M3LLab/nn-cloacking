"""Match each optimised macro cell to its nearest dataset entry and tile the
50×50 microstructures into a single image.

The feature vector used for matching depends on ``n_C_params`` in the run's
config:

* ``n_C_params=2`` — match on (λ, μ, ρ) (standardised L2). This is the same
  feature space the GMM regulariser fits, so it stays consistent with how the
  optimiser saw the dataset manifold.
* ``n_C_params=4`` — match on the four flat4 stiffness components
  (C[0,0,0,0], C[1,1,1,1], C[0,1,0,1], C[0,0,1,1]) plus ρ. The dataset's
  augmented-Voigt 4×4 ``C_eff`` is projected onto those same indices so the
  comparison is component-wise consistent with the optimiser's parameterisation.

The matched binary microstructure is laid out at the same (ix, iy) location the
macro cell occupies. The side-car ``.npz`` carries the matched per-cell stiffness
(``matched_cell_C_flat`` of shape ``(n_cells, n_C_params)``) and density
(``matched_cell_rho``) so downstream homogenised-FEM scripts can re-solve with
the snapped dataset values instead of the optimiser's continuous output.

Usage
-----

    python -m scripts.vis.tile_matched_microstructure \
        -p output/cell15_multifreq_cement/optimized_params.npz \
        -d output/ca_bulk_squared/stiffness.h5 \
        -c output/cell15_multifreq_cement/config.yaml \
        -o output/cell15_multifreq_cement/tiled_microstructure.png
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

# Reading a possibly-still-being-written h5 is fine without locks.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


def _build_cloak_mask(config_path: Path, n_x: int, n_y: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Reproduce CellDecomposition.cloak_mask + cell_centers from the run's config.

    We import the project's ``DerivedParams`` (lightweight — no gmsh/jax) and
    inline the geometry's ``in_cloak`` / ``in_defect`` formulas so this script
    doesn't pull in the gmsh-loaded geometry module.

    Returns ``(cloak_mask (n_cells,), defect_mask (n_cells,), cell_centers
    (n_cells, 2), info)`` where info carries the cloak bbox for optional
    rendering.
    """
    # Project's pydantic config + DerivedParams.
    from rayleigh_cloak.config import load_config, DerivedParams

    cfg = load_config(config_path)
    dp = DerivedParams.from_config(cfg)

    if cfg.geometry_type == "triangular":
        x_c, y_top = dp.x_c, dp.y_top
        a, b, c = dp.a, dp.b, dp.c
        x_min, x_max = x_c - c, x_c + c
        y_min, y_max = y_top - b, y_top
    elif cfg.geometry_type == "circular":
        x_c, y_c = dp.x_c, dp.y_c
        ri, rc = dp.ri, dp.rc
        x_min, x_max = x_c - rc, x_c + rc
        y_min, y_max = y_c - rc, y_c + rc
    else:
        raise ValueError(f"unsupported geometry_type={cfg.geometry_type!r}")

    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y
    cx = x_min + (np.arange(n_x) + 0.5) * cell_dx
    cy = y_min + (np.arange(n_y) + 0.5) * cell_dy
    gx, gy = np.meshgrid(cx, cy, indexing="ij")
    cell_centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (n_cells, 2)

    if cfg.geometry_type == "triangular":
        depth = y_top - cell_centers[:, 1]
        r = np.abs(cell_centers[:, 0] - x_c) / c
        d1 = a * (1.0 - r)
        d2 = b * (1.0 - r)
        cloak_mask = (r <= 1.0) & (depth >= d1) & (depth <= d2)
        defect_mask = (r <= 1.0) & (depth >= 0.0) & (depth <= d1)
    else:  # circular
        rad = np.sqrt((cell_centers[:, 0] - x_c) ** 2 + (cell_centers[:, 1] - y_c) ** 2)
        cloak_mask = (rad >= ri) & (rad <= rc)
        defect_mask = rad < ri

    info = {
        "geometry_type": cfg.geometry_type,
        "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
    }
    return cloak_mask, defect_mask, cell_centers, info


def _flat2_to_lame(cell_C_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ``n_C_params=2`` the flat layout is already [λ, μ]."""
    return cell_C_flat[:, 0], cell_C_flat[:, 1]


# flat4 ↔ orthotropic engineering constants (D2 squared assembly):
#
# ``rayleigh_cloak.materials.C_to_flatC`` lays the 4-component flat array as
#   [ C[0,0,0,0],  C[1,1,1,1],  C[0,1,0,1],  C[0,0,1,1] ]
# which, in the engineering-constant naming used by ``backfill_ortho_params.py``,
# is
#   [ C11, C22, C66, C12 ].
#
# The dataset exposes C11/C22/C12/C66 as 1-D arrays after the backfill, so we
# just stack them in that order.
_FLAT4_DATASET_FIELDS = ("C11", "C22", "C66", "C12")


def _resolve_grid(config_path: Path | None, n_cells: int) -> tuple[int, int]:
    """Return (n_x, n_y) for the macro grid.

    Reads from the run's ``config.yaml`` if provided. Otherwise tries to
    factor ``n_cells`` into the closest-to-square pair as a last resort.
    """
    if config_path is not None and config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        cells_cfg = cfg.get("cells", {}) or {}
        if "n_x" in cells_cfg and "n_y" in cells_cfg:
            nx, ny = int(cells_cfg["n_x"]), int(cells_cfg["n_y"])
            if nx * ny != n_cells:
                raise ValueError(
                    f"config.yaml says n_x*n_y = {nx*ny} but optimized_params has "
                    f"{n_cells} cells; refusing to guess."
                )
            return nx, ny
    # Fallback: factorise.
    for nx in range(int(np.sqrt(n_cells)), 0, -1):
        if n_cells % nx == 0:
            ny = n_cells // nx
            print(f"WARNING: no config.yaml; guessing n_x={nx}, n_y={ny}")
            return nx, ny
    raise ValueError(f"can't factor n_cells={n_cells} and no config given")


def _tile_rgb(
    geoms: np.ndarray,           # (n_cells, H, W) uint8 — only meaningful where cloak_mask=True
    cloak_mask: np.ndarray,      # (n_cells,) bool
    defect_mask: np.ndarray,     # (n_cells,) bool
    n_x: int,
    n_y: int,
    bg_rgb: np.ndarray,          # (3,) uint8 — outside-cloak/defect color
    fg_rgb: np.ndarray,          # (3,) uint8 — solid material color
    void_rgb: np.ndarray,        # (3,) uint8 — void color *inside* cloak microstructures
    defect_rgb: np.ndarray,      # (3,) uint8 — solid color for the defect region
) -> np.ndarray:
    """Tile per-cell content into an RGB image of shape (n_y*H, n_x*W, 3).

    Cell index convention is ``cell_idx = ix * n_y + iy`` (the codebase's
    CellDecomposition convention). Cell (ix, iy) is placed at row (n_y-1-iy)
    so that y-up is up in the saved image. Three categories per cell:
      * cloak  → render the matched microstructure (fg_rgb / void_rgb)
      * defect → solid defect_rgb
      * outside → solid bg_rgb
    """
    n_cells, H, W = geoms.shape
    assert n_cells == n_x * n_y, f"{n_cells} != {n_x}*{n_y}"
    canvas = np.empty((n_y * H, n_x * W, 3), dtype=np.uint8)
    canvas[..., :] = bg_rgb                                    # default = outside
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            row = n_y - 1 - iy        # y-up
            col = ix
            sl = (slice(row * H, (row + 1) * H), slice(col * W, (col + 1) * W))
            if cloak_mask[idx]:
                tile = geoms[idx]                              # (H, W) binary
                rgb = np.where(tile[..., None].astype(bool), fg_rgb, void_rgb)
                canvas[sl[0], sl[1], :] = rgb
            elif defect_mask[idx]:
                canvas[sl[0], sl[1], :] = defect_rgb
            # else: leave as bg_rgb
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-p", "--params", type=Path, required=True,
                    help="optimized_params.npz with cell_C_flat (n,2) and cell_rho (n,)")
    ap.add_argument("-d", "--dataset", type=Path,
                    default=Path("output/ca_bulk_squared/stiffness.h5"),
                    help="Stiffness HDF5 with lambda_, mu, rho, cells fields.")
    ap.add_argument("-c", "--config", type=Path, default=None,
                    help="Run config.yaml to read n_x/n_y from. Default: alongside --params.")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output PNG path (default: <params dir>/tiled_microstructure.png).")
    ap.add_argument("--rho-weight", type=float, default=1.0,
                    help="Multiplier on the standardised ρ component when computing distance. "
                         "1.0 treats ρ on equal footing with λ, μ (recommended). Set <1 to "
                         "downweight density and let stiffness drive the match.")
    ap.add_argument("--rng-seed", type=int, default=0,
                    help="Seed for tie-breaking when multiple entries are exactly equidistant.")
    args = ap.parse_args()

    # ── load optimisation result ────────────────────────────────────
    if not args.params.exists():
        raise SystemExit(f"missing params: {args.params}")
    npz = np.load(args.params)
    if "cell_C_flat" not in npz or "cell_rho" not in npz:
        raise SystemExit(
            f"{args.params} must contain cell_C_flat and cell_rho "
            f"(found {npz.files})"
        )
    cell_C_flat = npz["cell_C_flat"]
    cell_rho = npz["cell_rho"]
    n_cells, n_C_params = cell_C_flat.shape
    if n_C_params not in (2, 4):
        raise SystemExit(
            f"this script handles n_C_params∈{{2, 4}}; got {n_C_params}. "
            "Extend the feature-extraction block below to handle 6/10/16."
        )
    print(f"loaded {n_cells} optimised cells from {args.params} (n_C_params={n_C_params})")

    # ── grid layout + cloak mask ────────────────────────────────────
    config_path = args.config if args.config is not None else args.params.parent / "config.yaml"
    n_x, n_y = _resolve_grid(config_path, n_cells)
    print(f"grid: n_x={n_x}, n_y={n_y}")

    if not config_path.exists():
        raise SystemExit(
            f"--config {config_path} not found; needed to compute cloak_mask. "
            "Pass --config explicitly."
        )
    cloak_mask, defect_mask, _cell_centers, _geo_info = _build_cloak_mask(config_path, n_x, n_y)
    n_cloak = int(cloak_mask.sum())
    n_defect = int(defect_mask.sum())
    print(f"cloak cells: {n_cloak}/{n_cells}  ({n_cloak/n_cells:.1%})")
    print(f"defect cells: {n_defect}/{n_cells}  ({n_defect/n_cells:.1%})")

    # ── load dataset and build matching feature vectors ─────────────
    # Two matching modes, keyed by n_C_params:
    #   2 → match in (λ, μ, ρ)            (3-D)
    #   4 → match in (flat4 stiffness, ρ)  (5-D); dataset's augmented-Voigt
    #       C_eff is projected onto the flat4 layout.
    print(f"loading dataset {args.dataset} ...")
    t0 = time.time()
    with h5py.File(args.dataset, "r") as f:
        rho_ds = f["rho"][:]
        if n_C_params == 2:
            lam_ds = f["lambda_"][:]
            mu_ds = f["mu"][:]
            X_ds = np.column_stack([lam_ds, mu_ds, rho_ds]).astype(np.float64)
            feature_names = ["lambda", "mu", "rho"]
        else:  # n_C_params == 4
            missing = [k for k in _FLAT4_DATASET_FIELDS if k not in f]
            if missing:
                raise SystemExit(
                    f"{args.dataset}: missing orthotropic fields {missing}. "
                    "Run dataset/cellular_chiral/backfill_ortho_params.py to "
                    "populate C11/C22/C12/C66 from C_eff first."
                )
            flat4_ds = np.column_stack([f[k][:] for k in _FLAT4_DATASET_FIELDS])
            X_ds = np.column_stack([flat4_ds, rho_ds]).astype(np.float64)
            feature_names = [*_FLAT4_DATASET_FIELDS, "rho"]
        n_ds = X_ds.shape[0]
    print(f"  {n_ds} dataset cells, feature dim={X_ds.shape[1]} ({feature_names})  ({time.time()-t0:.1f}s)")

    mean = X_ds.mean(axis=0)
    std = X_ds.std(axis=0)
    Xs_ds = (X_ds - mean) / std

    # Query: per-cell (flat4 or λ/μ) + ρ, then standardised the same way.
    if n_C_params == 2:
        X_q_full = np.column_stack([cell_C_flat[:, 0], cell_C_flat[:, 1], cell_rho]).astype(np.float64)
    else:
        X_q_full = np.column_stack([cell_C_flat, cell_rho]).astype(np.float64)  # (n_cells, 5)

    cloak_indices = np.where(cloak_mask)[0]
    X_q = X_q_full[cloak_indices]
    Xs_q = (X_q - mean) / std
    # ρ is always the last column under both modes.
    Xs_q[:, -1] *= args.rho_weight
    Xs_ds_w = Xs_ds.copy()
    Xs_ds_w[:, -1] *= args.rho_weight

    # ── nearest neighbours (brute force; fine at 90-ish × 156k) ─────
    print(f"computing nearest neighbours for {cloak_indices.size} cloak cells ...")
    t0 = time.time()
    a2 = np.sum(Xs_q * Xs_q, axis=1, keepdims=True)               # (n_cloak, 1)
    b2 = np.sum(Xs_ds_w * Xs_ds_w, axis=1, keepdims=True).T       # (1, n_ds)
    cross = Xs_q @ Xs_ds_w.T                                       # (n_cloak, n_ds)
    d2 = a2 + b2 - 2.0 * cross
    cloak_match_idx = np.argmin(d2, axis=1).astype(np.int64)      # (n_cloak,)
    cloak_match_d = np.sqrt(np.maximum(d2[np.arange(cloak_indices.size), cloak_match_idx], 0.0))
    print(f"  done ({time.time()-t0:.1f}s)")
    print(
        f"  match distance (standardised L2):  mean={cloak_match_d.mean():.3f}  "
        f"median={np.median(cloak_match_d):.3f}  max={cloak_match_d.max():.3f}"
    )

    # ── pull the matched binary geometries + matched stiffness/density ──
    # h5py fancy-indexing requires strictly-increasing unique indices, so we
    # de-duplicate (multiple macro cells can match the same dataset entry)
    # and then scatter back by inverse map.
    unique_idx, inverse = np.unique(cloak_match_idx, return_inverse=True)
    print(f"reading {unique_idx.size} unique matched microstructures ...")
    t0 = time.time()
    with h5py.File(args.dataset, "r") as f:
        unique_geoms = f["cells"][unique_idx.tolist()]            # (n_unique, H, W)
        # Pull the matched dataset properties at the same indices so the
        # side-car npz can feed a downstream "matched homogenised" sweep.
        if n_C_params == 2:
            matched_lam = f["lambda_"][unique_idx.tolist()][inverse]
            matched_mu = f["mu"][unique_idx.tolist()][inverse]
            matched_cell_C_flat_cloak = np.column_stack([matched_lam, matched_mu])
        else:  # n_C_params == 4
            matched_cell_C_flat_cloak = np.column_stack([
                f[k][unique_idx.tolist()][inverse] for k in _FLAT4_DATASET_FIELDS
            ])  # (n_cloak, 4)
        matched_rho_cloak = f["rho"][unique_idx.tolist()][inverse]
    cloak_geoms = unique_geoms[inverse]                            # (n_cloak, H, W)
    print(f"  done ({time.time()-t0:.1f}s)")

    # Build full (n_cells,…) matched arrays. Non-cloak cells keep the optimiser's
    # original values (those are background cement, no snapping needed).
    matched_cell_C_flat = np.array(cell_C_flat, dtype=np.float64, copy=True)
    matched_cell_C_flat[cloak_indices] = matched_cell_C_flat_cloak
    matched_cell_rho = np.array(cell_rho, dtype=np.float64, copy=True)
    matched_cell_rho[cloak_indices] = matched_rho_cloak

    # Build a full (n_cells, H, W) array — non-cloak cells stay zero (= white
    # in cmap='gray_r'), so they show as empty in the tiled plot.
    H, W = cloak_geoms.shape[1:]
    geoms = np.zeros((n_cells, H, W), dtype=np.uint8)
    geoms[cloak_indices] = cloak_geoms

    # Per-cell match index for the side-car (-1 for non-cloak cells).
    match_idx = -np.ones(n_cells, dtype=np.int64)
    match_idx[cloak_indices] = cloak_match_idx
    match_d = np.full(n_cells, np.nan)
    match_d[cloak_indices] = cloak_match_d

    # ── tile + plot ────────────────────────────────────────────────
    # Region colors:
    #   outside (neither cloak nor defect) → warm tan / brown-yellow
    #   defect (obstacle being cloaked)    → white (uncolored)
    #   cloak material (microstructure 1)  → dark gray
    #   cloak void (microstructure 0)      → off-white (so brown stays unique
    #                                         to truly-outside cells)
    bg_rgb = np.array([238, 232, 170], dtype=np.uint8)   # #c8a063
    fg_rgb = np.array([ 90,  90,  90], dtype=np.uint8)   # #2e2e2e
    void_rgb = np.array([245, 245, 240], dtype=np.uint8) # #f5f5f0
    defect_rgb = np.array([255, 255, 255], dtype=np.uint8)

    canvas = _tile_rgb(geoms, cloak_mask, defect_mask, n_x, n_y,
                       bg_rgb, fg_rgb, void_rgb, defect_rgb)

    # Per-cell match-quality grid, laid out the same way _tile_rgb lays cells:
    # (n_y, n_x) with row 0 at the top (y-up).
    quality_grid = np.full((n_y, n_x), np.nan)
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            row = n_y - 1 - iy
            quality_grid[row, ix] = match_d[idx]

    out_path = args.output if args.output else args.params.parent / "tiled_microstructure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(8.0, 0.5 * n_x) + max(4.0, 0.25 * n_x)
    fig_h = max(6.0, 0.5 * n_y)
    fig, (ax, ax_q) = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [2.0, 1.0]},
    )
    ax.imshow(canvas, interpolation="nearest", aspect="equal")
    # Cell-boundary grid lines
    for ix in range(1, n_x):
        ax.axvline(ix * W - 0.5, color="C3", linewidth=0.4, alpha=0.6)
    for iy in range(1, n_y):
        ax.axhline(iy * H - 0.5, color="C3", linewidth=0.4, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{args.params.parent.name} — {n_x}×{n_y} macro cells, "
        f"cloak cells replaced by nearest dataset microstructure ({H}×{W})\n",
        # f"cloak={cloak_indices.size}/{n_cells}   "
        # f"match distance (std-L2): mean={cloak_match_d.mean():.2f}, "
        # f"median={np.median(cloak_match_d):.2f}"
        fontsize=11,
    )

    # Match-quality heatmap: green = small distance (good match), red = large
    # (bad match), yellow in between. Non-cloak cells are NaN and rendered with
    # the colormap's "bad" color (light gray) so they read as "not applicable".
    qcmap = plt.get_cmap("RdYlGn_r").copy()
    qcmap.set_bad(color="#dddddd")
    vmax = float(np.nanpercentile(cloak_match_d, 95)) if cloak_match_d.size else 1.0
    vmax = max(vmax, float(cloak_match_d.min()) + 1e-6) if cloak_match_d.size else 1.0
    im_q = ax_q.imshow(quality_grid, cmap=qcmap, interpolation="nearest",
                       aspect="equal", vmin=0.0, vmax=vmax)
    ax_q.set_xticks([])
    ax_q.set_yticks([])
    ax_q.set_title(
        f"match quality (std-L2 distance in {', '.join(feature_names)})",
        fontsize=11,
    )
    cbar = fig.colorbar(im_q, ax=ax_q, fraction=0.046, pad=0.04)
    cbar.set_label("standardised L2 distance")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved {out_path}")

    # Save the matching alongside the figure for reuse.
    side_npz = out_path.with_suffix(".npz")
    np.savez(
        side_npz,
        match_idx=match_idx,                       # -1 for non-cloak cells
        match_distance_std=match_d,
        cloak_mask=cloak_mask,
        n_x=np.int64(n_x),
        n_y=np.int64(n_y),
        H=np.int64(H),
        W=np.int64(W),
        feature_mean=mean,
        feature_std=std,
        feature_names=np.array(feature_names),
        rho_weight=np.float64(args.rho_weight),
        n_C_params=np.int64(n_C_params),
        # Per-cell snapped stiffness/density: cloak cells take the matched
        # dataset entry's value; non-cloak cells keep the optimiser's value
        # (they're cement background — no snapping). Downstream scripts can
        # plug these straight into the homogenised FEM solver.
        matched_cell_C_flat=matched_cell_C_flat,
        matched_cell_rho=matched_cell_rho,
    )
    print(f"saved {side_npz}")


if __name__ == "__main__":
    main()
