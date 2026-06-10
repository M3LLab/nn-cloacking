"""Pixel-level validation of an optimised cloak via frequency sweep.

Companion to ``frequency_sweep.py``. Where ``frequency_sweep.py`` evaluates
the *homogenised* macro-cell solution (each cell carries a (C, ρ) pair),
this script:

  1. matches each cloak macro cell to its nearest dataset entry by (λ, μ, ρ),
  2. tiles those 50×50 binary microstructures into one fine-grained image
     spanning the cloak bbox,
  3. runs frequency-domain elastodynamics with material assigned at the
     *pixel* level (solid cement vs. void) and a refined mesh,
  4. records  ⟨|u|⟩ / ⟨|u_ref|⟩  on the free surface beyond the cloak,
  5. plots the result alongside any pre-existing obstacle/ideal/optimised
     CSVs so you can see how much performance the snap-to-dataset costs.

Usage
-----

    python scripts/frequency_sweep_validated.py \\
        configs/triangular_optimize_neural_flat2.yaml \\
        output/cell15_multifreq_cement/optimized_params.npz \\
        --fmin 0.7 --fmax 3.3 --fstep 0.1 \\
        --refinement-factor 12

The default refinement factor is bumped well above the config's so the mesh
actually resolves the 50×50 microstructure inside each macro cell.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import jax
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.absorbing import make_xi_profile
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import (
    find_embedded_eval_node_indices,
    make_fixed_surface_eval_points,
    transmitted_displacement_ratio,
)
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.optimize import get_top_surface_beyond_cloak_indices
from rayleigh_cloak.problem import RayleighCloakProblem, build_problem, _make_dirichlet_bc, _make_top_surface
from rayleigh_cloak.solver import _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── helpers duplicated from scripts/vis/tile_matched_microstructure.py ─
# (scripts/ isn't a Python package, so we copy rather than import).


def _resolve_grid(config_path: Path, n_cells: int) -> tuple[int, int]:
    import yaml
    if config_path.exists():
        cfg = yaml.safe_load(open(config_path)) or {}
        cells_cfg = cfg.get("cells", {}) or {}
        if "n_x" in cells_cfg and "n_y" in cells_cfg:
            nx, ny = int(cells_cfg["n_x"]), int(cells_cfg["n_y"])
            if nx * ny != n_cells:
                raise ValueError(f"n_x*n_y={nx*ny} != n_cells={n_cells}")
            return nx, ny
    for nx in range(int(np.sqrt(n_cells)), 0, -1):
        if n_cells % nx == 0:
            return nx, n_cells // nx
    raise ValueError(f"can't factor n_cells={n_cells}")


def _build_cloak_mask(config_path: Path, n_x: int, n_y: int):
    """Reproduce CellDecomposition.cloak_mask without pulling in the gmsh-loaded
    geometry module. Triangular and circular geometries supported."""
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
    centers = np.stack([gx.ravel(), gy.ravel()], axis=-1)

    if cfg.geometry_type == "triangular":
        depth = y_top - centers[:, 1]
        r = np.abs(centers[:, 0] - x_c) / c
        d1 = a * (1.0 - r)
        d2 = b * (1.0 - r)
        cloak_mask = (r <= 1.0) & (depth >= d1) & (depth <= d2)
    else:
        rad = np.sqrt((centers[:, 0] - x_c) ** 2 + (centers[:, 1] - y_c) ** 2)
        cloak_mask = (rad >= ri) & (rad <= rc)

    info = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}
    return cloak_mask, centers, info


def _tile_image(geoms: np.ndarray, n_x: int, n_y: int) -> np.ndarray:
    """Tile (n_cells, H, W) into (n_y*H, n_x*W) with y-up, cell_idx=ix*n_y+iy."""
    n_cells, H, W = geoms.shape
    assert n_cells == n_x * n_y
    canvas = np.zeros((n_y * H, n_x * W), dtype=geoms.dtype)
    for ix in range(n_x):
        for iy in range(n_y):
            idx = ix * n_y + iy
            row = n_y - 1 - iy
            col = ix
            canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = geoms[idx]
    return canvas


# ── pixel-level FEM problem ─────────────────────────────────────────


class PixelMaterialProblem(RayleighCloakProblem):
    """Same elastodynamics as ``RayleighCloakProblem`` but with C(x), ρ(x) read
    from a binary canvas inside the cloak.

    Required class attributes (set by the factory below):
        _canvas_jnp     : jnp.ndarray (H_pix, W_pix) of 0/1
        _cloak_bbox     : (x_min, x_max, y_min, y_max) — physical extent of canvas
        _C_void         : (2,2,2,2) jnp tensor for void
        _rho_void       : float density for void
    """

    def custom_init(self):
        geo = self._geometry
        C0 = self._C0
        rho0 = self._rho0
        canvas = type(self)._canvas_jnp                  # (H_pix, W_pix) jnp
        x_min, x_max, y_min, y_max = type(self)._cloak_bbox
        H_pix, W_pix = canvas.shape
        C_void = type(self)._C_void
        rho_void = type(self)._rho_void
        xi_fn = type(self).__dict__["_xi_fn"]

        inv_dx = 1.0 / (x_max - x_min)
        inv_dy = 1.0 / (y_max - y_min)

        def _pixel_at(x):
            # Sample canvas at physical (x,y). The canvas was tiled with y-up
            # (top of image = high y) by tile_matched_microstructure._tile_image,
            # so we flip y when going from physical to image-row coordinates.
            x_norm = (x[0] - x_min) * inv_dx
            y_norm = (x[1] - y_min) * inv_dy
            col = jnp.clip((x_norm * W_pix).astype(jnp.int32), 0, W_pix - 1)
            row = jnp.clip(((1.0 - y_norm) * H_pix).astype(jnp.int32), 0, H_pix - 1)
            return canvas[row, col]

        def _C_eff_pt(x):
            in_clk = geo.in_cloak(x)
            is_solid = _pixel_at(x) > 0.5
            C_pixel = jnp.where(is_solid, C0, C_void)
            return jnp.where(in_clk, C_pixel, C0)

        def _rho_eff_pt(x):
            in_clk = geo.in_cloak(x)
            is_solid = _pixel_at(x) > 0.5
            rho_pixel = jnp.where(is_solid, rho0, rho_void)
            return jnp.where(in_clk, rho_pixel, rho0)

        xi_qp = jax.vmap(jax.vmap(xi_fn))(self.physical_quad_points)
        self._xi_qp = xi_qp

        self.internal_vars = [
            jax.vmap(jax.vmap(_C_eff_pt))(self.physical_quad_points),
            jax.vmap(jax.vmap(_rho_eff_pt))(self.physical_quad_points),
            xi_qp,
        ]

    # set_params is a no-op for the validation problem (no live parameter sweep)
    def set_params(self, _params):
        pass


def build_pixel_problem(
    mesh,
    cfg,
    params: DerivedParams,
    geometry,
    canvas: np.ndarray,
    cloak_bbox: tuple[float, float, float, float],
    void_ratio: float = 1e-6,
) -> PixelMaterialProblem:
    """Same plumbing as ``build_problem`` but inserts pixel-level material."""
    C0 = C_iso(params.lam, params.mu)
    C_void = C_iso(params.lam * void_ratio, params.mu * void_ratio)
    rho_void = params.rho0 * void_ratio
    canvas_jnp = jnp.asarray(canvas, dtype=jnp.float32)

    ProblemCls = type("PixelMaterialProblemInstance", (PixelMaterialProblem,), {
        "_omega":        params.omega,
        "_geometry":     geometry,
        "_is_reference": False,
        "_C0":           C0,
        "_rho0":         params.rho0,
        "_xi_fn":        make_xi_profile(params),
        "_x_src":        params.x_src,
        "_sigma_src":    params.sigma_src,
        "_F0":           params.F0,
        "_cell_decomp":  None,
        "_n_C_params":   cfg.cells.n_C_params,
        "_source_type":  cfg.source.source_type,
        "_wave_type":    cfg.source.wave_type,
        "_lam_param":    params.lam,
        "_mu_param":     params.mu,
        "_canvas_jnp":   canvas_jnp,
        "_cloak_bbox":   cloak_bbox,
        "_C_void":       C_void,
        "_rho_void":     rho_void,
    })

    return ProblemCls(
        mesh=mesh,
        vec=4,
        dim=2,
        ele_type=cfg.mesh.ele_type,
        dirichlet_bc_info=_make_dirichlet_bc(params),
        location_fns=[_make_top_surface(params)],
    )


# ── microstructure assembly ─────────────────────────────────────────


# flat4 layout (rayleigh_cloak.materials.C_to_flatC):
#   [ C[0,0,0,0],  C[1,1,1,1],  C[0,1,0,1],  C[0,0,1,1] ]  =  [ C11, C22, C66, C12 ].
_FLAT4_DATASET_FIELDS = ("C11", "C22", "C66", "C12")


def build_canvas(
    optimized_params_npz: Path,
    dataset_h5: Path,
    config_path: Path,
    rho_weight: float = 1.0,
):
    """Match cloak cells to dataset entries and build the tiled binary canvas.

    Mirrors ``scripts.vis.tile_matched_microstructure`` but returns the canvas
    in-memory rather than saving a PNG.

    Matching feature space depends on ``n_C_params`` in the optimised params:
        2 → (λ, μ, ρ)                3-D
        4 → (C11, C22, C66, C12, ρ)  5-D, using dataset's orthotropic params.

    Returns
    -------
    canvas : (n_y*H, n_x*W) uint8
    (n_x, n_y) : macro-grid dimensions
    (H, W) : per-cell pixel dimensions (50, 50 from the CA dataset)
    cloak_bbox : (x_min, x_max, y_min, y_max) in physical coords
    matched_cell_C_flat : (n_cells, n_C_params) — dataset entry's values on cloak
        cells, optimiser's values elsewhere (background cement, no snapping).
    matched_cell_rho : (n_cells,) — same convention.
    diag : misc stats for logging
    """
    npz = np.load(optimized_params_npz)
    cell_C_flat = npz["cell_C_flat"]
    cell_rho = npz["cell_rho"]
    n_cells, n_C = cell_C_flat.shape
    if n_C not in (2, 4):
        raise SystemExit(f"this script handles n_C_params∈{{2, 4}}; got {n_C}")

    n_x, n_y = _resolve_grid(config_path, n_cells)
    cloak_mask, _centers, info = _build_cloak_mask(config_path, n_x, n_y)
    cloak_bbox = (info["x_min"], info["x_max"], info["y_min"], info["y_max"])
    cloak_indices = np.where(cloak_mask)[0]

    # Standardisation must match the optimisation-time GMM standardisation,
    # which used the dataset's own mean/std (not a stored .npz).
    with h5py.File(dataset_h5, "r") as f:
        rho_ds = f["rho"][:]
        H, W = f["cells"].shape[1:]
        if n_C == 2:
            lam_ds = f["lambda_"][:]
            mu_ds = f["mu"][:]
            X_ds = np.column_stack([lam_ds, mu_ds, rho_ds]).astype(np.float64)
        else:  # n_C == 4
            missing = [k for k in _FLAT4_DATASET_FIELDS if k not in f]
            if missing:
                raise SystemExit(
                    f"{dataset_h5}: missing orthotropic fields {missing}. "
                    "Run dataset/cellular_chiral/backfill_ortho_params.py first."
                )
            flat4_ds = np.column_stack([f[k][:] for k in _FLAT4_DATASET_FIELDS])
            X_ds = np.column_stack([flat4_ds, rho_ds]).astype(np.float64)

    mean = X_ds.mean(axis=0)
    std = X_ds.std(axis=0)
    Xs_ds = (X_ds - mean) / std
    Xs_ds[:, -1] *= rho_weight                    # ρ is always the last column

    if n_C == 2:
        X_q = np.column_stack([cell_C_flat[:, 0], cell_C_flat[:, 1], cell_rho])
    else:
        X_q = np.column_stack([cell_C_flat, cell_rho])
    Xs_q = (X_q[cloak_indices] - mean) / std
    Xs_q[:, -1] *= rho_weight

    # Brute-force NN — cheap at ~100 × 150k.
    a2 = np.sum(Xs_q ** 2, axis=1, keepdims=True)
    b2 = np.sum(Xs_ds ** 2, axis=1, keepdims=True).T
    d2 = a2 + b2 - 2.0 * (Xs_q @ Xs_ds.T)
    cloak_match_idx = np.argmin(d2, axis=1)
    cloak_match_d = np.sqrt(np.maximum(d2[np.arange(cloak_indices.size), cloak_match_idx], 0.0))

    unique_idx, inverse = np.unique(cloak_match_idx, return_inverse=True)
    with h5py.File(dataset_h5, "r") as f:
        unique_geoms = f["cells"][unique_idx.tolist()]
        if n_C == 2:
            matched_cloak_C_flat = np.column_stack([
                f["lambda_"][unique_idx.tolist()][inverse],
                f["mu"][unique_idx.tolist()][inverse],
            ])
        else:
            matched_cloak_C_flat = np.column_stack([
                f[k][unique_idx.tolist()][inverse] for k in _FLAT4_DATASET_FIELDS
            ])
        matched_cloak_rho = f["rho"][unique_idx.tolist()][inverse]
    cloak_geoms = unique_geoms[inverse]

    # Full (n_cells,…) matched arrays: cloak cells get the snapped dataset
    # value; non-cloak cells keep the optimiser's value (cement background,
    # no snapping). These feed the matched-homogenised sweep.
    matched_cell_C_flat = np.array(cell_C_flat, dtype=np.float64, copy=True)
    matched_cell_C_flat[cloak_indices] = matched_cloak_C_flat
    matched_cell_rho = np.array(cell_rho, dtype=np.float64, copy=True)
    matched_cell_rho[cloak_indices] = matched_cloak_rho

    # Outside-cloak macro cells: solid cement (canvas=1 → solid in PixelMaterialProblem).
    geoms = np.ones((n_cells, H, W), dtype=np.uint8)
    geoms[cloak_indices] = cloak_geoms
    canvas = _tile_image(geoms, n_x, n_y)

    diag = {
        "n_cloak": int(cloak_indices.size),
        "n_cells": n_cells,
        "match_d_median": float(np.median(cloak_match_d)),
        "match_d_mean": float(cloak_match_d.mean()),
        "match_d_max": float(cloak_match_d.max()),
        "n_unique_dataset_entries": int(unique_idx.size),
        "n_C_params": int(n_C),
    }
    return canvas, (n_x, n_y), (H, W), cloak_bbox, matched_cell_C_flat, matched_cell_rho, diag


def build_canvas_from_cell_designs(
    optimized_params_npz: Path,
    dataset_h5: Path,
    config_path: Path,
    cell_designs_dir: Path,
    rho_weight: float = 1.0,
):
    """Like ``build_canvas`` but tile the INVERSE-DESIGNED microstructures
    (``cell_designs/cell_XXX/canvas.npy``) instead of the raw nearest-dataset
    cells, and carry their optimised homogenised stiffness
    (``weights.npz['pred_flat4'/'pred_rho']``) into the matched sweep.

    Builds the NN baseline first (for the cloak mask, bbox, grid and any cloak
    cells that don't yet have an inverse design) then OVERRIDES each cloak cell
    that has a design.  Returns the same tuple as ``build_canvas``.
    """
    (canvas, (n_x, n_y), (H, W), cloak_bbox,
     matched_cell_C_flat, matched_cell_rho, diag) = build_canvas(
        optimized_params_npz, dataset_h5, config_path, rho_weight=rho_weight,
    )

    cloak_mask, _centers, _info = _build_cloak_mask(config_path, n_x, n_y)
    cloak_idx = np.where(cloak_mask)[0]

    n_have = 0
    missing = []
    for idx in cloak_idx:
        cdir = cell_designs_dir / f"cell_{int(idx):03d}"
        cpath, wpath = cdir / "canvas.npy", cdir / "weights.npz"
        if not (cpath.exists() and wpath.exists()):
            missing.append(int(idx))
            continue
        cell_canvas = np.load(str(cpath)).astype(canvas.dtype)   # (H, W) binary
        # tile mapping from _tile_image: idx = ix*n_y + iy ; row = n_y-1-iy ; col = ix
        ix, iy = int(idx) // n_y, int(idx) % n_y
        row, col = n_y - 1 - iy, ix
        canvas[row * H:(row + 1) * H, col * W:(col + 1) * W] = cell_canvas
        w = np.load(str(wpath))
        pf = np.asarray(w["pred_flat4"])                # [C11, C22, C12, C66]
        # -> optimiser/dataset flat order [C11, C22, C66, C12]
        matched_cell_C_flat[idx] = pf[[0, 1, 3, 2]]
        matched_cell_rho[idx] = float(w["pred_rho"])
        n_have += 1

    diag.update(source="cell_designs (inverse design)",
                n_inverse_cells_used=n_have, n_missing_fell_back_to_nn=len(missing))
    if missing:
        print(f"  WARNING: {len(missing)} cloak cells have no inverse design yet "
              f"(fell back to NN): {missing[:12]}{'…' if len(missing) > 12 else ''}")
    print(f"  inverse-design cells tiled: {n_have}/{len(cloak_idx)}")
    return (canvas, (n_x, n_y), (H, W), cloak_bbox,
            matched_cell_C_flat, matched_cell_rho, diag)


# ── frequency sweep helpers (mirror frequency_sweep.py) ─────────────


def _save_csv(csv_path: Path, f_stars, ratios) -> None:
    with open(csv_path, "w") as f:
        f.write("f_star,u_ratio\n")
        for fs, r in zip(f_stars, ratios):
            f.write(f"{fs:.4f},{r:.6f}\n")
    print(f"  CSV → {csv_path}")


def _load_csv(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["f_star"], data["u_ratio"]


def _make_config_at_fstar(base_config, f_star: float, refinement_factor: int | None):
    """Copy of base config with overridden f_star and (optionally) refinement_factor."""
    updates = {"domain": base_config.domain.model_copy(update={"f_star": float(f_star)})}
    if refinement_factor is not None:
        updates["mesh"] = base_config.mesh.model_copy(update={"refinement_factor": int(refinement_factor)})
    return base_config.model_copy(update=updates)


def _surface_indices_at_f(cloak_mesh, geometry, dp, kept_nodes, loss_cfg=None):
    """Same convention as scripts/frequency_sweep.py: prefer the embedded
    fixed-x eval nodes when ``loss_cfg.n_eval_points > 0``."""
    if loss_cfg is not None and int(loss_cfg.n_eval_points) > 0:
        eval_xs = make_fixed_surface_eval_points(
            geometry, dp, int(loss_cfg.n_eval_points),
            noise_sigma=float(loss_cfg.eval_noise_sigma),
            seed=int(loss_cfg.eval_noise_seed),
        )
        cs_idx = find_embedded_eval_node_indices(
            cloak_mesh.points, eval_xs, dp.y_top,
        )
        return cs_idx, kept_nodes[cs_idx]

    x_left = dp.x_off
    x_right = dp.x_off + dp.W
    cs_idx = get_top_surface_beyond_cloak_indices(
        cloak_mesh.points, geometry, dp.y_top, x_left, x_right,
    )
    return cs_idx, kept_nodes[cs_idx]


def run_matched_homogenised_sweep(
    base_config,
    f_stars,
    matched_cell_C_flat: np.ndarray,
    matched_cell_rho: np.ndarray,
    csv_path: Path,
    solver_opts: dict,
) -> None:
    """Macro-level homogenised sweep using the *snapped* per-cell stiffness.

    Each cloak macro cell carries the (λ, μ, ρ) or flat4 stiffness of its
    nearest dataset entry instead of the optimiser's continuous output. The
    delta versus the optimised sweep isolates the error introduced by snapping
    to the dataset manifold; the further delta versus the pixel-level
    validated sweep is the error of homogenisation itself.
    """
    print(f"\n>>> Matched homogenised sweep ({len(f_stars)} freqs)")

    cell_C_flat_jnp = jnp.array(matched_cell_C_flat)
    cell_rho_jnp = jnp.array(matched_cell_rho)
    matched_params = (cell_C_flat_jnp, cell_rho_jnp)

    ratios = []
    for f_star in f_stars:
        t0 = time.time()
        print(f"  f* = {f_star:.2f} ", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star, refinement_factor=None)
        dp = DerivedParams.from_config(config)
        geometry = _create_geometry(config, dp)

        full_mesh = generate_mesh_full(config, dp, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

        ref_result = solve_reference(config, mesh=full_mesh)

        cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
        problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
        problem.set_params(matched_params)
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_val = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(u_val, ref_result.u, cs_idx, rs_idx)
        print(f"ratio={ratio:.4f}  ({time.time()-t0:.1f}s)")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


def run_validated_sweep(
    base_config,
    f_stars,
    refinement_factor: int,
    canvas: np.ndarray,
    cloak_bbox,
    void_ratio: float,
    csv_path: Path,
    solver_opts: dict,
) -> None:
    """Sweep validated case: pixel-material cloak, refined mesh."""
    print(f"\n>>> Validated sweep ({len(f_stars)} freqs, refinement={refinement_factor})")
    ratios = []
    for f_star in f_stars:
        t0 = time.time()
        print(f"  f* = {f_star:.2f} ", end="", flush=True)
        config = _make_config_at_fstar(base_config, f_star, refinement_factor)
        dp = DerivedParams.from_config(config)
        geometry = _create_geometry(config, dp)

        # Mesh per-frequency: domain dimensions depend on f_star (wavelength).
        full_mesh = generate_mesh_full(config, dp, geometry)
        cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
        print(f"[mesh nodes={len(cloak_mesh.points)}, cells={cloak_mesh.cells.shape[0]}] ", end="", flush=True)

        ref_result = solve_reference(config, mesh=full_mesh)

        problem = build_pixel_problem(
            cloak_mesh, config, dp, geometry,
            canvas=canvas, cloak_bbox=cloak_bbox, void_ratio=void_ratio,
        )
        sol_list = jax_fem_solver(problem, solver_options=solver_opts)
        u_val = np.asarray(sol_list[0])

        cs_idx, rs_idx = _surface_indices_at_f(
            cloak_mesh, geometry, dp, kept_nodes, loss_cfg=base_config.loss,
        )
        ratio = transmitted_displacement_ratio(u_val, ref_result.u, cs_idx, rs_idx)
        print(f"ratio={ratio:.4f}  ({time.time()-t0:.1f}s)")
        ratios.append(ratio)

    _save_csv(csv_path, f_stars.tolist(), ratios)


# ── plotting ────────────────────────────────────────────────────────


_CASE_STYLES = {
    "obstacle":  {"color": "black", "ls": "--", "marker": "s", "label": "Obstacle"},
    "ideal":     {"color": "C3",    "ls": "-",  "marker": "o", "label": "Ideal Cloak"},
    "optimized": {"color": "C0",    "ls": "-",  "marker": "D", "label": "Optimized (homogenised)"},
    "matched":   {"color": "C1",    "ls": "-",  "marker": "v", "label": "Matched (snapped homogenised)"},
    "validated": {"color": "C2",    "ls": "-",  "marker": "^", "label": "Validated (pixel-level)"},
}


def _load_comparison_module():
    """Load ``scripts/vis/plot_frequency_sweep_comparison.py`` by path.

    ``scripts/`` isn't an importable package, so we load the sibling vis module
    explicitly rather than copying its annotated-plot logic here.
    """
    import importlib.util
    mod_path = Path(__file__).resolve().parent / "vis" / "plot_frequency_sweep_comparison.py"
    spec = importlib.util.spec_from_file_location("plot_frequency_sweep_comparison", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plot_results(case_csvs: dict[str, Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    f_max = 0.0
    y_max = 0.0
    any_plotted = False
    for case, csv_path in case_csvs.items():
        if not csv_path.exists():
            continue
        any_plotted = True
        f_vals, ratios = _load_csv(csv_path)
        s = _CASE_STYLES[case]
        ax.plot(f_vals, ratios, color=s["color"], ls=s["ls"], marker=s["marker"],
                lw=1.5, markersize=4, label=s["label"])
        f_max = max(f_max, f_vals.max())
        y_max = max(y_max, ratios.max())

    if not any_plotted:
        plt.close(fig)
        return

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Cloaking performance vs frequency — pixel-level validation")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_max + 0.1)
    ax.set_ylim(0, max(y_max * 1.1, 1.15))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out_path}")


# ── main ────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", help="Path to YAML config file")
    p.add_argument("params", help="Path to optimized_params.npz")
    p.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5",
                   help="Stiffness HDF5 (cells + lambda_/mu/rho).")
    p.add_argument("--fmin", type=float, default=0.7)
    p.add_argument("--fmax", type=float, default=3.3)
    p.add_argument("--fstep", type=float, default=0.1)
    p.add_argument("--refinement-factor", type=int, default=25,
                   help="Override mesh.refinement_factor. Default 25 gives ~1 "
                        "FEM element per micro-pixel side (50-pixel-wide macro "
                        "cell ≈ 50 elements per side), which is the lowest "
                        "value at which the validation actually resolves the "
                        "microstructure features rather than aliasing them. "
                        "Tested on 64 GB box: ~140 k nodes, ~270 s/freq, "
                        "<5 GB peak. Configs typically use 3 — much too coarse "
                        "for pixel validation but sufficient for the "
                        "homogenised optimisation.")
    p.add_argument("--void-ratio", type=float, default=1e-6,
                   help="E_void / E_cement ratio for ersatz-material void.")
    p.add_argument("--rho-weight", type=float, default=1.0,
                   help="Weight on standardised ρ in the matching distance.")
    p.add_argument("--cell-designs", type=Path, default=None,
                   help="Directory of inverse-designed cells (cell_XXX/canvas.npy + "
                        "weights.npz). When given, tile THESE microstructures (and use "
                        "their pred_flat4/pred_rho for the matched sweep) instead of the "
                        "raw nearest-dataset cells. Cloak cells without a design fall back "
                        "to NN. Use a distinct -o to keep NN-baseline CSVs separate.")
    p.add_argument("-f", "--force", action="store_true",
                   help="Re-run solves even if existing CSVs are present.")
    p.add_argument("--no-matched", action="store_true",
                   help="Skip the matched (snapped) homogenised sweep.")
    p.add_argument("--no-validated", action="store_true",
                   help="Skip the pixel-level validated sweep.")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: <params dir>).")
    args = p.parse_args()

    base_config = load_config(args.config)
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.params).parent
    out_dir.mkdir(exist_ok=True, parents=True)

    csv_paths = {
        "obstacle":  out_dir / "frequency_sweep_obstacle.csv",
        "ideal":     out_dir / "frequency_sweep_ideal.csv",
        "optimized": out_dir / "frequency_sweep_optimized.csv",
        "matched":   out_dir / "frequency_sweep_matched.csv",
        "validated": out_dir / "frequency_sweep_validated.csv",
    }

    # ── build the canvas (matching) ─────────────────────────────────
    if args.cell_designs is not None:
        print(f"=== Tiling INVERSE-DESIGNED cells from {args.cell_designs} ===")
        canvas, (n_x, n_y), (H_pix, W_pix), cloak_bbox, matched_cell_C_flat, matched_cell_rho, diag = (
            build_canvas_from_cell_designs(
                Path(args.params), Path(args.dataset), Path(args.config),
                Path(args.cell_designs), rho_weight=args.rho_weight,
            )
        )
    else:
        print("=== Matching cloak cells & assembling canvas (nearest dataset cell) ===")
        canvas, (n_x, n_y), (H_pix, W_pix), cloak_bbox, matched_cell_C_flat, matched_cell_rho, diag = build_canvas(
            Path(args.params), Path(args.dataset), Path(args.config),
            rho_weight=args.rho_weight,
        )
    print(
        f"canvas shape: {canvas.shape}  ({n_y}×{H_pix}, {n_x}×{W_pix})  "
        f"cloak cells: {diag['n_cloak']}/{diag['n_cells']}  "
        f"unique dataset entries used: {diag['n_unique_dataset_entries']}\n"
        f"match-distance (std-L2): median={diag['match_d_median']:.3f}, "
        f"mean={diag['match_d_mean']:.3f}, max={diag['match_d_max']:.3f}  "
        f"(n_C_params={diag['n_C_params']})"
    )

    solver_opts = {
        "petsc_solver": {
            "ksp_type": base_config.solver.ksp_type,
            "pc_type": base_config.solver.pc_type,
        }
    }
    f_stars = np.arange(args.fmin, args.fmax + 0.5 * args.fstep, args.fstep)

    # ── matched homogenised sweep ───────────────────────────────────
    if not args.no_matched:
        if csv_paths["matched"].exists() and not args.force:
            print(f"matched CSV exists at {csv_paths['matched']}; skipping (use -f).")
        else:
            run_matched_homogenised_sweep(
                base_config=base_config,
                f_stars=f_stars,
                matched_cell_C_flat=matched_cell_C_flat,
                matched_cell_rho=matched_cell_rho,
                csv_path=csv_paths["matched"],
                solver_opts=solver_opts,
            )

    # ── pixel-level validated sweep ─────────────────────────────────
    if not args.no_validated:
        if csv_paths["validated"].exists() and not args.force:
            print(f"validated CSV exists at {csv_paths['validated']}; skipping (use -f).")
        else:
            run_validated_sweep(
                base_config=base_config,
                f_stars=f_stars,
                refinement_factor=args.refinement_factor,
                canvas=canvas,
                cloak_bbox=cloak_bbox,
                void_ratio=args.void_ratio,
                csv_path=csv_paths["validated"],
                solver_opts=solver_opts,
            )

    # Plot all available cases together.
    plot_results(csv_paths, out_dir / "frequency_sweep_validated.png")

    # Annotated head-to-head comparison — only worth drawing when at least two
    # of the directly-comparable cases (matched / optimized / validated) have a
    # CSV present (whether produced just now or by a prior frequency_sweep.py run).
    comparison_cases = ("optimized", "matched", "validated")
    available = [c for c in comparison_cases if csv_paths[c].exists()]
    if len(available) >= 2:
        _load_comparison_module().plot_comparison(
            out_dir, train_fstar=base_config.domain.f_star,
        )
    else:
        print(f"comparison plot skipped: only {available} present "
              f"(need ≥2 of {list(comparison_cases)}).")


if __name__ == "__main__":
    main()
