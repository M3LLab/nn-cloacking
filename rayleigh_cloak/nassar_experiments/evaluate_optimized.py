"""Evaluate cloaking distortion for optimized cell parameters.

Computes the three CloakingLoss metrics:
  - dist_boundary: distortion % on all four physical boundaries
  - dist_right:    distortion % on right physical boundary only
  - dist_outside:  distortion % over all physical nodes outside cloak

Usage::

    python -m rayleigh_cloak.nassar_experiments.evaluate_optimized \
        configs/circular_optimize.yaml \
        output/circular_optimize_16c/optimized_params.npz
"""

from __future__ import annotations

import sys

import numpy as np
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import compute_cloaking_loss
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import SolutionResult, _create_geometry, solve_reference

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


def evaluate(config_path: str, params_path: str) -> None:
    config = load_config(config_path)
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    # --- mesh + reference ---
    print("=== Generating mesh ===")
    full_mesh = generate_mesh_full(config, dp, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, {len(full_mesh.cells)} elements")

    print("=== Solving reference ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    print("=== Extracting submesh ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, {len(cloak_mesh.cells)} elements")

    # --- cell decomposition ---
    print("=== Setting up cells ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(dp.lam, dp.mu)
    CellMaterial(
        geometry, C0, dp.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
    )

    # --- load optimized params ---
    print(f"=== Loading optimized params from {params_path} ===")
    data = np.load(params_path)
    opt_params = (jnp.array(data["cell_C_flat"]), jnp.array(data["cell_rho"]))
    print(f"  cell_C_flat: {opt_params[0].shape}, cell_rho: {opt_params[1].shape}")

    # --- forward solve ---
    print("=== Forward solve with optimized params ===")
    problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
    problem.set_params(opt_params)

    solver_opts = {
        "petsc_solver": {
            "ksp_type": config.solver.ksp_type,
            "pc_type": config.solver.pc_type,
        }
    }
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u_cloak = np.asarray(sol_list[0])

    # --- compute CloakingLoss ---
    cloak_result = SolutionResult(
        u=u_cloak,
        mesh=cloak_mesh,
        config=config,
        params=dp,
        full_mesh=full_mesh,
        kept_nodes=kept_nodes,
    )

    loss = compute_cloaking_loss(cloak_result, ref_result, geometry)

    print("\n" + "=" * 60)
    print("CLOAKING DISTORTION (optimized params)")
    print("=" * 60)
    print(f"  dist_boundary (all 4 sides):  {loss.dist_boundary:.2f}%  ({loss.n_boundary} nodes)")
    print(f"  dist_right    (right side):   {loss.dist_right:.2f}%  ({loss.n_right} nodes)")
    print(f"  dist_outside  (outside cloak):{loss.dist_outside:.2f}%  ({loss.n_outside} nodes)")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <config.yaml> <optimized_params.npz>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
