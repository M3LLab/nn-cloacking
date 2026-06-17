import dataclasses

import numpy as np
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import _create_geometry, solve, solve_reference, jax_fem_solver, SolutionResult
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.loss import compute_cloaking_loss

import logging
logging.getLogger("jax_fem").setLevel(logging.WARNING)


cfg = load_config('configs/nassar.yaml')

# default parameters (from config)
#   nx_phys: 160
#   ny_phys: 160

# test for different mesh resolutions:
for mesh_res in [250, 200, 160, 120, 80, 60, 40]:
    cfg.mesh.nx_phys = mesh_res
    cfg.mesh.ny_phys = mesh_res
    params = DerivedParams.from_config(cfg)
    geometry = _create_geometry(cfg, params)
    print(f'\n=== Mesh {cfg.mesh.nx_phys}x{cfg.mesh.ny_phys} ===')

    # Shared mesh and reference
    full_mesh = generate_mesh_full(cfg, params, geometry)
    ref_result = solve_reference(cfg, mesh=full_mesh)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

    # Uncoated
    uncoated_result = solve(cfg.model_copy(update={'is_reference': True}), mesh=cloak_mesh)
    loss_uncoated = compute_cloaking_loss(
        dataclasses.replace(uncoated_result, kept_nodes=kept_nodes), ref_result, geometry)

    # # Continuous C_eff
    problem_c = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geometry)
    sol_c = jax_fem_solver(problem_c, solver_options={'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}})
    loss_continuous = compute_cloaking_loss(
        SolutionResult(u=np.asarray(sol_c[0]), mesh=cloak_mesh,
                       config=cfg.model_copy(update={'is_reference': False}),
                       params=params, kept_nodes=kept_nodes),
        ref_result, geometry)

    def fmt(loss):
        return (f'boundary={loss.dist_boundary:.2f}%  right={loss.dist_right:.2f}%'
                f'  outside={loss.dist_outside:.2f}%'
                f'  (n_bnd={loss.n_boundary}, n_right={loss.n_right}, n_outside={loss.n_outside})')

    print(f'Uncoated void:    {fmt(loss_uncoated)}')
    # print(f'Continuous C_eff: {fmt(loss_continuous)}')
