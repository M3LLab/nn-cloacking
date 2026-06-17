import numpy as np
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import _create_geometry, solve_reference, solve, jax_fem_solver
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.problem import build_problem, RayleighCloakProblem
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.nassar import NassarCellMaterial
from rayleigh_cloak.optimize import get_all_physical_boundary_indices

cfg = load_config('configs/nassar.yaml')
params = DerivedParams.from_config(cfg)
geometry = _create_geometry(cfg, params)

# Shared mesh and reference
full_mesh = generate_mesh_full(cfg, params, geometry)
ref_result = solve_reference(cfg, mesh=full_mesh)
u_ref = ref_result.u
cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)

bnd = get_all_physical_boundary_indices(
    np.asarray(cloak_mesh.points), params.x_off, params.y_off, params.W, params.H)
u_ref_bnd = u_ref[kept_nodes[bnd]]
ref_norm = np.sqrt(np.sum(u_ref_bnd**2))

# Uncoated
uncoated_result = solve(cfg.model_copy(update={'is_reference': True}), mesh=cloak_mesh)
d_uncoated = 100 * np.sqrt(np.sum((uncoated_result.u[bnd] - u_ref_bnd)**2)) / ref_norm

# Continuous C_eff
problem_c = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geometry)
sol_c = jax_fem_solver(problem_c, solver_options={'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}})
d_continuous = 100 * np.sqrt(np.sum((np.asarray(sol_c[0])[bnd] - u_ref_bnd)**2)) / ref_norm

print(f'Uncoated void:     {d_uncoated:.2f}%')
print(f'Continuous C_eff:  {d_continuous:.2f}%')
print()

# Nassar convergence
for n in [20, 50, 100, 200, 300]:
    cd = CellDecomposition(geometry, n, n)
    mat = NassarCellMaterial(geometry, params.lam, params.mu, params.rho0, cd)
    RayleighCloakProblem._nassar_cell_material = mat
    p = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geometry, cd)
    p.set_params(mat.get_initial_params())
    sol = jax_fem_solver(p, solver_options={'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}})
    d = 100 * np.sqrt(np.sum((np.asarray(sol[0])[bnd] - u_ref_bnd)**2)) / ref_norm
    print(f'Nassar {n:2d}x{n:2d} ({cd.n_cloak_cells:4d} cloak cells): {d:.2f}%')
    RayleighCloakProblem._nassar_cell_material = None
