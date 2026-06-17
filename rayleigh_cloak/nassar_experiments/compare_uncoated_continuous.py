import numpy as np
import jax.numpy as jnp
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import (
    _create_geometry, _plot_nassar_fields, solve_reference, solve, jax_fem_solver,
)
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.problem import build_problem, RayleighCloakProblem
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.nassar import NassarCellMaterial
from rayleigh_cloak.optimize import (
    get_right_boundary_indices, get_circular_boundary_indices,
)

cfg = load_config('configs/nassar.yaml')
cfg = cfg.model_copy(update={'source': cfg.source.model_copy(
    update={'x_src_factor': 0.15, 'sigma_factor': 1.0})})

params = DerivedParams.from_config(cfg)
geometry = _create_geometry(cfg, params)
output_dir = cfg.output_dir

print('Generating mesh...')
full_mesh = generate_mesh_full(cfg, params, geometry)
print(f'  {len(full_mesh.points)} nodes, {len(full_mesh.cells)} elements')

# 1. Reference
print('Solving reference...')
ref_result = solve_reference(cfg, mesh=full_mesh)
u_ref = ref_result.u

# 2. Uncoated void
cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
print(f'  Submesh: {len(cloak_mesh.points)} nodes')

print('Solving uncoated void...')
uncoated_result = solve(cfg.model_copy(update={'is_reference': True}), mesh=cloak_mesh)
u_uncoated = uncoated_result.u

# 3. Nassar cloak (100x100)
print('Solving Nassar 100x100...')
cell_decomp = CellDecomposition(geometry, 100, 100)
nassar_mat = NassarCellMaterial(geometry, params.lam, params.mu, params.rho0, cell_decomp)
RayleighCloakProblem._nassar_cell_material = nassar_mat
problem = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geometry, cell_decomp)
problem.set_params(nassar_mat.get_initial_params())
solver_opts = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
sol = jax_fem_solver(problem, solver_options=solver_opts)
u_cloak = np.asarray(sol[0])
RayleighCloakProblem._nassar_cell_material = None

# 4. Continuous push-forward C_eff
print('Solving continuous C_eff...')
problem2 = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geometry)
sol2 = jax_fem_solver(problem2, solver_options=solver_opts)
u_continuous = np.asarray(sol2[0])

# ── Distortion metrics ──
pts_sub = np.asarray(cloak_mesh.points)

# Circle r = 1.5*rc (primary metric)
bnd_circ = get_circular_boundary_indices(pts_sub, geometry.x_c, geometry.y_c, 1.5 * params.rc)
u_ref_circ = u_ref[kept_nodes[bnd_circ]]
rn_circ = np.sqrt(np.sum(u_ref_circ**2))

# Right boundary (for comparison)
x_right = params.x_off + params.W
bnd_right = get_right_boundary_indices(pts_sub, x_right)
u_ref_right = u_ref[kept_nodes[bnd_right]]
rn_right = np.sqrt(np.sum(u_ref_right**2))

def distortion(u, bnd, u_ref_bnd, rn):
    return 100 * np.sqrt(np.sum((u[bnd] - u_ref_bnd)**2)) / rn

print()
print(f'{"":30s}  {"circle":>8s}  {"right":>8s}')
print('-' * 52)
print(f'{"Uncoated void":30s}  '
      f'{distortion(u_uncoated, bnd_circ, u_ref_circ, rn_circ):8.2f}  '
      f'{distortion(u_uncoated, bnd_right, u_ref_right, rn_right):8.2f}')
print(f'{"Nassar 100x100":30s}  '
      f'{distortion(u_cloak, bnd_circ, u_ref_circ, rn_circ):8.2f}  '
      f'{distortion(u_cloak, bnd_right, u_ref_right, rn_right):8.2f}')
print(f'{"Continuous C_eff":30s}  '
      f'{distortion(u_continuous, bnd_circ, u_ref_circ, rn_circ):8.2f}  '
      f'{distortion(u_continuous, bnd_right, u_ref_right, rn_right):8.2f}')

# ── Displacement field plots ──
print('\nSaving displacement field plots...')

# Reference
_plot_nassar_fields(u_ref, full_mesh, u_ref, full_mesh, params, geometry, output_dir)

# Rename the cloak plots for each configuration
import os, shutil

for label, u_sol in [('uncoated', u_uncoated), ('nassar100', u_cloak), ('continuous', u_continuous)]:
    _plot_nassar_fields(u_ref, full_mesh, u_sol, cloak_mesh, params, geometry, output_dir)
    # Rename cloak outputs to include the label
    for comp in ['re_ux', 're_uy', 'div', 'curl']:
        src = os.path.join(output_dir, f'nassar_cloak_{comp}.png')
        dst = os.path.join(output_dir, f'nassar_{label}_{comp}.png')
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f'  {dst}')

print('\nDone.')
