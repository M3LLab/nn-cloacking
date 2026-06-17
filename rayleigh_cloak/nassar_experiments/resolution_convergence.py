import numpy as np
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import _create_geometry, solve_reference, solve, jax_fem_solver
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.problem import build_problem, RayleighCloakProblem
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.nassar import NassarCellMaterial
from rayleigh_cloak.optimize import get_right_boundary_indices


def circular_boundary_indices(pts, x_c, y_c, r_measure, tol_frac=0.05):
    """Find mesh nodes near a circle of radius r_measure around (x_c, y_c)."""
    dx = pts[:, 0] - x_c
    dy = pts[:, 1] - y_c
    r = np.sqrt(dx**2 + dy**2)
    tol = tol_frac * r_measure
    return np.where(np.abs(r - r_measure) < tol)[0]


def distortion_on(u, u_ref, idx):
    """Relative L2 distortion on selected nodes."""
    diff = u[idx] - u_ref[idx]
    rn = np.sqrt(np.sum(u_ref[idx]**2))
    if rn < 1e-30:
        return float('inf')
    return 100 * np.sqrt(np.sum(diff**2)) / rn


cfg = load_config('configs/nassar.yaml')
params = DerivedParams.from_config(cfg)
geo = _create_geometry(cfg, params)

print('Generating fine mesh...')
full_mesh = generate_mesh_full(cfg, params, geo)
print(f'  {len(full_mesh.points)} nodes, {len(full_mesh.cells)} elements')

ref = solve_reference(cfg, mesh=full_mesh)
cloak_mesh, kept = extract_submesh(full_mesh, geo)
print(f'  Submesh: {len(cloak_mesh.points)} nodes')

# --- Measurement indices ---
pts_sub = np.asarray(cloak_mesh.points)

# Right physical boundary (original metric — low |u_ref|, inflated distortion)
x_right = params.x_off + params.W
bnd_right = get_right_boundary_indices(pts_sub, x_right)
u_ref_right = ref.u[kept[bnd_right]]
rn_right = np.sqrt(np.sum(u_ref_right**2))

# Circular measurement at r = 1.5*rc (better metric — significant |u_ref|)
r_meas = 1.5 * params.rc
bnd_circ = circular_boundary_indices(pts_sub, geo.x_c, geo.y_c, r_meas)
u_ref_circ = ref.u[kept[bnd_circ]]
rn_circ = np.sqrt(np.sum(u_ref_circ**2))

print(f'  Right boundary: {len(bnd_right)} nodes, |u_ref| = {rn_right:.4e}')
print(f'  Circle r=1.5rc: {len(bnd_circ)} nodes, |u_ref| = {rn_circ:.4e}')
print(f'  (|u_ref| ratio: {rn_circ/rn_right:.0f}x — right boundary has much lower amplitude)')
print()

# Uncoated
unc = solve(cfg.model_copy(update={'is_reference': True}), mesh=cloak_mesh)
d_unc_right = distortion_on(unc.u, ref.u[kept], bnd_right)
d_unc_circ = distortion_on(unc.u, ref.u[kept], bnd_circ)
print(f'Uncoated void:       right={d_unc_right:7.2f}%   circle={d_unc_circ:6.2f}%')

# Continuous
solver_opts = {'petsc_solver': {'ksp_type': 'preonly', 'pc_type': 'lu'}}
p_c = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geo)
s_c = jax_fem_solver(p_c, solver_options=solver_opts)
u_cont = np.asarray(s_c[0])
d_cont_right = distortion_on(u_cont, ref.u[kept], bnd_right)
d_cont_circ = distortion_on(u_cont, ref.u[kept], bnd_circ)
print(f'Continuous c_eff:    right={d_cont_right:7.2f}%   circle={d_cont_circ:6.2f}%')
print()

# Nassar convergence
print(f'{"Config":>20s}  {"cells":>6s}  {"right%":>8s}  {"circle%":>8s}')
print('-' * 52)
for n in [5, 10, 20, 40, 60, 80, 100, 150, 200]:
    cd = CellDecomposition(geo, n, n)
    mat = NassarCellMaterial(geo, params.lam, params.mu, params.rho0, cd)
    RayleighCloakProblem._nassar_cell_material = mat
    p = build_problem(cloak_mesh, cfg.model_copy(update={'is_reference': False}), params, geo, cd)
    p.set_params(mat.get_initial_params())
    sol = jax_fem_solver(p, solver_options=solver_opts)
    u_nas = np.asarray(sol[0])
    d_right = distortion_on(u_nas, ref.u[kept], bnd_right)
    d_circ = distortion_on(u_nas, ref.u[kept], bnd_circ)
    print(f'  Nassar {n:3d}x{n:3d}  {cd.n_cloak_cells:6d}  {d_right:8.2f}  {d_circ:8.2f}')
    RayleighCloakProblem._nassar_cell_material = None
