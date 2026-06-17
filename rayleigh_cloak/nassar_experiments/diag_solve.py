"""Diagnostic: compare reference vs uncoated void solutions."""
import numpy as np
import jax.numpy as jnp
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import _create_geometry, solve_reference, solve
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.optimize import get_all_physical_boundary_indices

cfg = load_config('configs/nassar.yaml')
cfg = cfg.model_copy(update={'source': cfg.source.model_copy(
    update={'x_src_factor': 0.15, 'sigma_factor': 1.0})})

params = DerivedParams.from_config(cfg)
geometry = _create_geometry(cfg, params)
full_mesh = generate_mesh_full(cfg, params, geometry)

# 1. Reference
print("=== Solving reference (full mesh, no void) ===")
ref_result = solve_reference(cfg, mesh=full_mesh)
u_ref = ref_result.u
print(f"  u_ref shape: {u_ref.shape}")
print(f"  u_ref max abs: {np.max(np.abs(u_ref)):.6e}")
print(f"  u_ref L2 norm: {np.sqrt(np.sum(u_ref**2)):.6e}")
print(f"  u_ref Re(ux) range: [{u_ref[:, 0].min():.4e}, {u_ref[:, 0].max():.4e}]")
print(f"  u_ref Re(uy) range: [{u_ref[:, 1].min():.4e}, {u_ref[:, 1].max():.4e}]")
print(f"  u_ref Im(ux) range: [{u_ref[:, 2].min():.4e}, {u_ref[:, 2].max():.4e}]")
print(f"  u_ref Im(uy) range: [{u_ref[:, 3].min():.4e}, {u_ref[:, 3].max():.4e}]")

# 2. Uncoated void
cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
print(f"\n=== Solving uncoated (cloak mesh, void present, homogeneous material) ===")
uncoated_result = solve(cfg.model_copy(update={'is_reference': True}), mesh=cloak_mesh)
u_uncoated = uncoated_result.u
print(f"  u_uncoated shape: {u_uncoated.shape}")
print(f"  u_uncoated max abs: {np.max(np.abs(u_uncoated)):.6e}")
print(f"  u_uncoated L2 norm: {np.sqrt(np.sum(u_uncoated**2)):.6e}")
print(f"  u_uncoated Re(ux) range: [{u_uncoated[:, 0].min():.4e}, {u_uncoated[:, 0].max():.4e}]")
print(f"  u_uncoated Re(uy) range: [{u_uncoated[:, 1].min():.4e}, {u_uncoated[:, 1].max():.4e}]")
print(f"  u_uncoated Im(ux) range: [{u_uncoated[:, 2].min():.4e}, {u_uncoated[:, 2].max():.4e}]")
print(f"  u_uncoated Im(uy) range: [{u_uncoated[:, 3].min():.4e}, {u_uncoated[:, 3].max():.4e}]")

# 3. Distortion with DIFFERENT tolerances
print("\n=== Distortion comparison (different tolerances) ===")
pts_cloak = np.asarray(cloak_mesh.points)
for tol in [0.1, 0.01, 0.005, 0.002, 0.001]:
    bnd = get_all_physical_boundary_indices(
        pts_cloak, params.x_off, params.y_off, params.W, params.H, tol=tol)
    if len(bnd) == 0:
        print(f"  tol={tol:.4f}: no boundary nodes found!")
        continue
    u_ref_bnd = u_ref[kept_nodes[bnd]]
    ref_norm = np.sqrt(np.sum(u_ref_bnd**2))
    diff_norm = np.sqrt(np.sum((u_uncoated[bnd] - u_ref_bnd)**2))
    d = 100 * diff_norm / ref_norm
    print(f"  tol={tol:.4f}: {len(bnd):6d} nodes, "
          f"||u_ref||={ref_norm:.4e}, ||diff||={diff_norm:.4e}, distortion={d:.2f}%")

# 4. Check if solutions are actually different near the void
print("\n=== Solution near void boundary ===")
void_bnd_mask = np.abs(
    np.sqrt((pts_cloak[:, 0] - params.x_c)**2 + (pts_cloak[:, 1] - params.y_c)**2) - params.ri
) < 0.005
void_bnd_idx = np.where(void_bnd_mask)[0]
if len(void_bnd_idx) > 0:
    u_ref_void = u_ref[kept_nodes[void_bnd_idx]]
    u_unc_void = u_uncoated[void_bnd_idx]
    print(f"  {len(void_bnd_idx)} nodes near void boundary (r≈ri)")
    print(f"  u_ref max abs at void: {np.max(np.abs(u_ref_void)):.4e}")
    print(f"  u_uncoated max abs at void: {np.max(np.abs(u_unc_void)):.4e}")
    diff_void = np.sqrt(np.sum((u_unc_void - u_ref_void)**2))
    ref_void_norm = np.sqrt(np.sum(u_ref_void**2))
    print(f"  Local distortion at void: {100*diff_void/ref_void_norm:.2f}%")

# 5. Check on the right physical boundary only (wave propagation direction)
print("\n=== Right boundary only ===")
x_right = params.x_off + params.W
for tol in [0.01, 0.005, 0.002]:
    right_mask = np.abs(pts_cloak[:, 0] - x_right) < tol
    bnd_right = np.where(right_mask)[0]
    if len(bnd_right) == 0:
        continue
    u_ref_r = u_ref[kept_nodes[bnd_right]]
    ref_norm_r = np.sqrt(np.sum(u_ref_r**2))
    diff_norm_r = np.sqrt(np.sum((u_uncoated[bnd_right] - u_ref_r)**2))
    d_r = 100 * diff_norm_r / ref_norm_r
    print(f"  tol={tol:.4f}: {len(bnd_right)} nodes, distortion={d_r:.2f}%")

# 6. Check spatial pattern of difference
print("\n=== Spatial pattern of |u_uncoated - u_ref| ===")
u_ref_mapped = u_ref[kept_nodes]
diff = u_uncoated - u_ref_mapped
diff_mag = np.sqrt(np.sum(diff**2, axis=1))
ref_mag = np.sqrt(np.sum(u_ref_mapped**2, axis=1))
print(f"  Max |diff|: {diff_mag.max():.4e}")
print(f"  Mean |diff|: {diff_mag.mean():.4e}")
print(f"  Max |u_ref|: {ref_mag.max():.4e}")
print(f"  Mean |u_ref|: {ref_mag.mean():.4e}")
# Where is the max difference?
max_diff_idx = np.argmax(diff_mag)
pt_max = pts_cloak[max_diff_idx]
r_max = np.sqrt((pt_max[0] - params.x_c)**2 + (pt_max[1] - params.y_c)**2)
print(f"  Max diff at node {max_diff_idx}: position ({pt_max[0]:.4f}, {pt_max[1]:.4f}), r={r_max:.4f}")
