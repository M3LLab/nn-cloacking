"""Diagnostic: check mesh, boundary selection, and void properties."""
import numpy as np
import jax.numpy as jnp
from rayleigh_cloak.config import load_config, DerivedParams
from rayleigh_cloak.solver import _create_geometry
from rayleigh_cloak.mesh import generate_mesh_full, extract_submesh
from rayleigh_cloak.optimize import get_all_physical_boundary_indices

cfg = load_config('configs/nassar.yaml')
cfg = cfg.model_copy(update={'source': cfg.source.model_copy(
    update={'x_src_factor': 0.15, 'sigma_factor': 1.0})})

params = DerivedParams.from_config(cfg)
geometry = _create_geometry(cfg, params)

print("=== Physical dimensions ===")
print(f"  lambda_star (S-wave): {params.lambda_star:.5f} m")
print(f"  lambda_P:             {params.cp/40000:.5f} m")
print(f"  ri (void radius):     {params.ri:.5f} m  ({params.ri/params.lambda_star:.1f} lambda_S)")
print(f"  rc (cloak radius):    {params.rc:.5f} m")
print(f"  W x H (physical):     {params.W:.4f} x {params.H:.4f} m")
print(f"  W_total x H_total:    {params.W_total:.4f} x {params.H_total:.4f} m")
print(f"  x_off, y_off:         {params.x_off:.5f}, {params.y_off:.5f}")
print(f"  x_c, y_c:             {params.x_c:.5f}, {params.y_c:.5f}")
print(f"  L_pml:                {params.L_pml:.5f} m")
print(f"  x_src:                {params.x_src:.5f} m")
print(f"  sigma_src:            {params.sigma_src:.5f} m")

print("\n=== Mesh generation ===")
full_mesh = generate_mesh_full(cfg, params, geometry)
n_full = len(full_mesh.points)
n_elem_full = len(full_mesh.cells)
print(f"  Full mesh: {n_full} nodes, {n_elem_full} elements")

cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
n_cloak = len(cloak_mesh.points)
n_elem_cloak = len(cloak_mesh.cells)
n_removed = n_elem_full - n_elem_cloak
print(f"  Cloak mesh: {n_cloak} nodes, {n_elem_cloak} elements")
print(f"  Removed {n_removed} defect elements ({100*n_removed/n_elem_full:.1f}%)")
print(f"  Removed {n_full - n_cloak} orphan nodes")

# Check void: find the min/max of removed node positions
removed_mask = np.ones(n_full, dtype=bool)
removed_mask[kept_nodes] = False
if removed_mask.any():
    removed_pts = full_mesh.points[removed_mask]
    r_removed = np.sqrt((removed_pts[:, 0] - params.x_c)**2 + (removed_pts[:, 1] - params.y_c)**2)
    print(f"\n=== Void check ===")
    print(f"  Removed nodes: {removed_mask.sum()}")
    print(f"  Min r from center: {r_removed.min():.5f} m (should be ~0)")
    print(f"  Max r from center: {r_removed.max():.5f} m (should be ~ri={params.ri})")
else:
    print("\n  WARNING: No nodes were removed! Void is empty!")

# Boundary indices with different tolerances
print("\n=== Boundary selection ===")
pts = np.asarray(cloak_mesh.points)
for tol in [0.1, 0.01, 0.001, 0.0001]:
    bnd = get_all_physical_boundary_indices(
        pts, params.x_off, params.y_off, params.W, params.H, tol=tol)
    print(f"  tol={tol:.4f}: {len(bnd):6d} nodes ({100*len(bnd)/n_cloak:.1f}% of mesh)")

# Check element sizes
areas = []
for cell in full_mesh.cells:
    p0, p1, p2 = full_mesh.points[cell]
    area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
    areas.append(area)
areas = np.array(areas)
h_equiv = np.sqrt(areas * 4 / np.sqrt(3))  # equivalent element size
print(f"\n=== Element sizes ===")
print(f"  h_min: {h_equiv.min():.6f} m")
print(f"  h_max: {h_equiv.max():.6f} m")
print(f"  h_mean: {h_equiv.mean():.6f} m")
print(f"  Suggested tol: {h_equiv.max() * 0.5:.6f} m")

# Distance from void to nearest boundary node
print(f"\n=== Distance from void to boundaries ===")
bnd_exact = get_all_physical_boundary_indices(pts, params.x_off, params.y_off, params.W, params.H, tol=h_equiv.max())
r_bnd = np.sqrt((pts[bnd_exact, 0] - params.x_c)**2 + (pts[bnd_exact, 1] - params.y_c)**2)
print(f"  Min distance from void center to boundary nodes: {r_bnd.min():.4f} m")
print(f"  Max distance from void center to boundary nodes: {r_bnd.max():.4f} m")
print(f"  Void radius: {params.ri} m, cloak radius: {params.rc} m")
