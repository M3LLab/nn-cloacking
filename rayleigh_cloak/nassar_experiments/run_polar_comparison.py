"""Run polar vs Cartesian Nassar comparison with all loss metrics.

Solves five configurations on a shared mesh:
  1. Reference (no void, homogeneous)
  2. Uncoated void (void cutout, homogeneous background)
  3. Continuous C_eff (transformational push-forward at every quad point)
  4. Cartesian Nassar (piecewise-constant, square grid)
  5. Polar Nassar (piecewise-constant, N sectors × M layers)

Reports three loss metrics for each:
  - Circle r = 1.5rc  (near-field, good SNR)
  - All physical boundaries
  - All physical nodes outside cloak

Produces displacement field plots (Re(ux), div, curl) for each case.
"""

import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.cells_polar import PolarCellDecomposition
from rayleigh_cloak.config import DerivedParams, load_config
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.nassar import NassarCellMaterial, NassarPolarMaterial
from rayleigh_cloak.optimize import (
    get_all_physical_boundary_indices,
    get_circular_boundary_indices,
    get_outside_cloak_indices,
)
from rayleigh_cloak.problem import RayleighCloakProblem, build_problem
from rayleigh_cloak.solver import _create_geometry, jax_fem_solver, solve_reference


def distortion_pct(u, u_ref, idx):
    """100 * ||u - u_ref|| / ||u_ref|| on selected nodes."""
    diff = u[idx] - u_ref[idx]
    rn = np.sqrt(np.sum(u_ref[idx]**2))
    if rn < 1e-30:
        return float('inf')
    return 100.0 * np.sqrt(np.sum(diff**2)) / rn


def compute_grad_fields(u, mesh):
    """Per-element div and curl of Re(u) on TRI3."""
    pts = np.asarray(mesh.points)
    cells = np.asarray(mesh.cells)
    re_ux, re_uy = u[:, 0], u[:, 1]
    v = pts[cells]
    x1, y1 = v[:, 0, 0], v[:, 0, 1]
    x2, y2 = v[:, 1, 0], v[:, 1, 1]
    x3, y3 = v[:, 2, 0], v[:, 2, 1]
    det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    det = np.where(np.abs(det) < 1e-30, 1e-30, det)
    dNdx = np.column_stack([(y2-y3)/det, (y3-y1)/det, (y1-y2)/det])
    dNdy = np.column_stack([(x3-x2)/det, (x1-x3)/det, (x2-x1)/det])
    ux_e, uy_e = re_ux[cells], re_uy[cells]
    div = np.sum(dNdx * ux_e, axis=1) + np.sum(dNdy * uy_e, axis=1)
    curl = np.sum(dNdx * uy_e, axis=1) - np.sum(dNdy * ux_e, axis=1)
    return div, curl


def plot_field_comparison(results: dict, mesh_ref, mesh_cloak, params, geometry,
                          output_dir: str):
    """Plot Re(ux), div, curl for each case in a grid."""
    x_off, y_off = params.x_off, params.y_off
    W, H = params.W, params.H
    ri, rc = params.ri, params.rc
    x_c, y_c = geometry.x_c, geometry.y_c

    def phys_mask(cells, pts):
        centroids = pts[cells].mean(axis=1)
        return (
            (centroids[:, 0] >= x_off - 1e-8) &
            (centroids[:, 0] <= x_off + W + 1e-8) &
            (centroids[:, 1] >= y_off - 1e-8) &
            (centroids[:, 1] <= y_off + H + 1e-8)
        )

    def add_circles(ax):
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot((x_c - x_off) + ri * np.cos(theta),
                (y_c - y_off) + ri * np.sin(theta),
                '--', color='black', lw=0.8)
        ax.plot((x_c - x_off) + rc * np.cos(theta),
                (y_c - y_off) + rc * np.sin(theta),
                '--', color='black', lw=0.8)

    # Cases to plot
    cases = list(results.keys())
    n_cases = len(cases)

    # 3 rows: Re(ux), div, curl × n_cases columns
    fig, axes = plt.subplots(3, n_cases, figsize=(5 * n_cases, 15))
    if n_cases == 1:
        axes = axes[:, np.newaxis]

    row_labels = ['Re($u_x$)', 'div Re($\\mathbf{u}$)',
                   'curl Re($\\mathbf{u}$)']

    for col, name in enumerate(cases):
        u, mesh = results[name]
        pts = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells)
        pm = phys_mask(cells, pts)
        cells_p = cells[pm]
        tri = mtri.Triangulation(pts[:, 0] - x_off, pts[:, 1] - y_off, cells_p)

        # Re(ux)
        field = u[:, 0]
        vlim = np.percentile(np.abs(field), 97)
        if vlim < 1e-30:
            vlim = 1.0
        ax = axes[0, col]
        tc = ax.tricontourf(tri, field, levels=100, cmap='RdBu_r',
                            vmin=-vlim, vmax=vlim)
        add_circles(ax)
        ax.set_title(f'{name}')
        ax.set_aspect('equal')
        if col == 0:
            ax.set_ylabel(row_labels[0])

        # div, curl
        div_f, curl_f = compute_grad_fields(u, mesh)
        div_p, curl_p = div_f[pm], curl_f[pm]
        tri_cell = mtri.Triangulation(pts[:, 0] - x_off, pts[:, 1] - y_off, cells_p)

        for row, (vals, label) in enumerate(
                [(div_p, row_labels[1]), (curl_p, row_labels[2])], start=1):
            vlim = np.percentile(np.abs(vals), 97)
            if vlim < 1e-30:
                vlim = 1.0
            ax = axes[row, col]
            tp = ax.tripcolor(tri_cell, facecolors=vals, cmap='RdBu_r',
                              vmin=-vlim, vmax=vlim, edgecolors='none')
            add_circles(ax)
            ax.set_aspect('equal')
            if col == 0:
                ax.set_ylabel(label)

    fig.suptitle('Displacement field comparison', fontsize=14, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, 'field_comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {path}')


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/nassar_polar.yaml'
    cfg = load_config(config_path)
    params = DerivedParams.from_config(cfg)
    geometry = _create_geometry(cfg, params)

    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)

    solver_opts = {
        'petsc_solver': {'ksp_type': cfg.solver.ksp_type, 'pc_type': cfg.solver.pc_type}
    }

    # ── Step 1: Shared mesh ──
    print('=== Generating shared mesh ===')
    full_mesh = generate_mesh_full(cfg, params, geometry)
    print(f'  {len(full_mesh.points)} nodes, {len(full_mesh.cells)} elements')

    # ── Step 2: Reference (no void) ──
    print('=== Solving reference (no void) ===')
    ref_result = solve_reference(cfg, mesh=full_mesh)
    u_ref = ref_result.u

    # ── Step 3: Extract submesh ──
    cloak_mesh, kept = extract_submesh(full_mesh, geometry)
    print(f'  Submesh: {len(cloak_mesh.points)} nodes, '
          f'{len(cloak_mesh.cells)} elements')
    pts_sub = np.asarray(cloak_mesh.points)

    # ── Measurement indices ──
    circ_idx = get_circular_boundary_indices(
        pts_sub, geometry.x_c, geometry.y_c, 1.5 * params.rc)
    bnd_idx = get_all_physical_boundary_indices(
        pts_sub, params.x_off, params.y_off, params.W, params.H)
    out_idx = get_outside_cloak_indices(
        pts_sub, geometry, params.x_off, params.y_off, params.W, params.H)

    u_ref_sub = u_ref[kept]  # reference solution mapped to submesh

    print(f'  Measurement nodes: circle={len(circ_idx)}, '
          f'boundary={len(bnd_idx)}, outside={len(out_idx)}')

    # ── Results table ──
    header = f'{"Case":>25s}  {"circle%":>8s}  {"boundary%":>10s}  {"outside%":>9s}'
    sep = '-' * len(header)
    print(f'\n{header}')
    print(sep)

    results_plot = {}  # name -> (u, mesh)

    # ── Case 1: Uncoated void ──
    print('=== Solving uncoated void ===')
    p_unc = build_problem(
        cloak_mesh, cfg.model_copy(update={'is_reference': True}),
        params, geometry)
    s_unc = jax_fem_solver(p_unc, solver_options=solver_opts)
    u_unc = np.asarray(s_unc[0])
    d_unc = (distortion_pct(u_unc, u_ref_sub, circ_idx),
             distortion_pct(u_unc, u_ref_sub, bnd_idx),
             distortion_pct(u_unc, u_ref_sub, out_idx))
    print(f'{"Uncoated void":>25s}  {d_unc[0]:8.2f}  {d_unc[1]:10.2f}  {d_unc[2]:9.2f}')
    results_plot['Uncoated'] = (u_unc, cloak_mesh)

    # ── Case 2: Continuous C_eff ──
    print('=== Solving continuous C_eff ===')
    p_cont = build_problem(
        cloak_mesh, cfg.model_copy(update={'is_reference': False}),
        params, geometry)
    s_cont = jax_fem_solver(p_cont, solver_options=solver_opts)
    u_cont = np.asarray(s_cont[0])
    d_cont = (distortion_pct(u_cont, u_ref_sub, circ_idx),
              distortion_pct(u_cont, u_ref_sub, bnd_idx),
              distortion_pct(u_cont, u_ref_sub, out_idx))
    print(f'{"Continuous C_eff":>25s}  {d_cont[0]:8.2f}  {d_cont[1]:10.2f}  {d_cont[2]:9.2f}')
    results_plot['Continuous'] = (u_cont, cloak_mesh)

    # ── Case 3: Cartesian Nassar ──
    print('=== Solving Cartesian Nassar ===')
    n_cart = cfg.nassar.cell_n_x
    cd = CellDecomposition(geometry, n_cart, n_cart)
    mat_cart = NassarCellMaterial(geometry, params.lam, params.mu, params.rho0, cd)
    RayleighCloakProblem._nassar_cell_material = mat_cart
    p_cart = build_problem(
        cloak_mesh, cfg.model_copy(update={'is_reference': False}),
        params, geometry, cd)
    p_cart.set_params(mat_cart.get_initial_params())
    s_cart = jax_fem_solver(p_cart, solver_options=solver_opts)
    u_cart = np.asarray(s_cart[0])
    RayleighCloakProblem._nassar_cell_material = None
    d_cart = (distortion_pct(u_cart, u_ref_sub, circ_idx),
              distortion_pct(u_cart, u_ref_sub, bnd_idx),
              distortion_pct(u_cart, u_ref_sub, out_idx))
    print(f'{"Cartesian " + str(n_cart) + "x" + str(n_cart):>25s}  '
          f'{d_cart[0]:8.2f}  {d_cart[1]:10.2f}  {d_cart[2]:9.2f}')
    results_plot[f'Cart {n_cart}x{n_cart}'] = (u_cart, cloak_mesh)

    # ── Case 4: Polar Nassar ──
    print('=== Solving polar Nassar ===')
    N_pol = cfg.nassar.lattice_N
    M_pol = cfg.nassar.lattice_M
    pd = PolarCellDecomposition(
        ri=params.ri, rc=params.rc,
        x_c=geometry.x_c, y_c=geometry.y_c,
        N=N_pol, M=M_pol,
    )
    mat_polar = NassarPolarMaterial(
        geometry, params.lam, params.mu, params.rho0, pd)
    RayleighCloakProblem._nassar_cell_material = mat_polar
    p_polar = build_problem(
        cloak_mesh, cfg.model_copy(update={'is_reference': False}),
        params, geometry, pd)
    p_polar.set_params(mat_polar.get_initial_params())
    s_polar = jax_fem_solver(p_polar, solver_options=solver_opts)
    u_polar = np.asarray(s_polar[0])
    RayleighCloakProblem._nassar_cell_material = None
    d_polar = (distortion_pct(u_polar, u_ref_sub, circ_idx),
               distortion_pct(u_polar, u_ref_sub, bnd_idx),
               distortion_pct(u_polar, u_ref_sub, out_idx))
    print(f'{"Polar " + str(N_pol) + "x" + str(M_pol):>25s}  '
          f'{d_polar[0]:8.2f}  {d_polar[1]:10.2f}  {d_polar[2]:9.2f}')
    results_plot[f'Polar {N_pol}x{M_pol}'] = (u_polar, cloak_mesh)

    # ── Summary ──
    print(f'\n{header}')
    print(sep)
    print(f'{"Uncoated void":>25s}  {d_unc[0]:8.2f}  {d_unc[1]:10.2f}  {d_unc[2]:9.2f}')
    print(f'{"Continuous C_eff":>25s}  {d_cont[0]:8.2f}  {d_cont[1]:10.2f}  {d_cont[2]:9.2f}')
    print(f'{"Cartesian " + str(n_cart) + "x" + str(n_cart):>25s}  '
          f'{d_cart[0]:8.2f}  {d_cart[1]:10.2f}  {d_cart[2]:9.2f}')
    print(f'{"Polar " + str(N_pol) + "x" + str(M_pol):>25s}  '
          f'{d_polar[0]:8.2f}  {d_polar[1]:10.2f}  {d_polar[2]:9.2f}')

    # ── Plots ──
    print('\n=== Plotting field comparison ===')
    # Add reference for plot
    results_plot_full = {'Reference': (u_ref, full_mesh)}
    results_plot_full.update(results_plot)
    plot_field_comparison(results_plot_full, full_mesh, cloak_mesh,
                          params, geometry, output_dir)

    # Save results
    np.savez(
        os.path.join(output_dir, 'comparison_results.npz'),
        d_uncoated=d_unc, d_continuous=d_cont,
        d_cartesian=d_cart, d_polar=d_polar,
        n_cart=n_cart, N_pol=N_pol, M_pol=M_pol,
    )
    print(f'  Saved {output_dir}/comparison_results.npz')
    print('\nDone.')


if __name__ == '__main__':
    main()
