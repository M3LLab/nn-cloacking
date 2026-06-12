"""Plotting utilities for simulation results.

Delegates to the existing ``plot_results`` module when available, providing
a thin wrapper that accepts a ``SolutionResult``.

Plot displacement fields from saved simulation results.

Usage:  python plot_results.py [path/to/results.npz]
        Defaults to output/results.npz if no argument given.
"""
from __future__ import annotations
import os
import sys
from datetime import datetime
from typing import Literal

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from rayleigh_cloak.io import save_vtk
from rayleigh_cloak.solver import SolutionResult


NormType = Literal['asym_sigmoid', 'sigmoid', 'linear']


class SigmoidNorm(mcolors.Normalize):
    """Normalize using a sigmoid, stretching extremes and compressing the middle.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness : float
        Controls how sharp the transition is.  Higher = more contrast at
        the extremes, flatter middle.  Typical range: 5-30.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness=10.0, vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness = steepness
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # normalise data to [0, 1] linearly first
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # sigmoid centred on mid_norm
        s = 1.0 / (1.0 + np.exp(-self.steepness * (x - mid_norm)))
        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


class AsymSigmoidNorm(mcolors.Normalize):
    """Normalize using an asymmetric sigmoid with different steepness on each side.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness_left : float
        Steepness for values below *mid*.  Higher = sharper transition.
    steepness_right : float
        Steepness for values above *mid*.  Higher = sharper transition.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness_left=10.0, steepness_right=10.0,
                 vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness_left = steepness_left
        self.steepness_right = steepness_right
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # pick steepness per element depending on which side of mid_norm
        k = np.where(x <= mid_norm, self.steepness_left, self.steepness_right)
        s = 1.0 / (1.0 + np.exp(-k * (x - mid_norm)))

        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness_left * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness_right * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


def _build_norm(kind: NormType, vmin: float, vmax: float, mid: float,
                symmetric: bool = False) -> mcolors.Normalize:
    """Build a colormap normalizer.

    Parameters
    ----------
    kind : {'asym_sigmoid', 'sigmoid', 'linear'}
    vmin, vmax : data limits (after percentile clipping).
    mid : inflection point for sigmoid norms.
    symmetric : if True, use equal steepness on both sides (suited for
        signed data centred at zero).
    """
    if kind == 'linear':
        return mcolors.Normalize(vmin=vmin, vmax=vmax)
    if kind == 'sigmoid':
        steepness = 50 if symmetric else 10
        return SigmoidNorm(mid=mid, steepness=steepness, vmin=vmin, vmax=vmax)
    # default: 'asym_sigmoid'
    if symmetric:
        return AsymSigmoidNorm(mid=mid, steepness_left=50, steepness_right=50,
                               vmin=vmin, vmax=vmax)
    return AsymSigmoidNorm(mid=mid, steepness_left=2, steepness_right=30,
                           vmin=vmin, vmax=vmax)



def plot_npz_results(data_path, percentile=95,
                     norm_type: NormType = 'asym_sigmoid'):
    d = np.load(data_path)

    u       = d['u']           # (num_nodes, 4)
    pts_x   = d['pts_x']
    pts_y   = d['pts_y']
    x_src   = float(d['x_src'])
    y_top   = float(d['y_top'])
    x_off   = float(d['x_off'])
    y_off   = float(d['y_off'])
    W       = float(d['W'])
    H       = float(d['H'])
    x_src_phys = float(d['x_src_phys'])
    f_star  = float(d['f_star'])

    # ── Derived quantities ───────────────────────────────────────────────
    ux_R, uy_R, ux_I, uy_I = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
    mag = np.sqrt(ux_R**2 + uy_R**2 + ux_I**2 + uy_I**2)

    # ── Cloak geometry (must match boundaries.py) ──────────────────────
    a_cloak = 0.0774 * H                       # inner triangle depth
    b_cloak = 3 * a_cloak                      # outer triangle depth
    c_cloak = 0.309 * H / 2.0                  # half-width at surface
    x_c     = x_off + W / 2.0                  # cloak centre (extended coords)

    os.makedirs("output", exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Full-domain plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    vmin_mag = np.percentile(mag, 100 - percentile)
    vmax_mag = np.percentile(mag, percentile)
    norm = _build_norm(norm_type, vmin_mag, vmax_mag, mid=0.25*vmax_mag)
    tc = ax.tricontourf(pts_x, pts_y, mag, levels=100, cmap='RdBu_r', norm=norm)
    ax.plot(x_src, y_top, 'r*', markersize=16, label='source')

    ax.axvline(x_off,       color='cyan', ls='--', lw=0.7, label='PML interface')
    ax.axvline(x_off + W,   color='cyan', ls='--', lw=0.7)
    ax.axhline(y_off,       color='cyan', ls='--', lw=0.7)

    # Cloak outer boundary: (x_c-c, y_top) → (x_c, y_top-b) → (x_c+c, y_top)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - b_cloak, y_top],
            ls='--', color='yellow', lw=1.2)
    # Cloak inner boundary: (x_c-c, y_top) → (x_c, y_top-a) → (x_c+c, y_top)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - a_cloak, y_top],
            ls='--', color='yellow', lw=1.2)

    fig.colorbar(tc, ax=ax, shrink=0.8, label='|u|')
    ax.set_title(f"Symmetrized Triangular Cloak with Absorbing Layers  (f*={f_star})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.legend(loc='upper right', fontsize=8)
    ax.set_aspect('equal')

    fname_full = f"output/cloak_abslay_full_{stamp}.png"
    fig.savefig(fname_full, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Full-domain plot saved → {fname_full}")

    # ── Physical-domain-only plot ────────────────────────────────────────
    phys_mask = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8) &
                (pts_y >= y_off - 1e-8))

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    mag_phys = mag[phys_mask]
    # norm2 = SigmoidNorm(mid=0.2*mag_phys.max(), steepness=14,
    #                      vmin=mag_phys.min(), vmax=1*mag_phys.max())
    vmin_phys = np.percentile(mag_phys, 100 - percentile)
    vmax_phys = np.percentile(mag_phys, percentile)
    norm2 = _build_norm(norm_type, vmin_phys, vmax_phys, mid=0.25*vmax_phys)

    tc2 = ax2.tricontourf(pts_x[phys_mask] - x_off,
                        pts_y[phys_mask] - y_off,
                        mag_phys,
                        levels=100, cmap='RdBu_r', norm=norm2)
    ax2.plot(x_src_phys, H, 'r*', markersize=16)

    # Cloak boundaries in physical coords (centre at W/2, surface at y=H)
    x_c_phys = W / 2.0
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
            [H, H - b_cloak, H],
            ls='--', color='yellow', lw=1.2)
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
            [H, H - a_cloak, H],
            ls='--', color='yellow', lw=1.2)
    # ax2.legend(loc='upper right', fontsize=8)

    fig2.colorbar(tc2, ax=ax2, shrink=0.8, label='|u|')
    ax2.set_title(f"Physical domain only  (f*={f_star})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect('equal')

    fname_crop = f"output/cloak_abslay_phys_{stamp}.png"
    # fname_crop = f"output/cloak_abslay_phys.png"
    fig2.savefig(fname_crop, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Physical-domain plot saved → {fname_crop}")


def plot_vtk_results(vtk_path, percentile=95,
                     norm_type: NormType = 'asym_sigmoid'):
    """Plot displacement field from a VTK file using actual mesh connectivity."""
    import vtk as _vtk
    from vtk.util.numpy_support import vtk_to_numpy
    from matplotlib.collections import PolyCollection

    reader = _vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_path)
    reader.ReadAllScalarsOn()
    reader.ReadAllFieldsOn()
    reader.Update()
    grid = reader.GetOutput()

    # Extract points (2D)
    pts = vtk_to_numpy(grid.GetPoints().GetData())[:, :2]

    # Extract quad cells → list of (4, 2) vertex arrays
    cells = grid.GetCells()
    cell_array = vtk_to_numpy(cells.GetConnectivityArray())
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    quads = []
    for i in range(len(offsets) - 1):
        ids = cell_array[offsets[i]:offsets[i + 1]]
        quads.append(pts[ids])

    # Point data
    mag = vtk_to_numpy(grid.GetPointData().GetArray("mag_u"))

    # Per-cell colour = mean of vertex values
    cell_vals = np.array([mag[cell_array[offsets[i]:offsets[i+1]]].mean()
                          for i in range(len(offsets) - 1)])

    # Read metadata from field data
    fd = grid.GetFieldData()
    def _fval(name):
        return vtk_to_numpy(fd.GetArray(name))[0]
    x_src, y_top = _fval("x_src"), _fval("y_top")
    x_off, y_off = _fval("x_off"), _fval("y_off")
    W, H = _fval("W"), _fval("H")
    x_src_phys = _fval("x_src_phys")
    f_star = _fval("f_star")

    a_cloak = 0.0774 * H
    b_cloak = 3 * a_cloak
    c_cloak = 0.309 * H / 2.0
    x_c = x_off + W / 2.0

    os.makedirs("output", exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # ── Full-domain plot ─────────────────────────────────────────────────
    vmin_mag = np.percentile(mag, 100 - percentile)
    vmax_mag = np.percentile(mag, percentile)
    norm = _build_norm(norm_type, vmin_mag, vmax_mag, mid=0.25 * vmax_mag)

    fig, ax = plt.subplots(figsize=(14, 5))
    pc = PolyCollection(quads, array=cell_vals, cmap='RdBu_r', norm=norm,
                        edgecolors='face', linewidths=0.1, rasterized=True)
    ax.add_collection(pc)
    ax.autoscale_view()

    ax.plot(x_src, y_top, 'r*', markersize=16, label='source')
    ax.axvline(x_off, color='cyan', ls='--', lw=0.7, label='PML interface')
    ax.axvline(x_off + W, color='cyan', ls='--', lw=0.7)
    ax.axhline(y_off, color='cyan', ls='--', lw=0.7)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - b_cloak, y_top], ls='--', color='yellow', lw=1.2)
    ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
            [y_top, y_top - a_cloak, y_top], ls='--', color='yellow', lw=1.2)

    fig.colorbar(pc, ax=ax, shrink=0.8, label='|u|')
    ax.set_title(f"Triangular Cloak – VTK mesh  (f*={f_star})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_aspect('equal')

    fname_full = f"output/cloak_vtk_full.png" #_{stamp}
    fig.savefig(fname_full, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"VTK full-domain plot saved → {fname_full}")

    # ── Full-domain plot of Re(u_y) ──────────────────────────────────────
    u_full = vtk_to_numpy(grid.GetPointData().GetArray("u"))  # (num_nodes, 4)
    re_uy = u_full[:, 1]  # Re(u_y)

    cell_vals_re_uy_full = np.array([re_uy[cell_array[offsets[i]:offsets[i+1]]].mean()
                                     for i in range(len(offsets) - 1)])

    vlim_full = np.percentile(np.abs(re_uy), percentile)
    norm_re_full = _build_norm(norm_type, -vlim_full, vlim_full, mid=0, symmetric=True)

    fig_rf, ax_rf = plt.subplots(figsize=(14, 5))
    pc_rf = PolyCollection(quads, array=cell_vals_re_uy_full, cmap='RdBu_r',
                           norm=norm_re_full, edgecolors='face', linewidths=0.1,
                           rasterized=True)
    ax_rf.add_collection(pc_rf)
    ax_rf.autoscale_view()

    ax_rf.plot(x_src, y_top, 'r*', markersize=16, label='source')
    ax_rf.axvline(x_off, color='cyan', ls='--', lw=0.7, label='PML interface')
    ax_rf.axvline(x_off + W, color='cyan', ls='--', lw=0.7)
    ax_rf.axhline(y_off, color='cyan', ls='--', lw=0.7)
    ax_rf.plot([x_c - c_cloak, x_c, x_c + c_cloak],
               [y_top, y_top - b_cloak, y_top], ls='--', color='yellow', lw=1.2)
    ax_rf.plot([x_c - c_cloak, x_c, x_c + c_cloak],
               [y_top, y_top - a_cloak, y_top], ls='--', color='yellow', lw=1.2)

    fig_rf.colorbar(pc_rf, ax=ax_rf, shrink=0.8, label='Re(u_y)')
    ax_rf.set_title(f"Re(u_y) – Full domain  (f*={f_star})")
    ax_rf.set_xlabel("x"); ax_rf.set_ylabel("y")
    ax_rf.set_aspect('equal')

    fname_re_full = f"output/cloak_vtk_re_uy_full.png"
    fig_rf.savefig(fname_re_full, dpi=200, bbox_inches='tight')
    plt.close(fig_rf)
    print(f"VTK Re(u_y) full-domain plot saved → {fname_re_full}")

    # ── Physical-domain-only plot ────────────────────────────────────────
    # Keep cells whose centroid is inside physical domain
    centroids = np.array([q.mean(axis=0) for q in quads])
    phys_mask = ((centroids[:, 0] >= x_off - 1e-8) &
                 (centroids[:, 0] <= x_off + W + 1e-8) &
                 (centroids[:, 1] >= y_off - 1e-8))

    phys_quads = [quads[i] - np.array([x_off, y_off]) for i in range(len(quads)) if phys_mask[i]]
    phys_vals = cell_vals[phys_mask]

    vmin_pv = np.percentile(phys_vals, 100 - percentile)
    vmax_pv = np.percentile(phys_vals, percentile)
    norm2 = _build_norm(norm_type, vmin_pv, vmax_pv, mid=0.25 * vmax_pv)

    fig2, ax2 = plt.subplots(figsize=(13, 4))
    pc2 = PolyCollection(phys_quads, array=phys_vals, cmap='RdBu_r', norm=norm2,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax2.add_collection(pc2)
    ax2.autoscale_view()

    ax2.plot(x_src_phys, H, 'r*', markersize=16)
    x_c_phys = W / 2.0
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig2.colorbar(pc2, ax=ax2, shrink=0.8, label='|u|')
    ax2.set_title(f"Physical domain only – VTK mesh  (f*={f_star})")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_aspect('equal')

    fname_crop = f"output/cloak_vtk_phys.png" #_{stamp}
    fig2.savefig(fname_crop, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"VTK physical-domain plot saved → {fname_crop}")

    # ── Real part of u_y (physical domain) ──────────────────────────────
    cell_vals_re = cell_vals_re_uy_full
    phys_vals_re = cell_vals_re[phys_mask]

    vlim = np.percentile(np.abs(phys_vals_re), percentile)
    norm3 = _build_norm(norm_type, -vlim, vlim, mid=0, symmetric=True)

    fig3, ax3 = plt.subplots(figsize=(13, 4))
    pc3 = PolyCollection(phys_quads, array=phys_vals_re, cmap='RdBu_r', norm=norm3,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax3.add_collection(pc3)
    ax3.autoscale_view()

    ax3.plot(x_src_phys, H, 'r*', markersize=16)
    ax3.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax3.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig3.colorbar(pc3, ax=ax3, shrink=0.8, label='Re(u_y)')
    ax3.set_title(f"Re(u_y) – Physical domain  (f*={f_star})")
    ax3.set_xlabel("x"); ax3.set_ylabel("y")
    ax3.set_aspect('equal')

    fname_re = f"output/cloak_vtk_re_uy.png"  #_{stamp}
    fig3.savefig(fname_re, dpi=200, bbox_inches='tight')
    plt.close(fig3)
    print(f"VTK Re(u_y) plot saved → {fname_re}")

    # ── Total real displacement magnitude (physical domain) ─────────────
    re_ux = u_full[:, 0]  # Re(u_x)
    re_mag = np.sqrt(re_ux**2 + re_uy**2)

    cell_vals_re_mag = np.array([re_mag[cell_array[offsets[i]:offsets[i+1]]].mean()
                                 for i in range(len(offsets) - 1)])
    phys_vals_re_mag = cell_vals_re_mag[phys_mask]

    vmin_rm = np.percentile(phys_vals_re_mag, 100 - percentile)
    vmax_rm = np.percentile(phys_vals_re_mag, percentile)
    norm4 = _build_norm(norm_type, vmin_rm, vmax_rm, mid=0.25 * vmax_rm)

    fig4, ax4 = plt.subplots(figsize=(13, 4))
    pc4 = PolyCollection(phys_quads, array=phys_vals_re_mag, cmap='RdBu_r', norm=norm4,
                         edgecolors='face', linewidths=0.1, rasterized=True)
    ax4.add_collection(pc4)
    ax4.autoscale_view()

    ax4.plot(x_src_phys, H, 'r*', markersize=16)
    ax4.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - b_cloak, H], ls='--', color='yellow', lw=1.2)
    ax4.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
             [H, H - a_cloak, H], ls='--', color='yellow', lw=1.2)

    fig4.colorbar(pc4, ax=ax4, shrink=0.8, label='|Re(u)|')
    ax4.set_title(f"|Re(u)| – Physical domain  (f*={f_star})")
    ax4.set_xlabel("x"); ax4.set_ylabel("y")
    ax4.set_aspect('equal')

    fname_re_mag = f"output/cloak_vtk_re_mag.png" #_{stamp}
    fig4.savefig(fname_re_mag, dpi=200, bbox_inches='tight')
    plt.close(fig4)
    print(f"VTK |Re(u)| plot saved → {fname_re_mag}")


def _plot_field_on_phys_domain(
    values: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    params,
    save_path: str,
    title: str,
    cbar_label: str,
    percentile: float = 95,
    norm_type: NormType = 'linear',
    cells: np.ndarray | None = None,
    symmetric: bool = False,
) -> None:
    """Render a per-node scalar field on the physical domain.

    Shared by :func:`plot_displacement_field` (magnitude) and the
    Re(u_y) panel produced by :func:`plot_reference_field`.  Set
    ``symmetric=True`` for signed fields like Re(u_y) — the half-range
    is taken from the percentile of ``|values|`` so the colourbar is
    centred at zero.
    """
    import matplotlib.tri as mtri

    x_off, y_off = params.x_off, params.y_off
    W, H = params.W, params.H

    phys = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8) &
            (pts_y >= y_off - 1e-8))

    px = pts_x[phys] - x_off
    py = pts_y[phys] - y_off
    pv = values[phys]

    if symmetric:
        vlim = np.percentile(np.abs(pv), percentile)
        norm = _build_norm(norm_type, -vlim, vlim, mid=0.0, symmetric=True)
    else:
        vmin_v = np.percentile(pv, 100 - percentile)
        vmax_v = np.percentile(pv, percentile)
        norm = _build_norm(norm_type, vmin_v, vmax_v, mid=0.25 * vmax_v)

    if cells is not None:
        # Re-index the connectivity into the physical-domain submask so the
        # cut-out cloak void is preserved as a hole.
        cells = np.asarray(cells)
        # For quadratic (TRI6) meshes, matplotlib's Triangulation needs the
        # 3-node corner connectivity; the midside nodes are not triangulation
        # vertices (the |u| contour is a P1 plot of the corner values).
        if cells.ndim == 2 and cells.shape[1] > 3:
            cells = cells[:, :3]
        new_index = -np.ones(pts_x.shape[0], dtype=np.int64)
        new_index[np.where(phys)[0]] = np.arange(int(phys.sum()))
        cell_keep = phys[cells].all(axis=1)
        triang = mtri.Triangulation(px, py, new_index[cells[cell_keep]])
        tri_arg = (triang,)
    else:
        tri_arg = (px, py)

    fig, ax = plt.subplots(figsize=(13, 4))
    tc = ax.tricontourf(*tri_arg, pv, levels=100, cmap='RdBu_r', norm=norm)

    ax.plot(params.x_src - x_off, H, 'r*', markersize=12)

    a, b, c_hw = params.a, params.b, params.c
    xc = W / 2.0
    ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - b, H],
            ls='--', color='yellow', lw=1.2)
    ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - a, H],
            ls='--', color='yellow', lw=1.2)

    fig.colorbar(tc, ax=ax, shrink=0.8, label=cbar_label)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_displacement_field(
    u: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    params,
    save_path: str,
    title: str = "|u|",
    percentile: float = 95,
    norm_type: NormType = 'linear',
    cells: np.ndarray | None = None,
) -> None:
    """Plot the phase-invariant complex magnitude
    ``|U| = sqrt(|U_x|^2 + |U_y|^2)`` in the physical domain.

    Parameters
    ----------
    u : (num_nodes, 4)  — DOFs [Re(ux), Re(uy), Im(ux), Im(uy)]
    pts_x, pts_y : node coordinates
    params : DerivedParams (for domain extents and cloak geometry)
    save_path : output PNG path
    cells : (n_cells, 3) int, optional
        Triangle connectivity of the (sub-)mesh.  When provided, the
        plot uses this triangulation instead of letting matplotlib
        Delaunay-triangulate the points — required to preserve the
        cut-out cloak void as a hole rather than filling it in.
    """
    full_mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2
                       + u[:, 2] ** 2 + u[:, 3] ** 2)
    _plot_field_on_phys_domain(
        full_mag, pts_x, pts_y, params,
        save_path=save_path, title=title, cbar_label='|u|',
        percentile=percentile, norm_type=norm_type, cells=cells,
        symmetric=False,
    )


def plot_step_field(
    u: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    params,
    save_path: str,
    title: str = "|u|",
    percentile: float = 95,
    norm_type: NormType = 'linear',
    cells: np.ndarray | None = None,
) -> None:
    """Save per-step magnitude and in-phase real-displacement panels.

    Writes the phase-invariant magnitude ``|U|`` to ``save_path`` and
    the in-phase ``sqrt(Re(u_x)^2 + Re(u_y)^2)`` snapshot to
    ``<stem>_re_mag<ext>``.  The real-magnitude panel exposes the
    propagating wave that the envelope hides.
    """
    plot_displacement_field(
        u, pts_x, pts_y, params,
        save_path=save_path, title=title,
        percentile=percentile, norm_type=norm_type, cells=cells,
    )
    re_mag = np.sqrt(u[:, 0] ** 2 + u[:, 1] ** 2)
    root, ext = os.path.splitext(save_path)
    _plot_field_on_phys_domain(
        re_mag, pts_x, pts_y, params,
        save_path=f"{root}_re_mag{ext}",
        title=title.replace("|u|", "|Re(u)|"),
        cbar_label="|Re(u)|",
        percentile=percentile, norm_type=norm_type, cells=cells,
        symmetric=False,
    )


def plot_field_panels(
    u: np.ndarray,
    pts_x: np.ndarray,
    pts_y: np.ndarray,
    params,
    save_path: str,
    title: str = "|u|",
    percentile: float = 95,
    norm_type: NormType = 'linear',
    cells: np.ndarray | None = None,
) -> None:
    """Save four void-aware diagnostic panels for one displacement solution.

    Files written next to ``save_path`` (suffix on the filename stem):

      * ``''``         — |U| phase-invariant magnitude (envelope only)
      * ``'_re_ux'``   — Re(u_x), wavelength-visible component
      * ``'_re_uy'``   — Re(u_y), wavelength-visible component
      * ``'_re_mag'``  — ``sqrt(Re(u_x)^2 + Re(u_y)^2)``, the
        in-phase real-displacement snapshot

    Pass ``cells`` (the mesh connectivity) so a cut-out cloak void is
    preserved as a hole instead of being filled by Delaunay triangulation.
    Shared by :func:`plot_reference_field` and the forward-solve CLI (``run.py``).
    """
    plot_displacement_field(
        u, pts_x, pts_y, params,
        save_path=save_path, title=title,
        percentile=percentile, norm_type=norm_type, cells=cells,
    )

    root, ext = os.path.splitext(save_path)
    re_ux = u[:, 0]
    re_uy = u[:, 1]
    re_mag = np.sqrt(re_ux ** 2 + re_uy ** 2)

    panels = (
        ("_re_ux",  re_ux,  "Re(u_x)", True),
        ("_re_uy",  re_uy,  "Re(u_y)", True),
        ("_re_mag", re_mag, "|Re(u)|", False),
    )
    for suffix, vals, label, sym in panels:
        _plot_field_on_phys_domain(
            vals, pts_x, pts_y, params,
            save_path=f"{root}{suffix}{ext}",
            title=title.replace("|u|", label),
            cbar_label=label,
            percentile=percentile, norm_type=norm_type, cells=cells,
            symmetric=sym,
        )


def plot_reference_field(
    ref_result: SolutionResult,
    save_path: str,
    title: str = "Reference field |u|",
    percentile: float = 95,
    norm_type: NormType = 'linear',
) -> None:
    """Save four diagnostic panels for a reference (uncloaked) solve.

    Thin wrapper over :func:`plot_field_panels` that pulls the node
    coordinates and connectivity from ``ref_result``.
    """
    pts = np.asarray(ref_result.mesh.points)
    cells = np.asarray(ref_result.mesh.cells)
    plot_field_panels(
        ref_result.u, pts[:, 0], pts[:, 1], ref_result.params,
        save_path=save_path, title=title,
        percentile=percentile, norm_type=norm_type, cells=cells,
    )


def plot_results(result: SolutionResult, percentile: float = 95,
                 norm_type: NormType = 'linear') -> None:
    """Save VTK and call the standalone plotting code."""
    vtk_path = save_vtk(result)
    plot_vtk_results(vtk_path, percentile=percentile, norm_type=norm_type)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "output/results.npz"
    percentile = float(sys.argv[2]) if len(sys.argv) > 2 else 95
    norm_type: NormType = sys.argv[3] if len(sys.argv) > 3 else 'asym_sigmoid'  # type: ignore[assignment]
    if data_path.endswith('.vtk'):
        plot_vtk_results(data_path, percentile=percentile, norm_type=norm_type)
    else:
        plot_npz_results(data_path, percentile=percentile, norm_type=norm_type)