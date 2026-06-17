"""Plot C and rho profiles from optimized cell parameters along different lines.

Reads optimized_params.npz (cell_C_flat, cell_rho) and reconstructs the spatial
distribution on the cell grid.  Plots profiles along vertical and horizontal
lines through the cloak, comparing with the analytical (transformational) C_eff
and rho_eff.

Usage:
    python scripts/plot_optimized_profiles.py configs/triangular_optimize_neural_flat2.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak.config import DerivedParams, load_config
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_eff, C_iso, rho_eff, _get_converters


def main():
    parser = argparse.ArgumentParser(description="Plot optimized C/rho profiles")
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    dp = DerivedParams.from_config(cfg)
    geo = TriangularCloakGeometry.from_params(dp)

    outdir = Path(cfg.output_dir)
    npz_path = outdir / "optimized_params.npz"
    if not npz_path.exists():
        print(f"ERROR: {npz_path} not found")
        sys.exit(1)

    data = np.load(npz_path)
    cell_C_flat = data["cell_C_flat"]  # (n_cells, n_C_params)
    cell_rho = data["cell_rho"]        # (n_cells,)

    n_x = cfg.cells.n_x
    n_y = cfg.cells.n_y
    n_C_params = cfg.cells.n_C_params

    # Reconstruct cell grid (same logic as CellDecomposition)
    x_min = geo.x_c - geo.c
    x_max = geo.x_c + geo.c
    y_min = geo.y_top - geo.b
    y_max = geo.y_top
    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y

    cx = x_min + (np.arange(n_x) + 0.5) * cell_dx
    cy = y_min + (np.arange(n_y) + 0.5) * cell_dy

    # Reshape to (n_x, n_y, ...)
    C_flat_grid = cell_C_flat.reshape(n_x, n_y, n_C_params)
    rho_grid = cell_rho.reshape(n_x, n_y)

    # Get flat converters for this parameterization
    to_flat, from_flat = _get_converters(n_C_params)

    # Background material
    C0 = C_iso(dp.lam, dp.mu)
    C0_flat = np.array(to_flat(C0))  # (n_C_params,)

    # Labels for each flat parameterization
    _FLAT_LABELS = {
        2: [r"$\lambda$", r"$\mu$"],
        4: [r"$C_{1111}$", r"$C_{2222}$", r"$C_{1212}$", r"$C_{1122}$"],
        6: [r"$C_{1111}$", r"$C_{2222}$", r"$C_{1122}$",
            r"$C_{1212}$", r"$C_{2121}$", r"$C_{1221}$"],
        10: [r"$M_{11,11}$", r"$M_{11,22}$", r"$M_{11,12}$", r"$M_{11,21}$",
             r"$M_{22,22}$", r"$M_{22,12}$", r"$M_{22,21}$",
             r"$M_{12,12}$", r"$M_{12,21}$", r"$M_{21,21}$"],
        16: [f"$M_{{{i},{j}}}$" for i in range(4) for j in range(4)],
    }
    flat_labels = _FLAT_LABELS.get(n_C_params,
                                   [f"p{i}" for i in range(n_C_params)])

    # ── Define sample lines ─────────────────────────────────────────
    # Vertical lines at different x positions
    x_center = geo.x_c
    x_offcenter = geo.x_c + geo.c / 4.0
    x_offcenter2 = geo.x_c - geo.c / 4.0

    # Horizontal line at mid-depth
    y_mid = geo.y_top - (geo.a + geo.b) / 2.0

    lines = {
        f"Vertical x=x_c (center)": ("y", cx, cy, x_center),
        f"Vertical x=x_c+c/4": ("y", cx, cy, x_offcenter),
        f"Vertical x=x_c-c/4": ("y", cx, cy, x_offcenter2),
        f"Horizontal y=mid-depth": ("x", cx, cy, y_mid),
    }

    figdir = outdir / "profiles"
    figdir.mkdir(exist_ok=True)

    # ── Analytical reference along a dense line ──────────────────────
    def analytical_profile_vertical(x_val, n_pts=300):
        """Return (depths, flat_ref, rho_ref) for analytical C_eff along vertical."""
        depths = np.linspace(0, geo.b * 1.2, n_pts)
        ys = geo.y_top - depths
        pts = jnp.array([[x_val, y] for y in ys])
        C_ref = jax.vmap(lambda p: C_eff(p, geo, C0))(pts)
        flat_ref = np.array(jax.vmap(to_flat)(C_ref))  # (n_pts, n_C_params)
        rho_ref = np.array(jax.vmap(lambda p: rho_eff(p, geo, dp.rho0))(pts))
        return depths, flat_ref, rho_ref

    def analytical_profile_horizontal(y_val, n_pts=300):
        """Return (xs, flat_ref, rho_ref) for analytical C_eff along horizontal."""
        xs = np.linspace(x_min - 0.1 * geo.c, x_max + 0.1 * geo.c, n_pts)
        pts = jnp.array([[x, y_val] for x in xs])
        C_ref = jax.vmap(lambda p: C_eff(p, geo, C0))(pts)
        flat_ref = np.array(jax.vmap(to_flat)(C_ref))  # (n_pts, n_C_params)
        rho_ref = np.array(jax.vmap(lambda p: rho_eff(p, geo, dp.rho0))(pts))
        return xs, flat_ref, rho_ref

    # ── Plot each line ──────────────────────────────────────────────
    for title, (axis, cx_arr, cy_arr, fixed_val) in lines.items():
        if axis == "y":
            ix = np.argmin(np.abs(cx_arr - fixed_val))
            xcoords = geo.y_top - cy_arr  # depths
            flat_line = C_flat_grid[ix, :, :]     # (n_y, n_C_params)
            rho_line = rho_grid[ix, :]             # (n_y,)
            xlabel = "Depth from free surface [m]"
            ref_x, flat_ref, rho_ref = analytical_profile_vertical(fixed_val)
        else:
            iy = np.argmin(np.abs(cy_arr - fixed_val))
            xcoords = cx_arr
            flat_line = C_flat_grid[:, iy, :]      # (n_x, n_C_params)
            rho_line = rho_grid[:, iy]              # (n_x,)
            xlabel = "x position [m]"
            ref_x, flat_ref, rho_ref = analytical_profile_horizontal(fixed_val)

        # ── C components plot ──
        fig, ax = plt.subplots(figsize=(10, 6))
        for k in range(n_C_params):
            vals = flat_line[:, k]
            line_obj, = ax.plot(xcoords, vals / 1e9, 'o', markersize=3,
                                label=f"{flat_labels[k]} (opt)")
            # Analytical reference
            ax.plot(ref_x, flat_ref[:, k] / 1e9, '-',
                    color=line_obj.get_color(), alpha=0.6, linewidth=1.5)
            # Background reference
            ref_bg = C0_flat[k] / 1e9
            if abs(ref_bg) > 1e-6:
                ax.axhline(ref_bg, color=line_obj.get_color(),
                           ls=':', lw=0.6, alpha=0.4)

        if axis == "y":
            ax.axvline(geo.a, color='grey', ls='--', lw=2.0, label='inner (a)')
            ax.axvline(geo.b, color='grey', ls=':',  lw=2.0, label='outer (b)')

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Stiffness [GPa]")
        ax.set_title(f"C parameters — {title}")
        ax.legend(fontsize=7, ncol=2, loc="best")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        safe_title = title.replace(" ", "_").replace("=", "").replace("/", "")
        fname = figdir / f"C_{safe_title}.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
        plt.close(fig)

        # ── rho plot ──
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xcoords, rho_line, 'o', markersize=3, label=r"$\rho$ (opt)")
        ax.plot(ref_x, rho_ref, '-', alpha=0.6, linewidth=1.5,
                label=r"$\rho$ (analytical)")

        if axis == "y":
            ax.axvline(geo.a, color='grey', ls='--', lw=2.0, label='inner (a)')
            ax.axvline(geo.b, color='grey', ls=':',  lw=2.0, label='outer (b)')

        ax.axhline(dp.rho0, color='grey', ls=':', lw=1.0, alpha=0.5,
                   label=r"$\rho_0$ (background)")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\rho$ [kg/m³]")
        ax.set_title(f"Density — {title}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        fname = figdir / f"rho_{safe_title}.png"
        fig.savefig(fname, dpi=150)
        print(f"Saved → {fname}")
        plt.close(fig)

    # ── 2D heatmaps of native flat parameters + rho ───────────────────
    n_plots = n_C_params + 1  # flat params + rho
    ncols = min(n_plots, 3)
    nrows = (n_plots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes).ravel()

    for k in range(n_C_params):
        ax = axes[k]
        vals = C_flat_grid[:, :, k]
        im = ax.pcolormesh(cx, cy, (vals / 1e9).T, shading='auto')
        fig.colorbar(im, ax=ax, label="GPa")
        ax.set_title(flat_labels[k])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect('equal')

    # rho heatmap
    ax = axes[n_C_params]
    im = ax.pcolormesh(cx, cy, rho_grid.T, shading='auto')
    fig.colorbar(im, ax=ax, label="kg/m³")
    ax.set_title(r"$\rho$")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect('equal')

    # Hide unused axes
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(f"Optimized material fields (n_C_params={n_C_params})", fontsize=14)
    fig.tight_layout()
    fname = figdir / "heatmaps.png"
    fig.savefig(fname, dpi=150)
    print(f"Saved → {fname}")
    plt.close(fig)


if __name__ == "__main__":
    main()
