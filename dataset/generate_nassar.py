#!/usr/bin/env python3
"""
Generate pixelated images of the Nassar 2018 degenerate polar lattice cloak.

Reference:
  Nassar H, Chen YY, Huang GL. 2018.
  "A degenerate polar lattice for cloaking in full two-dimensional
   elastodynamics and statics." Proc. R. Soc. A 474: 20180523.

Lattice design parameters (eq. 2.15) — global (position-independent):
    θ = arctan(√(λ/(2μ+λ)))
    α = 2(μ+λ)√(λ/(2μ+λ))
    β = ((2μ+λ)² - λ²) / (2√λ √(2μ+λ))

Position-dependent (through f = (‖x‖ − rᵢ)/‖x‖):
    a/b = f √((2μ+λ)/λ)          radial/tangential cell aspect ratio
    κ   = λμ/(λ−μ) · (1−f)²/f   torsion spring constant
    b   = 2π‖x‖ / N              tangential cell size

Stability constraint: μ ≤ λ  (so that κ > 0 everywhere).

Each unit cell at polar position (r, φ) contains:
  - A rigid rectangular mass (orange)
  - Two diagonal springs α connecting to nodes at r1 = (a/2)m+(b/2)n
    and r2 = −(a/2)m+(b/2)n  (blue)
  - One tangential spring β connecting to node at r3 = −b·n  (red)

Usage:
    python gen_lattice.py [N]          # generate N designs (default 20)
    python gen_lattice.py 50 --seed 7  # reproducible
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon


# ---------------------------------------------------------------------------
# Physics / lattice parameter calculations
# ---------------------------------------------------------------------------

def global_params(lam: float, mu: float):
    """
    Compute global lattice parameters from Lamé constants.
    Stability requires mu <= lam.
    Returns (theta, alpha, beta).
    """
    if mu > lam:
        raise ValueError(f"Stability requires mu <= lam, got mu={mu:.4g} lam={lam:.4g}")
    theta = np.arctan(np.sqrt(lam / (2 * mu + lam)))
    alpha = 2 * (mu + lam) * np.sqrt(lam / (2 * mu + lam))
    beta  = ((2 * mu + lam) ** 2 - lam ** 2) / (
             2 * np.sqrt(lam) * np.sqrt(2 * mu + lam))
    return theta, alpha, beta


def local_params(r: float, ri: float, lam: float, mu: float, N_sectors: int):
    """
    Compute position-dependent unit-cell dimensions at radius r.
    Returns (a, b, f, kappa).
    """
    f     = (r - ri) / r
    b     = 2 * np.pi * r / N_sectors
    a     = b * f * np.sqrt((2 * mu + lam) / lam)
    kappa = (lam * mu / (lam - mu)) * (1 - f) ** 2 / f if lam != mu else np.inf
    return a, b, f, kappa


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def draw_square_lattice(ax, L, N_x, N_y, spring_px: int = 2, dpi: int = 100):
    """
    Draw the degenerate lattice on a square domain [-L/2, L/2]²
    tiled by N_x × N_y Cartesian unit cells.

    Cell dimensions are set by the grid so that spring endpoints land exactly
    on shared lattice nodes (grid corners), making the topology visible:
        a = L / N_x   (cell width  in x̂)
        b = L / N_y   (cell height in ŷ)

    spring_px : positive integer — spring stroke width in output pixels.
                Converted to matplotlib points via  lw = spring_px * 72 / dpi.

    Spring colours as in the polar drawing:
      α spring 1 → shared node at c + (a/2)x̂ + (b/2)ŷ   (blue)
      α spring 2 → shared node at c − (a/2)x̂ + (b/2)ŷ   (blue)
      β spring   → mass centre   at c          − b·ŷ      (red)
    """
    if spring_px < 1:
        raise ValueError(f"spring_px must be a positive integer, got {spring_px}")

    pts_per_px = 72 / dpi          # 1 pixel = 72/dpi points at this dpi
    lw_alpha   = spring_px * pts_per_px
    lw_beta    = max(1, spring_px - 1) * pts_per_px   # β one pixel thinner, ≥1 px
    node_px    = spring_px + 1     # node dot slightly larger than the spring

    a = L / N_x   # cell width  — α endpoints land on grid corners
    b = L / N_y   # cell height

    x_edges   = np.linspace(-L / 2, L / 2, N_x + 1)
    y_edges   = np.linspace(-L / 2, L / 2, N_y + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    m = np.array([1.0, 0.0])   # x̂
    n = np.array([0.0, 1.0])   # ŷ

    # ---- springs (drawn first, behind masses) ----
    for cx in x_centers:
        for cy in y_centers:
            c = np.array([cx, cy])
            nodes  = [
                c + (a / 2) * m + (b / 2) * n,   # α1: upper-right grid corner
                c - (a / 2) * m + (b / 2) * n,   # α2: upper-left  grid corner
                c - b * n,                         # β:  mass centre below
            ]
            colors = ['#2255cc', '#2255cc', '#cc3311']
            lws    = [lw_alpha, lw_alpha, lw_beta]
            for nd, col, lw in zip(nodes, colors, lws):
                ax.plot([c[0], nd[0]], [c[1], nd[1]],
                        '-', color=col, lw=lw, alpha=0.80,
                        solid_capstyle='round', zorder=2)

    # ---- lattice nodes at grid corners (shared α-spring endpoints) ----
    # marker size in points = node_px pixels converted to points
    ms = node_px * pts_per_px
    for nx_ in x_edges:
        for ny_ in y_edges:
            ax.plot(nx_, ny_, 'o', color='#2255cc',
                    markersize=ms, markeredgewidth=0.0,
                    alpha=0.85, zorder=3)

    # ---- masses (rectangles in local m-n frame) ----
    ha = 0.28 * a
    hb = 0.42 * b
    for cx in x_centers:
        for cy in y_centers:
            c = np.array([cx, cy])
            corners = np.array([
                c + ha * m + hb * n,
                c - ha * m + hb * n,
                c - ha * m - hb * n,
                c + ha * m - hb * n,
            ])
            ax.add_patch(MplPolygon(
                corners, closed=True,
                facecolor='#FF8C00', edgecolor='#7B3F00',
                linewidth=0.2, zorder=4
            ))


def make_square_image(params: dict, out_path: str,
                      image_size: int = 512, spring_px: int = 2):
    """Render one Cartesian square-lattice design and save as a PNG pixel image."""
    lam, mu = params['lam'], params['mu']
    rc      = params['rc']
    N_x     = params['N_sectors']   # reuse same count for x
    N_y     = params['N_layers']    # reuse same count for y

    L      = 2.0 * rc               # square domain side length
    domain = 0.55 * L               # half-width shown (same relative margin as polar)

    dpi = 100
    fig, ax = plt.subplots(figsize=(512 / dpi, 512 / dpi), dpi=dpi)
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_aspect('equal')
    ax.axis('off')

    fig.patch.set_facecolor('#c8ddf0')
    ax.set_facecolor('#c8ddf0')

    # square lattice region background
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((-L / 2, -L / 2), L, L,
                            color='#a0bedd', zorder=0))

    draw_square_lattice(ax, L, N_x, N_y, spring_px=spring_px, dpi=dpi)

    # boundary
    ax.add_patch(Rectangle((-L / 2, -L / 2), L, L,
                            fill=False, edgecolor='#222222',
                            linewidth=0.7, zorder=6))

    theta, _, _ = global_params(lam, mu)
    a = L / N_x
    b = L / N_y
    ax.set_title(
        f"λ={lam:.2f}  μ={mu:.2f}  θ={np.degrees(theta):.1f}°  "
        f"L={L:.2f}  a={a:.3f}  b={b:.3f}  Nx={N_x}  Ny={N_y}  spring={spring_px}px",
        fontsize=7, pad=3
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


def draw_lattice(ax, lam, mu, ri, rc, N_sectors, N_layers):
    """
    Draw the degenerate polar lattice inside the cloak annulus [ri, rc].

    Spring topology from each mass at (r, φ):
      α spring 1 → node at r1 = +(a/2)m + (b/2)n   (outer-CCW)
      α spring 2 → node at r2 = −(a/2)m + (b/2)n   (inner-CCW)
      β spring   → node at r3 =           −b·n      (CW midpoint)
    """
    r_edges   = np.linspace(ri, rc, N_layers + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    phi_step  = 2 * np.pi / N_sectors
    phi_centers = np.arange(N_sectors) * phi_step + phi_step / 2

    # ---- springs (drawn first, behind masses) ----
    for r in r_centers:
        a, b, f, _ = local_params(r, ri, lam, mu, N_sectors)
        for phi in phi_centers:
            m = np.array([ np.cos(phi),  np.sin(phi)])   # radial outward
            n = np.array([-np.sin(phi),  np.cos(phi)])   # tangential CCW
            c = r * m                                     # cell center

            nodes = [
                c + (a / 2) * m + (b / 2) * n,   # r1  (α spring)
                c - (a / 2) * m + (b / 2) * n,   # r2  (α spring)
                c - b * n,                         # r3  (β spring)
            ]
            colors = ['#2255cc', '#2255cc', '#cc3311']
            lws    = [0.7, 0.7, 0.55]
            for nd, col, lw in zip(nodes, colors, lws):
                ax.plot([c[0], nd[0]], [c[1], nd[1]],
                        '-', color=col, lw=lw, alpha=0.80,
                        solid_capstyle='round', zorder=2)

    # ---- masses (rectangles in local m-n frame) ----
    for r in r_centers:
        a, b, f, _ = local_params(r, ri, lam, mu, N_sectors)
        ha = 0.28 * a   # half-extent along m (radial)
        hb = 0.42 * b   # half-extent along n (tangential)
        for phi in phi_centers:
            m = np.array([ np.cos(phi),  np.sin(phi)])
            n = np.array([-np.sin(phi),  np.cos(phi)])
            c = r * m
            corners = np.array([
                c + ha * m + hb * n,
                c - ha * m + hb * n,
                c - ha * m - hb * n,
                c + ha * m - hb * n,
            ])
            ax.add_patch(MplPolygon(
                corners, closed=True,
                facecolor='#FF8C00', edgecolor='#7B3F00',
                linewidth=0.2, zorder=4
            ))


def make_image(params: dict, out_path: str, image_size: int = 512):
    """Render one lattice design and save as a PNG pixel image."""
    lam, mu  = params['lam'],      params['mu']
    ri,  rc  = params['ri'],       params['rc']
    N_sec    = params['N_sectors']
    N_lay    = params['N_layers']

    domain   = 1.55 * rc   # physical half-width shown

    dpi = 100
    fig, ax = plt.subplots(figsize=(image_size / dpi, image_size / dpi), dpi=dpi)
    ax.set_xlim(-domain, domain)
    ax.set_ylim(-domain, domain)
    ax.set_aspect('equal')
    ax.axis('off')

    # background: isotropic medium
    fig.patch.set_facecolor('#c8ddf0')
    ax.set_facecolor('#c8ddf0')

    # cloak annulus background (slightly darker)
    ax.add_patch(plt.Circle((0, 0), rc, color='#a0bedd', zorder=0))

    # lattice
    draw_lattice(ax, lam, mu, ri, rc, N_sec, N_lay)

    # inclusion void
    ax.add_patch(plt.Circle((0, 0), ri, color='#f0f0f0', zorder=5))

    # boundary circles
    for r in (ri, rc):
        ax.add_patch(plt.Circle((0, 0), r, fill=False,
                                color='#222222', lw=0.7, zorder=6))

    # parameter annotation
    theta, _, _ = global_params(lam, mu)
    ax.set_title(
        f"λ={lam:.2f}  μ={mu:.2f}  θ={np.degrees(theta):.1f}°  "
        f"rᵢ={ri:.2f}  rᶜ={rc:.2f}  N={N_sec}  M={N_lay}",
        fontsize=7, pad=3
    )

    fig.savefig(out_path, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parameter sampling
# ---------------------------------------------------------------------------

def sample_params(N: int, seed: int = 42) -> list[dict]:
    """
    Randomly sample N sets of valid lattice design parameters.

    Sampled variables:
      lam  ∈ [1, 5]           (normalised Lamé λ)
      mu   ∈ [0.05λ, λ]       (Lamé μ; upper bound ensures κ > 0)
      ri   ∈ [0.15, 0.40]     (inner cloak radius)
      rc   ∈ [ri+0.15, ri+0.50] (outer cloak radius)
      N    ∈ [2, 24]          (sectors)
      M    ∈ [1, 8]           (radial layers)
    """
    rng = np.random.default_rng(seed)
    designs = []
    for _ in range(N):
        lam   = rng.uniform(1.0, 5.0)
        mu    = rng.uniform(0.05 * lam, lam)
        ri    = rng.uniform(0.15, 0.40)
        rc    = ri + rng.uniform(0.15, 0.50)
        N_sec = int(rng.integers(2, 25))
        N_lay = int(rng.integers(1, 9))
        designs.append(dict(lam=lam, mu=mu, ri=ri, rc=rc,
                            N_sectors=N_sec, N_layers=N_lay))
    return designs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Nassar 2018 degenerate lattice images "
                    "(polar annular cloak or Cartesian square tiling).")
    parser.add_argument("N", nargs="?", type=int, default=20,
                        help="Number of designs to generate (default: 20)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512,
                        help="Image side length in pixels (default: 512)")
    parser.add_argument("--out", default="output/lattice_designs",
                        help="Output directory")
    parser.add_argument("--mode", choices=["polar", "square", "both"],
                        default="polar",
                        help="Geometry: polar annular cloak, Cartesian square "
                             "tiling, or both (default: polar)")
    parser.add_argument("--spring-px", type=int, default=2,
                        help="Spring stroke width in pixels, positive integer "
                             "(square mode only, default: 2)")
    args = parser.parse_args()
    if args.spring_px < 1:
        parser.error("--spring-px must be a positive integer")

    os.makedirs(args.out, exist_ok=True)
    designs = sample_params(args.N, seed=args.seed)

    meta_path = os.path.join(args.out, "params.json")
    with open(meta_path, "w") as f:
        json.dump(designs, f, indent=2)
    print(f"Parameters → {meta_path}")

    for i, p in enumerate(designs):
        theta, _, _ = global_params(p["lam"], p["mu"])
        print(f"[{i+1:3d}/{args.N}] "
              f"λ={p['lam']:.2f} μ={p['mu']:.2f} θ={np.degrees(theta):.1f}° "
              f"rᵢ={p['ri']:.2f} rᶜ={p['rc']:.2f} "
              f"N={p['N_sectors']} M={p['N_layers']}")

        if args.mode in ("polar", "both"):
            out_path = os.path.join(args.out, f"design_{i:04d}_polar.png")
            make_image(p, out_path, image_size=args.size)

        if args.mode in ("square", "both"):
            out_path = os.path.join(args.out, f"design_{i:04d}_square.png")
            make_square_image(p, out_path, image_size=args.size,
                              spring_px=args.spring_px)

    suffix = "" if args.mode == "polar" else f" ({args.mode})"
    print(f"\n✓  {args.N} designs saved to {args.out}/{suffix}")


if __name__ == "__main__":
    main()
