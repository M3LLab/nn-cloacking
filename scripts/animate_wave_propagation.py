"""Animate elastodynamic wave propagation in the time domain.

The forward FEM is frequency-domain (steady-state harmonic with PML). Each
node carries DOFs ``[Re(ux), Re(uy), Im(ux), Im(uy)]`` corresponding to a
complex displacement ``U(x) = U_R(x) + i U_I(x)``.  The PML/Rayleigh damping
in ``rayleigh_cloak.problem`` uses the ``e^{+i omega t}`` convention (the
stiffness multiplier is ``(1 + i xi)``), so the physical time-domain field
is::

    u(x,t) = Re[U(x) e^{+i omega t}]
           = U_R(x) cos(omega t) - U_I(x) sin(omega t)

We render frames at a sequence of phases ``phi = omega t`` in [0, 2*pi*N),
then stitch the frames into an MP4 with ffmpeg (bundled via
``imageio_ffmpeg``).  The approach mirrors jax-fem's
``applications/thermal_mechanical/animation.py`` (a sequence of VTK frames
is replaced here by a sequence of PNGs since we do not need ParaView).

Usage
-----

::

    python scripts/animate_wave_propagation.py output/multifreq_small
    python scripts/animate_wave_propagation.py output/cell20_cement_init

Arguments
---------
output_dir : positional
    Directory containing ``config.yaml`` and ``optimized_params.npz``.
    The MP4 is written to ``<output_dir>/wave_propagation.mp4``.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Allow ``python scripts/animate_wave_propagation.py`` from project root
# without requiring PYTHONPATH (mirrors how `run.py` is invoked).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

import jax.numpy as jnp
import imageio_ffmpeg
from jax_fem.solver import solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.plot import _build_norm
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry

import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── solve ────────────────────────────────────────────────────────────


def forward_solve(config, opt_params_npz: Path):
    """Run a single homogenised-cell forward solve with optimised params.

    Returns ``(u, cloak_mesh, geometry, derived_params)``.
    """
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    full_mesh = generate_mesh_full(config, dp, geometry)
    cloak_mesh, _kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  mesh: {len(cloak_mesh.points)} nodes, "
          f"{cloak_mesh.cells.shape[0]} elements")

    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(dp.lam, dp.mu)
    CellMaterial(
        geometry, C0, dp.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
    )

    npz = np.load(opt_params_npz)
    cell_C_flat = jnp.asarray(npz["cell_C_flat"])
    cell_rho = jnp.asarray(npz["cell_rho"])
    n_expected = cell_decomp.n_cells
    if cell_C_flat.shape[0] != n_expected:
        raise SystemExit(
            f"optimised params have {cell_C_flat.shape[0]} cells, but config "
            f"declares n_x*n_y={n_expected}"
        )

    problem = build_problem(cloak_mesh, config, dp, geometry, cell_decomp)
    problem.set_params((cell_C_flat, cell_rho))

    solver_opts = {"petsc_solver": {
        "ksp_type": config.solver.ksp_type,
        "pc_type": config.solver.pc_type,
    }}
    print(f"  solving at f*={config.domain.f_star:.3f} ...")
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u = np.asarray(sol_list[0])
    return u, cloak_mesh, geometry, dp


# ── animation ────────────────────────────────────────────────────────


def render_frames(
    u: np.ndarray,
    mesh,
    dp: DerivedParams,
    *,
    n_periods: int,
    n_frames_per_period: int,
    frame_dir: Path,
    f_star: float,
    percentile: float = 95,
    norm_type: str = "linear",
) -> int:
    """Render PNG frames of the time-resolved wave field, using the same
    visualisation as ``rayleigh_cloak.plot.plot_displacement_field`` (the
    per-step plot from ``run_optimize.py``).

    Plots ``|u(x, t)| = sqrt(u_x(t)**2 + u_y(t)**2)`` — the time-domain
    analogue of the static phase-invariant magnitude ``|U|`` — on the
    physical domain only, with the same 100-level ``RdBu_r``
    ``tricontourf``, percentile-clipped norm built via ``_build_norm``,
    source star, and yellow cloak outline.

    The colour-scale limits are fixed across frames using the percentile
    of ``|U(x)| = sqrt(|U_x|^2 + |U_y|^2)`` (the static field plotted by
    ``plot_displacement_field``), which bounds ``|u(x,t)|`` from above at
    every instant.
    """
    pts_x = np.asarray(mesh.points[:, 0])
    pts_y = np.asarray(mesh.points[:, 1])
    cells = np.asarray(mesh.cells)

    # Frequency-domain DOFs: U_x = Re_ux + i Im_ux, U_y = Re_uy + i Im_uy.
    Re_ux = u[:, 0]
    Re_uy = u[:, 1]
    Im_ux = u[:, 2]
    Im_uy = u[:, 3]

    # Physical-domain mask (mirrors plot_displacement_field).
    x_off, y_off, W, H = dp.x_off, dp.y_off, dp.W, dp.H
    phys = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8)
            & (pts_y >= y_off - 1e-8))
    px = pts_x[phys] - x_off
    py = pts_y[phys] - y_off

    # Re-index mesh connectivity into the physical-domain submask so the
    # cut-out cloak void is preserved as a hole (matplotlib's default
    # Delaunay triangulation would fill it in).
    new_index = -np.ones(pts_x.shape[0], dtype=np.int64)
    new_index[np.where(phys)[0]] = np.arange(int(phys.sum()))
    cell_keep = phys[cells].all(axis=1)
    triang = mtri.Triangulation(px, py, new_index[cells[cell_keep]])

    URx, UIx = Re_ux[phys], Im_ux[phys]
    URy, UIy = Re_uy[phys], Im_uy[phys]

    # Phase-invariant complex magnitude |U| = sqrt(|U_x|^2 + |U_y|^2),
    # used both as the colour-scale envelope here and as the static field
    # in rayleigh_cloak.plot.plot_displacement_field.  At any instant t,
    # |u(x,t)| <= |U(x)|, so this fully bounds every animation frame.
    full_mag = np.sqrt(URx ** 2 + UIx ** 2 + URy ** 2 + UIy ** 2)
    vmin_v = float(np.percentile(full_mag, 100 - percentile))
    vmax_v = float(np.percentile(full_mag, percentile))
    if vmax_v < 1e-30:
        vmax_v = 1.0
    norm = _build_norm(norm_type, vmin_v, vmax_v, mid=0.25 * vmax_v)
    levels = np.linspace(vmin_v, vmax_v, 100)

    # Cloak outline & source marker (physical coords) — same as
    # plot_displacement_field.
    a, b, c_hw = dp.a, dp.b, dp.c
    xc = W / 2.0
    x_src_phys = dp.x_src - x_off

    n_frames = int(n_periods * n_frames_per_period)
    frame_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n_frames):
        # e^{+i omega t} convention (matches rayleigh_cloak.problem):
        # u(x,t) = Re[U e^{+i phi}] = U_R cos(phi) - U_I sin(phi)
        phi = 2.0 * np.pi * (k / n_frames_per_period)
        c, s = np.cos(phi), np.sin(phi)
        ux_t = URx * c - UIx * s
        uy_t = URy * c - UIy * s
        mag_t = np.sqrt(ux_t ** 2 + uy_t ** 2)

        fig, ax = plt.subplots(figsize=(13, 4))
        tc = ax.tricontourf(triang, mag_t, levels=levels, cmap="RdBu_r",
                            norm=norm, extend="both")
        ax.plot(x_src_phys, H, "r*", markersize=12)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - b, H],
                ls="--", color="yellow", lw=1.2)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - a, H],
                ls="--", color="yellow", lw=1.2)

        fig.colorbar(tc, ax=ax, shrink=0.8, label="|u(t)|")
        period_idx = k // n_frames_per_period
        within = (k % n_frames_per_period) / n_frames_per_period
        ax.set_title(
            f"|u(t)|  (envelope = |U|)  |  f* = {f_star:.3f}  "
            f"|  period {period_idx + 1}/{n_periods}, t/T = {within:.2f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")

        path = frame_dir / f"frame_{k:04d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return n_frames


def encode_video(frame_dir: Path, out_path: Path, fps: int) -> None:
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg, "-y",
        "-framerate", str(fps),
        "-i", str(frame_dir / "frame_%04d.png"),
        "-pix_fmt", "yuv420p",
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(out_path),
    ]
    print("  ffmpeg:", " ".join(cmd))
    subprocess.run(cmd, check=True)


# ── main ─────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("output_dir", type=Path,
                   help="Directory with config.yaml + optimized_params.npz; "
                        "video is written here as wave_propagation.mp4")
    p.add_argument("--config", type=Path, default=None,
                   help="Override config (default: <output_dir>/config.yaml)")
    p.add_argument("--params", type=Path, default=None,
                   help="Override params (default: <output_dir>/optimized_params.npz)")
    p.add_argument("--f-star", type=float, default=None,
                   help="Frequency to animate (default: config domain.f_star)")
    p.add_argument("--n-periods", type=int, default=2)
    p.add_argument("--frames-per-period", type=int, default=30)
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--norm-type", choices=("linear", "sigmoid", "asym_sigmoid"),
                   default="linear",
                   help="Color norm (matches rayleigh_cloak.plot._build_norm)")
    p.add_argument("--keep-frames", action="store_true",
                   help="Do not delete the temporary frame directory")
    args = p.parse_args()

    out_dir: Path = args.output_dir
    config_path = args.config or (out_dir / "config.yaml")
    params_path = args.params or (out_dir / "optimized_params.npz")
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    if not params_path.exists():
        raise SystemExit(f"params not found: {params_path}")

    cfg = load_config(config_path)
    if args.f_star is not None:
        cfg = cfg.model_copy(update={
            "domain": cfg.domain.model_copy(update={"f_star": float(args.f_star)})
        })
    print(f"=== Forward solve in {out_dir} (f*={cfg.domain.f_star:.3f}) ===")
    u, cloak_mesh, _geometry, dp = forward_solve(cfg, params_path)

    f_tag = f"f{cfg.domain.f_star:.2f}"
    frame_dir = out_dir / f"_frames_{f_tag}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    print("=== Rendering frames ===")
    n = render_frames(
        u, cloak_mesh, dp,
        n_periods=args.n_periods,
        n_frames_per_period=args.frames_per_period,
        frame_dir=frame_dir,
        f_star=cfg.domain.f_star,
        norm_type=args.norm_type,
    )
    print(f"  wrote {n} frames -> {frame_dir}")

    out_path = out_dir / f"wave_propagation_{f_tag}.mp4"
    print("=== Encoding video ===")
    encode_video(frame_dir, out_path, fps=args.fps)
    print(f"  video -> {out_path}")

    if not args.keep_frames:
        shutil.rmtree(frame_dir)
        print(f"  cleaned up {frame_dir}")


if __name__ == "__main__":
    main()
