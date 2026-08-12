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

    # ideal (analytic transformation-optics) cloak, zoomed on the cloak
    python scripts/animate_wave_propagation.py output/multifreq_small \\
        --ideal --zoom 2.5

Arguments
---------
output_dir : positional
    Directory containing ``config.yaml`` and (unless ``--ideal``)
    ``optimized_params.npz``.  The MP4 is written here.
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
from rayleigh_cloak.mesh import extract_submesh
from rayleigh_cloak.plot import _build_norm
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry, _full_mesh, solve_reference

import logging

logging.getLogger("jax_fem").setLevel(logging.WARNING)


# ── solve ────────────────────────────────────────────────────────────


def forward_solve(config, opt_params_npz: Path | None, *, case: str = "optimized"):
    """Run a single forward solve for one of the three cloak scenarios.

    ``case`` selects what fills the domain, mirroring the sweeps in
    ``scripts/frequency_sweep.py``:

    ``"optimized"``
        Cut-out mesh, cloak filled with the optimised homogenised-cell
        materials from ``opt_params_npz``.
    ``"ideal"``
        Cut-out mesh, cloak filled with the analytic transformation-optics
        ``C_eff``/``rho_eff`` (``is_reference`` false, no cell
        decomposition) — ``run_ideal_sweep``'s case.  Needs no params.
    ``"reference"``
        Homogeneous half-space on the *full* mesh: no cloak and no defect
        cut-out at all — ``solve_reference``'s case.  Needs no params.

    Returns ``(u, mesh, geometry, derived_params)``.
    """
    dp = DerivedParams.from_config(config)
    geometry = _create_geometry(config, dp)

    full_mesh = _full_mesh(config, dp, geometry)

    if case == "reference":
        # Full mesh, no submesh extraction -> no void, homogeneous material.
        print(f"  mesh: {len(full_mesh.points)} nodes, "
              f"{full_mesh.cells.shape[0]} elements (full, no cut-out)")
        print(f"  solving at f*={config.domain.f_star:.3f} ...")
        ref = solve_reference(config, mesh=full_mesh)
        return np.asarray(ref.u), full_mesh, geometry, dp

    cloak_mesh, _kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  mesh: {len(cloak_mesh.points)} nodes, "
          f"{cloak_mesh.cells.shape[0]} elements")

    if case == "ideal":
        ideal_config = config.model_copy(update={"is_reference": False})
        problem = build_problem(cloak_mesh, ideal_config, dp, geometry)
    else:
        cell_decomp = CellDecomposition(
            geometry, config.cells.n_x, config.cells.n_y,
        )
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
                f"optimised params have {cell_C_flat.shape[0]} cells, but "
                f"config declares n_x*n_y={n_expected}"
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


#: Animatable fields, mirroring the static panels written by
#: ``rayleigh_cloak.plot.plot_field_panels``.  Each entry is
#: ``(per-frame value, colour-scale envelope, symmetric?, label)`` where the
#: envelope is a phase-invariant bound: at every instant the frame values lie
#: within it, so one percentile fixes the colour scale across all frames.
#:
#:   re_mag  |Re(u(t))| = sqrt(u_x(t)^2 + u_y(t)^2), bounded by |U|
#:   re_ux   Re(u_x(t)), signed, bounded by |U_x|
#:   re_uy   Re(u_y(t)), signed, bounded by |U_y|
FIELDS = ("re_mag", "re_ux", "re_uy")

_FIELD_LABEL = {
    "re_mag": "|Re(u(t))|",
    "re_ux": "Re(u_x(t))",
    "re_uy": "Re(u_y(t))",
}


def _field_envelope(field, URx, UIx, URy, UIy):
    """Phase-invariant bound on ``field``'s frame values (complex modulus)."""
    if field == "re_mag":
        return np.sqrt(URx ** 2 + UIx ** 2 + URy ** 2 + UIy ** 2)
    if field == "re_ux":
        return np.sqrt(URx ** 2 + UIx ** 2)
    return np.sqrt(URy ** 2 + UIy ** 2)


def _field_at_phase(field, URx, UIx, URy, UIy, c, s):
    """Evaluate ``field`` at phase ``phi``, with ``c, s = cos(phi), sin(phi)``.

    ``u(x,t) = Re[U e^{+i phi}] = U_R cos(phi) - U_I sin(phi)`` — the
    ``e^{+i omega t}`` convention used by ``rayleigh_cloak.problem``.
    """
    ux_t = URx * c - UIx * s
    uy_t = URy * c - UIy * s
    if field == "re_mag":
        return np.sqrt(ux_t ** 2 + uy_t ** 2)
    return ux_t if field == "re_ux" else uy_t


# ── transformation morph (reference -> ideal cloak) ──────────────────


def _morph_fraction(k: int, fpp: int, hold_periods: float,
                    morph_periods: float) -> float:
    """Smoothstepped morph fraction ``s`` in [0, 1] for frame ``k``.

    The wave keeps cycling throughout; ``s`` holds at 0 for ``hold_periods``,
    eases 0 -> 1 over ``morph_periods``, then holds at 1 for ``hold_periods``.
    """
    t_periods = k / fpp
    if morph_periods <= 0:
        return 1.0
    raw = (t_periods - hold_periods) / morph_periods
    raw = min(max(raw, 0.0), 1.0)
    return raw * raw * (3.0 - 2.0 * raw)   # smoothstep


def _warp_y(px: np.ndarray, py: np.ndarray, dp: DerivedParams,
            a_t: float) -> np.ndarray:
    """Push virtual (reference) coords through ``phi`` for inner depth ``a_t``.

    ``phi`` is the integral of :meth:`TriangularCloakGeometry.F_tensor` — the
    map whose Jacobian *is* ``F``.  It carries the intact half-space onto the
    half-space with a triangular void::

        x = X                                    (unchanged, F11=1, F12=0)
        d = a_t*(1-r) + D*(b - a_t)/b            r = |X-x_c|/c,  D = y_top - Y
        y = y_top - d

    inside the virtual triangle (``r<=1``, ``0<=D<=b(1-r)``), identity
    outside.  Verified against ``F_tensor``: ``dphi/dX == F`` exactly, and
    ``a_t=0`` gives the identity, so ``a_t = s*a`` sweeps a continuous family
    of *exact* cloaks from reference (s=0) to ideal (s=1).

    The free surface inside the opening (``D=0``) lands on the inner triangle
    edge ``d = a_t(1-r)`` — that descent is what opens the void.  The outer
    boundary ``D = b(1-r)`` is a fixed point, so the map is continuous with
    the identity outside and the coordinates are only displaced within the
    cloak.  Displacement values ride along unchanged: the BGM push-forward in
    ``materials.C_eff`` leaves the displacement indices untransformed, i.e.
    ``u_physical(phi(X)) = u_reference(X)``.
    """
    xc = dp.W / 2.0
    y_top, b, c_hw = dp.H, dp.b, dp.c
    r = np.abs(px - xc) / c_hw
    D = y_top - py
    inside = (r <= 1.0) & (D >= 0.0) & (D <= b * (1.0 - r))
    d_new = a_t * (1.0 - r) + D * (b - a_t) / b
    return np.where(inside, y_top - d_new, py)


def _view_box(
    dp: DerivedParams, zoom: float, zoom_depth: float | None = None,
) -> tuple[float, float, float, float]:
    """Axis limits (in physical coords) for a viewport around the cloak.

    The cloak is the triangle spanning ``x_c +- c`` at the free surface down
    to depth ``b``.  ``zoom`` scales that bounding box about the surface
    mid-point: ``zoom=1`` frames the cloak exactly, larger values pull back.

    Width and depth are scaled separately because the Rayleigh wave is a
    surface wave — it decays within roughly one wavelength of the free
    surface, so a viewport wide enough to show the wave crossing the cloak
    is deeper than it needs to be if both use one factor.  ``zoom_depth``
    defaults to ``zoom``.  Limits are clipped to the physical domain.
    """
    xc = dp.W / 2.0
    half_w = zoom * dp.c
    depth = (zoom if zoom_depth is None else zoom_depth) * dp.b
    return (
        max(0.0, xc - half_w), min(dp.W, xc + half_w),
        max(0.0, dp.H - depth), dp.H,
    )


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
    zoom: float | None = None,
    zoom_depth: float | None = None,
    field: str = "re_mag",
    clim: float | None = None,
    morph: bool = False,
    morph_periods: float = 4.0,
    hold_periods: float = 1.0,
    label: str = "",
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
    every instant.  When ``zoom`` is set the percentile is taken over the
    visible nodes only, so the colour scale resolves the near-cloak field
    instead of being set by the source-region amplitude off-screen.
    """
    pts_x = np.asarray(mesh.points[:, 0])
    pts_y = np.asarray(mesh.points[:, 1])
    cells = np.asarray(mesh.cells)
    # Quadratic (TRI6) meshes: matplotlib triangulates corner nodes only.
    if cells.ndim == 2 and cells.shape[1] > 3:
        cells = cells[:, :3]

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
    tri_conn = new_index[cells[cell_keep]]
    triang = mtri.Triangulation(px, py, tri_conn)

    URx, UIx = Re_ux[phys], Im_ux[phys]
    URy, UIy = Re_uy[phys], Im_uy[phys]

    envelope = _field_envelope(field, URx, UIx, URy, UIy)

    if zoom is not None:
        x_lo, x_hi, y_lo, y_hi = _view_box(dp, zoom, zoom_depth)
        visible = ((px >= x_lo) & (px <= x_hi) & (py >= y_lo) & (py <= y_hi))
        scale_mag = envelope[visible] if visible.any() else envelope
    else:
        x_lo, x_hi, y_lo, y_hi = 0.0, W, 0.0, H
        scale_mag = envelope

    # Signed components oscillate about zero, so they get a symmetric norm
    # (+-|U_i| percentile) on the diverging colormap; |Re(u(t))| is unsigned
    # and keeps the one-sided scale.
    symmetric = field != "re_mag"
    if symmetric:
        vlim = float(np.percentile(np.abs(scale_mag), percentile))
        if clim is not None:
            vlim = clim
        if vlim < 1e-30:
            vlim = 1.0
        vmin_v, vmax_v = -vlim, vlim
        norm = _build_norm(norm_type, vmin_v, vmax_v, mid=0.0, symmetric=True)
    else:
        vmin_v = float(np.percentile(scale_mag, 100 - percentile))
        vmax_v = float(np.percentile(scale_mag, percentile))
        if clim is not None:
            vmax_v = clim
        if vmax_v < 1e-30:
            vmax_v = 1.0
        norm = _build_norm(norm_type, vmin_v, vmax_v, mid=0.25 * vmax_v)
    levels = np.linspace(vmin_v, vmax_v, 100)
    print(f"  colour scale: [{vmin_v:.4g}, {vmax_v:.4g}]"
          f"{'  (--clim)' if clim is not None else ''}"
          f"  — pass --clim {vmax_v:.6g} to match this scale elsewhere")

    # Figure sized to the viewport so `aspect="equal"` leaves no dead space.
    aspect = (x_hi - x_lo) / max(y_hi - y_lo, 1e-12)
    fig_h = 5.0
    figsize = (min(max(fig_h * aspect + 2.0, 5.0), 16.0), fig_h)

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
        vals_t = _field_at_phase(field, URx, UIx, URy, UIy, c, s)

        # Morph: the nodal values never change — only where the nodes sit.
        # Carrying the reference field through phi_s IS the push-forward, so
        # the wave stays exact at every s rather than being tweened.
        if morph:
            s_morph = _morph_fraction(
                k, n_frames_per_period, hold_periods, morph_periods,
            )
            a_t = s_morph * a
            tri_k = mtri.Triangulation(px, _warp_y(px, py, dp, a_t), tri_conn)
        else:
            s_morph, a_t, tri_k = 1.0, a, triang

        # Constrained layout (not bbox_inches="tight") keeps every frame at
        # exactly figsize*dpi pixels — ffmpeg needs constant dimensions.
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        tc = ax.tricontourf(tri_k, vals_t, levels=levels, cmap="RdBu_r",
                            norm=norm, extend="both")
        ax.plot(x_src_phys, H, "r*", markersize=12)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - b, H],
                ls="--", color="yellow", lw=1.2)
        ax.plot([xc - c_hw, xc, xc + c_hw], [H, H - a_t, H],
                ls="--", color="yellow", lw=1.2)

        field_label = _FIELD_LABEL[field]
        fig.colorbar(tc, ax=ax, shrink=0.8, label=field_label)
        within = (k % n_frames_per_period) / n_frames_per_period
        prefix = f"{label}  |  " if label else ""
        if morph:
            # Live F components: the map's Jacobian at inner depth a_t.
            f21 = a_t / c_hw
            f22 = (b - a_t) / b
            ax.set_title(
                f"{prefix}{field_label}  |  f* = {f_star:.3f}\n"
                f"s = {s_morph:.2f}   a_t/a = {a_t / a:.2f}   "
                f"F = [[1, 0], [±{f21:.3f}, {f22:.3f}]]   t/T = {within:.2f}"
            )
        else:
            period_idx = k // n_frames_per_period
            ax.set_title(
                f"{prefix}{field_label}  |  f* = {f_star:.3f}  "
                f"|  period {period_idx + 1}/{n_periods}, t/T = {within:.2f}"
            )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal")

        path = frame_dir / f"frame_{k:04d}.png"
        fig.savefig(path, dpi=150)
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
    p.add_argument("--field", choices=FIELDS, default="re_mag",
                   help="Field to animate, matching the static panels of "
                        "rayleigh_cloak.plot.plot_field_panels: re_mag = "
                        "|Re(u(t))| (default), re_ux / re_uy = the signed "
                        "components on a symmetric colour scale")
    case_grp = p.add_mutually_exclusive_group()
    case_grp.add_argument("--ideal", action="store_true",
                          help="Animate the ideal cloak (analytic "
                               "transformation C_eff/rho_eff) instead of the "
                               "optimised cells; needs no optimized_params.npz")
    case_grp.add_argument("--reference", action="store_true",
                          help="Animate the reference field: homogeneous "
                               "half-space, no cloak and no defect cut-out; "
                               "needs no optimized_params.npz")
    case_grp.add_argument("--morph", action="store_true",
                          help="Animate the coordinate transformation itself: "
                               "solve the reference field, then carry it "
                               "through phi (the integral of F_tensor) as the "
                               "void opens from nothing to the full cloak. "
                               "The wave keeps cycling throughout and stays "
                               "exact at every step — no re-solve.")
    p.add_argument("--morph-periods", type=float, default=4.0,
                   help="Wave periods spanned by the morph (default: 4)")
    p.add_argument("--hold-periods", type=float, default=1.0,
                   help="Wave periods held at each end, reference before and "
                        "ideal cloak after (default: 1)")
    p.add_argument("--zoom", type=float, default=None,
                   help="Zoom on the cloak: viewport is the cloak bounding "
                        "box (half-width c, depth b at the top centre) scaled "
                        "by this factor. Omit for the full domain. E.g. 2.5")
    p.add_argument("--clim", type=float, default=None,
                   help="Force the colour-scale maximum instead of taking it "
                        "from this solve's percentile. Pass the same value to "
                        "two runs (the script prints the auto value it used) "
                        "so their videos share a scale and can be compared "
                        "frame to frame.")
    p.add_argument("--zoom-depth", type=float, default=None,
                   help="Scale the viewport depth separately from its width "
                        "(default: same as --zoom). Smaller values crop the "
                        "quiet deep field the Rayleigh wave never reaches.")
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
    if args.ideal:
        case, case_label = "ideal", "Ideal cloak"
    elif args.reference:
        case, case_label = "reference", "Reference (no cloak)"
    elif args.morph:
        # The morph animates the reference field carried through phi.
        case, case_label = "reference", "Reference → ideal cloak via F"
    else:
        case, case_label = "optimized", "Optimised cloak"

    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    if case == "optimized" and not params_path.exists():
        raise SystemExit(f"params not found: {params_path}")

    cfg = load_config(config_path)
    if args.f_star is not None:
        cfg = cfg.model_copy(update={
            "domain": cfg.domain.model_copy(update={"f_star": float(args.f_star)})
        })
    print(f"=== {case_label}: forward solve in {out_dir} "
          f"(f*={cfg.domain.f_star:.3f}) ===")
    u, cloak_mesh, _geometry, dp = forward_solve(
        cfg, params_path if case == "optimized" else None, case=case,
    )

    # Morph length is set by its schedule, not --n-periods.
    n_periods = args.n_periods
    if args.morph:
        n_periods = 2.0 * args.hold_periods + args.morph_periods

    f_tag = f"f{cfg.domain.f_star:.2f}"
    tag_case = "morph" if args.morph else case
    if tag_case != "optimized":
        f_tag = f"{tag_case}_{f_tag}"
    if args.field != "re_mag":
        f_tag = f"{f_tag}_{args.field}"
    if args.zoom is not None:
        f_tag = f"{f_tag}_zoom{args.zoom:g}"
        if args.zoom_depth is not None:
            f_tag = f"{f_tag}d{args.zoom_depth:g}"
    frame_dir = out_dir / f"_frames_{f_tag}"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)

    print("=== Rendering frames ===")
    n = render_frames(
        u, cloak_mesh, dp,
        n_periods=n_periods,
        n_frames_per_period=args.frames_per_period,
        frame_dir=frame_dir,
        f_star=cfg.domain.f_star,
        norm_type=args.norm_type,
        zoom=args.zoom,
        zoom_depth=args.zoom_depth,
        field=args.field,
        clim=args.clim,
        morph=args.morph,
        morph_periods=args.morph_periods,
        hold_periods=args.hold_periods,
        label=case_label,
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
