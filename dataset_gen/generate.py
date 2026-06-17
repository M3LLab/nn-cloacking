"""Core dataset generation: random samples + optimization trajectories.

Generates (cell_params, f_star) → transmission_loss data for training a
surrogate model.  The geometry (mesh, cell decomposition, cloak_mask) is
fixed; only cell material parameters and frequency vary.

Two sampling strategies:
  1. **Random**: perturb the push-forward initialisation with controlled noise.
  2. **Neural optimization trajectory**: run a neural-reparameterized
     optimisation (MLP maps cell coords → material params) and snapshot the
     *decoded* cell parameters at regular intervals.  These lie near local
     minima and are the most informative samples.  The neural method
     converges reliably (unlike direct cell-based Adam which has near-zero
     gradients and produces flat trajectories).

Output: HDF5 file with datasets:
  - cell_C_flat   (N, n_cells, n_C_params)   stiffness params per cell
  - cell_rho      (N, n_cells)                density per cell
  - f_star        (N,)                        normalised frequency
  - loss          (N,)                        transmission loss
  - sample_type   (N,)                        "random" or "opt_step_XXX"
  - cloak_mask    (n_cells,)                  shared boolean mask
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax_fem.solver import ad_wrapper, solver as jax_fem_solver

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.loss import resolve_loss_target, transmission_loss
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.neural_reparam import make_neural_reparam, run_optimization_neural
from rayleigh_cloak.optimize import (
    adam_init,
    adam_update,
    get_top_surface_beyond_cloak_indices,
    total_loss,
)
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry, solve_reference


# ---------------------------------------------------------------------------
# Infrastructure shared across all frequencies
# ---------------------------------------------------------------------------


@dataclass
class FixedGeometryContext:
    """Everything that stays constant when only f_star changes."""

    base_config: SimulationConfig
    geometry: object
    full_mesh: object
    cloak_mesh: object
    kept_nodes: np.ndarray
    cell_decomp: CellDecomposition
    cloak_mask: np.ndarray          # (n_cells,) bool
    params_init: tuple              # (cell_C_flat, cell_rho) from push-forward
    C0: jnp.ndarray                 # background isotropic C
    rho0: float
    solver_opts: dict


def build_fixed_context(config: SimulationConfig) -> FixedGeometryContext:
    """Build mesh, cell decomposition, and initial params once."""
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Building mesh (shared across all frequencies) ===")
    full_mesh = generate_mesh_full(config, params, geometry)
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Full: {len(full_mesh.points)} nodes | "
          f"Sub: {len(cloak_mesh.points)} nodes")

    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
    )
    print(f"  Cells: {cell_decomp.n_cells} total, "
          f"{cell_decomp.n_cloak_cells} in cloak")

    petsc_opts = {
        "ksp_type": config.solver.ksp_type,
        "pc_type": config.solver.pc_type,
    }
    if config.solver.pc_factor_mat_solver_type:
        petsc_opts["pc_factor_mat_solver_type"] = config.solver.pc_factor_mat_solver_type
    solver_opts = {"petsc_solver": petsc_opts}

    return FixedGeometryContext(
        base_config=config,
        geometry=geometry,
        full_mesh=full_mesh,
        cloak_mesh=cloak_mesh,
        kept_nodes=kept_nodes,
        cell_decomp=cell_decomp,
        cloak_mask=cell_decomp.cloak_mask,
        params_init=cell_mat.get_initial_params(),
        C0=C0,
        rho0=params.rho0,
        solver_opts=solver_opts,
    )


# ---------------------------------------------------------------------------
# Per-frequency problem setup
# ---------------------------------------------------------------------------


@dataclass
class FreqContext:
    """FEM problem + loss setup at a single frequency."""

    f_star: float
    dp: DerivedParams
    problem: object
    fwd_pred: object            # ad_wrapper
    surface_indices: np.ndarray
    u_ref_surface: jnp.ndarray  # reference u at surface nodes
    loss_fn: object             # JAX-traceable loss


def _make_config_at_fstar(base: SimulationConfig, f_star: float) -> SimulationConfig:
    return base.model_copy(
        update={"domain": base.domain.model_copy(update={"f_star": float(f_star)})}
    )


def build_freq_context(
    ctx: FixedGeometryContext,
    f_star: float,
) -> FreqContext:
    """Build FEM problem, reference solution, and loss at one frequency."""
    config_f = _make_config_at_fstar(ctx.base_config, f_star)
    dp_f = DerivedParams.from_config(config_f)

    # Reference solve
    ref_result = solve_reference(config_f, mesh=ctx.full_mesh)

    # Build problem with cell decomposition
    problem = build_problem(
        ctx.cloak_mesh, config_f, dp_f, ctx.geometry, ctx.cell_decomp,
    )
    fwd_pred = ad_wrapper(problem, ctx.solver_opts, ctx.solver_opts)

    # Loss target
    indices, u_ref_at_nodes, loss_fn = resolve_loss_target(
        ctx.base_config.loss.type,
        np.asarray(ctx.cloak_mesh.points),
        ctx.geometry, dp_f, ctx.kept_nodes, ref_result.u,
        loss_cfg=ctx.base_config.loss,
    )

    return FreqContext(
        f_star=f_star,
        dp=dp_f,
        problem=problem,
        fwd_pred=fwd_pred,
        surface_indices=indices,
        u_ref_surface=u_ref_at_nodes,
        loss_fn=loss_fn,
    )


# ---------------------------------------------------------------------------
# Evaluate loss for given params
# ---------------------------------------------------------------------------


def evaluate_loss(
    fctx: FreqContext,
    params: tuple[jnp.ndarray, jnp.ndarray],
) -> float:
    """Forward solve + transmission loss for one (params, freq) pair."""
    sol_list = fctx.fwd_pred(params)
    u = sol_list[0]
    loss_val = fctx.loss_fn(u, fctx.u_ref_surface, jnp.array(fctx.surface_indices))
    return float(loss_val)


# ---------------------------------------------------------------------------
# Random sampling
# ---------------------------------------------------------------------------


def generate_random_samples(
    ctx: FixedGeometryContext,
    fctx: FreqContext,
    n_samples: int,
    noise_scales: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate random perturbations of the push-forward init and evaluate loss.

    Parameters
    ----------
    noise_scales : list of relative noise magnitudes to cycle through.
        Default: [0.01, 0.05, 0.1, 0.2, 0.5] (1% to 50% of init magnitude).
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if noise_scales is None:
        noise_scales = [0.01, 0.05, 0.1, 0.2, 0.5]

    C_init, rho_init = ctx.params_init
    C_scale = float(jnp.std(C_init[ctx.cloak_mask]))
    rho_scale = float(jnp.std(rho_init[ctx.cloak_mask]))
    # Fallback: use mean if std is tiny (e.g. nearly uniform init)
    if C_scale < 1e-10:
        C_scale = float(jnp.mean(jnp.abs(C_init))) * 0.1
    if rho_scale < 1e-10:
        rho_scale = float(jnp.mean(jnp.abs(rho_init))) * 0.1

    samples = []
    for i in range(n_samples):
        scale = noise_scales[i % len(noise_scales)]

        # Perturb only cloak cells; background stays at C0/rho0
        C_noise = jnp.array(rng.normal(size=C_init.shape).astype(np.float64))
        rho_noise = jnp.array(rng.normal(size=rho_init.shape).astype(np.float64))

        mask_2d = jnp.array(ctx.cloak_mask[:, None], dtype=C_init.dtype)
        mask_1d = jnp.array(ctx.cloak_mask, dtype=rho_init.dtype)

        cell_C = C_init + scale * C_scale * C_noise * mask_2d
        cell_rho = rho_init + scale * rho_scale * rho_noise * mask_1d
        # Ensure positive density
        cell_rho = jnp.maximum(cell_rho, 100.0)

        params = (cell_C, cell_rho)

        t0 = time.time()
        loss = evaluate_loss(fctx, params)
        dt = time.time() - t0
        print(f"  random [{i+1}/{n_samples}] f*={fctx.f_star:.2f} "
              f"scale={scale:.2f} loss={loss:.4e} ({dt:.1f}s)")

        samples.append({
            "cell_C_flat": np.asarray(cell_C),
            "cell_rho": np.asarray(cell_rho),
            "f_star": fctx.f_star,
            "loss": loss,
            "sample_type": f"random_s{scale:.2f}",
        })

    return samples


# ---------------------------------------------------------------------------
# Spatially-smooth random fields (low-frequency sinusoidal + noise)
# ---------------------------------------------------------------------------


def generate_smooth_random_samples(
    ctx: FixedGeometryContext,
    fctx: FreqContext,
    n_samples: int,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Generate spatially-smooth random material fields and evaluate loss.

    Each sample draws base values near typical cement/rock ranges, then
    overlays low-frequency sinusoidal modulation (<2 periods across the
    cloak) plus small iid noise.  Only cloak cells are modulated;
    background stays at C0/rho0.

    The result is physically plausible spatial variation — not just
    white-noise perturbations of the push-forward init.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    centers = ctx.cell_decomp.cell_centers  # (n_cells, 2)
    mask = ctx.cloak_mask                   # (n_cells,) bool
    n_cells = ctx.cell_decomp.n_cells
    n_C_params = ctx.base_config.cells.n_C_params

    # Bounding box of cloak (for normalising coordinates to [0,1])
    cd = ctx.cell_decomp
    Lx = cd.x_max - cd.x_min
    Ly = cd.y_max - cd.y_min

    # Normalised coordinates in [0, 1]
    xn = (centers[:, 0] - cd.x_min) / Lx  # (n_cells,)
    yn = (centers[:, 1] - cd.y_min) / Ly

    # Background values (from push-forward init at non-cloak cells)
    C_bg = ctx.params_init[0][0]   # first cell (background), shape (n_C_params,)
    rho_bg = float(ctx.params_init[1][0])

    # Typical material ranges for random base values
    # For n_C_params=2: [λ, μ].  Use cement-like ranges as centre.
    lam_bg, mu_bg = float(C_bg[0]), float(C_bg[1])

    samples = []
    for i in range(n_samples):
        # --- Random base values (uniform within a range) ---
        # Draw base lam, mu, rho from broad distributions
        lam_base = rng.uniform(lam_bg * 0.3, lam_bg * 3.0)
        mu_base = rng.uniform(mu_bg * 0.3, mu_bg * 3.0)
        rho_base = rng.uniform(rho_bg * 0.5, rho_bg * 2.5)

        # --- Low-frequency sinusoidal spatial modulation ---
        # Random wave vector: 0.5-2 periods across the cloak
        n_modes = rng.integers(1, 4)  # 1-3 superimposed modes
        lam_field = np.full(n_cells, lam_base)
        mu_field = np.full(n_cells, mu_base)
        rho_field = np.full(n_cells, rho_base)

        for _ in range(n_modes):
            # Random spatial frequency: 0.5 to 2.0 full periods
            kx = rng.uniform(0.5, 2.0) * 2 * np.pi
            ky = rng.uniform(0.5, 2.0) * 2 * np.pi
            phase_lam = rng.uniform(0, 2 * np.pi)
            phase_mu = rng.uniform(0, 2 * np.pi)
            phase_rho = rng.uniform(0, 2 * np.pi)

            # Amplitude: 10-40% of base value
            amp_frac = rng.uniform(0.1, 0.4)

            lam_field += amp_frac * lam_base * np.sin(kx * xn + ky * yn + phase_lam)
            mu_field += amp_frac * mu_base * np.cos(kx * xn + ky * yn + phase_mu)
            rho_field += amp_frac * rho_base * np.sin(kx * xn + ky * yn + phase_rho)

        # --- Add small iid noise on top (5% of base) ---
        lam_field += rng.normal(0, 0.05 * abs(lam_base), n_cells)
        mu_field += rng.normal(0, 0.05 * abs(mu_base), n_cells)
        rho_field += rng.normal(0, 0.05 * rho_base, n_cells)

        # Ensure physical: positive μ and ρ
        mu_field = np.maximum(mu_field, mu_bg * 0.05)
        rho_field = np.maximum(rho_field, 100.0)

        # --- Assemble cell params (only cloak cells get the random field) ---
        cell_C = np.asarray(ctx.params_init[0]).copy()   # start from background
        cell_rho = np.asarray(ctx.params_init[1]).copy()

        if n_C_params == 2:
            cell_C[mask, 0] = lam_field[mask]
            cell_C[mask, 1] = mu_field[mask]
        else:
            # For higher-param models: scale all C params proportionally
            # relative to the isotropic (lam, mu) base
            scale_lam = lam_field / (lam_bg + 1e-30)
            scale_mu = mu_field / (mu_bg + 1e-30)
            scale_avg = 0.5 * (scale_lam + scale_mu)
            for ci in range(n_C_params):
                cell_C[mask, ci] = cell_C[mask, ci] * scale_avg[mask]

        cell_rho[mask] = rho_field[mask]

        params = (jnp.array(cell_C), jnp.array(cell_rho))

        t0 = time.time()
        loss = evaluate_loss(fctx, params)
        dt = time.time() - t0
        print(f"  smooth [{i+1}/{n_samples}] f*={fctx.f_star:.2f} "
              f"rho_base={rho_base:.0f} loss={loss:.4e} ({dt:.1f}s)")

        samples.append({
            "cell_C_flat": np.asarray(cell_C),
            "cell_rho": np.asarray(cell_rho),
            "f_star": fctx.f_star,
            "loss": loss,
            "sample_type": f"smooth_{n_modes}modes",
        })

    return samples


# ---------------------------------------------------------------------------
# Neural optimization trajectory sampling
# ---------------------------------------------------------------------------


def generate_opt_trajectory_neural(
    ctx: FixedGeometryContext,
    fctx: FreqContext,
    n_iters: int = 200,
    lr: float = 0.005,
    lr_end: float = 1e-6,
    lr_schedule: str = "cosine",
    lambda_l2: float = 0.0,
    snapshot_every: int = 5,
    hidden_size: int = 512,
    n_layers: int = 6,
    n_fourier: int = 64,
    seed: int = 42,
    output_scale: float = 0.1,
    theta_init: list | None = None,
    patience: int = 30,
    patience_min_delta: float = 1e-3,
) -> tuple[list[dict], list]:
    """Run neural-reparameterized optimisation and snapshot decoded cell params.

    An MLP maps cell-centre coordinates → material corrections on top of the
    push-forward initialisation.  Gradients flow through the FEM adjoint and
    then back through the network via JAX autodiff.  This converges reliably
    (several orders of magnitude per 100 steps), unlike direct cell-based
    Adam which produces effectively flat trajectories.

    Early stopping: if the best loss seen so far does not improve by at least
    ``patience_min_delta`` (relative) over ``patience`` consecutive steps,
    optimisation stops early.  This is frequency-agnostic — easy frequencies
    stop quickly, hard ones run longer.

    Parameters
    ----------
    theta_init : optional MLP weights for warm-starting from a previous
        frequency.  If None, weights are re-initialised from scratch.
    patience : number of consecutive non-improving steps before stopping.
        Set to 0 to disable early stopping.
    patience_min_delta : minimum relative improvement in best loss required
        to reset the patience counter.  E.g. 1e-3 means 0.1% improvement.

    Returns
    -------
    samples : list of dicts with decoded cell params + loss at each snapshot
    final_theta : MLP weights after the last update (for warm-starting the
        next frequency)
    """
    import math

    theta, reparam = make_neural_reparam(
        ctx.cell_decomp,
        ctx.params_init,
        hidden_size=hidden_size,
        n_layers=n_layers,
        n_fourier=n_fourier,
        seed=seed,
        output_scale=output_scale,
    )
    if theta_init is not None:
        theta = jax.tree.map(jnp.copy, theta_init)

    opt_state = adam_init(theta)
    boundary_indices_jnp = jnp.array(fctx.surface_indices)

    def _loss_fn(theta_):
        params = reparam.decode(theta_)
        sol_list = fctx.fwd_pred(params)
        u_cloak = sol_list[0]
        L_cloak = fctx.loss_fn(u_cloak, fctx.u_ref_surface, boundary_indices_jnp)
        if lambda_l2 > 0.0:
            from rayleigh_cloak.optimize import l2_regularization
            L_cloak = L_cloak + lambda_l2 * l2_regularization(params, ctx.params_init)
        return L_cloak

    loss_and_grad = jax.value_and_grad(_loss_fn)

    samples = []
    best_loss = float("inf")
    no_improve_count = 0

    def _snapshot(step, params, cloak_loss):
        cell_C, cell_rho = params
        samples.append({
            "cell_C_flat": np.asarray(cell_C),
            "cell_rho": np.asarray(cell_rho),
            "f_star": fctx.f_star,
            "loss": float(cloak_loss),
            "sample_type": f"opt_step_{step:04d}",
        })

    for step in range(n_iters):
        # Learning rate schedule
        t_frac = step / max(n_iters - 1, 1)
        if lr_schedule == "cosine":
            cur_lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * t_frac))
        elif lr_schedule == "linear":
            cur_lr = lr + (lr_end - lr) * t_frac
        else:
            cur_lr = lr

        t0 = time.time()
        loss_val, grads = loss_and_grad(theta)
        dt = time.time() - t0

        cloak_loss = float(loss_val)
        params = reparam.decode(theta)

        if step % snapshot_every == 0:
            _snapshot(step, params, cloak_loss)
            if step % max(10, snapshot_every) == 0:
                print(f"  neural_opt [{step}/{n_iters}] f*={fctx.f_star:.2f} "
                      f"loss={cloak_loss:.4e} lr={cur_lr:.2e} ({dt:.1f}s) [snapshot]")

        # Early stopping check
        if patience > 0:
            if cloak_loss < best_loss * (1.0 - patience_min_delta):
                best_loss = cloak_loss
                no_improve_count = 0
            else:
                no_improve_count += 1
            if no_improve_count >= patience:
                print(f"  neural_opt [early stop at {step}] f*={fctx.f_star:.2f} "
                      f"loss={cloak_loss:.4e} (no >{patience_min_delta*100:.1f}% "
                      f"improvement in {patience} steps)")
                updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
                theta = jax.tree.map(lambda p, u: p + u, theta, updates)
                break
        else:
            best_loss = min(best_loss, cloak_loss)

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    # Final snapshot after last update
    final_params = reparam.decode(theta)
    final_loss = evaluate_loss(fctx, final_params)
    _snapshot(step + 1, final_params, final_loss)
    print(f"  neural_opt [done at step {step+1}] f*={fctx.f_star:.2f} "
          f"final_loss={final_loss:.4e}")

    return samples, theta


# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------


def init_hdf5(path: Path, n_cells: int, n_C_params: int, cloak_mask: np.ndarray):
    """Create HDF5 file with resizable datasets."""
    with h5py.File(path, "w") as f:
        f.create_dataset("cloak_mask", data=cloak_mask.astype(np.bool_))
        f.attrs["n_cells"] = n_cells
        f.attrs["n_C_params"] = n_C_params

        maxshape_C = (None, n_cells, n_C_params)
        maxshape_rho = (None, n_cells)
        maxshape_scalar = (None,)

        f.create_dataset("cell_C_flat", shape=(0, n_cells, n_C_params),
                         maxshape=maxshape_C, dtype=np.float64, chunks=True)
        f.create_dataset("cell_rho", shape=(0, n_cells),
                         maxshape=maxshape_rho, dtype=np.float64, chunks=True)
        f.create_dataset("f_star", shape=(0,),
                         maxshape=maxshape_scalar, dtype=np.float64, chunks=True)
        f.create_dataset("loss", shape=(0,),
                         maxshape=maxshape_scalar, dtype=np.float64, chunks=True)

        dt = h5py.string_dtype()
        f.create_dataset("sample_type", shape=(0,),
                         maxshape=maxshape_scalar, dtype=dt, chunks=True)

    print(f"Initialised HDF5: {path}")


def append_samples(path: Path, samples: list[dict]):
    """Append a batch of samples to the HDF5 file."""
    if not samples:
        return

    n_new = len(samples)
    with h5py.File(path, "a") as f:
        n_old = f["loss"].shape[0]
        n_total = n_old + n_new

        for key in ("cell_C_flat", "cell_rho", "f_star", "loss", "sample_type"):
            f[key].resize(n_total, axis=0)

        for i, s in enumerate(samples):
            idx = n_old + i
            f["cell_C_flat"][idx] = s["cell_C_flat"]
            f["cell_rho"][idx] = s["cell_rho"]
            f["f_star"][idx] = s["f_star"]
            f["loss"][idx] = s["loss"]
            f["sample_type"][idx] = s["sample_type"]

    print(f"  Appended {n_new} samples → {n_total} total in {path.name}")


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------


@dataclass
class DatasetGenConfig:
    """Controls for the dataset generation run."""

    # Frequency grid (default: 79 freqs with narrow 0.05 steps)
    f_stars: list[float] = field(default_factory=lambda: [
        round(0.1 + i * 0.05, 4) for i in range(79)
    ])

    # Random perturbation samples per frequency
    n_random_per_freq: int = 25
    noise_scales: list[float] = field(
        default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.5]
    )

    # Spatially-smooth random fields per frequency
    n_smooth_per_freq: int = 25

    # Neural optimization trajectory per frequency.
    # The neural reparameterization (MLP maps cell coords → material
    # corrections) converges reliably — direct cell-based Adam produces
    # essentially flat trajectories due to very small per-cell gradients.
    opt_n_iters: int = 200
    opt_lr: float = 0.005
    opt_lr_end: float = 1e-6
    opt_lr_schedule: str = "cosine"
    opt_lambda_l2: float = 0.0
    opt_snapshot_every: int = 5

    # MLP architecture (should match the configs used in actual runs)
    opt_neural_hidden_size: int = 512
    opt_neural_n_layers: int = 6
    opt_neural_n_fourier: int = 64
    opt_neural_seed: int = 42
    opt_neural_output_scale: float = 0.1

    # Warm-start: pass the final MLP weights from frequency f as the
    # initialisation for f+1.  Speeds up convergence at each frequency
    # but produces correlated trajectories (less diverse samples).
    opt_warm_start: bool = False

    # Early stopping: halt when best loss hasn't improved by
    # opt_patience_min_delta (relative) in opt_patience consecutive steps.
    # Set opt_patience=0 to disable.
    opt_patience: int = 30
    opt_patience_min_delta: float = 4e-4

    # Which frequencies to run optimization on (subset of f_stars for cost)
    opt_f_stars: list[float] | None = None  # None → use all f_stars

    seed: int = 42
    output_path: str = "output/surrogate_dataset.h5"


def run_dataset_generation(
    base_config: SimulationConfig,
    gen_config: DatasetGenConfig | None = None,
):
    """Generate the full surrogate dataset.

    Parameters
    ----------
    base_config : SimulationConfig
        Base config (geometry, mesh, cells must be configured).
        f_star in this config is ignored; gen_config.f_stars is used.
    gen_config : DatasetGenConfig
        Controls frequencies, sample counts, etc.
    """
    if gen_config is None:
        gen_config = DatasetGenConfig()

    out_path = Path(gen_config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(gen_config.seed)

    # 1. Build fixed context (mesh, cells, init params)
    ctx = build_fixed_context(base_config)
    n_cells = ctx.cell_decomp.n_cells
    n_C_params = base_config.cells.n_C_params

    # 2. Init HDF5
    init_hdf5(out_path, n_cells, n_C_params, ctx.cloak_mask)

    # Also save the push-forward init as the first sample at each freq
    # (loss=0 is not meaningful; we'll evaluate it properly)

    opt_freqs = set(gen_config.opt_f_stars or gen_config.f_stars)
    # Keep opt_freqs in the same order as f_stars for warm-start to work
    opt_freqs_ordered = [f for f in gen_config.f_stars if f in opt_freqs]

    total_random = len(gen_config.f_stars) * gen_config.n_random_per_freq
    total_smooth = len(gen_config.f_stars) * gen_config.n_smooth_per_freq
    snapshots_per_freq = gen_config.opt_n_iters // gen_config.opt_snapshot_every + 1
    total_opt = len(opt_freqs) * snapshots_per_freq
    print(f"\n=== Dataset generation plan ===")
    print(f"  Frequencies: {len(gen_config.f_stars)} "
          f"({gen_config.f_stars[0]:.1f} to {gen_config.f_stars[-1]:.1f})")
    print(f"  Random perturbation samples: {total_random} "
          f"({gen_config.n_random_per_freq}/freq)")
    print(f"  Smooth random field samples: {total_smooth} "
          f"({gen_config.n_smooth_per_freq}/freq)")
    print(f"  Neural opt trajectory samples: ~{total_opt} "
          f"({len(opt_freqs)} freqs × ~{snapshots_per_freq} snapshots)")
    print(f"  Neural opt: {gen_config.opt_n_iters} iters, "
          f"lr {gen_config.opt_lr}→{gen_config.opt_lr_end} ({gen_config.opt_lr_schedule}), "
          f"MLP {gen_config.opt_neural_n_layers}×{gen_config.opt_neural_hidden_size}, "
          f"warm_start={gen_config.opt_warm_start}")
    print(f"  Estimated total: ~{total_random + total_smooth + total_opt}")
    print(f"  Output: {out_path}\n")

    prev_theta: list | None = None  # for warm-starting across frequencies

    # 3. Loop over frequencies
    for fi, f_star in enumerate(gen_config.f_stars):
        print(f"\n{'='*60}")
        print(f"  Frequency {fi+1}/{len(gen_config.f_stars)}: f* = {f_star:.2f}")
        print(f"{'='*60}")

        fctx = build_freq_context(ctx, f_star)

        # 3a. Evaluate push-forward init
        t0 = time.time()
        init_loss = evaluate_loss(fctx, ctx.params_init)
        print(f"  Init loss at f*={f_star:.2f}: {init_loss:.4e} "
              f"({time.time()-t0:.1f}s)")
        init_sample = [{
            "cell_C_flat": np.asarray(ctx.params_init[0]),
            "cell_rho": np.asarray(ctx.params_init[1]),
            "f_star": f_star,
            "loss": init_loss,
            "sample_type": "init",
        }]
        append_samples(out_path, init_sample)

        # 3b. Random samples
        print(f"\n--- Random samples at f*={f_star:.2f} ---")
        random_samples = generate_random_samples(
            ctx, fctx,
            n_samples=gen_config.n_random_per_freq,
            noise_scales=gen_config.noise_scales,
            rng=rng,
        )
        append_samples(out_path, random_samples)

        # 3c. Smooth random fields
        if gen_config.n_smooth_per_freq > 0:
            print(f"\n--- Smooth random fields at f*={f_star:.2f} ---")
            smooth_samples = generate_smooth_random_samples(
                ctx, fctx,
                n_samples=gen_config.n_smooth_per_freq,
                rng=rng,
            )
            append_samples(out_path, smooth_samples)

        # 3d. Neural optimization trajectory
        if f_star in opt_freqs:
            print(f"\n--- Neural opt trajectory at f*={f_star:.2f} ---")
            warm_theta = prev_theta if gen_config.opt_warm_start else None
            opt_samples, final_theta = generate_opt_trajectory_neural(
                ctx, fctx,
                n_iters=gen_config.opt_n_iters,
                lr=gen_config.opt_lr,
                lr_end=gen_config.opt_lr_end,
                lr_schedule=gen_config.opt_lr_schedule,
                lambda_l2=gen_config.opt_lambda_l2,
                snapshot_every=gen_config.opt_snapshot_every,
                hidden_size=gen_config.opt_neural_hidden_size,
                n_layers=gen_config.opt_neural_n_layers,
                n_fourier=gen_config.opt_neural_n_fourier,
                seed=gen_config.opt_neural_seed,
                output_scale=gen_config.opt_neural_output_scale,
                theta_init=warm_theta,
                patience=gen_config.opt_patience,
                patience_min_delta=gen_config.opt_patience_min_delta,
            )
            prev_theta = final_theta
            append_samples(out_path, opt_samples)

    # Summary
    with h5py.File(out_path, "r") as f:
        n_total = f["loss"].shape[0]
        losses = f["loss"][:]
    print(f"\n{'='*60}")
    print(f"  Dataset complete: {n_total} samples")
    print(f"  Loss range: [{losses.min():.4e}, {losses.max():.4e}]")
    print(f"  Output: {out_path}")
    print(f"{'='*60}")
