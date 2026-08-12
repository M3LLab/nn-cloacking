"""High-level solve interface.

``solve(config)`` runs a full forward simulation and returns a
``SolutionResult``.  Safe to call repeatedly (e.g. inside an optimisation
loop).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper, solver as jax_fem_solver

from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams, SimulationConfig
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_iso, CellMaterial
from rayleigh_cloak import mesh as _legacy_mesh
from rayleigh_cloak import mesh_uniform as _uniform_mesh
from rayleigh_cloak import mesh_cloak_uniform as _cloak_uniform_mesh
from rayleigh_cloak.mesh import extract_submesh
from rayleigh_cloak.optimize import (
    OptimizationResult,
    cloaking_distortion_percent,
    get_all_physical_boundary_indices,
    get_outside_cloak_indices,
    get_right_boundary_indices,
    get_top_surface_beyond_cloak_indices,
    run_optimization,
)
from rayleigh_cloak.material_prior import GMMPrior, load_gmm_prior
from rayleigh_cloak.neural_reparam import (
    FreqTarget,
    NeuralOptimizationResult,
    load_theta,
    make_neural_reparam,
    run_optimization_neural,
    run_optimization_neural_multifreq,
    run_optimization_neural_multifreq_minimax,
)
from rayleigh_cloak.loss import resolve_loss_target
from rayleigh_cloak.neural_reparam_topo import (
    TopoOptimizationResult,
    make_neural_reparam_topo,
    run_optimization_neural_topo,
)
from rayleigh_cloak.problem import build_problem


@dataclass
class SolutionResult:
    """Container for a completed simulation."""

    u: np.ndarray               # (num_nodes, 4)
    mesh: Mesh
    config: SimulationConfig
    params: DerivedParams
    full_mesh: Mesh | None = None       # full mesh before defect extraction
    kept_nodes: np.ndarray | None = None  # mapping submesh→full mesh nodes


def _create_geometry(cfg: SimulationConfig, params: DerivedParams):
    """Instantiate the geometry object specified by ``cfg.geometry_type``."""
    if cfg.geometry_type == "triangular":
        return TriangularCloakGeometry.from_params(params)
    if cfg.geometry_type == "circular":
        from rayleigh_cloak.geometry.circular import CircularCloakGeometry
        return CircularCloakGeometry(
            ri=params.ri, rc=params.rc,
            x_c=params.x_c, y_c=params.y_c,
            y_top=params.y_top,
        )
    raise ValueError(f"Unknown geometry_type: {cfg.geometry_type!r}")


def _petsc_opts(config: SimulationConfig) -> dict:
    """Build PETSc solver options dict from config."""
    opts = {
        "ksp_type": config.solver.ksp_type,
        "pc_type": config.solver.pc_type,
    }
    if config.solver.pc_factor_mat_solver_type:
        opts["pc_factor_mat_solver_type"] = config.solver.pc_factor_mat_solver_type
    return {"petsc_solver": opts}


def _mesh_module(config: SimulationConfig):
    """Select the mesh builder module from ``config.mesh.builder``.

    ``"uniform_tri6"`` -> :mod:`rayleigh_cloak.mesh_uniform` (uniform over the
    cloak *bounding box*); ``"uniform_cloak"`` ->
    :mod:`rayleigh_cloak.mesh_cloak_uniform` (uniform over the cloak *annulus*
    only, outside unchanged — the builder to use for pixel-level runs); anything
    else -> the legacy graded builder :mod:`rayleigh_cloak.mesh`. Default is
    legacy, so existing runs are unchanged.
    """
    if config.mesh.builder == "uniform_tri6":
        return _uniform_mesh
    if config.mesh.builder == "uniform_cloak":
        return _cloak_uniform_mesh
    return _legacy_mesh


def _single_mesh(config, params, geometry) -> Mesh:
    """Dispatch :func:`generate_mesh` (with-cutout / reference) on the builder."""
    return _mesh_module(config).generate_mesh(config, params, geometry)


def _full_mesh(config, params, geometry) -> Mesh:
    """Dispatch :func:`generate_mesh_full` (no cutout) on the builder."""
    return _mesh_module(config).generate_mesh_full(config, params, geometry)


def solve(
    config: SimulationConfig,
    mesh: Mesh | None = None,
) -> SolutionResult:
    """Run a full forward simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete specification of the problem.
    mesh : Mesh, optional
        Pre-built mesh.  If *None*, a mesh is generated from the config.

    Returns
    -------
    SolutionResult
        Solution array, mesh, config, and derived parameters.
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    if mesh is None:
        mesh = _single_mesh(config, params, geometry)
    problem = build_problem(mesh, config, params, geometry)

    solver_opts = _petsc_opts(config)

    print("Solving frequency-domain system with absorbing layers ...")
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u = sol_list[0]

    return SolutionResult(
        u=np.asarray(u),
        mesh=mesh,
        config=config,
        params=params,
    )


def solve_reference(
    config: SimulationConfig,
    mesh: Mesh | None = None,
) -> SolutionResult:
    """Convenience: solve the reference problem (no cloak)."""
    ref_config = config.model_copy(update={"is_reference": True})
    return solve(ref_config, mesh=mesh)


def solve_cell_based(config: SimulationConfig) -> SolutionResult:
    """Forward solve with piecewise-constant cell materials (no optimisation).

    Uses the initial cell parameters derived from the continuous
    transformational push-forward, evaluated at cell centres.
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Generating shared mesh ===")
    full_mesh = _full_mesh(config, params, geometry)

    print("=== Extracting submesh (removing defect) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Setting up cell decomposition ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
        init=config.cells.init,
        init_path=config.cells.init_path,
    )
    print(f"  {cell_decomp.n_cells} total cells, "
          f"{cell_decomp.n_cloak_cells} in cloak")

    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)
    problem.set_params(cell_mat.get_initial_params())

    solver_opts = _petsc_opts(config)
    print("Solving cell-based system ...")
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u = np.asarray(sol_list[0])

    return SolutionResult(u=u, mesh=cloak_mesh, config=config, params=params,
                          full_mesh=full_mesh, kept_nodes=kept_nodes)


def solve_optimization(config: SimulationConfig, step_callback=None) -> OptimizationResult:
    """Run cell-based material optimisation.

    Steps:
    1. Generate a single full-domain mesh (no defect cutout).
    2. Solve the reference problem on the full mesh.
    3. Extract submesh (defect elements removed) for the cloak solve.
    4. Build cell decomposition and FEM problem on the submesh.
    5. Optimise cell materials — boundary nodes match exactly (no interpolation).
    """
    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = _full_mesh(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    from rayleigh_cloak.plot import plot_reference_field
    plot_reference_field(
        ref_result,
        save_path=f"{config.output_dir}/reference_field.png",
    )

    print("=== Step 3: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements "
          f"({len(full_mesh.cells) - len(cloak_mesh.cells)} defect elements removed, "
          f"{len(full_mesh.points) - len(kept_nodes)} orphan nodes removed)")

    print("=== Step 4: Setting up cell decomposition ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
        init=config.cells.init,
        init_path=config.cells.init_path,
    )

    print(f"  {cell_decomp.n_cells} total cells, "
          f"{cell_decomp.n_cloak_cells} in cloak")

    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)

    # Loss target nodes from config
    boundary_indices, u_ref_boundary, loss_fn = resolve_loss_target(
        config.loss.type, np.asarray(cloak_mesh.points), geometry, params,
        kept_nodes, ref_result.u, loss_cfg=config.loss,
    )
    print(f"  {len(boundary_indices)} loss nodes ({config.loss.type})")

    neighbor_pairs = cell_decomp.get_neighbor_pairs()
    print(f"  {len(neighbor_pairs)} neighbor pairs for regularisation")

    print("=== Step 5: Optimising ===")
    solver_opts = _petsc_opts(config)
    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    # Load initial params: from file (warm-start) or from continuous push-forward
    if config.optimization.init_params:
        import jax.numpy as jnp
        data = np.load(config.optimization.init_params)
        params_init = (jnp.array(data["cell_C_flat"]), jnp.array(data["cell_rho"]))
        print(f"  Loaded init params from {config.optimization.init_params}")
    else:
        params_init = cell_mat.get_initial_params()

    # Per-step displacement plot callback
    pts_x = np.asarray(cloak_mesh.points[:, 0])
    pts_y = np.asarray(cloak_mesh.points[:, 1])
    cloak_cells = np.asarray(cloak_mesh.cells)
    step_dir = f"{config.output_dir}/opt_steps"

    def _plot_step(step: int, u: np.ndarray) -> None:
        from rayleigh_cloak.plot import plot_step_field
        plot_step_field(
            u, pts_x, pts_y, params,
            save_path=f"{step_dir}/step_{step:04d}.png",
            title=f"|u| — step {step}",
            cells=cloak_cells,
        )

    opt_cfg = config.optimization
    result = run_optimization(
        fwd_pred=fwd_pred,
        params_init=params_init,
        u_ref_boundary=u_ref_boundary,
        boundary_indices=boundary_indices,
        neighbor_pairs=neighbor_pairs,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lambda_l2=opt_cfg.lambda_l2,
        lambda_neighbor=opt_cfg.lambda_neighbor,
        plot_callback=_plot_step if opt_cfg.plot_every > 0 else None,
        plot_every=opt_cfg.plot_every,
        step_callback=step_callback,
        loss_fn=loss_fn,
    )
    return result


def _make_config_at_fstar(base_config: SimulationConfig, f_star: float):
    """Return a config copy with only f_star changed."""
    return base_config.model_copy(
        update={"domain": base_config.domain.model_copy(
            update={"f_star": float(f_star)}
        )}
    )


def solve_optimization_neural(
    config: SimulationConfig, step_callback=None,
) -> NeuralOptimizationResult:
    """Run cell-based optimisation with neural reparameterization.

    Same setup as ``solve_optimization`` (shared mesh, reference solve,
    submesh extraction, cell decomposition), but optimises MLP weights
    instead of raw cell parameters.

    When ``config.loss.multi_freq.f_stars`` is non-empty, builds one FEM
    problem per frequency and dispatches forward+adjoint solves in parallel
    via a thread pool.
    """
    import jax.numpy as jnp

    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = _full_mesh(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Step 3: Setting up cell decomposition ===")
    cell_decomp = CellDecomposition(geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    cell_mat = CellMaterial(
        geometry, C0, params.rho0, cell_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
        init=config.cells.init,
        init_path=config.cells.init_path,
        aniso_cauchy=config.cells.aniso_cauchy,
    )
    print(f"  {cell_decomp.n_cells} total cells, "
          f"{cell_decomp.n_cloak_cells} in cloak")

    print("=== Step 4: Setting up neural reparameterization ===")
    params_init = cell_mat.get_initial_params()
    neural_cfg = config.optimization.neural
    theta_init, reparam = make_neural_reparam(
        cell_decomp, params_init,
        hidden_size=neural_cfg.hidden_size,
        n_layers=neural_cfg.n_layers,
        n_fourier=neural_cfg.n_fourier,
        seed=neural_cfg.seed,
        output_scale=neural_cfg.output_scale,
        constrained=neural_cfg.constrained,
        kappa=neural_cfg.kappa,
        cap_anisotropy=neural_cfg.cap_anisotropy,
        anisotropy_ratio=neural_cfg.anisotropy_ratio,
        aniso_cauchy=config.cells.aniso_cauchy,
    )
    loaded_opt_state = None
    if neural_cfg.init_weights:
        theta_init, loaded_opt_state = load_theta(neural_cfg.init_weights)
        print(f"  Loaded MLP weights from {neural_cfg.init_weights}")
        if loaded_opt_state is not None:
            print(f"  Restored Adam state (t={loaded_opt_state.t})")

    n_weights = sum(p["W"].size + p["b"].size for p in theta_init)
    print(f"  MLP: {neural_cfg.n_layers} layers, "
          f"{neural_cfg.hidden_size} hidden, "
          f"{n_weights} total weights")

    solver_opts = _petsc_opts(config)

    pts_x = np.asarray(cloak_mesh.points[:, 0])
    pts_y = np.asarray(cloak_mesh.points[:, 1])
    cloak_cells = np.asarray(cloak_mesh.cells)
    step_dir = f"{config.output_dir}/opt_steps"

    def _plot_step(step: int, u: np.ndarray) -> None:
        from rayleigh_cloak.plot import plot_step_field
        plot_step_field(
            u, pts_x, pts_y, params,
            save_path=f"{step_dir}/step_{step:04d}.png",
            title=f"|u| — step {step}",
            cells=cloak_cells,
        )

    opt_cfg = config.optimization
    mf = config.loss.multi_freq

    # Optional dataset-prior penalty (loaded once, reused across freqs).
    gmm_cfg = config.loss.regularizations.material_cement_GMM
    gmm_prior: GMMPrior | None = None
    if gmm_cfg.enabled:
        print(f"=== Loading material-cement GMM prior from {gmm_cfg.path} ===")
        gmm_prior = load_gmm_prior(gmm_cfg.path, quantile=gmm_cfg.quantile)
        print(
            f"  K={int(gmm_prior.weights.shape[0])} components, "
            f"weight={gmm_cfg.weight}, threshold={float(gmm_prior.threshold):.4f}"
        )
    lambda_gmm = gmm_cfg.weight if gmm_cfg.enabled else 0.0
    n_C_params = config.cells.n_C_params

    # ── Multi-frequency path ───────────────────────────────────────
    # Build frequency list from either f_stars or f_min/f_max/f_step
    if mf.f_min is not None and mf.f_max is not None and mf.f_step is not None:
        f_stars = list(np.arange(mf.f_min, mf.f_max + 0.5 * mf.f_step, mf.f_step))
        weights = [1.0] * len(f_stars)
    elif mf.f_stars:
        f_stars = mf.f_stars
        weights = mf.weights if mf.weights else [1.0] * len(f_stars)
        if len(weights) != len(f_stars):
            raise ValueError(
                f"multi_freq.weights length ({len(weights)}) must match "
                f"f_stars length ({len(f_stars)})"
            )
    else:
        f_stars = []
        weights = []

    if f_stars:
        print(f"=== Step 5: Building per-frequency problems ({len(f_stars)} frequencies) ===")
        freq_targets: list[FreqTarget] = []
        for f_star, weight in zip(f_stars, weights):
            cfg_f = _make_config_at_fstar(config, f_star)
            dp_f = DerivedParams.from_config(cfg_f)

            # Reference solve at this frequency
            print(f"  f*={f_star:.2f}: solving reference ...", end="", flush=True)
            ref_f = solve_reference(cfg_f, mesh=full_mesh)

            from rayleigh_cloak.plot import plot_reference_field
            plot_reference_field(
                ref_f,
                save_path=f"{config.output_dir}/reference_field_f{f_star:.4f}.png",
                title=f"Reference field |u| — f*={f_star:.2f}",
            )

            # Build problem at this frequency (sets class attrs, then creates instance)
            problem_f = build_problem(cloak_mesh, cfg_f, dp_f, geometry, cell_decomp)
            fwd_pred_f = ad_wrapper(problem_f, solver_opts, solver_opts)

            # Loss target at this frequency
            indices_f, u_ref_f, loss_fn_f = resolve_loss_target(
                config.loss.type, np.asarray(cloak_mesh.points), geometry,
                dp_f, kept_nodes, ref_f.u, loss_cfg=config.loss,
            )
            print(f" {len(indices_f)} loss nodes, weight={weight:.2f}")

            freq_targets.append(FreqTarget(
                f_star=f_star,
                weight=weight,
                fwd_pred=fwd_pred_f,
                u_ref_boundary=u_ref_f,
                boundary_indices=indices_f,
                loss_fn=loss_fn_f,
            ))

        if mf.strategy == "minimax":
            print("=== Step 6: Optimising (multi-freq minimax neural reparam) ===")
            result = run_optimization_neural_multifreq_minimax(
                freq_targets=freq_targets,
                params_init=params_init,
                reparam=reparam,
                theta_init=theta_init,
                n_iters=opt_cfg.n_iters,
                lr=opt_cfg.lr,
                lr_end=opt_cfg.lr_end,
                lr_schedule=opt_cfg.lr_schedule,
                lambda_l2=opt_cfg.lambda_l2,
                plot_callback=_plot_step if opt_cfg.plot_every > 0 else None,
                plot_every=opt_cfg.plot_every,
                step_callback=step_callback,
                opt_state_init=loaded_opt_state,
                max_workers=mf.max_workers,
                gmm_prior=gmm_prior,
                lambda_gmm=lambda_gmm,
                n_C_params=n_C_params,
            )
        else:
            print("=== Step 6: Optimising (multi-freq neural reparam) ===")
            result = run_optimization_neural_multifreq(
                freq_targets=freq_targets,
                params_init=params_init,
                reparam=reparam,
                theta_init=theta_init,
                n_iters=opt_cfg.n_iters,
                lr=opt_cfg.lr,
                lr_end=opt_cfg.lr_end,
                lr_schedule=opt_cfg.lr_schedule,
                lambda_l2=opt_cfg.lambda_l2,
                plot_callback=_plot_step if opt_cfg.plot_every > 0 else None,
                plot_every=opt_cfg.plot_every,
                step_callback=step_callback,
                opt_state_init=loaded_opt_state,
                max_workers=mf.max_workers,
                gmm_prior=gmm_prior,
                lambda_gmm=lambda_gmm,
                n_C_params=n_C_params,
            )
        return result

    # ── Single-frequency path (original) ───────────────────────────
    print("=== Step 5: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    from rayleigh_cloak.plot import plot_reference_field
    plot_reference_field(
        ref_result,
        save_path=f"{config.output_dir}/reference_field.png",
    )

    problem = build_problem(cloak_mesh, config, params, geometry, cell_decomp)

    # Loss target nodes from config
    boundary_indices, u_ref_boundary, loss_fn = resolve_loss_target(
        config.loss.type, np.asarray(cloak_mesh.points), geometry, params,
        kept_nodes, ref_result.u, loss_cfg=config.loss,
    )
    print(f"  {len(boundary_indices)} loss nodes ({config.loss.type})")

    # Free-surface transmission metric (independent of the training loss): the
    # transmitted-displacement ratio on the surface *beyond* the cloak. The node
    # sets are fixed for the whole run, so we precompute them once and let the
    # optimisation loop evaluate the ratio every step from the field it already
    # solves for (no extra forward solve).
    trans_surface_case = get_top_surface_beyond_cloak_indices(
        np.asarray(cloak_mesh.points), geometry, params.y_top,
        params.x_off, params.x_off + params.W,
    )
    trans_surface_ref = kept_nodes[trans_surface_case]

    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    print("=== Step 6: Optimising (neural reparam) ===")
    result = run_optimization_neural(
        fwd_pred=fwd_pred,
        params_init=params_init,
        u_ref_boundary=u_ref_boundary,
        boundary_indices=boundary_indices,
        u_ref_full=ref_result.u,
        trans_surface_case=trans_surface_case,
        trans_surface_ref=trans_surface_ref,
        reparam=reparam,
        theta_init=theta_init,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lr_end=opt_cfg.lr_end,
        lr_schedule=opt_cfg.lr_schedule,
        lambda_l2=opt_cfg.lambda_l2,
        plot_callback=_plot_step if opt_cfg.plot_every > 0 else None,
        plot_every=opt_cfg.plot_every,
        step_callback=step_callback,
        opt_state_init=loaded_opt_state,
        loss_fn=loss_fn,
        gmm_prior=gmm_prior,
        lambda_gmm=lambda_gmm,
        n_C_params=n_C_params,
    )
    return result


def solve_optimization_neural_topo(
    config: SimulationConfig, step_callback=None,
) -> TopoOptimizationResult:
    """Run topology optimisation with pixel-level density prediction.

    Two-level coarse-graining:

    1. **Coarse cells** (``config.cells.n_x × n_y``): used only to compute
       target C_eff/rho_eff and match to dataset geometries.
    2. **Fine pixels** (coarse × ``pixel_per_cell``): the MLP predicts one
       density scalar per pixel.  Material is assigned via SIMP directly
       at the FEM quadrature-point level.
    """
    import jax.numpy as jnp

    from rayleigh_cloak.dataset_init import (
        build_pixel_targets,
        load_dataset,
        match_cells_to_dataset,
    )
    from rayleigh_cloak.materials import C_to_flat2, C_to_voigt4, symmetrize_stiffness

    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)
    topo_cfg = config.optimization.topo_neural

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = _full_mesh(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Solving reference problem (on full mesh) ===")
    ref_result = solve_reference(config, mesh=full_mesh)

    from rayleigh_cloak.plot import plot_reference_field
    plot_reference_field(
        ref_result,
        save_path=f"{config.output_dir}/reference_field.png",
    )

    print("=== Step 3: Extracting submesh (removing defect elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Step 4: Setting up coarse cell decomposition (for init) ===")
    coarse_decomp = CellDecomposition(
        geometry, config.cells.n_x, config.cells.n_y)
    C0 = C_iso(params.lam, params.mu)
    coarse_mat = CellMaterial(
        geometry, C0, params.rho0, coarse_decomp,
        n_C_params=config.cells.n_C_params,
        symmetrize_init=config.cells.symmetrize_init,
        init=config.cells.init,
        init_path=config.cells.init_path,
    )
    print(f"  Coarse: {coarse_decomp.n_cells} cells, "
          f"{coarse_decomp.n_cloak_cells} in cloak")

    print("=== Step 5: Building fine pixel grid ===")
    ppc = topo_cfg.pixel_per_cell
    n_fine_x = config.cells.n_x * ppc
    n_fine_y = config.cells.n_y * ppc
    fine_decomp = CellDecomposition(geometry, n_fine_x, n_fine_y)
    print(f"  Fine grid: {n_fine_x}×{n_fine_y} = {fine_decomp.n_cells} pixels, "
          f"{fine_decomp.n_cloak_cells} in cloak")

    print("=== Step 6: Matching coarse cells to dataset ===")
    dataset = load_dataset(topo_cfg.dataset_path)
    print(f"  Dataset: {len(dataset.geometries)} entries")

    # Compute target (λ, μ) per cell by symmetrizing the transformed C_eff
    from rayleigh_cloak.materials import C_eff as compute_C_eff, rho_eff as compute_rho_eff
    coarse_lam_mu = np.zeros((coarse_decomp.n_cells, 2))
    coarse_rho = np.zeros(coarse_decomp.n_cells)
    for i, center in enumerate(coarse_decomp.cell_centers):
        if coarse_decomp.cloak_mask[i]:
            x = jnp.array(center)
            C_i = compute_C_eff(x, geometry, C0)
            C_sym = symmetrize_stiffness(C_i)
            coarse_lam_mu[i] = np.asarray(C_to_flat2(C_sym))
            coarse_rho[i] = float(compute_rho_eff(x, geometry, params.rho0))
        else:
            coarse_lam_mu[i] = np.array([params.lam, params.mu])
            coarse_rho[i] = params.rho0

    matched_geoms, matched_idx = match_cells_to_dataset(
        coarse_lam_mu, coarse_rho, coarse_decomp.cloak_mask,
        dataset,
        rho_weight=topo_cfg.rho_weight,
    )
    print(f"  Matched {len(matched_idx)} cloak cells to dataset entries")

    # Build pixel-level density targets from matched geometries
    pixel_targets = build_pixel_targets(
        matched_geoms, config.cells.n_x, config.cells.n_y,
        ppc, coarse_decomp.cloak_mask,
    )

    print("=== Step 7: Setting up topology neural reparameterization ===")
    # Cement Lamé parameters
    E_c = topo_cfg.E_cement
    nu_c = topo_cfg.nu_micro
    lam_cement = E_c * nu_c / ((1 + nu_c) * (1 - 2 * nu_c))
    mu_cement = E_c / (2 * (1 + nu_c))
    C0_flat = jnp.array(C_to_flat2(C0))

    theta_init, reparam = make_neural_reparam_topo(
        fine_decomp,
        C0_flat=C0_flat,
        rho0=params.rho0,
        lam_cement=lam_cement,
        mu_cement=mu_cement,
        rho_cement=topo_cfg.rho_cement,
        pixel_targets=pixel_targets,
        hidden_size=topo_cfg.hidden_size,
        n_layers=topo_cfg.n_layers,
        n_fourier=topo_cfg.n_fourier,
        seed=topo_cfg.seed,
        simp_p=topo_cfg.simp_p,
        output_scale=topo_cfg.output_scale,
        density_eps=topo_cfg.density_eps,
        fourier_sigma=topo_cfg.fourier_sigma,
    )
    n_weights = sum(p["W"].size + p["b"].size for p in theta_init)
    print(f"  MLP: {topo_cfg.n_layers} layers, "
          f"{topo_cfg.hidden_size} hidden, "
          f"{n_weights} total weights, output_dim=1"
          f" (residual logit init from dataset targets)")

    # Build FEM problem with fine pixel decomposition
    # n_C_params MUST be 2 for isotropic SIMP material
    problem = build_problem(
        cloak_mesh, config, params, geometry, fine_decomp,
        n_C_params_override=2,
    )

    # Loss target nodes from config
    boundary_indices, u_ref_boundary, loss_fn = resolve_loss_target(
        config.loss.type, np.asarray(cloak_mesh.points), geometry, params,
        kept_nodes, ref_result.u, loss_cfg=config.loss,
    )
    print(f"  {len(boundary_indices)} loss nodes ({config.loss.type})")

    # Get initial material params from the decode (post-pretrain)
    params_init = reparam.decode(theta_init)

    solver_opts = _petsc_opts(config)
    fwd_pred = ad_wrapper(problem, solver_opts, solver_opts)

    pts_x = np.asarray(cloak_mesh.points[:, 0])
    pts_y = np.asarray(cloak_mesh.points[:, 1])
    cloak_cells = np.asarray(cloak_mesh.cells)
    step_dir = f"{config.output_dir}/opt_steps"

    def _plot_step(step: int, u: np.ndarray) -> None:
        from rayleigh_cloak.plot import plot_step_field
        plot_step_field(
            u, pts_x, pts_y, params,
            save_path=f"{step_dir}/step_{step:04d}.png",
            title=f"|u| — step {step}",
            cells=cloak_cells,
        )

    # Density visualisation: cloak mask reshaped to 2D for overlay
    cloak_mask_2d = np.array(fine_decomp.cloak_mask).reshape(
        fine_decomp.n_x, fine_decomp.n_y).T  # (n_y, n_x)
    density_dir = f"{config.output_dir}/density_steps"

    def _plot_density(step: int, density_grid: np.ndarray) -> None:
        from rayleigh_cloak.neural_reparam_topo import plot_density_grid
        plot_density_grid(
            density_grid, step,
            save_path=f"{density_dir}/density_{step:04d}.png",
            cloak_mask_2d=cloak_mask_2d,
        )

    print("=== Step 8: Optimising (topology neural reparam) ===")
    opt_cfg = config.optimization
    do_plot = opt_cfg.plot_every > 0
    result = run_optimization_neural_topo(
        fwd_pred=fwd_pred,
        params_init=params_init,
        u_ref_boundary=u_ref_boundary,
        boundary_indices=boundary_indices,
        reparam=reparam,
        theta_init=theta_init,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lr_end=opt_cfg.lr_end,
        lr_schedule=opt_cfg.lr_schedule,
        lambda_l2=opt_cfg.lambda_l2,
        lambda_bin=topo_cfg.lambda_bin,
        beta_start=topo_cfg.beta_start,
        beta_end=topo_cfg.beta_end,
        plot_callback=_plot_step if do_plot else None,
        density_callback=_plot_density if do_plot else None,
        plot_every=opt_cfg.plot_every,
        step_callback=step_callback,
        loss_fn=loss_fn,
    )
    return result


# ── Nassar cloaking pipeline ──────────────────────────────────────────


@dataclass
class NassarResult:
    """Result of a Nassar cloaking simulation."""
    u_ref: np.ndarray           # reference solution (no void)
    u_cloak: np.ndarray         # cloaked solution
    mesh_ref: Mesh              # full mesh (shared)
    mesh_cloak: Mesh            # submesh (void removed)
    kept_nodes: np.ndarray      # mapping from submesh to full mesh nodes
    distortion: float           # % cloaking distortion
    config: SimulationConfig
    params: DerivedParams


def _plot_nassar_fields(
    u_ref: np.ndarray,
    mesh_ref: Mesh,
    u_cloak: np.ndarray,
    mesh_cloak: Mesh,
    params: DerivedParams,
    geometry,
    output_dir: str,
) -> None:
    """Plot Re(ux) displacement fields for reference and cloaked solutions."""
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    os.makedirs(output_dir, exist_ok=True)

    x_off, y_off = params.x_off, params.y_off
    W, H = params.W, params.H
    ri, rc = params.ri, params.rc
    x_c, y_c = geometry.x_c, geometry.y_c

    def _phys_mask(cells, pts):
        """Return boolean mask for cells whose centroid is in the physical domain."""
        centroids = pts[cells].mean(axis=1)
        return (
            (centroids[:, 0] >= x_off - 1e-8) &
            (centroids[:, 0] <= x_off + W + 1e-8) &
            (centroids[:, 1] >= y_off - 1e-8) &
            (centroids[:, 1] <= y_off + H + 1e-8)
        )

    def _add_cloak_circles(ax):
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot((x_c - x_off) + ri * np.cos(theta),
                (y_c - y_off) + ri * np.sin(theta),
                '--', color='black', lw=1.0, label=f'r_i={ri*1e3:.0f}mm')
        ax.plot((x_c - x_off) + rc * np.cos(theta),
                (y_c - y_off) + rc * np.sin(theta),
                '--', color='black', lw=1.0, label=f'r_c={rc*1e3:.0f}mm')

    def _plot_field(u, mesh, title, fname, component=0):
        """Plot one displacement component on the physical domain."""
        pts = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells)
        field = u[:, component]

        phys = _phys_mask(cells, pts)
        cells_phys = cells[phys]
        tri = mtri.Triangulation(pts[:, 0] - x_off, pts[:, 1] - y_off, cells_phys)

        vlim = np.percentile(np.abs(field), 97)
        if vlim < 1e-30:
            vlim = 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        tc = ax.tricontourf(tri, field, levels=100, cmap='RdBu_r',
                            vmin=-vlim, vmax=vlim)
        fig.colorbar(tc, ax=ax, shrink=0.7, label=title)
        _add_cloak_circles(ax)

        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=7)

        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    def _compute_grad_fields(u, mesh):
        """Compute per-element div and curl of Re(u) on TRI3 elements.

        For TRI3, gradients are constant per element:
            div(u) = du_x/dx + du_y/dy     (P-wave / compressional)
            curl(u) = du_y/dx - du_x/dy    (S-wave / rotational, scalar in 2D)

        Returns (div, curl, centroids) arrays of shape (n_elements,).
        """
        pts = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells)
        re_ux = u[:, 0]
        re_uy = u[:, 1]

        # Vertices per element: (n_elem, 3, 2)
        v = pts[cells]
        x1, y1 = v[:, 0, 0], v[:, 0, 1]
        x2, y2 = v[:, 1, 0], v[:, 1, 1]
        x3, y3 = v[:, 2, 0], v[:, 2, 1]

        # 2 * area (signed)
        det = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        det = np.where(np.abs(det) < 1e-30, 1e-30, det)

        # Shape function gradients (constant per element):
        # dN1/dx = (y2-y3)/det,  dN2/dx = (y3-y1)/det,  dN3/dx = (y1-y2)/det
        # dN1/dy = (x3-x2)/det,  dN2/dy = (x1-x3)/det,  dN3/dy = (x2-x1)/det
        dNdx = np.column_stack([(y2-y3)/det, (y3-y1)/det, (y1-y2)/det])
        dNdy = np.column_stack([(x3-x2)/det, (x1-x3)/det, (x2-x1)/det])

        # Nodal values per element: (n_elem, 3)
        ux_e = re_ux[cells]
        uy_e = re_uy[cells]

        # Gradients: du/dx = sum_a N_a,x * u_a
        dux_dx = np.sum(dNdx * ux_e, axis=1)
        duy_dy = np.sum(dNdy * uy_e, axis=1)
        duy_dx = np.sum(dNdx * uy_e, axis=1)
        dux_dy = np.sum(dNdy * ux_e, axis=1)

        div = dux_dx + duy_dy
        curl = duy_dx - dux_dy
        centroids = v.mean(axis=1)

        return div, curl, centroids

    def _plot_cell_field(values, mesh, phys, title, fname, cmap='RdBu_r'):
        """Plot a per-element scalar field using tripcolor."""
        pts = np.asarray(mesh.points)
        cells = np.asarray(mesh.cells)
        cells_phys = cells[phys]
        vals_phys = values[phys]

        tri = mtri.Triangulation(pts[:, 0] - x_off, pts[:, 1] - y_off, cells_phys)

        vlim = np.percentile(np.abs(vals_phys), 97)
        if vlim < 1e-30:
            vlim = 1.0

        fig, ax = plt.subplots(figsize=(8, 8))
        tp = ax.tripcolor(tri, facecolors=vals_phys, cmap=cmap,
                          vmin=-vlim, vmax=vlim, edgecolors='none')
        fig.colorbar(tp, ax=ax, shrink=0.7, label=title)
        _add_cloak_circles(ax)

        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=7)

        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved {path}")

    # ── Displacement components ──
    # Reference: Re(ux) on full mesh
    _plot_field(u_ref, mesh_ref, 'Re(u_x) — reference (no void)',
                'nassar_ref_re_ux.png', component=0)

    # Cloak: Re(ux) on submesh
    _plot_field(u_cloak, mesh_cloak, 'Re(u_x) — cloaked',
                'nassar_cloak_re_ux.png', component=0)

    # Reference: Re(uy) on full mesh
    _plot_field(u_ref, mesh_ref, 'Re(u_y) — reference (no void)',
                'nassar_ref_re_uy.png', component=1)

    # Cloak: Re(uy) on submesh
    _plot_field(u_cloak, mesh_cloak, 'Re(u_y) — cloaked',
                'nassar_cloak_re_uy.png', component=1)

    # ── Divergence and curl ──
    pts_ref = np.asarray(mesh_ref.points)
    cells_ref = np.asarray(mesh_ref.cells)
    phys_ref = _phys_mask(cells_ref, pts_ref)
    div_ref, curl_ref, _ = _compute_grad_fields(u_ref, mesh_ref)
    _plot_cell_field(div_ref, mesh_ref, phys_ref,
                     'div Re(u) — reference (no void)', 'nassar_ref_div.png')
    _plot_cell_field(curl_ref, mesh_ref, phys_ref,
                     'curl Re(u) — reference (no void)', 'nassar_ref_curl.png')

    pts_cl = np.asarray(mesh_cloak.points)
    cells_cl = np.asarray(mesh_cloak.cells)
    phys_cl = _phys_mask(cells_cl, pts_cl)
    div_cl, curl_cl, _ = _compute_grad_fields(u_cloak, mesh_cloak)
    _plot_cell_field(div_cl, mesh_cloak, phys_cl,
                     'div Re(u) — cloaked', 'nassar_cloak_div.png')
    _plot_cell_field(curl_cl, mesh_cloak, phys_cl,
                     'curl Re(u) — cloaked', 'nassar_cloak_curl.png')


def solve_nassar(config: SimulationConfig) -> NassarResult:
    """Run the Nassar cloaking pipeline.

    Steps:
    1. Generate full mesh (no void cutout).
    2. Solve reference (homogeneous background, no void).
    3. Extract submesh (void elements removed).
    4. Build cell decomposition + NassarCellMaterial.
    5. Solve cloaked problem with Nassar cell materials.
    6. Compute cloaking distortion (%).
    """
    import jax.numpy as jnp
    from rayleigh_cloak.nassar import NassarCellMaterial
    from rayleigh_cloak.problem import RayleighCloakProblem

    params = DerivedParams.from_config(config)
    geometry = _create_geometry(config, params)

    print("=== Step 1: Generating shared mesh ===")
    full_mesh = _full_mesh(config, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} elements")

    print("=== Step 2: Solving reference problem (homogeneous, no void) ===")
    ref_result = solve_reference(config, mesh=full_mesh)
    u_ref = ref_result.u

    print("=== Step 3: Extracting submesh (removing void elements) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(f"  Submesh: {len(cloak_mesh.points)} nodes, "
          f"{len(cloak_mesh.cells)} elements")

    print("=== Step 4: Setting up Nassar cell materials ===")
    if config.nassar.polar:
        from rayleigh_cloak.cells_polar import PolarCellDecomposition
        from rayleigh_cloak.nassar import NassarPolarMaterial
        cell_decomp = PolarCellDecomposition(
            ri=params.ri, rc=params.rc,
            x_c=geometry.x_c, y_c=geometry.y_c,
            N=config.nassar.lattice_N, M=config.nassar.lattice_M,
        )
        nassar_mat = NassarPolarMaterial(
            geometry, params.lam, params.mu, params.rho0, cell_decomp,
        )
        print(f"  Polar grid: {config.nassar.lattice_N} sectors × "
              f"{config.nassar.lattice_M} layers = {cell_decomp.n_cells} cells")
    else:
        n_x = config.nassar.cell_n_x
        n_y = config.nassar.cell_n_y
        cell_decomp = CellDecomposition(geometry, n_x, n_y)
        nassar_mat = NassarCellMaterial(
            geometry, params.lam, params.mu, params.rho0, cell_decomp,
        )
        print(f"  {cell_decomp.n_cells} total cells, "
              f"{cell_decomp.n_cloak_cells} in cloak")

    # Attach Nassar material to the problem class
    RayleighCloakProblem._nassar_cell_material = nassar_mat

    cloak_config = config.model_copy(update={"is_reference": False})
    problem = build_problem(cloak_mesh, cloak_config, params, geometry, cell_decomp)

    # Set initial Nassar params
    params_init = nassar_mat.get_initial_params()
    problem.set_params(params_init)

    print("=== Step 5: Solving cloaked problem ===")
    solver_opts = _petsc_opts(config)
    sol_list = jax_fem_solver(problem, solver_options=solver_opts)
    u_cloak = np.asarray(sol_list[0])

    print("=== Step 6: Computing cloaking distortion ===")

    # Circular measurement at r = 1.5*rc (primary metric)
    from rayleigh_cloak.optimize import get_circular_boundary_indices
    circ_indices = get_circular_boundary_indices(
        np.asarray(cloak_mesh.points),
        geometry.x_c, geometry.y_c, 1.5 * params.rc,
    )
    u_ref_circ = jnp.array(u_ref[kept_nodes[circ_indices]])
    u_cloak_circ = jnp.array(u_cloak[circ_indices])
    diff_circ = u_cloak_circ - u_ref_circ
    ref_norm_sq_circ = jnp.sum(u_ref_circ ** 2) + 1e-30
    distortion = float(100.0 * jnp.sqrt(jnp.sum(diff_circ ** 2) / ref_norm_sq_circ))
    print(f"  {len(circ_indices)} nodes on circle r=1.5rc")
    print(f"  Cloaking distortion (circle): {distortion:.2f}%")

    # Also report all-boundary metric for comparison
    boundary_indices = get_all_physical_boundary_indices(
        np.asarray(cloak_mesh.points),
        params.x_off, params.y_off, params.W, params.H,
    )
    u_ref_boundary = jnp.array(u_ref[kept_nodes[boundary_indices]])
    u_cloak_boundary = jnp.array(u_cloak[boundary_indices])
    diff_bnd = u_cloak_boundary - u_ref_boundary
    ref_norm_sq_bnd = jnp.sum(u_ref_boundary ** 2) + 1e-30
    dist_bnd = float(100.0 * jnp.sqrt(jnp.sum(diff_bnd ** 2) / ref_norm_sq_bnd))
    print(f"  {len(boundary_indices)} nodes on all physical boundaries")
    print(f"  Cloaking distortion (boundaries): {dist_bnd:.2f}%")

    # ── Displacement field plots ──
    print("=== Step 7: Plotting displacement fields ===")
    _plot_nassar_fields(
        u_ref, full_mesh, u_cloak, cloak_mesh,
        params, geometry, config.output_dir,
    )

    # Clean up class attribute
    RayleighCloakProblem._nassar_cell_material = None

    return NassarResult(
        u_ref=u_ref, u_cloak=u_cloak,
        mesh_ref=full_mesh, mesh_cloak=cloak_mesh,
        kept_nodes=kept_nodes, distortion=distortion,
        config=config, params=params,
    )
