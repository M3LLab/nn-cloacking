"""High-level solve / optimise interface for the 3D Rayleigh cloak."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax_fem.generate_mesh import Mesh
from jax_fem.solver import ad_wrapper, solver as jax_fem_solver

from rayleigh_cloak_3d.config import DerivedParams3D, SimulationConfig3D
from rayleigh_cloak_3d.geometry.conical import ConicalCloakGeometry
from rayleigh_cloak_3d.loss import resolve_loss_target
from rayleigh_cloak_3d.material_field import (
    CellDecomposedNeuralField,
    CellDecomposition3D,
    ContinuousNeuralField,
)
from rayleigh_cloak_3d.materials import C_iso_3d
from rayleigh_cloak_3d.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak_3d.optimize import (
    NeuralOptimizationResult,
    cloaking_loss,
    run_optimization,
)
from rayleigh_cloak_3d.problem import build_problem


@dataclass
class SolutionResult3D:
    u: np.ndarray                       # (num_nodes, 6)
    mesh: Mesh
    config: SimulationConfig3D
    params: DerivedParams3D
    full_mesh: Mesh | None = None
    kept_nodes: np.ndarray | None = None


def _create_geometry(cfg: SimulationConfig3D, params: DerivedParams3D):
    if cfg.geometry_type == "conical":
        return ConicalCloakGeometry.from_params(params)
    raise ValueError(f"Unknown geometry_type: {cfg.geometry_type!r}")


def _petsc_opts(cfg: SimulationConfig3D) -> dict:
    opts = {
        "ksp_type": cfg.solver.ksp_type,
        "pc_type": cfg.solver.pc_type,
    }
    if cfg.solver.pc_factor_mat_solver_type:
        opts["pc_factor_mat_solver_type"] = cfg.solver.pc_factor_mat_solver_type
    return {"petsc_solver": opts}


def _solver_opts_pair(cfg: SimulationConfig3D) -> tuple[dict, dict]:
    """Return (forward_opts, adjoint_opts) dicts for the configured backend.

    For the cuDSS backend the adjoint options carry ``_is_adjoint=True`` so
    the custom solver knows to apply the J-trick (sign-flip of imaginary
    DOFs on both sides of the solve) and reuse the forward factorisation
    — key to fitting a full AD step into the 2 min/iter budget.
    """
    if cfg.solver.backend == "cudss":
        from rayleigh_cloak_3d.cudss_solver import make_forward_and_adjoint
        return make_forward_and_adjoint({
            "hybrid_memory": cfg.solver.cudss_hybrid_memory,
            "verbose": cfg.solver.cudss_verbose,
        })
    opts = _petsc_opts(cfg)
    return opts, opts


def solve(cfg: SimulationConfig3D, mesh: Mesh | None = None) -> SolutionResult3D:
    """Run a forward simulation (reference or with continuous push-forward cloak)."""
    params = DerivedParams3D.from_config(cfg)
    geometry = _create_geometry(cfg, params)

    if mesh is None:
        mesh = generate_mesh_full(cfg, params, geometry)

    problem = build_problem(mesh, cfg, params, geometry)

    print("Solving 3D frequency-domain system ...")
    fwd_opts, _ = _solver_opts_pair(cfg)
    sol_list = jax_fem_solver(problem, solver_options=fwd_opts)
    return SolutionResult3D(
        u=np.asarray(sol_list[0]),
        mesh=mesh,
        config=cfg,
        params=params,
    )


def solve_reference(cfg: SimulationConfig3D, mesh: Mesh | None = None) -> SolutionResult3D:
    ref_cfg = cfg.model_copy(update={"is_reference": True})
    return solve(ref_cfg, mesh=mesh)


def _build_material_field(
    cfg: SimulationConfig3D,
    params: DerivedParams3D,
    geometry,
):
    """Instantiate the MaterialField implementation selected by the config."""
    C0 = C_iso_3d(params.lam, params.mu)
    nf = cfg.optimization.neural
    mode = cfg.cells.mode

    if mode == "continuous":
        mf = ContinuousNeuralField(
            geometry=geometry,
            C0=C0,
            rho0=params.rho0,
            n_C_params=cfg.cells.n_C_params,
            n_fourier=nf.n_fourier,
            output_scale=nf.output_scale,
            symmetrize_init=cfg.cells.symmetrize_init,
        )
        print(f"  MaterialField: continuous (QP-level), n_C_params={cfg.cells.n_C_params}")
        return mf

    if mode == "grid":
        cd = CellDecomposition3D(
            geometry, cfg.cells.n_x, cfg.cells.n_y, cfg.cells.n_z,
        )
        print(
            f"  CellDecomposition3D: {cd.n_cells} cells "
            f"({cd.n_x}×{cd.n_y}×{cd.n_z}), {cd.n_cloak_cells} inside cloak"
        )
        mf = CellDecomposedNeuralField(
            cell_decomp=cd,
            C0=C0,
            rho0=params.rho0,
            n_C_params=cfg.cells.n_C_params,
            n_fourier=nf.n_fourier,
            output_scale=nf.output_scale,
            symmetrize_init=cfg.cells.symmetrize_init,
        )
        return mf

    raise ValueError(f"Unknown cells.mode: {mode!r}")


def solve_optimization_neural(
    cfg: SimulationConfig3D,
    step_callback=None,
    on_material_field_ready=None,
) -> NeuralOptimizationResult:
    """Single-frequency neural-reparam optimisation in 3D.

    Steps:
      1. Generate full-domain mesh (no defect cut-out).
      2. Reference solve on the full mesh.
      3. Extract submesh (defect tets removed).
      4. Build material field (continuous or cell-decomposed) and FEM problem.
      5. Resolve loss target nodes and interpolate the reference there.
      6. Run Adam over MLP weights.
    """
    params = DerivedParams3D.from_config(cfg)
    geometry = _create_geometry(cfg, params)

    print("=== Step 1: Generating full 3D mesh ===")
    full_mesh = generate_mesh_full(cfg, params, geometry)
    print(f"  Full mesh: {len(full_mesh.points)} nodes, "
          f"{len(full_mesh.cells)} tets")

    print("=== Step 2: Reference solve (homogeneous, full mesh) ===")
    ref = solve_reference(cfg, mesh=full_mesh)

    print("=== Step 3: Extracting submesh (removing defect) ===")
    cloak_mesh, kept_nodes = extract_submesh(full_mesh, geometry)
    print(
        f"  Submesh: {len(cloak_mesh.points)} nodes, {len(cloak_mesh.cells)} tets "
        f"({len(full_mesh.cells) - len(cloak_mesh.cells)} defect tets removed, "
        f"{len(full_mesh.points) - len(kept_nodes)} orphan nodes removed)"
    )

    print("=== Step 4: Building material field + FEM problem ===")
    material_field = _build_material_field(cfg, params, geometry)
    problem = build_problem(cloak_mesh, cfg, params, geometry)
    material_field.bind_mesh(np.asarray(problem.physical_quad_points))

    if on_material_field_ready is not None:
        on_material_field_ready(material_field)

    print("=== Step 5: Resolving loss target ===")
    cloak_pts = np.asarray(cloak_mesh.points)
    target_idx = resolve_loss_target(cfg, params, geometry, cloak_pts)
    # Reference solution is defined on the full mesh — map via kept_nodes.
    u_ref_full = np.asarray(ref.u)
    u_ref_on_submesh = u_ref_full[kept_nodes]           # (n_submesh_nodes, 6)
    u_ref_target = jnp.asarray(u_ref_on_submesh[target_idx])
    print(f"  {len(target_idx)} target nodes ({cfg.loss.type})")

    print("=== Step 6: Optimising MLP weights ===")
    fwd_opts, adj_opts = _solver_opts_pair(cfg)
    fwd_solve = ad_wrapper(problem, fwd_opts, adj_opts)

    # Wrap so the optimiser's forward takes theta directly.
    def fwd_pred(theta):
        return fwd_solve(material_field.evaluate(theta))

    nf = cfg.optimization.neural
    theta_init = material_field.init_theta(nf.hidden_size, nf.n_layers, nf.seed)

    opt_cfg = cfg.optimization
    return run_optimization(
        fwd_pred=fwd_pred,
        theta_init=theta_init,
        u_ref_target=u_ref_target,
        target_indices=target_idx,
        n_iters=opt_cfg.n_iters,
        lr=opt_cfg.lr,
        lr_end=opt_cfg.lr_end,
        lr_schedule=opt_cfg.lr_schedule,
        loss_fn=cloaking_loss,
        step_callback=step_callback,
    )
