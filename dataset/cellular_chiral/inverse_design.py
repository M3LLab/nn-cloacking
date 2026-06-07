"""Neural-field-driven unit cell inverse design.

Optimizes a 50×50 binary microstructure – parameterized by a 25×25 neural
field assembled with D2/mirror (squared) symmetry – to match target effective
stiffness properties via periodic FEM homogenization.

Pipeline
--------
1. MLP(pixel_coords_25x25) → sigmoid → soft 25×25 quadrant.
   Border pixels are clamped to the CA-pipeline pattern (all-solid except
   centered gate openings on each edge, exactly as in ``generator.CAConfig``).
2. Squared assembly: quadrant → 50×50 canvas via mirror reflections.
3. Periodic elastodynamic FEM homogenization at frequency ``f_star``
   (4 load cases, SIMP-soft material so the pipeline is differentiable).
4. Volume-averaged stress → flat4 = (C11, C22, C12, C66).
5. Loss = sum_i  weights_i * (flat4_i - target_i)^2.

All steps are JAX-differentiable.  Gradients flow from the loss back through
the FEM (via jax-fem's implicit-adjoint ``ad_wrapper``), through the symmetric
assembly, and into the MLP weights.
"""
from __future__ import annotations

import contextlib
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp

with contextlib.redirect_stdout(io.StringIO()):
    from jax_fem.problem import Problem
    from jax_fem.solver import ad_wrapper
    from jax_fem.generate_mesh import Mesh

from dataset.stiffness.calc_fem import (
    E_CEMENT,
    NU,
    RHO_CEMENT,
    build_periodic_pmat,
    make_structured_tri_mesh,
)
from rayleigh_cloak.neural_reparam import (
    fourier_features,
    init_mlp,
    load_theta,
    mlp_forward,
    save_theta,
)
from rayleigh_cloak.optimize import AdamState, adam_init, adam_update


# ── CA border constants (must match generator.CAConfig defaults) ───────

_QUADRANT_N: int = 25
_GATE_WIDTH: int = 5
_GATE_START: int = (_QUADRANT_N - _GATE_WIDTH) // 2   # 10
_GATE_END: int = _GATE_START + _GATE_WIDTH             # 15

# 4 load cases: augmented Voigt [e11, e22, e12, e21]
_LOAD_CASES: list[np.ndarray] = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [0.0, 1.0]]),
    np.array([[0.0, 1.0], [0.0, 0.0]]),
    np.array([[0.0, 0.0], [1.0, 0.0]]),
]


# ── CA border constraints on the 25×25 quadrant ───────────────────────

def make_quadrant_border(
    gate_width: int = _GATE_WIDTH,
    N: int = _QUADRANT_N,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return (border_values, interior_mask) for the N×N quadrant.

    Matches the CA pipeline's _reverse_map convention:
      grid border pixels = 1 (live) → reverse_map → void = 0  (white)
      grid gate   pixels = 0 (dead) → reverse_map → material = 1  (black)

    border_values : (N, N) float32 — 0=void on border, 1=material at gate.
    interior_mask : (N, N) bool   — True for pixels the MLP controls.

    Only the top row and left column are frozen; bottom row and right column
    follow automatically from the flipud / fliplr assembly.
    """
    gs = (N - gate_width) // 2
    ge = gs + gate_width

    border_vals = np.zeros((N, N), dtype=np.float32)  # border = void
    border_vals[0, gs:ge] = 1.0    # top row gate = material
    border_vals[gs:ge, 0] = 1.0    # left col gate = material

    interior = np.ones((N, N), dtype=bool)
    interior[0, :] = False   # top row fixed
    interior[:, 0] = False   # left col fixed

    return jnp.array(border_vals), jnp.array(interior)


# ── Squared assembly (mirror symmetry, D2) ────────────────────────────

def assemble_squared(quadrant: jnp.ndarray) -> jnp.ndarray:
    """(N, N) soft quadrant → (2N, 2N) canvas with D2 mirror symmetry.

    Two perpendicular mirror axes (horizontal + vertical center lines) and
    a 180° rotation — but not 90° rotation, so C11 ≠ C22 in general.

    Layout:
        TL = quadrant           | TR = fliplr(quadrant)
        BL = flipud(quadrant)   | BR = flipud(fliplr(quadrant))
    """
    tr = jnp.fliplr(quadrant)
    top = jnp.concatenate([quadrant, tr], axis=1)
    bot = jnp.concatenate([jnp.flipud(quadrant), jnp.flipud(tr)], axis=1)
    return jnp.concatenate([top, bot], axis=0)


# ── Neural field ──────────────────────────────────────────────────────

def make_cell_features(N: int = _QUADRANT_N, n_fourier: int = 32) -> jnp.ndarray:
    """Fourier features for the N×N pixel-center grid, shape (N*N, 4*n_fourier)."""
    xs = np.linspace(0.0, 1.0, N)
    ys = np.linspace(0.0, 1.0, N)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)   # (N*N, 2)
    return fourier_features(jnp.array(coords, dtype=jnp.float32), n_fourier)


@dataclass
class CellNeuralField:
    """MLP-backed neural field for a 25×25 quadrant assembled into 50×50.

    Attributes
    ----------
    features         : (N*N, n_feat) Fourier input features for the MLP
    border_vals      : (N, N) clamped CA border values (0 or 1)
    interior_mask    : (N, N) bool — True where the MLP output is used
    N                : quadrant size (default 25)
    initial_quadrant : (N, N) float32 in (0, 1) — nearest-neighbour soft
                       canvas for residual initialisation.  When provided,
                       the MLP output is a logit-space correction on top of
                       this initial canvas so the optimisation starts at the
                       NN cell and refines from there.  None → standard
                       sigmoid-only decode (starts at 0.5 everywhere).
    """

    features: jnp.ndarray
    border_vals: jnp.ndarray
    interior_mask: jnp.ndarray
    N: int = _QUADRANT_N
    initial_quadrant: jnp.ndarray | None = None

    def decode_quadrant(self, theta: list[dict]) -> jnp.ndarray:
        """theta → soft (N, N) quadrant with CA borders clamped.

        With ``initial_quadrant``: occ = sigmoid(logit(initial) + mlp_out).
        Without              : occ = sigmoid(mlp_out).
        """
        raw = mlp_forward(theta, self.features)[:, 0].reshape(self.N, self.N)
        if self.initial_quadrant is not None:
            logit_init = jnp.log(self.initial_quadrant / (1.0 - self.initial_quadrant))
            occ = jax.nn.sigmoid(logit_init + raw)
        else:
            occ = jax.nn.sigmoid(raw)
        return jnp.where(self.interior_mask, occ, self.border_vals)

    def decode_canvas(self, theta: list[dict]) -> jnp.ndarray:
        """theta → soft (2N, 2N) canvas via squared assembly."""
        return assemble_squared(self.decode_quadrant(theta))

    def binarize(self, theta: list[dict], threshold: float = 0.5) -> np.ndarray:
        """Decode and threshold to a hard {0, 1} (2N, 2N) uint8 canvas."""
        canvas = np.asarray(self.decode_canvas(theta))
        return (canvas > threshold).astype(np.uint8)


def make_cell_neural_field(
    n_fourier: int = 32,
    hidden_size: int = 128,
    n_layers: int = 4,
    seed: int = 42,
    N: int = _QUADRANT_N,
    initial_quadrant: np.ndarray | None = None,
    initial_soft_eps: float = 0.05,
) -> tuple[list[dict], CellNeuralField]:
    """Initialise MLP weights and a CellNeuralField.

    Parameters
    ----------
    initial_quadrant  : (N, N) uint8 or float32 array — nearest-neighbour
                        cell quadrant (top-left N×N of a 2N×2N dataset cell).
                        Binary {0,1} values are soft-clipped to
                        (initial_soft_eps, 1-initial_soft_eps) so the logit
                        is finite.  When provided the MLP is initialised to
                        output ≈ 0 so optimisation starts at the NN cell.
    initial_soft_eps  : clipping margin applied to binary initial_quadrant
                        (default 0.05 → logit ≈ ±2.94).

    Returns
    -------
    theta  : list of {W, b} dicts (MLP weights)
    field  : CellNeuralField with decode_canvas(theta) → (2N, 2N) soft canvas
    """
    features = make_cell_features(N, n_fourier)
    n_in = features.shape[1]

    layer_sizes = [n_in] + [hidden_size] * (n_layers - 1) + [1]
    key = jax.random.PRNGKey(seed)
    theta = init_mlp(key, layer_sizes)
    # Small last layer: MLP output ≈ 0 at init.
    # With initial_quadrant: starts at NN cell.  Without: starts at sigmoid(0)=0.5.
    theta[-1]["W"] = theta[-1]["W"] * 0.01
    theta[-1]["b"] = theta[-1]["b"] * 0.0

    border_vals, interior_mask = make_quadrant_border(_GATE_WIDTH, N)

    init_q_jnp = None
    if initial_quadrant is not None:
        q = np.asarray(initial_quadrant, dtype=np.float32)
        q = np.clip(q, initial_soft_eps, 1.0 - initial_soft_eps)
        init_q_jnp = jnp.array(q)

    nf = CellNeuralField(
        features=features,
        border_vals=border_vals,
        interior_mask=interior_mask,
        N=N,
        initial_quadrant=init_q_jnp,
    )
    return theta, nf


# ── Differentiable periodic-FEM homogenization problem ────────────────


class _DynHomogProblem(Problem):
    """Periodic homogenization: SIMP stiffness + inertia at angular freq omega.

    Class attributes injected via type() before instantiation
    (pattern from multiscale_generation/fem.py):
        _eps_macro_qp : (n_elems, n_qp, 2, 2) — fixed macrostrain for this load case
        _omega        : float — 2π f_star (0 for static)
        _E_solid, _E_void_ratio, _rho_solid, _rho_void_ratio, _nu, _simp_p : floats
        _row_idx      : (n_elems, n_qp) int — quadrature point → canvas row
        _col_idx      : (n_elems, n_qp) int — quadrature point → canvas col
    """

    def custom_init(self) -> None:
        # Dummy occupancy — replaced on first set_params call
        cls = type(self)
        self.internal_vars = [
            jnp.zeros(cls._row_idx.shape, dtype=jnp.float32),
            cls._eps_macro_qp,
        ]

    def set_params(self, canvas: jnp.ndarray) -> None:
        """canvas (2N, 2N) soft occupancy → rebuild occ_field internal var."""
        cls = type(self)
        occ_qp = canvas[cls._row_idx, cls._col_idx]   # differentiable gather
        self.internal_vars = [occ_qp, cls._eps_macro_qp]

    def get_tensor_map(self):
        E_solid = type(self)._E_solid
        E_void_ratio = type(self)._E_void_ratio
        nu = type(self)._nu
        simp_p = type(self)._simp_p

        def stress(u_grad, occ_q, eps_macro_q):
            E_void = E_solid * E_void_ratio
            E = E_void + (E_solid - E_void) * jnp.clip(occ_q, 0.0, 1.0) ** simp_p
            lam = E * nu / ((1 + nu) * (1 - 2 * nu))
            mu = E / (2 * (1 + nu))
            I = jnp.eye(2)
            C = lam * jnp.einsum("ij,kl->ijkl", I, I) + mu * (
                jnp.einsum("ik,jl->ijkl", I, I) + jnp.einsum("il,jk->ijkl", I, I)
            )
            return jnp.einsum("ijkl,kl->ij", C, u_grad + eps_macro_q)

        return stress

    def get_mass_map(self):
        omega = type(self)._omega
        rho_solid = type(self)._rho_solid
        rho_void_ratio = type(self)._rho_void_ratio

        def inertia(u, _x, occ_q, _eps_macro_q):
            rho = rho_solid * (
                rho_void_ratio + (1.0 - rho_void_ratio) * jnp.clip(occ_q, 0.0, 1.0)
            )
            return -omega ** 2 * rho * u

        return inertia


# ── HomogSetup: pre-built FEM infrastructure ──────────────────────────


@dataclass
class HomogSetup:
    """Pre-built FEM infrastructure for differentiable periodic homogenization.

    Built once per (canvas size, f_star, material params).
    ``fwd_preds[i](canvas)`` is differentiable w.r.t. canvas via JAX-FEM's
    implicit-adjoint custom VJP.
    """

    cells: np.ndarray             # (n_elems, n_npe) int
    shape_grads: jnp.ndarray      # (n_elems, n_qp, n_npe, 2)
    JxW: jnp.ndarray              # (n_elems, n_qp)
    row_idx: jnp.ndarray          # (n_elems, n_qp) int — qp → canvas row
    col_idx: jnp.ndarray          # (n_elems, n_qp) int — qp → canvas col
    fwd_preds: list[Any]          # len-4 list of callable(canvas) → sol_list
    problems: list[Any]           # _DynHomogProblem × 4
    load_cases: list[np.ndarray]
    E_solid: float
    E_void_ratio: float
    nu: float
    simp_p: float
    rho_solid: float = RHO_CEMENT
    rho_void_ratio: float = 1e-6


def _qp_to_pixel(
    points: np.ndarray,
    cells_np: np.ndarray,
    canvas_H: int,
    canvas_W: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Map TRI3 element centroids → pixel (row, col) in the (H, W) canvas.

    Image convention: row 0 = top = high y.
    Returns (row_idx, col_idx) each shaped (n_elems, 1) for broadcasting over n_qp.
    """
    centroids = points[cells_np].mean(axis=1)   # (n_elems, 2)
    x, y = centroids[:, 0], centroids[:, 1]
    col = np.clip((x * canvas_W).astype(int), 0, canvas_W - 1)
    row = np.clip(((1.0 - y) * canvas_H).astype(int), 0, canvas_H - 1)
    return row[:, None], col[:, None]   # (n_elems, 1)


def build_homog_setup(
    canvas_N: int = 50,
    f_star: float = 0.0,
    simp_p: float = 1.0,
    E_solid: float = E_CEMENT,
    E_void_ratio: float = 1e-6,
    rho_solid: float = RHO_CEMENT,
    rho_void_ratio: float = 1e-6,
    nu: float = NU,
    solver_opts: dict | None = None,
) -> HomogSetup:
    """Build mesh, 4 periodic homogenization problems, and ad_wrapper callables.

    Parameters
    ----------
    canvas_N      : side length of the square pixel canvas (= 2 * quadrant_N)
    f_star        : excitation frequency [Hz]; 0 → static homogenization
    simp_p        : SIMP penalization exponent
    E_void_ratio  : void/solid Young's modulus ratio (ersatz material)
    rho_void_ratio: void/solid density ratio
    solver_opts   : jax-fem solver options dict; default = {'umfpack_solver': {}}
    """
    if solver_opts is None:
        solver_opts = {"umfpack_solver": {}}

    omega = 2.0 * math.pi * f_star

    points, cells_np = make_structured_tri_mesh(canvas_N)
    mesh = Mesh(points, cells_np, ele_type="TRI3")
    P_mat = build_periodic_pmat(canvas_N, vec=2)

    def corner(pt):
        return jnp.isclose(pt[0], 0.0, atol=1e-5) & jnp.isclose(pt[1], 0.0, atol=1e-5)

    dirichlet_bc_info = [
        [corner, corner],
        [0, 1],
        [lambda p: 0.0, lambda p: 0.0],
    ]

    row_idx_np, col_idx_np = _qp_to_pixel(points, cells_np, canvas_N, canvas_N)
    row_idx = jnp.array(row_idx_np)
    col_idx = jnp.array(col_idx_np)

    fwd_preds: list[Any] = []
    problems: list[Any] = []

    for k, eps_mac in enumerate(_LOAD_CASES):
        eps_qp = np.broadcast_to(eps_mac[None, None], (len(cells_np), 1, 2, 2)).copy()

        ProbCls = type(
            f"_DynHomog_{k}",
            (_DynHomogProblem,),
            {
                "_eps_macro_qp":   jnp.array(eps_qp),
                "_omega":          omega,
                "_E_solid":        E_solid,
                "_E_void_ratio":   E_void_ratio,
                "_rho_solid":      rho_solid,
                "_rho_void_ratio": rho_void_ratio,
                "_nu":             nu,
                "_simp_p":         simp_p,
                "_row_idx":        row_idx,
                "_col_idx":        col_idx,
            },
        )
        prob = ProbCls(
            mesh=mesh,
            vec=2,
            dim=2,
            ele_type="TRI3",
            dirichlet_bc_info=dirichlet_bc_info,
        )
        prob.P_mat = P_mat
        problems.append(prob)
        fwd_preds.append(ad_wrapper(prob, solver_opts, solver_opts))

    fe = problems[0].fes[0]
    return HomogSetup(
        cells=np.array(fe.cells),
        shape_grads=jnp.array(fe.shape_grads),
        JxW=jnp.array(fe.JxW),
        row_idx=row_idx,
        col_idx=col_idx,
        fwd_preds=fwd_preds,
        problems=problems,
        load_cases=_LOAD_CASES,
        E_solid=E_solid,
        E_void_ratio=E_void_ratio,
        nu=nu,
        simp_p=simp_p,
        rho_solid=rho_solid,
        rho_void_ratio=rho_void_ratio,
    )


# ── Differentiable average stress ─────────────────────────────────────


def _avg_stress(
    canvas: jnp.ndarray,
    sol: jnp.ndarray,
    eps_macro: np.ndarray,
    setup: HomogSetup,
) -> jnp.ndarray:
    """Volume-averaged stress (2×2) for one homogenization load case.

    Differentiable w.r.t. canvas via two paths:
    - Direct material path: canvas → occ_qp → C(occ) → sigma
    - Adjoint path:         canvas → sol (via ad_wrapper VJP) → u_grad → sigma
    """
    occ_qp = canvas[setup.row_idx, setup.col_idx]   # (n_elems, n_qp)

    E_void = setup.E_solid * setup.E_void_ratio
    E = E_void + (setup.E_solid - E_void) * jnp.clip(occ_qp, 0.0, 1.0) ** setup.simp_p
    nu = setup.nu
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))

    cell_sol = sol[setup.cells]                                            # (n_elems, n_npe, 2)
    u_grads = jnp.einsum("cni,cqnj->cqij", cell_sol, setup.shape_grads)   # (n_elems, n_qp, 2, 2)
    total_grads = u_grads + eps_macro[None, None]

    tr_g = total_grads[..., 0, 0] + total_grads[..., 1, 1]
    sigma = (
        lam[..., None, None] * tr_g[..., None, None] * jnp.eye(2)
        + mu[..., None, None] * (total_grads + jnp.swapaxes(total_grads, -2, -1))
    )

    total_area = jnp.sum(setup.JxW)
    return jnp.sum(sigma * setup.JxW[..., None, None], axis=(0, 1)) / total_area


# ── flat4 and loss ────────────────────────────────────────────────────


def compute_flat4(canvas: jnp.ndarray, setup: HomogSetup) -> jnp.ndarray:
    """Run 4 periodic FEM solves → flat4 = [C11, C22, C12, C66].

    Fully differentiable w.r.t. canvas via JAX-FEM's ad_wrapper adjoint.
    """
    C_cols = []
    for fwd_pred, eps_mac in zip(setup.fwd_preds, setup.load_cases):
        sol = fwd_pred(canvas)[0]                                  # (n_nodes, 2)
        sig = _avg_stress(canvas, sol, eps_mac, setup)             # (2, 2)
        C_cols.append(jnp.stack([sig[0, 0], sig[1, 1], sig[0, 1], sig[1, 0]]))

    C_eff = jnp.stack(C_cols, axis=1)   # (4, 4): rows=stress, cols=load_case
    C11 = C_eff[0, 0]
    C22 = C_eff[1, 1]
    C12 = 0.5 * (C_eff[0, 1] + C_eff[1, 0])
    C66 = 0.5 * (C_eff[2, 2] + C_eff[3, 3])
    return jnp.stack([C11, C22, C12, C66])


def compute_rho_eff(canvas: jnp.ndarray, setup: HomogSetup) -> jnp.ndarray:
    """Effective density from soft canvas: rho_solid * mean(SIMP_rho_fraction).

    Uses the same SIMP-linear density model as the FEM mass matrix, so the
    loss gradient is consistent with the stiffness gradient path.
    Differentiable w.r.t. canvas.
    """
    occ = jnp.clip(canvas, 0.0, 1.0)
    rho_frac = setup.rho_void_ratio + (1.0 - setup.rho_void_ratio) * occ
    return setup.rho_solid * jnp.mean(rho_frac)


def flat4_loss(
    flat4: jnp.ndarray,
    target: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Weighted relative L2: sum_i w_i ((f_i - t_i) / |t_i|)^2."""
    rel = (flat4 - target) / (jnp.abs(target) + 1e-30)
    return jnp.sum(weights * rel ** 2)


def gate_connectivity_loss(
    canvas: jnp.ndarray,
    n_steps: int = 200,
    gate_width: int = _GATE_WIDTH,
    quadrant_N: int = _QUADRANT_N,
    sharpness: float = 10.0,
) -> jnp.ndarray:
    """Strict differentiable gate connectivity loss via sharpened multi-source flood.

    Sharpens occupancy with sigmoid((occ - 0.5) * sharpness) so pixels below
    0.5 nearly block the flood and pixels above 0.5 nearly pass it — much
    closer to binary connectivity than raw soft occupancy.

    Floods simultaneously from all 4 gate groups (top, bottom, left, right)
    as separate channels and checks all 12 pairwise reachability values.
    Returns a value in [0, 3]: 0 when every gate pair is fully reachable.
    """
    canvas = canvas.astype(jnp.float32)
    H, W = canvas.shape
    N = quadrant_N
    gs = (N - gate_width) // 2
    ge = gs + gate_width

    # Sharpen: sub-0.5 pixels nearly block the flood; super-0.5 nearly pass.
    occ = jax.nn.sigmoid((canvas - 0.5) * sharpness)   # (H, W)

    # Build 4 source masks: top, bottom, left, right — each activates both
    # gate slots on that edge (one per quadrant, e.g. left has rows gs:ge and H-ge:H-gs).
    top_s = np.zeros((H, W), dtype=np.float32)
    top_s[0, gs:ge] = 1.0
    top_s[0, W - ge : W - gs] = 1.0

    bot_s = np.zeros((H, W), dtype=np.float32)
    bot_s[H - 1, gs:ge] = 1.0
    bot_s[H - 1, W - ge : W - gs] = 1.0

    lft_s = np.zeros((H, W), dtype=np.float32)
    lft_s[gs:ge, 0] = 1.0
    lft_s[H - ge : H - gs, 0] = 1.0

    rgt_s = np.zeros((H, W), dtype=np.float32)
    rgt_s[gs:ge, W - 1] = 1.0
    rgt_s[H - ge : H - gs, W - 1] = 1.0

    sources = jnp.array(np.stack([top_s, bot_s, lft_s, rgt_s]))   # (4, H, W)

    @jax.checkpoint
    def _step(reach: jnp.ndarray, _) -> tuple[jnp.ndarray, None]:
        # reach: (4, H, W).  Zero-pad to prevent wrap-around paths.
        nbr = jnp.maximum(
            jnp.maximum(
                jnp.pad(reach[:, :-1, :], ((0, 0), (1, 0), (0, 0))),  # from above
                jnp.pad(reach[:, 1:, :],  ((0, 0), (0, 1), (0, 0))),  # from below
            ),
            jnp.maximum(
                jnp.pad(reach[:, :, :-1], ((0, 0), (0, 0), (1, 0))),  # from left
                jnp.pad(reach[:, :, 1:],  ((0, 0), (0, 0), (0, 1))),  # from right
            ),
        )
        return jnp.maximum(sources, occ[None] * nbr), None

    reach, _ = jax.lax.scan(_step, sources, None, length=n_steps)   # (4, H, W)

    def _gate_reach(reach_k: jnp.ndarray, edge: int) -> jnp.ndarray:
        """Mean reach at both gate slots of the given edge (0=top,1=bot,2=left,3=right)."""
        if edge == 0:  # top
            return 0.5 * (reach_k[0, gs:ge].mean() + reach_k[0, W - ge : W - gs].mean())
        elif edge == 1:  # bottom
            return 0.5 * (reach_k[H - 1, gs:ge].mean() + reach_k[H - 1, W - ge : W - gs].mean())
        elif edge == 2:  # left
            return 0.5 * (reach_k[gs:ge, 0].mean() + reach_k[H - ge : H - gs, 0].mean())
        else:  # right
            return 0.5 * (reach_k[gs:ge, W - 1].mean() + reach_k[H - ge : H - gs, W - 1].mean())

    # Sum (1 - reach) over all 12 ordered (source, target) pairs; divide by 4
    # so the range stays [0, 3] and weight_conn doesn't need retuning.
    loss = jnp.zeros(())
    for src in range(4):
        for tgt in range(4):
            if tgt != src:
                loss = loss + (1.0 - _gate_reach(reach[src], tgt))
    return loss / 4.0


# ── Optimization result ────────────────────────────────────────────────


@dataclass
class CellDesignResult:
    """Result of cell inverse-design optimization."""

    theta: list[dict]
    best_theta: list[dict]         # weights at lowest loss
    opt_state: AdamState
    loss_history: list[float] = field(default_factory=list)
    flat4_history: list[list[float]] = field(default_factory=list)
    rho_history: list[float] = field(default_factory=list)
    conn_history: list[float] = field(default_factory=list)


# ── Optimization loop ──────────────────────────────────────────────────


def run_cell_design(
    neural_field: CellNeuralField,
    setup: HomogSetup,
    target_flat4: jnp.ndarray | np.ndarray,
    theta_init: list[dict],
    target_rho: float | None = None,
    weights: jnp.ndarray | None = None,
    weight_rho: float = 1.0,
    weight_conn: float = 10.0,
    conn_steps: int = 200,
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "cosine",
    opt_state_init: AdamState | None = None,
    step_callback=None,
    tol: float = 0.001,
) -> CellDesignResult:
    """Optimize MLP weights to produce a cell matching ``target_flat4`` and ``target_rho``.

    Stops early when the maximum relative error across all components (flat4 and
    rho when provided) falls below ``tol`` (default 0.001 = 0.1 %).

    Parameters
    ----------
    neural_field    : CellNeuralField (from make_cell_neural_field)
    setup           : HomogSetup (from build_homog_setup)
    target_flat4    : (4,) array — [C11, C22, C12, C66] targets
    theta_init      : initial MLP weights
    target_rho      : target effective density [kg/m³]; None → rho not penalized
    weights         : (4,) stiffness loss weights; None → uniform 0.25 each
    weight_rho      : scalar weight for the rho L2 term (default 1.0)
    n_iters         : maximum number of Adam steps (default 100)
    lr, lr_end      : initial and final learning rate (None → constant lr)
    lr_schedule     : "cosine" or "linear"
    opt_state_init  : AdamState for warm restart; None → fresh
    step_callback   : optional callable(step, loss, flat4, rho, theta, opt_state)
    tol             : early-stop threshold on max relative error (default 0.001)
    weight_conn     : weight for gate-to-gate connectivity loss (default 10.0); 0 → disabled
    conn_steps      : flood iterations for connectivity loss (default 100; ≥ grid diameter)
    """
    if weights is None:
        weights = jnp.ones(4) * 0.25
    weights = jnp.asarray(weights)
    target = jnp.asarray(target_flat4, dtype=jnp.float32)
    use_rho = target_rho is not None
    t_rho = float(target_rho) if use_rho else 0.0

    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = opt_state_init if opt_state_init is not None else adam_init(theta)

    use_conn = weight_conn > 0.0

    def _loss_with_aux(t):
        canvas = neural_field.decode_canvas(t)
        flat4 = compute_flat4(canvas, setup)
        rho = compute_rho_eff(canvas, setup)
        conn = (gate_connectivity_loss(canvas, n_steps=conn_steps)
                if use_conn else jnp.zeros(()))
        L = flat4_loss(flat4, target, weights)
        if use_rho:
            rel_rho = (rho - t_rho) / (jnp.abs(t_rho) + 1e-30)
            L = L + weight_rho * rel_rho ** 2
        if use_conn:
            L = L + weight_conn * conn
        return L, (flat4, rho, conn)

    loss_and_grad = jax.value_and_grad(_loss_with_aux, has_aux=True)

    loss_history: list[float] = []
    flat4_history: list[list[float]] = []
    rho_history: list[float] = []
    conn_history: list[float] = []
    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)

    target_str = "[" + ", ".join(f"{float(v):.3e}" for v in target) + "]"
    rho_str = f"  rho_target={t_rho:.1f}" if use_rho else ""
    print(f"  Cell inverse design | target flat4 = {target_str}{rho_str}")
    conn_str = f"  weight_conn={weight_conn}  conn_steps={conn_steps}" if use_conn else ""
    print(f"  weights = {[float(w) for w in weights]}  weight_rho={weight_rho}{conn_str}  iters={n_iters}  lr={lr}")

    for step in range(n_iters):
        t_frac = step / max(n_iters - 1, 1)
        if lr_end is None:
            cur_lr = lr
        elif lr_schedule == "cosine":
            cur_lr = lr_end + 0.5 * (lr - lr_end) * (1.0 + math.cos(math.pi * t_frac))
        else:
            cur_lr = lr + (lr_end - lr) * t_frac

        (loss_val, (flat4, rho, conn)), grads = loss_and_grad(theta)
        loss_float = float(loss_val)
        flat4_list = [float(v) for v in flat4]
        rho_float = float(rho)
        conn_float = float(conn)

        loss_history.append(loss_float)
        flat4_history.append(flat4_list)
        rho_history.append(rho_float)
        conn_history.append(conn_float)

        grad_norm = float(
            jnp.sqrt(sum(jnp.sum(l["W"] ** 2) + jnp.sum(l["b"] ** 2) for l in grads))
        )
        flat4_str = "[" + ", ".join(f"{v:.3e}" for v in flat4_list) + "]"
        rho_print = f"  rho={rho_float:.1f}" if use_rho else ""
        conn_print = f"  conn={conn_float:.3f}" if use_conn else ""
        print(
            f"  Step {step:4d} | loss={loss_float:.4e}"
            f"  flat4={flat4_str}"
            f"{rho_print}"
            f"{conn_print}"
            f"  lr={cur_lr:.2e}  |grad|={grad_norm:.3e}"
        )

        if loss_float < best_loss:
            best_loss = loss_float
            best_theta = jax.tree.map(jnp.copy, theta)

        if step_callback is not None:
            step_callback(step, loss_float, flat4_list, rho_float, theta, opt_state)

        # Early stopping: check max relative error across all components
        rel_err_flat4 = np.abs((np.array(flat4_list) - np.array(target_flat4))
                               / (np.abs(np.array(target_flat4)) + 1e-30))
        max_rel_err = float(rel_err_flat4.max())
        if use_rho:
            rel_err_rho = abs((rho_float - t_rho) / (abs(t_rho) + 1e-30))
            max_rel_err = max(max_rel_err, rel_err_rho)
        if max_rel_err < tol:
            print(f"  Early stop at step {step}: max rel err {max_rel_err:.2e} < tol {tol:.2e}")
            break

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    return CellDesignResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        loss_history=loss_history,
        flat4_history=flat4_history,
        rho_history=rho_history,
        conn_history=conn_history,
    )


# ── Weight I/O (delegates to neural_reparam helpers) ──────────────────


def save_design(
    path: str,
    theta: list[dict],
    opt_state: AdamState | None = None,
    **extra: np.ndarray,
) -> None:
    """Save MLP weights, optional Adam state, and arbitrary extra arrays.

    Extra keyword arguments (e.g. target_flat4=..., best_flat4=...) are
    stored alongside the weights in the same .npz file.
    """
    arrays: dict[str, np.ndarray] = {}
    for i, layer in enumerate(theta):
        arrays[f"W_{i}"] = np.asarray(layer["W"])
        arrays[f"b_{i}"] = np.asarray(layer["b"])
    arrays["n_layers"] = np.array(len(theta))
    if opt_state is not None:
        arrays["adam_t"] = np.array(opt_state.t)
        for i, layer in enumerate(opt_state.m):
            arrays[f"adam_m_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_m_b_{i}"] = np.asarray(layer["b"])
        for i, layer in enumerate(opt_state.v):
            arrays[f"adam_v_W_{i}"] = np.asarray(layer["W"])
            arrays[f"adam_v_b_{i}"] = np.asarray(layer["b"])
    for k, v in extra.items():
        arrays[k] = np.asarray(v)
    np.savez(path, **arrays)


def load_design(path: str) -> tuple[list[dict], AdamState | None, dict]:
    """Load MLP weights, optional Adam state, and extra metadata from .npz.

    Returns (theta, opt_state, meta) where meta holds any arrays beyond the
    standard weight / Adam keys (e.g. target_flat4, best_flat4).
    """
    theta, opt_state = load_theta(path)
    npz_path = path if path.endswith(".npz") else path + ".npz"
    data = np.load(npz_path)
    _weight_prefixes = ("W_", "b_", "adam_m_W_", "adam_m_b_", "adam_v_W_", "adam_v_b_")
    _scalar_keys = {"n_layers", "adam_t"}
    meta = {
        k: data[k]
        for k in data.files
        if k not in _scalar_keys and not any(k.startswith(p) for p in _weight_prefixes)
    }
    return theta, opt_state, meta
