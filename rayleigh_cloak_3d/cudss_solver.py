"""cuDSS (GPU direct sparse) solver for the frequency-domain cloak.

Dimension-agnostic: it takes an assembled PETSc matrix + RHS from jax-fem and
hands the CSR to cuDSS, so the same module drives both the 3D design solve
(``rayleigh_cloak_3d``, vec=6) and the 2D pixel-level validation
(``design-validation/run_validation.py``, vec=4). The only place the
dimension enters is ``cfg['vec']``, read by :func:`_signflip`.

Why
---
- MUMPS does not fit the factor on a 32 GB host; AMG / BiCGStab diverge
  because the operator is indefinite (K − ω²M + Rayleigh-damping PML).
- cuDSS is a modern GPU sparse direct solver that tolerates indefinite
  systems and keeps the factor in VRAM. An A6000 (48 GB) handles the
  problem up to roughly n_phys = 28 in this setup.
- On the 2D pixel validation it turned the factorisation from the cost
  centre into a rounding error: 4.0M DOF / 176M nnz factorised in 1.8 s,
  whole validation 89 s, versus ~180-220 s and 23-28 GB peak host RSS for a
  third the node count under the CPU path.

Design
------
One persistent :class:`DirectSolver` **per sparsity pattern**. jax-fem's
row-elimination Dirichlet BCs keep the pattern fixed across iterations, so
after a one-time ``plan()`` (reorder + symbolic) every subsequent iter pays
only the numeric ``factorize()`` + ``solve()``.

Forward and adjoint are **separate solvers** (keyed by their own CSR
patterns). The adjoint cache is populated on the first adjoint call and
then refactored cheaply every step. No special handling of the 2×2 real
block structure — we trade a 2× per-iter factor cost for code simplicity.

Knobs (via ``solver_options['cudss']``)
--------------------------------------
    hybrid_memory : bool = False
        Spill factor to pinned host memory. Pushes the problem-size ceiling
        up by roughly 1.5× at the cost of 1.5-2× solve time.
    exclusive_gpu : bool = False
        Hold at most ONE factor in VRAM, freeing the other pattern's solver
        first. Makes the peak ``max(fwd, adj)`` instead of ``fwd + adj``, at
        the cost of re-planning each call (~10 s symbolic). This is what lets
        the adjoint run at 1 element/pixel at all.
    signflip_adjoint : bool = False
        Reuse the forward factor for ``Aᵀ`` via ``J A J = Aᵀ``. Valid for the
        3D vec=6 operator; NOT valid for the 2D vec=4 problem — see
        :func:`_adjoint_via_signflip`. Off by default.
    vec : int = 4
        DOFs per node, used only to build ``J``. Set 6 for the 3D solve.
    verify_signflip : bool = False
        Check the identity numerically on first use (materialises a second CSR
        copy of ``Aᵀ`` — expensive; prefer ``tools/verify_signflip.py``).
    verbose : bool = True
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp


# --- locate the cuDSS multithreading layer once ---------------------------
def _find_mt_lib() -> str:
    """Full path to libcudss_mtlayer_gomp.so.*; falls back to soname."""
    candidates: list[str] = []
    try:
        import nvidia  # type: ignore
        for p in nvidia.__path__:
            candidates += glob.glob(os.path.join(p, "cu12", "lib",
                                                  "libcudss_mtlayer_gomp.so*"))
    except Exception:
        pass
    candidates.sort(reverse=True)  # prefer "...so.0" over "...so"
    for c in candidates:
        if os.path.exists(c):
            return c
    return "libcudss_mtlayer_gomp.so.0"


_MT_LIB = _find_mt_lib()


# --- persistent state per sparsity pattern --------------------------------
@dataclass
class _CachedSolver:
    solver: object        # nvmath DirectSolver
    A_d: object           # cupyx csr_matrix (values updated in place)
    b_d: object           # cupy ndarray (RHS buffer, reused)
    n: int
    nnz: int
    plan_time: float


_CACHE: Dict[Tuple[int, int, int], _CachedSolver] = {}


def _pattern_key(A: sp.csr_matrix) -> Tuple[int, int, int]:
    # Lightweight hash on a prefix of the CSR pattern.
    h = hash(A.indptr.tobytes()[:256] + A.indices.tobytes()[:256])
    return (A.shape[0], A.nnz, h)


def _to_csr(A_pet) -> sp.csr_matrix:
    n = A_pet.getSize()[0]
    indptr, indices, data = A_pet.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=(n, n))


def _get_or_create(A: sp.csr_matrix, cfg: dict, label: str) -> _CachedSolver:
    import cupy as cp
    import cupyx.scipy.sparse as csp
    from nvmath.sparse.advanced import (
        DirectSolver,
        DirectSolverOptions,
        ExecutionCUDA,
        HybridMemoryModeOptions,
    )

    key = _pattern_key(A)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    t0 = time.perf_counter()
    A_d = csp.csr_matrix(
        (cp.asarray(A.data),
         cp.asarray(A.indices),
         cp.asarray(A.indptr)),
        shape=A.shape,
    )
    b_d = cp.zeros(A.shape[0], dtype=A.data.dtype)

    opts = DirectSolverOptions(multithreading_lib=_MT_LIB)
    exec_cfg = None
    if cfg.get("hybrid_memory", False):
        exec_cfg = ExecutionCUDA(
            device_id=0,
            hybrid_memory_mode_options=HybridMemoryModeOptions(hybrid_memory_mode=True),
        )

    solver = DirectSolver(A_d, b_d, options=opts, execution=exec_cfg)
    solver.plan()
    cp.cuda.runtime.deviceSynchronize()
    t_plan = time.perf_counter() - t0

    cached = _CachedSolver(solver=solver, A_d=A_d, b_d=b_d,
                            n=A.shape[0], nnz=A.nnz, plan_time=t_plan)
    _CACHE[key] = cached
    if cfg.get("verbose", True):
        print(f"[cuDSS] {label}: planned n={cached.n:,} nnz={cached.nnz:,} "
              f"plan={t_plan:.2f}s (1-time)", flush=True)
    return cached


_LAST_FWD: Optional[_CachedSolver] = None


def _signflip(n_dof: int, vec: int = 4):
    """Diagonal J: +1 on the Re DOFs, -1 on the Im DOFs of each node block."""
    import cupy as cp
    half = vec // 2
    per_node = np.concatenate([np.ones(half), -np.ones(vec - half)])
    d = np.tile(per_node, n_dof // vec + 1)[:n_dof]
    return cp.asarray(d)


def _adjoint_via_signflip(A_pet, b, cfg):
    """Solve ``A^T y = b`` reusing the FORWARD factor, via ``J A J = A^T``.

    Our FE discretisation stores the complex system as a real one with Re and
    Im DOFs interleaved per node; the operator then satisfies ``J A J = A^T``
    with J flipping the sign of the Im DOFs. So

        A^T y = b   <=>   A x = J b,  y = J x

    which means the adjoint needs no second factorisation and no second cuDSS
    solver. That matters twice over: it halves adjoint wall time, and it halves
    GPU memory — with a separate ``A^T`` solver the two factors together
    overflow the A6000 at ~4.9M DOF (cuDSS ALLOC_FAILED), which is what blocks
    gradient-based optimisation at 1 element/pixel.

    Verified numerically on the first call against the ``A^T`` jax-fem actually
    passes; falls back to an independent solver if the identity does not hold.
    """
    import cupy as cp
    global _LAST_FWD
    if _LAST_FWD is None:
        return None
    fwd = _LAST_FWD
    n = fwd.n
    if A_pet is not None and _LAST_FWD.n != n:
        return None
    J = _signflip(n, int(cfg.get("vec", 4)))

    # Verification materialises a second CSR copy of A^T (~2.7 GB at 4.9M DOF /
    # 223M nnz) at the exact moment the AD tape is also live, which is enough to
    # push a 31 GB host over. Off by default; verify once on a small mesh
    # instead (tools/verify_signflip.py) — the identity is a property of the
    # discretisation, not of the mesh size.
    if cfg.get("verify_signflip", False) and not getattr(fwd, "_signflip_ok", False):
        AT = _to_csr(A_pet)
        rng = np.random.default_rng(0)
        v = rng.standard_normal(n)
        AT_v = AT @ v
        v_d = cp.asarray(v)
        JAJ_v = cp.asnumpy(J * (fwd.A_d @ (J * v_d)))
        err = np.linalg.norm(AT_v - JAJ_v) / max(np.linalg.norm(AT_v), 1e-30)
        if err > 1e-8:
            print(f"[cuDSS] sign-flip adjoint identity FAILED (rel err {err:.2e}); "
                  f"falling back to an independent A^T factorisation", flush=True)
            return None
        fwd._signflip_ok = True
        if cfg.get("verbose", True):
            print(f"[cuDSS] sign-flip adjoint verified (rel err {err:.2e}) — "
                  f"reusing the forward factor, no second factorisation",
                  flush=True)

    fwd.b_d[:] = J * cp.asarray(np.asarray(b))
    x_d = fwd.solver.solve()
    return cp.asnumpy(J * x_d)


def _vram() -> str:
    """``vram=<in-use>/<total> GB`` from the driver, or '' if unavailable.

    The factor lives in VRAM, so this is the number that decides whether the
    next refinement level fits. ``mem_get_info`` reports the whole device, which
    is what we want: it counts cuDSS's own allocations, which sit outside CuPy's
    pool and so are invisible to ``mempool.used_bytes()``.
    """
    try:
        import cupy as cp
        free, total = cp.cuda.runtime.memGetInfo()
        return f"vram={(total - free) / 2**30:.1f}/{total / 2**30:.1f}GB"
    except Exception:
        return ""


def cudss_solver(A_pet, b, x0, solver_options):
    """Linear solver for jax-fem. Plug in via ``solver_options['custom_solver']``.

    Used for both forward and adjoint. jax-fem passes ``A`` on forward and
    ``A.transpose()`` on adjoint. By default the adjoint reuses the forward
    factor through the sign-flip identity (see ``_adjoint_via_signflip``); set
    ``cudss.signflip_adjoint = False`` to give it its own persistent solver
    instead, which costs a second factorisation and a second factor in VRAM.
    """
    import cupy as cp
    global _LAST_FWD

    cfg = solver_options.get("cudss", {})
    is_adj = bool(solver_options.get("_is_adjoint", False))
    label = "adj" if is_adj else "fwd"

    t_all0 = time.perf_counter()

    # NOTE: signflip_adjoint defaults OFF. The identity J A J = A^T holds for the
    # 3-D vec=6 operator (bench/adjoint_trick.py) but NOT for this 2-D vec=4
    # problem: tools/verify_signflip.py measures a relative error of 4.6e-3,
    # almost certainly because jax-fem's row-elimination Dirichlet BCs zero rows
    # without zeroing the matching columns, so A is not exactly
    # complex-symmetric. That error would corrupt every gradient. Use
    # `exclusive_gpu` below to fit forward+adjoint on one card instead.
    if is_adj and cfg.get("signflip_adjoint", False):
        y = _adjoint_via_signflip(A_pet, b, cfg)
        if y is not None:
            if cfg.get("verbose", True):
                print(f"[cuDSS] adj via sign-flip (forward factor reused) "
                      f"total={time.perf_counter()-t_all0:.2f}s", flush=True)
            return y

    A = _to_csr(A_pet)

    # Exclusive mode: hold at most ONE factor in VRAM. Forward and adjoint have
    # different sparsity patterns, so the default cache keeps both alive and the
    # two factors together overflow the card (measured: the forward alone peaks
    # at ~30.8 GB of 46 GB at 4.86M DOF, and the adjoint then dies with
    # ALLOC_FAILED). Freeing the other one first makes the peak max(fwd, adj)
    # instead of fwd+adj, at the cost of re-planning each call (~10 s symbolic).
    # That is what lets the adjoint run at 1 element/pixel at all.
    if cfg.get("exclusive_gpu", False):
        key_now = _pattern_key(A)
        for k in list(_CACHE):
            if k != key_now:
                try:
                    _CACHE[k].solver.free()
                except Exception:
                    pass
                del _CACHE[k]

    t0 = time.perf_counter()
    cached = _get_or_create(A, cfg, label)
    t_get = time.perf_counter() - t0
    if not is_adj:
        _LAST_FWD = cached

    t0 = time.perf_counter()
    cached.A_d.data[:] = cp.asarray(A.data)
    cached.b_d[:] = cp.asarray(np.asarray(b))
    cp.cuda.runtime.deviceSynchronize()
    t_upload = time.perf_counter() - t0

    t0 = time.perf_counter()
    cached.solver.factorize()
    cp.cuda.runtime.deviceSynchronize()
    t_fact = time.perf_counter() - t0

    t0 = time.perf_counter()
    x_d = cached.solver.solve()
    cp.cuda.runtime.deviceSynchronize()
    t_solve = time.perf_counter() - t0

    x = cp.asnumpy(x_d)

    if cfg.get("verbose", True):
        b_arr = np.asarray(b)
        r = A @ x - b_arr
        rel = np.linalg.norm(r) / max(np.linalg.norm(b_arr), 1e-30)
        total = time.perf_counter() - t_all0
        print(f"[cuDSS] {label} cache={t_get:.2f}s up={t_upload:.2f}s "
              f"fact={t_fact:.2f}s solve={t_solve:.2f}s total={total:.2f}s "
              f"rel_res={rel:.1e} {_vram()}", flush=True)

    return x


def make_forward_and_adjoint(base_cfg: dict | None = None):
    """Return (fwd_opts, adj_opts) dicts for ``ad_wrapper``."""
    cfg = dict(base_cfg or {})
    fwd_opts = {"custom_solver": cudss_solver, "cudss": cfg}
    adj_opts = {"custom_solver": cudss_solver, "cudss": cfg, "_is_adjoint": True}
    return fwd_opts, adj_opts


def clear_cache() -> None:
    """Free all cached cuDSS solvers (e.g. when changing mesh)."""
    for c in _CACHE.values():
        try:
            c.solver.free()
        except Exception:
            pass
    _CACHE.clear()
