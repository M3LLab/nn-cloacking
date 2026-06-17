"""Compute effective stiffness for the bulk-generated CA dataset and save to HDF5.

Reads ``cells.npy`` and ``live_fractions.npy`` produced by
``dataset.cellular_chiral.bulk_generate``, runs FEM homogenization on every
unique cell, and writes the four independent orthotropic (D2-class) stiffness
parameters (C11, C22, C12, C66) plus effective density, the full augmented
Voigt stiffness, and the binary geometry to one HDF5 file.

The squared (mirror) assembly has two orthogonal mirror axes -- point group D2,
NOT D4 -- so the two axial stiffnesses C11 (x) and C22 (y) differ in general
(median C11/C22 ratio ~2x in ca_bulk_squared). They are stored separately; do
NOT assume C11 == C22.

The fields ``lambda_`` and ``mu`` are also written for backward compatibility
with older readers; they alias ``C12`` and ``C66`` respectively. Be aware
those names assume isotropy that does NOT hold for these cells -- use
``C11/C22/C12/C66`` directly when you need the full elastic response.

Homogenizer
-----------

Two paths, selected with ``--homog`` (provenance recorded in the HDF5 attrs
``homog_method`` / ``homog_ele_type`` / ``homog_mesh_N`` / ``homog_elem_per_pixel``):

* ``hifi`` (DEFAULT): quadratic **TRI6** elements on a configurable ``--mesh-N``
  grid (default 100 = 2 elements/pixel for a 50x50 cell), with a single
  factorization shared across the 4 load cases. This is the accurate path.
* ``legacy``: the original linear **TRI3 at exactly 1 element/pixel**. Kept for
  back-compat / reproducing the historical dataset. A mesh-convergence study
  (``results/mesh_artifact/SUMMARY.md``) showed this **over-predicts stiffness**
  by ~12%% (bulky cells) up to **>130%% (thin/low-vf cells)** because 1-pixel-wide
  ligaments are a single linear element wide and cannot bend.

Identical and "really similar" geometries are filtered out: a cell is skipped
if its raw bytes (exact match) or its block-pooled fingerprint (near-duplicate
match) has already been seen.

At the end the script prints how many designs were kept per live_fraction bin
and how many were ignored as duplicates / FEM failures.

Usage
-----

    # Default: hi-fi TRI6 @ 2 elements/pixel
    python -m dataset.cellular_chiral.bulk_stiffness \
        -i output/ca_bulk_squared \
        -o output/ca_bulk_squared/stiffness_tri6.h5

    # Reproduce the historical dataset (legacy linear TRI3 @ 1 element/pixel)
    python -m dataset.cellular_chiral.bulk_stiffness --homog legacy \
        -o output/ca_bulk_squared/stiffness.h5

    # Higher accuracy for thin cells (4 elements/pixel) -- slower
    python -m dataset.cellular_chiral.bulk_stiffness --mesh-N 200

    # Process a slice (useful when 2M is too large to chew through at once)
    python -m dataset.cellular_chiral.bulk_stiffness --start 0 --num 50000

    # Resume an interrupted run, write to the same file
    python -m dataset.cellular_chiral.bulk_stiffness --resume
"""
from __future__ import annotations

import os as _os
import argparse
import contextlib
import io
import logging
import multiprocessing as mp
import signal
import sys
import time
from pathlib import Path

import functools

import h5py
import numpy as np
from tqdm import tqdm

# jax-fem prints a banner on import; keep stdout clean
with contextlib.redirect_stdout(io.StringIO()):
    import jax.numpy as jnp
    from jax_fem.generate_mesh import Mesh
    from jax_fem.solver import solver as jax_fem_solver

    from ..stiffness.calc_fem import (
        E_CEMENT,
        NU,
        RHO_CEMENT,
        HomogenizationProblem,
        assign_material,
        build_periodic_pmat,
        compute_average_stress,
        make_structured_tri_mesh,
    )
    from ..stiffness.calc_fem_hifi import compute_stiffness_hifi, estimate_and_warn

logging.getLogger("jax_fem").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Stiffness from in-memory array
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _get_mesh(N: int):
    points, cells = make_structured_tri_mesh(N)
    mesh = Mesh(points, cells, ele_type="TRI3")
    P_mat = build_periodic_pmat(N, vec=2)
    return points, cells, mesh, P_mat


_LOAD_CASES = [
    np.array([[1.0, 0.0], [0.0, 0.0]]),  # e11
    np.array([[0.0, 0.0], [0.0, 1.0]]),  # e22
    np.array([[0.0, 1.0], [0.0, 0.0]]),  # e12
    np.array([[0.0, 0.0], [1.0, 0.0]]),  # e21
]


def _compute_stiffness(
    pixel_image: np.ndarray,
    rho_solid: float = RHO_CEMENT,
) -> tuple[np.ndarray, float, float]:
    """Run periodic FEM homogenization on a single (N, N) pixel image.

    Mirrors ``dataset.stiffness.calc_fem.compute_effective_stiffness`` but
    accepts the array directly so we don't round-trip through a temp file.
    Returns (C_eff (4x4 augmented Voigt), rho_eff, volume_fraction).

    ``rho_solid`` is the density of the solid phase used to compute the
    effective density ``rho_eff = vf * rho_solid``. Defaults to RHO_CEMENT
    for back-compat with the original pipeline.
    """
    N = pixel_image.shape[0]
    assert pixel_image.shape == (N, N), f"expected square image, got {pixel_image.shape}"
    vf = float(pixel_image.astype(np.float64).mean())

    points, cells, mesh, P_mat = _get_mesh(N)
    E_field = assign_material(pixel_image, points, cells, num_quads=1)

    def corner(point):
        return jnp.isclose(point[0], 0.0, atol=1e-5) & jnp.isclose(point[1], 0.0, atol=1e-5)

    dirichlet_bc_info = [[corner, corner], [0, 1], [lambda p: 0.0, lambda p: 0.0]]

    C_eff = np.zeros((4, 4))
    for col, eps_macro in enumerate(_LOAD_CASES):
        HomogenizationProblem._eps_macro = eps_macro
        HomogenizationProblem._E_field = E_field
        problem = HomogenizationProblem(
            mesh=mesh,
            vec=2,
            dim=2,
            ele_type="TRI3",
            dirichlet_bc_info=dirichlet_bc_info,
        )
        problem.P_mat = P_mat
        sol_list = jax_fem_solver(problem, solver_options={"umfpack_solver": {}})
        sol = sol_list[0]
        avg_stress = compute_average_stress(problem, sol, eps_macro, E_field)
        C_eff[0, col] = float(avg_stress[0, 0])
        C_eff[1, col] = float(avg_stress[1, 1])
        C_eff[2, col] = float(avg_stress[0, 1])
        C_eff[3, col] = float(avg_stress[1, 0])

    rho_eff = vf * rho_solid
    return C_eff, rho_eff, vf


def _stiffness(
    pixel_image: np.ndarray,
    rho_solid: float,
    method: str,
    mesh_N: int,
    ele_type: str,
) -> tuple[np.ndarray, float, float]:
    """Dispatch to the hi-fi (TRI6/TRI3, factor-once, mesh res N) or the legacy
    (TRI3 @ 1-element-per-pixel) homogenizer. Both return (C_eff 4x4, rho, vf)."""
    if method == "legacy":
        return _compute_stiffness(pixel_image, rho_solid=rho_solid)
    return compute_stiffness_hifi(
        pixel_image, N=mesh_N, ele_type=ele_type, rho_solid=rho_solid
    )


def _ortho_params_from_C(C: np.ndarray) -> tuple[float, float, float, float]:
    """Extract the four independent orthotropic (D2-class) stiffness constants.

    Augmented Voigt order is [sigma_11, sigma_22, sigma_12, sigma_21] vs
    [e_11, e_22, e_12, e_21]. The squared (mirror) assembly is symmetric about
    both the x and y center lines -- point group D2 -- which zeroes the
    axial<->shear coupling but does NOT force C11 == C22. The four independent
    entries are:
        C11 = C[0,0]            (x-axial)
        C22 = C[1,1]            (y-axial)  -- distinct from C11 for D2
        C12 = C[0,1] = C[1,0]   (lateral coupling; equal by major symmetry)
        C66 = C[2,2] = C[3,3]   (engineering shear)
    C12 and C66 average their physically-equal pairs to dampen FEM numerical
    noise. C11 and C22 are kept raw because they are genuinely different
    quantities (median C11/C22 ratio ~2x in ca_bulk_squared) -- averaging them,
    as the earlier D4 assumption did, discarded the dominant in-plane anisotropy.
    """
    C11 = float(C[0, 0])
    C22 = float(C[1, 1])
    C12 = 0.5 * (C[0, 1] + C[1, 0])
    C66 = 0.5 * (C[2, 2] + C[3, 3])
    return C11, C22, float(C12), float(C66)


# ---------------------------------------------------------------------------
# Duplicate / near-duplicate detection
# ---------------------------------------------------------------------------

def _exact_key(cell: np.ndarray) -> bytes:
    return cell.tobytes()


def _fuzzy_key(cell: np.ndarray, pool: int) -> bytes | None:
    """Block-pool the binary cell into a smaller fingerprint.

    A block (pool x pool) is treated as 1 if more than half of its pixels are 1.
    Returns None when pool <= 1 (i.e. fuzzy dedup is disabled).
    """
    if pool <= 1:
        return None
    H, W = cell.shape
    h = H // pool
    w = W // pool
    if h == 0 or w == 0:
        return None
    cropped = cell[: h * pool, : w * pool]
    pooled = (cropped.reshape(h, pool, w, pool).mean(axis=(1, 3)) > 0.5).astype(np.uint8)
    return pooled.tobytes()


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

CHUNK = 256


def _create_datasets(f: h5py.File, img_shape: tuple[int, int]) -> None:
    H, W = img_shape
    f.create_dataset(
        "cells",
        shape=(0, H, W),
        maxshape=(None, H, W),
        dtype=np.uint8,
        chunks=(CHUNK, H, W),
        compression="gzip",
        compression_opts=4,
    )
    f.create_dataset("C_eff", shape=(0, 4, 4), maxshape=(None, 4, 4), dtype=np.float64, chunks=(CHUNK, 4, 4))
    for name in ("C11", "C22", "C12", "C66", "lambda_", "mu", "rho", "vf", "vol", "live_fraction"):
        f.create_dataset(name, shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(CHUNK,))
    f.create_dataset("source_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(CHUNK,))

    f.attrs["E_cement"] = E_CEMENT
    f.attrs["nu"] = NU
    f.attrs["voigt_order"] = "[sigma_11, sigma_22, sigma_12, sigma_21]"
    f.attrs["stiffness_params"] = "C11, C22, C12, C66 (orthotropic D2 class, 2D)"
    f.attrs["lame_aliases"] = "lambda_ := C12, mu := C66 (back-compat; assume isotropy)"


def _append(
    f: h5py.File,
    cell: np.ndarray,
    C: np.ndarray,
    C11: float,
    C22: float,
    C12: float,
    C66: float,
    rho: float,
    vf: float,
    lf: float,
    src_idx: int,
) -> None:
    idx = f["C_eff"].shape[0]
    for key in ("cells", "C_eff", "C11", "C22", "C12", "C66", "lambda_", "mu",
                "rho", "vf", "vol", "live_fraction", "source_idx"):
        f[key].resize(idx + 1, axis=0)
    f["cells"][idx] = cell.astype(np.uint8)
    f["C_eff"][idx] = C
    f["C11"][idx] = C11
    f["C22"][idx] = C22
    f["C12"][idx] = C12
    f["C66"][idx] = C66
    f["lambda_"][idx] = C12  # back-compat alias (would-be lambda if isotropic)
    f["mu"][idx] = C66       # back-compat alias (engineering shear modulus)
    f["rho"][idx] = rho
    f["vf"][idx] = vf
    f["vol"][idx] = vf  # solid volume fraction, alias matching 3D pipeline naming
    f["live_fraction"][idx] = lf
    f["source_idx"][idx] = src_idx


# ---------------------------------------------------------------------------
# Timeout context (POSIX)
# ---------------------------------------------------------------------------

class _Timeout:
    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False


# ---------------------------------------------------------------------------
# Up-front dedup pass (cheap, runs in main process)
# ---------------------------------------------------------------------------

def _dedup_pass(
    cells: np.ndarray,
    indices: np.ndarray,
    fuzzy_pool: int,
    seen_exact: set[bytes] | None = None,
    seen_fuzzy: set[bytes] | None = None,
) -> tuple[list[int], list[int]]:
    """Walk indices, partition into (kept, dup) based on exact + fuzzy keys.

    The two ``seen_*`` sets may be passed in pre-populated (from a resume run);
    they are mutated in place so the caller can keep using them.
    """
    if seen_exact is None:
        seen_exact = set()
    if seen_fuzzy is None:
        seen_fuzzy = set()
    kept: list[int] = []
    dup: list[int] = []

    chunk = 4096
    n = len(indices)
    pbar = tqdm(range(0, n, chunk), desc="dedup scan", unit_scale=chunk, unit="cell")
    for s0 in pbar:
        srcs = indices[s0 : s0 + chunk]
        block = np.asarray(cells[srcs])  # (B, H, W) uint8
        for k, src in enumerate(srcs):
            cell = block[k]
            ek = cell.tobytes()
            if ek in seen_exact:
                dup.append(int(src))
                continue
            fk = _fuzzy_key(cell, fuzzy_pool)
            if fk is not None and fk in seen_fuzzy:
                dup.append(int(src))
                continue
            seen_exact.add(ek)
            if fk is not None:
                seen_fuzzy.add(fk)
            kept.append(int(src))
        pbar.set_postfix(kept=len(kept), dup=len(dup))
    return kept, dup


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

# Module-level globals get populated in each worker process by ``_worker_init``.
# Using globals (rather than passing the memmap through the queue every call)
# is the standard pattern for sharing read-only data with a Pool — fork()
# means the OS shares the underlying file pages with all workers for free.
_W_CELLS: np.ndarray | None = None
_W_RHO_SOLID: float = RHO_CEMENT
_W_METHOD: str = "hifi"
_W_MESH_N: int = 100
_W_ELE_TYPE: str = "TRI6"


def _worker_init(
    cells_path_str: str,
    rho_solid: float = RHO_CEMENT,
    method: str = "hifi",
    mesh_N: int = 100,
    ele_type: str = "TRI6",
) -> None:
    global _W_CELLS, _W_RHO_SOLID, _W_METHOD, _W_MESH_N, _W_ELE_TYPE
    _W_RHO_SOLID = float(rho_solid)
    _W_METHOD = str(method)
    _W_MESH_N = int(mesh_N)
    _W_ELE_TYPE = str(ele_type)

    # Pin this worker to a distinct logical CPU so its thread pool can't
    # oversubscribe the box. Fall back silently on platforms without
    # sched_setaffinity (non-Linux). The "distinct CPU" decision is made by
    # round-robin on the worker's PID, which is good enough — we don't need
    # perfect uniqueness, just no two workers fighting on the exact same core.
    try:
        ncpu = _os.cpu_count() or 1
        cpu = _os.getpid() % ncpu
        _os.sched_setaffinity(0, {cpu})
    except (AttributeError, OSError):
        pass

    _W_CELLS = np.load(cells_path_str, mmap_mode="r")
    # Warmup is intentionally minimal — JAX-FEM compile is ~1 s on this box.
    # We DON'T solve a real cell here, because some cells make jax-fem's
    # Newton iteration oscillate near tolerance and never converge. If a
    # worker happened to warm up on such a cell, it would hang at startup
    # with no per-sample timeout to rescue it. Instead we let the first real
    # task pay the ~1 s compile cost; that task is wrapped in _Timeout so a
    # bad-luck non-converging cell can't hang the worker.


def _worker_run(args: tuple[int, int]) -> tuple[int, np.ndarray | None, float | None, float | None, str | None]:
    src_idx, timeout = args
    assert _W_CELLS is not None
    cell = np.asarray(_W_CELLS[src_idx], dtype=np.int8)
    try:
        with _Timeout(timeout):
            C, rho, vf = _stiffness(cell, _W_RHO_SOLID, _W_METHOD, _W_MESH_N, _W_ELE_TYPE)
        return (src_idx, C, rho, vf, None)
    except TimeoutError:
        return (src_idx, None, None, None, "TIMEOUT")
    except Exception as exc:  # noqa: BLE001 — surface anything as a per-sample failure
        return (src_idx, None, None, None, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _report_bins(
    live_fractions: np.ndarray,
    accepted_idx: np.ndarray,
    skipped_idx: np.ndarray,
    failed_idx: np.ndarray,
    n_bins: int,
    lf_min: float,
    lf_max: float,
) -> str:
    edges = np.linspace(lf_min, lf_max, n_bins + 1)

    def _hist(idx: np.ndarray) -> np.ndarray:
        if len(idx) == 0:
            return np.zeros(n_bins, dtype=np.int64)
        return np.histogram(live_fractions[idx], bins=edges)[0]

    accepted_h = _hist(accepted_idx)
    skipped_h = _hist(skipped_idx)
    failed_h = _hist(failed_idx)

    total = len(accepted_idx) + len(skipped_idx) + len(failed_idx)
    lines = [
        "=" * 78,
        f"{'live_fraction bin':>22}  {'accepted':>10}  {'duplicate':>10}  {'failed':>10}",
        "-" * 78,
    ]
    for i in range(n_bins):
        lines.append(
            f"  [{edges[i]:.3f}, {edges[i+1]:.3f})       {accepted_h[i]:10d}  "
            f"{skipped_h[i]:10d}  {failed_h[i]:10d}"
        )
    lines += [
        "-" * 78,
        f"{'TOTAL':>22}  {len(accepted_idx):10d}  {len(skipped_idx):10d}  {len(failed_idx):10d}",
        f"  processed: {total}    accept rate: {len(accepted_idx)/max(1,total):.2%}",
        "=" * 78,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("output/ca_bulk_squared"),
        help="Directory holding cells.npy and live_fractions.npy",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HDF5 path (default: <input>/stiffness.h5)",
    )
    parser.add_argument("--start", type=int, default=0, help="First sample index to process")
    parser.add_argument(
        "--num",
        type=int,
        default=None,
        help="Number of samples to process from --start (default: all remaining)",
    )
    parser.add_argument(
        "--fuzzy-pool",
        type=int,
        default=5,
        help="Block size for the near-duplicate fingerprint (set to 1 to disable). "
        "A 50x50 cell with --fuzzy-pool=5 yields a 10x10 binary signature.",
    )
    parser.add_argument(
        "--no-fuzzy-dedup",
        action="store_true",
        help="Disable near-duplicate detection (still drops exact duplicates).",
    )
    parser.add_argument("--bins", type=int, default=10, help="Number of live_fraction bins for the summary")
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-sample FEM timeout in seconds. Legacy TRI3@1px cells solve in "
             "<1s; hi-fi TRI6 cells take a few seconds at N=100 and longer at high "
             "N. 60s is a comfortable upper bound; raise it for large --mesh-N.",
    )
    parser.add_argument(
        "--homog",
        choices=("hifi", "legacy"),
        default="hifi",
        help="Homogenizer. 'hifi' (default): quadratic TRI6, configurable mesh "
             "resolution --mesh-N, single factorization shared by the 4 load cases "
             "(accurate; the legacy TRI3@1px path over-predicts stiffness by "
             "12%%-130%%). 'legacy': original linear TRI3 at 1 element/pixel "
             "(kept for back-compat / reproducing the old dataset).",
    )
    parser.add_argument(
        "--mesh-N",
        type=int,
        default=100,
        help="hi-fi homogenization mesh resolution (NxN), decoupled from the "
             "pixel image. Default 100 = 2 elements/pixel for a 50x50 cell. A loud "
             "warning fires below ~2 elem/pixel; below 1 the microstructure aliases.",
    )
    parser.add_argument(
        "--ele-type",
        choices=("TRI6", "TRI3"),
        default="TRI6",
        help="hi-fi element type: TRI6 (quadratic, default, ~2x more accurate per "
             "resolution) or TRI3 (linear).",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Number of FEM worker processes (1 = sequential). Each worker JIT-compiles "
             "JAX-FEM independently at startup (~5-10 min wall on first init).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing HDF5 file, skipping source indices already present.",
    )
    parser.add_argument(
        "--rho-solid",
        type=float,
        default=RHO_CEMENT,
        help=f"Density of the solid phase (kg/m^3). Effective rho = vf * rho_solid. "
             f"Default {RHO_CEMENT} (cement) preserves back-compat with the original pipeline.",
    )
    args = parser.parse_args()

    in_dir: Path = args.input
    cells_path = in_dir / "cells.npy"
    lf_path = in_dir / "live_fractions.npy"
    if not cells_path.exists() or not lf_path.exists():
        print(f"Missing cells.npy or live_fractions.npy in {in_dir}", file=sys.stderr)
        sys.exit(1)

    out_path: Path = args.output if args.output else in_dir / "stiffness.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cells = np.load(cells_path, mmap_mode="r")
    live_fractions = np.load(lf_path)
    N_total = cells.shape[0]
    H, W = cells.shape[1:]
    assert live_fractions.shape == (N_total,), "cells.npy and live_fractions.npy length mismatch"

    start = max(0, args.start)
    end = N_total if args.num is None else min(N_total, start + args.num)
    indices = np.arange(start, end)

    fuzzy_pool = 1 if args.no_fuzzy_dedup else max(1, args.fuzzy_pool)
    n_workers = max(1, args.workers)

    # Report the homogenization mesh and warn loudly if it under-resolves the
    # microstructure (the whole point of the hi-fi path).
    if args.homog == "hifi":
        print(estimate_and_warn(args.mesh_N, int(H), args.ele_type))
    else:
        print(f"  homogenization: legacy TRI3 @ 1 element/pixel (N={H})")

    accepted_src: list[int] = []
    failed_src: list[int] = []

    seen_exact: set[bytes] = set()
    seen_fuzzy: set[bytes] = set()

    mode = "a" if args.resume and out_path.exists() else "w"
    with h5py.File(out_path, mode) as f:
        if "C_eff" not in f:
            _create_datasets(f, (H, W))
            f.attrs["fuzzy_pool"] = fuzzy_pool
            f.attrs["rho_solid"] = float(args.rho_solid)
            # Keep legacy attr name in sync for older readers.
            f.attrs["rho_cement"] = float(args.rho_solid)
            # Record how the stiffness was homogenized (provenance).
            f.attrs["homog_method"] = args.homog
            if args.homog == "hifi":
                f.attrs["homog_ele_type"] = args.ele_type
                f.attrs["homog_mesh_N"] = int(args.mesh_N)
                f.attrs["homog_elem_per_pixel"] = float(args.mesh_N) / float(H)
            else:
                f.attrs["homog_ele_type"] = "TRI3"
                f.attrs["homog_mesh_N"] = int(H)
                f.attrs["homog_elem_per_pixel"] = 1.0

        # Resume: pre-populate dedup sets from already-stored cells, drop those
        # source indices from the work list.
        already_done: set[int] = set()
        if mode == "a" and f["cells"].shape[0] > 0:
            print(f"Resuming: {f['cells'].shape[0]} samples already in {out_path}")
            stored = f["source_idx"][:]
            already_done = set(int(s) for s in stored)
            indices = np.array([i for i in indices if int(i) not in already_done], dtype=indices.dtype)
            for k in tqdm(range(f["cells"].shape[0]), desc="rebuild dedup", unit="cell"):
                c = np.asarray(f["cells"][k])
                seen_exact.add(_exact_key(c))
                fk = _fuzzy_key(c, fuzzy_pool)
                if fk is not None:
                    seen_fuzzy.add(fk)

        # ---- Phase 1: pre-filter duplicates (fast, no FEM) -------------------
        kept_idx, dup_idx = _dedup_pass(cells, indices, fuzzy_pool, seen_exact, seen_fuzzy)
        skipped_src = dup_idx
        print(f"After dedup: {len(kept_idx)} unique / {len(dup_idx)} duplicate "
              f"out of {len(indices)} candidate samples")

        # ---- Phase 2: FEM ----------------------------------------------------
        t0 = time.time()
        if n_workers <= 1:
            # No explicit warmup. JAX-FEM compile is ~1 s and a non-converging
            # cell could hang an unwrapped warmup forever. Let the first real
            # task pay the compile cost — it's already wrapped in _Timeout.
            pbar = tqdm(kept_idx, desc="FEM (sequential)", unit="cell")
            for src_idx in pbar:
                cell = np.asarray(cells[src_idx], dtype=np.uint8)
                try:
                    with _Timeout(args.timeout):
                        C, rho, vf = _stiffness(
                            cell.astype(np.int8), args.rho_solid,
                            args.homog, args.mesh_N, args.ele_type,
                        )
                except TimeoutError:
                    tqdm.write(f"  TIMEOUT (>{args.timeout}s) at src={src_idx}")
                    failed_src.append(src_idx)
                    continue
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"  FAILED at src={src_idx}: {type(exc).__name__}: {exc}")
                    failed_src.append(src_idx)
                    continue

                C11, C22, C12, C66 = _ortho_params_from_C(C)
                _append(f, cell, C, C11, C22, C12, C66, rho, vf, float(live_fractions[src_idx]), src_idx)
                accepted_src.append(src_idx)
                if f["C_eff"].shape[0] % CHUNK == 0:
                    f.flush()
                pbar.set_postfix(kept=len(accepted_src), fail=len(failed_src))
        else:
            print(f"Spawning {n_workers} FEM workers (spawn start method).")
            # Use spawn rather than fork. Forking *after* the parent has
            # imported JAX/PETSc copies threads, allocators, and JIT state into
            # each child, which makes XLA compile multiples of times slower
            # (and sometimes deadlock). Spawn gives each worker a clean
            # interpreter at the cost of re-importing modules on startup.
            ctx = mp.get_context("spawn")
            tasks = [(int(s), int(args.timeout)) for s in kept_idx]
            chunksize = max(1, min(8, len(tasks) // (n_workers * 32) or 1))
            with ctx.Pool(
                processes=n_workers,
                initializer=_worker_init,
                initargs=(str(cells_path), float(args.rho_solid),
                          args.homog, int(args.mesh_N), args.ele_type),
            ) as pool:
                pbar = tqdm(
                    pool.imap_unordered(_worker_run, tasks, chunksize=chunksize),
                    total=len(tasks),
                    desc=f"FEM ({n_workers} workers)",
                    unit="cell",
                )
                for src_idx, C, rho, vf, err in pbar:
                    if err is None:
                        cell = np.asarray(cells[src_idx], dtype=np.uint8)
                        C11, C22, C12, C66 = _ortho_params_from_C(C)
                        _append(f, cell, C, C11, C22, C12, C66, rho, vf, float(live_fractions[src_idx]), src_idx)
                        accepted_src.append(src_idx)
                        if f["C_eff"].shape[0] % CHUNK == 0:
                            f.flush()
                    else:
                        failed_src.append(src_idx)
                        tqdm.write(f"  {err} at src={src_idx}")
                    pbar.set_postfix(kept=len(accepted_src), fail=len(failed_src))

        f.flush()
        elapsed = time.time() - t0

    # ----- summary -----
    summary = _report_bins(
        live_fractions,
        np.array(accepted_src, dtype=np.int64),
        np.array(skipped_src, dtype=np.int64),
        np.array(failed_src, dtype=np.int64),
        n_bins=args.bins,
        lf_min=float(live_fractions.min()),
        lf_max=float(live_fractions.max()),
    )
    n_processed = len(accepted_src) + len(failed_src)
    print(f"\nFEM phase processed {n_processed} samples in {elapsed:.1f}s "
          f"({elapsed/max(1,n_processed):.3f}s/cell wall, {n_workers} worker(s))")
    print(summary)
    print(f"\nWrote {out_path}  (kept {len(accepted_src)} unique designs)")


if __name__ == "__main__":
    main()
