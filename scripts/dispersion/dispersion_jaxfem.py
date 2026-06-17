#!/usr/bin/env python3
"""Bloch-Floquet dispersion using the rayleigh_cloak config + material pipeline.

Config-driven alternative to ``dispersion_ideal.py``.  Physical parameters are
derived from a :class:`SimulationConfig` YAML (consistent with the forward
solver) rather than hardcoded.

Reuses from rayleigh_cloak:
  - ``SimulationConfig`` / ``DerivedParams`` — parameter derivation
  - ``TriangularCloakGeometry.from_params`` — geometry object
  - ``C_eff``, ``rho_eff``, ``C_iso`` — material assignment

The FEM assembly and eigenvalue solve remain standalone (P1 triangles, ARPACK)
since JAX-FEM is designed for BVP solves, not generalized eigenvalue problems.

Usage::

    python scripts/dispersion_jaxfem.py                            # default config
    python scripts/dispersion_jaxfem.py configs/continuous.yaml    # explicit config
    python scripts/dispersion_jaxfem.py --force --n-kpts 30
    python scripts/dispersion_jaxfem.py --H-factor 1.5             # taller unit cell
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from rayleigh_cloak.config import DerivedParams, SimulationConfig, load_config
from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_iso, C_eff as C_eff_fn, rho_eff as rho_eff_fn, _get_converters


# ── Unit-cell parameters from config ─────────────────────────────────


def unit_cell_params(cfg: SimulationConfig, H_factor: float = 1.0) -> dict:
    """Derive unit-cell parameters from a SimulationConfig.

    The physical material properties (rho0, cs, lam, mu, etc.) come from
    ``DerivedParams``.  The unit-cell dimensions (H, L_c) and cloak geometry
    (a, b, c) are placed in a local coordinate system (no PML offsets).

    Parameters
    ----------
    cfg : SimulationConfig
        Loaded YAML configuration.
    H_factor : float
        Additional scaling for unit-cell height.  The baseline H comes
        from ``cfg.domain.H_factor * lambda_star``; this multiplier is
        applied on top.  Triangle dimensions are **not** scaled.
    """
    params = DerivedParams.from_config(cfg)

    # Use config-derived physical quantities
    rho0 = params.rho0
    cs = params.cs
    mu = params.mu
    lam = params.lam
    nu = params.nu
    cR = params.cR
    ls = params.lambda_star

    # Unit-cell geometry — baseline H from config
    H_base = cfg.domain.H_factor * ls
    H = H_base * H_factor

    # Cloak dimensions from config factors applied to baseline H
    a = cfg.geometry.a_factor * H_base
    b = cfg.geometry.b_factor * a
    c = cfg.geometry.c_factor * H_base

    # Unit-cell width: 2 * lambda_star (BZ edge at k_norm = 0.25)
    L_c = 2.0 * ls

    return dict(
        rho0=rho0, cs=cs, mu=mu, lam=lam, nu=nu, cR=cR,
        lambda_star=ls, H=H, L_c=L_c,
        a=a, b=b, c=c, x_c=L_c / 2, y_top=H,
    )


# ── Mesh generation ───────────────────────────────────────────────────


def generate_mesh(case: str, p: dict, h_elem: float = 0.08,
                  h_fine: float = 0.03):
    """Generate TRI3 unit-cell mesh via gmsh.

    Uses the same mesh topology as ``dispersion_ideal.py``: periodic left/right
    boundaries for Bloch BCs, Dirichlet bottom, traction-free top.

    Parameters
    ----------
    case : ``"reference"`` or ``"ideal_cloak"``
    p : dict from :func:`unit_cell_params`
    h_elem, h_fine : global and fine mesh sizes

    Returns
    -------
    nodes : (N, 2)
    elems : (N_e, 3) int
    left_nodes, right_nodes, bottom_nodes : lists of node indices
    right_to_left : dict  right-idx → left-idx  (Bloch master–slave pairs)
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("unit_cell")

    L_c = p["L_c"]
    H = p["H"]
    a = p["a"]
    b = p["b"]
    c = p["c"]
    x_c = p["x_c"]

    geo = gmsh.model.geo

    # Corner points
    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)
    p2 = geo.addPoint(L_c, 0.0, 0.0, h_elem)
    p3 = geo.addPoint(L_c, H, 0.0, h_elem)
    p4 = geo.addPoint(0.0, H, 0.0, h_elem)

    # Cloak vertices (same refinement for both cases)
    pt_L = geo.addPoint(x_c - c, H, 0.0, h_fine)
    pt_R = geo.addPoint(x_c + c, H, 0.0, h_fine)
    pt_apex = geo.addPoint(x_c, H - a, 0.0, h_fine)
    oc_apex = geo.addPoint(x_c, H - b, 0.0, h_fine)

    if case == "reference":
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        l_top1 = geo.addLine(p3, pt_R)
        l_top_mid = geo.addLine(pt_R, pt_L)
        l_top2 = geo.addLine(pt_L, p4)
        l_left = geo.addLine(p4, p1)

        loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, l_top_mid, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([loop])
        geo.synchronize()

        tl_right = geo.addLine(pt_R, pt_apex)
        tl_left = geo.addLine(pt_apex, pt_L)
        geo.synchronize()
        gmsh.model.mesh.embed(0, [pt_apex, oc_apex], 2, surf)
        gmsh.model.mesh.embed(1, [tl_right, tl_left], 2, surf)
    else:
        l_bot = geo.addLine(p1, p2)
        l_right = geo.addLine(p2, p3)
        l_top1 = geo.addLine(p3, pt_R)
        l_top2 = geo.addLine(pt_L, p4)
        l_left = geo.addLine(p4, p1)
        tl_right = geo.addLine(pt_R, pt_apex)
        tl_left = geo.addLine(pt_apex, pt_L)

        outer_loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, tl_right, tl_left, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([outer_loop])
        geo.synchronize()
        gmsh.model.mesh.embed(0, [oc_apex], 2, surf)

    # Refinement around cloak triangle edges
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [tl_right, tl_left])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 100)

    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField", fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin", h_fine)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax", h_elem)
    gmsh.model.mesh.field.setNumber(ft, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMax", b * 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)

    # Periodic BCs: right (slave) ↔ left (master)
    gmsh.model.mesh.setPeriodic(
        1, [l_right], [l_left],
        [1, 0, 0, L_c,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1],
    )

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    # Extract nodes
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2].copy()
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}

    # Extract TRI3 elements
    etypes, _, enode_tags = gmsh.model.mesh.getElements(dim=2)
    tris = []
    for etype, entags in zip(etypes, enode_tags):
        if etype == 2:
            conn = entags.reshape(-1, 3)
            for row in conn:
                tris.append([tag2idx[int(r)] for r in row])
    elems = np.array(tris, dtype=int)

    # Boundary node lists
    def line_nodes(tag):
        ntags, _, _ = gmsh.model.mesh.getNodes(dim=1, tag=tag)
        return [tag2idx[int(t)] for t in ntags]

    left_nodes = line_nodes(l_left)
    right_nodes = line_nodes(l_right)
    bottom_nodes = line_nodes(l_bot)

    # Periodic (Bloch) node pairing
    _, slave_tags, master_tags, _ = gmsh.model.mesh.getPeriodicNodes(1, l_right)
    right_to_left = {
        tag2idx[int(s)]: tag2idx[int(m)]
        for s, m in zip(slave_tags, master_tags)
    }

    gmsh.finalize()
    return nodes, elems, left_nodes, right_nodes, bottom_nodes, right_to_left


# ── Material assignment via rayleigh_cloak geometry ───────────────────


def element_materials(nodes, elems, case: str, p: dict):
    """Return C_elems (n_e,2,2,2,2) and rho_elems (n_e,) at centroids.

    Uses ``TriangularCloakGeometry`` from rayleigh_cloak for consistent
    material evaluation with the forward solver.
    """
    rho0 = p["rho0"]
    C0 = np.array(C_iso(p["lam"], p["mu"]))
    n_e = len(elems)

    if case == "reference":
        return (np.broadcast_to(C0, (n_e, 2, 2, 2, 2)).copy(),
                np.full(n_e, rho0))

    # Build geometry from config-derived parameters
    geo = TriangularCloakGeometry(
        a=p["a"], b=p["b"], c=p["c"],
        x_c=p["x_c"], y_top=p["y_top"],
    )
    C0_jax = jnp.array(C0)

    centroids = jnp.array(nodes[elems].mean(axis=1))
    C_elems = np.array(jax.vmap(lambda x: C_eff_fn(x, geo, C0_jax))(centroids))
    rho_elems = np.array(jax.vmap(lambda x: rho_eff_fn(x, geo, rho0))(centroids))

    return C_elems, rho_elems


def element_materials_optimized(nodes, elems, p: dict,
                                params_npz: str, n_C_params: int = 2,
                                n_x: int = 50, n_y: int = 50):
    """Assign materials from optimized cell-based parameters.

    Loads ``cell_C_flat`` and ``cell_rho`` from an .npz file, maps each
    element centroid to the cell grid covering the cloak bounding box,
    and returns per-element C and rho arrays.

    Material values are clamped to physically valid ranges for the
    eigenvalue dispersion analysis (positive density, positive shear
    modulus).  The forward BVP solver tolerates negative Lamé parameters
    as artifacts of isotropic approximation of Cosserat behaviour, but
    the generalised eigenvalue problem requires positive-definite K & M.
    """
    rho0 = p["rho0"]
    C0 = np.array(C_iso(p["lam"], p["mu"]))
    n_e = len(elems)

    # Load optimized parameters
    data = np.load(params_npz)
    cell_C_flat = data["cell_C_flat"].copy()   # (n_cells, n_C_params)
    cell_rho = data["cell_rho"].copy()         # (n_cells,)

    # ── Clamp to physical ranges ──────────────────────────────────────
    # For n_C_params=2: flat = [λ, μ].  Need μ > 0 for positive-definite K,
    # and ρ > 0 for positive-definite M.
    rho_min = 0.01 * rho0       # 1% of background density
    mu_min = 0.01 * p["mu"]     # 1% of background shear modulus

    n_rho_clamped = int((cell_rho < rho_min).sum())
    if n_rho_clamped:
        print(f"  ⚠ Clamping {n_rho_clamped} cells with ρ < {rho_min:.1f} "
              f"(min was {cell_rho.min():.1f})")
    cell_rho = np.maximum(cell_rho, rho_min)

    if n_C_params == 2:
        n_mu_clamped = int((cell_C_flat[:, 1] < mu_min).sum())
        if n_mu_clamped:
            print(f"  ⚠ Clamping {n_mu_clamped} cells with μ < {mu_min:.1f} "
                  f"(min was {cell_C_flat[:, 1].min():.1f})")
        cell_C_flat[:, 1] = np.maximum(cell_C_flat[:, 1], mu_min)

    # Cloak bounding box (same as CellDecomposition)
    x_c = p["x_c"]
    y_top = p["y_top"]
    c = p["c"]
    b = p["b"]
    x_min, x_max = x_c - c, x_c + c
    y_min, y_max = y_top - b, y_top
    cell_dx = (x_max - x_min) / n_x
    cell_dy = (y_max - y_min) / n_y

    # Convert flat params → full tensors
    _, from_flat = _get_converters(n_C_params)
    cell_C_full = np.array(jax.vmap(from_flat)(jnp.array(cell_C_flat)))  # (n_cells,2,2,2,2)

    # Map each centroid to a cell
    centroids = nodes[elems].mean(axis=1)  # (n_e, 2)
    ix = np.floor((centroids[:, 0] - x_min) / cell_dx).astype(int)
    iy = np.floor((centroids[:, 1] - y_min) / cell_dy).astype(int)

    C_elems = np.broadcast_to(C0, (n_e, 2, 2, 2, 2)).copy()
    rho_elems = np.full(n_e, rho0)

    in_grid = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
    cell_idx = np.clip(ix * n_y + iy, 0, n_x * n_y - 1)

    C_elems[in_grid] = cell_C_full[cell_idx[in_grid]]
    rho_elems[in_grid] = cell_rho[cell_idx[in_grid]]

    return C_elems, rho_elems


# ── FEM assembly ──────────────────────────────────────────────────────

_PAIRS = [(0, 0), (1, 1), (0, 1), (1, 0)]


def assemble_KM(nodes, elems, C_elems, rho_elems, lumped: bool = False):
    """Assemble global stiffness K and mass M (sparse CSR).

    DOF ordering: [ux_0, uy_0, ux_1, uy_1, ...].
    Cosserat (full-gradient): σ_{ij} = C_{ijkl} ∂u_k/∂x_l.
    """
    N = len(nodes)
    n_e = len(elems)

    Cv = np.zeros((n_e, 4, 4))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            Cv[:, I, J] = C_elems[:, i, j, k, l]

    x = nodes[elems, 0]
    y = nodes[elems, 1]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

    areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    dN = np.zeros((n_e, 3, 2))
    dN[:, 0, 0] = (y1 - y2) / (2 * areas)
    dN[:, 1, 0] = (y2 - y0) / (2 * areas)
    dN[:, 2, 0] = (y0 - y1) / (2 * areas)
    dN[:, 0, 1] = (x2 - x1) / (2 * areas)
    dN[:, 1, 1] = (x0 - x2) / (2 * areas)
    dN[:, 2, 1] = (x1 - x0) / (2 * areas)

    B = np.zeros((n_e, 3, 4, 2))
    B[:, :, 0, 0] = dN[:, :, 0]
    B[:, :, 1, 1] = dN[:, :, 1]
    B[:, :, 2, 0] = dN[:, :, 1]
    B[:, :, 3, 1] = dN[:, :, 0]

    BtC = np.einsum("eApi,epq->eAqi", B, Cv)
    Klocal = np.einsum("eAqi,eBqj,e->eABij", BtC, B, areas)

    Mcoeff = np.zeros((n_e, 3, 3))
    if lumped:
        for A in range(3):
            Mcoeff[:, A, A] = rho_elems * areas / 3.0
    else:
        for A in range(3):
            for Bb in range(3):
                Mcoeff[:, A, Bb] = rho_elems * areas / 12.0 * (2 if A == Bb else 1)

    n_entries_K = n_e * 9 * 4
    n_entries_M = n_e * 9 * 2

    Kr = np.empty(n_entries_K, dtype=np.float64)
    Kc = np.empty(n_entries_K, dtype=np.int32)
    Kv = np.empty(n_entries_K, dtype=np.float64)
    Mr = np.empty(n_entries_M, dtype=np.float64)
    Mc = np.empty(n_entries_M, dtype=np.int32)
    Mv = np.empty(n_entries_M, dtype=np.float64)

    ik = im = 0
    for A in range(3):
        for Bb in range(3):
            for di in range(2):
                for dj in range(2):
                    rows = 2 * elems[:, A] + di
                    cols = 2 * elems[:, Bb] + dj
                    vals = Klocal[:, A, Bb, di, dj]
                    Kr[ik:ik + n_e] = rows
                    Kc[ik:ik + n_e] = cols
                    Kv[ik:ik + n_e] = vals
                    ik += n_e
            for d in range(2):
                rows = 2 * elems[:, A] + d
                cols = 2 * elems[:, Bb] + d
                vals = Mcoeff[:, A, Bb]
                Mr[im:im + n_e] = rows
                Mc[im:im + n_e] = cols
                Mv[im:im + n_e] = vals
                im += n_e

    size = 2 * N
    K = sp.csr_matrix((Kv, (Kr, Kc)), shape=(size, size))
    M = sp.csr_matrix((Mv, (Mr, Mc)), shape=(size, size))
    return K, M


# ── Bloch eigenvalue problem ──────────────────────────────────────────


def bloch_eigenproblem(K, M, nodes, bottom_nodes, right_to_left, k, L_c,
                       n_eigs: int = 40):
    """Apply Bloch + Dirichlet BCs and solve generalised eigenvalue problem.

    Returns
    -------
    omega : (n_eigs,) angular frequencies (rad/s)
    vecs : (n_free, n_eigs) complex eigenvectors
    free_nodes : list of free node indices
    """
    N = len(nodes)
    bset = set(bottom_nodes)

    right_nb = [n for n in right_to_left if n not in bset]
    right_set = set(right_nb)

    free_nodes = [n for n in range(N) if n not in bset and n not in right_set]
    free_idx = {n: i for i, n in enumerate(free_nodes)}
    n_free = len(free_nodes)

    phase = np.exp(1j * k * L_c)
    rows, cols, vals = [], [], []

    for n in free_nodes:
        f = free_idx[n]
        for d in range(2):
            rows.append(2 * n + d)
            cols.append(2 * f + d)
            vals.append(1.0 + 0j)

    for n_R in right_nb:
        n_L = right_to_left[n_R]
        if n_L not in free_idx:
            continue
        f = free_idx[n_L]
        for d in range(2):
            rows.append(2 * n_R + d)
            cols.append(2 * f + d)
            vals.append(phase)

    T = sp.csr_matrix(
        (vals, (rows, cols)), shape=(2 * N, 2 * n_free), dtype=complex
    )

    Th = T.conj().T
    K_r = Th @ (K @ T)
    M_r = Th @ (M @ T)

    K_r = 0.5 * (K_r + K_r.conj().T)
    M_r = 0.5 * (M_r + M_r.conj().T)

    n_eigs = min(n_eigs, 2 * n_free - 6)
    try:
        from scipy.sparse.linalg import eigsh, splu, LinearOperator

        K_r_csc = K_r.tocsc()
        lu = splu(K_r_csc, permc_spec="MMD_AT_PLUS_A")
        n_dof = K_r_csc.shape[0]
        OPinv = LinearOperator(
            shape=(n_dof, n_dof),
            matvec=lu.solve,
            dtype=complex,
        )
        omega_sq, vecs = eigsh(
            K_r, k=n_eigs, M=M_r,
            sigma=0.0, OPinv=OPinv, which="LM",
            tol=1e-9, maxiter=2000,
        )
        idx = np.argsort(omega_sq)
        omega_sq = omega_sq[idx]
        vecs = vecs[:, idx]
    except Exception:
        K_d = K_r.toarray()
        M_d = M_r.toarray()
        K_d = 0.5 * (K_d + K_d.conj().T)
        M_d = 0.5 * (M_d + M_d.conj().T)
        omega_sq, vecs = scipy.linalg.eigh(
            K_d, M_d, subset_by_index=[0, n_eigs - 1]
        )

    omega = np.sqrt(np.maximum(omega_sq, 0.0))
    return omega, vecs, free_nodes


# ── IPR computation ───────────────────────────────────────────────────


def compute_ipr(vecs, free_nodes, nodes, elems, right_to_left, bottom_nodes):
    """Inverse participation ratio for each mode.

    IPR = A_total * Σ_n(A_n |u_n|^4) / (Σ_n(A_n |u_n|^2))^2
    """
    N = len(nodes)
    n_eigs = vecs.shape[1]
    bset = set(bottom_nodes)
    free_idx = {n: i for i, n in enumerate(free_nodes)}

    node_area = np.zeros(N)
    for tri in elems:
        xy = nodes[tri]
        ae = 0.5 * abs(
            (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
            - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        )
        node_area[tri] += ae / 3.0

    for n_R, n_L in right_to_left.items():
        if n_R not in bset and n_L in free_idx:
            node_area[n_L] += node_area[n_R]

    A_total = node_area[free_nodes].sum()

    iprs = np.zeros(n_eigs)
    for m in range(n_eigs):
        v = vecs[:, m]
        u_sq = np.array([
            abs(v[2 * free_idx[n]]) ** 2 + abs(v[2 * free_idx[n] + 1]) ** 2
            for n in free_nodes
        ])
        An = node_area[free_nodes]
        denom = (An @ u_sq) ** 2
        if denom > 0:
            iprs[m] = A_total * (An @ u_sq ** 2) / denom
    return iprs


# ── Sweep ─────────────────────────────────────────────────────────────


def run_sweep(case: str, p: dict, k_vals: np.ndarray, n_eigs: int,
              h_elem: float, h_fine: float, out_dir: Path, force: bool,
              lumped: bool = False, tag: str | None = None,
              workers: int = 1,
              params_npz: str | None = None,
              n_C_params: int = 2,
              n_cells_x: int = 50, n_cells_y: int = 50):
    """Run or load a Bloch-Floquet sweep for one case."""
    if tag is None:
        tag = f"h{h_elem:g}_hf{h_fine:g}{'_lumped' if lumped else ''}"
    npz_path = out_dir / f"dispersion_{case}_{tag}.npz"

    if npz_path.exists() and not force:
        print(f"  Loading cached {npz_path.name}")
        d = np.load(npz_path)
        return d["ks"], d["fs"], d["iprs"]

    # For the optimized cloak, use the same mesh topology as ideal_cloak
    # (triangle cut out from the surface)
    mesh_case = "ideal_cloak" if case == "optimized_cloak" else case

    print(f"\n=== {case.upper()} — generating mesh ===")
    t0 = time.time()
    nodes, elems, left_nodes, right_nodes, bottom_nodes, right_to_left = \
        generate_mesh(mesh_case, p, h_elem=h_elem, h_fine=h_fine)
    print(f"  Mesh: {len(nodes)} nodes, {len(elems)} elements  ({time.time() - t0:.1f}s)")

    print(f"  Computing materials at {len(elems)} element centroids …")
    t1 = time.time()
    if case == "optimized_cloak" and params_npz is not None:
        C_elems, rho_elems = element_materials_optimized(
            nodes, elems, p, params_npz,
            n_C_params=n_C_params, n_x=n_cells_x, n_y=n_cells_y,
        )
    else:
        C_elems, rho_elems = element_materials(nodes, elems, case, p)
    print(f"  Done ({time.time() - t1:.1f}s)")

    print(f"  Assembling K and M (lumped={lumped}) …")
    t2 = time.time()
    K, M = assemble_KM(nodes, elems, C_elems, rho_elems, lumped=lumped)
    print(f"  Done ({time.time() - t2:.1f}s)")

    cR = p["cR"]
    ls = p["lambda_star"]
    L_c = p["L_c"]

    ks_out, fs_out, iprs_out = [], [], []
    n_k = len(k_vals)
    n_workers = min(workers, n_k)

    def _solve_one(ki_k):
        ki, k = ki_k
        t3 = time.time()
        print(f"  [k {ki + 1:3d}/{n_k}] start  k_norm={k / (2 * np.pi):.3f}",
              flush=True)
        omega, vecs, free_nodes = bloch_eigenproblem(
            K, M, nodes, bottom_nodes, right_to_left, k, L_c, n_eigs=n_eigs
        )
        ipr = compute_ipr(
            vecs, free_nodes, nodes, elems, right_to_left, bottom_nodes
        )
        f_star = omega / (2 * np.pi * cR / ls)
        k_norm = k / (2 * np.pi)
        dt = time.time() - t3
        print(f"  [k {ki + 1:3d}/{n_k}] done   k_norm={k_norm:.3f}  "
              f"n_eigs={len(f_star)}  ({dt:.2f}s)", flush=True)
        return ki, k_norm, f_star, ipr

    if n_workers > 1:
        print(f"  Solving {n_k} k-points with {n_workers} threads …")
        t_all = time.time()
        results = [None] * n_k
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_solve_one, (ki, k)): ki
                       for ki, k in enumerate(k_vals)}
            for fut in as_completed(futures):
                ki, k_norm, f_star, ipr = fut.result()
                results[ki] = (k_norm, f_star, ipr)
        for _ki, (k_norm, f_star, ipr) in enumerate(results):
            ks_out.extend([k_norm] * len(f_star))
            fs_out.extend(f_star.tolist())
            iprs_out.extend(ipr.tolist())
        print(f"  All k-points done ({time.time() - t_all:.1f}s)")
    else:
        for ki, k in enumerate(k_vals):
            _, k_norm, f_star, ipr = _solve_one((ki, k))
            ks_out.extend([k_norm] * len(f_star))
            fs_out.extend(f_star.tolist())
            iprs_out.extend(ipr.tolist())

    ks_out = np.array(ks_out)
    fs_out = np.array(fs_out)
    iprs_out = np.array(iprs_out)

    np.savez(npz_path, ks=ks_out, fs=fs_out, iprs=iprs_out)
    print(f"  Saved → {npz_path}")
    return ks_out, fs_out, iprs_out


# ── Plotting ──────────────────────────────────────────────────────────


def plot_dispersion(ref_data, cloak_data, p: dict, out_path: Path,
                    ipr_threshold: float = 2.0, f_max: float = 2.2,
                    cloak_label: str = "Ideal Cloak"):
    """Combined dispersion plot (paper Fig 3d-e style)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    L_c = p["L_c"]
    k_edge = (np.pi / L_c) / (2 * np.pi)
    x_pad = 0.012

    ks_r, fs_r, iprs_r = ref_data
    ks_c, fs_c, iprs_c = cloak_data

    ipr_cap = 15.0  # cap colorbar to avoid purple saturation
    norm = Normalize(vmin=1.0, vmax=ipr_cap)
    cmap = cm.turbo

    for ks, fs, iprs, marker, label in [
        (ks_r, fs_r, iprs_r, "D", "Reference"),
        (ks_c, fs_c, iprs_c, "o", cloak_label),
    ]:
        mask = fs <= f_max
        bulk = mask & (iprs < ipr_threshold)
        surface = mask & (iprs >= ipr_threshold)

        ax.scatter(ks[bulk], fs[bulk],
                   s=3, marker=".",
                   c=np.clip(iprs[bulk], 1.0, ipr_cap), norm=norm, cmap=cmap,
                   edgecolors="none", alpha=0.7, zorder=3)

        is_cloak = marker == "o"
        ax.scatter(ks[surface], fs[surface],
                   s=28.0 * (1.0 if is_cloak else 1.9),
                   marker=marker,
                   c=np.clip(iprs[surface], 1.0, ipr_cap), norm=norm, cmap=cmap,
                   edgecolors="black" if is_cloak else "none",
                   linewidths=0.9 if is_cloak else 0,
                   alpha=0.95, zorder=4,
                   label=label)

    # Folded Rayleigh guide lines
    k_line = np.linspace(0, k_edge, 300)
    labelled = False
    m = 0
    while True:
        f_up = 2 * m * k_edge + k_line
        f_down = 2 * (m + 1) * k_edge - k_line
        if f_up[0] > f_max:
            break
        for branch in (f_up, f_down):
            mask_b = branch <= f_max
            if not mask_b.any():
                continue
            ax.plot(k_line[mask_b], branch[mask_b], "k--",
                    lw=0.8, alpha=0.45, zorder=2,
                    label=(r"Rayleigh Analytic" if not labelled else None))
            labelled = True
        m += 1

    ax.set_xlim(-x_pad, k_edge + x_pad)
    ax.set_ylim(0, f_max)
    ax.set_xticks(np.arange(0.0, k_edge + 1e-6, 0.05))
    ax.set_xlabel(r"$\xi = k\lambda^* / (2\pi)$", fontsize=12)
    ax.set_ylabel(r"$f^* = f / (c_R / \lambda^*)$", fontsize=12)
    ax.set_title("Bloch-Floquet Dispersion (config-driven)", fontsize=13,
                 fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
    ax.tick_params(direction="out", length=4)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="0.85")

    fig.subplots_adjust(left=0.10, right=0.86, top=0.92, bottom=0.10)
    cbar_ax = fig.add_axes([0.885, 0.10, 0.022, 0.82])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("IPR", fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {out_path}")


def plot_single(data, p: dict, out_path: Path, label: str = "Reference",
                marker: str = "D", ipr_threshold: float = 2.0,
                f_max: float = 2.2, show_rayleigh: bool = True):
    """Dispersion plot for a single dataset (with optional Rayleigh lines)."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": True,
        "axes.spines.right": True,
    })

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    L_c = p["L_c"]
    k_edge = (np.pi / L_c) / (2 * np.pi)
    x_pad = 0.012

    ks, fs, iprs = data
    ipr_cap = 15.0
    norm = Normalize(vmin=1.0, vmax=ipr_cap)
    cmap = cm.turbo

    mask = fs <= f_max
    bulk = mask & (iprs < ipr_threshold)
    surface = mask & (iprs >= ipr_threshold)

    ax.scatter(ks[bulk], fs[bulk],
               s=3, marker=".",
               c=np.clip(iprs[bulk], 1.0, ipr_cap), norm=norm, cmap=cmap,
               edgecolors="none", alpha=0.7, zorder=3)

    ax.scatter(ks[surface], fs[surface],
               s=35, marker=marker,
               c=np.clip(iprs[surface], 1.0, ipr_cap), norm=norm, cmap=cmap,
               edgecolors="black", linewidths=0.8,
               alpha=0.95, zorder=4, label=label)

    if show_rayleigh:
        k_line = np.linspace(0, k_edge, 300)
        labelled = False
        m = 0
        while True:
            f_up = 2 * m * k_edge + k_line
            f_down = 2 * (m + 1) * k_edge - k_line
            if f_up[0] > f_max:
                break
            for branch in (f_up, f_down):
                mask_b = branch <= f_max
                if not mask_b.any():
                    continue
                ax.plot(k_line[mask_b], branch[mask_b], "k--",
                        lw=0.8, alpha=0.45, zorder=2,
                        label=(r"Rayleigh Analytic" if not labelled else None))
                labelled = True
            m += 1

    ax.set_xlim(-x_pad, k_edge + x_pad)
    ax.set_ylim(0, f_max)
    ax.set_xticks(np.arange(0.0, k_edge + 1e-6, 0.05))
    ax.set_xlabel(r"$\xi = k\lambda^* / (2\pi)$", fontsize=12)
    ax.set_ylabel(r"$f^* = f / (c_R / \lambda^*)$", fontsize=12)
    ax.set_title(f"Bloch-Floquet Dispersion — {label}", fontsize=13,
                 fontweight="bold", pad=10)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
    ax.tick_params(direction="out", length=4)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="0.85")

    fig.subplots_adjust(left=0.10, right=0.86, top=0.92, bottom=0.10)
    cbar_ax = fig.add_axes([0.885, 0.10, 0.022, 0.82])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("IPR", fontsize=10)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=3)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Bloch-Floquet dispersion (config-driven, rayleigh_cloak pipeline)"
    )
    ap.add_argument("config", nargs="?", default="configs/continuous.yaml",
                    help="Path to SimulationConfig YAML (default: configs/continuous.yaml)")
    ap.add_argument("--n-kpts", type=int, default=50)
    ap.add_argument("--n-eigs", type=int, default=40)
    ap.add_argument("--h-elem", type=float, default=0.08)
    ap.add_argument("--h-fine", type=float, default=0.03)
    ap.add_argument("--ipr-thr", type=float, default=2.0)
    ap.add_argument("--f-max", type=float, default=2.2)
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if cached .npz files exist")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory (default: <config.output_dir>/dispersion)")
    ap.add_argument("--lumped-mass", action="store_true")
    ap.add_argument("--H-factor", type=float, default=1.0,
                    help="Scale unit-cell height (triangle dims stay fixed)")
    ap.add_argument("--case", choices=["both", "reference", "ideal_cloak",
                                       "optimized", "optimized_vs_ref"],
                    default="both",
                    help="Which cases to compute. 'optimized' runs optimized_cloak only; "
                         "'optimized_vs_ref' compares reference vs optimized cloak.")
    ap.add_argument("--workers", "-j", type=int, default=1)
    ap.add_argument("--params-npz", type=str, default=None,
                    help="Path to optimized_params.npz (cell_C_flat, cell_rho)")
    ap.add_argument("--n-C-params", type=int, default=2,
                    help="Number of flat stiffness params per cell (default: 2 = isotropic)")
    ap.add_argument("--n-cells-x", type=int, default=50,
                    help="Number of cells in x for optimized grid (default: 50)")
    ap.add_argument("--n-cells-y", type=int, default=50,
                    help="Number of cells in y for optimized grid (default: 50)")
    args = ap.parse_args()

    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}")
        sys.exit(1)
    cfg = load_config(cfg_path)
    print(f"Config: {cfg_path}")

    # Derive unit-cell parameters from config
    p = unit_cell_params(cfg, H_factor=args.H_factor)

    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg.output_dir) / "dispersion"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Physical params (from config):")
    print(f"  rho0 = {p['rho0']},  cs = {p['cs']}")
    print(f"  cR = {p['cR']:.2f} m/s,  λ* = {p['lambda_star']}")
    print(f"  H = {p['H']:.3f},  L_c = {p['L_c']}")
    print(f"  a = {p['a']:.4f},  b = {p['b']:.4f},  c = {p['c']:.4f}")
    print(f"  BZ edge: k_norm = {(np.pi / p['L_c']) / (2 * np.pi):.3f}")
    print(f"  Mesh: h_elem={args.h_elem}, h_fine={args.h_fine}")
    print(f"  k-points: {args.n_kpts},  eigenvalues/k: {args.n_eigs}")
    print(f"  Workers: {args.workers}")
    if args.params_npz:
        print(f"  Optimized params: {args.params_npz}")
        print(f"  Cell grid: {args.n_cells_x}x{args.n_cells_y}, n_C_params={args.n_C_params}")
    print(f"  Output: {out_dir}\n")

    L_c = p["L_c"]
    k_vals = np.linspace(np.pi / (100 * L_c), np.pi / L_c, args.n_kpts)

    sweep_kw = dict(
        n_eigs=args.n_eigs, h_elem=args.h_elem, h_fine=args.h_fine,
        out_dir=out_dir, force=args.force,
        lumped=args.lumped_mass, workers=args.workers,
        params_npz=args.params_npz,
        n_C_params=args.n_C_params,
        n_cells_x=args.n_cells_x, n_cells_y=args.n_cells_y,
    )

    ref_data = cloak_data = None
    if args.case in ("both", "reference", "optimized_vs_ref"):
        ref_data = run_sweep("reference", p, k_vals, **sweep_kw)
    if args.case in ("both", "ideal_cloak"):
        cloak_data = run_sweep("ideal_cloak", p, k_vals, **sweep_kw)
    if args.case in ("optimized", "optimized_vs_ref"):
        if args.params_npz is None:
            print("ERROR: --params-npz is required for optimized_cloak case")
            sys.exit(1)
        cloak_data = run_sweep("optimized_cloak", p, k_vals, **sweep_kw)

    if ref_data is not None and cloak_data is not None:
        suffix = ""
        if args.lumped_mass:
            suffix += "_lumped"
        if args.H_factor != 1.0:
            suffix += f"_H{args.H_factor:g}"
        if args.case in ("optimized", "optimized_vs_ref"):
            suffix += "_optimized"
        cloak_label = ("Optimized Cloak" if args.case in ("optimized", "optimized_vs_ref")
                       else "Ideal Cloak")
        plot_dispersion(
            ref_data, cloak_data, p,
            out_path=out_dir / f"dispersion_comparison{suffix}.png",
            ipr_threshold=args.ipr_thr,
            f_max=args.f_max,
            cloak_label=cloak_label,
        )

    # Additional individual figures
    plot_kw = dict(ipr_threshold=args.ipr_thr, f_max=args.f_max)
    if ref_data is not None:
        plot_single(
            ref_data, p,
            out_path=out_dir / "dispersion_reference_vs_rayleigh.png",
            label="Reference", marker="D", show_rayleigh=True, **plot_kw,
        )
    if cloak_data is not None:
        cloak_label_single = ("Optimized Cloak"
                              if args.case in ("optimized", "optimized_vs_ref")
                              else "Ideal Cloak")
        plot_single(
            cloak_data, p,
            out_path=out_dir / f"dispersion_{cloak_label_single.lower().replace(' ', '_')}_vs_rayleigh.png",
            label=cloak_label_single, marker="o", show_rayleigh=True, **plot_kw,
        )


if __name__ == "__main__":
    main()
