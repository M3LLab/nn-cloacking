#!/usr/bin/env python3
"""Bloch-Floquet dispersion curves with IPR analysis.

Reproduces Fig 3(d)-(e) from Chatzopoulos et al. (2023):
  - Reference unit cell: plain rectangle, homogeneous material
  - Ideal cloak unit cell: triangular void + Cosserat transformation elasticity

Bloch-Floquet in x:  u(x + L_c, y) = exp(i k L_c) u(x, y)
Dirichlet bottom:    u(x, 0) = 0
Traction-free top and defect boundaries.

Usage::

    python scripts/dispersion_ideal.py
    python scripts/dispersion_ideal.py --n-kpts 30 --n-eigs 30
    python scripts/dispersion_ideal.py --force        # recompute even if cached
    python scripts/dispersion_ideal.py --h-elem 0.1   # coarser mesh, faster
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp

from rayleigh_cloak.geometry.triangular import TriangularCloakGeometry
from rayleigh_cloak.materials import C_iso, C_eff as C_eff_fn, rho_eff as rho_eff_fn


# ── Physical parameters ───────────────────────────────────────────────

def derive_params(H_factor: float = 1.0) -> dict:
    """Compute physical parameters matching configs/continuous.yaml.

    Parameters
    ----------
    H_factor : multiplies the baseline unit-cell height.  Use >1 to reduce
        interaction between the Rayleigh mode and the Dirichlet bottom at
        low frequencies.
    """
    rho0 = 1600.0
    cs   = 300.0
    cp   = np.sqrt(3.0) * cs
    mu   = rho0 * cs**2
    lam  = rho0 * cp**2 - 2 * mu
    nu   = lam / (2 * (lam + mu))
    cR   = cs * (0.826 + 1.14 * nu) / (1 + nu)

    ls  = 1.0                       # lambda_star
    H   = 4.305 * ls * H_factor     # unit cell height (scaled)
    L_c = 2.0 * ls                  # unit cell width (BZ edge at k_norm = 0.25)

    # Triangle dimensions are tied to the *baseline* H (absolute, λ*-scaled) —
    # do NOT scale with H_factor, otherwise the cloak geometry drifts.
    H_base = 4.305 * ls
    a = 0.0774 * H_base    # inner triangle depth
    b = 3.0 * a            # outer triangle depth  (≈ lambda_star)
    c = 0.1545 * H_base    # half-width at surface

    return dict(rho0=rho0, cs=cs, mu=mu, lam=lam, nu=nu, cR=cR,
                lambda_star=ls, H=H, L_c=L_c,
                a=a, b=b, c=c, x_c=L_c/2, y_top=H)


# ── Mesh generation ───────────────────────────────────────────────────

def generate_mesh(case: str, p: dict, h_elem: float = 0.08,
                  h_fine: float = 0.03):
    """Generate TRI3 unit-cell mesh via gmsh.

    Parameters
    ----------
    case : "reference" | "ideal_cloak"
    p    : physical parameter dict from derive_params()
    h_elem, h_fine : global and fine mesh sizes

    Returns
    -------
    nodes   : (N, 2)
    elems   : (N_e, 3) int  – node indices
    left_nodes, right_nodes, bottom_nodes : lists of node indices
    right_to_left : dict  right-idx → left-idx  (Bloch master-slave pairs)
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("unit_cell")

    L_c = p["L_c"]
    H   = p["H"]
    a   = p["a"]
    b   = p["b"]
    c   = p["c"]
    x_c = p["x_c"]

    geo = gmsh.model.geo

    # ── corner points ────────────────────────────────────────────────
    p1 = geo.addPoint(0.0, 0.0, 0.0, h_elem)   # bottom-left
    p2 = geo.addPoint(L_c, 0.0, 0.0, h_elem)   # bottom-right
    p3 = geo.addPoint(L_c,   H, 0.0, h_elem)   # top-right
    p4 = geo.addPoint(0.0,   H, 0.0, h_elem)   # top-left

    # Shared cloak vertices (same h_fine for both cases so node density matches)
    pt_L    = geo.addPoint(x_c - c, H,     0.0, h_fine)
    pt_R    = geo.addPoint(x_c + c, H,     0.0, h_fine)
    pt_apex = geo.addPoint(x_c,     H - a, 0.0, h_fine)
    oc_apex = geo.addPoint(x_c,     H - b, 0.0, h_fine)

    if case == "reference":
        # Full rectangle — top edge split at cloak opening points so node
        # placement matches the cutout mesh.  Inner triangle edges are
        # embedded (not cut) so the mesh refines identically to the cloak
        # case without creating a void.
        l_bot     = geo.addLine(p1, p2)
        l_right   = geo.addLine(p2, p3)
        l_top1    = geo.addLine(p3, pt_R)
        l_top_mid = geo.addLine(pt_R, pt_L)
        l_top2    = geo.addLine(pt_L, p4)
        l_left    = geo.addLine(p4, p1)

        loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, l_top_mid, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([loop])
        geo.synchronize()

        # Embed inner triangle edges + apex + outer apex for identical
        # refinement topology
        tl_right = geo.addLine(pt_R, pt_apex)
        tl_left  = geo.addLine(pt_apex, pt_L)
        geo.synchronize()
        gmsh.model.mesh.embed(0, [pt_apex, oc_apex], 2, surf)
        gmsh.model.mesh.embed(1, [tl_right, tl_left], 2, surf)

        top_lines = [l_top1, l_top_mid, l_top2]

    else:  # ideal_cloak – cut out triangular defect
        l_bot    = geo.addLine(p1, p2)
        l_right  = geo.addLine(p2, p3)
        l_top1   = geo.addLine(p3, pt_R)
        l_top2   = geo.addLine(pt_L, p4)
        l_left   = geo.addLine(p4, p1)
        tl_right = geo.addLine(pt_R, pt_apex)
        tl_left  = geo.addLine(pt_apex, pt_L)

        outer_loop = geo.addCurveLoop([
            l_bot, l_right, l_top1, tl_right, tl_left, l_top2, l_left
        ])
        surf = geo.addPlaneSurface([outer_loop])
        geo.synchronize()
        gmsh.model.mesh.embed(0, [oc_apex], 2, surf)

        top_lines = [l_top1, l_top2]

    # Refinement field (identical for both cases) around cloak triangle edges
    fd = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(fd, "CurvesList", [tl_right, tl_left])
    gmsh.model.mesh.field.setNumber(fd, "Sampling", 100)

    ft = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(ft, "InField",  fd)
    gmsh.model.mesh.field.setNumber(ft, "SizeMin",  h_fine)
    gmsh.model.mesh.field.setNumber(ft, "SizeMax",  h_elem)
    gmsh.model.mesh.field.setNumber(ft, "DistMin",  0.0)
    gmsh.model.mesh.field.setNumber(ft, "DistMax",  b * 2.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(ft)

    # Periodic BCs: right (slave) ↔ left (master).
    # affineTransform maps master → slave: (0,y) → (L_c, y)
    gmsh.model.mesh.setPeriodic(
        1, [l_right], [l_left],
        [1, 0, 0, L_c,
         0, 1, 0, 0,
         0, 0, 1, 0,
         0, 0, 0, 1],
    )

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(1)

    # ── extract nodes ────────────────────────────────────────────────
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    nodes = coords.reshape(-1, 3)[:, :2].copy()   # (N, 2)
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}

    # ── extract TRI3 elements ────────────────────────────────────────
    etypes, _, enode_tags = gmsh.model.mesh.getElements(dim=2)
    tris = []
    for etype, entags in zip(etypes, enode_tags):
        if etype == 2:   # TRI3
            conn = entags.reshape(-1, 3)
            for row in conn:
                tris.append([tag2idx[int(r)] for r in row])
    elems = np.array(tris, dtype=int)

    # ── extract boundary node lists ──────────────────────────────────
    def line_nodes(tag):
        ntags, _, _ = gmsh.model.mesh.getNodes(dim=1, tag=tag)
        return [tag2idx[int(t)] for t in ntags]

    left_nodes   = line_nodes(l_left)
    right_nodes  = line_nodes(l_right)
    bottom_nodes = line_nodes(l_bot)

    # ── periodic (Bloch) node pairing ───────────────────────────────
    # getPeriodicNodes(dim, tag) → tagMaster, nodeTags, nodeTagsMaster, affineTransform
    _, slave_tags, master_tags, _ = gmsh.model.mesh.getPeriodicNodes(1, l_right)
    right_to_left = {
        tag2idx[int(s)]: tag2idx[int(m)]
        for s, m in zip(slave_tags, master_tags)
    }

    gmsh.finalize()
    return nodes, elems, left_nodes, right_nodes, bottom_nodes, right_to_left


# ── Material assignment ───────────────────────────────────────────────

def element_materials(nodes, elems, case: str, p: dict):
    """Return C_elems (n_e,2,2,2,2) and rho_elems (n_e,) at centroids."""
    rho0 = p["rho0"]
    C0   = np.array(C_iso(p["lam"], p["mu"]))   # (2,2,2,2)
    n_e  = len(elems)

    if case == "reference":
        return np.broadcast_to(C0, (n_e, 2, 2, 2, 2)).copy(), \
               np.full(n_e, rho0)

    # ideal_cloak: vectorised via jax.vmap
    geo = TriangularCloakGeometry(
        a=p["a"], b=p["b"], c=p["c"],
        x_c=p["x_c"], y_top=p["y_top"],
    )
    C0_jax = jnp.array(C0)

    centroids = jnp.array(nodes[elems].mean(axis=1))   # (n_e, 2)

    C_elems   = np.array(jax.vmap(lambda x: C_eff_fn(x, geo, C0_jax))(centroids))
    rho_elems = np.array(jax.vmap(lambda x: rho_eff_fn(x, geo, rho0))(centroids))

    return C_elems, rho_elems


# ── FEM assembly ──────────────────────────────────────────────────────

# Augmented-Voigt index pairs  (matches materials.py _PAIRS)
_PAIRS = [(0, 0), (1, 1), (0, 1), (1, 0)]


def assemble_KM(nodes, elems, C_elems, rho_elems, lumped: bool = False):
    """Assemble global stiffness K and mass M as sparse CSR matrices.

    DOF ordering: [ux_0, uy_0, ux_1, uy_1, ...].
    Cosserat (full-gradient) formulation: σ_{ij} = C_{ijkl} ∂u_k/∂x_l.
    B-matrix uses augmented-Voigt ordering (0,0),(1,1),(0,1),(1,0) for (k,l).
    """
    N   = len(nodes)
    n_e = len(elems)

    # Convert C tensors to augmented-Voigt (4×4) ─────────────────────
    Cv = np.zeros((n_e, 4, 4))
    for I, (i, j) in enumerate(_PAIRS):
        for J, (k, l) in enumerate(_PAIRS):
            Cv[:, I, J] = C_elems[:, i, j, k, l]

    # Element geometry ────────────────────────────────────────────────
    x = nodes[elems, 0]   # (n_e, 3)
    y = nodes[elems, 1]
    x0, x1, x2 = x[:, 0], x[:, 1], x[:, 2]
    y0, y1, y2 = y[:, 0], y[:, 1], y[:, 2]

    areas = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))

    # Shape-function gradients (n_e, 3 nodes, 2 directions)
    dN = np.zeros((n_e, 3, 2))
    dN[:, 0, 0] = (y1 - y2) / (2 * areas)
    dN[:, 1, 0] = (y2 - y0) / (2 * areas)
    dN[:, 2, 0] = (y0 - y1) / (2 * areas)
    dN[:, 0, 1] = (x2 - x1) / (2 * areas)
    dN[:, 1, 1] = (x0 - x2) / (2 * areas)
    dN[:, 2, 1] = (x1 - x0) / (2 * areas)

    # B-matrix (n_e, 3 nodes, 4 voigt-rows, 2 DOFs)
    B = np.zeros((n_e, 3, 4, 2))
    B[:, :, 0, 0] = dN[:, :, 0]
    B[:, :, 1, 1] = dN[:, :, 1]
    B[:, :, 2, 0] = dN[:, :, 1]
    B[:, :, 3, 1] = dN[:, :, 0]

    # Local stiffness  K_e[A,B,i,j] = Σ_{p,q} B[A,p,i] Cv[p,q] B[B,q,j] * area
    BtC    = np.einsum("eApi,epq->eAqi", B, Cv)
    Klocal = np.einsum("eAqi,eBqj,e->eABij", BtC, B, areas)  # (n_e,3,3,2,2)

    # Local mass.
    #   Consistent: M_e[A,B] = rho * area/12 * (1 + δ_AB)
    #   Lumped (row-sum): M_e[A,B] = rho * area/3 * δ_AB
    # Row-sum lumping preserves total mass (3 nodes × area/3 = area) and
    # removes off-diagonal coupling, which is known to reduce numerical
    # dispersion at high wavenumber for P1 triangles.
    Mcoeff = np.zeros((n_e, 3, 3))
    if lumped:
        for A in range(3):
            Mcoeff[:, A, A] = rho_elems * areas / 3.0
    else:
        for A in range(3):
            for Bb in range(3):
                Mcoeff[:, A, Bb] = rho_elems * areas / 12.0 * (2 if A == Bb else 1)

    # Build COO triplets for sparse assembly ──────────────────────────
    n_entries_K = n_e * 9 * 4      # 3×3 node pairs × 2×2 DOF pairs
    n_entries_M = n_e * 9 * 2      # 3×3 node pairs × 2 (diagonal only)

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
                    rows = 2 * elems[:, A]  + di
                    cols = 2 * elems[:, Bb] + dj
                    vals = Klocal[:, A, Bb, di, dj]
                    Kr[ik:ik+n_e] = rows
                    Kc[ik:ik+n_e] = cols
                    Kv[ik:ik+n_e] = vals
                    ik += n_e
            for d in range(2):
                rows = 2 * elems[:, A]  + d
                cols = 2 * elems[:, Bb] + d
                vals = Mcoeff[:, A, Bb]
                Mr[im:im+n_e] = rows
                Mc[im:im+n_e] = cols
                Mv[im:im+n_e] = vals
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
    omega  : (n_eigs,) non-negative angular frequencies  (rad/s)
    vecs   : (n_free, n_eigs) complex eigenvectors in reduced DOF space
    free_nodes : list of node indices included in the reduced system
    """
    N    = len(nodes)
    bset = set(bottom_nodes)

    # Classify: right-not-bottom → Bloch slaves; everything else is free
    right_nb = [n for n in right_to_left if n not in bset]
    right_set = set(right_nb)

    free_nodes = [n for n in range(N) if n not in bset and n not in right_set]
    free_idx   = {n: i for i, n in enumerate(free_nodes)}
    n_free     = len(free_nodes)

    # Build complex transformation matrix T  (shape: 2N × 2*n_free)
    # u_full = T @ u_free
    phase = np.exp(1j * k * L_c)

    rows, cols, vals = [], [], []

    # free → identity
    for n in free_nodes:
        f = free_idx[n]
        for d in range(2):
            rows.append(2 * n + d)
            cols.append(2 * f + d)
            vals.append(1.0 + 0j)

    # right slaves → Bloch phase × corresponding left master
    for n_R in right_nb:
        n_L = right_to_left[n_R]
        if n_L not in free_idx:
            continue   # corresponding left node is also Dirichlet (corner)
        f = free_idx[n_L]
        for d in range(2):
            rows.append(2 * n_R + d)
            cols.append(2 * f + d)
            vals.append(phase)

    T = sp.csr_matrix(
        (vals, (rows, cols)), shape=(2 * N, 2 * n_free), dtype=complex
    )

    # Reduced complex Hermitian sparse system
    Th  = T.conj().T                   # (2*n_free, 2N) sparse complex
    K_r = Th @ (K @ T)                 # (2*n_free, 2*n_free) sparse complex
    M_r = Th @ (M @ T)

    # Enforce Hermitian symmetry (numerical noise from float arithmetic)
    K_r = 0.5 * (K_r + K_r.conj().T)
    M_r = 0.5 * (M_r + M_r.conj().T)

    # Solve for lowest n_eigs eigenvalues via ARPACK (shift-and-invert)
    n_eigs = min(n_eigs, 2 * n_free - 6)
    try:
        from scipy.sparse.linalg import eigsh, splu, LinearOperator

        # SuperLU (used internally by eigsh sigma= shift-and-invert) is NOT
        # thread-safe.  Instead, factorize explicitly here and pass OPinv so
        # ARPACK never calls SuperLU internally.
        # MMD_AT_PLUS_A reordering drastically reduces fill-in (10-50x less
        # memory than no reordering on unordered FEM meshes).
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
        # eigsh returns eigenvalues in arbitrary order; sort ascending
        idx      = np.argsort(omega_sq)
        omega_sq = omega_sq[idx]
        vecs     = vecs[:, idx]
    except Exception:
        # Dense fallback (small systems)
        K_d = K_r.toarray()
        M_d = M_r.toarray()
        K_d = 0.5 * (K_d + K_d.conj().T)
        M_d = 0.5 * (M_d + M_d.conj().T)
        omega_sq, vecs = scipy.linalg.eigh(
            K_d, M_d, subset_by_index=[0, n_eigs - 1]
        )

    # Discard near-zero or negative (numerical noise)
    omega = np.sqrt(np.maximum(omega_sq, 0.0))
    return omega, vecs, free_nodes


# ── IPR computation ───────────────────────────────────────────────────

def compute_ipr(vecs, free_nodes, nodes, elems, right_to_left, bottom_nodes):
    """Inverse participation ratio for each mode.

    IPR = A_total * Σ_n(A_n |u_n|^4) / (Σ_n(A_n |u_n|^2))^2

    A high IPR (> ~3) indicates a surface-localised (Rayleigh) mode.
    """
    N       = len(nodes)
    n_eigs  = vecs.shape[1]
    bset    = set(bottom_nodes)
    free_idx = {n: i for i, n in enumerate(free_nodes)}

    # Lumped nodal areas
    node_area = np.zeros(N)
    for tri in elems:
        xy  = nodes[tri]
        ae  = 0.5 * abs(
            (xy[1, 0] - xy[0, 0]) * (xy[2, 1] - xy[0, 1])
            - (xy[2, 0] - xy[0, 0]) * (xy[1, 1] - xy[0, 1])
        )
        node_area[tri] += ae / 3.0

    # For boundary nodes not in free_nodes (right slaves): add area of
    # their Bloch partner to the left (master) node so we don't lose area
    for n_R, n_L in right_to_left.items():
        if n_R not in bset and n_L in free_idx:
            node_area[n_L] += node_area[n_R]

    A_total = node_area[free_nodes].sum()

    iprs = np.zeros(n_eigs)
    for m in range(n_eigs):
        v = vecs[:, m]
        u_sq = np.array([
            abs(v[2 * free_idx[n]])**2 + abs(v[2 * free_idx[n] + 1])**2
            for n in free_nodes
        ])
        An = node_area[free_nodes]
        denom = (An @ u_sq) ** 2
        if denom > 0:
            iprs[m] = A_total * (An @ u_sq**2) / denom
    return iprs


# ── Full sweep ────────────────────────────────────────────────────────

def run_sweep(case: str, p: dict, k_vals: np.ndarray, n_eigs: int,
              h_elem: float, h_fine: float, out_dir: Path, force: bool,
              lumped: bool = False, H_factor: float = 1.0,
              tag: str | None = None, workers: int = 1):
    """Run or load the Bloch-Floquet sweep for one case.

    Returns arrays:
      ks_out  : (M,) repeated k values
      fs_out  : (M,) normalised frequencies f* = ω / (2π cR / λ*)
      iprs_out: (M,) IPR per mode
    """
    if tag is None:
        tag = (f"h{h_elem:g}_hf{h_fine:g}"
               f"{'_lumped' if lumped else ''}"
               f"{'' if H_factor == 1.0 else f'_H{H_factor:g}'}")
    npz_path = out_dir / f"dispersion_{case}_{tag}.npz"

    if npz_path.exists() and not force:
        print(f"  Loading cached {npz_path.name}")
        d = np.load(npz_path)
        return d["ks"], d["fs"], d["iprs"]

    print(f"\n=== {case.upper()} — generating mesh ===")
    t0 = time.time()
    nodes, elems, left_nodes, right_nodes, bottom_nodes, right_to_left = \
        generate_mesh(case, p, h_elem=h_elem, h_fine=h_fine)
    print(f"  Mesh: {len(nodes)} nodes, {len(elems)} elements  ({time.time()-t0:.1f}s)")

    print(f"  Computing materials at {len(elems)} element centroids …")
    t1 = time.time()
    C_elems, rho_elems = element_materials(nodes, elems, case, p)
    print(f"  Done ({time.time()-t1:.1f}s)")

    print(f"  Assembling K and M (lumped={lumped}) …")
    t2 = time.time()
    K, M = assemble_KM(nodes, elems, C_elems, rho_elems, lumped=lumped)
    print(f"  Done ({time.time()-t2:.1f}s)")

    cR   = p["cR"]
    ls   = p["lambda_star"]
    L_c  = p["L_c"]

    ks_out, fs_out, iprs_out = [], [], []

    n_k = len(k_vals)
    n_workers = min(workers, n_k)

    def _solve_one(ki_k):
        ki, k = ki_k
        t3 = time.time()
        print(f"  [k {ki+1:3d}/{n_k}] start  k_norm={k/(2*np.pi):.3f}", flush=True)
        omega, vecs, free_nodes = bloch_eigenproblem(
            K, M, nodes, bottom_nodes, right_to_left, k, L_c, n_eigs=n_eigs
        )
        ipr = compute_ipr(
            vecs, free_nodes, nodes, elems, right_to_left, bottom_nodes
        )
        f_star = omega / (2 * np.pi * cR / ls)
        k_norm = k / (2 * np.pi)
        dt = time.time() - t3
        print(f"  [k {ki+1:3d}/{n_k}] done   k_norm={k_norm:.3f}  "
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
        for ki, (k_norm, f_star, ipr) in enumerate(results):
            ks_out.extend([k_norm] * len(f_star))
            fs_out.extend(f_star.tolist())
            iprs_out.extend(ipr.tolist())
        print(f"  All k-points done ({time.time()-t_all:.1f}s)")
    else:
        for ki, k in enumerate(k_vals):
            t3 = time.time()
            _, k_norm, f_star, ipr = _solve_one((ki, k))

            ks_out.extend([k_norm] * len(f_star))
            fs_out.extend(f_star.tolist())
            iprs_out.extend(ipr.tolist())

            dt = time.time() - t3
            print(f"  k {ki+1:3d}/{n_k}  k_norm={k_norm:.3f}  "
                  f"n_eigs={len(f_star)}  ({dt:.2f}s)")

    ks_out   = np.array(ks_out)
    fs_out   = np.array(fs_out)
    iprs_out = np.array(iprs_out)

    np.savez(npz_path, ks=ks_out, fs=fs_out, iprs=iprs_out)
    print(f"  Saved → {npz_path}")
    return ks_out, fs_out, iprs_out


# ── Plotting ──────────────────────────────────────────────────────────

def plot_dispersion(
    ref_data,
    cloak_data,
    p: dict,
    out_path: Path,
    ipr_threshold: float = 2.0,
    f_max: float = 2.2,
):
    """Combined dispersion plot (paper Fig 3d-e style, single panel).

    Reference modes shown as hollow diamonds, ideal-cloak modes as filled
    circles coloured by IPR.  Surface (Rayleigh) modes are distinguished
    from bulk via the IPR threshold.
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top":   True,
        "axes.spines.right": True,
    })

    fig, ax_only = plt.subplots(figsize=(8.5, 6.0))
    axes = [ax_only]

    L_c    = p["L_c"]
    k_edge = (np.pi / L_c) / (2 * np.pi)   # = 0.25
    x_pad  = 0.012                         # left-margin so x=0 markers fit

    ks_r, fs_r, iprs_r = ref_data
    ks_c, fs_c, iprs_c = cloak_data

    # Colour + size scale: linear in IPR, green → yellow
    ipr_max = float(max(iprs_r.max(), iprs_c.max()))
    norm = Normalize(vmin=1.0, vmax=ipr_max)
    cmap = cm.turbo

    def _sizes(ipr):
        """Constant marker size (IPR shown only via colour)."""
        return np.full_like(ipr, 28.0, dtype=float)

    # (title, ax, list[(ks, fs, iprs, marker, label)])
    panels = [
        ("Combined", axes[0], [
            (ks_r, fs_r, iprs_r, "D", "Reference"),
            (ks_c, fs_c, iprs_c, "o", "Ideal Cloak"),
        ]),
    ]

    for title, ax, datasets in panels:
        for ks, fs, iprs, marker, label in datasets:
            mask = fs <= f_max
            bulk    = mask & (iprs <  ipr_threshold)
            surface = mask & (iprs >= ipr_threshold)

            # Bulk modes: tiny coloured dots
            ax.scatter(ks[bulk], fs[bulk],
                       s=3, marker=".",
                       c=iprs[bulk], norm=norm, cmap=cmap,
                       edgecolors="none", alpha=0.7, zorder=3)

            # Surface modes: full-size markers
            is_cloak = marker == "o"
            ax.scatter(ks[surface], fs[surface],
                       s=_sizes(iprs[surface]) * (1.0 if is_cloak else 1.9),
                       marker=marker,
                       c=iprs[surface], norm=norm, cmap=cmap,
                       edgecolors="black" if is_cloak else "none",
                       linewidths=0.9 if is_cloak else 0,
                       alpha=0.95, zorder=4,
                       label=label)

        # Folded Rayleigh guide lines
        k_line = np.linspace(0, k_edge, 300)
        labelled = False
        m = 0
        while True:
            f_up   = 2 * m * k_edge + k_line
            f_down = 2 * (m + 1) * k_edge - k_line
            if f_up[0] > f_max:
                break
            for branch in (f_up, f_down):
                mask_b = branch <= f_max
                if not mask_b.any():
                    continue
                ax.plot(k_line[mask_b], branch[mask_b], "k--",
                        lw=0.8, alpha=0.45, zorder=2,
                        label=(r"Rayleigh Analytic"
                               if not labelled else None))
                labelled = True
            m += 1

        ax.set_xlim(-x_pad, k_edge + x_pad)
        ax.set_ylim(0, f_max)
        ax.set_xticks(np.arange(0.0, k_edge + 1e-6, 0.05))
        ax.set_xlabel(r"$\xi = k\lambda^* / (2\pi)$", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
        ax.tick_params(direction="out", length=4)
        ax.legend(fontsize=9, loc="upper left",
                  framealpha=0.9, edgecolor="0.85")

    axes[0].set_ylabel(r"$f^* = f / (c_R / \lambda^*)$", fontsize=12)

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


# ── main ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Bloch-Floquet dispersion + IPR for ideal cloak unit cells"
    )
    ap.add_argument("--n-kpts",  type=int,   default=50,
                    help="Number of k-points from 0 to BZ edge (default 50)")
    ap.add_argument("--n-eigs",  type=int,   default=40,
                    help="Eigenvalues per k-point (default 40)")
    ap.add_argument("--h-elem",  type=float, default=0.08,
                    help="Global mesh element size (default 0.08)")
    ap.add_argument("--h-fine",  type=float, default=0.03,
                    help="Fine mesh size near cloak (default 0.03)")
    ap.add_argument("--ipr-thr", type=float, default=2.0,
                    help="IPR threshold for surface mode (default 2.0)")
    ap.add_argument("--f-max",   type=float, default=2.2,
                    help="Max normalised frequency on plot (default 2.2)")
    ap.add_argument("--force",   action="store_true",
                    help="Recompute even if cached .npz files exist")
    ap.add_argument("--out-dir", default="output/dispersion",
                    help="Output directory (default output/dispersion)")
    ap.add_argument("--lumped-mass", action="store_true",
                    help="Use row-sum lumped mass (debug: reduces P1 "
                         "high-frequency dispersion stiffening)")
    ap.add_argument("--H-factor", type=float, default=1.0,
                    help="Scale unit-cell height by this factor (debug: "
                         "raise to reduce Dirichlet-bottom interaction at "
                         "low frequency). Triangle dims stay fixed.")
    ap.add_argument("--case", choices=["both", "reference", "ideal_cloak"],
                    default="both",
                    help="Which case(s) to run (default both)")
    # Default: single process, BLAS uses all cores for each eigsh call.
    # Multiple workers cause OOM (each holds SpLU + ARPACK workspace).
    ap.add_argument("--workers", "-j", type=int,
                    default=1,
                    help="Number of parallel threads for k-point solves "
                         "(default: 1, let BLAS parallelise internally)")
    args = ap.parse_args()

    p       = derive_params(H_factor=args.H_factor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Physical params:")
    print(f"  cR = {p['cR']:.2f} m/s,  λ* = {p['lambda_star']},  "
          f"H = {p['H']:.3f},  L_c = {p['L_c']}")
    print(f"  a = {p['a']:.4f},  b = {p['b']:.4f},  c = {p['c']:.4f}")
    print(f"  BZ edge: k_norm = {(np.pi/p['L_c'])/(2*np.pi):.3f}")
    print(f"  Mesh: h_elem={args.h_elem}, h_fine={args.h_fine}")
    print(f"  k-points: {args.n_kpts},  eigenvalues/k: {args.n_eigs}")
    print(f"  Workers: {args.workers}\n")

    L_c    = p["L_c"]
    # Paper §5.1: k_x1 ∈ [π/(100 L_c), π/L_c] — skip Γ to avoid doubly
    # degenerate modes that ARPACK returns in an arbitrary basis (spurious
    # high-IPR points at k=0).
    k_vals = np.linspace(np.pi / (100 * L_c), np.pi / L_c, args.n_kpts)

    sweep_kw = dict(
        n_eigs=args.n_eigs, h_elem=args.h_elem, h_fine=args.h_fine,
        out_dir=out_dir, force=args.force,
        lumped=args.lumped_mass, H_factor=args.H_factor,
        workers=args.workers,
    )

    ref_data = cloak_data = None
    if args.case in ("both", "reference"):
        ref_data = run_sweep("reference", p, k_vals, **sweep_kw)
    if args.case in ("both", "ideal_cloak"):
        cloak_data = run_sweep("ideal_cloak", p, k_vals, **sweep_kw)

    if ref_data is not None and cloak_data is not None:
        suffix = ""
        if args.lumped_mass: suffix += "_lumped"
        if args.H_factor != 1.0: suffix += f"_H{args.H_factor:g}"
        plot_dispersion(
            ref_data, cloak_data, p,
            out_path=out_dir / f"dispersion_comparison{suffix}.png",
            ipr_threshold=args.ipr_thr,
            f_max=args.f_max,
        )


if __name__ == "__main__":
    main()
