"""Fill the upsampling holes that diffusion missed, by adjoint-guided inverse design.

``upsample_targets`` proposed 13 877 coordinates in the 5-D condition space
``(C11, C22, C12, C66, vol)`` that the CA dataset does not cover.  The diffusion
generator was then asked to hit them; after homogenisation and re-thinning,
``stiffness_tri6_uniform_v2.h5`` still has **no sample within the thinning radius
of 8 695 of them** (7 367 interior holes + 1 328 frontier).  Those are the ones a
conditional generator cannot reach by sampling, because it only interpolates what
it was trained on.

Each run starts from the nearest mirror-symmetric dataset cell and edits it, one
small group of pixels at a time, using the FEM adjoint from
``inverse_design.compute_flat4`` to rank which pixels to flip.  See
:func:`refine_pixels` for why the module's MLP-field optimiser
(``run_cell_design``) is not used by default -- in short, it cannot express a
move smaller than "the whole interface at once", and these designs need moves of
a few pixels.  It is still reachable with ``--neural-iters``.

Fidelity
--------
The dataset is homogenised with **quadratic TRI6 on a 100x100 mesh** (2 elements
per pixel).  ``inverse_design.build_homog_setup``'s old default -- linear TRI3 at
1 element/pixel -- is over-stiff by 5-120 % depending on the component (worst on
C66 and on thin, low-vf cells), so a design optimised against it misses its
target under the dataset homogeniser by far more than the hole is wide.  Every
run here uses ``ele_type="TRI6"`` in two stages:

* **coarse** ``mesh_N=50``  -- 1 element/pixel, ~4.7 s per value+grad.  Runs
  0.2-7 % stiff, which is corrected by a measured per-geometry bias factor;
  cheap enough to do the geometry-forming work.
* **fine**   ``mesh_N=100`` -- the dataset's exact mesh, ~43 s per value+grad.
  Reproduces stored dataset values to ~1e-9 relative, so at this stage the
  objective *is* the validation number.

Success criterion
-----------------
Two conditions, both checked on the binarised cell re-homogenised by
``calc_fem_hifi`` at TRI6/N=100 -- the dataset's own code path, not the
optimiser's model:

1. **hit** -- the result lands within the thinning radius of the target's
   rank-space centre.  This is the same test that declared the diffusion pass to
   have failed.
2. **new support** -- it is further than that radius from every existing sample.
   Without this a design can hit its target and still be discarded by
   blue-noise thinning as a duplicate of a row that is already there.

Condition 2 is the binding one.  The median unfilled hole centre sits just
1.15 radii from the nearest existing sample, so the region that satisfies both
is the far side of the hole, and ``--min-isolation`` skips the holes with no room
at all.

Usage
-----

    python -m dataset.cellular_chiral.fill_gaps_inverse \
        -o output/ca_bulk_squared/inverse_fill --n-targets 4

    # re-run specific target ids from upsample_targets_v1.csv
    python -m dataset.cellular_chiral.fill_gaps_inverse --target-ids 8 1274 5561

    # work the whole queue with N workers; resumable, safe to kill and restart
    for w in $(seq 0 9); do
        python -m dataset.cellular_chiral.fill_gaps_inverse --all \
            --worker $w --n-workers 10 -o output/ca_bulk_squared/inverse_fill &
    done

Then summarise and export with ``fill_gaps_report``.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

FEATURES = ("C11", "C22", "C12", "C66", "vol")
RHO_SOLID = 2300.0


# --------------------------------------------------------------------------- #
# rank-space plumbing (knots come straight out of the thinning .npz)
# --------------------------------------------------------------------------- #
def load_knots(subset_npz: Path):
    z = np.load(subset_npz, allow_pickle=False)
    return [(z[f"knot_v_{f}"], z[f"knot_q_{f}"]) for f in FEATURES], float(z["radius"])


def to_rank(P: np.ndarray, knots) -> np.ndarray:
    out = np.empty_like(np.atleast_2d(P), dtype=np.float64)
    A = np.atleast_2d(P)
    for j in range(A.shape[1]):
        v, q = knots[j]
        out[:, j] = np.interp(A[:, j], v, q)
    return out


def rank_slopes(P: np.ndarray, knots, h: float = 1e-3) -> np.ndarray:
    """d(rank)/d(value) at ``P`` — the local density of the dataset per axis.

    Central difference on the stored empirical CDF, with the step taken as a
    fraction ``h`` of each axis' own knot range so it is scale-free.
    """
    g = np.empty(len(knots))
    for j, (v, q) in enumerate(knots):
        d = h * (v[-1] - v[0])
        g[j] = (np.interp(P[j] + d, v, q) - np.interp(P[j] - d, v, q)) / (2 * d)
    return g


def rank_aligned_weights(target: np.ndarray, knots) -> tuple[np.ndarray, float]:
    """Loss weights that make ``flat4_loss`` + rho term ~= squared rank distance.

    The acceptance test is Euclidean distance in rank space, but ``flat4_loss``
    measures *relative* error, and the two disagree badly: C12 near zero has a
    huge relative error for a rank displacement of nothing, while C11 sits where
    the dataset is dense and a 2 % move crosses several percentiles.  Optimising
    relative error therefore spends the budget on the wrong components.

    Linearising rank around the target, ``rank_j - rank_j(t_j) ~= g_j (C_j - t_j)
    = (g_j t_j) * rel_j``, so weighting component ``j`` by ``(g_j t_j)^2`` turns
    the loss into the squared rank distance — directly comparable to ``radius^2``.
    """
    g = rank_slopes(target, knots)
    w = (g * target) ** 2
    return w[:4], float(w[4])


# --------------------------------------------------------------------------- #
# target selection
# --------------------------------------------------------------------------- #
def _is_d2(cell: np.ndarray) -> bool:
    """True iff the 50x50 cell is the squared (D2 mirror) assembly of its own quadrant."""
    q = cell[:25, :25]
    return np.array_equal(
        cell, np.block([[q, q[:, ::-1]], [q[::-1], q[::-1, ::-1]]])
    )


def select_failed(dataset: Path, targets_npz: Path, subset_npz: Path,
                  min_abs_c12: float = 0.0):
    """Return (info dict, per-target records) for holes the diffusion pass missed.

    Only interior holes (``region == 0``) are candidates.  The frontier targets
    are extrapolations past the sampled envelope, so several of their rank
    coordinates sit at the clamped end of the empirical CDF and a rank-space
    distance stops discriminating; they need their own acceptance test.

    ``min_abs_c12`` is available for the relative-error objective, where a target
    with C12 ~ 0 would absorb the whole budget for no rank-space movement.  It
    defaults to off, because the objective used here is the rank distance itself.
    """
    knots, radius = load_knots(subset_npz)

    t = np.load(targets_npz, allow_pickle=False)
    tab, cols = t["targets"], [str(c) for c in t["columns"]]
    col = {c: i for i, c in enumerate(cols)}
    P = tab[:, [col[c] for c in FEATURES]]
    centres = t["rank_centres"]
    region = t["region"].astype(int)
    encl = tab[:, col["enclosure"]]
    tid = tab[:, col["target_id"]].astype(int)

    with h5py.File(dataset, "r") as f:
        Xv = np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)
    Rv = to_rank(Xv, knots)
    d_all, _ = cKDTree(Rv).query(centres, k=1, workers=-1)

    failed = d_all > radius
    ok = failed & (region == 0) & (np.abs(P[:, 2]) >= min_abs_c12)
    info = dict(n_targets=len(P), n_failed=int(failed.sum()),
                n_failed_interior=int((failed & (region == 0)).sum()),
                n_candidates=int(ok.sum()), radius=radius)
    return info, dict(tid=tid, P=P, centres=centres, encl=encl, d_all=d_all,
                      cand=np.flatnonzero(ok), Rv=Rv, Xv=Xv, knots=knots,
                      radius=radius)


def pick_stratified(rec, n: int, seed: int = 0) -> np.ndarray:
    """``n`` candidates spread over volume fraction, best-enclosed first in each bin."""
    cand = rec["cand"]
    vol = rec["P"][cand, 4]
    edges = np.quantile(vol, np.linspace(0, 1, n + 1))
    out = []
    for b in range(n):
        m = (vol >= edges[b]) & (vol <= edges[b + 1] if b == n - 1 else vol < edges[b + 1])
        idx = cand[m]
        if not len(idx):
            continue
        out.append(idx[np.lexsort((-rec["d_all"][idx], -rec["encl"][idx]))[0]])
    return np.array(out, dtype=int)


def nearest_symmetric_cells(dataset: Path, rec, i: int, n_seeds: int = 1,
                            k: int = 64, tree=None):
    """The ``n_seeds`` closest D2-symmetric dataset cells to target ``i``.

    Only D2-symmetric cells can seed the design: the optimiser works on a
    quadrant and mirrors it, so a seed that is not already its own mirror would
    be silently altered before the first iteration.  Two thirds of the
    diffusion-generated rows in v2 fail that test, hence the search over
    neighbours rather than taking the single nearest row.
    """
    out = []
    with h5py.File(dataset, "r") as f:
        t = cKDTree(rec["Rv"]) if tree is None else tree
        d, idx = t.query(rec["centres"][i], k=k, workers=-1)
        for dist, row in zip(np.atleast_1d(d), np.atleast_1d(idx)):
            cell = f["cells"][int(row)]
            if _is_d2(cell):
                out.append((int(row), float(dist), cell,
                            np.array([f[c][int(row)] for c in FEATURES])))
                if len(out) >= n_seeds:
                    break
    if not out:
        raise RuntimeError(f"no D2-symmetric cell among the {k} nearest neighbours")
    return out


# --------------------------------------------------------------------------- #
# one design run
# --------------------------------------------------------------------------- #
def soft_seed_quadrant(cell: np.ndarray, amp: float = 3.0, width: float = 2.5) -> np.ndarray:
    """Seed occupancy ``sigmoid(amp * tanh(sdf / width))`` from the cell's own SDF.

    Handing the raw binary seed to ``make_cell_neural_field`` puts *every* pixel
    at the same logit distance from the 0.5 threshold, so as the MLP output grows
    they all cross it within a step or two of each other: the design sits frozen
    for ~50 iterations and then flips as a block, severing the load path (C11
    fell by 4 orders of magnitude in a measured run).  Lowering the learning rate
    only postpones the cliff.

    Grading the seed by the signed distance to the material interface breaks that
    degeneracy in the way shape optimisation wants: a pixel touching the boundary
    starts at logit ~1.1 and flips readily, one buried 6 pixels deep starts at
    ~2.9 and essentially does not.  The design then evolves by moving interfaces,
    a few pixels at a time.  Thresholding at 0.5 still returns the seed exactly,
    since sdf > 0 iff the pixel is solid.

    The distance transform is taken on the periodic 3x3 tiling, so pixels near
    the cell border see their true neighbours across the wrap.
    """
    from scipy.ndimage import distance_transform_edt

    c = (np.asarray(cell) > 0).astype(np.uint8)
    tile = np.tile(c, (3, 3))
    H, W = c.shape
    sdf = (distance_transform_edt(tile) - distance_transform_edt(1 - tile))[H:2 * H, W:2 * W]
    return 1.0 / (1.0 + np.exp(-amp * np.tanh(sdf / width)))


def _assemble(q: np.ndarray) -> np.ndarray:
    """25x25 quadrant -> 50x50 squared (D2 mirror) canvas."""
    return np.block([[q, q[:, ::-1]], [q[::-1], q[::-1, ::-1]]])


def _periodic_labels(cell: np.ndarray) -> np.ndarray:
    """Connected components of the solid phase on the periodic torus (0 = void)."""
    from scipy.ndimage import label

    lab, n = label(cell > 0)
    if n <= 1:
        return lab
    parent = np.arange(n + 1)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in list(zip(lab[0], lab[-1])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[ra] = rb
    return np.array([root(v) if v else 0 for v in range(n + 1)])[lab]


def _shape_ok(cell: np.ndarray, max_components: int, gate_width: int = 5) -> bool:
    """Reject geometries with new floating material or a severed gate network.

    Left unconstrained the flipper discovers that isolated single pixels are a
    cheap way to buy volume fraction: they move ``vol`` toward its target while
    contributing nothing to stiffness.  A first run ended with 17 disconnected
    solid components -- properties on target, geometry unbuildable.  Checking
    this before the solve is also a speed-up, since the rejected candidate never
    reaches the FEM.
    """
    lab = _periodic_labels(cell)
    if len(set(lab.ravel()) - {0}) > max_components:
        return False
    H, W = cell.shape
    N = H // 2
    gs, ge = (N - gate_width) // 2, (N - gate_width) // 2 + gate_width
    sets = [
        set(lab[0, gs:ge]) | set(lab[0, W - ge:W - gs]),
        set(lab[-1, gs:ge]) | set(lab[-1, W - ge:W - gs]),
        set(lab[gs:ge, 0]) | set(lab[H - ge:H - gs, 0]),
        set(lab[gs:ge, -1]) | set(lab[H - ge:H - gs, -1]),
    ]
    sets = [s - {0} for s in sets]
    return bool(all(sets) and set.intersection(*sets))


def refine_pixels(q0, setup, centre, knots, bias=None, max_iters=60, k0=16,
                  r_in=None, r_out=0.0, w_hole=1.0, w_keep=0.0, hard_r_in=None,
                  label="", probe=12, nn_tree=None, max_components=None,
                  axis_w=None, n_pairs=3):
    """Greedy adjoint-guided pixel flips on the quadrant.  Returns (q, history).

    Why not the neural field
    ------------------------
    ``run_cell_design`` moves the design by growing a smooth MLP output field
    until pixels cross the 0.5 threshold.  Measured on these targets, that gives
    no usable step size: at lr 1e-4 nothing crosses in 80 iterations, and at
    3e-4/1e-3 the whole interface crosses within one step of itself -- every
    ligament thins by a pixel simultaneously and C22 falls by four orders of
    magnitude.  Halving the learning rate does not help, because *which* pixels
    cross together is set by the spatial shape of the MLP output, not by how fast
    it grows; the same block just crosses later.  The seed cells here are already
    within a few percent of their target, so the design needs moves of a few
    pixels, which that parameterisation cannot express.

    What this does instead
    ----------------------
    One adjoint solve gives dL/d(pixel) for every pixel at once.  Flipping
    quadrant pixel (i,j) toggles four canvas pixels (D2 mirror), so its
    sensitivity is the sum over the four images, and the predicted change is
    ``s * (1 - 2q)``.  Flip the ``k`` most promising, re-evaluate exactly, and
    accept only if the true objective dropped -- otherwise halve ``k`` and retry.
    ``k`` therefore adapts: large while the design is far away, down to a single
    pixel near the optimum.

    The linearised sensitivity ranks flips well but predicts their size poorly --
    a flip changes a pixel by a full unit, far outside where the derivative is
    valid -- so when no prefix of the ranking works, the next ``probe`` flips are
    tried one at a time before declaring a local minimum.  Every accepted move is
    verified by a real solve, so a mis-ranked flip costs a forward pass, never a
    wrong answer.

    Objective
    ---------
    One hinge per goal, each zero once its goal is met::

        L = w_hole * max(0, |r - centre| - r_in)^2
          + w_keep * max(0, r_out - d_nn)^2

    with ``d_nn`` the rank distance to the nearest point already in the dataset.
    ``hard_r_in``, when set, additionally *rejects* any move that leaves
    ``|r - centre|`` above it, whatever it does to the cost.

    These are meant to be run one at a time, not summed -- see
    :func:`design_one`.  Both goals in one weighted sum lets the search settle at
    a compromise that satisfies neither, and it does: the keep-out term starts at
    its maximum (the seed *is* a dataset row, so ``d_nn`` = 0) and dominates, so
    the design is dragged away from its target before it ever gets close.
    Measured on the first two queue targets that way: 0.0329 -> 0.0406 and
    0.0361 -> 0.0409 against a radius of 0.0276, i.e. both ended further from
    their hole than the seed they started from.  Running the hole hinge alone
    first, then the keep-out hinge under a hard "stay in the hole" constraint,
    keeps what the first phase won.

    The gradient w.r.t. the stiffness components is the chain rule through the
    empirical CDF, ``dL/dr_j * d(rank_j)/d(C_j)``, with the slope re-evaluated at
    the current point each iteration.

    ``bias`` (coarse/fine ratio) divides the forward values, so a cheap mesh can
    be driven toward what the dataset mesh will report.
    """
    import jax, jax.numpy as jnp
    from dataset.cellular_chiral.inverse_design import compute_flat4

    b = np.ones(4) if bias is None else np.asarray(bias, float)
    frozen = np.zeros_like(q0, dtype=bool)
    frozen[0, :] = True     # quadrant top row / left column are the cell border
    frozen[:, 0] = True

    def _objective(r):
        """(L, search direction, d_centre, d_nn) at rank-space point ``r``.

        ``axis_w`` scales the returned gradient per coordinate but never the cost.
        The cost decides which moves are *accepted*, and that has to stay the
        acceptance test itself -- the plain 5-D rank distance -- or the search can
        report success on something the dataset will reject.  The gradient only
        decides which flips are *proposed*, so tilting it toward the axes that
        block the misses (C11 and C66) can widen the search without ever letting
        a worse design through.
        """
        diff = r - centre
        dc = float(np.linalg.norm(diff))
        L, grad = 0.0, np.zeros_like(r)
        if w_hole > 0.0 and dc > r_in:
            L += w_hole * (dc - r_in) ** 2
            grad = grad + 2.0 * w_hole * (dc - r_in) * diff / max(dc, 1e-12)
        dn = np.inf
        if w_keep > 0.0 and nn_tree is not None and r_out > 0.0:
            d, j = nn_tree.query(r, k=1)
            dn = float(d)
            if dn < r_out:
                short = r_out - dn
                L += w_keep * short ** 2
                # push directly away from the crowding neighbour
                away = (r - nn_tree.data[j]) / max(dn, 1e-12)
                grad = grad - 2.0 * w_keep * short * away
        # Tilt only the hole-seeking direction.  Phase B's direction is "away
        # from the crowding neighbour", which is a geometric fact about where the
        # dataset already is; scaling its axes distorts the escape route.  Applied
        # to both phases the weights drove one test target from 0.0334 to 0.0161
        # on distance-to-hole while giving back its clearance, 0.0289 -> 0.0202,
        # trading an almost-hit that had room for a solid hit that had none.
        if axis_w is not None and w_hole > 0.0:
            grad = grad * axis_w
        return L, grad, dc, dn

    def evaluate(q, want_grad):
        canvas = jnp.array(_assemble(q).astype(np.float64))
        if want_grad:
            flat4, vjp = jax.vjp(lambda c: compute_flat4(c, setup), canvas)
        else:
            flat4, vjp = compute_flat4(canvas, setup), None
        pt = np.concatenate([np.asarray(flat4) / b, [float(q.mean())]])
        L, dL_dr, dc, dn = _objective(to_rank(pt[None], knots)[0])
        pt = np.concatenate([pt, [dc, dn]])            # carry the two distances
        if not want_grad:
            return L, pt, None
        sl = rank_slopes(pt[:5], knots)
        g = np.asarray(vjp(jnp.array(dL_dr[:4] * sl[:4] / b))[0])
        g = g + dL_dr[4] * sl[4] / g.size               # volume-fraction path
        n = q.shape[0]
        s = (g[:n, :n] + g[:n, n:][:, ::-1] + g[n:, :n][::-1] + g[n:, n:][::-1, ::-1])
        return L, pt, s

    def flipped(q, take):
        c = q.copy()
        c.ravel()[take] ^= 1
        return c

    q = np.array(q0, dtype=np.uint8)
    if max_components is None:
        max_components = max(len(set(_periodic_labels(_assemble(q)).ravel()) - {0}), 1)
    L, pt, _ = evaluate(q, False)
    hist = [(L, pt)]
    goal = (f"want d_target<={r_in:.4f}" if w_hole > 0 else
            f"want d_nearest>={r_out:.4f}, holding d_target<={hard_r_in:.4f}")
    print(f"  {label}start: cost {np.sqrt(L):.4f}  d_target={pt[5]:.4f}  "
          f"d_nearest={pt[6]:.4f}  [{goal}; <= {max_components} component(s)]",
          flush=True)
    if L <= 0.0:
        print(f"  {label}goal already met — nothing to do", flush=True)
        return q, hist
    _, _, s = evaluate(q, True)

    k = k0
    for it in range(max_iters):
        pred = (s * (1 - 2 * q.astype(float)))          # predicted dL per flip
        pred[frozen] = np.inf
        flat = pred.ravel()
        order = np.argsort(flat)

        # candidate moves, best guess first: prefixes of the ranking, then
        # singles, then volume-neutral add/remove pairs.
        #
        # The pairs matter because ``vol`` is one of the five coordinates and is
        # usually the binding one: every pure add or remove moves it by 4/2500,
        # so once the volume is right, any single flip spoils it and the search
        # reports a local minimum with the stiffness still well off target.  A
        # simultaneous add and remove leaves the volume untouched and changes
        # only the stiffness anisotropy -- a direction no single flip can express.
        ks, kk = [], k
        while kk >= 1:
            ks.append(kk)
            kk //= 2
        free = order[~frozen.ravel()[order]]
        solid = q.ravel()[free].astype(bool)
        adds, rems = free[~solid][:n_pairs], free[solid][:n_pairs]
        moves = ([order[:n] for n in ks]
                 + [order[m:m + 1] for m in range(1, probe)]
                 + [np.array([a, r]) for a in adds for r in rems])

        taken = None
        solves = 0
        for trial, take in enumerate(moves):
            # Multi-pixel prefixes are only meaningful while the linearisation
            # says they help.  Single flips are tried regardless of sign: a flip
            # moves a pixel by a whole unit, far outside where a derivative is
            # valid, and near a local optimum the prediction is wrong often
            # enough that skipping the positives is what ends most runs early.
            if trial < len(ks) and flat[take].min() >= 0:
                continue
            cand = flipped(q, take)
            if not _shape_ok(_assemble(cand), max_components):
                continue                                # free rejection, no FEM
            # spend the adjoint on the first solve: it is the one most likely to
            # be accepted, and then its sensitivities are already in hand
            Lc, ptc, sc = evaluate(cand, want_grad=(solves == 0))
            solves += 1
            if hard_r_in is not None and ptc[5] > hard_r_in:
                continue                                # would leave the hole
            if Lc < L:
                taken = (take, Lc, ptc, sc)
                k = len(take) if trial < len(ks) else 1
                break
        if taken is None:
            print(f"  {label}iter {it:3d}: no improving flip among "
                  f"{len(moves)} candidates — local minimum, stop", flush=True)
            break

        take, L, pt, s = taken
        q = flipped(q, take)
        hist.append((L, pt))
        print(f"  {label}iter {it:3d}: {len(take):3d} px  cost {np.sqrt(L):.4f}  "
              f"d_target={pt[5]:.4f}  d_nearest={pt[6]:.4f}  "
              + "  ".join(f"{c}={v:.4g}" for c, v in zip(FEATURES, pt)), flush=True)
        if L <= 0.0:
            print(f"  {label}goal met — stop", flush=True)
            break
        if s is None:
            _, _, s = evaluate(q, True)
        k = min(max(k, 1) * 2, k0)                      # grow back after a success
    return q, hist


def design_one(target_flat4, target_vol, init_cell, setups, args,
               knots=None, centre=None, radius=None, nn_tree=None):
    """Coarse-mesh then dataset-mesh inverse design.  Returns (cell, history).

    Stage 1 runs on ``mesh_N=50`` with the target divided by the measured
    coarse/fine bias, so the cheap model aims where the dataset model will land;
    stage 2 runs on the dataset's own ``mesh_N=100``, where the objective *is*
    the validation number.

    ``--neural-iters`` optionally prepends ``run_cell_design`` (the MLP field).
    It defaults to 0 because on these targets it cannot take a useful step --
    see :func:`refine_pixels` for the measurement.
    """
    import jax.numpy as jnp
    from dataset.cellular_chiral.inverse_design import compute_flat4

    coarse, fine = setups
    target_flat4 = np.asarray(target_flat4, float)
    target = np.concatenate([target_flat4, [float(target_vol)]])

    def bias(cell) -> np.ndarray:
        """coarse(cell) / fine(cell) — what mesh_N=50 reads high by, on this geometry.

        The coarse mesh is 0.2-7 % stiff relative to the dataset mesh, the same
        order as the correction the design has to make, so optimising the coarse
        model against the raw target aims at the wrong place.  Dividing by this
        ratio makes the cheap stage aim where the expensive stage will land.
        """
        c = jnp.array(np.asarray(cell, dtype=np.float64))
        return np.asarray(compute_flat4(c, coarse)) / np.asarray(compute_flat4(c, fine))

    q = np.array(init_cell[:25, :25], dtype=np.uint8)
    history = {}

    if args.neural_iters:
        from dataset.cellular_chiral.inverse_design import (
            make_cell_neural_field, run_cell_design)
        w4, w_rho = (rank_aligned_weights(target, knots) if args.rank_weights
                     else (np.full(4, 0.25), args.weight_rho))
        theta0, nf = make_cell_neural_field(
            n_fourier=args.n_fourier, hidden_size=args.hidden, n_layers=args.layers,
            seed=args.seed, initial_soft_eps=1e-3,
            initial_quadrant=soft_seed_quadrant(init_cell, args.sdf_amp,
                                                args.sdf_width)[:25, :25])
        b = bias(_assemble(q))
        print("\n  ---- stage 0: neural field (run_cell_design) on mesh_N=50 ----",
              flush=True)
        r = run_cell_design(
            nf, coarse, target_flat4 * b, theta0, target_rho=RHO_SOLID * target_vol,
            weights=w4, weight_rho=w_rho, weight_conn=args.weight_conn,
            n_iters=args.neural_iters, lr=args.lr, lr_end=args.lr * 0.1,
            beta_init=1.0, beta_final=1.0, beta_warmup_frac=0.0, beta_ramp_frac=0.0,
            tol=args.tol, straight_through=True,
            revert_on_blowup=args.revert_on_blowup or None)
        q = nf.binarize(r.best_theta)[:25, :25]
        history["neural_loss"] = np.array(r.loss_history)

    b = bias(_assemble(q))
    print(f"\n  ---- stage 1: pixel refinement, TRI6 @ mesh_N=50 ----"
          f"\n  coarse/fine bias on the seed: "
          + "  ".join(f"{c}={v:.4f}" for c, v in zip(FEATURES[:4], b)), flush=True)
    # Aim slightly inside the ball on the coarse mesh: the bias correction is
    # good to ~1 %, not exact, so stopping exactly on the boundary there would
    # land outside it on the dataset mesh about half the time.
    # Each mesh runs the two goals in sequence: reach the hole, then move within
    # it until the design is clear of every existing sample.  Phase B's hard
    # constraint is what makes the order safe -- it can never give back the hit
    # phase A won.  The coarse mesh aims deeper than acceptance needs because its
    # model is 0.2-7 % off the dataset mesh even after the bias correction.
    axis_w = (np.array(args.axis_weights) if args.axis_weights else None)
    common = dict(k0=args.k0, probe=args.probe, nn_tree=nn_tree,
                  axis_w=axis_w, n_pairs=args.n_pairs)
    half = max(args.coarse_iters // 2, 1)
    q, h1 = refine_pixels(q, coarse, centre, knots, bias=b, max_iters=half,
                          r_in=args.stop_frac * radius, w_hole=1.0, w_keep=0.0,
                          label="coarse A ", **common)
    # the bias drifts as the geometry moves, so re-measure it before phase B
    b = bias(_assemble(q))
    print(f"  bias re-estimated: "
          + "  ".join(f"{c}={v:.4f}" for c, v in zip(FEATURES[:4], b)), flush=True)
    # Hold whatever phase A reached, even if it fell short of its own goal --
    # a hard bound tighter than the current point rejects every move including
    # the ones that would help, and phase B does nothing at all.
    q, h1b = refine_pixels(q, coarse, centre, knots, bias=b,
                           max_iters=args.coarse_iters - half,
                           r_out=args.keep_out * radius, w_hole=0.0, w_keep=1.0,
                           hard_r_in=max(args.stop_frac * radius, h1[-1][1][5]),
                           label="coarse B ", **common)
    history["coarse"] = np.array([L for L, _ in h1 + h1b])

    print("\n  ---- stage 2: pixel refinement, TRI6 @ mesh_N=100 (dataset mesh) ----",
          flush=True)
    # On the dataset mesh the thresholds *are* the acceptance test, so no margin
    # is needed beyond a hair of tie-breaking at the boundary.
    # A probe on the dataset mesh costs ~22 s a shot, so it is kept short here;
    # the coarse mesh is where the wide search belongs.
    fine_common = dict(k0=max(args.k0 // 4, 4), probe=args.fine_probe,
                       nn_tree=nn_tree, axis_w=axis_w, n_pairs=args.n_pairs)
    q, h2 = refine_pixels(q, fine, centre, knots, bias=None,
                          max_iters=args.fine_iters,
                          r_in=args.fine_stop_frac * radius, w_hole=1.0, w_keep=0.0,
                          label="fine A   ", **fine_common)
    q, h2b = refine_pixels(q, fine, centre, knots, bias=None,
                           max_iters=args.fine_iters,
                           r_out=1.01 * radius, w_hole=0.0, w_keep=1.0,
                           hard_r_in=max(args.fine_stop_frac * radius, h2[-1][1][5]),
                           label="fine B   ", **fine_common)
    history["fine"] = np.array([L for L, _ in h2 + h2b])
    return _assemble(q), history


def connectivity_report(cell: np.ndarray, gate_width: int = 5) -> dict:
    """Exact binary check that the solid phase still ties the four gates together.

    ``gate_connectivity_loss`` is a differentiable surrogate that never reaches 0
    (its sharpened flood decays ~0.993 per hop, so a perfectly connected 50x50
    cell scores ~0.9), which makes it useless as a pass/fail test and dangerous
    as a loss term -- at the default weight it is ~300x the stiffness term.  This
    is the honest version: connected components of the solid phase on the
    periodic torus, then whether one component reaches all four gates.
    """
    lab = _periodic_labels(cell)
    n = len(set(lab.ravel()) - {0})
    return dict(gates_connected=bool(n and _shape_ok(cell, n, gate_width)),
                n_components=int(n))


_SEEN: dict[str, list | None] = {}


def _rss_mb() -> float:
    """Resident set size of this worker, MB.

    Logged per target because the first long run degraded from 24 targets/hour to
    zero over ~36 h while holding 2.6 GB per worker, and nothing in the per-target
    output showed it coming.  A number in the log is what makes that visible next
    time.
    """
    try:
        with open("/proc/self/status") as fh:
            return int(fh.read().split("VmRSS:")[1].split()[0]) / 1024
    except (OSError, IndexError, ValueError):
        return float("nan")


def _accepted_points(out_dir: Path) -> np.ndarray:
    """Rank-space points of every accepted design written so far, by any worker.

    Files already parsed are cached, so re-reading the directory once per target
    stays cheap as the run grows into the thousands.  Results are written via a
    rename, so a file seen here is never half-written.
    """
    for f in out_dir.glob("target_*.json"):
        if f.name in _SEEN:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue                                  # retry on the next pass
        _SEEN[f.name] = d["rank"] if (d.get("accepted") and "rank" in d) else None
    pts = [v for v in _SEEN.values() if v is not None]
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 5))


def _is_better(payload_hit: bool, payload_acc: bool, d_hit: float, prev: dict) -> bool:
    """Does a retry beat the stored result?  Acceptance first, then distance."""
    if payload_acc != prev.get("accepted", False):
        return payload_acc
    if payload_hit != prev.get("hit", False):
        return payload_hit
    return d_hit < prev["d_hit"]


def validate(cell: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Re-homogenise with the dataset's own code path: TRI6, N=100.

    Returns (flat4, vf, C_eff 4x4).  This is ``bulk_stiffness``'s exact call, so
    the numbers are directly comparable with the stored dataset columns and the
    full ``C_eff`` can be written straight into a v3 HDF5.
    """
    from dataset.cellular_chiral.bulk_stiffness import _ortho_params_from_C
    from dataset.stiffness.calc_fem_hifi import compute_stiffness_hifi

    C, _rho, vf = compute_stiffness_hifi(cell.astype(np.uint8), N=100, ele_type="TRI6")
    return np.array(_ortho_params_from_C(C)), float(vf), np.asarray(C)


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-d", "--dataset", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("-t", "--targets", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/upsample_targets_v1.npz"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/inverse_fill"))
    p.add_argument("--n-targets", type=int, default=4)
    p.add_argument("--target-ids", type=int, nargs="*", default=None,
                   help="explicit target_id values from upsample_targets_v1.csv")
    p.add_argument("--only", type=int, default=None,
                   help="run only the k-th selected target (for parallel driving)")
    p.add_argument("--all", action="store_true",
                   help="work the whole unfilled interior-hole queue, best-enclosed "
                        "first; resumable and safe to run as several workers")
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1,
                   help="with --all, take every n-th target starting at --worker")
    p.add_argument("--max-targets", type=int, default=0)
    p.add_argument("--min-isolation", type=float, default=1.1,
                   help="with --all, skip holes whose centre is closer than this "
                        "multiple of the radius to existing data: there is no room "
                        "there for a design that both lands in the hole and "
                        "survives thinning")
    p.add_argument("--min-abs-c12", type=float, default=0.0)
    p.add_argument("--n-seeds", type=int, default=2,
                   help="symmetric dataset cells to try as starting geometry, "
                        "nearest first; stops at the first accepted design")
    p.add_argument("--no-dynamic-dedup", dest="dynamic_dedup", action="store_false",
                   help="do not skip targets already covered by an earlier design")
    p.add_argument("--retry-unaccepted", action="store_true",
                   help="re-run only the targets an earlier pass failed to land, keeping the new result only if it beats the stored one. Pair with a different search setting, e.g. --axis-weights 3 1 1 3 1")
    p.add_argument("--dry-run", action="store_true",
                   help="print the queue and exit without building the FEM setups")
    p.add_argument("--coarse-iters", type=int, default=80)
    p.add_argument("--fine-iters", type=int, default=25)
    p.add_argument("--k0", type=int, default=16,
                   help="max pixels flipped per refinement step (halved on rejection)")
    p.add_argument("--probe", type=int, default=28,
                   help="individual flips tried on the coarse mesh, in sensitivity "
                        "order, once no prefix of the ranking improves")
    p.add_argument("--axis-weights", type=float, nargs=5, default=None,
                   metavar=("C11", "C22", "C12", "C66", "VOL"),
                   help="tilt the PROPOSAL gradient per axis (accept/reject stays on the true unweighted rank distance); e.g. 3 1 1 3 1 to push on the axes that block most misses")
    p.add_argument("--n-pairs", type=int, default=3,
                   help="top-N adds x top-N removes tried as volume-neutral pair moves; every single flip shifts vol by ~0.07 radii, which is what blocks stiffness-only moves")
    p.add_argument("--fine-probe", type=int, default=10,
                   help="same, on the dataset mesh, where each try costs ~22 s")
    p.add_argument("--keep-out", type=float, default=1.06,
                   help="coarse-stage target for the distance to the nearest "
                        "existing dataset point, as a multiple of the thinning "
                        "radius; anything closer than 1.0 is discarded by thinning. "
                        "Keep this close to 1: a hole centre typically sits only "
                        "1.02-1.25 radii from the nearest sample, so a larger "
                        "demand is unsatisfiable inside the hole and the search "
                        "abandons the hole to chase it")
    p.add_argument("--w-keep", type=float, default=1.0,
                   help="weight of the keep-out hinge relative to the hole hinge")
    p.add_argument("--stop-frac", type=float, default=0.7,
                   help="coarse-stage stop, as a fraction of the radius; leaves "
                        "margin for the coarse/fine model difference")
    p.add_argument("--fine-stop-frac", type=float, default=0.97,
                   help="dataset-mesh stop, as a fraction of the radius")
    p.add_argument("--neural-iters", type=int, default=0,
                   help="iterations of the MLP-field optimiser (run_cell_design) "
                        "before pixel refinement; 0 = skip, see refine_pixels")
    p.add_argument("--lr", type=float, default=3e-4,
                   help="learning rate for the optional neural-field stage")
    p.add_argument("--weight-rho", type=float, default=1.0)
    p.add_argument("--weight-conn", type=float, default=0.0,
                   help="weight of the differentiable gate-connectivity surrogate. "
                        "Default 0: it has a large non-zero floor even for valid "
                        "dataset cells, so at any useful weight it swamps the "
                        "stiffness loss. Connectivity is checked exactly on the "
                        "final binary cell instead.")
    p.add_argument("--sdf-amp", type=float, default=3.0,
                   help="max |logit| of the distance-graded seed (deep interior)")
    p.add_argument("--sdf-width", type=float, default=2.5,
                   help="pixels over which the seed logit saturates away from the "
                        "material interface; smaller = only the boundary is mobile")
    p.add_argument("--revert-on-blowup", type=float, default=3.0,
                   help="roll back and halve the lr when a step's loss exceeds this "
                        "multiple of the best so far (0 = never)")
    p.add_argument("--tol", type=float, default=1e-4,
                   help="early stop on max relative error; the acceptance test is "
                        "the rank-space ball, so this is left tight enough to "
                        "effectively defer to it")
    p.add_argument("--no-rank-weights", dest="rank_weights", action="store_false",
                   help="weight the four stiffness components equally in relative "
                        "error instead of by their rank-space sensitivity")
    p.add_argument("--n-fourier", type=int, default=32)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    import logging

    import jax
    jax.config.update("jax_enable_x64", True)

    # Silence jax_fem's per-solve DEBUG chatter with a *filter*, not setLevel:
    # importing jax_fem runs setup_logger(), which unconditionally forces its
    # logger back to DEBUG, so any setLevel here is undone by the import below.
    # Filters survive setLevel, so this holds whatever the import order becomes.
    #
    # This is not cosmetic.  A wedged Newton loop (jax_fem/solver.py has no
    # iteration cap) emits DEBUG lines forever: attempt 3 wrote 895 MB per stuck
    # worker, and -- far worse -- kept the log mtime fresh, which is exactly the
    # liveness signal fill_gaps_launch.sh's watchdog uses.  The run sat wedged
    # for 10 days with the watchdog watching a log that never stopped moving.
    logging.getLogger("jax_fem").addFilter(
        lambda r: r.levelno >= logging.WARNING)

    from dataset.cellular_chiral.inverse_design import build_homog_setup

    info, rec = select_failed(args.dataset, args.targets, args.subset,
                              min_abs_c12=args.min_abs_c12)
    radius = rec["radius"]
    print(f"targets {info['n_targets']}  |  no v2 sample within radius {radius:.5f}: "
          f"{info['n_failed']} ({info['n_failed_interior']} interior holes)  |  "
          f"candidates: {info['n_candidates']}")

    if args.retry_unaccepted:
        # Second pass over targets a previous pass could not land.  Meant to be
        # run with a different search setting -- e.g.
        # ``--axis-weights 3 1 1 3 1``, which converted 2 of 6 recorded misses in
        # a controlled test but also cost 1 of 6 already-accepted targets, so it
        # is worth applying only where there is nothing left to lose.  Results
        # are kept only when they beat what is already on disk.
        done = {}
        for f in args.output.glob("target_*.json"):
            with open(f) as fh:
                d = json.load(fh)
            if not d.get("accepted"):
                done[d["target_id"]] = d
        pos = {int(t): i for i, t in enumerate(rec["tid"])}
        sel = np.array(sorted(pos[t] for t in done if t in pos), dtype=int)
        if args.n_workers > 1:
            sel = sel[args.worker :: args.n_workers]
        print(f"  retrying {len(sel)} unaccepted target(s)")
    elif args.target_ids:
        sel = np.array([int(np.flatnonzero(rec["tid"] == t)[0]) for t in args.target_ids])
    elif args.all:
        # Two filters, then rank by attainability evidence.
        #
        # ``--min-isolation`` keeps only holes with room: a design has to land
        # inside the hole *and* stay a radius clear of every existing sample, and
        # the median unfilled hole centre is just 1.15 radii from the nearest one.
        #
        # Ordering is by enclosure (occupied neighbours out of 242), NOT by
        # isolation.  Sorting by isolation looks right and is badly wrong: the
        # most rank-isolated holes are concentrated in the near-solid corner
        # (38 % of the d_all >= 1.5r tail has vol > 0.7), where the marginal CDFs
        # are so steep that the smallest move a 50x50 cell can make -- one
        # mirrored pixel quad, 4/2500 of the volume -- is already 0.20 radii on
        # the vol axis alone.  Those holes are quantisation artefacts of the rank
        # transform, not reachable gaps; runs there found no improving flip at
        # all despite the seed matching the target's physical properties to 0.05 %.
        cand = rec["cand"]
        cand = cand[rec["d_all"][cand] >= args.min_isolation * radius]
        sel = cand[np.lexsort((-rec["d_all"][cand], -rec["encl"][cand]))]
        print(f"  queue: {len(sel)} holes with d_all >= {args.min_isolation:g} x radius,"
              f" best-enclosed first")
        if args.n_workers > 1:
            sel = sel[args.worker :: args.n_workers]
        # Drop finished targets BEFORE --max-targets caps the pass.  Capping
        # first means a restarted worker gets the same already-done prefix of its
        # slice every time, skips all of it, and does no work at all -- which is
        # exactly how the second attempt made 9 targets of progress in 115 hours.
        n_slice = len(sel)
        # ``timeout_*.json`` marks a target whose FEM solve wedged and had to be
        # killed by the watchdog; it is skipped like a finished one, otherwise
        # every restart re-runs it and hangs again.
        sel = np.array([i for i in sel
                        if not (args.output / f"target_{int(rec['tid'][i]):06d}.json").exists()
                        and not (args.output / f"timeout_{int(rec['tid'][i]):06d}.json").exists()],
                       dtype=int)
        print(f"  worker slice {n_slice}, of which {n_slice - len(sel)} already done,"
              f" {len(sel)} outstanding")
        if len(sel) == 0:
            print("QUEUE_EMPTY")          # unambiguous marker for the supervisor
            return
        if args.max_targets:
            sel = sel[: args.max_targets]
    else:
        sel = pick_stratified(rec, args.n_targets)
    if args.only is not None:
        sel = sel[args.only : args.only + 1]

    print(f"\nrunning {len(sel)} target(s)"
          + (f" (worker {args.worker}/{args.n_workers})" if args.n_workers > 1 else ""))
    for i in sel[:8]:
        print(f"  tid {rec['tid'][i]:6d}  encl {int(rec['encl'][i]):3d}  "
              f"d_v2 {rec['d_all'][i]:.4f}  "
              + "  ".join(f"{c}={v:.4g}" for c, v in zip(FEATURES, rec["P"][i])))

    if args.dry_run:
        return

    coarse = build_homog_setup(canvas_N=50, mesh_N=50, ele_type="TRI6", f_star=0.0)
    fine = build_homog_setup(canvas_N=50, mesh_N=100, ele_type="TRI6", f_star=0.0)
    tree = cKDTree(rec["Rv"])
    args.output.mkdir(parents=True, exist_ok=True)
    extra: list[np.ndarray] = []          # rank points of designs accepted so far

    for i in sel:
        tid = int(rec["tid"][i])
        tgt = rec["P"][i]
        out_json = args.output / f"target_{tid:06d}.json"
        prev = None
        if out_json.exists():
            if not args.retry_unaccepted:
                print(f"target {tid}: already done, skipping", flush=True)
                continue
            with open(out_json) as fh:
                prev = json.load(fh)

        # Designs land inside their own hole but also displace nearby ones, and
        # workers run blind to each other, so re-read everything accepted so far
        # and skip whatever is already covered.  Cheap next to one FEM solve.
        if args.dynamic_dedup:
            extra = _accepted_points(args.output)
            if len(extra):
                d_extra = np.min(np.linalg.norm(extra - rec["centres"][i], axis=1))
                if d_extra <= radius:
                    print(f"target {tid}: already covered by an earlier design "
                          f"(d={d_extra:.4f}) — skipping", flush=True)
                    continue

        live_tree = (cKDTree(np.vstack([rec["Rv"], extra])) if len(extra) else tree)
        seeds = nearest_symmetric_cells(args.dataset, rec, i,
                                        n_seeds=args.n_seeds, tree=tree)
        t0 = time.time()
        best = None
        for attempt, (row, d_nn, init_cell, nn_props) in enumerate(seeds):
            print(f"\n{'='*78}\ntarget {tid}   [rss {_rss_mb():.0f} MB]   "
                  + "  ".join(f"{c}={v:.5g}" for c, v in zip(FEATURES, tgt))
                  + f"\n  seed {attempt}: dataset row {row}, rank distance {d_nn:.4f}  "
                  + "  ".join(f"{c}={v:.5g}" for c, v in zip(FEATURES, nn_props)),
                  flush=True)

            cell, runs = design_one(tgt[:4], tgt[4], init_cell, (coarse, fine), args,
                                    knots=rec["knots"], centre=rec["centres"][i],
                                    radius=radius, nn_tree=live_tree)
            got, vf, C_eff = validate(cell)
            achieved = np.concatenate([got, [vf]])
            # the seed is a dataset row, so its TRI6/N=100 properties are the
            # stored columns — no need to re-solve them
            init_achieved = nn_props
            r_ach = to_rank(achieved[None], rec["knots"])[0]
            d_hit = float(np.linalg.norm(r_ach - rec["centres"][i]))
            d_hit0 = float(np.linalg.norm(
                to_rank(init_achieved[None], rec["knots"])[0] - rec["centres"][i]))
            d_new, _ = live_tree.query(r_ach, k=1)
            conn = connectivity_report(cell)
            conn0 = connectivity_report(init_cell)
            # No worse than the seed geometrically: a handful of dataset cells
            # carry floating islands of their own, and demanding a single
            # component would reject every design grown from one of those
            # regardless of how well its properties landed.
            accepted = bool(d_hit <= radius and float(d_new) > radius
                            and conn["gates_connected"]
                            and conn["n_components"] <= max(conn0["n_components"], 1))
            cand_result = (cell, C_eff, achieved, init_achieved, d_hit, d_hit0,
                           float(d_new), conn, conn0, accepted, row, d_nn,
                           init_cell, nn_props, runs)
            if best is None or d_hit < best[4]:
                best = cand_result
            if accepted:
                break
            if attempt + 1 < len(seeds):
                print(f"  seed {attempt} not accepted (d_hit={d_hit:.4f}) — "
                      f"retrying from the next-nearest symmetric cell", flush=True)

        (cell, C_eff, achieved, init_achieved, d_hit, d_hit0, d_new, conn, conn0,
         accepted, row, d_nn, init_cell, nn_props, runs) = best
        rel = achieved / tgt - 1.0
        rel0 = init_achieved / tgt - 1.0

        print(f"\n  VALIDATION (calc_fem_hifi, TRI6, N=100) — {time.time()-t0:.0f}s")
        for j, c in enumerate(FEATURES):
            print(f"    {c:4s} target={tgt[j]:12.5e}  seed={init_achieved[j]:12.5e}"
                  f" ({rel0[j]:+7.2%})  designed={achieved[j]:12.5e} ({rel[j]:+7.2%})")
        print(f"    rank distance to target centre : {d_hit:.4f}  "
              f"(seed {d_hit0:.4f}, radius {radius:.4f}) -> "
              f"{'HIT' if d_hit <= radius else 'MISS'}")
        print(f"    rank distance to nearest v2 row: {float(d_new):.4f}   "
              f"({'genuinely new support' if float(d_new) > radius else 'lands on existing data'})")
        print(f"    binary connectivity: all four gates joined="
              f"{conn['gates_connected']}  solid components={conn['n_components']}"
              f" (seed {conn0['n_components']})")
        print(f"    ACCEPTED: {accepted}   "
              f"(needs hit + support beyond the radius + no new floating material)")

        if prev is not None and not _is_better(payload_hit=(d_hit <= radius),
                                               payload_acc=accepted, d_hit=d_hit,
                                               prev=prev):
            print(f"  retry did not beat the stored result "
                  f"(d_hit {prev['d_hit']:.4f} -> {d_hit:.4f}) — keeping the old one",
                  flush=True)
            continue

        np.savez_compressed(
            args.output / f"target_{tid:06d}.npz",
            cell=cell, target=tgt, achieved=achieved, rel=rel, C_eff=C_eff,
            accepted=accepted, init_cell=init_cell, init_row=row, init_props=nn_props,
            init_achieved=init_achieved, d_hit=d_hit, d_hit_seed=d_hit0,
            d_nearest_v2=float(d_new), radius=radius,
            enclosure=rec["encl"][i], target_id=tid,
            connectivity=np.array([conn["gates_connected"], conn["n_components"]]),
            **{f"hist_{k}": v for k, v in runs.items()},
        )
        payload = dict(target_id=tid, target=tgt.tolist(),
                       achieved=achieved.tolist(), rel=rel.tolist(),
                       rank=to_rank(achieved[None], rec["knots"])[0].tolist(),
                       # per-axis rank residual: on a miss this says which
                       # coordinate the local search could not move
                       rank_residual=(to_rank(achieved[None], rec["knots"])[0]
                                      - rec["centres"][i]).tolist(),
                       seed=init_achieved.tolist(), rel_seed=rel0.tolist(),
                       d_hit=d_hit, d_hit_seed=d_hit0,
                       d_nearest_v2=float(d_new), radius=radius,
                       hit=bool(d_hit <= radius), accepted=accepted,
                       enclosure=int(rec["encl"][i]), init_row=row,
                       init_rank_distance=d_nn, connectivity=conn,
                       n_pixels_changed=int((cell != init_cell).sum()),
                       seconds=time.time() - t0)
        tmp = args.output / f".target_{tid:06d}.json"
        with open(tmp, "w") as fh:                  # atomic: other workers read this dir
            json.dump(payload, fh, indent=2)
        tmp.rename(out_json)
        print(f"  wrote {args.output / f'target_{tid:06d}.npz'}", flush=True)


if __name__ == "__main__":
    main()
