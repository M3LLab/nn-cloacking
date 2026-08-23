"""Thin the homogenised dataset to a near-uniform subset of the 5-D condition space.

The June-17 ``stiffness.h5`` (TRI6 @ 2 elements/pixel) is heavily clumped in the
5-D diffusion conditioning space ``(C11, C22, C12, C66, vol)``: a single bin of a
coarse 8^5 grid holds ~31%% of all rows, while box-counting puts the intrinsic
dimension of the occupied support at ~2.5-3. Training or nearest-neighbour
snapping against that distribution is dominated by the near-solid corner.

This module writes a **non-destructive index subset** (the HDF5 is never
rewritten) in two stages.

Stage 1 -- symmetry dedup
    Homogenisation is invariant under periodic translation of the unit cell, but
    ``bulk_stiffness``'s dedup keys off raw bytes (and an optional block-pooled
    fingerprint, which was *disabled* for the June-17 run: ``fuzzy_pool=1``), so
    cyclic translates survive as distinct rows with bit-identical properties.
    Candidate groups are found as connected components of the 1e-9 neighbourhood
    graph in z-scored property space, then each pair is *confirmed* by phase
    correlation plus an exact ``np.roll`` comparison -- so a group is only
    collapsed when the geometries are genuinely the same cell.

    Rows that merely happen to share properties while being geometrically
    distinct are **kept**: for a conditional generator, several distinct
    microstructures hitting one target C is signal, not redundancy.

Stage 2 -- blue-noise thinning
    Each coordinate is mapped through its own empirical CDF (mid-rank, so equal
    values stay equal) to [0, 1], making "uniform" mean uniform in quantile
    rather than in raw units -- without this the near-solid pile sets the scale
    for every radius. Then Poisson-disk elimination: walk the points in a seeded
    random order, accept a point if no already-accepted point lies within
    ``radius``, and kill everything inside that ball.

    Two properties matter downstream. Elimination adapts to the support, so no
    empty-bin bookkeeping is needed in 5-D; and every *discarded* row is by
    construction within ``radius`` of a kept one, which bounds how much worse the
    "snap each cloak cell to its nearest dataset entry" step can get. The support
    boundary (per-axis extremes plus extremes along random directions) is seeded
    first so the attainable envelope -- the compliant tail and the negative-C12
    lobe -- survives thinning intact.

Usage
-----

    python -m dataset.cellular_chiral.thin_uniform \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/subset_uniform_v1.npz \
        --target 150000

    # fix the radius directly instead of bisecting for a target count
    python -m dataset.cellular_chiral.thin_uniform --radius 0.031

Consumers read ``idx`` from the ``.npz`` and index the HDF5 with it. The file
also stores the quantile knots of the rank transform, so the identical mapping
can be applied to new points later without re-reading the dataset.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

FEATURES = ("C11", "C22", "C12", "C66", "vol")
CELL_SIZE = 50

# Categorical slots 1-2 and the single-hue blue ramp of the reference palette.
_BEFORE = "#2a78d6"
_AFTER = "#eb6834"
_SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_INK = "#0b0b0b"
_INK2 = "#52514e"
_SURFACE = "#fcfcfb"


# --------------------------------------------------------------------------- #
# Stage 1: periodic-translation dedup
# --------------------------------------------------------------------------- #
def _is_cyclic_translate(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff ``b`` equals ``a`` rolled by some (dy, dx) on the periodic torus.

    The candidate shift is located by phase correlation (one FFT pair) and then
    *verified* exactly, so a spurious correlation peak can never merge two
    genuinely different cells.
    """
    if a.shape != b.shape:
        return False
    if np.array_equal(a, b):
        return True
    fa = np.fft.rfft2(a.astype(np.float64))
    fb = np.fft.rfft2(b.astype(np.float64))
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    return np.array_equal(np.roll(a, (-int(dy), -int(dx)), axis=(0, 1)), b)


def _union_find(n: int, pairs) -> np.ndarray:
    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return np.array([find(i) for i in range(n)])


def dedup_translations(
    h5_path: Path, Z: np.ndarray, tol: float = 1e-9, verbose: bool = True
) -> np.ndarray:
    """Return a boolean mask dropping all but one row of each translation class."""
    t0 = time.time()
    tree = cKDTree(Z)
    pairs = tree.query_pairs(tol, output_type="ndarray")
    if verbose:
        print(f"  [stage 1] {len(pairs)} candidate pairs within {tol:g} "
              f"({time.time() - t0:.0f}s)")
    keep = np.ones(len(Z), dtype=bool)
    if len(pairs) == 0:
        return keep

    involved = np.unique(pairs)
    local = {int(g): k for k, g in enumerate(involved)}
    comp = _union_find(len(involved), [(local[int(i)], local[int(j)]) for i, j in pairs])

    with h5py.File(h5_path, "r") as f:
        cells = f["cells"][np.sort(involved)]
    cell_of = {int(g): cells[k] for k, g in enumerate(np.sort(involved))}

    n_groups = n_dropped = 0
    for c in np.unique(comp):
        members = involved[comp == c]
        if len(members) < 2:
            continue
        n_groups += 1
        # representatives of the distinct geometries seen so far in this group
        reps: list[int] = []
        for m in members:
            m = int(m)
            if any(_is_cyclic_translate(cell_of[m], cell_of[r]) for r in reps):
                keep[m] = False
                n_dropped += 1
            else:
                reps.append(m)
    if verbose:
        print(f"  [stage 1] {n_groups} groups -> dropped {n_dropped} periodic "
              f"translates ({time.time() - t0:.0f}s total)")
    return keep


# --------------------------------------------------------------------------- #
# Stage 2: rank transform + Poisson-disk elimination
# --------------------------------------------------------------------------- #
def rank_transform(X: np.ndarray, n_knots: int = 8192):
    """Per-column empirical CDF using mid-ranks; equal values map to equal output.

    Returns ``(R, knots)`` where ``knots[j] = (values, quantiles)`` reproduces the
    mapping for unseen points via ``np.interp``.
    """
    n, d = X.shape
    R = np.empty_like(X, dtype=np.float64)
    knots = []
    for j in range(d):
        vals, inv, cnt = np.unique(X[:, j], return_inverse=True, return_counts=True)
        upper = np.cumsum(cnt)
        mid = (upper - cnt / 2.0) / n          # mid-rank of each distinct value
        R[:, j] = mid[inv]
        if len(vals) > n_knots:
            sel = np.linspace(0, len(vals) - 1, n_knots).astype(np.int64)
            knots.append((vals[sel], mid[sel]))
        else:
            knots.append((vals, mid))
    return R, knots


def _boundary_seeds(R: np.ndarray, n_dirs: int, seed: int) -> np.ndarray:
    """Indices on the support boundary: per-axis extremes + extremes along random rays."""
    rng = np.random.default_rng(seed)
    idx = set()
    for j in range(R.shape[1]):
        idx.add(int(np.argmin(R[:, j])))
        idx.add(int(np.argmax(R[:, j])))
    dirs = rng.normal(size=(n_dirs, R.shape[1]))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for k in range(0, n_dirs, 32):                     # chunked: (n, n_dirs) is huge
        proj = R @ dirs[k : k + 32].T
        idx.update(int(i) for i in np.argmax(proj, axis=0))
    return np.array(sorted(idx), dtype=np.int64)


def poisson_disk(R: np.ndarray, radius: float, seed: int, seeds: np.ndarray | None = None):
    """Greedy elimination. Returns kept indices; every dropped row is < radius from one."""
    n = len(R)
    tree = cKDTree(R)
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    if seeds is not None and len(seeds):
        rest = np.setdiff1d(order, seeds, assume_unique=False)
        order = np.concatenate([seeds, rest])

    alive = np.ones(n, dtype=bool)
    kept = []
    for i in order:
        if not alive[i]:
            continue
        kept.append(int(i))
        alive[tree.query_ball_point(R[i], radius)] = False
    return np.array(sorted(kept), dtype=np.int64)


def solve_radius(R, target, seed, seeds, lo=0.002, hi=0.30, tol=0.02, max_iter=14, verbose=True):
    """Bisect the radius so ``len(kept)`` lands within ``tol`` of ``target``."""
    best = None
    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        t0 = time.time()
        kept = poisson_disk(R, mid, seed, seeds)
        if verbose:
            print(f"  [stage 2] r={mid:.5f} -> N={len(kept)} ({time.time() - t0:.0f}s)")
        if best is None or abs(len(kept) - target) < abs(len(best[1]) - target):
            best = (mid, kept)
        if abs(len(kept) - target) <= tol * target:
            return mid, kept
        if len(kept) > target:      # too many points kept -> larger radius
            lo = mid
        else:
            hi = mid
    return best


# --------------------------------------------------------------------------- #
# Stage 3: verification
# --------------------------------------------------------------------------- #
def _occupancy_cv(Y: np.ndarray, B: int) -> tuple[int, float]:
    lo, hi = Y.min(0), Y.max(0)
    idx = np.clip(((Y - lo) / (hi - lo + 1e-15) * B).astype(np.int64), 0, B - 1)
    flat = np.zeros(len(Y), dtype=np.int64)
    for j in range(Y.shape[1]):
        flat = flat * B + idx[:, j]
    _, cnt = np.unique(flat, return_counts=True)
    return len(cnt), float(cnt.std() / cnt.mean())


def verify(X, Z, R, kept, radius) -> dict:
    """Coverage / uniformity / snap-error report for the chosen subset."""
    rep = {"n_before": len(X), "n_after": len(kept), "radius": radius}
    dropped = np.setdiff1d(np.arange(len(X)), kept, assume_unique=False)

    for B in (16, 24, 32):
        ob, cb = _occupancy_cv(R, B)
        oa, ca = _occupancy_cv(R[kept], B)
        rep[f"occ_B{B}"] = (ob, oa)
        rep[f"cv_B{B}"] = (cb, ca)

    # snap error: how far a discarded row now sits from the nearest kept row
    tree_z = cKDTree(Z[kept])
    dz, _ = tree_z.query(Z[dropped], k=1, workers=-1)
    rep["snap_z"] = (float(dz.mean()), float(np.percentile(dz, 99)), float(dz.max()))
    tree_r = cKDTree(R[kept])
    dr, _ = tree_r.query(R[dropped], k=1, workers=-1)
    rep["snap_rank"] = (float(dr.mean()), float(np.percentile(dr, 99)), float(dr.max()))

    # nearest-neighbour spacing inside each set (the "no near-duplicates" claim)
    for name, Y in (("before", R), ("after", R[kept])):
        d, _ = cKDTree(Y).query(Y, k=2, workers=-1)
        rep[f"nn_{name}"] = (float(np.percentile(d[:, 1], 1)),
                             float(np.median(d[:, 1])))

    # envelope retention, per raw feature
    rep["envelope"] = {
        f: (float(X[:, j].min()), float(X[:, j].max()),
            float(X[kept, j].min()), float(X[kept, j].max()))
        for j, f in enumerate(FEATURES)
    }
    return rep


def print_report(rep: dict) -> None:
    print("\n" + "=" * 78)
    print(f"  {rep['n_before']} -> {rep['n_after']} rows "
          f"({100 * rep['n_after'] / rep['n_before']:.1f}%), radius={rep['radius']:.5f} (rank units)")
    print("=" * 78)
    print("\n  uniformity (rank space; CV over occupied bins, lower = flatter)")
    for B in (16, 24, 32):
        ob, oa = rep[f"occ_B{B}"]
        cb, ca = rep[f"cv_B{B}"]
        print(f"    B={B:2d}^5   occupied {ob:7d} -> {oa:7d}    CV {cb:8.2f} -> {ca:6.2f}")
    print("\n  nearest-neighbour spacing (rank units)")
    for k in ("before", "after"):
        p1, med = rep[f"nn_{k}"]
        print(f"    {k:6s}  p1={p1:.5f}  median={med:.5f}")
    print("\n  snap error for discarded rows (distance to nearest kept row)")
    for k, unit in (("snap_rank", "rank"), ("snap_z", "z-score")):
        m, p99, mx = rep[k]
        print(f"    {unit:8s} mean={m:.4f}  p99={p99:.4f}  max={mx:.4f}")
    print("\n  envelope retention (raw units)")
    for f, (lo, hi, klo, khi) in rep["envelope"].items():
        flag = "" if (klo <= lo and khi >= hi) else "  <-- SHRUNK"
        print(f"    {f:5s} [{lo:11.4g}, {hi:11.4g}] -> [{klo:11.4g}, {khi:11.4g}]{flag}")
    print()


def plot(X, R, kept, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", _SEQ)
    fig, axes = plt.subplots(2, 5, figsize=(21, 8), facecolor=_SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=_INK2, labelsize=9)
        ax.grid(True, color="#ebeae6", linewidth=0.8)
        ax.set_axisbelow(True)

    # row 1 -- marginals, density-normalised so the two sizes are comparable
    for j, name in enumerate(FEATURES):
        ax = axes[0, j]
        lo, hi = X[:, j].min(), X[:, j].max()
        bins = np.linspace(lo, hi, 90)
        ax.hist(X[:, j], bins=bins, density=True, color=_BEFORE,
                alpha=0.85, label="before" if j == 0 else None)
        ax.hist(X[kept, j], bins=bins, density=True, histtype="step",
                color=_AFTER, linewidth=2, label="after" if j == 0 else None)
        ax.set_yscale("log")
        ax.set_title(name, fontsize=11, color=_INK)
        if j == 0:
            ax.set_ylabel("density (log)", fontsize=10, color=_INK2)
            ax.legend(frameon=False, fontsize=10, labelcolor=_INK2)

    # row 2 -- rank-space occupancy before/after, spacing, and per-bin counts
    for k, (title, Y) in enumerate((("before", R), ("after", R[kept]))):
        ax = axes[1, k]
        ax.hexbin(Y[:, 0], Y[:, 3], gridsize=60, bins="log", cmap=cmap,
                  mincnt=1, linewidths=0)
        ax.set_title(f"rank C11 x C66 — {title}", fontsize=11, color=_INK)
        ax.set_xlabel("rank C11", fontsize=10, color=_INK2)
        if k == 0:
            ax.set_ylabel("rank C66", fontsize=10, color=_INK2)

    ax = axes[1, 2]
    for name, Y, c in (("before", R, _BEFORE), ("after", R[kept], _AFTER)):
        d, _ = cKDTree(Y).query(Y, k=2, workers=-1)
        ax.hist(np.log10(np.maximum(d[:, 1], 1e-8)), bins=80, density=True,
                histtype="step", color=c, linewidth=2, label=name)
    ax.set_title("1-NN distance (rank units)", fontsize=11, color=_INK)
    ax.set_xlabel("log10 distance", fontsize=10, color=_INK2)
    ax.legend(frameon=False, fontsize=10, labelcolor=_INK2)

    for k, B in enumerate((24, 32)):
        ax = axes[1, 3 + k]
        for name, Y, c in (("before", R, _BEFORE), ("after", R[kept], _AFTER)):
            lo, hi = R.min(0), R.max(0)
            idx = np.clip(((Y - lo) / (hi - lo + 1e-15) * B).astype(np.int64), 0, B - 1)
            flat = np.zeros(len(Y), dtype=np.int64)
            for j in range(5):
                flat = flat * B + idx[:, j]
            _, cnt = np.unique(flat, return_counts=True)
            ax.hist(np.log10(cnt), bins=50, density=True, histtype="step",
                    color=c, linewidth=2, label=name)
        ax.set_title(f"points per occupied bin (B={B}^5)", fontsize=11, color=_INK)
        ax.set_xlabel("log10 count", fontsize=10, color=_INK2)
        if k == 0:
            ax.legend(frameon=False, fontsize=10, labelcolor=_INK2)

    fig.suptitle("Uniform thinning of the 5-D condition space (C11, C22, C12, C66, vol)",
                 fontsize=14, color=_INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130, facecolor=_SURFACE)
    print(f"  wrote {out_png}")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("--target", type=int, default=150_000,
                   help="target subset size (ignored when --radius is given)")
    p.add_argument("--radius", type=float, default=None,
                   help="fix the Poisson-disk radius in rank units instead of bisecting")
    p.add_argument("--seed", type=int, default=777, help="matches split_v1.json's seed")
    p.add_argument("--n-dirs", type=int, default=512,
                   help="random directions used to seed support-boundary points")
    p.add_argument("--no-dedup", action="store_true",
                   help="skip stage 1 (periodic-translation dedup)")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    with h5py.File(args.input, "r") as f:
        X = np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)
        attrs = {k: f.attrs[k] for k in ("homog_ele_type", "homog_elem_per_pixel")
                 if k in f.attrs}
    print(f"loaded {len(X)} rows from {args.input}  {attrs}")

    Z = (X - X.mean(0)) / X.std(0)

    if args.no_dedup:
        surv = np.arange(len(X))
    else:
        surv = np.flatnonzero(dedup_translations(args.input, Z))
        print(f"  [stage 1] {len(X)} -> {len(surv)} rows after symmetry dedup")

    R_all, knots = rank_transform(X[surv])
    seeds = _boundary_seeds(R_all, args.n_dirs, args.seed)
    print(f"  [stage 2] {len(seeds)} boundary seeds")

    if args.radius is not None:
        radius = args.radius
        kept_local = poisson_disk(R_all, radius, args.seed, seeds)
        print(f"  [stage 2] r={radius:.5f} -> N={len(kept_local)}")
    else:
        radius, kept_local = solve_radius(R_all, args.target, args.seed, seeds)

    idx = np.sort(surv[kept_local])
    rep = verify(X[surv], Z[surv], R_all, kept_local, radius)
    print_report(rep)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        idx=idx,
        radius=radius,
        seed=args.seed,
        feature_order=np.array(FEATURES),
        survived_dedup=surv,
        n_source=len(X),
        source=str(args.input),
        **{f"knot_v_{f}": knots[j][0] for j, f in enumerate(FEATURES)},
        **{f"knot_q_{f}": knots[j][1] for j, f in enumerate(FEATURES)},
    )
    print(f"  wrote {args.output}  ({len(idx)} indices)")

    if not args.no_plot:
        plot(X[surv], R_all, kept_local, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
