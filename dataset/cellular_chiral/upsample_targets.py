"""Emit (C, rho) targets for inverse-design upsampling of the under-covered regions.

Thinning (``thin_uniform``) can only flatten the dense side of the distribution --
it cannot create samples where the dataset has none. What is left after thinning is
a set of **holes**: pockets of the 5-D condition space (C11, C22, C12, C66, vol)
that are enclosed by real data on essentially every side but contain no sample.
Those are the coordinates worth spending inverse-design runs on.

Method
------
Work in the same rank (per-marginal empirical-CDF) space the thinning used, on a
grid of bin width ``1/bins`` (default 1/24 ~ 1.5x the Poisson radius). For every
empty bin, count how many of its 242 neighbours are occupied by real data. That
count -- ``enclosure`` -- is the attainability evidence: a bin walled in on 3/4 of
its neighbourhood is a genuine hole in a sampled manifold, whereas a bin touching
only a handful of occupied neighbours is most likely just outside the support and
would waste a run.

Surviving bin centres are mapped back to physical units through the stored
quantile knots, then screened by ``bounds.check_attainable``: the Hashin-Shtrikman
upper bound on the in-plane bulk modulus, the Voigt (rule-of-mixtures) ceiling on
the individual components, and positive-definiteness. Those hold for *any*
cement/void microstructure at the given volume fraction, so they reject impossible
targets on physics rather than on what the dataset happens to contain. The HS bulk
bound is the sharp one -- the stiffest real cells sit at 94% of it.

Density caveat
--------------
``rho`` is not an independent target: the dataset satisfies ``rho = 2300 * vol``
exactly (rho_solid x volume fraction). It is emitted for completeness, but the
quantity a run actually has to hit is ``vol`` -- and ``inverse_design.py``'s loss
only covers ``flat4 = (C11, C22, C12, C66)``, so hitting ``vol`` needs an added
volume penalty or a post-hoc filter on the achieved volume fraction.

Usage
-----

    python -m dataset.cellular_chiral.upsample_targets \
        -i output/ca_bulk_squared/stiffness.h5 \
        -s output/ca_bulk_squared/subset_uniform_v1.npz \
        -o output/ca_bulk_squared/upsample_targets_v1 \
        --bins 24 --min-enclosure 100

Writes ``<out>.csv`` (review / hand-edit) and ``<out>.npz`` (``targets`` array,
column order in ``columns``), ranked so that truncating to the first N rows keeps
the most confidently-attainable, most-isolated holes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from dataset.cellular_chiral.bounds import RHO_SOLID, check_attainable
from dataset.cellular_chiral.thin_uniform import FEATURES, rank_transform

_BEFORE = "#2a78d6"
_AFTER = "#eb6834"
_FRONTIER = "#1baf7a"
_INK, _INK2, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def _neighbour_offsets() -> np.ndarray:
    o = np.array(np.meshgrid(*[[-1, 0, 1]] * 5, indexing="ij")).reshape(5, -1).T
    return o[np.any(o != 0, axis=1)]


def find_holes(R: np.ndarray, bins: int, min_enclosure: int):
    """Empty grid bins ranked by how many of their 242 neighbours hold real data."""
    B = bins
    pw = np.array([B**4, B**3, B**2, B, 1], dtype=np.int64)
    idx = np.clip((R * B).astype(np.int64), 0, B - 1)
    keys = idx @ pw
    occ, first = np.unique(keys, return_index=True)
    occ_digits = idx[first]

    parts = []
    for o in _neighbour_offsets():
        cand = occ_digits + o
        ok = np.all((cand >= 0) & (cand < B), axis=1)
        parts.append(cand[ok] @ pw)
    cand_keys, counts = np.unique(np.concatenate(parts), return_counts=True)

    empty = ~np.isin(cand_keys, occ, assume_unique=True)
    cand_keys, counts = cand_keys[empty], counts[empty]
    sel = counts >= min_enclosure
    cand_keys, counts = cand_keys[sel], counts[sel]

    digits = np.empty((len(cand_keys), 5), dtype=np.int64)
    rem = cand_keys.copy()
    for j in range(5):
        digits[:, j], rem = np.divmod(rem, pw[j])
    centres = (digits + 0.5) / B
    return centres, counts, len(occ)


def frontier_targets(X: np.ndarray, vol_bins: int, levels, per_bin: int, seed: int):
    """Targets *beyond* the sampled envelope but inside the HS bulk bound.

    The enclosure heuristic can only propose holes surrounded by existing data, so
    it never reaches the stiff frontier: at a given volume fraction the dataset tops
    out near 0.94 of the HS bound, and everything between that and 1.0 is attainable
    in principle yet unsampled -- the CA generator simply never produced it.

    Each target is built by scaling a real, high-utilisation cell's whole tensor by
    a factor ``s``, which scales K linearly and leaves anisotropy and positive-
    definiteness intact, chosen so the target lands at a prescribed fraction of the
    HS bound. These are genuine extrapolation: ``enclosure`` is 0 because no
    dataset evidence supports them, only the physics.
    """
    from dataset.cellular_chiral.bounds import bulk_2d, hs_upper_2d

    rng = np.random.default_rng(seed)
    C11, C22, C12, C66, vol = X.T
    k_hs, _ = hs_upper_2d(vol)
    util = bulk_2d(C11, C22, C12) / k_hs

    edges = np.linspace(vol.min(), vol.max(), vol_bins + 1)
    rows, scales, utils = [], [], []
    for b in range(vol_bins):
        m = (vol >= edges[b]) & (vol <= edges[b + 1] if b == vol_bins - 1
                                 else vol < edges[b + 1])
        if m.sum() < 50:
            continue
        idx = np.flatnonzero(m)
        top = idx[np.argsort(util[idx])[-max(per_bin * 4, 50):]]   # stiffest in bin
        for lvl in levels:
            pick = rng.choice(top, size=min(per_bin, len(top)), replace=False)
            for i in pick:
                s = lvl / util[i]
                if s <= 1.0:
                    continue                      # already at or past this level
                rows.append([C11[i] * s, C22[i] * s, C12[i] * s, C66[i] * s, vol[i]])
                scales.append(s)
                utils.append(lvl)
    if not rows:
        return np.empty((0, 5)), np.empty(0), np.empty(0)
    return np.array(rows), np.array(scales), np.array(utils)


def inv_rank(centres: np.ndarray, knots) -> np.ndarray:
    """Rank-space -> physical units through the stored per-column quantile knots."""
    out = np.empty_like(centres)
    for j in range(centres.shape[1]):
        v, q = knots[j]
        out[:, j] = np.interp(centres[:, j], q, v)
    return out


def fwd_rank(P: np.ndarray, knots) -> np.ndarray:
    """Physical units -> rank space (clamped past the ends of the knot range)."""
    out = np.empty_like(P)
    for j in range(P.shape[1]):
        v, q = knots[j]
        out[:, j] = np.interp(P[:, j], v, q)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/upsample_targets_v1"))
    p.add_argument("--bins", type=int, default=24,
                   help="grid resolution per axis in rank space (24 -> width 0.0417)")
    p.add_argument("--min-enclosure", type=int, default=100,
                   help="minimum occupied neighbours out of 242 for a bin to qualify")
    p.add_argument("--max-targets", type=int, default=0,
                   help="truncate the ranked list (0 = keep all)")
    p.add_argument("--frontier-per-bin", type=int, default=40,
                   help="frontier targets per volume-fraction bin per level (0 = none)")
    p.add_argument("--frontier-levels", type=str, default="0.96,0.98",
                   help="fractions of the HS bulk bound to aim at; the dataset tops "
                        "out at 0.94, and 1.00 is attainable only by idealised "
                        "assemblages, not a 50x50 binary cell")
    p.add_argument("--frontier-vol-bins", type=int, default=24)
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    z = np.load(args.subset, allow_pickle=False)
    surv = z["survived_dedup"]
    with h5py.File(args.input, "r") as f:
        X = np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)[surv]
    R, knots = rank_transform(X)
    radius = float(z["radius"])
    print(f"reference coverage: {len(X)} rows, thinning radius {radius:.5f}")

    centres, enclosure, n_occ = find_holes(R, args.bins, args.min_enclosure)
    print(f"grid {args.bins}^5 (width {1/args.bins:.4f}): {n_occ} occupied bins, "
          f"{len(centres)} empty bins with >={args.min_enclosure}/242 occupied neighbours")

    P = inv_rank(centres, knots)                       # C11, C22, C12, C66, vol
    region = np.zeros(len(P), dtype=np.int64)          # 0 = interior hole

    if args.frontier_per_bin:
        levels = [float(v) for v in args.frontier_levels.split(",")]
        F, _, _ = frontier_targets(X, args.frontier_vol_bins, levels,
                                   args.frontier_per_bin, args.seed)
        print(f"frontier: {len(F)} candidates beyond the sampled envelope "
              f"at HS utilisation {levels}")
        P = np.vstack([P, F])
        centres = np.vstack([centres, fwd_rank(F, knots)])
        enclosure = np.concatenate([enclosure, np.zeros(len(F), dtype=enclosure.dtype)])
        region = np.concatenate([region, np.ones(len(F), dtype=np.int64)])

    C11, C22, C12, C66, vol = P.T

    ok, diag = check_attainable(C11, C22, C12, C66, vol=vol)
    print(f"attainability screen (HS bulk + Voigt + positive-definite): "
          f"{np.sum(~ok)} rejected, {np.sum(ok)} kept")
    for k, v in diag.items():
        bad = np.sum(v > 1.0 + 1e-9) if k != "pd_margin" else np.sum(v <= 0)
        print(f"    {k:16s} max={v.max():.4f}  violating={bad}")
    P, centres, enclosure, region = P[ok], centres[ok], enclosure[ok], region[ok]
    diag = {k: v[ok] for k, v in diag.items()}
    pd_margin, hs_util = diag["pd_margin"], diag["hs_K_util"]
    C11, C22, C12, C66, vol = P.T

    tree_r = cKDTree(R)
    nn_rank, _ = tree_r.query(centres, k=1, workers=-1)
    mu, sd = X.mean(0), X.std(0)
    tree_z = cKDTree((X - mu) / sd)
    nn_z, _ = tree_z.query((P - mu) / sd, k=1, workers=-1)

    # most confidently attainable first; isolation breaks ties
    # interior holes first (ranked by enclosure, then isolation), frontier after
    order = np.lexsort((-nn_rank, -enclosure, region))
    if args.max_targets:
        order = order[: args.max_targets]
    P, centres, enclosure, region = P[order], centres[order], enclosure[order], region[order]
    pd_margin, hs_util = pd_margin[order], hs_util[order]
    nn_rank, nn_z = nn_rank[order], nn_z[order]
    C11, C22, C12, C66, vol = P.T
    rho = RHO_SOLID * vol

    cols = ["target_id", "region", "C11", "C22", "C12", "C66", "rho", "vol",
            "enclosure", "nn_dist_rank", "nn_dist_z", "hs_K_util", "pd_margin",
            "anisotropy"]
    table = np.column_stack([
        np.arange(1, len(P) + 1), region, C11, C22, C12, C66, rho, vol,
        enclosure, nn_rank, nn_z, hs_util, pd_margin, C11 / C22,
    ])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output.with_suffix(".csv")
    hdr = (
        f"inverse-design upsampling targets for {args.input.name}\n"
        f"C11,C22,C12,C66 in Pa; rho in kg/m^3; vol = volume fraction (rho = {RHO_SOLID:g}*vol)\n"
        f"inverse_design.py optimises flat4=(C11,C22,C12,C66); vol needs its own penalty\n"
        f"region: 0 = interior hole (data on all sides), 1 = frontier beyond the "
        f"sampled envelope but inside the HS bound (extrapolation, harder)\n"
        f"enclosure = occupied neighbours out of 242 (attainability evidence, higher = safer; 0 for frontier)\n"
        f"nn_dist_rank/nn_dist_z = distance to the nearest existing dataset entry\n"
        f"hs_K_util = (C11+C22+2C12)/4 divided by the Hashin-Shtrikman upper bound "
        f"at this vol; <=1 required, real cells reach 0.94\n"
        f"pd_margin = 1 - C12^2/(C11*C22) > 0; anisotropy = C11/C22\n"
        f"rows are ranked: truncate from the top\n" + ",".join(cols)
    )
    np.savetxt(csv_path, table, delimiter=",", header=hdr,
               fmt=["%d", "%d"] + ["%.6e"] * 5 + ["%.6f", "%d"] + ["%.6f"] * 5)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        targets=table, columns=np.array(cols), region=region,
        flat4=np.column_stack([C11, C22, C12, C66]), rho=rho, vol=vol,
        rank_centres=centres, bins=args.bins, min_enclosure=args.min_enclosure,
        source=str(args.input), subset=str(args.subset), rho_solid=RHO_SOLID,
    )
    print(f"wrote {csv_path}\nwrote {args.output.with_suffix('.npz')}  ({len(P)} targets)")

    n_hole, n_front = int(np.sum(region == 0)), int(np.sum(region == 1))
    print(f"\n  {n_hole} interior-hole targets + {n_front} frontier targets")
    print("\n  target spread (raw units)")
    for j, f in enumerate(FEATURES):
        print(f"    {f:5s} [{P[:, j].min():11.4g}, {P[:, j].max():11.4g}]   "
              f"dataset [{X[:, j].min():11.4g}, {X[:, j].max():11.4g}]")
    print(f"    rho   [{rho.min():11.4g}, {rho.max():11.4g}]")
    print(f"\n  enclosure: min={enclosure.min()} median={np.median(enclosure):.0f} "
          f"max={enclosure.max()}")
    print(f"  distance to nearest existing entry (rank units): "
          f"min={nn_rank.min():.4f} median={np.median(nn_rank):.4f} max={nn_rank.max():.4f}"
          f"   [thinning radius {radius:.4f}]")

    if not args.no_plot:
        _plot(X, R, centres, region, args.output.with_suffix(".png"))


def _plot(X, R, centres, region, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = [(0, 3, "C11", "C66"), (0, 1, "C11", "C22"),
             (2, 4, "C12", "vol"), (3, 4, "C66", "vol")]
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor=_SURFACE)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(R), 60000, replace=False)
    for ax, (a, b, la, lb) in zip(axes, pairs):
        ax.set_facecolor(_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.grid(True, color="#ebeae6", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_INK2, labelsize=9)
        first = a == 0 and b == 3
        ax.scatter(R[sub, a], R[sub, b], s=1, c=_BEFORE, alpha=0.25,
                   linewidths=0, label="existing data" if first else None)
        h = region == 0
        ax.scatter(centres[h, a], centres[h, b], s=6, c=_AFTER,
                   linewidths=0, label="interior holes" if first else None)
        ax.scatter(centres[~h, a], centres[~h, b], s=14, c=_FRONTIER, marker="^",
                   linewidths=0, label="HS frontier" if first else None)
        ax.set_xlabel(f"rank {la}", fontsize=10, color=_INK2)
        ax.set_ylabel(f"rank {lb}", fontsize=10, color=_INK2)
        if a == 0 and b == 3:
            ax.legend(frameon=False, fontsize=10, labelcolor=_INK2, markerscale=4)
    fig.suptitle("Where the upsampling targets sit relative to existing coverage "
                 "(rank space)", fontsize=13, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130, facecolor=_SURFACE)
    print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
