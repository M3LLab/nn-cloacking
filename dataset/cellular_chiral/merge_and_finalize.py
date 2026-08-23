"""Merge the newly homogenised diffusion cells with the parent pool and re-thin.

The parent dataset and the generated cells are combined *before* thinning rather
than after, because the rank transform and the Poisson-disk acceptance both depend
on the full point set: a generated cell is only worth keeping if it is far from
every parent cell *and* from every other generated cell, and that can only be
decided jointly.

The radius is held fixed rather than the count. "Uniform" means constant spacing,
so the honest way to report the effect of upsampling is that the spacing stays put
and the point count rises by however much genuinely new support was added.

Writes the final HDF5 with the parent's exact schema, plus two provenance columns:
``provenance`` (0 = original CA cells, 1 = diffusion-generated) and ``parent_row``
(row index into whichever source file that row came from).

Usage
-----

    python -m dataset.cellular_chiral.merge_and_finalize \
        -p output/ca_bulk_squared/stiffness.h5 \
        -n output/ca_bulk_squared/upsample/stiffness_gen.h5 \
        -s output/ca_bulk_squared/subset_uniform_v1.npz \
        -o output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from dataset.cellular_chiral.thin_uniform import (
    FEATURES, _boundary_seeds, poisson_disk, print_report, rank_transform, verify,
)

COPY = ["C11", "C22", "C12", "C66", "C_eff", "cells", "lambda_", "live_fraction",
        "mu", "rho", "source_idx", "vf", "vol"]
CHUNK = 8192


def _load5(path: Path, rows=None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        if rows is None:
            return np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)
        return np.stack([f[k][:][rows] for k in FEATURES], 1).astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--parent", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-n", "--new", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/stiffness_gen.h5"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("--radius", type=float, default=None,
                   help="Poisson-disk radius; default reuses the v1 radius")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force")

    z = np.load(args.subset, allow_pickle=False)
    radius = float(args.radius if args.radius is not None else z["radius"])
    surv = z["survived_dedup"]                       # parent rows past translation dedup

    Xp = _load5(args.parent, surv)
    Xn = _load5(args.new)
    print(f"parent {len(Xp)} (post-dedup)  +  new {len(Xn)}  = {len(Xp) + len(Xn)}")

    X = np.vstack([Xp, Xn])
    prov = np.concatenate([np.zeros(len(Xp), np.int64), np.ones(len(Xn), np.int64)])
    src_row = np.concatenate([surv, np.arange(len(Xn))])

    # cross-set property duplicates (a generated cell reproducing a parent one)
    Z = (X - X.mean(0)) / X.std(0)
    pairs = cKDTree(Z).query_pairs(1e-9, output_type="ndarray")
    drop = np.zeros(len(X), bool)
    for i, j in pairs:                               # keep the parent row of each pair
        a, b = (i, j) if prov[i] <= prov[j] else (j, i)
        if not drop[a]:
            drop[b] = True
    print(f"  {int(drop.sum())} cross-set property duplicates dropped")
    keep0 = ~drop
    X, prov, src_row = X[keep0], prov[keep0], src_row[keep0]

    R, knots = rank_transform(X)
    seeds = _boundary_seeds(R, 512, args.seed)
    t0 = time.time()
    kept = poisson_disk(R, radius, args.seed, seeds)
    print(f"  Poisson-disk r={radius:.5f} -> {len(kept)} rows ({time.time()-t0:.0f}s)")

    rep = verify(X, (X - X.mean(0)) / X.std(0), R, kept, radius)
    print_report(rep)

    n_par = int(np.sum(prov[kept] == 0))
    n_new = int(np.sum(prov[kept] == 1))
    print(f"  final composition: {n_par} original CA + {n_new} diffusion-generated "
          f"({100 * n_new / len(kept):.1f}% new)")
    print(f"  generated-cell yield: {n_new}/{len(Xn)} = {n_new / len(Xn):.3f}")

    # ---- materialise ------------------------------------------------------
    kp = np.sort(src_row[kept][prov[kept] == 0])
    kn = np.sort(src_row[kept][prov[kept] == 1])
    n_out = len(kp) + len(kn)
    with h5py.File(args.parent, "r") as fp, h5py.File(args.new, "r") as fn, \
            h5py.File(args.output, "w") as dst:
        for k, v in fp.attrs.items():
            dst.attrs[k] = v
        for name in COPY:
            d = fp[name]
            out = dst.create_dataset(name, shape=(n_out,) + d.shape[1:],
                                     maxshape=d.maxshape, dtype=d.dtype,
                                     chunks=d.chunks, compression=d.compression,
                                     compression_opts=d.compression_opts)
            for k, v in d.attrs.items():
                out.attrs[k] = v
            for s in range(0, len(kp), CHUNK):
                e = min(s + CHUNK, len(kp))
                out[s:e] = d[kp[s:e]]
            dn = fn[name]
            for s in range(0, len(kn), CHUNK):
                e = min(s + CHUNK, len(kn))
                out[len(kp) + s : len(kp) + e] = dn[kn[s:e]]
            print(f"    {name:15s} {out.shape}")
        dst.create_dataset("provenance",
                           data=np.concatenate([np.zeros(len(kp), np.int64),
                                                np.ones(len(kn), np.int64)]),
                           dtype=np.int64, chunks=(min(256, n_out),), maxshape=(None,))
        dst.create_dataset("parent_row", data=np.concatenate([kp, kn]),
                           dtype=np.int64, chunks=(min(256, n_out),), maxshape=(None,))
        dst["provenance"].attrs["desc"] = "0 = original CA cell, 1 = diffusion-generated"
        dst["parent_row"].attrs["desc"] = "row in the source file named by provenance"
        dst.attrs["subset_parent"] = str(args.parent)
        dst.attrs["subset_generated"] = str(args.new)
        dst.attrs["subset_n"] = n_out
        dst.attrs["subset_n_from_parent"] = len(kp)
        dst.attrs["subset_n_from_generated"] = len(kn)
        dst.attrs["subset_radius_rank_units"] = radius
        dst.attrs["subset_seed"] = args.seed
        dst.attrs["subset_method"] = (
            "translation dedup + rank-space Poisson-disk thinning of "
            "(parent + diffusion-upsampled) pool")
    print(f"\nwrote {args.output}  ({n_out} rows, "
          f"{args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
