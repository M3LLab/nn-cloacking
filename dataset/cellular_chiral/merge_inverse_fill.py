"""Fold the accepted inverse-designed cells into the v2 subset to make v3.

Unlike ``merge_and_finalize``, this does *not* re-thin the parent pool.  v2 is
already a Poisson-disk sample at ``radius``; re-ranking the union shifts every
coordinate by ~0.1 % and a greedy pass over the shifted points can evict rows
that v2 legitimately kept.  v3 is therefore a strict superset of v2: every v2
row survives, with its ``provenance``/``parent_row`` intact.

The new rows are judged in the *pipeline's* rank space -- the empirical-CDF knots
stored in ``subset_uniform_v1.npz``, which is what the thinning radius, the hole
list and ``fill_gaps_inverse``'s acceptance test are all defined in.  Re-fitting
the transform on v2 instead gives a different (and here ~1.6x larger) distance
for the same pair, so the stored radius would no longer mean what it says.
Two filters apply:

* new-vs-v2 -- already guaranteed ``> radius`` by the acceptance rule, re-checked
  here rather than trusted;
* new-vs-new -- two designs aimed at neighbouring holes can land within a radius
  of each other; the greedy pass keeps the first and drops the rest.

``provenance`` gains a third value: 2 = inverse-designed.  ``live_fraction`` is
-1 for those rows (no CA seed) and must be masked, not read as a density.

Usage
-----

    python -m dataset.cellular_chiral.merge_inverse_fill \
        -p output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5 \
        -n output/ca_bulk_squared/inverse_fill/accepted.h5 \
        -o output/ca_bulk_squared/stiffness_tri6_uniform_v3.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from dataset.cellular_chiral.fill_gaps_inverse import load_knots, to_rank
from dataset.cellular_chiral.thin_uniform import FEATURES

COPY = ["C11", "C22", "C12", "C66", "C_eff", "cells", "lambda_", "live_fraction",
        "mu", "rho", "source_idx", "vf", "vol"]
CHUNK = 8192


def _load5(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-p", "--parent", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("-n", "--new", type=Path,
                   default=Path("output/ca_bulk_squared/inverse_fill/accepted.h5"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v3.h5"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"),
                   help="npz holding the pipeline's rank knots and thinning radius")
    p.add_argument("--radius", type=float, default=None,
                   help="Poisson-disk radius; default reuses the subset npz")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force")

    knots, radius_npz = load_knots(args.subset)
    radius = float(args.radius if args.radius is not None else radius_npz)

    with h5py.File(args.parent, "r") as f:
        n_par = f["C11"].shape[0]
        has_prov = "provenance" in f
        r_attr = float(f.attrs["subset_radius_rank_units"])
    if abs(r_attr - radius_npz) > 1e-12:
        print(f"  ! parent radius attr {r_attr:.9f} != subset npz {radius_npz:.9f}")

    Xp, Xn = _load5(args.parent), _load5(args.new)
    print(f"parent {len(Xp)}  +  candidates {len(Xn)}   radius {radius:.6f}")

    Rp, Rn = to_rank(Xp, knots), to_rank(Xn, knots)

    # ---- new vs parent ----------------------------------------------------- #
    d_par, _ = cKDTree(Rp).query(Rn, k=1)
    far = d_par > radius
    print(f"  new-vs-parent : {int((~far).sum())} dropped (inside the radius), "
          f"{int(far.sum())} kept   [d range {d_par.min():.4f}-{d_par.max():.4f}]")

    # ---- new vs new (greedy, first-come) ----------------------------------- #
    idx = np.flatnonzero(far)
    tree = cKDTree(Rn[idx])
    alive = np.ones(len(idx), bool)
    keep_local = []
    for i in range(len(idx)):
        if not alive[i]:
            continue
        keep_local.append(i)
        alive[tree.query_ball_point(Rn[idx[i]], radius)] = False
        alive[i] = False
    keep_local = np.array(sorted(keep_local), dtype=np.int64)
    kn = np.sort(idx[keep_local])
    print(f"  new-vs-new    : {len(idx) - len(kn)} dropped (mutually too close), "
          f"{len(kn)} kept")

    n_out = n_par + len(kn)
    print(f"  v3 = {n_par} parent + {len(kn)} inverse-designed = {n_out} rows "
          f"(+{100 * len(kn) / n_par:.3f}%)")

    # ---- materialise -------------------------------------------------------- #
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
            for s in range(0, n_par, CHUNK):
                e = min(s + CHUNK, n_par)
                out[s:e] = d[s:e]
            out[n_par:] = fn[name][:][kn]
            print(f"    {name:15s} {out.shape}")

        prov_par = (fp["provenance"][:] if has_prov
                    else np.zeros(n_par, np.int64))
        row_par = (fp["parent_row"][:] if has_prov
                   else np.arange(n_par, dtype=np.int64))
        for name, v, desc in (
            ("provenance", np.concatenate([prov_par, np.full(len(kn), 2, np.int64)]),
             "0 = original CA cell, 1 = diffusion-generated, 2 = inverse-designed"),
            ("parent_row", np.concatenate([row_par, kn]),
             "row in the source file named by provenance"),
        ):
            dst.create_dataset(name, data=v, dtype=np.int64,
                               chunks=(min(256, n_out),), maxshape=(None,))
            dst[name].attrs["desc"] = desc

        dst.attrs["subset_inverse_designed"] = str(args.new)
        dst.attrs["subset_n"] = n_out
        dst.attrs["subset_n_from_inverse"] = len(kn)
        dst.attrs["subset_method"] = (
            "v2 (translation dedup + rank-space Poisson-disk thinning of parent + "
            "diffusion-upsampled pool), plus inverse-designed gap fills accepted at "
            "the same radius in v2's rank space; no re-thinning of v2")
        dst.attrs["live_fraction_note"] = (
            "-1 sentinel on provenance==2 rows: no CA seed, mask before use")
    print(f"\nwrote {args.output}  ({n_out} rows, "
          f"{args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
