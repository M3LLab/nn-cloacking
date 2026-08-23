"""Materialise a subset index (from ``thin_uniform``) as a standalone stiffness HDF5.

``thin_uniform`` deliberately writes only an index array, so the source dataset is
never rewritten. This script turns that index into a real HDF5 with **exactly the
same structure** as the parent — same dataset names, dtypes, chunking, compression
and resizability, same attributes — so every existing reader
(``diffusion_dataset``, ``fit_scalers``, ``fit_gmm``, ``tile_matched_microstructure``)
works against it unchanged.

Two things are added, both additive:

* a ``parent_row`` dataset mapping each output row back to its row in the parent
  file (``source_idx`` is left alone -- it still points into ``cells.npy``),
* ``subset_*`` attributes recording the provenance of the thinning.

Usage
-----

    python -m dataset.cellular_chiral.materialize_subset \
        -i output/ca_bulk_squared/stiffness.h5 \
        -s output/ca_bulk_squared/subset_uniform_v1.npz \
        -o output/ca_bulk_squared/stiffness_tri6_uniform_v1.h5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

CHUNK_ROWS = 8192  # rows copied per pass; bounds peak memory on `cells`


def materialize(src_path: Path, idx: np.ndarray, dst_path: Path,
                meta: dict | None = None, verbose: bool = True) -> None:
    idx = np.asarray(idx, dtype=np.int64)
    if not np.all(np.diff(idx) > 0):
        idx = np.unique(idx)  # h5py fancy indexing requires strictly increasing
    n_out = len(idx)

    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w") as dst:
        n_src = src[next(iter(src))].shape[0]
        if idx[-1] >= n_src:
            raise ValueError(f"index {idx[-1]} out of range for {n_src} source rows")

        for k, v in src.attrs.items():
            dst.attrs[k] = v

        for name in src:
            d = src[name]
            if d.shape[0] != n_src:
                raise ValueError(f"{name}: leading axis {d.shape[0]} != {n_src}")
            out = dst.create_dataset(
                name,
                shape=(n_out,) + d.shape[1:],
                maxshape=d.maxshape,
                dtype=d.dtype,
                chunks=d.chunks,
                compression=d.compression,
                compression_opts=d.compression_opts,
            )
            for k, v in d.attrs.items():
                out.attrs[k] = v
            t0 = time.time()
            for s in range(0, n_out, CHUNK_ROWS):
                out[s : s + CHUNK_ROWS] = d[idx[s : s + CHUNK_ROWS]]
            if verbose:
                print(f"  {name:15s} {str(out.shape):20s} {time.time() - t0:6.1f}s")

        pr = dst.create_dataset("parent_row", data=idx, dtype=np.int64,
                                chunks=(min(256, n_out),), maxshape=(None,))
        pr.attrs["desc"] = f"row index into the parent file {src_path.name}"

        dst.attrs["subset_parent"] = str(src_path)
        dst.attrs["subset_n_parent"] = n_src
        dst.attrs["subset_n"] = n_out
        for k, v in (meta or {}).items():
            dst.attrs[f"subset_{k}"] = v


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v1.h5"))
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = p.parse_args()

    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    z = np.load(args.subset, allow_pickle=False)
    idx = z["idx"]
    meta = {
        "method": "translation-dedup + rank-space Poisson-disk thinning",
        "index_file": str(args.subset),
        "radius_rank_units": float(z["radius"]),
        "seed": int(z["seed"]),
        "feature_order": ", ".join(str(f) for f in z["feature_order"]),
        "n_after_dedup": int(len(z["survived_dedup"])),
    }
    print(f"{args.input} -> {args.output}   ({len(idx)} of {int(z['n_source'])} rows)")
    materialize(args.input, idx, args.output, meta)
    print(f"wrote {args.output}  "
          f"({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
