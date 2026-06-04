"""Fix the orthotropic stiffness params in an existing stiffness HDF5 in place.

The squared (mirror) assembly is D2, not D4, so C11 (x-axial) and C22 (y-axial)
are genuinely different. Older files (and the buggy ``backfill_d4_params.py``)
stored ``C11 = 0.5*(C[0,0] + C[1,1])`` -- the average of the two axes -- and
never wrote ``C22`` at all, silently discarding the dominant in-plane
anisotropy.

The full augmented Voigt tensor ``C_eff`` is already stored, so the correct
constants can be recovered offline without re-running FEM:
    C11 = C[:,0,0]                         (x-axial; was wrongly averaged)
    C22 = C[:,1,1]                         (y-axial; was missing)
    C12 = 0.5*(C[:,0,1] + C[:,1,0])        (unchanged; equal pair averaged)
    C66 = 0.5*(C[:,2,2] + C[:,3,3])        (unchanged; equal pair averaged)

Idempotent: safe to re-run. The operation is fully reversible because C_eff is
never touched.

Usage
-----
    python -m dataset.cellular_chiral.backfill_ortho_params \
        output/ca_bulk_squared/stiffness.h5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", type=Path, help="stiffness.h5 to update in place")
    args = ap.parse_args()

    with h5py.File(args.path, "a") as f:
        if "C_eff" not in f:
            raise SystemExit(f"{args.path}: no C_eff dataset found")
        C = f["C_eff"][:]
        N = C.shape[0]

        params = {
            "C11": C[:, 0, 0],
            "C22": C[:, 1, 1],
            "C12": 0.5 * (C[:, 0, 1] + C[:, 1, 0]),
            "C66": 0.5 * (C[:, 2, 2] + C[:, 3, 3]),
        }
        for name, vals in params.items():
            if name in f:
                f[name][...] = vals
                print(f"  overwrote {name:>4}  (N={N})")
            else:
                f.create_dataset(name, data=vals, chunks=(min(N, 256),),
                                 maxshape=(None,), dtype=np.float64)
                print(f"  created   {name:>4}  (N={N})")

        # lambda_/mu remain aliases of C12/C66 -- keep them in sync.
        if "lambda_" in f:
            f["lambda_"][...] = params["C12"]
        if "mu" in f:
            f["mu"][...] = params["C66"]

        f.attrs["stiffness_params"] = "C11, C22, C12, C66 (orthotropic D2 class, 2D)"
        f.attrs["lame_aliases"] = "lambda_ := C12, mu := C66 (back-compat; assume isotropy)"

    # Report the anisotropy that the old D4 average was hiding.
    ratio = np.maximum(params["C11"], params["C22"]) / np.clip(
        np.minimum(params["C11"], params["C22"]), 1e-30, None)
    print(f"  C11/C22 axis-stiffness ratio: median {np.median(ratio):.2f}x, "
          f"p90 {np.percentile(ratio, 90):.2f}x, max {ratio.max():.2f}x")
    print(f"done -> {args.path}")


if __name__ == "__main__":
    main()
