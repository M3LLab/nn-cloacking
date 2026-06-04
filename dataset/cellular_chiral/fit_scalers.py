"""Fit per-channel StandardScalers on (C11, C22, C12, C66) from stiffness.h5.

Matches the dataset's orthotropic (D2) naming (`scaler_C11`, `scaler_C22`,
`scaler_C12`, `scaler_C66`) so the 2D diffusion code can reuse the same joblib
filenames. `vol` is left in raw [0, 1] units (the 3D pipeline does the same).

Usage:

    python -m dataset.cellular_chiral.fit_scalers \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o microstructure_generation_2d/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input", type=Path,
        default=Path("output/ca_bulk_squared/stiffness.h5"),
        help="Path to stiffness.h5",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        default=Path("microstructure_generation_2d"),
        help="Directory for scaler_C11 / scaler_C22 / scaler_C12 / scaler_C66 joblib files",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as f:
        for name in ("C11", "C22", "C12", "C66"):
            arr = f[name][:].astype(np.float64).reshape(-1, 1)
            scaler = StandardScaler().fit(arr)
            out_path = args.output / f"scaler_{name}"
            joblib.dump(scaler, out_path)
            print(
                f"  {name}: mean={scaler.mean_[0]:.4g} scale={scaler.scale_[0]:.4g} "
                f"-> {out_path}"
            )


if __name__ == "__main__":
    main()
