#!/usr/bin/env python
"""Augment a surrogate dataset with best-sample frequency sweeps.

For each unique source frequency in an existing HDF5 dataset, takes the sample
with the lowest loss and re-evaluates it at a grid of N target frequencies
spanning the dataset's frequency range (same C and rho, only omega changes).
The new (f_star, loss) entries are appended to the same HDF5 file with
sample_type ``augment_from_f{source:.3f}``.

The base YAML config must be the same one used to generate the dataset
(geometry, mesh, cells, loss type must match).

Usage::

    # Default: 32 target freqs across the dataset's existing range
    python scripts/augment_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        output/surrogate_dataset.h5

    # Custom target grid
    python scripts/augment_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        output/surrogate_dataset.h5 --n-target 32 --f-min 0.5 --f-max 3.0

    # Use the top-3 best samples per source freq (3x more evaluations)
    python scripts/augment_surrogate_dataset.py configs/surrogate_dataset.yaml \\
        output/surrogate_dataset.h5 --top-k 3
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# Swallow the jax-fem banner (a bare `print` at module import).
with contextlib.redirect_stdout(io.StringIO()):
    import jax_fem  # noqa: F401

logging.getLogger("jax_fem").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.ERROR)

from rayleigh_cloak import load_config
from dataset_gen.augment import AugmentConfig, run_dataset_augmentation


def main():
    parser = argparse.ArgumentParser(
        description="Augment surrogate dataset with best-sample frequency sweeps",
    )
    parser.add_argument(
        "config",
        help="Base YAML config (same one used to generate the dataset)",
    )
    parser.add_argument(
        "dataset",
        help="Path to existing HDF5 dataset to augment (modified in place)",
    )

    parser.add_argument("--n-target", type=int, default=32,
                        help="Number of target frequencies (default: 32)")
    parser.add_argument("--f-min", type=float, default=None,
                        help="Min target frequency (default: dataset's own min)")
    parser.add_argument("--f-max", type=float, default=None,
                        help="Max target frequency (default: dataset's own max)")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Number of best samples per source frequency (default: 1)")

    args = parser.parse_args()

    base_config = load_config(args.config)
    if not base_config.cells.enabled:
        print("Error: cells.enabled must be true in the base config.")
        sys.exit(1)

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: dataset file not found: {dataset_path}")
        sys.exit(1)

    aug_config = AugmentConfig(
        n_target_freqs=args.n_target,
        f_min=args.f_min,
        f_max=args.f_max,
        top_k=args.top_k,
    )

    print(f"Base config: {args.config}")
    print(f"Dataset:     {dataset_path}")
    print(f"Target:      {args.n_target} freqs "
          f"(f_min={args.f_min if args.f_min is not None else 'auto'}, "
          f"f_max={args.f_max if args.f_max is not None else 'auto'})")
    print(f"Top-k:       {args.top_k} best samples per source freq")
    print()

    run_dataset_augmentation(base_config, dataset_path, aug_config)


if __name__ == "__main__":
    main()
