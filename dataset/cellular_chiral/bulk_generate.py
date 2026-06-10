"""Bulk-generate squared-assembly CA unit cells with varied live_fraction.

Stores all structures in one contiguous memmapped .npy plus a sidecar of the
per-sample live_fraction values. No PNGs are produced.

    python -m dataset.cellular_chiral.bulk_generate -n 1000000 -o output/ca_bulk_squared --assembly squared
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .generator import CAConfig, generate_unit_cell


_LF: np.ndarray | None = None
_BASE_SEED: int = 0


def _init_worker(live_fractions: np.ndarray, base_seed: int) -> None:
    global _LF, _BASE_SEED
    _LF = live_fractions
    _BASE_SEED = base_seed


def _generate_one(idx: int, assembly: str) -> tuple[int, np.ndarray]:
    lf = float(_LF[idx])
    cell, _ = generate_unit_cell(
        config=CAConfig(live_fraction=lf),
        seed=_BASE_SEED + idx,
        assembly=assembly,
    )
    return idx, cell.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bulk CA unit-cell generator")
    parser.add_argument("-n", "--num", type=int, default=1_000_000, help="Number of cells")
    parser.add_argument("--assembly", type=str, default="squared", help="Assembly type (chiral or squared). Default is squared.")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Base RNG seed")
    parser.add_argument("-o", "--output", type=Path, default=Path("output/unit_cells_D2"))
    parser.add_argument("--lf-min", type=float, default=0.20, help="Min live_fraction")
    parser.add_argument("--lf-max", type=float, default=0.80, help="Max live_fraction")
    parser.add_argument("-j", "--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--max-bytes", type=int, default=50 * 1024**3,
                        help="Hard cap on total cells.npy size in bytes (default 50 GiB)")
    args = parser.parse_args()

    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    live_fractions = rng.uniform(args.lf_min, args.lf_max, size=args.num).astype(np.float32)

    probe, _ = generate_unit_cell(
        config=CAConfig(live_fraction=float(live_fractions[0])),
        seed=args.seed, assembly=args.assembly,
    )
    H, W = probe.shape

    bytes_total = args.num * H * W
    if bytes_total > args.max_bytes:
        raise SystemExit(
            f"Refusing: {args.num} x {H}x{W} = {bytes_total/1e9:.2f} GB exceeds cap "
            f"{args.max_bytes/1e9:.2f} GB."
        )

    cells_path = out_dir / "cells.npy"
    cells = np.lib.format.open_memmap(
        cells_path, mode="w+", dtype=np.uint8, shape=(args.num, H, W)
    )

    chunksize = max(64, args.num // max(1, args.workers * 64))
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(live_fractions, args.seed),
    ) as pool:
        for idx, cell in tqdm(
            pool.imap_unordered(functools.partial(_generate_one, assembly=args.assembly), range(args.num), chunksize=chunksize),
            total=args.num, desc="generate", unit="cell",
        ):
            cells[idx] = cell

    cells.flush()
    np.save(out_dir / "live_fractions.npy", live_fractions)

    print(
        f"Saved {args.num} cells of shape {H}x{W} ({bytes_total/1e9:.2f} GB) -> {cells_path}\n"
        f"live_fractions sampled uniformly in [{args.lf_min}, {args.lf_max}] "
        f"-> {out_dir / 'live_fractions.npy'}"
    )


if __name__ == "__main__":
    main()
