"""Extend an existing bulk CA dataset with N more cells in a given live_fraction
range, written to a NEW directory (the source is NEVER modified).

Produces a self-contained copy: ``cells.npy`` and ``live_fractions.npy`` hold the
original samples first, followed by the newly generated ones. Auxiliary files
(``stiffness.h5``, ``gmm_*.npz``) are copied verbatim so the output is a true
superset-copy. Because the originals keep their positions, every ``source_idx``
already stored in the copied ``stiffness.h5`` still points at the same geometry;
the appended cells have NO stiffness yet -- fill them in later with:

    python -m dataset.cellular_chiral.bulk_stiffness \
        -i <output> --start <N_old> --resume

New cells are seeded with ``seed_offset + i`` (default seed_offset = N_old) so
they continue the source's ``seed == global_index`` scheme and cannot collide
with existing structures.

Example -- append 500k dense cells (low live_fraction == high solid fraction)::

    python -m dataset.cellular_chiral.extend_dataset \
        --source output/ca_bulk_squared -o output/ca_bulk_squared_dense \
        -n 500000 --lf-min 0.0 --lf-max 0.20 --assembly squared
"""
from __future__ import annotations

import argparse
import functools
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Reuse the exact generation worker the original dataset was built with, so the
# CA parameters, seeding (seed == base + index) and parallel pattern all match.
from .bulk_generate import _generate_one, _init_worker

_AUX_FILES = ("stiffness.h5", "gmm_lambda_mu_rho.npz")
_COPY_CHUNK = 50_000  # rows per streaming-copy block (~125 MB at 50x50 uint8)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", type=Path, default=Path("output/ca_bulk_squared"),
                        help="Existing dataset dir (read-only) holding cells.npy + live_fractions.npy")
    parser.add_argument("-o", "--output", type=Path, required=True,
                        help="New dataset dir to create (must not already contain cells.npy)")
    parser.add_argument("-n", "--num", type=int, default=500_000, help="Number of NEW cells to append")
    parser.add_argument("--assembly", type=str, default="squared", help="Assembly type (squared or chiral)")
    parser.add_argument("--lf-min", type=float, default=0.0, help="Min live_fraction for new cells")
    parser.add_argument("--lf-max", type=float, default=0.20, help="Max live_fraction for new cells")
    parser.add_argument("--seed-offset", type=int, default=None,
                        help="Base seed for new cells (seed=offset+i). Default: N_old (continues the scheme).")
    parser.add_argument("--lf-seed", type=int, default=None,
                        help="RNG seed for sampling the new live_fractions. Default: seed-offset.")
    parser.add_argument("-j", "--workers", type=int, default=mp.cpu_count())
    parser.add_argument("--no-aux", action="store_true",
                        help="Do not copy stiffness.h5 / gmm_*.npz into the new dir.")
    args = parser.parse_args()

    src: Path = args.source
    out: Path = args.output
    src_cells_path = src / "cells.npy"
    src_lf_path = src / "live_fractions.npy"
    if not src_cells_path.exists() or not src_lf_path.exists():
        raise SystemExit(f"Missing cells.npy / live_fractions.npy in {src}")

    out.mkdir(parents=True, exist_ok=True)
    out_cells_path = out / "cells.npy"
    if out_cells_path.exists():
        raise SystemExit(f"Refusing to overwrite existing {out_cells_path}")

    src_cells = np.load(src_cells_path, mmap_mode="r")
    src_lf = np.load(src_lf_path)
    n_old, H, W = src_cells.shape
    assert src_lf.shape == (n_old,), "source cells/live_fractions length mismatch"
    n_new = args.num
    n_total = n_old + n_new

    seed_offset = args.seed_offset if args.seed_offset is not None else n_old
    lf_seed = args.lf_seed if args.lf_seed is not None else seed_offset

    # Sample the new live_fractions (float32 to match the source sidecar).
    lf_rng = np.random.default_rng(lf_seed)
    new_lf = lf_rng.uniform(args.lf_min, args.lf_max, size=n_new).astype(np.float32)

    print(f"Source : {src}  ({n_old} cells, {H}x{W})")
    print(f"Append : {n_new} cells, live_fraction ~ U[{args.lf_min}, {args.lf_max}], "
          f"seed = {seed_offset}+i")
    print(f"Output : {out}  ({n_total} cells total, "
          f"{n_total*H*W/1e9:.2f} GB)\n")

    # Allocate the combined cells.npy and stream the originals in.
    cells = np.lib.format.open_memmap(
        out_cells_path, mode="w+", dtype=np.uint8, shape=(n_total, H, W)
    )
    for a in tqdm(range(0, n_old, _COPY_CHUNK), desc="copy source", unit_scale=_COPY_CHUNK, unit="cell"):
        b = min(a + _COPY_CHUNK, n_old)
        cells[a:b] = src_cells[a:b]

    # Generate the new cells in parallel directly into the tail.
    chunksize = max(64, n_new // max(1, args.workers * 64))
    with mp.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(new_lf, seed_offset),
    ) as pool:
        for local_idx, cell in tqdm(
            pool.imap_unordered(
                functools.partial(_generate_one, assembly=args.assembly),
                range(n_new), chunksize=chunksize,
            ),
            total=n_new, desc="generate", unit="cell",
        ):
            cells[n_old + local_idx] = cell
    cells.flush()

    # Combined live_fractions sidecar.
    np.save(out / "live_fractions.npy", np.concatenate([src_lf, new_lf]))

    # Copy auxiliary files verbatim (kept consistent: source_idx < n_old still valid).
    if not args.no_aux:
        for name in _AUX_FILES:
            p = src / name
            if p.exists():
                shutil.copy2(p, out / name)
                print(f"copied aux: {name}")

    print(
        f"\nDone. {out_cells_path} now holds {n_total} cells "
        f"(orig [0:{n_old}] + new [{n_old}:{n_total}]).\n"
        f"Stiffness for the new cells is NOT computed; run:\n"
        f"    python -m dataset.cellular_chiral.bulk_stiffness "
        f"-i {out} --start {n_old} --resume"
    )


if __name__ == "__main__":
    main()
