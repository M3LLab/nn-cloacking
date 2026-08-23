"""Pick which diffusion-generated cells are worth homogenising.

``generated_v2.h5`` holds 7 samples per condition. All 7 aim at the *same* point
in the 5-D space, so homogenising all of them mostly buys near-duplicates of each
other -- at the thinning radius at most one or two can survive per condition. This
script cuts the pool down before any FEM is spent.

Three filters, in order:

1. **Exact-byte duplicates** within the generated set.
2. **Periodic-translation duplicates**, keyed on the cyclic autocorrelation
   ``A(dy,dx) = sum_ij c[i,j] c[i+dy, j+dx]`` (mod 50). It is *exactly* invariant
   under cyclic shift and integer-valued for a binary cell, so it buckets translates
   together with no floating-point tolerance to tune; collisions (homometric but
   genuinely different cells) are then resolved by an exact ``np.roll`` check. The
   same signature is computed for the parent dataset so generated cells that merely
   reproduce an existing design are dropped too.
3. **Best-K per condition** on the free ``|vol - target_vol|`` proxy. ``vol`` is
   exact from the geometry, so this ranks how on-target a sample is without solving
   anything -- and it is one of the five coordinates outright.

Usage
-----

    python -m dataset.cellular_chiral.select_generated \
        -g output/ca_bulk_squared/upsample/generated_v2.h5 \
        -p output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/upsample/selected_v2.npz \
        --per-target 3
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

BATCH = 8192


def autocorr_keys(cells_ds, n: int, verbose: bool = True) -> np.ndarray:
    """Exact translation-invariant 16-byte key per cell, computed in batches."""
    import hashlib

    keys = np.empty(n, dtype="S16")
    t0 = time.time()
    for s in range(0, n, BATCH):
        c = np.asarray(cells_ds[s : s + BATCH], dtype=np.float64)
        F = np.fft.rfft2(c, axes=(1, 2))
        A = np.fft.irfft2(F * np.conj(F), s=c.shape[1:], axes=(1, 2))
        A = np.rint(A).astype(np.int32)          # integer-exact for binary cells
        for k, row in enumerate(A):
            keys[s + k] = hashlib.blake2b(row.tobytes(), digest_size=16).digest()
        if verbose and (s // BATCH) % 4 == 0:
            print(f"    autocorr {s + len(c)}/{n}  ({time.time() - t0:.0f}s)", flush=True)
    return keys


def _is_translate(a: np.ndarray, b: np.ndarray) -> bool:
    if np.array_equal(a, b):
        return True
    fa, fb = np.fft.rfft2(a.astype(np.float64)), np.fft.rfft2(b.astype(np.float64))
    cross = fa * np.conj(fb)
    mag = np.abs(cross)
    mag[mag == 0] = 1.0
    corr = np.fft.irfft2(cross / mag, s=a.shape)
    dy, dx = np.unravel_index(np.argmax(corr), corr.shape)
    return np.array_equal(np.roll(a, (-int(dy), -int(dx)), axis=(0, 1)), b)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-g", "--generated", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/generated_v2.h5"))
    p.add_argument("-p", "--parent", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/selected_v2.npz"))
    p.add_argument("--per-target", type=int, default=7,
                   help="cap of samples kept per condition; the pilot showed the "
                        "vol proxy does NOT predict which sample lands on target, "
                        "so the default keeps them all and lets thinning decide")
    p.add_argument("--max-dvol", type=float, default=1.0,
                   help="drop samples whose realised vol misses the target by more")
    p.add_argument("--export-dir", type=Path, default=None,
                   help="also write cells.npy + live_fractions.npy here, the input "
                        "layout bulk_stiffness expects")
    p.add_argument("--skip-parent", action="store_true",
                   help="skip the cross-check against the parent dataset's geometries")
    args = p.parse_args()

    g = h5py.File(args.generated, "r")
    n = g["cells"].shape[0]
    tid, vol, tvol = g["target_id"][:], g["vol"][:], g["target_vol"][:]
    region = g["region"][:]
    print(f"generated: {n} cells, {np.unique(tid).size} conditions")

    print("  computing translation-invariant keys (generated)")
    gk = autocorr_keys(g["cells"], n)

    alive = np.ones(n, dtype=bool)

    # ---- 1+2: collapse translation classes inside the generated set ----------
    order = np.argsort(gk, kind="stable")
    groups, start = [], 0
    sk = gk[order]
    for i in range(1, len(sk) + 1):
        if i == len(sk) or sk[i] != sk[start]:
            if i - start > 1:
                groups.append(order[start:i])
            start = i
    n_tr = 0
    for grp in groups:
        cells = g["cells"][np.sort(grp)]
        idx = np.sort(grp)
        reps = []
        for k, row in enumerate(idx):
            if any(_is_translate(cells[k], cells[r]) for r in reps):
                alive[row] = False
                n_tr += 1
            else:
                reps.append(k)
    print(f"  dropped {n_tr} duplicate/translate geometries within the generated set")

    # ---- cross-check against the parent dataset -----------------------------
    if not args.skip_parent:
        with h5py.File(args.parent, "r") as pf:
            m = pf["cells"].shape[0]
            print(f"  computing translation-invariant keys (parent, {m} cells)")
            pk = autocorr_keys(pf["cells"], m)
        seen = np.isin(gk, pk)
        n_seen = int(np.sum(seen & alive))
        alive &= ~seen
        print(f"  dropped {n_seen} generated cells already present in the parent")

    print(f"  {alive.sum()} unique geometries remain")

    # ---- 3: best-K per condition on the free vol proxy ----------------------
    dvol = np.abs(vol - tvol)
    alive &= dvol <= args.max_dvol
    print(f"  {alive.sum()} remain after |dvol| <= {args.max_dvol}")

    idx_alive = np.flatnonzero(alive)
    keyorder = np.lexsort((dvol[idx_alive], tid[idx_alive]))
    idx_sorted = idx_alive[keyorder]
    t_sorted = tid[idx_sorted]
    rank_in_group = np.ones(len(idx_sorted), dtype=np.int64)
    rank_in_group[0] = 0
    for i in range(1, len(idx_sorted)):
        rank_in_group[i] = 0 if t_sorted[i] != t_sorted[i - 1] else rank_in_group[i - 1] + 1
    sel = np.sort(idx_sorted[rank_in_group < args.per_target])

    print(f"\n  selected {len(sel)} cells "
          f"({np.unique(tid[sel]).size} conditions covered of {np.unique(tid).size})")
    print(f"    region 0 (holes): {int(np.sum(region[sel] == 0))}   "
          f"region 1 (frontier): {int(np.sum(region[sel] == 1))}")
    print(f"    |dvol| of selected: med={np.median(dvol[sel]):.4f} "
          f"p90={np.percentile(dvol[sel], 90):.4f} max={dvol[sel].max():.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, idx=sel, target_id=tid[sel], region=region[sel],
                        dvol=dvol[sel], per_target=args.per_target,
                        generated=str(args.generated))
    print(f"  wrote {args.output}")

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        out = np.empty((len(sel), 50, 50), dtype=np.uint8)
        for s in range(0, len(sel), BATCH):
            out[s : s + BATCH] = g["cells"][sel[s : s + BATCH]]
        np.save(args.export_dir / "cells.npy", out)
        np.save(args.export_dir / "live_fractions.npy", g["live_fraction"][:][sel])
        np.savez(args.export_dir / "map.npz", gen_row=sel, target_id=tid[sel],
                 region=region[sel])
        print(f"  exported {len(sel)} cells to {args.export_dir} "
              f"(cells.npy + live_fractions.npy + map.npz)")


if __name__ == "__main__":
    main()
