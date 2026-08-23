"""Summarise an inverse-design gap-filling run and export the accepted cells.

Reads the per-target ``.json``/``.npz`` files written by ``fill_gaps_inverse``
and answers the only question that matters: how much of the hole list the run
actually closed, measured the same way the holes were found.

Three outputs:

* a printed summary -- acceptance rate, how far each design moved in rank space,
  how many pixels it took, and the coverage of the *whole* target list once
  every accepted design is counted (one design can close its neighbours' holes
  too, so coverage is higher than the number of runs);
* ``report.png`` -- rank-space coverage before/after plus the per-target
  improvement distribution;
* ``accepted.h5`` -- the accepted cells in the parent dataset's schema, ready
  for ``merge_and_finalize`` to fold into a v3 subset.

``live_fraction`` is written as -1: it is the CA seed density, and an
inverse-designed cell has no CA seed.  Anything training on that column must
mask the sentinel rather than read it as a density.

Usage
-----

    python -m dataset.cellular_chiral.fill_gaps_report \
        -i output/ca_bulk_squared/inverse_fill
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from dataset.cellular_chiral.fill_gaps_inverse import FEATURES, RHO_SOLID, select_failed

_BEFORE, _AFTER, _NEW = "#2a78d6", "#eb6834", "#1baf7a"
_INK, _INK2, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load_results(in_dir: Path) -> list[dict]:
    out = []
    for f in sorted(in_dir.glob("target_*.json")):
        with open(f) as fh:
            d = json.load(fh)
        d["_npz"] = f.with_suffix(".npz")
        out.append(d)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path,
                   default=Path("output/ca_bulk_squared/inverse_fill"))
    p.add_argument("-d", "--dataset", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("-t", "--targets", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/upsample_targets_v1.npz"))
    p.add_argument("-s", "--subset", type=Path,
                   default=Path("output/ca_bulk_squared/subset_uniform_v1.npz"))
    p.add_argument("--no-export", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    res = load_results(args.input)
    if not res:
        raise SystemExit(f"no target_*.json under {args.input}")

    info, rec = select_failed(args.dataset, args.targets, args.subset)
    radius = rec["radius"]
    acc = [r for r in res if r.get("accepted")]
    hit = [r for r in res if r["hit"]]

    secs = np.array([r["seconds"] for r in res])
    d0 = np.array([r["d_hit_seed"] for r in res])
    d1 = np.array([r["d_hit"] for r in res])
    npx = np.array([r.get("n_pixels_changed", -1) for r in res])

    print(f"runs {len(res)}   hit the ball {len(hit)} ({len(hit)/len(res):.0%})   "
          f"accepted {len(acc)} ({len(acc)/len(res):.0%})")
    print(f"  rank distance to target: seed median {np.median(d0):.4f} -> "
          f"designed median {np.median(d1):.4f}   (radius {radius:.4f})")
    if len(acc):
        da = np.array([r["d_nearest_v2"] for r in acc])
        print(f"  accepted designs sit {da.min():.4f}-{da.max():.4f} from the nearest "
              f"existing row (median {np.median(da):.4f}); all above the radius by "
              f"construction")
    ok = npx >= 0
    if ok.any():
        print(f"  pixels changed vs the seed cell: median {np.median(npx[ok]):.0f}  "
              f"p90 {np.percentile(npx[ok], 90):.0f}  max {npx[ok].max():.0f} of 2500")
    print(f"  wall clock per target: median {np.median(secs):.0f}s  "
          f"p90 {np.percentile(secs, 90):.0f}s  total {secs.sum()/3600:.1f} h")
    miss = [r for r in res if not r["hit"]]
    if miss:
        print(f"  {len(miss)} misses:")
        for r in miss[:15]:
            res_txt = ("  worst axis " + max(
                zip(FEATURES, r["rank_residual"]), key=lambda kv: abs(kv[1]))[0]
                if "rank_residual" in r else "")
            print(f"    tid {r['target_id']:6d}: {r['d_hit_seed']:.4f} -> "
                  f"{r['d_hit']:.4f}  (enclosure {r.get('enclosure', -1)}){res_txt}")
        if "rank_residual" in miss[0]:
            R = np.abs(np.array([r["rank_residual"] for r in miss]))
            worst = np.bincount(R.argmax(1), minlength=5)
            print("    axis that blocks the miss: "
                  + "  ".join(f"{c}={n}" for c, n in zip(FEATURES, worst)))

    # ---- coverage of the whole hole list --------------------------------- #
    cand = rec["cand"]
    unfilled = rec["d_all"][cand] > radius
    pts = np.array([r["rank"] for r in acc]) if acc else np.empty((0, 5))
    if len(pts):
        d_new, _ = cKDTree(pts).query(rec["centres"][cand], k=1)
        now_filled = (d_new <= radius) & unfilled
        print(f"\ncoverage: {int(unfilled.sum())} interior holes were unfilled; "
              f"{len(acc)} accepted designs close {int(now_filled.sum())} of them "
              f"({now_filled.sum()/max(unfilled.sum(),1):.1%})  "
              f"[{now_filled.sum()/max(len(acc),1):.2f} holes closed per design]")

    # ---- export ---------------------------------------------------------- #
    if not args.no_export and acc:
        out_h5 = args.input / "accepted.h5"
        cells = np.stack([np.load(r["_npz"])["cell"] for r in acc]).astype(np.uint8)
        C_eff = np.stack([np.load(r["_npz"])["C_eff"] for r in acc]).astype(np.float64)
        A = np.array([r["achieved"] for r in acc], dtype=np.float64)
        with h5py.File(args.dataset, "r") as src, h5py.File(out_h5, "w") as dst:
            for k, v in src.attrs.items():
                dst.attrs[k] = v
            data = {
                "C11": A[:, 0], "C22": A[:, 1], "C12": A[:, 2], "C66": A[:, 3],
                "lambda_": A[:, 2], "mu": A[:, 3], "vol": A[:, 4], "vf": A[:, 4],
                "rho": RHO_SOLID * A[:, 4], "C_eff": C_eff, "cells": cells,
                "live_fraction": np.full(len(acc), -1.0),
                "source_idx": np.array([r["target_id"] for r in acc], dtype=np.int64),
            }
            for name, v in data.items():
                d = src[name]
                dst.create_dataset(name, data=v.astype(d.dtype), chunks=d.chunks,
                                   maxshape=d.maxshape, compression=d.compression,
                                   compression_opts=d.compression_opts)
                for k, val in d.attrs.items():
                    dst[name].attrs[k] = val
            dst.attrs["origin"] = "inverse_design pixel refinement (fill_gaps_inverse)"
            dst.attrs["live_fraction_note"] = "-1 sentinel: no CA seed for these cells"
        print(f"\nwrote {out_h5}  ({len(acc)} cells, parent schema)")

    if not args.no_plot:
        _plot(rec, res, acc, radius, args.input / "report.png")


def _plot(rec, res, acc, radius, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    R = rec["Rv"]
    cand = rec["cand"]
    pts = np.array([r["rank"] for r in acc]) if acc else np.empty((0, 5))
    pairs = [(0, 3, "C11", "C66"), (0, 1, "C11", "C22"), (2, 4, "C12", "vol")]

    fig, axes = plt.subplots(1, 4, figsize=(21, 5), facecolor=_SURFACE)
    rng = np.random.default_rng(0)
    sub = rng.choice(len(R), min(60000, len(R)), replace=False)
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
        ax.scatter(R[sub, a], R[sub, b], s=1, c=_BEFORE, alpha=0.22, linewidths=0,
                   label="v2 dataset" if first else None)
        ax.scatter(rec["centres"][cand, a], rec["centres"][cand, b], s=4, c=_AFTER,
                   alpha=0.5, linewidths=0, label="unfilled holes" if first else None)
        if len(pts):
            ax.scatter(pts[:, a], pts[:, b], s=26, c=_NEW, marker="D", linewidths=0,
                       label="inverse-designed" if first else None)
        ax.set_xlabel(f"rank {la}", fontsize=10, color=_INK2)
        ax.set_ylabel(f"rank {lb}", fontsize=10, color=_INK2)
        if first:
            ax.legend(frameon=False, fontsize=10, labelcolor=_INK2, markerscale=3)

    ax = axes[3]
    ax.set_facecolor(_SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d8d7d3")
    ax.grid(True, color="#ebeae6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=_INK2, labelsize=9)
    d0 = np.array([r["d_hit_seed"] for r in res])
    d1 = np.array([r["d_hit"] for r in res])
    ax.scatter(d0, d1, s=28, c=_NEW, linewidths=0)
    lim = max(d0.max(), d1.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=_INK2, lw=0.8, ls="--")
    ax.axhline(radius, color=_AFTER, lw=1.2)
    ax.text(lim * 0.02, radius * 1.05, "thinning radius", color=_AFTER, fontsize=9)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("seed cell: rank distance to target", fontsize=10, color=_INK2)
    ax.set_ylabel("designed cell", fontsize=10, color=_INK2)

    fig.suptitle("Inverse design of the holes the diffusion upsampling missed",
                 fontsize=13, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130, facecolor=_SURFACE)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
