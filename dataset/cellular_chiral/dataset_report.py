"""Distribution report for a thinned/upsampled stiffness dataset.

Prints marginal statistics, uniformity metrics, physical-bound utilisation and
provenance breakdown, and writes a comparison figure against the reference
(pre-thinning) pool and any earlier version.

Usage
-----

    python -m dataset.cellular_chiral.dataset_report \
        -f output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5 \
        -r output/ca_bulk_squared/stiffness.h5 \
        -v1 output/ca_bulk_squared/stiffness_tri6_uniform_v1.h5 \
        -t output/ca_bulk_squared/upsample/upsample_targets_v1.npz \
        -o output/ca_bulk_squared/dataset_report_v2.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from dataset.cellular_chiral.bounds import check_attainable
from dataset.cellular_chiral.thin_uniform import FEATURES, _occupancy_cv, rank_transform

_REF = "#2a78d6"
_V1 = "#eb6834"
_V2 = "#1baf7a"
_INK, _INK2, _SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
UNITS = {"C11": "Pa", "C22": "Pa", "C12": "Pa", "C66": "Pa", "vol": "-"}

# provenance codes as written by merge_and_finalize / merge_inverse_fill
PROV = {0: ("original CA", _REF), 1: ("diffusion-generated", _V2),
        2: ("inverse-designed", "#d4276e")}


def _lw(k: int) -> float:
    """Taper line width down the series so a curve that exactly overlays an
    earlier one (v2 vs v3 differ by 0.1 % of rows) stays visible under it."""
    return 3.4 - 1.1 * k


def load5(path: Path):
    with h5py.File(path, "r") as f:
        return np.stack([f[k][:] for k in FEATURES], 1).astype(np.float64)


def marginals(name: str, X: np.ndarray) -> None:
    print(f"\n  {name}  (n = {len(X)})")
    print(f"    {'coord':6s} {'min':>11s} {'p1':>11s} {'p25':>11s} {'median':>11s} "
          f"{'p75':>11s} {'p99':>11s} {'max':>11s}")
    for j, f in enumerate(FEATURES):
        q = np.percentile(X[:, j], [0, 1, 25, 50, 75, 99, 100])
        print(f"    {f:6s} " + " ".join(f"{v:11.4g}" for v in q))
    rho = 2300.0 * X[:, 4]
    q = np.percentile(rho, [0, 1, 25, 50, 75, 99, 100])
    print(f"    {'rho':6s} " + " ".join(f"{v:11.4g}" for v in q))


def common_metric(sets):
    """One rank transform shared by every set.

    Rank-transforming each set on itself would be meaningless for comparison: an
    already-uniform subset gets its marginals re-spread to [0,1] and then looks
    lumpy in 5-D. Uniformity of a subset only means something measured in a metric
    fixed by the pool, so the knots are fitted once on the union.
    """
    knots = rank_transform(np.vstack(sets))[1]

    def apply(X):
        out = np.empty_like(X, dtype=np.float64)
        for j in range(X.shape[1]):
            v, q = knots[j]
            out[:, j] = np.interp(X[:, j], v, q)
        return out

    return apply


def uniformity(name: str, R: np.ndarray) -> None:
    print(f"\n  {name}")
    for B in (16, 24, 32):
        occ, cv = _occupancy_cv(R, B)
        print(f"    B={B:2d}^5  occupied={occ:7d}  CV={cv:6.2f}")
    d, _ = cKDTree(R).query(R, k=2, workers=-1)
    d1 = d[:, 1]
    print(f"    1-NN spacing: p1={np.percentile(d1,1):.5f}  "
          f"median={np.median(d1):.5f}  p99={np.percentile(d1,99):.5f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-f", "--final", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v2.h5"))
    p.add_argument("-r", "--reference", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness.h5"))
    p.add_argument("--v1", type=Path,
                   default=Path("output/ca_bulk_squared/stiffness_tri6_uniform_v1.h5"))
    p.add_argument("-t", "--targets", type=Path,
                   default=Path("output/ca_bulk_squared/upsample/upsample_targets_v1.npz"))
    p.add_argument("-o", "--output", type=Path,
                   default=Path("output/ca_bulk_squared/dataset_report_v2.png"))
    args = p.parse_args()

    Xf, Xr = load5(args.final), load5(args.reference)
    Xv1 = load5(args.v1) if args.v1.exists() else None
    with h5py.File(args.final, "r") as f:
        prov = f["provenance"][:]
        attrs = dict(f.attrs)

    prev = args.v1.stem.split("_")[-1] if Xv1 is not None else None
    final = args.final.stem.split("_")[-1]

    print("=" * 78)
    print(f"  {args.final.name}: {len(Xf)} rows   "
          f"{attrs.get('homog_ele_type')} @ {attrs.get('homog_elem_per_pixel')} elem/pixel")
    print("  provenance: " + "  +  ".join(
        f"{int((prov == k).sum())} {PROV.get(k, (f'code {k}', None))[0]} "
        f"({100 * (prov == k).mean():.2f}%)" for k in np.unique(prov)))
    print("=" * 78)

    print("\n" + "-" * 78 + "\nMARGINAL DISTRIBUTIONS\n" + "-" * 78)
    marginals(f"reference pool ({args.reference.name})", Xr)
    if Xv1 is not None:
        marginals(f"{prev}  <-- BEFORE", Xv1)
    marginals(f"{final}  <-- AFTER", Xf)

    print("\n" + "-" * 78 +
          "\nUNIFORMITY (all sets in ONE common rank metric fitted on the union)\n"
          + "-" * 78)
    sets = [Xr, Xf] + ([Xv1] if Xv1 is not None else [])
    to_rank = common_metric(sets)
    uniformity("reference pool", to_rank(Xr))
    if Xv1 is not None:
        uniformity(f"{prev}  <-- BEFORE", to_rank(Xv1))
    uniformity(f"{final}  <-- AFTER", to_rank(Xf))

    print("\n" + "-" * 78 + "\nPHYSICAL BOUND UTILISATION\n" + "-" * 78)
    for name, X in (("reference pool", Xr), (f"{final} final", Xf)):
        ok, d = check_attainable(*X[:, :4].T, vol=X[:, 4])
        print(f"\n  {name}: violations={int((~ok).sum())}")
        for k, v in d.items():
            if k == "pd_margin":
                print(f"    {k:16s} min={v.min():.4f}  median={np.median(v):.4f}")
            else:
                print(f"    {k:16s} median={np.median(v):.4f}  p99={np.percentile(v,99):.4f}"
                      f"  max={v.max():.4f}")

    print("\n" + "-" * 78 + "\nENVELOPE CHANGE (reference pool -> final)\n" + "-" * 78)
    for j, f in enumerate(FEATURES):
        lo0, hi0 = Xr[:, j].min(), Xr[:, j].max()
        lo1, hi1 = Xf[:, j].min(), Xf[:, j].max()
        print(f"    {f:5s} [{lo0:11.4g}, {hi0:11.4g}] -> [{lo1:11.4g}, {hi1:11.4g}]"
              f"   span x{(hi1-lo1)/(hi0-lo0):.2f}")

    # did the requested holes actually get filled?
    if args.targets.exists():
        z = np.load(args.targets, allow_pickle=False)
        cols = list(z["columns"])
        P = np.column_stack([z["targets"][:, cols.index(c)] for c in FEATURES])
        reg = z["region"]
        Rall, knots = rank_transform(np.vstack([Xr, Xf, P]))
        nr, nf = len(Xr), len(Xf)
        Rr, Rf, Rt = Rall[:nr], Rall[nr:nr+nf], Rall[nr+nf:]
        rad = float(attrs.get("subset_radius_rank_units", 0.0276))
        d_before, _ = cKDTree(Rr).query(Rt, k=1, workers=-1)
        d_after, _ = cKDTree(Rf).query(Rt, k=1, workers=-1)
        print("\n" + "-" * 78 + "\nREQUESTED-TARGET COVERAGE\n" + "-" * 78)
        for label, m in (("interior holes", reg == 0), ("HS frontier", reg == 1)):
            print(f"\n  {label} (n={int(m.sum())})")
            print(f"    distance to nearest dataset entry, before: "
                  f"median={np.median(d_before[m]):.4f}  within r={np.mean(d_before[m]<rad):.3f}")
            print(f"    distance to nearest dataset entry, after:  "
                  f"median={np.median(d_after[m]):.4f}  within r={np.mean(d_after[m]<rad):.3f}")

    _plot(Xr, Xv1, Xf, prov, args.output, prev, final)


def _plot(Xr, Xv1, Xf, prov, out_png: Path, prev=None, final="final") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(21, 8), facecolor=_SURFACE)
    for ax in axes.ravel():
        ax.set_facecolor(_SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d8d7d3")
        ax.tick_params(colors=_INK2, labelsize=9)
        ax.grid(True, color="#ebeae6", linewidth=0.8)
        ax.set_axisbelow(True)

    series = [("reference pool", Xr, _REF)]
    if Xv1 is not None:
        series.append((f"{prev} (before)", Xv1, _V1))
    series.append((f"{final} (after)", Xf, _V2))

    # rows unique to the final set (highest provenance code): too few to move a
    # marginal, so mark where they land instead of letting them vanish
    newest = int(np.max(prov))
    new_m = (prov == newest) if newest > 1 else np.zeros(len(prov), bool)
    new_lab, new_c = PROV.get(newest, ("new", "#d4276e"))

    for j, f in enumerate(FEATURES):
        ax = axes[0, j]
        lo = min(s[1][:, j].min() for s in series)
        hi = max(s[1][:, j].max() for s in series)
        bins = np.linspace(lo, hi, 90)
        # widest line first: "before" and "after" can coincide to the pixel
        for k, (lab, X, c) in enumerate(series):
            ax.hist(X[:, j], bins=bins, density=True, histtype="step", color=c,
                    linewidth=_lw(k), alpha=0.9, label=lab if j == 0 else None)
        ax.set_yscale("log")
        if new_m.any():
            # axis-fraction y, so the rug sits on the baseline of a log axis
            ax.vlines(Xf[new_m, j], 0, 0.06, transform=ax.get_xaxis_transform(),
                      color=new_c, linewidth=0.9, alpha=0.75, zorder=4,
                      label=f"{new_lab} ({int(new_m.sum())})" if j == 0 else None)
        ax.set_title(f"{f}  [{UNITS[f]}]", fontsize=11, color=_INK)
        if j == 0:
            ax.set_ylabel("density (log)", fontsize=10, color=_INK2)
            ax.legend(frameon=False, fontsize=9, labelcolor=_INK2)

    # rho
    ax = axes[1, 0]
    for k, (lab, X, c) in enumerate(series):
        ax.hist(2300 * X[:, 4], bins=90, density=True, histtype="step",
                color=c, linewidth=_lw(k), alpha=0.9, label=lab)
    ax.set_title("rho  [kg/m^3]", fontsize=11, color=_INK)
    ax.set_ylabel("density", fontsize=10, color=_INK2)

    # HS utilisation
    ax = axes[1, 1]
    for k, (lab, X, c) in enumerate(series):
        _, d = check_attainable(*X[:, :4].T, vol=X[:, 4])
        ax.hist(d["hs_K_util"], bins=90, density=True, histtype="step",
                color=c, linewidth=_lw(k), alpha=0.9)
    ax.set_title("HS bulk-bound utilisation", fontsize=11, color=_INK)
    ax.set_xlabel("K / K_HS+", fontsize=10, color=_INK2)

    # anisotropy
    ax = axes[1, 2]
    for k, (lab, X, c) in enumerate(series):
        ax.hist(np.log10(X[:, 0] / X[:, 1]), bins=90, density=True,
                histtype="step", color=c, linewidth=_lw(k), alpha=0.9)
    ax.set_title("anisotropy  log10(C11/C22)", fontsize=11, color=_INK)

    # per-bin occupancy
    ax = axes[1, 3]
    to_rank = common_metric([s[1] for s in series])
    for k, (lab, X, c) in enumerate(series):
        R = to_rank(X)
        B = 24
        idx = np.clip((R * B).astype(np.int64), 0, B - 1)
        flat = np.zeros(len(R), dtype=np.int64)
        for j in range(5):
            flat = flat * B + idx[:, j]
        _, cnt = np.unique(flat, return_counts=True)
        ax.hist(np.log10(cnt), bins=60, density=True, histtype="step",
                color=c, linewidth=_lw(k), alpha=0.9)
    ax.set_title("points per occupied bin (B=24^5)", fontsize=11, color=_INK)
    ax.set_xlabel("log10 count", fontsize=10, color=_INK2)

    # provenance in the final set
    ax = axes[1, 4]
    Rf = to_rank(Xf)
    # rarest class last and largest, or 223 inverse-designed rows vanish under 218k
    for k in sorted(np.unique(prov), key=lambda k: -int((prov == k).sum())):
        lab, c = PROV.get(k, (f"code {k}", _INK2))
        m = prov == k
        rare = m.sum() < 0.01 * len(prov)
        ax.scatter(Rf[m, 0], Rf[m, 3], s=30 if rare else 1, c=c,
                   alpha=0.9 if rare else 0.3, linewidths=0,
                   marker="D" if rare else "o", zorder=3 if rare else 1,
                   label=f"{lab} ({int(m.sum())})")
    ax.set_title("final set by provenance", fontsize=11, color=_INK)
    ax.set_xlabel("rank C11", fontsize=10, color=_INK2)
    ax.set_ylabel("rank C66", fontsize=10, color=_INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK2, markerscale=1)

    fig.suptitle(f"Dataset {final}: {prev or 'reference'} -> {final}, "
                 "TRI6 @ 2 elem/pixel", fontsize=14, color=_INK, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=130, facecolor=_SURFACE)
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
