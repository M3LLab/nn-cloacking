"""Plot histograms of C11, C12, C66 and vol for an existing stiffness HDF5 file.

Reads the file written by ``dataset.cellular_chiral.bulk_stiffness``,
prints summary statistics, and saves a 4-panel histogram figure.
Non-positive values are excluded from each panel (logged as skipped count).

Usage
-----

    python -m dataset.cellular_chiral.dataset_stats \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/stats.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


def _stats(name: str, x: np.ndarray, unit: str = "") -> str:
    pcts = np.percentile(x, [1, 5, 25, 50, 75, 95, 99])
    return (
        f"{name:>8} ({unit:<7}) "
        f"n={x.size:7d}  "
        f"mean={x.mean():.4e}  std={x.std():.4e}  "
        f"min={x.min():.4e}  max={x.max():.4e}\n"
        f"           percentiles  1%={pcts[0]:.4e}  5%={pcts[1]:.4e}  "
        f"25%={pcts[2]:.4e}  50%={pcts[3]:.4e}  "
        f"75%={pcts[4]:.4e}  95%={pcts[5]:.4e}  99%={pcts[6]:.4e}"
    )


def _plot_hist(
    ax,
    x: np.ndarray,
    label: str,
    unit: str,
    log_x: bool,
    bins: int,
    nonzero_only: bool = True,
) -> None:
    if nonzero_only:
        skipped = (x <= 0).sum()
        x = x[x > 0]
    else:
        skipped = 0

    if log_x:
        edges = np.logspace(np.log10(x.min()), np.log10(x.max()), bins + 1)
        ax.hist(x, bins=edges, color="steelblue", edgecolor="white", linewidth=0.6)
        ax.set_xscale("log")
    else:
        ax.hist(x, bins=bins, color="steelblue", edgecolor="white", linewidth=0.6)

    ax.axvline(np.median(x), color="crimson", linestyle="--", linewidth=1.2,
               label=f"median = {np.median(x):.3e}")
    ax.axvline(x.mean(), color="darkorange", linestyle=":", linewidth=1.2,
               label=f"mean = {x.mean():.3e}")

    title = f"{label}  (n={x.size}"
    if skipped:
        title += f", {skipped} ≤0 excluded"
    title += ")"
    ax.set_xlabel(f"{label}  [{unit}]" if unit else label, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc="best")


def _density_scatter(ax, x: np.ndarray, y: np.ndarray, xlabel: str, ylabel: str, subsample: int = 5000) -> None:
    """Scatter plot coloured by local point density via KDE."""
    rng = np.random.default_rng(0)
    if len(x) > subsample:
        idx = rng.choice(len(x), subsample, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y

    kde = gaussian_kde(np.vstack([xs, ys]))
    density = kde(np.vstack([xs, ys]))
    order = np.argsort(density)

    sc = ax.scatter(xs[order], ys[order], c=density[order], s=4, cmap="viridis", linewidths=0)
    plt.colorbar(sc, ax=ax, label="Density")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def _hs_upper(vf: np.ndarray, kappa_s: float, mu_s: float) -> tuple[np.ndarray, np.ndarray]:
    """2D Hashin-Shtrikman upper bounds for a porous solid (void phase → 0 moduli).

    Plane-strain formulation; uses the in-plane 2D bulk modulus κ = λ+μ.

    Returns (K_HS+, G_HS+) as arrays matching vf.
    """
    K_hs = kappa_s * mu_s * vf / (kappa_s * (1 - vf) + mu_s)
    G_hs = mu_s - (1 - vf) * 2 * mu_s * (kappa_s + mu_s) / (
        kappa_s * (2 - vf) + 2 * mu_s * (1 - vf)
    )
    return K_hs, G_hs


def plot_derived_properties(C11: np.ndarray, C12: np.ndarray, C66: np.ndarray, vol: np.ndarray, out_path: Path) -> None:
    """Scatter plots of bulk modulus, shear modulus, and Poisson's ratio."""
    K  = (C11 + C12) / 2          # bulk modulus (2D)
    G  = C66                       # shear modulus
    nu = C12 / C11                 # Poisson's ratio (apparent, plane-strain: ν/(1-ν) for solid)

    # Solid-phase moduli (plane-strain, cement: E=30 GPa, ν=0.2)
    E_s, nu_s = 30e9, 0.2
    mu_s    = E_s / (2 * (1 + nu_s))
    kappa_s = E_s / (2 * (1 + nu_s) * (1 - 2 * nu_s))

    vf_line = np.linspace(0, 1, 500)
    K_hs, G_hs = _hs_upper(vf_line, kappa_s, mu_s)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    _density_scatter(axes[0], vol, K  / 1e9, "Volume fraction", "Bulk modulus $K$ [GPa]")
    _density_scatter(axes[1], vol, G  / 1e9, "Volume fraction", "Shear modulus $G$ [GPa]")
    _density_scatter(axes[2], vol, nu,        "Volume fraction", r"Poisson's ratio $\nu$")

    axes[0].plot(vf_line, K_hs / 1e9, "r-", lw=1.5, label="HS upper bound")
    axes[1].plot(vf_line, G_hs / 1e9, "r-", lw=1.5, label="HS upper bound")

    for ax in axes[:2]:
        ax.legend(fontsize=8)

    fig.suptitle("Derived mechanical properties — dataset coverage", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-i", "--input", type=Path,
                        default=Path("output/ca_bulk_squared/stiffness.h5"))
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--log", action="store_true",
                        help="Log-scale x-axis for C11, C12, C66.")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    out_path = args.output or args.input.parent / "stats.png"

    with h5py.File(args.input, "r") as f:
        C11 = f["C11"][:]
        C12 = f["C12"][:]
        C66 = f["C66"][:]
        vol = f["vol"][:]

    print(f"Loaded {C11.size} samples from {args.input}\n")
    print("Statistics")
    print("-" * 90)
    for name, arr, unit in [("C11", C11, "Pa"),
                              ("C12", C12, "Pa"),
                              ("C66", C66, "Pa"),
                              ("vol", vol, "-")]:
        print(_stats(name, arr, unit))
    print("-" * 90)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _plot_hist(axes[0], C11, r"$C_{11}$", "Pa",  log_x=args.log, bins=args.bins, nonzero_only=False)
    _plot_hist(axes[1], C12, r"$C_{12}$", "Pa",  log_x=args.log, bins=args.bins, nonzero_only=False)
    _plot_hist(axes[2], C66, r"$C_{66}$", "Pa",  log_x=args.log, bins=args.bins, nonzero_only=False)
    _plot_hist(axes[3], vol, r"vol-frac",  "",   log_x=False,     bins=args.bins, nonzero_only=False)

    fig.suptitle(f"Stiffness parameter distributions — {args.input.name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved figure to {out_path}")

    derived_path = out_path.parent / (out_path.stem + "_derived.png")
    plot_derived_properties(C11, C12, C66, vol, derived_path)


if __name__ == "__main__":
    main()
