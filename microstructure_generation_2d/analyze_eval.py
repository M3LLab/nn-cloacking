"""Analyze and visualize eval_nn_comparison.py results.

Compares diffusion-generated samples vs nearest-neighbor retrieval from the
training set, relative to the validation target stiffness.

Usage
-----
    python -m microstructure_generation_2d.analyze_eval \\
        --csv output/diffusion_ca_squared/v2/eval_nn/comparison.csv \\
        --output-dir output/diffusion_ca_squared/v2/eval_nn/analysis

    # overlay multiple runs
    python -m microstructure_generation_2d.analyze_eval \\
        --csv output/diffusion_ca_squared/v2/eval_nn/comparison.csv \\
               output/diffusion_ca_squared/v3/eval_nn/comparison.csv \\
        --labels v2 v3 \\
        --output-dir output/analysis_comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def rel_err(pred: pd.Series, target: pd.Series) -> pd.Series:
    denom = target.abs().clip(lower=1.0)
    return (pred - target).abs() / denom


def _summary(s: pd.Series, label: str) -> dict:
    return {
        "label": label,
        "mean":   s.mean(),
        "median": s.median(),
        "p25":    s.quantile(0.25),
        "p75":    s.quantile(0.75),
        "p90":    s.quantile(0.90),
        "p95":    s.quantile(0.95),
    }


def compute_errors(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with per-sample relative errors for all properties."""
    out = pd.DataFrame()
    out["comparison_id"] = df["comparison_id"]
    for prop in ("C11", "C12", "C66"):
        target = df[prop]
        out[f"{prop}_generated_relerr"] = rel_err(df[f"{prop}_generated"], target)
        out[f"{prop}_nearest_relerr"]   = rel_err(df[f"{prop}_dataset"],   target)
    out["vol_generated_abserr"] = (df["vol_generated"] - df["vol"]).abs()
    out["vol_nearest_abserr"]   = (df["vol_dataset"]   - df["vol"]).abs()
    return out


def combined_relerr(errs: pd.DataFrame, source: str) -> pd.Series:
    """Mean relative error across C11, C12, C66 for a given source."""
    cols = [f"{p}_{source}_relerr" for p in ("C11", "C12", "C66")]
    return errs[cols].mean(axis=1)


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_stats_table(all_stats: list[list[dict]]) -> None:
    """Print a table of statistics.

    all_stats: list of groups; each group is a list of _summary() dicts.
    """
    header = f"{'Metric':<28}  {'mean':>8}  {'median':>8}  {'p25':>8}  {'p75':>8}  {'p90':>8}  {'p95':>8}"
    print(header)
    print("-" * len(header))
    for group in all_stats:
        for row in group:
            print(
                f"{row['label']:<28}  "
                f"{row['mean']:>8.3%}  "
                f"{row['median']:>8.3%}  "
                f"{row['p25']:>8.3%}  "
                f"{row['p75']:>8.3%}  "
                f"{row['p90']:>8.3%}  "
                f"{row['p95']:>8.3%}"
            )
        print()


def print_vol_stats_table(all_stats: list[list[dict]]) -> None:
    header = f"{'Metric':<28}  {'mean':>8}  {'median':>8}  {'p25':>8}  {'p75':>8}  {'p90':>8}  {'p95':>8}"
    print(header)
    print("-" * len(header))
    for group in all_stats:
        for row in group:
            print(
                f"{row['label']:<28}  "
                f"{row['mean']:>8.4f}  "
                f"{row['median']:>8.4f}  "
                f"{row['p25']:>8.4f}  "
                f"{row['p75']:>8.4f}  "
                f"{row['p90']:>8.4f}  "
                f"{row['p95']:>8.4f}"
            )
        print()


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

COLORS = {"generated": "#2196F3", "nearest": "#FF9800"}
PROPS = ["C11", "C12", "C66"]
PROP_LABELS = {"C11": "$C_{11}$", "C12": "$C_{12}$", "C66": "$C_{66}$"}


def plot_error_distributions(
    error_dfs: list[pd.DataFrame],
    run_labels: list[str],
    out_path: Path,
) -> None:
    """CDF and histogram of relative errors for each stiffness component."""
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    max_x = 1.0  # clip display at 100 % relative error

    for col, prop in enumerate(PROPS):
        ax_cdf  = fig.add_subplot(gs[0, col])
        ax_hist = fig.add_subplot(gs[1, col])

        for run_idx, (errs, rlabel) in enumerate(zip(error_dfs, run_labels)):
            lsgen = "-"  if len(error_dfs) == 1 else ["-", "--", ":"][run_idx % 3]
            lsnn  = "--" if len(error_dfs) == 1 else lsgen
            alpha = 0.85 if len(error_dfs) == 1 else 0.7

            for source, ls in (("generated", lsgen), ("nearest", lsnn)):
                col_name = f"{prop}_{source}_relerr"
                vals = errs[col_name].clip(upper=max_x).values
                label = f"{source}" if len(error_dfs) == 1 else f"{rlabel} {source}"
                color = COLORS[source]

                # CDF
                sorted_v = np.sort(vals)
                cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
                ax_cdf.plot(sorted_v, cdf, color=color, ls=ls, lw=1.5,
                            alpha=alpha, label=label)

                # Histogram
                ax_hist.hist(vals, bins=50, range=(0, max_x),
                             color=color, alpha=0.35, density=True,
                             label=label, histtype="stepfilled")
                ax_hist.hist(vals, bins=50, range=(0, max_x),
                             color=color, alpha=0.8, density=True,
                             histtype="step", lw=1.2)

        ax_cdf.set_title(f"{PROP_LABELS[prop]} — CDF of rel. error", fontsize=9)
        ax_cdf.set_xlabel("Relative error", fontsize=8)
        ax_cdf.set_ylabel("Cumulative fraction", fontsize=8)
        ax_cdf.legend(fontsize=7)
        ax_cdf.grid(True, lw=0.4, alpha=0.5)
        ax_cdf.set_xlim(0, max_x)

        ax_hist.set_title(f"{PROP_LABELS[prop]} — distribution of rel. error", fontsize=9)
        ax_hist.set_xlabel("Relative error", fontsize=8)
        ax_hist.set_ylabel("Density", fontsize=8)
        ax_hist.legend(fontsize=7)
        ax_hist.grid(True, lw=0.4, alpha=0.5)
        ax_hist.set_xlim(0, max_x)

    fig.suptitle("Stiffness relative-error: diffusion generated vs nearest-neighbor retrieval", fontsize=10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_vol_distribution(
    error_dfs: list[pd.DataFrame],
    run_labels: list[str],
    out_path: Path,
) -> None:
    """CDF of absolute volume fraction error."""
    fig, (ax_cdf, ax_hist) = plt.subplots(1, 2, figsize=(10, 4))

    for run_idx, (errs, rlabel) in enumerate(zip(error_dfs, run_labels)):
        ls = ["-", "--", ":"][run_idx % 3]
        for source in ("generated", "nearest"):
            col_name = f"vol_{source}_abserr"
            vals = errs[col_name].values
            label = source if len(error_dfs) == 1 else f"{rlabel} {source}"
            color = COLORS[source]

            sorted_v = np.sort(vals)
            cdf = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
            ax_cdf.plot(sorted_v, cdf, color=color, ls=ls, lw=1.5, label=label)
            ax_hist.hist(vals, bins=50, color=color, alpha=0.35, density=True,
                         histtype="stepfilled", label=label)
            ax_hist.hist(vals, bins=50, color=color, alpha=0.8, density=True,
                         histtype="step", lw=1.2)

    for ax in (ax_cdf, ax_hist):
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
        ax.set_xlabel("Absolute vol-frac error", fontsize=9)
    ax_cdf.set_ylabel("Cumulative fraction", fontsize=9)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_cdf.set_title("Volume fraction error — CDF", fontsize=9)
    ax_hist.set_title("Volume fraction error — distribution", fontsize=9)
    fig.suptitle("Volume fraction: diffusion generated vs nearest-neighbor retrieval", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_scatter_generated_vs_nearest(
    df: pd.DataFrame,
    errs: pd.DataFrame,
    out_path: Path,
) -> None:
    """Scatter: per-sample combined rel-error for generated vs nearest-neighbor."""
    gen_err = combined_relerr(errs, "generated")
    near_err = combined_relerr(errs, "nearest")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(near_err, gen_err, s=6, alpha=0.3, color="#555")
    lim = max(near_err.quantile(0.98), gen_err.quantile(0.98)) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1, label="equal error")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Nearest-neighbor mean rel. error (C11/C12/C66)", fontsize=9)
    ax.set_ylabel("Generated mean rel. error (C11/C12/C66)", fontsize=9)
    ax.set_title("Per-sample: generated vs nearest-neighbor accuracy\n(below diagonal = generated wins)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.4, alpha=0.5)

    frac_gen_wins = (gen_err < near_err).mean()
    ax.text(0.05, 0.95, f"Generated better: {frac_gen_wins:.1%}",
            transform=ax.transAxes, fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_generated_histograms(df: pd.DataFrame, out_path: Path) -> None:
    """Histograms of generated C11, C12, C66 overlaid with the target distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, prop in zip(axes, PROPS):
        ax.hist(df[prop],                bins=60, alpha=0.5, color="#444",              density=True, label="Target (val)")
        ax.hist(df[f"{prop}_generated"], bins=60, alpha=0.6, color=COLORS["generated"], density=True, label="Generated")
        ax.set_xlabel(PROP_LABELS[prop], fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.set_title(PROP_LABELS[prop], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, lw=0.4, alpha=0.5)
    fig.suptitle("Distribution of generated vs target stiffness values", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_stiffness_vs_vol(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of each C parameter vs volume fraction for target, generated, and nearest-neighbor."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax, prop in zip(axes, PROPS):
        ax.scatter(df["vol"],          df[prop],                s=4, alpha=0.25, color="#444",  label="Target (val)")
        ax.scatter(df["vol_generated"], df[f"{prop}_generated"], s=4, alpha=0.25, color=COLORS["generated"], label="Generated")
        ax.scatter(df["vol_dataset"],   df[f"{prop}_dataset"],   s=4, alpha=0.25, color=COLORS["nearest"],   label="Nearest-neighbor")
        ax.set_xlabel("Volume fraction", fontsize=9)
        ax.set_ylabel(PROP_LABELS[prop], fontsize=9)
        ax.set_title(f"{PROP_LABELS[prop]} vs vol-frac", fontsize=9)
        ax.grid(True, lw=0.4, alpha=0.5)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.suptitle("Stiffness vs volume fraction: target, generated, nearest-neighbor",
                 fontsize=10, y=1.06)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_target_vs_predicted(df: pd.DataFrame, out_path: Path) -> None:
    """Scatter of target vs generated/nearest for each stiffness property."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for col, prop in enumerate(PROPS):
        target = df[prop]
        for row, (source, label) in enumerate([("generated", "Generated"), ("dataset", "Nearest-neighbor")]):
            ax = axes[row, col]
            pred = df[f"{prop}_{source}"]
            lim_lo = min(target.min(), pred.min())
            lim_hi = max(target.quantile(0.99), pred.quantile(0.99))
            ax.scatter(target, pred, s=4, alpha=0.2, color=COLORS[source if source != "dataset" else "nearest"])
            ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "r--", lw=1)
            ax.set_xlabel(f"Target {PROP_LABELS[prop]}", fontsize=8)
            ax.set_ylabel(f"{label} {PROP_LABELS[prop]}", fontsize=8)
            ax.set_title(f"{label} — {PROP_LABELS[prop]}", fontsize=8)
            ax.set_xlim(lim_lo, lim_hi)
            ax.set_ylim(lim_lo, lim_hi)
            ax.grid(True, lw=0.4, alpha=0.5)
            relerr = rel_err(pred, target).mean()
            ax.text(0.05, 0.95, f"mean rel err: {relerr:.2%}",
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8))

    fig.suptitle("Target vs predicted stiffness: generated and nearest-neighbor", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", nargs="+", required=True,
                        help="Path(s) to comparison.csv from eval_nn_comparison.py")
    parser.add_argument("--labels", nargs="*", default=None,
                        help="Labels for each CSV (defaults to v0, v1, ...)")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    csv_paths = [Path(p) for p in args.csv]
    labels = args.labels if args.labels else [f"v{i}" for i in range(len(csv_paths))]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs   = [pd.read_csv(p) for p in csv_paths]
    errss = [compute_errors(df) for df in dfs]

    # ---- print statistics ----
    for df, errs, label in zip(dfs, errss, labels):
        n = len(df)
        print(f"\n{'='*60}")
        print(f"Run: {label}  ({n} samples)")
        print(f"{'='*60}")

        print("\n--- Stiffness relative error ---")
        stats_groups = []
        for prop in PROPS:
            group = [
                _summary(errs[f"{prop}_generated_relerr"], f"{prop:>3} generated"),
                _summary(errs[f"{prop}_nearest_relerr"],   f"{prop:>3} nearest-neighbor"),
            ]
            stats_groups.append(group)
        print_stats_table(stats_groups)

        print("--- Combined (mean over C11/C12/C66) relative error ---")
        combined_stats = [
            [
                _summary(combined_relerr(errs, "generated"), "generated"),
                _summary(combined_relerr(errs, "nearest"),   "nearest-neighbor"),
            ]
        ]
        print_stats_table(combined_stats)

        print("--- Volume fraction absolute error ---")
        vol_stats = [
            [
                _summary(errs["vol_generated_abserr"], "generated"),
                _summary(errs["vol_nearest_abserr"],   "nearest-neighbor"),
            ]
        ]
        print_vol_stats_table(vol_stats)

        gen_wins = (combined_relerr(errs, "generated") < combined_relerr(errs, "nearest")).mean()
        print(f"\nSamples where generated outperforms nearest-neighbor: {gen_wins:.1%}")

    # ---- plots ----
    # error distributions per property
    plot_error_distributions(
        errss, labels,
        out_dir / "error_distributions.png",
    )

    # volume fraction errors
    plot_vol_distribution(
        errss, labels,
        out_dir / "vol_error_distribution.png",
    )

    # scatter: generated vs nearest, per sample
    if len(dfs) == 1:
        plot_scatter_generated_vs_nearest(
            dfs[0], errss[0],
            out_dir / "scatter_generated_vs_nearest.png",
        )

        # target vs predicted scatter
        plot_target_vs_predicted(
            dfs[0],
            out_dir / "target_vs_predicted.png",
        )

        # generated C value histograms
        plot_generated_histograms(
            dfs[0],
            out_dir / "generated_histograms.png",
        )

        # stiffness vs volume fraction
        plot_stiffness_vs_vol(
            dfs[0],
            out_dir / "stiffness_vs_vol.png",
        )


if __name__ == "__main__":
    main()
