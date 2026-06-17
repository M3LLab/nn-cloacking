"""Compare frequency sweeps across different experiments.

Loads frequency sweep CSVs from multiple experiments and plots them together
for visual comparison. The optimized_params.npz location is automatically
inferred from the output_dir specified in each config.

Usage::

    # Compare three experiments with auto-detected paths
    python scripts/compare_frequency_sweeps.py \\
        configs/cauchy_tri_pt2.yaml \\
        configs/cauchy_tri_top.yaml \\
        configs/cauchy_tri_multifreq.yaml

    # With custom labels
    python scripts/compare_frequency_sweeps.py \\
        configs/cauchy_tri_pt2.yaml "Surface Loss" \\
        configs/cauchy_tri_top.yaml "Topology" \\
        configs/cauchy_tri_multifreq.yaml "Multi-freq"

    # Compare only ideal or optimized cases
    python scripts/compare_frequency_sweeps.py --sweep-type ideal \\
        configs/cauchy_tri_pt2.yaml configs/cauchy_tri_top.yaml

    # Specify output directory for comparison plot
    python scripts/compare_frequency_sweeps.py \\
        configs/cauchy_tri_pt2.yaml configs/cauchy_tri_top.yaml \\
        --output /tmp/comparison_output

    # Compare specific sweep types only (can specify multiple)
    python scripts/compare_frequency_sweeps.py \\
        configs/cauchy_tri_pt2.yaml configs/cauchy_tri_top.yaml \\
        --sweep-type ideal optimized
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rayleigh_cloak import load_config


# ── CSV I/O ───────────────────────────────────────────────────────────


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load frequency sweep CSV, return (f_star, u_ratio)."""
    if not csv_path.exists():
        return None, None
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["f_star"], data["u_ratio"]


# ── plotting ──────────────────────────────────────────────────────────


# Styles for different sweep types (obstacle, ideal, optimized)
SWEEP_STYLES = {
    "obstacle": {"color": "black", "ls": "--", "marker": "s", "alpha": 0.7},
    "ideal":    {"color": "C3",    "ls": "-",  "marker": "o", "alpha": 0.7},
    "optimized": {"color": "C0",   "ls": "-",  "marker": "D", "alpha": 0.7},
}

# Color palette for different experiments
EXPERIMENT_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
]


def plot_comparison(
    experiments: list[dict],
    sweep_types: list[str],
    out_dir: Path,
) -> None:
    """Plot frequency sweeps from all experiments.

    Args:
        experiments: List of dicts with keys:
            - label: experiment name
            - config_path: path to config
            - output_dir: output directory from config
        sweep_types: List of sweep types to plot (obstacle, ideal, optimized)
        out_dir: Directory to save comparison plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    f_max = 0.0
    y_max = 0.0
    has_data = False

    for exp_idx, exp in enumerate(experiments):
        exp_label = exp["label"]
        exp_output_dir = Path(exp["output_dir"])
        exp_color = EXPERIMENT_COLORS[exp_idx % len(EXPERIMENT_COLORS)]

        for sweep_type in sweep_types:
            csv_path = exp_output_dir / f"frequency_sweep_{sweep_type}.csv"
            f_vals, ratio_vals = _load_csv(csv_path)

            if f_vals is None:
                continue

            has_data = True
            style = SWEEP_STYLES[sweep_type]

            # Create label combining experiment and sweep type
            if sweep_type == "optimized":
                label_suffix = "(optimized)"
            elif sweep_type == "ideal":
                label_suffix = "(ideal)"
            else:  # obstacle
                label_suffix = "(obstacle)"

            label = f"{exp_label} {label_suffix}"

            # Use experiment color, modify line style per sweep type
            ax.plot(
                f_vals, ratio_vals,
                color=exp_color,
                ls=style["ls"],
                marker=style["marker"],
                lw=1.5,
                markersize=5,
                alpha=style["alpha"],
                label=label,
            )

            f_max = max(f_max, f_vals.max())
            y_max = max(y_max, ratio_vals.max())

    if not has_data:
        print("WARNING: No data found to plot!")
        return

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)", fontsize=11)
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$", fontsize=11)
    ax.set_title("Cloaking Performance Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_max + 0.1)
    ax.set_ylim(0, max(y_max * 1.1, 1.15))

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / "frequency_sweep_comparison.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Comparison plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare frequency sweeps across multiple experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Parse positional args: config_path [label] config_path [label] ...
    parser.add_argument(
        "config_and_labels",
        nargs="+",
        help="Alternating config paths and optional labels. "
             "Example: config1.yaml label1 config2.yaml label2",
    )

    parser.add_argument(
        "--sweep-type",
        nargs="+",
        choices=["obstacle", "ideal", "optimized"],
        default=["ideal", "optimized"],
        help="Which sweep types to include in plot (default: ideal optimized)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory to save comparison plot. "
             "If not specified, uses first experiment's output_dir",
    )

    args = parser.parse_args()

    # Parse config_path and optional label pairs
    experiments = []
    i = 0
    while i < len(args.config_and_labels):
        config_path_str = args.config_and_labels[i]

        # Check if next arg is a label (doesn't look like a path)
        label = None
        if i + 1 < len(args.config_and_labels):
            next_arg = args.config_and_labels[i + 1]
            # If next arg doesn't end with .yaml and doesn't contain /, assume it's a label
            if not next_arg.endswith(".yaml") and "/" not in next_arg:
                label = next_arg
                i += 2
            else:
                i += 1
        else:
            i += 1

        # Load config to get output_dir
        config_path = Path(config_path_str)
        if not config_path.exists():
            print(f"ERROR: Config not found: {config_path}")
            continue

        config = load_config(str(config_path))
        output_dir = Path(config.output_dir)

        # Use config name or provided label
        if label is None:
            label = config_path.stem

        experiments.append({
            "label": label,
            "config_path": str(config_path),
            "output_dir": str(output_dir),
        })

        print(f"Added experiment: '{label}' (config: {config_path}, output: {output_dir})")

    if not experiments:
        parser.error("No valid experiments found")

    # Determine output directory
    if args.output is None:
        out_dir = Path(experiments[0]["output_dir"]) / "comparison"
    else:
        out_dir = args.output

    print(f"\n=== Comparing {len(experiments)} experiments ===")
    print(f"Sweep types: {', '.join(args.sweep_type)}")
    print(f"Output dir: {out_dir}\n")

    plot_comparison(experiments, args.sweep_type, out_dir)

    print("Done!")


if __name__ == "__main__":
    main()
