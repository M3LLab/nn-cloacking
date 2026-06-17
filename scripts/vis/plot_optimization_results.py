"""Process and plot results from a cell-based optimization run.

Reads loss_history.csv (produced by run_optimize.py) or parses a stdout
log file, then plots loss curves and prints before/after comparison.

Usage:
    python plot_optimization_results.py                          # default output dir
    python plot_optimization_results.py output/cell_based        # explicit dir
    python plot_optimization_results.py --log run.log            # parse stdout log
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_from_csv(csv_path: Path) -> dict[str, np.ndarray]:
    """Load loss history from the CSV written by run_optimize.py."""
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return {
        "step": data[:, 0].astype(int),
        "total": data[:, 1],
        "cloak": data[:, 2],
        "l2_reg": data[:, 3],
        "neighbor": data[:, 4],
    }


def load_from_log(log_path: Path) -> dict[str, np.ndarray]:
    """Parse optimization stdout log lines into loss arrays.

    Expected format per line:
      Step  NNN | total = X.XXe-XX  cloak = X.XXe-XX  L2_reg = X.XXe-XX  neighbor = X.XXe-XX
    """
    pattern = re.compile(
        r"Step\s+(\d+)\s*\|\s*total\s*=\s*([0-9.eE+\-]+)\s+"
        r"cloak\s*=\s*([0-9.eE+\-]+)\s+"
        r"L2_reg\s*=\s*([0-9.eE+\-]+)\s+"
        r"neighbor\s*=\s*([0-9.eE+\-]+)"
    )
    steps, totals, cloaks, l2s, neighbors = [], [], [], [], []
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                totals.append(float(m.group(2)))
                cloaks.append(float(m.group(3)))
                l2s.append(float(m.group(4)))
                neighbors.append(float(m.group(5)))

    if not steps:
        raise ValueError(f"No optimization log lines found in {log_path}")

    return {
        "step": np.array(steps),
        "total": np.array(totals),
        "cloak": np.array(cloaks),
        "l2_reg": np.array(l2s),
        "neighbor": np.array(neighbors),
    }


def print_comparison(d: dict[str, np.ndarray]) -> None:
    """Print before/after loss comparison."""
    n = len(d["step"])
    print(f"\nOptimization: {n} iterations recorded")
    print(f"{'':20s} {'Initial':>14s} {'Final':>14s} {'Change':>10s}")
    print("-" * 62)
    for key in ("total", "cloak", "l2_reg", "neighbor"):
        v0, vf = d[key][0], d[key][-1]
        if v0 > 0:
            pct = 100 * (vf - v0) / v0
            print(f"  {key:18s} {v0:14.6e} {vf:14.6e} {pct:+9.1f}%")
        else:
            print(f"  {key:18s} {v0:14.6e} {vf:14.6e}       N/A")
    print()


def plot_losses(d: dict[str, np.ndarray], save_path: str) -> None:
    """Plot total and component losses vs iteration."""
    steps = d["step"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

    # Top: total + cloaking loss
    ax1.semilogy(steps, d["total"], "k-", lw=1.5, label="total")
    ax1.semilogy(steps, d["cloak"], "C0-", lw=1, label="cloak")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Cell-based optimization loss")

    # Bottom: regularization terms
    ax2.semilogy(steps, d["l2_reg"], "C1-", lw=1, label="L2 reg")
    ax2.semilogy(steps, d["neighbor"], "C2-", lw=1, label="neighbor")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Regularisation")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot optimization loss curves")
    parser.add_argument("output_dir", nargs="?", default="output/cell_based",
                        help="Directory containing loss_history.csv")
    parser.add_argument("--log", type=str, default=None,
                        help="Path to stdout log file (alternative to CSV)")
    parser.add_argument("--save", type=str, default=None,
                        help="Output plot path (default: <output_dir>/loss_comparison.png)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    if args.log:
        d = load_from_log(Path(args.log))
    else:
        csv_path = out / "loss_history.csv"
        if not csv_path.exists():
            print(f"Error: {csv_path} not found. Either:")
            print(f"  - Wait for run_optimize.py to finish (writes CSV at the end)")
            print(f"  - Pipe output to a log file and use: {sys.argv[0]} --log run.log")
            sys.exit(1)
        d = load_from_csv(csv_path)

    print_comparison(d)

    save_path = args.save or str(out / "loss_comparison.png")
    plot_losses(d, save_path)


if __name__ == "__main__":
    main()
