"""Visualize frequency sweep CSVs with optimization target band highlighted.

Usage::

    python scripts/vis/plot_frequency_sweep.py output/latent_ae_optimize_demo3/ \\
        --f-min 1.5 --f-max 2.5

    # Custom step for band shading ticks:
    python scripts/vis/plot_frequency_sweep.py output/latent_ae_optimize_demo3/ \\
        --f-min 1.5 --f-max 2.5 --f-step 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASE_STYLES = {
    "obstacle":  {"color": "black", "ls": "--", "marker": "s", "label": "Obstacle"},
    "ideal":     {"color": "C3",    "ls": "-",  "marker": "o", "label": "Ideal Cloak"},
    "optimized": {"color": "C0",    "ls": "-",  "marker": "D", "label": "Optimized"},
}


def _load_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return data["f_star"], data["u_ratio"]


def plot_sweep(out_dir: Path, f_min: float | None, f_max: float | None,
               f_step: float | None) -> None:
    csv_names = {
        "obstacle":  "frequency_sweep_obstacle.csv",
        "ideal":     "frequency_sweep_ideal.csv",
        "optimized": "frequency_sweep_optimized.csv",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    f_global_max = 0.0
    y_global_max = 0.0
    band_avgs: list[tuple[str, float, str]] = []  # (case, avg, color)
    for case, fname in csv_names.items():
        csv_path = out_dir / fname
        if not csv_path.exists():
            continue
        f_vals, ratio_vals = _load_csv(csv_path)
        style = CASE_STYLES[case]
        ax.plot(f_vals, ratio_vals,
                color=style["color"], ls=style["ls"], marker=style["marker"],
                lw=1.5, markersize=4, label=style["label"])
        f_global_max = max(f_global_max, f_vals.max())
        y_global_max = max(y_global_max, ratio_vals.max())

        if f_min is not None and f_max is not None:
            mask = (f_vals >= f_min) & (f_vals <= f_max)
            if mask.any():
                band_avgs.append((style["label"], float(ratio_vals[mask].mean()), style["color"]))

    # Optimization target band
    if f_min is not None and f_max is not None:
        ax.axvspan(f_min, f_max, alpha=0.12, color="C0", label=f"Optimization band [{f_min}–{f_max}]")
        ax.axvline(f_min, color="C0", ls=":", lw=1.0, alpha=0.7)
        ax.axvline(f_max, color="C0", ls=":", lw=1.0, alpha=0.7)
        if f_step is not None:
            for f_tick in np.arange(f_min, f_max + f_step * 0.5, f_step):
                ax.axvline(f_tick, color="C0", ls=":", lw=0.6, alpha=0.4)

        # Annotate average values inside the band
        if band_avgs:
            x_mid = (f_min + f_max) / 2
            lines = [r"$\langle\cdot\rangle_{\rm band}$:"] + [
                f"  {label}: {avg:.3f}" for label, avg, _ in band_avgs
            ]
            ax.text(
                x_mid, 0.04, "\n".join(lines),
                transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="C0", alpha=0.8),
            )

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Cloaking performance vs frequency")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, f_global_max + 0.1)
    ax.set_ylim(0, max(y_global_max * 1.1, 1.15))

    plot_path = out_dir / "frequency_sweep_vis.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot frequency sweep with optimization band")
    parser.add_argument("out_dir", help="Output directory containing frequency_sweep_*.csv")
    parser.add_argument("--f-min", type=float, default=None, help="Optimization band lower bound")
    parser.add_argument("--f-max", type=float, default=None, help="Optimization band upper bound")
    parser.add_argument("--f-step", type=float, default=None,
                        help="Optimization frequency step (for tick marks within band)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    plot_sweep(out_dir, args.f_min, args.f_max, args.f_step)


if __name__ == "__main__":
    main()
