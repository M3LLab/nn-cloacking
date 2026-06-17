#!/usr/bin/env python3
"""Compare optimized material parameters between neural_reparam and topo experiments.

Usage:
    python scripts/compare_neural_vs_topo.py <exp1_dir> <exp2_dir> [--save <output.png>]

Example:
    python scripts/compare_neural_vs_topo.py \
        output/triangular_optimize_neural_flat2 \
        output/topo3 \
        --save output/comparison_neural_vs_topo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_experiment(exp_dir: str) -> dict:
    """Load optimized params and loss history from an experiment directory."""
    d = np.load(Path(exp_dir) / "optimized_params.npz")
    loss = np.genfromtxt(
        Path(exp_dir) / "loss_history.csv", delimiter=",", names=True,
    )
    return {
        "cell_C_flat": d["cell_C_flat"],
        "cell_rho": d["cell_rho"],
        "loss": loss,
        "name": Path(exp_dir).name,
    }


def analyze_simp_feasibility(
    target_C: np.ndarray,
    target_rho: np.ndarray,
    cloak_mask: np.ndarray,
    lam_cement: float = 8.333e9,
    mu_cement: float = 12.5e9,
    rho_cement: float = 2300.0,
    simp_p: float = 3.0,
) -> dict:
    """Quantify how well SIMP can represent the target material field.

    Returns dict with per-cell optimal densities and residual errors.
    """
    lam_t = target_C[cloak_mask, 0]
    mu_t = target_C[cloak_mask, 1]
    rho_t = target_rho[cloak_mask]

    # Density needed to match each parameter independently
    d_lam = np.clip((lam_t / lam_cement) ** (1.0 / simp_p), 0, 1)
    d_mu = np.clip((mu_t / mu_cement) ** (1.0 / simp_p), 0, 1)
    d_rho = np.clip(rho_t / rho_cement, 0, 1)

    # Best compromise density (least-squares over normalised residuals)
    # Sweep d in [0,1] for each cell and pick the one minimising
    # sum of squared relative errors
    d_grid = np.linspace(0, 1, 1000)[:, None]  # (1000, 1)
    lam_pred = lam_cement * d_grid ** simp_p    # (1000, n_cloak)
    mu_pred = mu_cement * d_grid ** simp_p
    rho_pred = rho_cement * d_grid

    err = (
        ((lam_pred - lam_t) / (lam_t + 1e-30)) ** 2
        + ((mu_pred - mu_t) / (mu_t + 1e-30)) ** 2
        + ((rho_pred - rho_t) / (rho_t + 1e-30)) ** 2
    )
    best_idx = np.argmin(err, axis=0)
    d_best = d_grid[best_idx, 0]
    residual = err[best_idx, np.arange(len(lam_t))]

    return {
        "d_lam": d_lam,
        "d_mu": d_mu,
        "d_rho": d_rho,
        "d_best": d_best,
        "residual": residual,
        "lam_target": lam_t,
        "mu_target": mu_t,
        "rho_target": rho_t,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp1_dir", help="Path to neural_reparam experiment output")
    parser.add_argument("exp2_dir", help="Path to topo experiment output")
    parser.add_argument("--save", default=None, help="Output PNG path")
    parser.add_argument("--lam-cement", type=float, default=8.333e9)
    parser.add_argument("--mu-cement", type=float, default=12.5e9)
    parser.add_argument("--rho-cement", type=float, default=2300.0)
    parser.add_argument("--simp-p", type=float, default=3.0)
    args = parser.parse_args()

    exp1 = load_experiment(args.exp1_dir)
    exp2 = load_experiment(args.exp2_dir)

    # Identify cloak cells in exp1 (non-background)
    bg_rho = 1600.0
    cloak1 = np.abs(exp1["cell_rho"] - bg_rho) > 1.0

    feas = analyze_simp_feasibility(
        exp1["cell_C_flat"], exp1["cell_rho"], cloak1,
        lam_cement=args.lam_cement,
        mu_cement=args.mu_cement,
        rho_cement=args.rho_cement,
        simp_p=args.simp_p,
    )

    # ── Print summary ──
    print("=" * 72)
    print(f"Experiment 1 (neural_reparam): {exp1['name']}")
    print(f"  Cells: {len(exp1['cell_rho'])} ({cloak1.sum()} cloak)")
    print(f"  Final cloak_pct: {np.sqrt(max(exp1['loss']['cloak'][-1], 0)) * 100:.2f}%")
    print(f"Experiment 2 (topo): {exp2['name']}")
    print(f"  Cells: {len(exp2['cell_rho'])}")
    print(f"  Final cloak_pct: {np.sqrt(max(exp2['loss']['cloak'][-1], 0)) * 100:.2f}%")
    print()
    print("SIMP feasibility analysis (can SIMP represent exp1's solution?):")
    print(f"  d needed for lam: [{feas['d_lam'].min():.4f}, {feas['d_lam'].max():.4f}]")
    print(f"  d needed for mu:  [{feas['d_mu'].min():.4f}, {feas['d_mu'].max():.4f}]")
    print(f"  d needed for rho: [{feas['d_rho'].min():.4f}, {feas['d_rho'].max():.4f}]")
    print(f"  Max |d_lam - d_rho|: {np.max(np.abs(feas['d_lam'] - feas['d_rho'])):.4f}")
    print(f"  Max |d_mu  - d_rho|: {np.max(np.abs(feas['d_mu'] - feas['d_rho'])):.4f}")
    print(f"  Best-compromise d: [{feas['d_best'].min():.4f}, {feas['d_best'].max():.4f}]")
    print(f"  Mean relative residual: {feas['residual'].mean():.4f}")
    ratio = feas["lam_target"] / feas["mu_target"]
    print(f"  Required lam/mu ratio: [{ratio.min():.4f}, {ratio.max():.4f}]")
    print(f"  SIMP locked lam/mu:    {args.lam_cement / args.mu_cement:.4f}")
    n_infeasible_rho = np.sum(exp1["cell_rho"][cloak1] > args.rho_cement)
    print(f"  Cells with rho > rho_cement: {n_infeasible_rho}/{cloak1.sum()}")
    print("=" * 72)

    # ── Plot ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: Loss curves
    ax = axes[0, 0]
    cloak_pct1 = np.sqrt(np.maximum(exp1["loss"]["cloak"], 0)) * 100
    cloak_pct2 = np.sqrt(np.maximum(exp2["loss"]["cloak"], 0)) * 100
    ax.semilogy(exp1["loss"]["step"], cloak_pct1, label=exp1["name"])
    ax.semilogy(exp2["loss"]["step"], cloak_pct2, label=exp2["name"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Cloak error %")
    ax.set_title("Cloaking loss convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Row 1: lam/mu ratio histograms
    ax = axes[0, 1]
    ax.hist(ratio, bins=50, alpha=0.7, label="Target (exp1 cloak)")
    ax.axvline(args.lam_cement / args.mu_cement, color="red", ls="--",
               label=f"SIMP locked = {args.lam_cement / args.mu_cement:.3f}")
    ax.axvline(1.0, color="green", ls="--", alpha=0.5, label="Background = 1.0")
    ax.set_xlabel("lam / mu")
    ax.set_ylabel("Count")
    ax.set_title("Required lam/mu ratio in cloak")
    ax.legend(fontsize=8)

    # Row 1: Density requirements scatter
    ax = axes[0, 2]
    ax.scatter(feas["d_lam"], feas["d_rho"], s=3, alpha=0.4, label="d_lam vs d_rho")
    ax.scatter(feas["d_mu"], feas["d_rho"], s=3, alpha=0.4, label="d_mu vs d_rho")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="d_C = d_rho (ideal)")
    ax.set_xlabel("d from stiffness")
    ax.set_ylabel("d from density")
    ax.set_title("SIMP density conflict")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.6)

    # Row 2: Material distributions exp1
    ax = axes[1, 0]
    C1 = exp1["cell_C_flat"]
    ax.hist(C1[cloak1, 0], bins=50, alpha=0.6, label="lam (exp1 cloak)")
    ax.hist(C1[cloak1, 1], bins=50, alpha=0.6, label="mu (exp1 cloak)")
    ax.axvline(1.44e8, color="k", ls="--", alpha=0.5, label="Background")
    ax.set_xlabel("C value (Pa)")
    ax.set_title(f"Exp1 ({exp1['name']}) stiffness")
    ax.legend(fontsize=8)

    # Row 2: Material distributions exp2
    ax = axes[1, 1]
    C2 = exp2["cell_C_flat"]
    # For topo, identify cloak cells as those with rho != rho0
    bg2 = np.abs(exp2["cell_rho"] - bg_rho) < 1.0
    if bg2.sum() == len(exp2["cell_rho"]):
        # All cells are far from 1600, use different criterion
        bg2 = np.zeros(len(exp2["cell_rho"]), dtype=bool)
    cloak2 = ~bg2
    if cloak2.sum() > 0:
        ax.hist(C2[cloak2, 0], bins=50, alpha=0.6, label="lam (exp2 cloak)")
        ax.hist(C2[cloak2, 1], bins=50, alpha=0.6, label="mu (exp2 cloak)")
    ax.set_xlabel("C value (Pa)")
    ax.set_title(f"Exp2 ({exp2['name']}) stiffness")
    ax.legend(fontsize=8)

    # Row 2: Density distribution (exp2 implied)
    ax = axes[1, 2]
    d_implied = exp2["cell_rho"] / args.rho_cement
    d_implied_cloak = d_implied[cloak2] if cloak2.sum() > 0 else d_implied
    ax.hist(d_implied_cloak, bins=50, alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", ls="--", alpha=0.5, label="Void")
    ax.axvline(1, color="red", ls="--", alpha=0.5, label="Solid")
    ax.set_xlabel("Implied SIMP density")
    ax.set_ylabel("Count")
    ax.set_title(f"Exp2 density distribution (binarisation)")
    ax.legend(fontsize=8)

    fig.suptitle("Neural reparam vs Topology SIMP: material parameter comparison", fontsize=14)
    fig.tight_layout()

    save_path = args.save or str(
        Path(args.exp1_dir).parent / "comparison_neural_vs_topo.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figure to {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
