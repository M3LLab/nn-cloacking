#!/usr/bin/env python3
"""Debug reference dispersion vs analytic Rayleigh.

Runs the **reference** (no cloak) sweep for several configurations and
overlays the fundamental Rayleigh branch against the analytic line
f* = k_norm.  The three debug axes are:

  1. Mesh convergence:  h_elem = {0.08, 0.04, 0.02}  (baseline mass)
  2. Mass lumping:      best h from (1), with --lumped-mass
  3. Taller cell:       best h from (1), with --H-factor 2.0

Outputs:
  output/dispersion/debug_convergence.png   — three reference bands
  output/dispersion/debug_lumped.png        — consistent vs lumped
  output/dispersion/debug_Hfactor.png       — H=1x vs H=2x (low-f IPR focus)

The .npz caches produced by each sub-run are reused by the main
`dispersion_ideal.py` script via its filename-based cache.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.dispersion.dispersion_ideal import derive_params, run_sweep


# ── helpers ──────────────────────────────────────────────────────────────


def extract_surface_branch(ks, fs, iprs, k_edge=0.25, n_bins=40,
                           ipr_thr=2.0, f_max=2.5):
    """Per-k-bin, pick the highest-IPR surface mode among the lowest
    folded-Rayleigh window. Returns branch k, f, IPR arrays (with NaNs
    where no surface mode exists)."""
    edges = np.linspace(ks.min(), k_edge, n_bins + 1)
    out_k, out_f, out_ipr = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ks >= lo) & (ks < hi) & (fs <= f_max) & (iprs >= ipr_thr)
        if not m.any():
            continue
        idx = np.where(m)[0]
        # Among surface modes, the true fundamental Rayleigh is the lowest f
        j = idx[np.argmin(fs[idx])]
        out_k.append(0.5 * (lo + hi))
        out_f.append(fs[j])
        out_ipr.append(iprs[j])
    return np.array(out_k), np.array(out_f), np.array(out_ipr)


def plot_branches(results, out_path, title, f_max=2.5, k_edge=0.25,
                  ipr_thr=2.0):
    """results: list of (label, (ks, fs, iprs))."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    ax_scat, ax_band, ax_ipr = axes

    markers = ["o", "s", "^", "D", "v"]
    colors  = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

    # Analytic folded Rayleigh (f*=ξ mirrored around BZ edge, repeat upwards)
    k_line = np.linspace(0, k_edge, 300)
    for m in range(0, 6):
        up   = 2 * m * k_edge + k_line
        down = 2 * (m + 1) * k_edge - k_line
        if up[0] > f_max:
            break
        for br, lbl in ((up, "Analytic" if m == 0 else None),
                        (down, None)):
            mask = br <= f_max
            if mask.any():
                for ax in (ax_scat, ax_band):
                    ax.plot(k_line[mask], br[mask], "k--", lw=0.9, alpha=0.55,
                            zorder=2, label=lbl)

    for (label, data), mk, col in zip(results, markers, colors):
        ks, fs, iprs = data
        # Panel 1: full scatter of surface modes (IPR ≥ threshold)
        m_surf = (fs <= f_max) & (iprs >= ipr_thr)
        ax_scat.scatter(ks[m_surf], fs[m_surf], s=10, marker=mk, c=col,
                        alpha=0.55, edgecolors="none", label=label, zorder=3)

        # Panel 2: extracted fundamental surface branch (f vs k)
        kb, fb, ib = extract_surface_branch(ks, fs, iprs, k_edge=k_edge,
                                            ipr_thr=ipr_thr, f_max=f_max)
        ax_band.plot(kb, fb, marker=mk, c=col, lw=1.3, ms=5,
                     alpha=0.9, label=label, zorder=4)

        # Panel 3: IPR along branch
        ax_ipr.plot(kb, ib, marker=mk, c=col, lw=1.3, ms=5, label=label)

    for ax, ttl in ((ax_scat, "All surface modes (IPR≥thr)"),
                    (ax_band, "Fundamental Rayleigh branch")):
        ax.set_xlim(0, k_edge + 0.005)
        ax.set_ylim(0, f_max)
        ax.set_xlabel(r"$\xi = k\lambda^*/(2\pi)$")
        ax.set_ylabel(r"$f^* = f\lambda^*/c_R$")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    ax_ipr.set_xlim(0, k_edge + 0.005)
    ax_ipr.set_xlabel(r"$\xi$")
    ax_ipr.set_ylabel("IPR along fundamental branch")
    ax_ipr.axhline(3.4, color="k", lw=0.6, ls=":", alpha=0.6)
    ax_ipr.set_title("IPR (3.4 dotted)")
    ax_ipr.grid(alpha=0.3)
    ax_ipr.legend(fontsize=8, loc="lower right")

    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")


# ── driver ───────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-kpts", type=int, default=30)
    ap.add_argument("--n-eigs", type=int, default=120)
    ap.add_argument("--f-max",  type=float, default=2.5)
    ap.add_argument("--out-dir", default="output/dispersion")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-step", type=int, action="append", default=[],
                    help="Skip step index (1, 2, 3). Repeatable.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip = set(args.skip_step)

    L_c = 2.0
    k_vals = np.linspace(np.pi / (100 * L_c), np.pi / L_c, args.n_kpts)

    # ── Step 1: mesh convergence ────────────────────────────────────────
    p_base = derive_params(H_factor=1.0)
    h_list = [0.08, 0.04, 0.02]
    h_fine = 0.03

    step1_results = []
    if 1 not in skip:
        print("\n======== STEP 1: mesh convergence (reference, consistent M) ========")
        for h in h_list:
            hf = min(h_fine, h)   # don't let h_fine exceed h_elem
            print(f"\n--- h_elem={h}, h_fine={hf} ---")
            t0 = time.time()
            data = run_sweep(
                "reference", p_base, k_vals,
                n_eigs=args.n_eigs, h_elem=h, h_fine=hf,
                out_dir=out_dir, force=args.force,
                lumped=False, H_factor=1.0,
            )
            print(f"  total {time.time()-t0:.1f}s")
            step1_results.append((f"h={h}", data))

        plot_branches(
            step1_results,
            out_dir / "debug_convergence.png",
            title="Step 1: reference mesh convergence (consistent mass)",
            f_max=args.f_max,
        )

    # Pick best h for subsequent steps — smallest converged
    best_h = 0.04    # sensible default, override after looking at step 1

    # ── Step 2: lumped mass ─────────────────────────────────────────────
    if 2 not in skip:
        print(f"\n======== STEP 2: mass lumping (h={best_h}) ========")
        hf = min(h_fine, best_h)
        data_consistent = run_sweep(
            "reference", p_base, k_vals,
            n_eigs=args.n_eigs, h_elem=best_h, h_fine=hf,
            out_dir=out_dir, force=args.force,
            lumped=False, H_factor=1.0,
        )
        data_lumped = run_sweep(
            "reference", p_base, k_vals,
            n_eigs=args.n_eigs, h_elem=best_h, h_fine=hf,
            out_dir=out_dir, force=args.force,
            lumped=True, H_factor=1.0,
        )
        plot_branches(
            [(f"consistent h={best_h}", data_consistent),
             (f"lumped h={best_h}",     data_lumped)],
            out_dir / "debug_lumped.png",
            title=f"Step 2: consistent vs lumped mass (h={best_h})",
            f_max=args.f_max,
        )

    # ── Step 4: H-factor sweep at h=0.08 ────────────────────────────────
    if 4 not in skip:
        print("\n======== STEP 4: H_factor sweep {1.0, 1.5, 2.0, 3.0} @ h=0.08 ========")
        h_sweep  = 0.08
        hf_sweep = 0.03
        sweep_results = []
        for Hf in (1.0, 1.5, 2.0, 3.0):
            p_i = derive_params(H_factor=Hf)
            data = run_sweep(
                "reference", p_i, k_vals,
                n_eigs=args.n_eigs, h_elem=h_sweep, h_fine=hf_sweep,
                out_dir=out_dir, force=args.force,
                lumped=False, H_factor=Hf,
            )
            sweep_results.append((f"H={Hf}", data))
        plot_branches(
            sweep_results,
            out_dir / "debug_Hsweep.png",
            title=f"Step 4: H_factor sweep (h={h_sweep})",
            f_max=args.f_max,
        )

    # ── Step 3: taller unit cell ────────────────────────────────────────
    if 3 not in skip:
        print(f"\n======== STEP 3: H_factor=2.0 (h={best_h}, consistent) ========")
        hf = min(h_fine, best_h)
        data_H1 = run_sweep(
            "reference", p_base, k_vals,
            n_eigs=args.n_eigs, h_elem=best_h, h_fine=hf,
            out_dir=out_dir, force=args.force,
            lumped=False, H_factor=1.0,
        )
        p_tall = derive_params(H_factor=2.0)
        data_H2 = run_sweep(
            "reference", p_tall, k_vals,
            n_eigs=args.n_eigs, h_elem=best_h, h_fine=hf,
            out_dir=out_dir, force=args.force,
            lumped=False, H_factor=2.0,
        )
        plot_branches(
            [(f"H=1.0 h={best_h}", data_H1),
             (f"H=2.0 h={best_h}", data_H2)],
            out_dir / "debug_Hfactor.png",
            title=f"Step 3: unit-cell height H_factor=1 vs 2 (h={best_h})",
            f_max=args.f_max,
        )


if __name__ == "__main__":
    main()
