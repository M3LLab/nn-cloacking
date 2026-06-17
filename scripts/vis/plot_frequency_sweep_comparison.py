"""Comparison plot for a frequency sweep of any experiment.

Reads the frequency-sweep CSVs (optimized / matched / pixel-validated) from an
experiment's output directory and draws whichever exist on one axes. Saved next
to the CSVs as ``frequency_sweep_comparison.png``.

Usage
-----

    # point at a config (output dir + training f* are read from it)
    python scripts/vis/plot_frequency_sweep_comparison.py \\
        configs/cell20_phys_flat4_materialreg4.yaml

    # or point straight at an output directory
    python scripts/vis/plot_frequency_sweep_comparison.py \\
        output/cell20_phys_flat4_materialreg4

    # overrides
    python scripts/vis/plot_frequency_sweep_comparison.py <target> \\
        --out-dir output/foo --title "My run" --train-fstar 2.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# csv filename → plot style. Cases whose CSV is absent are skipped silently-ish.
CASES = [
    ("frequency_sweep_obstacle.csv",
     {"color": "black", "marker": "s", "ls": "--", "label": "Obstacle (no cloak)"}),
    ("frequency_sweep_ideal.csv",
     {"color": "C3", "marker": "o", "ls": "-", "label": "Ideal (analytic transform)"}),
    ("frequency_sweep_symmetrized.csv",
     {"color": "C4", "marker": "P", "ls": "-", "label": "Symmetrized-ideal (Cauchy)"}),
    ("frequency_sweep_optimized.csv",
     {"color": "C0", "marker": "D", "ls": "-", "label": "Optimized (continuous homogenised)"}),
    ("frequency_sweep_matched.csv",
     {"color": "C1", "marker": "v", "ls": "-", "label": "Matched (snapped to dataset)"}),
    ("frequency_sweep_validated.csv",
     {"color": "C2", "marker": "^", "ls": "-", "label": "Validated (pixel-level cement)"}),
]


def _load(csv_path: Path):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    return np.atleast_1d(data["f_star"]), np.atleast_1d(data["u_ratio"])


def plot_comparison(out_dir: Path, *, train_fstar: float | None = None,
                    title: str | None = None, out_path: Path | None = None,
                    verbose: bool = True) -> Path | None:
    """Draw whichever ``frequency_sweep_*.csv`` exist in ``out_dir`` on one axes.

    Returns the saved figure path, or ``None`` if no CSVs were found. Importable
    so other scripts (e.g. ``frequency_sweep_validated.py``) can render the same
    annotated comparison without shelling out.
    """
    out_dir = Path(out_dir)
    out_path = Path(out_path) if out_path is not None else out_dir / "frequency_sweep_comparison.png"

    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    f_lo, f_hi, y_hi = np.inf, -np.inf, 0.0
    any_plotted = False
    for name_csv, style in CASES:
        path = out_dir / name_csv
        if not path.exists():
            if verbose:
                print(f"missing (skipped): {path}")
            continue
        any_plotted = True
        fs, r = _load(path)
        ax.plot(fs, r, color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=1.6, markersize=6, label=style["label"])
        for x, y in zip(fs, r):
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                        xytext=(0, 6), ha="center", fontsize=7,
                        color=style["color"], alpha=0.8)
        f_lo, f_hi = min(f_lo, fs.min()), max(f_hi, fs.max())
        y_hi = max(y_hi, r.max())

    if not any_plotted:
        plt.close(fig)
        if verbose:
            print(f"no frequency_sweep_*.csv found in {out_dir}")
        return None

    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6,
               label="Perfect cloak (=1.0)")
    if train_fstar is not None and f_lo <= train_fstar <= f_hi:
        ax.axvline(train_fstar, color="gray", ls=":", lw=0.8, alpha=0.5,
                   label=rf"Training $f^*={train_fstar:g}$")
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title(title or f"{out_dir.name} — frequency sweep comparison")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1, fontsize=9)
    ax.grid(True, alpha=0.3)
    pad = 0.05 * (f_hi - f_lo) if f_hi > f_lo else 0.05
    ax.set_xlim(f_lo - pad, f_hi + pad)
    ax.set_ylim(0, max(1.1, y_hi * 1.1))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"saved {out_path}")
    return out_path


# ── multi-method comparison (NN vs neural-field inverse vs diffusion) ──

# Each entry: (subdir under opt_dir, label, color). The matched (homogenised)
# curve is drawn solid; the validated (pixel-level) curve, if present, dashed.
_METHOD_SERIES = [
    ("val_nn",        "Nearest neighbour",        "C1"),
    ("val_inverse",   "Inverse design (neural field)", "C4"),
    ("val_diffusion", "Diffusion (best-of-N)",    "C2"),
]


def plot_method_comparison(
    opt_dir: Path,
    *,
    methods: list[tuple[str, str, str]] | None = None,
    train_fstar: float | None = None,
    title: str | None = None,
    out_path: Path | None = None,
    include_validated: bool = True,
    verbose: bool = True,
) -> Path | None:
    """Overlay the per-method frequency sweeps of one optimised cloak.

    Reads ``<opt_dir>/<subdir>/frequency_sweep_matched.csv`` (and, when
    ``include_validated``, ``frequency_sweep_validated.csv``) for each method in
    ``methods`` and draws them on one axes — matched (homogenised) solid,
    validated (pixel-level) dashed. Baselines (obstacle/ideal/optimized) found
    directly in ``opt_dir`` are drawn too. Saved as
    ``<opt_dir>/frequency_sweep_method_comparison.png``.
    """
    opt_dir = Path(opt_dir)
    methods = methods or _METHOD_SERIES
    out_path = Path(out_path) if out_path is not None else opt_dir / "frequency_sweep_method_comparison.png"

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    f_lo, f_hi, y_hi = np.inf, -np.inf, 0.0
    any_plotted = False
    summary = []

    # method-agnostic baselines living directly in opt_dir
    for name_csv, style in CASES[:3]:   # obstacle, ideal, optimized
        path = opt_dir / name_csv
        if not path.exists():
            continue
        any_plotted = True
        fs, r = _load(path)
        ax.plot(fs, r, color=style["color"], marker=style["marker"],
                ls=style["ls"], lw=1.4, markersize=5, label=style["label"], alpha=0.7)
        f_lo, f_hi = min(f_lo, fs.min()), max(f_hi, fs.max())
        y_hi = max(y_hi, r.max())

    for subdir, label, color in methods:
        mpath = opt_dir / subdir / "frequency_sweep_matched.csv"
        if mpath.exists():
            any_plotted = True
            fs, r = _load(mpath)
            err = float(np.abs(1.0 - r).mean())
            ax.plot(fs, r, color=color, marker="v", ls="-", lw=1.9, markersize=6,
                    label=f"{label} — homogenised (mean |1-ratio| {err:.3f})")
            for x, y in zip(fs, r):
                ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=6.5, color=color, alpha=0.85)
            f_lo, f_hi = min(f_lo, fs.min()), max(f_hi, fs.max())
            y_hi = max(y_hi, r.max())
            summary.append((f"{label} (homogenised)", float(r.mean()), err))
        elif verbose:
            print(f"missing (skipped): {mpath}")

        if include_validated:
            vpath = opt_dir / subdir / "frequency_sweep_validated.csv"
            if vpath.exists():
                any_plotted = True
                fs, r = _load(vpath)
                err = float(np.abs(1.0 - r).mean())
                ax.plot(fs, r, color=color, marker="^", ls="--", lw=1.3, markersize=5,
                        alpha=0.8, label=f"{label} — pixel (mean |1-ratio| {err:.3f})")
                f_lo, f_hi = min(f_lo, fs.min()), max(f_hi, fs.max())
                y_hi = max(y_hi, r.max())
                summary.append((f"{label} (pixel)", float(r.mean()), err))

    if not any_plotted:
        plt.close(fig)
        if verbose:
            print(f"no per-method frequency_sweep CSVs found under {opt_dir}")
        return None

    ax.axhline(1.0, color="green", ls="--", lw=1.0, alpha=0.6, label="Perfect cloak (=1.0)")
    if train_fstar is not None and f_lo <= train_fstar <= f_hi:
        ax.axvline(train_fstar, color="gray", ls=":", lw=0.8, alpha=0.5,
                   label=rf"Training $f^*={train_fstar:g}$")
    ax.set_xlabel(r"$f^*$ (normalised frequency)")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$  (=1.0 is perfect cloak)")
    ax.set_title(title or f"{opt_dir.name} — cell-generation method comparison")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=8)
    ax.grid(True, alpha=0.3)
    pad = 0.05 * (f_hi - f_lo) if f_hi > f_lo else 0.05
    ax.set_xlim(f_lo - pad, f_hi + pad)
    ax.set_ylim(0, max(1.1, y_hi * 1.1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"saved {out_path}\n  series plotted ({len(summary)}):")
        for lab, m, e in summary:
            print(f"    {lab:42s} mean ratio {m:.4f}   mean |1-ratio| {e:.4f}")
    return out_path


def _resolve_target(target: Path) -> tuple[Path, float | None, str]:
    """Return (out_dir, train_fstar, name) from a config file or a directory.

    A config file lets us pull ``output_dir`` and the training ``f_star``;
    a directory is used as-is with no training-frequency marker.
    """
    if target.is_dir():
        return target, None, target.name
    if target.suffix in (".yaml", ".yml") and target.exists():
        # Import lazily so plotting works without the full solver stack.
        from rayleigh_cloak import load_config
        cfg = load_config(target)
        return Path(cfg.output_dir), float(cfg.domain.f_star), Path(cfg.output_dir).name
    raise SystemExit(f"target must be an existing config (.yaml) or directory; got {target}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="Experiment config (.yaml) or output directory.")
    p.add_argument("--out-dir", default=None,
                   help="Override the output directory holding the CSVs.")
    p.add_argument("--title", default=None, help="Override the plot title.")
    p.add_argument("--train-fstar", type=float, default=None,
                   help="Draw a vertical marker at the training f*. "
                        "Defaults to the config's domain.f_star when a config "
                        "is passed; omitted otherwise.")
    p.add_argument("--methods", action="store_true",
                   help="Multi-method mode: overlay the per-method sweeps "
                        "(val_nn / val_inverse / val_diffusion subdirs of the "
                        "output dir) instead of the single-dir 5-case plot.")
    p.add_argument("--no-validated", action="store_true",
                   help="In --methods mode, omit the pixel-level validated curves.")
    args = p.parse_args()

    out_dir, train_fstar, _ = _resolve_target(Path(args.target))
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    if args.train_fstar is not None:
        train_fstar = args.train_fstar

    if args.methods:
        saved = plot_method_comparison(
            out_dir, train_fstar=train_fstar, title=args.title,
            include_validated=not args.no_validated,
        )
        if saved is None:
            raise SystemExit(f"no per-method frequency_sweep CSVs found under {out_dir}")
        return

    saved = plot_comparison(out_dir, train_fstar=train_fstar, title=args.title)
    if saved is None:
        raise SystemExit(f"no frequency_sweep_*.csv found in {out_dir}")


if __name__ == "__main__":
    main()
