"""Plot right-boundary cloaking distortion vs iteration for two optimisation approaches."""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from rayleigh_cloak.config import load_config

# ── Config ─────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot loss history for an experiment.")
parser.add_argument("source", help="Path to YAML config or loss CSV file")
parser.add_argument("-n", type=int, default=None, help="Number of steps to plot (default: all)")
args = parser.parse_args()

source = Path(args.source)
if source.suffix in (".yaml", ".yml"):
    cfg = load_config(str(source))
    output_dir = Path(cfg.output_dir)
    loss_csv = output_dir / "loss_history.csv"
elif source.suffix == ".csv":
    loss_csv = source
    output_dir = loss_csv.parent
else:
    parser.error(f"unsupported file type '{source.suffix}', expected .yaml/.yml or .csv")

# ── Style ──────────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (5.5, 3.8),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Load data ──────────────────────────────────────────────────────────
loss = np.genfromtxt(
    loss_csv,
    delimiter=",", names=True,
)

N = args.n if args.n is not None else len(loss)

neural_steps = loss["step"][:N].astype(int)
neural_D = 100.0 * np.sqrt(loss["cloak"][:N])

# ── Reference lines (from Table 1 & Table 3 in the paper) ─────────────
D_continuous = 27.51   # continuous c_eff at 160×160 mesh
D_nassar_80 = 45.88    # Nassar 80×80 cell initialisation

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

ax.plot(neural_steps, neural_D, "-", color="C1", linewidth=1.5, label="Neural reparam. (INR)")

ax.set_xlabel("Optimisation step")
ax.set_ylabel("Right-boundary distortion $\\mathcal{D}$ [%]")
ax.set_xlim(0, N - 1)
ax.set_ylim(bottom=0)
ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
ax.grid(True, linewidth=0.3, alpha=0.6)

fig.tight_layout()
out_path = output_dir / "loss.png"
fig.savefig(out_path)
print(f"Saved to {out_path}")
plt.show()
