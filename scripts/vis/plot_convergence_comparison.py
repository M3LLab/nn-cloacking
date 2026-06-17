"""Plot right-boundary cloaking distortion vs iteration for two optimisation approaches."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

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
adam = np.genfromtxt(
    "output/circular_optimize_10c/loss_history.csv",
    delimiter=",", names=True,
)
neural = np.genfromtxt(
    "output/circular_optimize_neural/loss_history.csv",
    delimiter=",", names=True,
)

N = 55  # first 55 steps

adam_steps = adam["step"][:N].astype(int)
adam_D = 100.0 * np.sqrt(adam["cloak"][:N])

neural_steps = neural["step"][:N].astype(int)
neural_D = 100.0 * np.sqrt(neural["cloak"][:N])

# ── Reference lines (from Table 1 & Table 3 in the paper) ─────────────
D_continuous = 27.51   # continuous c_eff at 160×160 mesh
D_nassar_80 = 45.88    # Nassar 80×80 cell initialisation

# ── Plot ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots()

ax.plot(adam_steps, adam_D, "-", color="C0", linewidth=1.5, label="Direct (Adam)")
ax.plot(neural_steps, neural_D, "-", color="C1", linewidth=1.5, label="Neural reparam. (INR)")

ax.axhline(D_nassar_80, color="C3", linestyle="--", linewidth=1.0,
           label=f"Nassar 80×80 init ({D_nassar_80:.1f}%)")
ax.axhline(D_continuous, color="C2", linestyle=":", linewidth=1.0,
           label=f"Continuous $\\mathbf{{c}}_{{\\mathrm{{eff}}}}$ ({D_continuous:.1f}%)")

ax.set_xlabel("Optimisation step")
ax.set_ylabel("Right-boundary distortion $\\mathcal{D}$ [%]")
ax.set_xlim(0, N - 1)
ax.set_ylim(bottom=0)
ax.legend(frameon=True, fancybox=False, edgecolor="0.7")
ax.grid(True, linewidth=0.3, alpha=0.6)

fig.tight_layout()
fig.savefig("output/convergence_comparison.pdf")
fig.savefig("output/convergence_comparison.png")
print("Saved to output/convergence_comparison.{pdf,png}")
plt.show()
