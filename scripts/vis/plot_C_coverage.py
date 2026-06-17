"""
Plot distribution of 2D plane-strain C tensor components across the full dataset.

Reads only *binary_C files (144 bytes each) — no voxels loaded.
2D components extracted from the 6×6 Voigt matrix at indices [0,1,5]:
  C11 = C[0,0], C22 = C[1,1], C12 = C[0,1], C66 = C[5,5], C16 = C[0,5]

Usage:
    python scripts/plot_C_coverage.py [dataset_root]

Defaults to /home/david/Downloads/structures_dataset/3/
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

DATASET_ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "/home/david/Downloads/structures_dataset/3/"
)

# ── 1. Discover all binary_C files ───────────────────────────────────────────

all_C_paths = sorted(DATASET_ROOT.glob("**/*binary_C"))
print(f"Dataset root : {DATASET_ROOT}")
print(f"Total samples: {len(all_C_paths):,}")

# Per-split counts
splits = {}
for p in all_C_paths:
    split = p.parent.name if p.parent != DATASET_ROOT else "(root)"
    splits[split] = splits.get(split, 0) + 1
print("\nSamples per split:")
for name, count in sorted(splits.items(), key=lambda x: -x[1]):
    print(f"  {name:30s}  {count:>8,}")

# ── 2. Load all C tensors ─────────────────────────────────────────────────────

print(f"\nLoading {len(all_C_paths):,} C tensors …")
C_all = np.empty((len(all_C_paths), 6, 6), dtype=np.float32)
for i, p in enumerate(all_C_paths):
    with open(p, "rb") as f:
        C_all[i] = np.fromfile(f, dtype=np.float32).reshape(6, 6)
    if (i + 1) % 10_000 == 0:
        print(f"  {i+1:>7,} / {len(all_C_paths):,}")
print("Done.")

# 2D plane-strain submatrix indices [0,1,5]
C11 = C_all[:, 0, 0]
C22 = C_all[:, 1, 1]
C12 = C_all[:, 0, 1]
C66 = C_all[:, 5, 5]
C16 = C_all[:, 0, 5]   # should be ≈ 0 for cubic symmetry

# ── 3. Print statistics ───────────────────────────────────────────────────────

components = {"C11": C11, "C22": C22, "C12": C12, "C66": C66, "C16 (≈0?)": C16}

print(f"\n{'Component':<14} {'min':>12} {'max':>12} {'mean':>12} {'std':>12} {'|max|/C11_mean':>15}")
C11_mean = C11.mean()
for name, vals in components.items():
    print(
        f"{name:<14} {vals.min():>12.3g} {vals.max():>12.3g}"
        f" {vals.mean():>12.3g} {vals.std():>12.3g}"
        f" {np.abs(vals).max() / C11_mean:>15.4f}"
    )

# Check cubic symmetry quality
print(f"\nCubic symmetry check:")
print(f"  max |C11 - C22| / C11_mean = {np.abs(C11 - C22).max() / C11_mean:.4e}")
print(f"  mean |C16| / C11_mean      = {np.abs(C16).mean() / C11_mean:.4e}")

# ── 4. Plot distributions ─────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 9))
fig.suptitle(
    f"2D Plane-Strain C Tensor Coverage  (N = {len(all_C_paths):,})",
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

plot_specs = [
    (0, 0, "C11",     C11,  "steelblue"),
    (0, 1, "C22",     C22,  "steelblue"),
    (0, 2, "C12",     C12,  "darkorange"),
    (1, 0, "C66",     C66,  "seagreen"),
    (1, 1, "C16 (≈0?)", C16, "firebrick"),
]

for row, col, label, vals, color in plot_specs:
    ax = fig.add_subplot(gs[row, col])
    ax.hist(vals, bins=120, color=color, alpha=0.75, edgecolor="none")
    ax.set_title(label, fontsize=12)
    ax.set_xlabel("Stiffness (Pa)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.text(
        0.97, 0.95,
        f"μ={vals.mean():.3g}\nσ={vals.std():.3g}",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=8, color="black",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
    )

# 2D scatter: C11 vs C12 coloured by C66 (coverage map)
ax_scatter = fig.add_subplot(gs[1, 2])
sc = ax_scatter.scatter(C11[::10], C12[::10], c=C66[::10], s=1, alpha=0.3,
                         cmap="viridis", rasterized=True)
plt.colorbar(sc, ax=ax_scatter, label="C66 (Pa)")
ax_scatter.set_xlabel("C11 (Pa)", fontsize=9)
ax_scatter.set_ylabel("C12 (Pa)", fontsize=9)
ax_scatter.set_title("C11 vs C12 (colored by C66)", fontsize=10)
ax_scatter.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

out_path = Path("output/C_coverage.png")
out_path.parent.mkdir(exist_ok=True)
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")
plt.show()
