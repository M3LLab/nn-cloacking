#!/usr/bin/env bash
# Diagnose where the extra high-IPR modes in the optimized-cloak dispersion
# come from: geometry (triangle cutout) vs material optimization (cell map).
#
# Produces exactly three comparison plots in <output_dir>/dispersion/:
#   dispersion_comparison_ideal_vs_ref.png
#   dispersion_comparison_optimized_vs_ref.png
#   dispersion_comparison_ideal_vs_optimized.png
#
# Interpretation:
#   - ideal_vs_ref shows the effect of the triangle cutout + analytic C_eff
#     (any high-IPR modes here are geometric / Cosserat-approx).
#   - optimized_vs_ref shows the effect of the optimized material.
#   - ideal_vs_optimized isolates what the optimization adds/removes on top
#     of the ideal analytic cloak (this is usually the most informative).
#
# Usage:
#   ./run_dispersion_diagnose.sh
#   ./run_dispersion_diagnose.sh -f                               # force rerun
#   ./run_dispersion_diagnose.sh <config.yaml> <params.npz>       # custom inputs
#   ./run_dispersion_diagnose.sh --ipr-thr 3.0 --f-max 3.0        # plot bounds
#   ./run_dispersion_diagnose.sh <cfg> <npz> --ipr-thr 2.0 -f

set -e
cd "$(dirname "$0")/.."

CONFIG=""
PARAMS=""
FORCE=""
IPR_THR="2.5"
F_MAX="4.0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--force)   FORCE="-f"; shift ;;
        --ipr-thr)    IPR_THR="$2"; shift 2 ;;
        --f-max)      F_MAX="$2"; shift 2 ;;
        *)
            if [[ -z "$CONFIG" ]]; then
                CONFIG="$1"
            elif [[ -z "$PARAMS" ]]; then
                PARAMS="$1"
            fi
            shift ;;
    esac
done

CONFIG="${CONFIG:-best_configs/latest/latent_ae_optimize_sweep.yaml}"
PARAMS="${PARAMS:-best_configs/latest/optimized_params.npz}"

echo "=============================================================="
echo " Config:         $CONFIG"
echo " Optimized npz:  $PARAMS"
echo " Force rerun:    ${FORCE:-no}"
echo " IPR threshold:  $IPR_THR"
echo " f* max:         $F_MAX"
echo "=============================================================="

# ── 1. Inspect optimized params ───────────────────────────────────────
echo
echo "─── Inspecting $PARAMS ───"
PYTHONPATH="$(pwd)" python - "$CONFIG" "$PARAMS" <<'PY'
import sys, numpy as np, yaml
CONFIG, PARAMS = sys.argv[1], sys.argv[2]
d = np.load(PARAMS)
cfg = yaml.safe_load(open(CONFIG))
rho0 = float(cfg["material"]["rho0"])
cs   = float(cfg["material"]["cs"])
mu0  = rho0 * cs**2
print("keys         :", list(d.keys()))
for k in d.keys():
    print(f"  {k:20s} shape={d[k].shape}  dtype={d[k].dtype}")
C = d["cell_C_flat"]; rho = d["cell_rho"]
print()
print(f"n_C_params (from npz)       : {C.shape[1]}")
print(f"n_cells                     : {C.shape[0]}")
print(f"background mu0 = rho*cs^2   : {mu0:.1f}")
print(f"background rho0             : {rho0:.1f}")
print()
print(f"cell_C_flat col-0 range     : [{C[:,0].min():.3e}, {C[:,0].max():.3e}]  (lambda if iso)")
if C.shape[1] >= 2:
    print(f"cell_C_flat col-1 range     : [{C[:,1].min():.3e}, {C[:,1].max():.3e}]  (mu if iso)")
print(f"cell_rho range              : [{rho.min():.3e}, {rho.max():.3e}]")
print()
rho_floor = 0.01*rho0
mu_floor  = 0.01*mu0
print(f"Would-clamp at rho_min={rho_floor:.2f}: {int((rho<rho_floor).sum())} / {rho.size}")
if C.shape[1] >= 2:
    print(f"Would-clamp at mu_min={mu_floor:.2f}: {int((C[:,1]<mu_floor).sum())} / {C.shape[0]}")
    print(f"Cells with negative mu      : {int((C[:,1]<0).sum())}")
print(f"Cells with negative lambda  : {int((C[:,0]<0).sum())}")
PY

NC=$(PYTHONPATH="$(pwd)" python -c "import numpy as np; print(np.load('$PARAMS')['cell_C_flat'].shape[1])")
echo
echo "Using --n-C-params $NC"

OUTDIR=$(PYTHONPATH="$(pwd)" python - "$CONFIG" <<'PY'
import sys, yaml, pathlib
cfg = yaml.safe_load(open(sys.argv[1]))
print(pathlib.Path(cfg["output_dir"]) / "dispersion")
PY
)

# Helper: remove the per-case "*_vs_rayleigh" single plots produced by the
# inner script — we only want comparison plots here.
clean_single_plots() {
    rm -f "$OUTDIR/dispersion_reference_vs_rayleigh.png" \
          "$OUTDIR/dispersion_ideal_cloak_vs_rayleigh.png" \
          "$OUTDIR/dispersion_optimized_cloak_vs_rayleigh.png"
}

# ── 2. reference + ideal_cloak  (isolates triangle/geometry effect) ───
echo
echo "=============================================================="
echo " Run A:  reference + ideal analytic cloak  (--case both)"
echo "=============================================================="
./shell_commands/dispersion_run_jax.sh "$CONFIG" --case both --ipr-thr "$IPR_THR" --f-max "$F_MAX" $FORCE
clean_single_plots
# Rename so run B doesn't overwrite and name is self-describing
if [[ -f "$OUTDIR/dispersion_comparison.png" ]]; then
    mv "$OUTDIR/dispersion_comparison.png" "$OUTDIR/dispersion_comparison_ideal_vs_ref.png"
    echo "Saved → $OUTDIR/dispersion_comparison_ideal_vs_ref.png"
fi

# ── 3. reference + optimized_cloak  (isolates optimization effect) ────
echo
echo "=============================================================="
echo " Run B:  reference + optimized cloak  (--case optimized_vs_ref)"
echo "=============================================================="
./shell_commands/dispersion_run_jax.sh "$CONFIG" \
    --params-npz "$PARAMS" \
    --n-C-params "$NC" \
    --case optimized_vs_ref \
    --ipr-thr "$IPR_THR" \
    --f-max "$F_MAX" \
    $FORCE
clean_single_plots
if [[ -f "$OUTDIR/dispersion_comparison_optimized.png" ]]; then
    mv "$OUTDIR/dispersion_comparison_optimized.png" \
       "$OUTDIR/dispersion_comparison_optimized_vs_ref.png"
    echo "Saved → $OUTDIR/dispersion_comparison_optimized_vs_ref.png"
fi

# ── 4. ideal_cloak vs optimized_cloak  (most informative comparison) ──
echo
echo "=============================================================="
echo " Plot C:  ideal cloak vs optimized cloak"
echo "=============================================================="
PYTHONPATH="$(pwd)" python - "$CONFIG" "$IPR_THR" "$F_MAX" <<'PY'
import sys, pathlib, numpy as np, yaml
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

cfg = yaml.safe_load(open(sys.argv[1]))
ipr_thr = float(sys.argv[2])
f_max   = float(sys.argv[3])
out_dir = pathlib.Path(cfg["output_dir"]) / "dispersion"

# Matches dispersion_run_jax.sh defaults (h-elem 0.08, h-fine 0.03)
tag = "h0.08_hf0.03"
ideal_npz = out_dir / f"dispersion_ideal_cloak_{tag}.npz"
opt_npz   = out_dir / f"dispersion_optimized_cloak_{tag}.npz"
if not ideal_npz.exists() or not opt_npz.exists():
    print(f"ERROR: expected cached sweeps not found:\n  {ideal_npz}\n  {opt_npz}")
    sys.exit(1)
ideal = np.load(ideal_npz)
opt   = np.load(opt_npz)

# Rayleigh-line drawing needs BZ edge; L_c = 2*lambda_star ⇒ k_edge_norm = 0.25.
k_edge = 0.25
ipr_cap = 15.0

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": True, "axes.spines.right": True,
})
fig, ax = plt.subplots(figsize=(8.5, 6.0))
norm = Normalize(vmin=1.0, vmax=ipr_cap)
cmap = cm.turbo

for d, marker, size, edge, lw, label in [
    (ideal, "D", 28*1.9, "none",  0.0, "Ideal Cloak"),
    (opt,   "o", 28*1.0, "black", 0.9, "Optimized Cloak"),
]:
    ks, fs, iprs = d["ks"], d["fs"], d["iprs"]
    mask = fs <= f_max
    bulk    = mask & (iprs <  ipr_thr)
    surface = mask & (iprs >= ipr_thr)
    ax.scatter(ks[bulk], fs[bulk], s=3, marker=".",
               c=np.clip(iprs[bulk], 1.0, ipr_cap), norm=norm, cmap=cmap,
               edgecolors="none", alpha=0.7, zorder=3)
    ax.scatter(ks[surface], fs[surface], s=size, marker=marker,
               c=np.clip(iprs[surface], 1.0, ipr_cap), norm=norm, cmap=cmap,
               edgecolors=edge, linewidths=lw, alpha=0.95, zorder=4, label=label)

# Folded Rayleigh guide lines
k_line = np.linspace(0, k_edge, 300)
labelled = False; m = 0
while True:
    f_up   = 2*m*k_edge + k_line
    f_down = 2*(m+1)*k_edge - k_line
    if f_up[0] > f_max: break
    for branch in (f_up, f_down):
        b = branch <= f_max
        if not b.any(): continue
        ax.plot(k_line[b], branch[b], "k--", lw=0.8, alpha=0.45, zorder=2,
                label=("Rayleigh Analytic" if not labelled else None))
        labelled = True
    m += 1

ax.set_xlim(-0.012, k_edge + 0.012)
ax.set_ylim(0, f_max)
ax.set_xticks(np.arange(0.0, k_edge + 1e-6, 0.05))
ax.set_xlabel(r"$\xi = k\lambda^* / (2\pi)$", fontsize=12)
ax.set_ylabel(r"$f^* = f / (c_R / \lambda^*)$", fontsize=12)
ax.set_title("Bloch-Floquet Dispersion: Ideal vs Optimized Cloak",
             fontsize=13, fontweight="bold", pad=10)
ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
ax.tick_params(direction="out", length=4)
ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="0.85")

fig.subplots_adjust(left=0.10, right=0.86, top=0.92, bottom=0.10)
cbar_ax = fig.add_axes([0.885, 0.10, 0.022, 0.82])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label("IPR", fontsize=10); cbar.outline.set_visible(False)
cbar.ax.tick_params(length=3)

out_path = out_dir / "dispersion_comparison_ideal_vs_optimized.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"Plot saved → {out_path}")
PY

echo
echo "=============================================================="
echo " Done.  Three plots in $OUTDIR :"
echo "   dispersion_comparison_ideal_vs_ref.png        (geometry only)"
echo "   dispersion_comparison_optimized_vs_ref.png    (optimized vs flat)"
echo "   dispersion_comparison_ideal_vs_optimized.png  (opt vs ideal cloak)"
echo "=============================================================="
