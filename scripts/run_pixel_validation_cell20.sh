#!/usr/bin/env bash
# Pixel-level validation for output/cell20_phys_flat4_materialreg4 @ f*=2.0.
#
# Two phases:
#   1. convergence sweep (one design, refinement swept) -> is the pixel-macro
#      solve even convergent on this machine? watch `ratio` climb toward the
#      homogenised value (~0.967 for diffusion).
#   2. single-refinement pixel-level FEM for all three design sets.
#
# The macro solve uses a sparse DIRECT LU factorisation (the frequency-domain
# elastodynamic operator + Rayleigh PML is INDEFINITE, so iterative solvers
# diverge). RAM is the binding constraint and grows ~ nodes^1.8. Rough budget,
# extrapolated from measured TRI6 points at f*=2.0:
#
#   refinement | nodes | DOF   | peak RAM | wall (measured cores)
#   -----------+-------+-------+----------+----------------------
#        8*     | 0.30M | 1.2M  |  18 GB   |  ~8 min      (* measured)
#       12      | 0.49M | 2.0M  |  ~45 GB  |  ~18 min
#       16      | 0.76M | 3.0M  |  ~100 GB |  ~40 min
#       20      | 1.10M | 4.4M  |  ~200 GB |  ~1.3 h
#       24      | 1.53M | 6.1M  |  ~350 GB |  ~2 h
#       32      | 2.60M | 10.4M |  ~900 GB |  ~5 h
#
# Memory does NOT drop with more cores (it's LU fill-in); wall time does.
# Set OMP_NUM_THREADS to the box's core count for the MUMPS factorisation.
#
# Usage:
#   scripts/run_pixel_validation_cell20.sh <phase> [REF]
#     phase = converge            # run the convergence sweep (phase 1)
#     phase = validate  REF       # run all-3 pixel FEM at refinement REF
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-}:$PWD"
D=output/cell20_phys_flat4_materialreg4
DATASET=output/ca_bulk_squared/stiffness.h5
CFG=$D/config_tri6.yaml
PARAMS=$D/optimized_params.npz

phase=${1:?"phase: converge | validate"}

if [[ "$phase" == "converge" ]]; then
  REFS=${2:-8,12,16,20,24}
  python scripts/mesh_convergence_validated.py \
    "$CFG" "$PARAMS" --dataset "$DATASET" --f-star 2.0 \
    --refinements "$REFS" \
    --cell-designs "$D/cell_designs_diffusion" \
    --bail-on-oom -o "$D/conv_tri6_heavy"

elif [[ "$phase" == "validate" ]]; then
  REF=${2:?"give a refinement, e.g. 20"}
  for SET in diffusion inverse bestof; do
    case $SET in
      diffusion) DIR=cell_designs_diffusion;;
      inverse)   DIR=cell_designs;;
      bestof)    DIR=cell_designs_bestof;;
    esac
    echo "=== pixel-level FEM: $SET (refinement=$REF) ==="
    python scripts/frequency_sweep_validated.py \
      "$CFG" "$PARAMS" --dataset "$DATASET" \
      --fmin 2.0 --fmax 2.0 --fstep 0.1 \
      --no-matched --refinement-factor "$REF" \
      --cell-designs "$D/$DIR" \
      -o "$D/val_${SET}_tri6"
  done
else
  echo "unknown phase '$phase' (want: converge | validate)" >&2; exit 1
fi
