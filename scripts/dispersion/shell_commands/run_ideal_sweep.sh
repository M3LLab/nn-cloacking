#!/usr/bin/env bash
# Ideal cloaking frequency sweep (f* = 0.1 → 4.0, step 0.1)
# Uses analytical transformation-elasticity C_eff/rho_eff — no optimization.
#
# Output: output/continuous/frequency_sweep_ideal.csv
#         output/continuous/frequency_sweep.png
#
# Usage:
#   ./run_ideal_sweep.sh           # run sweep (skips if CSV already exists)
#   ./run_ideal_sweep.sh -f        # force re-run even if CSV exists

set -e
cd "$(dirname "$0")/.."

PYTHONPATH="$(pwd)" python scripts/frequency_sweep.py \
    --no-obstacle --no-optimized \
    "$@" \
    configs/continuous.yaml
