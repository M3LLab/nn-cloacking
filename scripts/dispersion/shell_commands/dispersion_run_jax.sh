#!/usr/bin/env bash
# Bloch-Floquet dispersion curves using the rayleigh_cloak config pipeline.
#
# Config-driven alternative to run_dispersion.sh.  Physical parameters are
# derived from a SimulationConfig YAML via DerivedParams (consistent with
# the forward solver) rather than hardcoded.
#
# Output (default): <output_dir>/dispersion/dispersion_comparison.png
#                   <output_dir>/dispersion/dispersion_*.npz
#
# Usage:
#   ./dispersion_run_jax.sh                               # default (configs/continuous.yaml)
#   ./dispersion_run_jax.sh configs/continuous.yaml        # explicit config
#   ./dispersion_run_jax.sh -f                             # force re-run
#   ./dispersion_run_jax.sh configs/continuous.yaml -f     # config + force
#   ./dispersion_run_jax.sh --h-elem 0.12                  # coarser mesh
#
# Optimized cloak dispersion (reference vs optimized):
#   ./dispersion_run_jax.sh best_configs/cauchy_tri_top.yaml \
#       --params-npz best_configs/optimized_params.npz \
#       --case optimized_vs_ref

set -e
cd "$(dirname "$0")/.."

# Parse: first positional .yaml arg → config, rest passed through.
config=""
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f)  args+=("--force");  shift ;;
        *.yaml|*.yml)
            if [[ -z "$config" ]]; then
                config="$1"
            else
                args+=("$1")
            fi
            shift ;;
        *)   args+=("$1");       shift ;;
    esac
done

# Default config
if [[ -z "$config" ]]; then
    config="configs/continuous.yaml"
fi

PYTHONPATH="$(pwd)" python scripts/dispersion/dispersion_jaxfem.py \
    "$config" \
    --n-kpts 50 \
    --n-eigs 550 \
    --h-elem 0.08 \
    --h-fine 0.03 \
    --ipr-thr 2.5\
    --f-max 4.0 \
    "${args[@]}"
