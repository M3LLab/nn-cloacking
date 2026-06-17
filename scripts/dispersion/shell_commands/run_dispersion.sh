#!/usr/bin/env bash
# Bloch-Floquet dispersion curves with IPR analysis.
# Reproduces Fig 3(d)-(e) from Chatzopoulos et al. (2023).
#
# Output: output/dispersion/dispersion_comparison.png
#         output/dispersion/dispersion_reference.npz
#         output/dispersion/dispersion_ideal_cloak.npz
#
# Usage:
#   ./run_dispersion.sh           # run (skips if NPZ files already exist)
#   ./run_dispersion.sh -f        # force re-run
#   ./run_dispersion.sh --h-elem 0.12   # coarser mesh, faster

set -e
cd "$(dirname "$0")/.."

# Translate shorthand flags for the Python script.
# -f        → --force
args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -f)  args+=("--force");          shift ;;
        *)   args+=("$1");               shift ;;
    esac
done

# Paper reference (Chatzopoulos et al. 2023, §5.1):
#   unit-cell H = 4.305 λ*, L_c = 2.0 λ* (implicit, BZ edge k*=0.25),
#   a = 0.0774 H, c = 0.1545 H, b = 3a ≈ λ* ;
#   IPR > 3.5 to distinguish surface modes; >500 eigenvalues per k to reach f*=2.5.
PYTHONPATH="$(pwd)" python scripts/dispersion/dispersion_ideal.py \
    --n-kpts 50 \
    --n-eigs 550 \
    --h-elem 0.08 \
    --h-fine 0.03 \
    --ipr-thr 3.5 \
    --f-max 2.5 \
    --H-factor 1.0 \
    --out-dir output/dispersion \
    "${args[@]}"
