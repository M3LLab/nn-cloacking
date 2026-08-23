#!/usr/bin/env bash
# Launch (or resume) the inverse-design gap-filling queue in the background.
#
# Each worker takes every N-th target from the enclosure-ranked list of interior
# holes that v2 still does not cover, and writes one .json/.npz per target into
# $OUT.  Workers are independent and stateless apart from that directory, so the
# run is resumable: re-running this script skips whatever is already on disk.
#
#   ./dataset/cellular_chiral/fill_gaps_launch.sh [n_workers] [out_dir]
#
# Progress:  python -m dataset.cellular_chiral.fill_gaps_report -i $OUT
# Stop:      touch $OUT/STOP    (clean, at the next target boundary)
#            pkill -f fill_gaps_inverse   (immediate)
#
# Why the watchdog exists (read before removing it)
# -------------------------------------------------
# Some seed geometries produce a near-singular stiffness matrix whose UMFPACK
# factorisation never returns.  The worker then sits at 100 % CPU holding
# ~2.4 GB, in `bias()` on the first solve of a target, indefinitely -- observed
# stuck for 3.5 days.  Three separate multi-day runs were lost to this:
#
#   attempt 1  8 workers, no bound     305 targets, then 3.5 days of nothing
#   attempt 2  restart every 20        9 targets in 115 h (a queue-filter bug)
#   attempt 3  restart every 20        46 targets in 72 h, 4 of 6 workers wedged
#
# Restarting on a target boundary does NOT catch it, because the hang happens
# mid-chunk and the boundary is never reached.  Only an external watchdog does.
# It was initially misdiagnosed twice -- first as a JAX retrace leak (disproved:
# fresh vs hoisted callables measure identically), then as memory exhaustion
# (disproved: the box had 18 GB free while every worker was wedged).  The real
# signature is 100 % CPU with a silent log.
#
# $CHUNK still bounds worker lifetime, which caps the ~1 MB/FEM-call growth in
# the jax-fem path (RSS 1196 -> 1300 MB over 100 calls); that is a real but
# secondary effect.
set -euo pipefail

N=${1:-6}
OUT=${2:-output/ca_bulk_squared/inverse_fill}
CHUNK=${CHUNK:-20}                 # targets per worker lifetime
STALL=${STALL:-1800}               # seconds of log silence before the watchdog fires
PY=${PY:-/root/miniconda3/envs/jax-fem-env/bin/python}
cd "$(dirname "$0")/../.."

mkdir -p "$OUT/logs"
rm -f "$OUT/STOP"

for w in $(seq 0 $((N - 1))); do
    # setsid + full redirection: the supervisor must not inherit this script's
    # stdout, or launching it from a pipeline blocks until every worker exits.
    setsid bash -c '
        w=$1; N=$2; OUT=$3; CHUNK=$4; PY=$5; STALL=$6
        while [ ! -f "$OUT/STOP" ]; do
            LOG="$OUT/logs/worker_$w.log"
            PYTHONPATH=. PYTHONUNBUFFERED=1 "$PY" \
                -m dataset.cellular_chiral.fill_gaps_inverse \
                --all --worker "$w" --n-workers "$N" -o "$OUT" \
                --max-targets "$CHUNK" \
                --coarse-iters 60 --fine-iters 12 --n-seeds 1 \
                >> "$LOG" 2>&1 &
            PID=$!
            # Watchdog.  Some seed geometries give a near-singular stiffness
            # matrix whose UMFPACK factorisation never returns: the process sits
            # at 100 % CPU holding ~2.4 GB, indefinitely.  It happens mid-chunk,
            # so bounding the worker lifetime cannot catch it.  Three runs were
            # lost to this before it was identified.  If the log goes quiet for
            # STALL seconds, blacklist the target being worked and kill it.
            while kill -0 $PID 2>/dev/null; do
                sleep 60
                QUIET=$(( $(date +%s) - $(stat -c %Y "$LOG" 2>/dev/null || date +%s) ))
                [ "$QUIET" -lt "$STALL" ] && continue
                TID=$(grep -ao "^target [0-9]*" "$LOG" | tail -1 | awk "{print \$2}")
                if [ -n "$TID" ]; then
                    printf "{\"target_id\": %s, \"timeout_s\": %s, \"worker\": %s}\n" \
                        "$TID" "$QUIET" "$w" \
                        > "$OUT/$(printf "timeout_%06d.json" "$TID")"
                    echo "watchdog: target $TID wedged for ${QUIET}s - killed and blacklisted" \
                        >> "$LOG"
                fi
                kill -9 $PID 2>/dev/null || true
                break
            done
            wait $PID 2>/dev/null || true
            # Stop only when the worker itself reports an empty slice.
            # Inferring it from a change in the shared result count is racy and
            # wrong: other workers write to the same directory, and a pass that
            # legitimately skips finished targets adds nothing of its own.
            # (No apostrophes in here - this whole block is single-quoted.)
            if tail -5 "$OUT/logs/worker_$w.log" | grep -q "QUEUE_EMPTY"; then
                echo "worker $w: slice complete" >> "$OUT/logs/worker_$w.log"
                break
            fi
        done
    ' _ "$w" "$N" "$OUT" "$CHUNK" "$PY" "$STALL" </dev/null >/dev/null 2>&1 &
    sleep 2                      # stagger: the v2 load and KD-tree build are I/O heavy
done
echo "launched $N workers (restart every $CHUNK targets, watchdog ${STALL}s) -> $OUT"
echo "logs: $OUT/logs   stop cleanly: touch $OUT/STOP"
