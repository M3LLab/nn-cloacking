#!/usr/bin/env bash
# Extract per-frame JPEGs for the \animategraphics calls in presentation.tex.
# The frame rate passed to \animategraphics must match the source video's
# (both clips below are 20 fps); its first/last indices are 1 and the frame
# count printed here. Widths are set from the on-slide size, not the source.
set -euo pipefail
cd "$(dirname "$0")"

extract() {  # <video> <outdir> <target width>
    local src=$1 dir=$2 width=$3
    rm -rf "$dir" && mkdir -p "$dir"
    ffmpeg -v error -i "$src" -vf "scale=$width:-2" -q:v 3 -vsync 0 \
        "$dir/frame-%03d.jpg"
    echo "$dir: $(ls "$dir" | wc -l) frames, $(du -sh "$dir" | cut -f1)"
}

extract wave_propagation_uy_f2.00.mp4                  wave_frames  900
extract wave_propagation_morph_f2.00_zoom2.5d1.5.mp4   morph_frames 900
