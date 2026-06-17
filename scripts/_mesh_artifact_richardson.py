"""Push TRI6 homogenisation to high resolution on a single cell and extrapolate.

Answers: does the effective C of a (possibly thin) cell PLATEAU under refinement
(genuine ligament) or keep DROPPING toward ~0 (corner-touching pixels that a
coarse mesh spuriously bridges)? Sweeps TRI6 at increasing N for one dataset
cell, prints C and the step-to-step change, and Richardson-extrapolates C_inf
from the last three points (assuming power-law h-convergence).

Usage:
    PYTHONPATH=. python scripts/_mesh_artifact_richardson.py \
        --idx 103600 --mesh-res 100,150,200,300,400
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import h5py
import numpy as np

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "_mac", _HERE / "_mesh_artifact_homog_convergence.py")
_mac = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mac)


def richardson(ns, vals):
    """Estimate C_inf and apparent order p from the last 3 (N, C) points,
    assuming C(h) = C_inf + a*h^p with h = 1/N and a geometric-ish N triple."""
    (n1, n2, n3), (c1, c2, c3) = ns[-3:], vals[-3:]
    h1, h2, h3 = 1.0 / n1, 1.0 / n2, 1.0 / n3
    d12, d23 = c1 - c2, c2 - c3
    if abs(d23) < 1e-30 or d12 * d23 <= 0:
        return c3, float("nan")  # not in asymptotic regime / noise
    # p from ratio of successive differences with the actual h's
    p = np.log(abs(d12 / d23)) / np.log((h1 - h2) / (h2 - h3) + 1e-30)
    r = (h3 ** p)
    # C_inf via two-point Richardson on the finest pair using order p
    c_inf = c3 + (c3 - c2) * (h3 ** p) / ((h2 ** p) - (h3 ** p) + 1e-30)
    return float(c_inf), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="output/ca_bulk_squared/stiffness.h5")
    ap.add_argument("--idx", type=int, default=None,
                    help="dataset cell index; default = lowest-vf (thinnest) cell")
    ap.add_argument("--mesh-res", default="100,150,200,300,400")
    ap.add_argument("--img-res", type=int, default=50)
    args = ap.parse_args()

    mesh_res = [int(r) for r in args.mesh_res.split(",") if r.strip()]
    with h5py.File(args.dataset, "r") as f:
        if args.idx is None:
            args.idx = int(np.argsort(f["vf"][:])[0])
        img = f["cells"][args.idx].astype(np.uint8)
        vf = float(f["vf"][args.idx])
        ds = (float(f["C11"][args.idx]), float(f["C22"][args.idx]),
              float(f["C12"][args.idx]), float(f["C66"][args.idx]))

    # Connectivity diagnostic: does the solid phase connect top<->bottom and
    # left<->right via EDGE adjacency (4-connectivity) only? If a path needs a
    # diagonal (corner) hop, a coarse mesh bridges it spuriously.
    def connected_4(mask, axis):
        from collections import deque
        H, W = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        dq = deque()
        if axis == 0:  # top row -> bottom row
            starts = [(0, c) for c in range(W) if mask[0, c]]
            goal = lambda r, c: r == H - 1
        else:          # left col -> right col
            starts = [(r, 0) for r in range(H) if mask[r, 0]]
            goal = lambda r, c: c == W - 1
        for s in starts:
            seen[s] = True; dq.append(s)
        while dq:
            r, c = dq.popleft()
            if goal(r, c):
                return True
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True; dq.append((nr, nc))
        return False

    m = img > 0
    c_v = connected_4(m, 0)
    c_h = connected_4(m, 1)
    print(f"cell idx={args.idx}  vf={vf:.3f}")
    print(f"dataset C (TRI3@1/pix) = C11={ds[0]:.3e} C22={ds[1]:.3e} C12={ds[2]:.3e} C66={ds[3]:.3e}")
    print(f"4-connectivity (edge-only): top<->bottom={c_v}  left<->right={c_h}  "
          f"(False => load path relies on diagonal/corner touch => coarse-mesh over-stiffness)\n")

    print(f"  {'N':>5} {'el/pix':>7} {'C11':>11} {'C22':>11} {'C12':>11} {'C66':>11} {'dC22%':>8}")
    C = {}
    prev22 = None
    for n in mesh_res:
        v = _mac.compute_C_flat4(img, n, ele_type="TRI6")
        C[n] = v
        d = "" if prev22 is None else f"{(v[1]-prev22)/prev22*100:>7.1f}%"
        prev22 = v[1]
        print(f"  {n:>5d} {n/args.img_res:>7.2f} {v[0]:>11.3e} {v[1]:>11.3e} "
              f"{v[2]:>11.3e} {v[3]:>11.3e} {d:>8}", flush=True)

    if len(mesh_res) >= 3:
        ns = mesh_res
        print("\n  Richardson extrapolation (last 3 points):")
        names = ["C11", "C22", "C12", "C66"]
        for k in range(4):
            vals = [C[n][k] for n in ns]
            cinf, p = richardson(ns, vals)
            finest = vals[-1]
            rem = abs(finest - cinf) / (abs(cinf) + 1e-30) * 100
            print(f"    {names[k]}: C_inf={cinf:.3e}  apparent order p={p:.2f}  "
                  f"finest(N={ns[-1]}) still {rem:.1f}% above C_inf")


if __name__ == "__main__":
    main()
