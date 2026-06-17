"""Sweep over cell resolutions for multi-frequency neural optimization.

Derives the maximum cell count from mesh resolution (1 node per cell in the
cloak region, accounting for refinement), then sweeps down to 5×5.

Usage::

    python scripts/cell_sweep.py configs/cauchy_tri_multifreq_cell_sweep.yaml

The config's cells.n_x / cells.n_y and loss.multi_freq.f_stars / weights
are overridden by this script.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def derive_max_cells(cfg_data: dict) -> tuple[int, int]:
    """Compute max (n_x, n_y) so that each cell ≈ 1 mesh node in the cloak region.

    The mesh element size follows the same logic as ``rayleigh_cloak.mesh``:
        h_elem = min(W_total / nx_total, H_total / ny_total)
        h_fine = h_elem / refinement_factor

    where nx_total = n_pml_x + nx_phys + n_pml_x,  ny_total = n_pml_y + ny_phys.
    The cloak bbox spans 2*c in x and b in y, so max cells = bbox / h_fine.
    """
    dom = cfg_data.get("domain", {})
    geo = cfg_data.get("geometry", {})
    msh = cfg_data.get("mesh", {})
    ab = cfg_data.get("absorbing", {})

    lambda_star = dom.get("lambda_star", 1.0)
    H_factor = dom.get("H_factor", 4.305)
    W_factor = dom.get("W_factor", 12.5)

    H = H_factor * lambda_star
    W = W_factor * lambda_star
    L_pml = ab.get("L_pml_factor", 1.0) * lambda_star

    W_total = 2 * L_pml + W
    H_total = L_pml + H  # PML only on bottom for triangular

    n_pml_x = msh.get("n_pml_x", 32)
    n_pml_y = msh.get("n_pml_y", 32)
    nx_phys = msh.get("nx_phys", 50)
    ny_phys = msh.get("ny_phys", 30)
    refinement_factor = msh.get("refinement_factor", 3)

    nx_total = n_pml_x + nx_phys + n_pml_x
    ny_total = n_pml_y + ny_phys

    h_elem = min(W_total / nx_total, H_total / ny_total)
    h_fine = h_elem / refinement_factor

    # Cloak geometry
    a_factor = geo.get("a_factor", 0.0774)
    b_factor = geo.get("b_factor", 3.0)
    c_factor = geo.get("c_factor", 0.1545)

    a = a_factor * H
    b = b_factor * a
    c = c_factor * H  # half-width

    bbox_x = 2 * c
    bbox_y = b

    max_nx = int(np.floor(bbox_x / h_fine))
    max_ny = int(np.floor(bbox_y / h_fine))

    return max(max_nx, 5), max(max_ny, 5)


def build_cell_sizes(max_nx: int, max_ny: int, min_n: int = 5,
                     n_steps: int = 10) -> list[tuple[int, int]]:
    """Build a list of (n_x, n_y) from max down to min_n×min_n.

    Uses geometric spacing to produce ~n_steps configurations.
    n_y is scaled by the cloak bbox aspect ratio.
    """
    aspect = max_ny / max_nx

    nx_values = np.geomspace(max_nx, min_n, num=n_steps)
    nx_values = sorted(set(int(round(v)) for v in nx_values), reverse=True)

    sizes = []
    seen = set()
    for nx in nx_values:
        ny = max(min_n, int(round(nx * aspect)))
        pair = (nx, ny)
        if pair not in seen:
            seen.add(pair)
            sizes.append(pair)

    # Ensure (min_n, min_n) is included as the last entry
    if sizes[-1] != (min_n, min_n):
        if (min_n, min_n) not in seen:
            sizes.append((min_n, min_n))

    return sizes


# def make_f_stars_uniform(n: int = 16, f_min: float = 1.0, f_max: float = 3.0):
#     """Return n equally-spaced frequencies and uniform weights."""
#     f_stars = np.linspace(f_min, f_max, n).tolist()
#     weights = [1.0] * n
#     return [round(f, 4) for f in f_stars], weights


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config_path = Path(sys.argv[1])
    with open(config_path) as f:
        cfg_data = yaml.safe_load(f)

    # Derive max cells from mesh
    max_nx, max_ny = derive_max_cells(cfg_data)
    print(f"Max cell resolution (1 node/cell): {max_nx} x {max_ny}")

    # Build sweep list
    sizes = build_cell_sizes(max_nx, max_ny)
    print(f"Sweep: {len(sizes)} configurations")
    for nx, ny in sizes:
        print(f"  {nx} x {ny}")

    # Build uniform f_stars
    # f_stars, weights = make_f_stars_uniform(16)
    # print(f"\nFrequencies ({len(f_stars)}): {f_stars[0]:.2f} .. {f_stars[-1]:.2f}")

    base_output_dir = cfg_data.get("output_dir", "output/cell_sweep")

    for i, (nx, ny) in enumerate(sizes[::-1]):
        run_tag = f"cells_{nx}x{ny}"
        run_output = str(Path(base_output_dir) / run_tag)

        print(f"\n{'='*60}")
        print(f"  [{i+1}/{len(sizes)}] Running {run_tag}")
        print(f"  Output: {run_output}")
        print(f"{'='*60}")

        # Patch config
        run_cfg = dict(cfg_data)
        run_cfg["cells"] = dict(cfg_data.get("cells", {}))
        run_cfg["cells"]["n_x"] = nx
        run_cfg["cells"]["n_y"] = ny

        run_cfg["loss"] = dict(cfg_data.get("loss", {}))
        # if run_cfg["loss"].get("type") == "multi_freq":
        #     run_cfg["loss"]["multi_freq"] = dict(cfg_data["loss"].get("multi_freq", {}))
        #     run_cfg["loss"]["multi_freq"]["f_stars"] = f_stars
        #     run_cfg["loss"]["multi_freq"]["weights"] = weights

        run_cfg["output_dir"] = run_output

        # Write temp config (outside output dir to avoid SameFileError
        # when run_optimize.py copies config into output_dir)
        tmp_dir = Path(base_output_dir) / "_tmp_configs"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_config = tmp_dir / f"{run_tag}.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(run_cfg, f, default_flow_style=False, sort_keys=False)

        # Run optimization with early stopping
        cmd = [sys.executable, "run_optimize.py", str(tmp_config)]
        print(f"  Command: {' '.join(cmd)}")
        early_stopped = False
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        loss_pattern = re.compile(r"New best loss\s+([\d.eE+\-]+)")
        for line in proc.stdout:
            print(line, end="", flush=True)
            m = loss_pattern.search(line)
            if m and float(m.group(1)) < 1e-4:
                print(f"  Early stop: loss {float(m.group(1)):.4e} < 1e-4, terminating.")
                proc.terminate()
                proc.wait()
                early_stopped = True
                break
        if not early_stopped:
            proc.wait()

        if proc.returncode not in (0, -15):  # -15 = SIGTERM (early stop)
            print(f"  WARNING: run {run_tag} exited with code {proc.returncode}")
        else:
            print(f"  Completed {run_tag}" + (" (early stop)" if early_stopped else ""))

    print(f"\n{'='*60}")
    print(f"  Cell sweep complete. Results in {base_output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
