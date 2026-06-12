"""5 fresh-seed basin-sampling runs for the 1x1 single-tile cloak.

Goal: sample more local optima ("basins") of the (degenerate, multimodal)
single-tile objective so they can be visualised in C-tensor space afterwards.

Design: the basin a run lands in is selected by the random MLP init (seed) at a
fixed freedom setting -- seeds 7/42/123 already landed in qualitatively different
basins. So we hold the *most productive* freedom config fixed (= E3_kappa99:
anisotropy_ratio=30 capped, kappa=0.99, output_scale=0.3, lr=1e-3, net 512/6/64
-- the lineage that produced the 0.82-0.85 family) and sweep only the seed, all
FRESH (no init_weights). One variable -> clean attribution.

Per-step C-tensor + rho logging is handled by run_optimize.py automatically for
1x1 runs (loss_history.csv gains C11,C22,C66,C12,rho columns), so each run's full
trajectory into its basin is recorded.

Creates configs only -- does NOT launch anything.
"""
import copy
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = "output/A_1x1_ceiling"

# fresh seeds, none reused from earlier runs (those used 7/42/123)
SEEDS = [3, 19, 37, 64, 88]

BASE = {
    "is_reference": False,
    "geometry_type": "triangular",
    "material": {"rho0": 1600.0, "cs": 300.0},
    "domain": {"f_star": 2.0, "lambda_star": 1.0, "H_factor": 4.305, "W_factor": 12.5},
    "geometry": {"a_factor": 0.0774, "b_factor": 3.0, "c_factor": 0.1545},
    "absorbing": {"L_pml_factor": 1.0, "xi_max": 4.0, "pml_pow": 2},
    "mesh": {
        "n_pml_x": 32, "n_pml_y": 32, "nx_phys": 60, "ny_phys": 40,
        "refinement_factor": 2, "ele_type": "TRI6", "builder": "uniform_tri6",
        "embed_macro_grid": False, "refinement_factor_outside": 0.5,
    },
    "source": {"x_src_factor": 0.05, "sigma_factor": 0.01, "F0": 1.0},
    "solver": {"ksp_type": "preonly", "pc_type": "lu",
               "pc_factor_mat_solver_type": "mumps"},
    "cells": {
        "enabled": True, "init": "dataset_centroid",
        "init_path": "dataset/gmm_flat4.npz",
        "n_x": 1, "n_y": 1, "n_C_params": 4,
        "symmetrize_init": False, "confine_to_cloak": True,
    },
    "loss": {
        "type": "magnitude_band_integral", "depth": 0.5,
        "regularizations": {
            "material_cement_GMM": {
                "enabled": False, "path": "dataset/gmm_flat4.npz",
                "weight": 0.0, "quantile": 0.25,
            }
        },
    },
    "optimization": {
        "method": "neural", "n_iters": 1000,
        "lr": 1.0e-3, "lr_end": 1.0e-06, "lr_schedule": "cosine",
        "lambda_l2": 0.0, "lambda_neighbor": 0.0, "plot_every": 0,
        "neural": {
            "hidden_size": 512, "n_layers": 6, "n_fourier": 64, "seed": 42,
            "output_scale": 0.3, "constrained": True, "kappa": 0.99,
            "cap_anisotropy": True, "anisotropy_ratio": 30.0,
        },
    },
}

for seed in SEEDS:
    cfg = copy.deepcopy(BASE)
    cfg["optimization"]["neural"]["seed"] = seed
    name = f"basin_s{seed}"
    cfg["output_dir"] = f"{OUT_ROOT}/{name}"
    path = HERE / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}  (seed={seed} -> {cfg['output_dir']})")
