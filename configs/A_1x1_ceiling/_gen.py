"""Generate 5 single-tile (1x1 cell) configs that push the transmission ceiling.

All share the EXACT mesh / loss / init of the user's running 1x1 experiments so
results are comparable. They keep n_x=n_y=1, n_C_params=4, method=neural,
constrained=True (PD-safe, realizable material). The sweep varies only the knobs
that set the ceiling for a single homogeneous anisotropic block:
  - anisotropy cap (cap_anisotropy / anisotropy_ratio)  -> diagonal C11/C22 reach
  - kappa                                               -> shear/normal coupling C12 reach
  - output_scale                                        -> material reach per unit MLP output
  - lr (+ fresh start, no init_weights)                 -> escape the ~0.756 plateau
  - one tiny-net variant                                -> kill the weight-ballooning pathology
"""
import copy
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_ROOT = "output/A_1x1_ceiling"

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
        "method": "neural", "n_iters": 1500,
        "lr": 1.0e-3, "lr_end": 1.0e-06, "lr_schedule": "cosine",
        "lambda_l2": 0.0, "lambda_neighbor": 0.0, "plot_every": 0,
        "neural": {
            "hidden_size": 512, "n_layers": 6, "n_fourier": 64, "seed": 42,
            "output_scale": 0.3, "constrained": True, "kappa": 0.95,
            "cap_anisotropy": True, "anisotropy_ratio": 30.0,
        },
    },
}

# name -> (lr, hidden, layers, fourier, seed, output_scale, kappa, cap, ratio)
MATRIX = {
    # capped @30, default reach/coupling, higher LR + fresh start
    "E1_aniso30":  (1.0e-3, 512, 6, 64,  42, 0.3, 0.95, True,  30.0),
    # cap OFF: unbounded (but PD) diagonal anisotropy
    "E2_anisoOff": (1.0e-3, 512, 6, 64,  42, 0.3, 0.95, False, 30.0),
    # stronger shear/normal coupling (kappa 0.95 -> 0.99)
    "E3_kappa99":  (1.0e-3, 512, 6, 64,  42, 0.3, 0.99, True,  30.0),
    # max material freedom: cap off, big reach, strong coupling, higher LR
    "E4_maxreach": (1.5e-3, 512, 6, 64,   7, 0.5, 0.99, False, 30.0),
    # same freedom but a tiny net (no weight-ballooning) + high LR + new seed
    "E5_tinynet":  (2.0e-3,  64, 3, 16, 123, 0.5, 0.99, False, 30.0),
}

for name, (lr, hid, lay, fou, seed, osc, kappa, cap, ratio) in MATRIX.items():
    cfg = copy.deepcopy(BASE)
    o = cfg["optimization"]
    o["lr"] = lr
    n = o["neural"]
    n["hidden_size"] = hid
    n["n_layers"] = lay
    n["n_fourier"] = fou
    n["seed"] = seed
    n["output_scale"] = osc
    n["kappa"] = kappa
    n["cap_anisotropy"] = cap
    n["anisotropy_ratio"] = ratio
    cfg["output_dir"] = f"{OUT_ROOT}/{name}"
    path = HERE / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f"wrote {path}")
