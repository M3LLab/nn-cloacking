"""Perturbed warm-restart seeds around the best run (E3_kappa99, trans 0.821).

Why perturb instead of just changing `seed`: with `init_weights` set, the loaded
weights overwrite the seed-based init, the Fourier features are deterministic, and
the optimization is deterministic -> different seeds give bit-identical runs. To get
genuine seed diversity around the best basin we add a small seed-controlled Gaussian
kick to the loaded weights (scaled per-layer by that layer's weight std) and RESET
Adam, so each run explores a slightly different neighborhood of E3's optimum.

Keeps E3's exact architecture (512/6/64) + kappa=0.99 + anisotropy_ratio=30 +
output_scale=0.3 so the loaded weights decode to the same material at eps=0.
Scans the kick magnitude eps across seeds to bracket the useful range.
"""
import copy
import numpy as np
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # repo root
SRC = ROOT / "output/A_1x1_ceiling/E3_kappa99/best_weights.npz"
BASE_CFG = HERE / "E3_kappa99.yaml"
INIT_DIR = ROOT / "output/A_1x1_ceiling/_warm_inits"
INIT_DIR.mkdir(parents=True, exist_ok=True)

# name -> (seed, eps)  eps = perturbation std as a fraction of each layer's weight std
RUNS = {
    "warm_s101_e02": (101, 0.02),
    "warm_s202_e05": (202, 0.05),
    "warm_s303_e10": (303, 0.10),
    "warm_s404_e20": (404, 0.20),
}

src = np.load(SRC)
n_layers = int(src["n_layers"])
base_cfg = yaml.safe_load(open(BASE_CFG))

for name, (seed, eps) in RUNS.items():
    rng = np.random.default_rng(seed)
    out = {"n_layers": np.array(n_layers)}  # drop adam_* -> Adam restarts fresh
    moved = []
    for i in range(n_layers):
        for p in ("W", "b"):
            a = src[f"{p}_{i}"].astype(np.float64)
            std = a.std() if a.std() > 0 else 1.0
            ap = a + eps * std * rng.standard_normal(a.shape)
            out[f"{p}_{i}"] = ap
            if p == "W":
                moved.append(np.linalg.norm(ap - a) / (np.linalg.norm(a) + 1e-12))
    init_path = INIT_DIR / f"{name}.npz"
    np.savez(init_path, **out)

    cfg = copy.deepcopy(base_cfg)
    cfg["optimization"]["n_iters"] = 1000
    cfg["optimization"]["neural"]["seed"] = seed
    cfg["optimization"]["neural"]["init_weights"] = str(init_path)
    cfg["output_dir"] = f"output/A_1x1_ceiling/{name}"
    cfg_path = HERE / f"{name}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False, default_flow_style=False)
    print(f"{name}: seed={seed} eps={eps}  mean |dW|/|W|={np.mean(moved):.3f}  "
          f"-> {cfg_path.name}, {init_path.name}")
