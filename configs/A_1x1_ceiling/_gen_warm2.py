"""Round-2 low-eps warm restarts, chained from the current frontier.

warm_s101_e02 (eps=0.02 restart from E3) lifted 0.821 -> 0.850. This repeats that
exact move from the NEW best (warm_s101_e02's 0.850 weights): small seed-controlled
weight kicks in 5 directions, Adam reset, same constrained config (R=30 capped,
kappa=0.99, output_scale=0.3, lr=1e-3, net 512/6/64). Scans the low-eps band
(0.01-0.03) since eps=0.02 was the productive scale and larger kicks drained back
to the source basin.

Per-step C+rho logging is automatic for 1x1 runs (run_optimize.py).
"""
import copy
import numpy as np
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "output/A_1x1_ceiling/warm_s101_e02/best_weights.npz"  # the 0.850 frontier
BASE_CFG = HERE / "E3_kappa99.yaml"   # same freedom config warm_s101_e02 was trained under
INIT_DIR = ROOT / "output/A_1x1_ceiling/_warm_inits"
INIT_DIR.mkdir(parents=True, exist_ok=True)

# name -> (seed, eps)  eps = kick std as a fraction of each layer's weight std
RUNS = {
    "warm2_s11_e010": (11, 0.010),
    "warm2_s12_e015": (12, 0.015),
    "warm2_s13_e020": (13, 0.020),
    "warm2_s14_e025": (14, 0.025),
    "warm2_s15_e030": (15, 0.030),
}

src = np.load(SRC)
n_layers = int(src["n_layers"])
base_cfg = yaml.safe_load(open(BASE_CFG))

for name, (seed, eps) in RUNS.items():
    rng = np.random.default_rng(seed)
    out = {"n_layers": np.array(n_layers)}  # drop adam_* -> fresh Adam
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
    print(f"{name}: seed={seed} eps={eps}  mean|dW|/|W|={np.mean(moved):.3f}  -> {cfg_path.name}")
