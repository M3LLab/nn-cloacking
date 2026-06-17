"""Multi-frequency latent-space optimization.

Freeze a trained ``LatentAutoencoder`` and optimize a single latent vector ``z``
against ``α·L_minmax + (1−α)·L_mean`` over a frequency bandwidth evaluated by
the JAX FEM solver (via ``latent_ae.fem_bridge``). ``α`` follows an optional
schedule (none | linear | cosine) from ``alpha_start`` to ``alpha_end``; LR
follows the same schedule types.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import logging
import contextlib
import io

import numpy as np
import torch
import yaml
from pydantic import BaseModel
from tqdm import tqdm

from latent_ae.evaluate import load_model
with contextlib.redirect_stdout(io.StringIO()):
    from latent_ae.fem_bridge import FEMContext, MultiFreqFEMLoss, build_freq_targets
    from latent_ae.model import LatentAutoencoder
    from rayleigh_cloak.config import load_config
from surrogate.dataset import (
    SurrogateBatch,
    SurrogateDataset,
    SurrogateSample,
    collate_surrogate,
)

Schedule = str  # "none" | "linear" | "cosine"

logging.getLogger("jax_fem").setLevel(logging.ERROR)


class OptimizeConfig(BaseModel):
    # paths
    ae_checkpoint: str = "output/latent_ae_train_hi_fi/best.pt"
    dataset_path: str = "output/surrogate_dataset/surrogate_high_fidelity.h5"
    out_dir: str = "output/latent_ae_optimize_demo"
    fem_base_config: str = "configs/cauchy_tri.yaml"

    # frequency bandwidth
    f_min: float = 1.5
    f_max: float = 2.5
    f_step: float = 0.25
    max_workers: int = 0        # 0 -> len(freqs)

    # loss blending
    alpha_start: float = 1.0
    alpha_end: float = 0.0
    alpha_schedule: Schedule = "cosine"

    # initialization
    init_strategy: str = "barycenter_of_best_per_f"
    barycenter_nearest_f_tol: float = 0.1

    # optimization
    n_iters: int = 500
    lr: float = 5.0e-3
    lr_schedule: Schedule = "cosine"
    lr_end: float = 5.0e-5
    optimizer: str = "adam"

    # material validity penalty: weight * mean(relu(floor - x)^2) over cloak
    # cells, applied to rho and to diagonal C-flat entries (λ/μ-like).
    material_weight: float = 0.0
    material_floor: float = 0.0

    # bookkeeping
    plot_every: int = 25
    save_every: int = 25
    seed: int = 0


def load_optimize_config(path: str | Path) -> OptimizeConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return OptimizeConfig(**(data or {}))


# Indices of "diagonal" entries of the flat C parameterization that must
# stay positive for a physically-admissible stiffness (normal/shear diagonals).
# Off-diagonal couplings may legitimately be negative and are not penalized.
_C_DIAG_INDICES: dict[int, list[int]] = {
    2:  [0, 1],         # [λ, μ]
    6:  [0, 1, 3, 4],   # normal-xx, normal-yy, shear-diag, shear-diag
    10: [0, 4, 7, 9],   # diagonal of symmetric 4×4 Voigt
    16: [0, 5, 10, 15], # diagonal of full 4×4 Voigt
}


def _material_penalty(
    C: torch.Tensor,        # (X, Y, P)
    rho: torch.Tensor,      # (X, Y)
    cloak_mask: torch.Tensor | None,
    floor: float,
) -> torch.Tensor:
    n_P = C.shape[-1]
    if n_P not in _C_DIAG_INDICES:
        raise ValueError(f"material_weight>0 unsupported for n_C_params={n_P}")
    C_diag = C[..., _C_DIAG_INDICES[n_P]]                    # (X, Y, k)
    rho_viol = torch.relu(floor - rho).pow(2)                # (X, Y)
    C_viol = torch.relu(floor - C_diag).pow(2)               # (X, Y, k)
    if cloak_mask is not None:
        m = cloak_mask.to(rho.device, dtype=rho.dtype)
        denom = m.sum().clamp_min(1.0)
        return rho_viol.mul(m).sum() / denom \
             + C_viol.mul(m[..., None]).sum() / (denom * C_viol.shape[-1])
    return rho_viol.mean() + C_viol.mean()


def _schedule(kind: Schedule, start: float, end: float, t: int, T: int) -> float:
    if kind == "none":
        return start
    frac = t / max(T - 1, 1)
    if kind == "linear":
        return start + (end - start) * frac
    if kind == "cosine":
        return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * frac))
    raise ValueError(f"Unknown schedule: {kind!r}")


def _build_frequencies(f_min: float, f_max: float, f_step: float) -> list[float]:
    """Half-open arange with tolerance for the end, matches rayleigh_cloak style."""
    return [float(f) for f in np.arange(f_min, f_max + 0.5 * f_step, f_step)]


def _best_sample_per_frequency(
    ds: SurrogateDataset,
    f_stars: list[float],
    tol: float,
) -> list[int]:
    """Pick the lowest-stored-loss dataset index with f_star within ±tol of each target f."""
    f_all = ds.f_star.numpy()
    loss_all = ds.loss.numpy()   # transformed; lower = better regardless of transform
    chosen = []
    for f in f_stars:
        near = np.where(np.abs(f_all - f) <= tol)[0]
        if len(near) == 0:
            raise ValueError(
                f"No dataset sample within {tol} of f={f}. "
                f"Try loosening barycenter_nearest_f_tol."
            )
        best = near[int(np.argmin(loss_all[near]))]
        chosen.append(int(best))
    return chosen


def _encode_barycenter(
    model: LatentAutoencoder,
    ds: SurrogateDataset,
    indices: list[int],
    device: str,
) -> torch.Tensor:
    """Encode the listed samples, take z_sh only, average to a single latent."""
    samples = [ds[i] for i in indices]
    batch = collate_surrogate(samples).to(device)
    with torch.no_grad():
        z_sh = model.encode_shared(batch)            # (len(indices), D)
    return z_sh.mean(dim=0, keepdim=True)             # (1, D)


def optimize(config: OptimizeConfig) -> dict:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

    # ── AE load ───────────────────────────────────────────────────────
    model, ae_cfg = load_model(config.ae_checkpoint, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # ── dataset (for barycenter init) ─────────────────────────────────
    ds = SurrogateDataset(
        config.dataset_path,
        loss_transform=ae_cfg["loss_transform"],
        loss_clip_floor=ae_cfg["loss_clip_floor"],
    )

    # ── frequency grid ────────────────────────────────────────────────
    f_stars = _build_frequencies(config.f_min, config.f_max, config.f_step)
    print(f"Optimizing over {len(f_stars)} frequencies: {f_stars}")

    # ── latent init: barycenter of best-per-f ─────────────────────────
    print("Selecting best-per-f dataset samples ...")
    best_idx = _best_sample_per_frequency(ds, f_stars, config.barycenter_nearest_f_tol)
    for f, i in zip(f_stars, best_idx):
        print(f"  f={f:.3f}: idx={i} f_star={float(ds.f_star[i]):.3f} "
              f"loss(stored)={float(ds.loss[i]):.3e} type={ds.sample_type[i]}")

    z = _encode_barycenter(model, ds, best_idx, device)
    z = z.detach().clone().requires_grad_(True)
    print(f"z_init shape: {tuple(z.shape)}, norm={float(z.norm()):.3f}")

    # ── FEM context ───────────────────────────────────────────────────
    print(f"Building FEM context from {config.fem_base_config} ...")
    base_cfg = load_config(config.fem_base_config)
    fem_ctx: FEMContext = build_freq_targets(
        base_cfg, f_stars, max_workers=config.max_workers,
    )

    # ── optimizer ─────────────────────────────────────────────────────
    if config.optimizer == "adam":
        opt = torch.optim.Adam([z], lr=config.lr)
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer!r}")

    history: dict = {
        "per_freq_loss": [],         # list of lists (n_iters × n_freq)
        "minmax": [],
        "mean": [],
        "blended": [],
        "material": [],
        "alpha": [],
        "lr": [],
        "f_stars": f_stars,
        "best_iter": -1,
        "best_minmax": float("inf"),
        "best_blended": float("inf"),
    }

    try:
        for step in tqdm(range(config.n_iters), desc="latent-opt"):
            alpha = _schedule(config.alpha_schedule, config.alpha_start, config.alpha_end,
                              step, config.n_iters)
            cur_lr = _schedule(config.lr_schedule, config.lr, config.lr_end,
                               step, config.n_iters)
            for g in opt.param_groups:
                g["lr"] = cur_lr

            C_grid, rho_grid = model.decode_physical(z)     # (1, X, Y, P), (1, X, Y)
            per_f = MultiFreqFEMLoss.apply(C_grid[0], rho_grid[0], fem_ctx)

            L_minmax = per_f.max()
            L_mean = per_f.mean()
            L_fem = alpha * L_minmax + (1.0 - alpha) * L_mean
            if config.material_weight > 0.0:
                L_mat = _material_penalty(
                    C_grid[0], rho_grid[0], model.cloak_mask, config.material_floor,
                )
            else:
                L_mat = torch.zeros((), device=C_grid.device)
            L = L_fem + config.material_weight * L_mat

            # Snapshot pre-step state so saved params match the saved loss.
            z_snapshot = z.detach().cpu().clone()
            C_snapshot = C_grid[0].detach().cpu().clone()
            rho_snapshot = rho_grid[0].detach().cpu().clone()

            opt.zero_grad()
            L.backward()
            opt.step()

            per_f_np = per_f.detach().cpu().numpy().tolist()
            history["per_freq_loss"].append(per_f_np)
            history["minmax"].append(float(L_minmax.item()))
            history["mean"].append(float(L_mean.item()))
            history["blended"].append(float(L.item()))
            history["material"].append(float(L_mat.item()))
            history["alpha"].append(float(alpha))
            history["lr"].append(float(cur_lr))

            if float(L.item()) < history["best_blended"]:
                history["best_blended"] = float(L.item())
                history["best_minmax"] = float(L_minmax.item())
                history["best_iter"] = step
                torch.save({
                    "iter": step,
                    "z": z_snapshot,
                    "per_freq_loss": per_f_np,
                    "f_stars": f_stars,
                }, out_dir / "best_z.pt")

            if (step + 1) % config.save_every == 0 or step == config.n_iters - 1:
                np.savez(
                    out_dir / f"iter-{step:04d}.npz",
                    z=z_snapshot.numpy(),
                    C=C_snapshot.numpy(),
                    rho=rho_snapshot.numpy(),
                    per_freq_loss=np.array(per_f_np),
                    f_stars=np.array(f_stars),
                )

            per_f_str = " ".join(f"{f:.2f}:{l:.2e}" for f, l in zip(f_stars, per_f_np))
            print(
                f"  step {step:4d} α={alpha:.3f} lr={cur_lr:.2e}  "
                f"minmax={L_minmax.item():.4e} mean={L_mean.item():.4e} "
                f"mat={L_mat.item():.4e} blend={L.item():.4e}  [{per_f_str}]"
            )

            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

    finally:
        fem_ctx.shutdown()

    return history


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/latent_ae_optimize.yaml"
    optimize(load_optimize_config(path))
