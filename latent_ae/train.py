"""Training loop for the frequency-aware autoencoder.

Writes to ``config.out_dir``:
    history.json       — per-epoch losses (train/val totals + components)
    best.pt            — checkpoint of the lowest-val-total epoch
    last.pt            — most recent checkpoint
    epoch-NNNN.pt      — periodic snapshots every ``save_every`` epochs
    config.json        — serialized TrainConfig
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from latent_ae.loss import (
    freq_decorrelation_loss,
    performance_loss,
    ranking_loss,
    reconstruction_loss,
)
from latent_ae.model import LatentAutoencoder, compute_norm_stats
from surrogate.dataset import (
    LossTransform,
    SurrogateDataset,
    collate_surrogate,
)


class TrainConfig(BaseModel):
    data_path: str = "output/surrogate_dataset/surrogate_high_fidelity.h5"
    out_dir: str = "output/latent_ae_train_hi_fi"

    # data
    batch_size: int = 128
    val_fraction: float = 0.15
    num_workers: int = 2
    seed: int = 0
    loss_transform: LossTransform = "log_clip"
    loss_clip_floor: float = 1.0e-8

    # model
    z_dim: int = 128
    fourier_bands: int = 16
    f_min: float = 0.1
    f_max: float = 4.0
    residual_hidden: int = 128
    decoder_hidden_dims: tuple[int, int, int] = (256, 128, 64)
    perf_hidden: int = 128

    # loss weights / ranking hyperparameters
    recon_weight: float = 1.0
    perf_weight: float = 1.0
    rank_weight: float = 0.3
    freq_decorr_weight: float = 0.05
    rho_weight: float = 1.0
    rank_margin: float = 0.5
    rank_gap: float = 0.25
    rank_topk: int = 8
    n_freq_bins: int = 8

    # optimization
    lr: float = 1.0e-4
    weight_decay: float = 1.0e-2
    n_epochs: int = 300

    # checkpointing
    save_every: int = 10
    resume_from: str | None = None


def load_train_config(path: str | Path) -> TrainConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**(data or {}))


_COMPONENT_KEYS = ("total", "rec", "perf", "rank", "freq")


def _fresh_history() -> dict:
    keys = [f"train_{k}" for k in _COMPONENT_KEYS] + [f"val_{k}" for k in _COMPONENT_KEYS]
    hist = {k: [] for k in keys}
    hist["val_perf_r2"] = []
    hist["val_perf_rmse"] = []
    hist["best_val"] = float("inf")
    hist["best_epoch"] = -1
    return hist


def _save_checkpoint(path, *, model, optimizer, epoch, history, cloak_mask) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "cloak_mask": cloak_mask.cpu() if cloak_mask is not None else None,
        },
        path,
    )


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - float(np.sum((y_pred - y_true) ** 2)) / ss_tot)


def _compute_losses(
    model: LatentAutoencoder,
    batch,
    cfg: TrainConfig,
    cloak_mask,
):
    out = model.encode(batch)
    L_rec = reconstruction_loss(out, batch, model, rho_weight=cfg.rho_weight, cloak_mask=cloak_mask)
    L_perf = performance_loss(out, batch)
    L_rank = ranking_loss(
        z_sh=out.z_sh, f_star=batch.f_star, loss_target=batch.loss,
        f_min=cfg.f_min, f_max=cfg.f_max,
        n_freq_bins=cfg.n_freq_bins,
        rank_topk=cfg.rank_topk,
        rank_margin=cfg.rank_margin,
        rank_gap=cfg.rank_gap,
    )
    L_freq = freq_decorrelation_loss(out.z_sh, batch.f_star, model.fourier)
    L_total = (
        cfg.recon_weight * L_rec
        + cfg.perf_weight * L_perf
        + cfg.rank_weight * L_rank
        + cfg.freq_decorr_weight * L_freq
    )
    return L_total, {"rec": L_rec, "perf": L_perf, "rank": L_rank, "freq": L_freq}, out


def _run_epoch(
    model: LatentAutoencoder,
    loader: DataLoader,
    device: str,
    cfg: TrainConfig,
    cloak_mask,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> tuple[dict, np.ndarray, np.ndarray]:
    train = optimizer is not None
    model.train(train)

    sums = {k: 0.0 for k in ("total", "rec", "perf", "rank", "freq")}
    n_samples = 0
    y_true_parts, y_pred_parts = [], []

    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for batch in pbar:
            batch = batch.to(device)
            L_total, comp, out = _compute_losses(model, batch, cfg, cloak_mask)
            if train:
                optimizer.zero_grad()
                L_total.backward()
                optimizer.step()

            bs = len(batch)
            sums["total"] += L_total.item() * bs
            for k, v in comp.items():
                sums[k] += v.item() * bs
            n_samples += bs

            y_true_parts.append(batch.loss.detach().cpu())
            y_pred_parts.append(out.loss_pred.detach().cpu())

            pbar.set_postfix(loss=f"{L_total.item():.3e}")

    means = {k: v / max(n_samples, 1) for k, v in sums.items()}
    y_true = torch.cat(y_true_parts).numpy().astype(np.float64)
    y_pred = torch.cat(y_pred_parts).numpy().astype(np.float64)
    return means, y_true, y_pred


def train(config: TrainConfig) -> tuple[LatentAutoencoder, dict]:
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

    # ── data ──────────────────────────────────────────────────────────
    ds = SurrogateDataset(
        config.data_path,
        loss_transform=config.loss_transform,
        loss_clip_floor=config.loss_clip_floor,
    )
    n_val = max(1, int(len(ds) * config.val_fraction))
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)

    dl_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=collate_surrogate,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **dl_kwargs)

    # ── model ─────────────────────────────────────────────────────────
    n_C_params = ds.C.shape[-1]
    grid_hw = (ds.n_x, ds.n_y)
    norm_stats = compute_norm_stats(ds)
    print("  Normalization stats (from training set, cloak cells):")
    print(f"    C_mean={norm_stats['C_mean'].tolist()}  C_std={norm_stats['C_std'].tolist()}")
    print(f"    rho_mean={float(norm_stats['rho_mean']):.3e}  rho_std={float(norm_stats['rho_std']):.3e}")
    model = LatentAutoencoder(
        n_C_params=n_C_params,
        grid_hw=grid_hw,
        cloak_mask=ds.cloak_mask,
        z_dim=config.z_dim,
        fourier_bands=config.fourier_bands,
        f_min=config.f_min,
        f_max=config.f_max,
        residual_hidden=config.residual_hidden,
        decoder_dims=tuple(config.decoder_hidden_dims),
        perf_hidden=config.perf_hidden,
        **norm_stats,
    ).to(device)

    cloak_mask = ds.cloak_mask.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # ── resume ────────────────────────────────────────────────────────
    start_epoch = 0
    history = _fresh_history()
    if config.resume_from:
        state = torch.load(config.resume_from, map_location="cpu")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history = state["history"]
        for k in (f"train_{k}" for k in _COMPONENT_KEYS):
            history.setdefault(k, [])
        for k in (f"val_{k}" for k in _COMPONENT_KEYS):
            history.setdefault(k, [])
        history.setdefault("val_perf_r2", [])
        history.setdefault("val_perf_rmse", [])
        start_epoch = state["epoch"] + 1
        print(f"Resumed from {config.resume_from} (epoch {state['epoch']})")

    # ── loop ──────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.n_epochs):
        train_m, _, _ = _run_epoch(
            model, train_loader, device, config, cloak_mask,
            optimizer=optimizer, desc=f"epoch {epoch} train",
        )
        val_m, y_true, y_pred = _run_epoch(
            model, val_loader, device, config, cloak_mask,
            optimizer=None, desc=f"epoch {epoch} val",
        )

        val_total = val_m["total"]
        val_perf_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        val_perf_r2 = _r2(y_true, y_pred)

        for k in _COMPONENT_KEYS:
            history[f"train_{k}"].append(train_m[k])
            history[f"val_{k}"].append(val_m[k])
        history["val_perf_r2"].append(val_perf_r2)
        history["val_perf_rmse"].append(val_perf_rmse)

        if val_total < history["best_val"]:
            history["best_val"] = val_total
            history["best_epoch"] = epoch
            _save_checkpoint(
                out_dir / "best.pt",
                model=model, optimizer=optimizer,
                epoch=epoch, history=history, cloak_mask=ds.cloak_mask,
            )

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        _save_checkpoint(
            out_dir / "last.pt",
            model=model, optimizer=optimizer,
            epoch=epoch, history=history, cloak_mask=ds.cloak_mask,
        )
        if (epoch + 1) % config.save_every == 0 or epoch == config.n_epochs - 1:
            _save_checkpoint(
                out_dir / f"epoch-{epoch:04d}.pt",
                model=model, optimizer=optimizer,
                epoch=epoch, history=history, cloak_mask=ds.cloak_mask,
            )

        print(
            f"epoch {epoch:4d}  "
            f"train[tot={train_m['total']:.3e} rec={train_m['rec']:.3e} "
            f"perf={train_m['perf']:.3e} rank={train_m['rank']:.3e} freq={train_m['freq']:.3e}]  "
            f"val[tot={val_m['total']:.3e} perf={val_m['perf']:.3e}]  "
            f"val_perf_R²={val_perf_r2:+.3f}  "
            f"best={history['best_val']:.3e}@{history['best_epoch']}"
        )

    return model, history


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/latent_ae_train.yaml"
    train(load_train_config(path))
