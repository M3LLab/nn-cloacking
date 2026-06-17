"""Training loop for the cloak-loss surrogate.

Writes to ``config.out_dir``:
    history.json       — per-epoch train/val MSE, updated every epoch
    best.pt            — checkpoint of the lowest-val-loss epoch so far
    last.pt            — most recent checkpoint (easy resume target)
    epoch-NNNN.pt      — periodic snapshots, every ``save_every`` epochs
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from surrogate.dataset import LossTransform, SurrogateDataset, collate_surrogate
from surrogate.model import ForwardFEM_CNN


class TrainConfig(BaseModel):
    data_path: str = "output/surrogate_dataset.h5"
    out_dir: str = "output/surrogate"

    # data
    batch_size: int = 64
    val_fraction: float = 0.1
    num_workers: int = 2
    seed: int = 0
    loss_transform: LossTransform = "none"
    loss_clip_floor: float = 1e-8

    # model
    z_dim: int = 128
    fourier_bands: int = 8
    decoder_hidden: int = 128
    f_min: float = 0.1
    f_max: float = 4.0

    # optimization
    lr: float = 1e-4
    weight_decay: float = 1e-5
    n_epochs: int = 100

    # checkpointing
    save_every: int = 10
    resume_from: str | None = None


def load_train_config(path: str | Path) -> TrainConfig:
    """Load a ``TrainConfig`` from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return TrainConfig(**(data or {}))


def _save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        },
        path,
    )


def _load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict]:
    state = torch.load(path, map_location="cpu")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["epoch"], state["history"]


_VAL_METRIC_KEYS = (
    "val_rmse", "val_r2",
    "val_mse_log", "val_rmse_log", "val_r2_log",
    "val_r2_bot20", "val_r2_log_bot20",
)


def _fresh_history() -> dict:
    return {
        "train": [], "val": [],
        **{k: [] for k in _VAL_METRIC_KEYS},
        "best_val": float("inf"), "best_epoch": -1,
    }


def _to_log_space(y: np.ndarray, transform: str, floor: float = 1e-8) -> np.ndarray:
    if transform == "log_clip":
        return y
    if transform == "none":
        return np.log(np.clip(y, floor, None))
    raise ValueError(f"Unknown loss_transform: {transform!r}")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - float(np.sum((y_pred - y_true) ** 2)) / ss_tot)


@torch.no_grad()
def _compute_val_metrics(
    model: ForwardFEM_CNN,
    loader: DataLoader,
    device: str,
    *,
    loss_transform: str,
    desc: str,
) -> dict:
    """One val pass: MSE (training objective) plus R²/RMSE in native and log space,
    plus R² restricted to the bottom 20% of true loss."""
    model.eval()
    y_true_parts, y_pred_parts = [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        batch = batch.to(device)
        pred = model.forward_at(batch).predicted_cloaking
        y_true_parts.append(batch.loss.detach().cpu())
        y_pred_parts.append(pred.detach().cpu())
        pbar.set_postfix(loss=f"{F.mse_loss(pred, batch.loss).item():.3e}")
    y_true = torch.cat(y_true_parts).numpy().astype(np.float64)
    y_pred = torch.cat(y_pred_parts).numpy().astype(np.float64)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    r2 = _r2(y_true, y_pred)

    yt_log = _to_log_space(y_true, loss_transform)
    yp_log = _to_log_space(y_pred, loss_transform)
    mse_log = float(np.mean((yp_log - yt_log) ** 2))
    rmse_log = float(np.sqrt(mse_log))
    r2_log = _r2(yt_log, yp_log)

    # Ordering is preserved by log_clip, so quantile on stored y_true is fine.
    sel = y_true <= float(np.quantile(y_true, 0.20))
    r2_bot20 = _r2(y_true[sel], y_pred[sel])
    r2_log_bot20 = _r2(yt_log[sel], yp_log[sel])

    return {
        "mse": mse, "rmse": rmse, "r2": r2,
        "mse_log": mse_log, "rmse_log": rmse_log, "r2_log": r2_log,
        "r2_bot20": r2_bot20, "r2_log_bot20": r2_log_bot20,
    }


def _run_epoch(
    model: ForwardFEM_CNN,
    loader: DataLoader,
    device: str,
    *,
    optimizer: torch.optim.Optimizer | None,
    desc: str,
) -> float:
    """Returns the sample-weighted mean MSE over the loader."""
    train = optimizer is not None
    model.train(train)
    loss_sum, n_samples = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for batch in pbar:
            batch = batch.to(device)
            pred = model.forward_at(batch).predicted_cloaking
            loss = F.mse_loss(pred, batch.loss)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = len(batch)
            loss_sum += loss.item() * bs
            n_samples += bs
            pbar.set_postfix(loss=f"{loss.item():.3e}")
    return loss_sum / max(n_samples, 1)


def train(config: TrainConfig) -> tuple[ForwardFEM_CNN, dict]:
    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))

    # --- data ---------------------------------------------------------------
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

    # --- model --------------------------------------------------------------
    n_C_params = ds.C.shape[-1]
    model = ForwardFEM_CNN(
        n_C_params=n_C_params,
        z_dim=config.z_dim,
        fourier_bands=config.fourier_bands,
        decoder_hidden=config.decoder_hidden,
        f_min=config.f_min,
        f_max=config.f_max,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # --- resume -------------------------------------------------------------
    start_epoch = 0
    history = _fresh_history()
    if config.resume_from:
        last_epoch, history = _load_checkpoint(
            config.resume_from, model=model, optimizer=optimizer,
        )
        # backfill keys added in later versions so list lengths stay aligned
        for k in _VAL_METRIC_KEYS:
            history.setdefault(k, [])
        start_epoch = last_epoch + 1
        print(f"Resumed from {config.resume_from} (epoch {last_epoch})")

    # --- loop ---------------------------------------------------------------
    for epoch in range(start_epoch, config.n_epochs):
        train_loss = _run_epoch(model, train_loader, device,
                                optimizer=optimizer, desc=f"epoch {epoch} train")
        val_m = _compute_val_metrics(
            model, val_loader, device,
            loss_transform=ds.loss_transform, desc=f"epoch {epoch} val",
        )
        val_loss = val_m["mse"]

        history["train"].append(train_loss)
        history["val"].append(val_loss)
        history["val_rmse"].append(val_m["rmse"])
        history["val_r2"].append(val_m["r2"])
        history["val_mse_log"].append(val_m["mse_log"])
        history["val_rmse_log"].append(val_m["rmse_log"])
        history["val_r2_log"].append(val_m["r2_log"])
        history["val_r2_bot20"].append(val_m["r2_bot20"])
        history["val_r2_log_bot20"].append(val_m["r2_log_bot20"])
        if val_loss < history["best_val"]:
            history["best_val"] = val_loss
            history["best_epoch"] = epoch
            _save_checkpoint(out_dir / "best.pt", model=model, optimizer=optimizer,
                             epoch=epoch, history=history)

        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        _save_checkpoint(out_dir / "last.pt", model=model, optimizer=optimizer,
                         epoch=epoch, history=history)
        if (epoch + 1) % config.save_every == 0 or epoch == config.n_epochs - 1:
            _save_checkpoint(out_dir / f"epoch-{epoch:04d}.pt",
                             model=model, optimizer=optimizer,
                             epoch=epoch, history=history)

        print(
            f"epoch {epoch:4d}  train={train_loss:.4e}  val={val_loss:.4e}"
            f"  R²={val_m['r2']:+.3f}  R²_log={val_m['r2_log']:+.3f}"
            f"  R²_bot20={val_m['r2_bot20']:+.3f}  R²_log_bot20={val_m['r2_log_bot20']:+.3f}"
            f"  best={history['best_val']:.4e}@{history['best_epoch']}"
        )

    return model, history


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "configs/surrogate_train.yaml"
    train(load_train_config(path))
