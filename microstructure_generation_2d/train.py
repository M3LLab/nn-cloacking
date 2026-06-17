"""Train the 2D conditional diffusion model from a YAML config.

Usage:

    python -m microstructure_generation_2d.train --config configs/diffusion_ca_squared.yaml
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from .network.model_trainer import DiffusionModel
from .utils.utils import ensure_directory, find_best_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default="configs/diffusion_ca_squared.yaml",
        help="Path to YAML training config",
    )
    parser.add_argument("--max-epochs-override", type=int, default=None)
    parser.add_argument("--limit-train-batches", type=float, default=None)
    parser.add_argument("--limit-val-batches", type=float, default=None)
    parser.add_argument("--devices", default=None,
                        help="passed to Lightning Trainer (e.g. '1', '-1', 'auto')")
    parser.add_argument("--accelerator", default=None,
                        help="passed to Lightning Trainer (e.g. 'gpu', 'cpu', 'auto')")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    name = cfg.get("name", "v1")
    results_folder = os.path.join(cfg["results_folder"], name)
    ensure_directory(results_folder)
    cfg["results_folder"] = results_folder

    seed_everything(cfg.get("seed", 777))

    model_kwargs = dict(
        h5_path=cfg["h5_path"],
        scaler_dir=cfg["scaler_dir"],
        split_path=cfg["split_path"],
        results_folder=results_folder,
        val_frac=cfg.get("val_frac", 0.02),
        split_seed=cfg.get("split_seed", 777),
        image_size=cfg.get("image_size", 64),
        base_channels=cfg.get("base_channels", 64),
        attention_resolutions=cfg.get("attention_resolutions", "8,4"),
        attention_sizes=cfg.get("attention_sizes", None),
        dim_mults=cfg.get("dim_mults", None),
        with_attention=cfg.get("with_attention", True),
        num_heads=cfg.get("num_heads", 4),
        dropout=cfg.get("dropout", 0.1),
        use_tensor_condition=cfg.get("use_tensor_condition", True),
        noise_schedule=cfg.get("noise_schedule", "linear"),
        lr_schedule=cfg.get("lr_schedule", "none"),
        batch_size=cfg.get("batch_size", 64),
        lr=cfg.get("lr", 2e-4),
        optimizer_name=cfg.get("optimizer_name", "adam"),
        ema_rate=cfg.get("ema_rate", 0.9999),
        training_epoch=cfg.get("training_epoch", 100),
        gradient_clip_val=cfg.get("gradient_clip_val", 1.0),
        save_every_epoch=cfg.get("save_every_epoch", 5),
        num_workers=cfg.get("num_workers", None),
        compressed=cfg.get("compressed", False),
        num_res_blocks=cfg.get("num_res_blocks", 1),
        parameterization=cfg.get("parameterization", "x0"),
        min_snr_gamma=cfg.get("min_snr_gamma", 0.0),
    )
    model = DiffusionModel(**model_kwargs)

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=results_folder, name="logs", default_hp_metric=False
    )

    checkpoint_cb = ModelCheckpoint(
        monitor="current_epoch",
        dirpath=results_folder,
        filename="{epoch:02d}",
        save_top_k=10,
        save_last=cfg.get("save_last", True),
        every_n_epochs=cfg.get("save_every_epoch", 5),
        mode="max",
    )
    best_ckpt_cb = ModelCheckpoint(
        monitor="val_loss",
        dirpath=results_folder,
        filename="best",
        save_top_k=1,
        mode="min",
    )

    max_epochs = args.max_epochs_override or cfg.get("training_epoch", 100)
    accelerator = args.accelerator or ("gpu" if _has_gpu() else "cpu")
    devices = args.devices or (1 if accelerator == "gpu" else 1)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        logger=tb_logger,
        log_every_n_steps=10,
        callbacks=[checkpoint_cb, best_ckpt_cb],
        precision=cfg.get("precision", "32-true"),
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
    )

    last_ckpt = None
    if cfg.get("continue_training", False):
        last = Path(results_folder) / "last.ckpt"
        if last.exists():
            last_ckpt = str(last)
        else:
            best_epoch = find_best_epoch(results_folder)
            if best_epoch:
                cand = Path(results_folder) / f"epoch={best_epoch:02d}.ckpt"
                if cand.exists():
                    last_ckpt = str(cand)

    trainer.fit(model, ckpt_path=last_ckpt)


def _has_gpu() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


if __name__ == "__main__":
    main()
