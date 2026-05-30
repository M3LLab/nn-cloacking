"""PyTorch-Lightning module that wraps OccupancyDiffusion + dataloaders."""
from __future__ import annotations

import copy
import os
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from dataset.cellular_chiral.diffusion_dataset import (
    CABulkDiffusionDataset,
    load_or_make_split,
)

from .model import OccupancyDiffusion
from .model_utils import EMA
from ..utils.utils import set_requires_grad, update_moving_average


class DiffusionModel(LightningModule):
    def __init__(
        self,
        h5_path: str = "output/ca_bulk_squared/stiffness.h5",
        scaler_dir: str = "microstructure_generation_2d",
        split_path: str = "microstructure_generation_2d/split.json",
        results_folder: str = "./results",
        val_frac: float = 0.02,
        split_seed: int = 777,
        image_size: int = 64,
        base_channels: int = 64,
        lr: float = 2e-4,
        batch_size: int = 64,
        attention_resolutions: str = "8,4",
        optimizer_name: str = "adam",
        with_attention: bool = True,
        num_heads: int = 4,
        dropout: float = 0.1,
        ema_rate: float = 0.9999,
        verbose: bool = False,
        save_every_epoch: int = 5,
        training_epoch: int = 100,
        gradient_clip_val: float = 1.0,
        use_tensor_condition: bool = True,
        noise_schedule: str = "linear",
        num_workers: int | None = None,
        lr_schedule: str = "none",
        compressed: bool = False,
        dim_mults=None,
        attention_sizes=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.results_folder = Path(results_folder)
        self.compressed = compressed
        self.model = OccupancyDiffusion(
            image_size=image_size,
            base_channels=base_channels,
            attention_resolutions=attention_resolutions,
            attention_sizes=attention_sizes,
            dim_mults=dim_mults,
            with_attention=with_attention,
            dropout=dropout,
            use_tensor_condition=use_tensor_condition,
            num_heads=num_heads,
            noise_schedule=noise_schedule,
            verbose=verbose,
            compressed=compressed,
        )

        self.batch_size = batch_size
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.use_tensor_condition = use_tensor_condition
        self.gradient_clip_val = gradient_clip_val
        self.lr_schedule = lr_schedule
        self.training_epoch = training_epoch
        self.ema_updater = EMA(ema_rate)
        self.ema_model = copy.deepcopy(self.model)
        self.reset_parameters()
        set_requires_grad(self.ema_model, False)

        self.num_workers = (
            num_workers if num_workers is not None else max(1, (os.cpu_count() or 2) // 2)
        )

        self.h5_path = h5_path
        self.scaler_dir = scaler_dir
        self.split_path = Path(split_path)
        self.val_frac = val_frac
        self.split_seed = split_seed
        self._split = None

    # ------------------------------------------------------------------ EMA --
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def update_EMA(self):
        update_moving_average(self.ema_model, self.model, self.ema_updater)

    # ----------------------------------------------------------- optimizers --
    def configure_optimizers(self):
        if self.optimizer_name == "adamw":
            opt = AdamW(self.model.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam":
            opt = Adam(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(self.optimizer_name)
        if self.lr_schedule == "cosine":
            sched = CosineAnnealingLR(opt, T_max=self.training_epoch, eta_min=1e-6)
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
        return opt

    # ---------------------------------------------------------------- data --
    def _split_dict(self) -> dict:
        if self._split is None:
            import h5py
            with h5py.File(self.h5_path, "r") as f:
                n = f["cells"].shape[0]
            self._split = load_or_make_split(
                n=n, val_frac=self.val_frac, seed=self.split_seed, split_path=self.split_path
            )
        return self._split

    def _loader(self, indices, shuffle: bool, cfg_dropout: bool) -> DataLoader:
        ds = CABulkDiffusionDataset(
            h5_path=self.h5_path,
            scaler_dir=self.scaler_dir,
            indices=indices,
            cfg_dropout=cfg_dropout,
            compressed=self.compressed,
        )
        return DataLoader(
            ds, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, drop_last=shuffle,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self._split_dict()["train"], shuffle=True, cfg_dropout=True)

    def val_dataloader(self):
        return self._loader(self._split_dict()["val"], shuffle=False, cfg_dropout=False)

    # -------------------------------------------------------------- steps --
    def training_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        tensor_feature = batch["tensor_feature"] if self.use_tensor_condition else None
        loss = self.model.training_loss(occupancy, tensor_feature).mean()

        self.log("loss", loss.detach().item(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
        opt.step()
        self.update_EMA()

    def validation_step(self, batch, batch_idx):
        occupancy = batch["occupancy"]
        tensor_feature = batch["tensor_feature"] if self.use_tensor_condition else None
        with torch.no_grad():
            loss = self.model.training_loss(occupancy, tensor_feature).mean()
        self.log("val_loss", loss.detach().item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        self.log("current_epoch", float(self.current_epoch), logger=True)
        if self.lr_schedule == "cosine":
            sched = self.lr_schedulers()
            if sched is not None:
                sched.step()
                self.log("lr", sched.get_last_lr()[0], logger=True)
        return super().on_train_epoch_end()
