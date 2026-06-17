"""Evaluation helpers for the frequency-aware autoencoder.

Primary entry points:
    load_model(checkpoint_path)          — rebuild model from a saved checkpoint
    encode_dataset(model, dataset)       — run the encoder over all samples
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from latent_ae.model import LatentAutoencoder
from surrogate.dataset import (
    SurrogateDataset,
    collate_surrogate,
    invert_loss_transform,
)


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[LatentAutoencoder, dict]:
    """Load a LatentAutoencoder from a training checkpoint.

    Reads ``config.json`` next to the checkpoint to recover the architecture.
    Returns (model, config_dict).
    """
    ckpt_path = Path(checkpoint_path)
    cfg_path = ckpt_path.parent / "config.json"
    cfg = json.loads(cfg_path.read_text())

    state = torch.load(ckpt_path, map_location=device)
    cloak_mask = state.get("cloak_mask", None)

    # Infer n_C_params from the state dict (stem input channels = P + 1)
    stem_weight = state["model"]["trunk.stem.0.weight"]
    in_channels = stem_weight.shape[1]
    n_C_params = in_channels - 1

    # Infer grid from cloak_mask if present, else fall back to dataset.
    if cloak_mask is not None:
        grid_hw = tuple(cloak_mask.shape)
    else:
        # Read once from the dataset to get dims
        ds = SurrogateDataset(cfg["data_path"])
        grid_hw = (ds.n_x, ds.n_y)
        cloak_mask = ds.cloak_mask

    # Pull normalization buffers from the saved state_dict so model init matches.
    model = LatentAutoencoder(
        n_C_params=n_C_params,
        grid_hw=grid_hw,
        cloak_mask=cloak_mask,
        z_dim=cfg["z_dim"],
        fourier_bands=cfg["fourier_bands"],
        f_min=cfg["f_min"],
        f_max=cfg["f_max"],
        residual_hidden=cfg["residual_hidden"],
        decoder_dims=tuple(cfg["decoder_hidden_dims"]),
        perf_hidden=cfg["perf_hidden"],
        C_mean=state["model"].get("C_mean"),
        C_std=state["model"].get("C_std"),
        rho_mean=state["model"].get("rho_mean"),
        rho_std=state["model"].get("rho_std"),
    )
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def encode_dataset(
    model: LatentAutoencoder,
    dataset: SurrogateDataset,
    batch_size: int = 128,
    device: str = "cpu",
) -> dict:
    """Run the encoder on every sample. Returns numpy arrays keyed by field.

    Keys: z_sh, z_f, z_f_shift, z, loss_pred, f_star, loss (inverted), sample_type.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_surrogate,
    )
    z_sh_p, z_f_p, z_shift_p, z_p, lp_p = [], [], [], [], []
    for batch in loader:
        batch = batch.to(device)
        out = model.encode(batch)
        z_sh_p.append(out.z_sh.cpu().numpy())
        z_f_p.append(out.z_f.cpu().numpy())
        z_shift_p.append(out.z_f_shift.cpu().numpy())
        z_p.append(out.z.cpu().numpy())
        lp_p.append(out.loss_pred.cpu().numpy())

    loss_raw = invert_loss_transform(
        dataset.loss.numpy().astype(np.float64),
        transform=dataset.loss_transform,
    )

    return {
        "z_sh":        np.concatenate(z_sh_p, axis=0),
        "z_f":         np.concatenate(z_f_p, axis=0),
        "z_f_shift":   np.concatenate(z_shift_p, axis=0),
        "z":           np.concatenate(z_p, axis=0),
        "loss_pred":   np.concatenate(lp_p, axis=0),
        "f_star":      dataset.f_star.numpy(),
        "loss":        loss_raw,
        "loss_stored": dataset.loss.numpy(),
        "sample_type": dataset.sample_type,
    }
