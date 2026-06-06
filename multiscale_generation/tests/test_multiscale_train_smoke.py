"""End-to-end smoke for the multiscale NF training step (``predict_structure``).

Runs one (or ``MULTISCALE_STEPS``) trajectory-coupled optimisation step through the
full torch<->JAX bridge with the *real* pieces:

* diffusion checkpoint ``output/diffusion_ca_squared/v1_ortho/last.ckpt`` (5-dim
  ortho conditioning),
* the flat4 (``n_C_params=4``) cloak config + warm-start MLP weights from the
  ``cell20_flat4_materialreg`` run.

This exercises the chain conditions(θ) → diffusion step → decode+tile → pixel FEM →
loss → VJP → Adam(θ). It runs the full-structure FEM once per diffusion step, so it
is heavy — intended for a GPU box. It **skips** if the checkpoint / config / weights
are absent. Tunables via env: ``MULTISCALE_CKPT``, ``MULTISCALE_STEPS`` (default 1),
``MULTISCALE_REFINEMENT`` (default 2), ``MULTISCALE_LR`` (default 1e-2).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "output" / "diffusion_ca_squared" / "v1_ortho" / "last.ckpt"
RAY_CONFIG = REPO / "output" / "cell20_flat4_materialreg" / "config.yaml"
NF_WEIGHTS = REPO / "output" / "cell20_flat4_materialreg" / "best_weights.npz"
SCALER_DIR = REPO / "microstructure_generation_2d"


def test_predict_structure_smoke():
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import torch

    ckpt = Path(os.environ.get("MULTISCALE_CKPT", DEFAULT_CKPT))
    for p in (ckpt, RAY_CONFIG, NF_WEIGHTS, SCALER_DIR / "scaler_C11"):
        if not p.exists():
            pytest.skip(f"missing required file: {p}")

    from multiscale_generation.config import (
        CellDecompositionConfig, DiffusionConfig, MultiscaleConfig, NeuralFieldConfig,
    )
    from multiscale_generation.model import MultiscaleDiffusionModel

    steps = int(os.environ.get("MULTISCALE_STEPS", "1"))
    refinement = int(os.environ.get("MULTISCALE_REFINEMENT", "2"))
    lr = float(os.environ.get("MULTISCALE_LR", "1e-2"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # NeuralField hyper-parameters match the run that produced best_weights.npz
    # (config.yaml: hidden_size=512, n_layers=6, n_fourier=64); init_weights warm-starts θ.
    cfg = MultiscaleConfig(
        diffusion=DiffusionConfig(ckpt=ckpt, scaler_dir=SCALER_DIR, steps=steps, tensor_w=2.0),
        cell_decomposition=CellDecompositionConfig(config_path=RAY_CONFIG),
        neural_field=NeuralFieldConfig(
            hidden_size=512, n_layers=6, n_fourier=64, seed=42, output_scale=0.1,
            init_weights=NF_WEIGHTS,
        ),
    )
    model = MultiscaleDiffusionModel.from_config(device, cfg)

    canvas, theta, loss_history = model.predict_structure(
        lr=lr, refinement_factor=refinement, void_ratio=1e-2, simp_p=1.0,
    )

    assert len(loss_history) == steps
    assert np.all(np.isfinite(loss_history)), loss_history
    assert np.isfinite(np.asarray(canvas)).all()
    theta_norm = sum(float((np.asarray(l["W"]) ** 2).sum()) for l in theta)
    assert np.isfinite(theta_norm) and theta_norm > 0
