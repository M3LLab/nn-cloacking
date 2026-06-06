"""Tests for the diffusion conditioning seam in ``multiscale_generation.model``.

The 2D diffusion is conditioned on a 5-dim vector in the fixed order
``(C11, C22, C12, C66, vol)`` (``microstructure_generation_2d.network.model.TENSOR_DIM``,
matching the training dataset / ``generate.py``). The neural field decodes a flat
stiffness layout that is *not* in that order (flat4 ``C_to_flatC`` is
``[C11, C22, C66, C12]``), so a reorder is required before conditioning.

* ``test_moduli_in_diffusion_order_flat4`` — always on (no checkpoint / mesh):
  guards the reorder helper against the flat4 layout, and documents the old
  ``flat[:, :3]`` bug.
* ``test_cell_condition_matches_manual_reorder`` — integration: builds the model
  from a :class:`MultiscaleConfig`, feeds the real neural field at
  ``output/cell20_flat4_materialreg/optimized_params.npz``, and checks the 5-dim
  conditions match an independent flat4 reorder + scale. Skips if the diffusion
  checkpoint / scalers are absent.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
NF_PARAMS = REPO / "output" / "cell20_flat4_materialreg" / "optimized_params.npz"
RAY_CONFIG = REPO / "output" / "cell20_flat4_materialreg" / "config.yaml"
SCALER_DIR = REPO / "microstructure_generation_2d"
# The orthotropic (5-dim) diffusion checkpoint; override with MULTISCALE_CKPT.
DEFAULT_CKPT = REPO / "output" / "diffusion_ca_squared" / "v1_ortho" / "last.ckpt"


# ── always-on: the reorder helper ───────────────────────────────────


def test_moduli_in_diffusion_order_flat4():
    """``moduli_in_diffusion_order`` must recover (C11, C22, C12, C66) from a
    flat4 layout, which stores them as [C11, C22, C66, C12]."""
    import jax.numpy as jnp
    from rayleigh_cloak.materials import _get_converters
    from multiscale_generation.model import moduli_in_diffusion_order

    # Distinct values so any mis-ordering is caught.
    C11, C22, C12, C66 = 11.0, 22.0, 12.0, 66.0
    _, from_flat = _get_converters(4)  # flatC: [C11, C22, C66, C12]
    flat4 = jnp.asarray([C11, C22, C66, C12])
    C = from_flat(flat4)  # (2,2,2,2)

    got = np.asarray(moduli_in_diffusion_order(C, xp=jnp))
    np.testing.assert_allclose(got, [C11, C22, C12, C66])

    # Regression: the old `flat[:3]` slice mislabels C66 as C12 and drops C12.
    assert not np.allclose(np.asarray(flat4)[:3], [C11, C22, C12])


# ── integration: real neural field through the real model ───────────


def test_cell_condition_matches_manual_reorder():
    pytest.importorskip("torch")
    # NB: don't gate on zipfile.is_zipfile — torch's streaming zip checkpoints are
    # not parseable by Python's zipfile, so let load_from_checkpoint be the judge.
    ckpt = Path(os.environ.get("MULTISCALE_CKPT", DEFAULT_CKPT))
    if not ckpt.is_file():
        pytest.skip(f"no diffusion checkpoint at {ckpt}")
    for p in (NF_PARAMS, RAY_CONFIG, SCALER_DIR / "scaler_C11"):
        if not p.exists():
            pytest.skip(f"missing required file: {p}")

    import torch
    from multiscale_generation.config import (
        CellDecompositionConfig, DiffusionConfig, MultiscaleConfig, NeuralFieldConfig,
    )
    from multiscale_generation.model import MultiscaleDiffusionModel
    from microstructure_generation_2d.network.model import TENSOR_DIM

    # Hyper-parameters match the run that produced optimized_params.npz (config.yaml).
    cfg = MultiscaleConfig(
        diffusion=DiffusionConfig(ckpt=ckpt, scaler_dir=SCALER_DIR, steps=10, tensor_w=2.0),
        cell_decomposition=CellDecompositionConfig(config_path=RAY_CONFIG),
        neural_field=NeuralFieldConfig(
            hidden_size=512, n_layers=6, n_fourier=64, seed=42, output_scale=0.1,
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiscaleDiffusionModel.from_config(device, cfg)

    data = np.load(NF_PARAMS)
    cell_C_flat = data["cell_C_flat"]  # (300, 4) flat4 = [C11, C22, C66, C12]
    cell_rho = data["cell_rho"]        # (300,)

    cond = model._cell_condition(cell_C_flat, cell_rho)  # (300, 5)
    assert cond.shape == (cell_C_flat.shape[0], TENSOR_DIM)
    assert TENSOR_DIM == 5

    # Independent expectation: reorder flat4 [C11,C22,C66,C12] -> [C11,C22,C12,C66]
    # (indices [0,1,3,2]), z-score with the on-disk scalers, append vol.
    moduli = cell_C_flat[:, [0, 1, 3, 2]].astype(np.float64)
    scalers = [joblib.load(SCALER_DIR / f"scaler_{n}") for n in ("C11", "C22", "C12", "C66")]
    scaled = np.stack([s.transform(moduli[:, i:i + 1])[:, 0] for i, s in enumerate(scalers)], axis=1)
    rho0 = model._cloak_layout().dp.rho0
    vol = np.clip(cell_rho / rho0, 0.0, 1.0)
    expected = np.concatenate([scaled, vol[:, None]], axis=1).astype(np.float32)

    np.testing.assert_allclose(cond, expected, rtol=1e-5, atol=1e-6)
