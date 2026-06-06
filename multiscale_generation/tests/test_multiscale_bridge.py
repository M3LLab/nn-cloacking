"""Tests for the torch<->JAX gradient bridge in ``predict_structure``.

Layered so the suite runs anywhere:

* ``test_tile_image_torch_matches_numpy`` / ``test_decode_tile_gradcheck`` —
  always run (torch only, no external files); guard the differentiable decode+tile.
* ``test_pixel_objective_gradient_fd`` — opt-in (set ``MULTISCALE_RUN_FEM=1``);
  finite-difference checks ``value_and_grad_canvas`` (builds a coarse mesh + FEM
  solve, ~30s). This is the regression for the main risk: ``ad_wrapper``
  differentiability w.r.t. the canvas.
* ``test_predict_structure_integration`` — the end-to-end smoke test; **skips**
  unless a valid diffusion checkpoint + scalers are present. Configure via env:
  ``MULTISCALE_CKPT``, ``MULTISCALE_SCALER_DIR``, ``MULTISCALE_CONFIG``,
  ``MULTISCALE_STEPS``, ``MULTISCALE_REFINEMENT``, ``MULTISCALE_N_C_PARAMS``.
"""
from __future__ import annotations

import os
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

# The whole module needs torch (multiscale_generation.model imports it at top).
torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = REPO / "output" / "diffusion_ca_squared" / "v2" / "last.ckpt"
DEFAULT_SCALER_DIR = REPO / "microstructure_generation_2d"
DEFAULT_CONFIG = REPO / "configs" / "cauchy_tri.yaml"


# ── always-on: differentiable decode + tile ─────────────────────────


def test_tile_image_torch_matches_numpy():
    """The torch tiling must reproduce the numpy ``tile_image`` exactly."""
    from multiscale_generation.fem import tile_image
    from multiscale_generation.model import tile_image_torch

    n_x, n_y, H = 3, 4, 5
    geoms = np.arange(n_x * n_y * H * H).reshape(n_x * n_y, H, H).astype(np.float32)
    ref = tile_image(geoms, n_x, n_y)
    got = tile_image_torch(torch.tensor(geoms), n_x, n_y).numpy()
    assert np.array_equal(ref, got)


@pytest.mark.parametrize("compressed,S", [(True, 6), (False, 64)])
def test_decode_tile_gradcheck(compressed, S):
    """``_assemble_canvas_torch`` is differentiable w.r.t. ``x_start``."""
    from multiscale_generation.model import MultiscaleDiffusionModel

    # ``compressed`` is a property on real instances (resolves None from the
    # checkpoint); the stub bypasses __init__, so expose it directly.
    stub = SimpleNamespace(device="cpu", compressed=compressed)
    fn = MultiscaleDiffusionModel._assemble_canvas_torch.__get__(stub, MultiscaleDiffusionModel)
    n_x, n_y, n_cells = 2, 2, 4
    cloak_idx = np.array([0, 3])
    x_start = torch.rand(len(cloak_idx), 1, S, S, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda xs: fn(xs, cloak_idx, n_cells, n_x, n_y), (x_start,),
        eps=1e-6, atol=1e-6, rtol=1e-4,
    )


# ── opt-in: differentiable pixel FEM (FD gradcheck of the main risk) ──


@pytest.mark.skipif(
    os.environ.get("MULTISCALE_RUN_FEM") != "1",
    reason="set MULTISCALE_RUN_FEM=1 to run the FEM finite-difference gradcheck (~30s)",
)
def test_pixel_objective_gradient_fd():
    import jax.numpy as jnp
    from rayleigh_cloak import load_config
    from multiscale_generation.fem import (
        build_cloak_layout, build_pixel_objective, structure_cloaking_loss,
    )

    cfg = load_config(str(DEFAULT_CONFIG))
    bbox = build_cloak_layout(cfg).cloak_bbox
    Hc = Wc = 48
    obj = build_pixel_objective(
        cfg, bbox, (Hc, Wc), refinement_factor=2,
        void_ratio=1e-2, simp_p=1.0, binarize=False,
    )
    rng = np.random.default_rng(0)
    canvas = jnp.asarray(rng.uniform(0.2, 0.8, size=(Hc, Wc)).astype(np.float32))
    loss, g = obj(canvas)
    g = np.asarray(g)
    assert np.isfinite(g).all() and np.abs(g).max() > 0

    # FD-check the most influential (in-cloak) pixels.
    eps = 1e-2
    for idx in np.argsort(np.abs(g).ravel())[::-1][:4]:
        r, c = divmod(int(idx), Wc)
        cp = np.array(canvas); cp[r, c] += eps
        cm = np.array(canvas); cm[r, c] -= eps
        fd = (float(obj.loss_only(jnp.asarray(cp))) - float(obj.loss_only(jnp.asarray(cm)))) / (2 * eps)
        an = g[r, c]
        assert abs(fd - an) / (abs(fd) + abs(an) + 1e-12) < 0.05, (r, c, an, fd)

    # Bridge forward loss must match the numpy structure_cloaking_loss (binarized).
    bin_canvas = (np.asarray(canvas) > 0.5).astype(np.float32)
    obj_bin = build_pixel_objective(
        cfg, bbox, (Hc, Wc), refinement_factor=2,
        void_ratio=1e-2, simp_p=1.0, binarize=True,
    )
    bridge = float(obj_bin.loss_only(jnp.asarray(bin_canvas)))
    ref, _, _ = structure_cloaking_loss(
        bin_canvas, cfg, bbox, refinement_factor=2,
        void_ratio=1e-2, simp_p=1.0, binarize=True,
    )
    assert abs(bridge - float(ref)) / (abs(float(ref)) + 1e-12) < 1e-6


# ── end-to-end integration (needs a real diffusion checkpoint) ───────


def _checkpoint_ok(path: Path) -> bool:
    """Valid (complete) PyTorch-Lightning zip checkpoint?"""
    return path.is_file() and zipfile.is_zipfile(path)


def test_predict_structure_integration():
    """One short ``predict_structure`` trajectory: loss finite, θ updates, no NaNs.

    Uses ``n_C_params=6`` so ``conditions_fn`` is runnable — this exercises the
    full bridge plumbing end-to-end; the physical condition mapping (which Voigt
    components feed the diffusion) is the separate open seam.
    """
    ckpt = Path(os.environ.get("MULTISCALE_CKPT", DEFAULT_CKPT))
    scaler_dir = Path(os.environ.get("MULTISCALE_SCALER_DIR", DEFAULT_SCALER_DIR))
    config = os.environ.get("MULTISCALE_CONFIG", str(DEFAULT_CONFIG))

    if not _checkpoint_ok(ckpt):
        pytest.skip(f"no valid diffusion checkpoint at {ckpt}")
    if not (scaler_dir / "scaler_C11").is_file():
        pytest.skip(f"no scalers under {scaler_dir}")

    from rayleigh_cloak import load_config
    from multiscale_generation.model import MultiscaleDiffusionModel

    steps = int(os.environ.get("MULTISCALE_STEPS", "3"))
    refinement = int(os.environ.get("MULTISCALE_REFINEMENT", "2"))
    n_c_params = int(os.environ.get("MULTISCALE_N_C_PARAMS", "6"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_config = SimpleNamespace(
        scaler_dir=scaler_dir, ckpt=str(ckpt),
        steps=steps, tensor_w=2.0, compressed=False,  # synced from generator below
    )
    model = MultiscaleDiffusionModel(
        device, diffusion_config,
        SimpleNamespace(config_path=config),
        SimpleNamespace(hidden_size=64, n_layers=3, n_fourier=16, seed=0, output_scale=0.1),
    )
    # Trust the loaded generator's own layout flag (25x25 quadrant => compressed).
    diffusion_config.compressed = bool(getattr(model.generator, "compressed", False))

    # Override n_C_params for the reparam (conditions_fn assumes >=3 stiffness comps).
    ray_cfg = load_config(config)
    model._ray_cfg = ray_cfg.model_copy(update={
        "cells": ray_cfg.cells.model_copy(update={"n_C_params": n_c_params})})

    canvas, theta, loss_history = model.predict_structure(
        lr=1e-2, refinement_factor=refinement, void_ratio=1e-2, simp_p=1.0,
    )

    assert len(loss_history) == steps
    assert all(np.isfinite(loss_history))
    assert np.isfinite(np.asarray(canvas)).all()
    theta_norm = sum(float((np.asarray(l["W"]) ** 2).sum()) for l in theta)
    assert np.isfinite(theta_norm)
