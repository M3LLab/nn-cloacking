"""Pydantic configuration for the multiscale diffusion model.

Collects every path / hyper-parameter :class:`MultiscaleDiffusionModel` needs into
three nested models:

* :class:`DiffusionConfig`      — the trained 2D diffusion checkpoint, the fitted
  ``StandardScaler`` directory, and sampling settings.
* :class:`CellDecompositionConfig` — the rayleigh ``SimulationConfig`` YAML that
  describes the cloak geometry / cell grid.
* :class:`NeuralFieldConfig`    — the ``NeuralReparam`` MLP hyper-parameters (must
  match train time) plus an optional warm-start weights path.

``MultiscaleConfig`` composes the three so a run can be described by one object;
``MultiscaleDiffusionModel.from_config`` consumes it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel


class DiffusionConfig(BaseModel):
    """Trained 2D diffusion checkpoint + the condition scalers it was fit with."""

    ckpt: Path                       # PyTorch-Lightning .ckpt for DiffusionModel
    scaler_dir: Path                 # dir holding scaler_C11/C22/C12/C66 (joblib)
    steps: int = 50                  # number of denoise steps per sample
    tensor_w: float = 2.0            # classifier-free guidance scale
    # 25x25 mirror-quadrant (v4+) decodes to 50x50 via unfold; legacy 64x64
    # decodes via centre-crop. ``None`` => take the checkpoint's saved hparam.
    compressed: Optional[bool] = None


class CellDecompositionConfig(BaseModel):
    """Locates the rayleigh ``SimulationConfig`` describing the cloak layout."""

    config_path: Path                # YAML loadable by rayleigh_cloak.load_config


class NeuralFieldConfig(BaseModel):
    """``NeuralReparam`` MLP settings; must match the values used at train time.

    ``init_weights`` (or ``ckpt``) points at an .npz of saved MLP weights
    (``W_0``/``b_0``/.../``n_layers`` — see ``rayleigh_cloak.neural_reparam.save_theta``)
    for the torch<->JAX bridge. Leave unset to start from the near-zero init.
    """

    hidden_size: int = 256
    n_layers: int = 4
    n_fourier: int = 32
    seed: int = 42
    output_scale: float = 0.1
    init_weights: Optional[Path] = None
    ckpt: Optional[Path] = None


class OptimizeConfig(BaseModel):
    """Hyper-parameters for one ``predict_structure`` training trajectory.

    The number of optimizer steps equals the number of diffusion steps
    (``DiffusionConfig.steps``), so step count lives there, not here.
    """

    lr: float = 1e-3
    refinement_factor: Optional[int] = None  # FEM mesh refinement (None => config's)
    void_ratio: float = 1e-6                 # SIMP min density (void stiffness floor)
    simp_p: float = 3.0                      # SIMP penalisation exponent
    binarize: bool = False                   # threshold the canvas before the FEM
    freeze: bool = False                     # if True: NF frozen, no grads/optimizer (eval-only)
    plot_every: int = 1                      # save a structure image every N steps (0 = off)
    ckpt_every: int = 0                      # save θ every N steps (0 = final only)
    output_dir: Optional[Path] = None        # where to write artifacts (CLI --output-dir overrides)


class MultiscaleConfig(BaseModel):
    """Everything :class:`MultiscaleDiffusionModel` needs, in one object."""

    diffusion: DiffusionConfig
    cell_decomposition: CellDecompositionConfig
    neural_field: NeuralFieldConfig
    optimize: OptimizeConfig = OptimizeConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "MultiscaleConfig":
        """Load a :class:`MultiscaleConfig` from a YAML file (see ``configs/multiscale_generation``)."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**(data or {}))
