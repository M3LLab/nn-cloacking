
from pathlib import Path

import joblib

from rayleigh_cloak.neural_reparam import load_theta, make_neural_reparam


def load_scalers(scaler_dir):
    """Load the four fitted ``StandardScaler``s in the diffusion conditioning
    order ``(C11, C22, C12, C66)`` (see ``microstructure_generation_2d`` /
    ``diffusion_dataset``). Returns them in that order."""
    scaler_dir = Path(scaler_dir)
    return tuple(
        joblib.load(scaler_dir / f"scaler_{name}")
        for name in ("C11", "C22", "C12", "C66")
    )


def load_neural_field(nf_config, cell_decomp, params_init):
    """Build the JAX ``(reparam, theta)`` the bridge optimises.

    The neural field is a small MLP (``NeuralReparam``) that maps cell-center
    Fourier features to per-cell ``(C_flat, rho)``; its weights ``theta`` are the
    optimisation variables. The reparam is reconstructed from the cloak ``cell_
    decomp`` + initial cell materials (so ``decode`` matches train time), then the
    pretrained weights are loaded if a checkpoint is given — otherwise the
    near-zero init from :func:`make_neural_reparam` is returned (the residual
    starts at the continuous push-forward).

    ``nf_config`` is expected to expose the MLP hyper-parameters used at train
    time (``hidden_size``, ``n_layers``, ``n_fourier``, ``seed``, ``output_scale``)
    and, optionally, a weights path (``init_weights`` or ``ckpt``). These mirror
    ``config.optimization.neural`` in ``solve_optimization_neural``.

    Returns
    -------
    reparam : NeuralReparam
        Has ``decode(theta) -> (cell_C_flat, cell_rho)`` and ``cloak_mask``.
    theta : list[dict]
        MLP weights (pretrained if available, else the near-zero init).
    """
    theta_init, reparam = make_neural_reparam(
        cell_decomp, params_init,
        hidden_size=nf_config.hidden_size,
        n_layers=nf_config.n_layers,
        n_fourier=nf_config.n_fourier,
        seed=nf_config.seed,
        output_scale=nf_config.output_scale,
        constrained=getattr(nf_config, "constrained", False),
        kappa=getattr(nf_config, "kappa", 0.95),
        cap_anisotropy=getattr(nf_config, "cap_anisotropy", True),
        anisotropy_ratio=getattr(nf_config, "anisotropy_ratio", 15.0),
    )

    weights_path = getattr(nf_config, "init_weights", None) or getattr(nf_config, "ckpt", None)
    if weights_path:
        theta_init, _ = load_theta(weights_path)

    return reparam, theta_init
