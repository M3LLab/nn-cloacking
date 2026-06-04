
import joblib

from rayleigh_cloak.neural_reparam import load_theta, make_neural_reparam


def load_scalers(scaler_dir):
    scaler_C11 = joblib.load(scaler_dir / "scaler_C11")
    scaler_C12 = joblib.load(scaler_dir / "scaler_C12")
    scaler_C66 = joblib.load(scaler_dir / "scaler_C66")

    return scaler_C11, scaler_C12, scaler_C66


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
    )

    weights_path = getattr(nf_config, "init_weights", None) or getattr(nf_config, "ckpt", None)
    if weights_path:
        theta_init, _ = load_theta(weights_path)

    return reparam, theta_init
