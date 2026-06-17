"""MLP + Fourier features for 3D material fields.

Mirrors :mod:`rayleigh_cloak.neural_reparam` with 3D Fourier features
(``6 * n_freq`` components: sin/cos per axis × ``n_freq`` frequencies).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# ── MLP (pure JAX) ───────────────────────────────────────────────────


def _init_layer(key, n_in: int, n_out: int) -> dict:
    k1, _ = jax.random.split(key)
    bound = jnp.sqrt(6.0 / (n_in + n_out))
    W = jax.random.uniform(k1, (n_in, n_out), minval=-bound, maxval=bound)
    b = jnp.zeros(n_out)
    return {"W": W, "b": b}


def init_mlp(key, layer_sizes: list[int]) -> list[dict]:
    params = []
    for i in range(len(layer_sizes) - 1):
        key, sub = jax.random.split(key)
        params.append(_init_layer(sub, layer_sizes[i], layer_sizes[i + 1]))
    return params


def mlp_forward(params: list[dict], x: jnp.ndarray) -> jnp.ndarray:
    h = x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    last = params[-1]
    return h @ last["W"] + last["b"]


# ── 3D Fourier features ──────────────────────────────────────────────


def fourier_features_3d(xyz: jnp.ndarray, n_freq: int = 32) -> jnp.ndarray:
    """Map ``(n, 3)`` coordinates to ``(n, 6*n_freq)`` Fourier features.

    Uses linearly-spaced integer frequencies ``[1, n_freq]`` per axis and
    concatenates ``sin``/``cos`` for each axis.
    """
    freqs = jnp.linspace(1.0, float(n_freq), n_freq)
    px = xyz[:, 0:1] * freqs[None, :]
    py = xyz[:, 1:2] * freqs[None, :]
    pz = xyz[:, 2:3] * freqs[None, :]
    return jnp.concatenate([
        jnp.sin(px), jnp.cos(px),
        jnp.sin(py), jnp.cos(py),
        jnp.sin(pz), jnp.cos(pz),
    ], axis=-1)


# ── weight I/O ───────────────────────────────────────────────────────


def save_theta(theta: list[dict], path: str) -> None:
    arrays = {"n_layers": np.array(len(theta))}
    for i, layer in enumerate(theta):
        arrays[f"W_{i}"] = np.asarray(layer["W"])
        arrays[f"b_{i}"] = np.asarray(layer["b"])
    np.savez(path, **arrays)


def load_theta(path: str) -> list[dict]:
    data = np.load(path)
    n_layers = int(data["n_layers"])
    return [
        {"W": jnp.array(data[f"W_{i}"]), "b": jnp.array(data[f"b_{i}"])}
        for i in range(n_layers)
    ]
