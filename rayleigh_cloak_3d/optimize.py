"""Adam-based neural-reparam optimisation for the 3D cloak.

The optimizer is agnostic to the material-field implementation (continuous
vs cell-decomposed). It takes a differentiable ``fwd_pred`` whose input is
``theta`` (MLP weights) — the material-field evaluation is baked in by
the caller when building ``fwd_pred``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np


# ── loss components ──────────────────────────────────────────────────


def cloaking_loss(
    u_cloak: jnp.ndarray,
    u_ref_target: jnp.ndarray,
    target_indices: jnp.ndarray,
) -> jnp.ndarray:
    """Relative L2 displacement difference on a target node set."""
    diff = u_cloak[target_indices] - u_ref_target
    ref_norm_sq = jnp.sum(u_ref_target ** 2) + 1e-30
    return jnp.sum(diff ** 2) / ref_norm_sq


# ── Adam ─────────────────────────────────────────────────────────────


@dataclass
class AdamState:
    m: Any = None
    v: Any = None
    t: int = 0


def adam_init(params) -> AdamState:
    m = jax.tree.map(jnp.zeros_like, params)
    v = jax.tree.map(jnp.zeros_like, params)
    return AdamState(m=m, v=v, t=0)


def adam_update(
    grads,
    state: AdamState,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple:
    t = state.t + 1
    m = jax.tree.map(lambda m_, g: beta1 * m_ + (1 - beta1) * g, state.m, grads)
    v = jax.tree.map(lambda v_, g: beta2 * v_ + (1 - beta2) * g ** 2, state.v, grads)
    m_hat = jax.tree.map(lambda m_: m_ / (1 - beta1 ** t), m)
    v_hat = jax.tree.map(lambda v_: v_ / (1 - beta2 ** t), v)
    updates = jax.tree.map(
        lambda mh, vh: -lr * mh / (jnp.sqrt(vh) + eps), m_hat, v_hat,
    )
    return updates, AdamState(m=m, v=v, t=t)


# ── optimisation loop ────────────────────────────────────────────────


@dataclass
class NeuralOptimizationResult:
    theta: list[dict]
    best_theta: list[dict]
    opt_state: AdamState
    loss_history: list[float] = field(default_factory=list)


def _schedule(lr: float, lr_end: float | None, schedule: str, step: int, n_iters: int) -> float:
    if lr_end is None:
        return lr
    t_frac = step / max(n_iters - 1, 1)
    if schedule == "cosine":
        return lr_end + 0.5 * (lr - lr_end) * (1 + math.cos(math.pi * t_frac))
    return lr + (lr_end - lr) * t_frac


def run_optimization(
    fwd_pred: Callable,
    theta_init: list[dict],
    u_ref_target: jnp.ndarray,
    target_indices: np.ndarray,
    n_iters: int = 100,
    lr: float = 1e-3,
    lr_end: float | None = None,
    lr_schedule: str = "linear",
    loss_fn: Callable | None = None,
    step_callback: Callable | None = None,
) -> NeuralOptimizationResult:
    """Run Adam optimisation over MLP weights ``theta``.

    ``fwd_pred(theta)`` must return a list whose first element is the FEM
    displacement solution (shape ``(n_nodes, 6)``). The material field is
    already wired inside ``fwd_pred`` by the caller.
    """
    if loss_fn is None:
        loss_fn = cloaking_loss

    theta = jax.tree.map(jnp.copy, theta_init)
    opt_state = adam_init(theta)
    target_indices_j = jnp.asarray(target_indices)

    best_loss = float("inf")
    best_theta = jax.tree.map(jnp.copy, theta)
    history: list[float] = []

    def _loss(theta):
        sol_list = fwd_pred(theta)
        u = sol_list[0]
        return loss_fn(u, u_ref_target, target_indices_j)

    loss_and_grad = jax.value_and_grad(_loss)

    for step in range(n_iters):
        cur_lr = _schedule(lr, lr_end, lr_schedule, step, n_iters)
        loss_val, grads = loss_and_grad(theta)
        loss_f = float(loss_val)
        history.append(loss_f)

        grad_norm = float(jnp.sqrt(sum(
            jnp.sum(l["W"] ** 2) + jnp.sum(l["b"] ** 2) for l in grads
        )))
        print(
            f"  Step {step:4d} | loss = {loss_f:.4e}"
            f"  pct = {np.sqrt(max(loss_f, 0)) * 100:.2f}%"
            f"  lr = {cur_lr:.2e}  |grad| = {grad_norm:.4e}"
        )

        if loss_f < best_loss:
            best_loss = loss_f
            best_theta = jax.tree.map(jnp.copy, theta)

        if step_callback is not None:
            step_callback(step, loss_f, theta)

        updates, opt_state = adam_update(grads, opt_state, lr=cur_lr)
        theta = jax.tree.map(lambda p, u: p + u, theta, updates)

    return NeuralOptimizationResult(
        theta=theta,
        best_theta=best_theta,
        opt_state=opt_state,
        loss_history=history,
    )
