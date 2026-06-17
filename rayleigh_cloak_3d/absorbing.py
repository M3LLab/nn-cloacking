"""Absorbing-layer (Rayleigh-damping) profile for 3D half-space.

PML damping is applied on five faces: x = 0 and x = W_total, y = 0 and
y = W_total, z = 0 (bottom). The top face z = z_top is a free surface.
"""

from __future__ import annotations

from typing import Callable

import jax.numpy as jnp

from rayleigh_cloak_3d.config import DerivedParams3D


def make_xi_profile(params: DerivedParams3D) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a closure ``xi(x)`` capturing domain parameters.

    The returned function is JAX-traceable and suitable for ``jax.vmap``.
    Contributions from each PML face are additive; inside the physical
    region the profile is zero.
    """
    x_off = params.x_off
    y_off = params.y_off
    z_off = params.z_off
    W = params.W
    L_pml = params.L_pml
    xi_max = params.xi_max
    pml_pow = params.pml_pow

    def xi_profile(x: jnp.ndarray) -> jnp.ndarray:
        d_xmin = jnp.maximum(x_off - x[0], 0.0)
        d_xmax = jnp.maximum(x[0] - (x_off + W), 0.0)
        xi_x = xi_max * (jnp.maximum(d_xmin, d_xmax) / L_pml) ** pml_pow

        d_ymin = jnp.maximum(y_off - x[1], 0.0)
        d_ymax = jnp.maximum(x[1] - (y_off + W), 0.0)
        xi_y = xi_max * (jnp.maximum(d_ymin, d_ymax) / L_pml) ** pml_pow

        d_zbot = jnp.maximum(z_off - x[2], 0.0)
        xi_z = xi_max * (d_zbot / L_pml) ** pml_pow

        return xi_x + xi_y + xi_z

    return xi_profile
