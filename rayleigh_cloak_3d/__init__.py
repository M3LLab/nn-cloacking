"""3D Rayleigh-wave cloak simulation package.

Sibling of ``rayleigh_cloak`` (2D) with a reworked material-field layer: the
neural network is the primary object that produces (C, rho) at the FEM
quadrature points, and the cell decomposition is one optional implementation
of that interface rather than a mandatory layer.

Coordinate convention: x, y are horizontal, z is vertical (positive up).
The free surface lies at z = H_total; the bottom at z = 0.
"""

from rayleigh_cloak_3d.config import (
    DerivedParams3D,
    SimulationConfig3D,
    load_config,
)


def __getattr__(name: str):
    if name in ("solve", "solve_reference", "solve_optimization_neural"):
        from rayleigh_cloak_3d import solver
        return getattr(solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimulationConfig3D",
    "DerivedParams3D",
    "load_config",
    "solve",
    "solve_reference",
    "solve_optimization_neural",
]
