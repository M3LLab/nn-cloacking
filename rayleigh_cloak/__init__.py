"""Modular cloaking simulation package.

Public API::

    from cloak import solve, solve_reference, load_config, SimulationConfig

    # From YAML
    config = load_config("configs/default.yaml")
    result = solve(config)

    # Programmatic
    from cloak.config import SimulationConfig, MaterialConfig
    config = SimulationConfig(material=MaterialConfig(rho0=2000))
    result = solve(config)

    # Reference field
    ref = solve_reference(config)
"""

from rayleigh_cloak.config import DerivedParams, SimulationConfig, load_config

# Lazy imports for solver components (they pull in jax-fem / gmsh which
# may not be available in lightweight environments).


def __getattr__(name: str):
    if name in ("solve", "solve_reference", "solve_cell_based", "SolutionResult"):
        from rayleigh_cloak import solver

        return getattr(solver, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "SimulationConfig",
    "DerivedParams",
    "load_config",
    "solve",
    "solve_reference",
    "solve_cell_based",
    "SolutionResult",
]
