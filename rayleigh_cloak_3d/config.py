"""Pydantic configuration models and derived physical parameters (3D).

Coordinate convention: (x, y) horizontal, z vertical (positive up).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel


# ── user-facing config sections ──────────────────────────────────────


class MaterialConfig(BaseModel):
    rho0: float = 1600.0
    cs: float = 300.0


class DomainConfig(BaseModel):
    f_star: float = 2.0
    lambda_star: float = 1.0
    H_factor: float = 4.305   # depth (z extent of physical region) / lambda
    W_factor: float = 6.0     # horizontal extent (x and y, symmetric) / lambda


class ConicalCloakConfig(BaseModel):
    a_factor: float = 0.0774   # inner cone depth  = a_factor * H
    b_factor: float = 3.0      # outer cone depth  = b_factor * a
    c_factor: float = 0.1545   # surface radius    = c_factor * H


class AbsorbingConfig(BaseModel):
    L_pml_factor: float = 1.0
    xi_max: float = 4.0
    pml_pow: int = 2


class MeshConfig(BaseModel):
    n_pml: int = 8              # cells across each PML layer
    n_phys: int = 24            # cells across the physical region per axis (x,y,z)
    refinement_factor: int = 2  # h_fine = h_elem / refinement_factor near the cloak
    ele_type: str = "TET10"


class SourceConfig(BaseModel):
    """Gaussian point-like vertical traction on the free surface."""
    source_type: Literal["gaussian_point"] = "gaussian_point"
    x_src_factor: float = 0.25   # x_src (physical) = x_src_factor * W
    y_src_factor: float = 0.5    # y_src (physical) = y_src_factor * W
    sigma_factor: float = 0.05   # Gaussian footprint std = sigma_factor * lambda
    F0: float = 1.0              # traction amplitude (negative = downward)


class SolverConfig(BaseModel):
    backend: Literal["petsc", "cudss"] = "petsc"
    # PETSc options (used when backend == 'petsc').
    ksp_type: str = "preonly"
    pc_type: str = "lu"
    pc_factor_mat_solver_type: str = ""   # e.g. "mumps" for large 3D
    # cuDSS options (used when backend == 'cudss'). See rayleigh_cloak_3d.cudss_solver.
    cudss_hybrid_memory: bool = False
    cudss_verbose: bool = True
    # Reuse the forward factor for the adjoint via J A J = A^T. Halves adjoint
    # wall time and VRAM. Valid for this 3D vec=6 operator; the 2D vec=4
    # validation problem must leave it off (measured 4.6e-3 rel err there).
    cudss_signflip_adjoint: bool = False
    # Hold at most one factor in VRAM. Alternative to the sign-flip when the
    # forward and adjoint factors cannot coexist on the card.
    cudss_exclusive_gpu: bool = False


class CellConfig(BaseModel):
    """Material-field discretisation.

    ``mode = "continuous"``: the neural field is evaluated directly at each
    FEM quadrature point. ``mode = "grid"``: evaluated at the centre of each
    cell of a regular ``n_x × n_y × n_z`` grid over the cloak bounding box.
    """
    mode: Literal["continuous", "grid"] = "continuous"
    n_x: int = 10
    n_y: int = 10
    n_z: int = 10
    n_C_params: Literal[2, 3, 9, 21] = 2   # 2 = isotropic, 3 = cubic, 9 = orthotropic, 21 = full
    symmetrize_init: bool = True


class NeuralReparamConfig(BaseModel):
    hidden_size: int = 256
    n_layers: int = 4
    n_fourier: int = 32
    seed: int = 42
    output_scale: float = 0.1
    init_weights: str = ""


class OptimizationConfig(BaseModel):
    method: Literal["neural"] = "neural"
    n_iters: int = 500
    lr: float = 1e-3
    lr_end: float | None = None
    lr_schedule: Literal["linear", "cosine"] = "linear"
    lambda_l2: float = 0.0
    plot_every: int = 0
    init_params: str = ""
    neural: NeuralReparamConfig = NeuralReparamConfig()


class LossConfig(BaseModel):
    model_config = {"extra": "ignore"}
    type: Literal["top_surface", "outside_cloak", "right_boundary"] = "top_surface"


class SimulationConfig3D(BaseModel):
    is_reference: bool = False
    geometry_type: Literal["conical"] = "conical"

    material: MaterialConfig = MaterialConfig()
    domain: DomainConfig = DomainConfig()
    geometry: ConicalCloakConfig = ConicalCloakConfig()
    absorbing: AbsorbingConfig = AbsorbingConfig()
    mesh: MeshConfig = MeshConfig()
    source: SourceConfig = SourceConfig()
    solver: SolverConfig = SolverConfig()
    cells: CellConfig = CellConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    loss: LossConfig = LossConfig()
    output_dir: str = "output"


def load_config(path: str | Path) -> SimulationConfig3D:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SimulationConfig3D(**(data or {}))


# ── derived / computed quantities ────────────────────────────────────


@dataclass(frozen=True)
class DerivedParams3D:
    """Physical quantities derived from :class:`SimulationConfig3D`.

    All lengths are in the *extended-mesh* coordinate system (i.e. including
    PML padding).  (x, y) are horizontal; z is vertical (free surface on top).
    """

    # material
    rho0: float
    mu: float
    lam: float
    nu: float
    cs: float
    cp: float
    cR: float

    # frequency
    omega: float

    # domain extents (physical)
    W: float           # horizontal, x and y (square footprint)
    H: float           # depth, z extent of physical region
    lambda_star: float

    # PML
    L_pml: float
    xi_max: float
    pml_pow: int

    # extended domain
    W_total: float
    H_total: float
    x_off: float
    y_off: float
    z_off: float       # physical floor in extended coords (= L_pml)

    # convenience
    z_top: float       # free surface z (= H_total)
    x_c: float         # cloak axis x
    y_c: float         # cloak axis y

    # conical cloak geometry
    a: float
    b: float
    c: float

    # source
    x_src: float
    y_src: float
    sigma_src: float
    F0: float

    @staticmethod
    def from_config(cfg: SimulationConfig3D) -> DerivedParams3D:
        mat = cfg.material
        dom = cfg.domain
        geo = cfg.geometry
        ab = cfg.absorbing
        src = cfg.source

        rho0 = mat.rho0
        cs = mat.cs
        cp = np.sqrt(3.0) * cs
        mu = rho0 * cs ** 2
        lam = rho0 * cp ** 2 - 2 * mu
        nu = lam / (2 * (lam + mu))
        cR = cs * (0.826 + 1.14 * nu) / (1 + nu)

        lambda_star = dom.lambda_star
        omega = 2 * np.pi * dom.f_star * cR / lambda_star

        H = dom.H_factor * lambda_star
        W = dom.W_factor * lambda_star

        a = geo.a_factor * H
        b = geo.b_factor * a
        c = geo.c_factor * H

        L_pml = ab.L_pml_factor * lambda_star
        W_total = 2 * L_pml + W          # PML on both horizontal sides (x and y)
        H_total = L_pml + H               # PML on bottom only; free surface on top
        x_off = L_pml
        y_off = L_pml
        z_off = L_pml
        z_top = H_total
        x_c = x_off + W / 2.0
        y_c = y_off + W / 2.0

        x_src = x_off + src.x_src_factor * W
        y_src = y_off + src.y_src_factor * W
        sigma_src = src.sigma_factor * lambda_star

        return DerivedParams3D(
            rho0=rho0, mu=mu, lam=lam, nu=nu, cs=cs, cp=cp, cR=cR,
            omega=omega,
            W=W, H=H, lambda_star=lambda_star,
            L_pml=L_pml, xi_max=ab.xi_max, pml_pow=ab.pml_pow,
            W_total=W_total, H_total=H_total,
            x_off=x_off, y_off=y_off, z_off=z_off,
            z_top=z_top, x_c=x_c, y_c=y_c,
            a=a, b=b, c=c,
            x_src=x_src, y_src=y_src, sigma_src=sigma_src, F0=src.F0,
        )
