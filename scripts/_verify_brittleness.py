"""Is the matched-sweep floor a brittleness cliff (no bug) or a pipeline bug?

At the design frequency, sweep params = (1-a)*optimized + a*matched and watch
the cloaking ratio. If it collapses to the ~0.6 floor for small a (cliff well
below the 4.4% matching error), then improving matching from 17%->4.4% can't
help — both are past the cliff. Adds a random-direction control of the same
per-element magnitude as (matched-opt) to test direction- vs magnitude-sensitivity.
"""
import sys
from pathlib import Path
import numpy as np
import jax.numpy as jnp
from jax_fem.solver import solver as jax_fem_solver

sys.path.insert(0, "scripts")
import frequency_sweep_validated as V

from rayleigh_cloak import load_config
from rayleigh_cloak.cells import CellDecomposition
from rayleigh_cloak.config import DerivedParams
from rayleigh_cloak.loss import transmitted_displacement_ratio
from rayleigh_cloak.mesh import extract_submesh, generate_mesh_full
from rayleigh_cloak.problem import build_problem
from rayleigh_cloak.solver import _create_geometry, solve_reference

# Common evaluation mesh/geometry for BOTH models (geometry is identical across
# the two configs; only mesh resolution differs — so fix it to remove that confound).
EVAL_CFG = "configs/cell20_flat4_materialreg4_pt3.yaml"
# Model whose (optimized, matched) pair we ramp between:
MODEL_CFG = sys.argv[1] if len(sys.argv) > 1 else "configs/cell20_flat4_materialreg4_pt3.yaml"
PARAMS = sys.argv[2] if len(sys.argv) > 2 else "output/cell20_flat4_materialreg4_pt3/optimized_params.npz"
DS = "output/ca_bulk_squared/stiffness.h5"
FSTAR = 2.0
print(f"EVAL_CFG={EVAL_CFG}\nMODEL_CFG={MODEL_CFG}\nPARAMS={PARAMS}")

base = load_config(EVAL_CFG)
_, _, _, _, matched_C, matched_rho, diag = V.build_canvas(
    Path(PARAMS), Path(DS), Path(MODEL_CFG), rho_weight=1.0)
npz = np.load(PARAMS)
opt_C, opt_rho = npz["cell_C_flat"], npz["cell_rho"]
print(f"unique dataset entries matched: {diag['n_unique_dataset_entries']}")

# Build the design-freq problem once; only params change between solves.
cfg = V._make_config_at_fstar(base, FSTAR, refinement_factor=None)
dp = DerivedParams.from_config(cfg)
geom = _create_geometry(cfg, dp)
full = generate_mesh_full(cfg, dp, geom)
cloak, kept = extract_submesh(full, geom)
ref = solve_reference(cfg, mesh=full)
cd = CellDecomposition(geom, cfg.cells.n_x, cfg.cells.n_y)
problem = build_problem(cloak, cfg, dp, geom, cd)
cs, rs = V._surface_indices_at_f(cloak, geom, dp, kept, loss_cfg=base.loss)
solver_opts = {"petsc_solver": {"ksp_type": base.solver.ksp_type,
                                "pc_type": base.solver.pc_type}}

def ratio_for(C, rho):
    problem.set_params((jnp.asarray(C), jnp.asarray(rho)))
    sol = jax_fem_solver(problem, solver_options=solver_opts)
    return float(transmitted_displacement_ratio(np.asarray(sol[0]), ref.u, cs, rs))

dC, drho = matched_C - opt_C, matched_rho - opt_rho
# pooled median rel error of the full snap, for axis labelling
mask_nz = np.abs(opt_C) > 0
relmed = np.median(np.abs(dC[mask_nz]) / np.abs(opt_C[mask_nz]))
print(f"full-snap pooled median rel error = {100*relmed:.1f}%\n")

print("=== interpolation opt -> matched (design f*=2.0) ===")
print(f"{'alpha':>6} {'~rel%':>7} {'ratio':>8}")
for a in [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0]:
    r = ratio_for(opt_C + a * dC, opt_rho + a * drho)
    print(f"{a:6.2f} {100*a*relmed:7.1f} {r:8.4f}", flush=True)

print("\n=== random-direction control, same |matched-opt| magnitude ===")
for seed in (0, 1, 2):
    rng = np.random.default_rng(seed)
    sC = rng.choice([-1.0, 1.0], size=dC.shape)
    sr = rng.choice([-1.0, 1.0], size=drho.shape)
    r = ratio_for(opt_C + sC * np.abs(dC), opt_rho + sr * np.abs(drho))
    print(f"  seed={seed}: ratio={r:.4f}", flush=True)
