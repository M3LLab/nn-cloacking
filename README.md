# nn-cloacking

Modular frequency-domain 2D elastodynamics code for a triangular Rayleigh-wave cloak with absorbing layers.

The current codebase is package-based. The main workflow is:

- load a YAML config,
- derive physical and mesh parameters,
- build a Gmsh mesh for the physical domain plus PML region,
- solve the split real/imaginary JAX-FEM system,
- save results to NPZ and VTK,
- generate PNG plots from the VTK output.

For installation and environment notes, see [setup.md](setup.md).

## What is in the repo

Main entry point:

- `run.py` — runs one simulation from a YAML config.

Core package:

- `rayleigh_cloak/config.py` — config models and derived parameters.
- `rayleigh_cloak/solver.py` — high-level `solve()` and `solve_reference()` API.
- `rayleigh_cloak/problem.py` — JAX-FEM problem definition.
- `rayleigh_cloak/mesh.py` — Gmsh mesh generation.
- `rayleigh_cloak/materials.py` — isotropic and transformed material tensors.
- `rayleigh_cloak/absorbing.py` — absorbing-layer damping profile.
- `rayleigh_cloak/plot.py` — VTK/NPZ plotting utilities.
- `rayleigh_cloak/io.py` — NPZ and VTK writers.
- `rayleigh_cloak/geometry/triangular.py` — triangular cloak geometry.

Configs:

- `configs/default.yaml` — cloak simulation.
- `configs/reference.yaml` — reference simulation with `is_reference: true`.

Other folders:

- `docs/` — theory notes and figures.
- `tests/` — diagnostic tests and scripts.
- `output/` — generated mesh/results/plots.
- `patches/solver.py` — our patched copy of `jax_fem/solver.py`. The `jax-fem/`
  checkout itself is gitignored, so this is the tracked original; copy it over
  after re-cloning jax-fem.
- `design-validation/` — **git submodule**, the pixel-level FEM validation rig.

### The design-validation submodule

`design-validation` ([M3LLab/design-validation](https://github.com/M3LLab/design-validation))
takes a finished design and re-solves it with material assigned per *pixel*
rather than per homogenised cell — the exact check the homogenised design only
approximates. It is a separate repo because it has to clone and run standalone
on a big validation machine, with its own PETSc/MUMPS build.

```bash
git clone --recurse-submodules git@github.com:M3LLab/nn-cloaking.git
# or, in an existing checkout:
git submodule update --init
```

It carries copies of `rayleigh_cloak/`, `patches/solver.py` and
`rayleigh_cloak_3d/cudss_solver.py` under its `vendor/`, so that a standalone
clone needs nothing from here. **This repo is the source of truth for all
three** — fix them here, then refresh the copies:

```bash
cd design-validation
python tools/sync_vendor.py           # parent -> vendor/
python tools/sync_vendor.py --check   # fail if vendor/ has drifted
```

The direction is one-way. `install.sh` over there runs `--check`, so drift
fails the install instead of going unnoticed — which is how the two versions
silently forked for months before the trees were merged.

## What changed relative to the older layout

This repository no longer uses the older top-level scripts mentioned in the previous README such as `boundaries.py`, `compute_reference_field.py`, `plot_results.py`, or `triangle.py`.

The current workflow is driven by `run.py` and the `rayleigh_cloak` package.

## Running the code

Default cloak simulation:

```bash
python run.py
```

Reference simulation:

```bash
python run.py configs/reference.yaml
```

Programmatic use:

```python
from rayleigh_cloak import load_config, solve, solve_reference

config = load_config("configs/default.yaml")
result = solve(config)

ref_result = solve_reference(config)
```

## Config format

The YAML files are parsed by `SimulationConfig` in `rayleigh_cloak/config.py`.

Top-level keys currently used by the solver are:

- `is_reference`
- `geometry_type`
- `material`
- `domain`
- `geometry`
- `absorbing`
- `mesh`
- `source`
- `solver`
- `output_dir`

Important notes:

- `geometry_type` currently supports `triangular`.
- `is_reference: true` keeps the same extended domain and absorbing layers but disables the cloak/defect behavior.
- `configs/reference.yaml` contains an extra `material.mu_factor` field, but `MaterialConfig` currently only uses `rho0` and `cs`.

## Outputs

Running `python run.py` writes outputs under `output/`.

Generated files include:

- `results_<f_star>.npz`
- `results_<f_star>.vtk`
- `_cloak_mesh.msh`
- `cloak_vtk_full.png`
- `cloak_vtk_phys.png`
- `cloak_vtk_re_uy_full.png`
- `cloak_vtk_re_uy.png`
- `cloak_vtk_re_mag.png`

### NPZ contents

The NPZ writer stores:

- `u` — nodal degrees of freedom with shape `(num_nodes, 4)`
- `pts_x`, `pts_y`
- `x_src`, `y_top`
- `x_off`, `y_off`
- `W`, `H`
- `x_src_phys`
- `f_star`

Per-node ordering in `u` is:

- `u[:, 0]` = Re($u_x$)
- `u[:, 1]` = Re($u_y$)
- `u[:, 2]` = Im($u_x$)
- `u[:, 3]` = Im($u_y$)

### VTK contents

The VTK writer stores:

- point data `mag_u`
- point data `u`
- field data `x_src`, `y_top`, `x_off`, `y_off`, `W`, `H`, `x_src_phys`, `f_star`

## Plotting saved results

The plotting utility lives in `rayleigh_cloak/plot.py`.

Plot a saved NPZ:

```bash
python -m rayleigh_cloak.plot output/results_2.00.npz
```

Plot a saved VTK:

```bash
python -m rayleigh_cloak.plot output/results_2.00.vtk
```

Optional extra arguments are:

```bash
python -m rayleigh_cloak.plot <path> <percentile> <norm_type>
```

where `norm_type` is one of:

- `linear`
- `sigmoid`
- `asym_sigmoid`

## Solver/model summary

- 2D frequency-domain elastodynamics.
- Four DOFs per node: Re($u_x$), Re($u_y$), Im($u_x$), Im($u_y$).
- Absorbing layers are modeled with a position-dependent Rayleigh-damping ratio $\xi(x)$.
- The current geometry implementation is a triangular cloak beneath the free surface.
- The mesh generator always reads triangle cells from Gmsh output.

## Environment notes

The checked-in `environment.yml` looks like a fully exported environment, not a minimal cross-platform spec. It includes many platform-specific packages, including Linux-only entries, so it should not be assumed to work unchanged on macOS or Apple Silicon.

If `conda env create -f environment.yml` works on your machine, use it. Otherwise create a fresh environment and install the packages actually used by the codebase, notably:

- `numpy`
- `pyyaml`
- `pydantic`
- `jax`
- `jax-fem`
- `gmsh`
- `meshio`
- `matplotlib`
- `vtk`

You will also need whatever PETSc/JAX-FEM stack your local setup requires for the linear solve.

## Current caveats

- Some docstrings and tests still refer to the older monolithic script layout.
- In particular, several files under `tests/` still import `boundaries.py`, which is not present in the current package-based layout.
- The README should therefore be treated as the description of the current runtime workflow, not of the older pre-refactor structure.
