from dataclasses import dataclass
from typing import Callable, Optional

import torch
import numpy as np

from dataset.cellular_chiral.diffusion_dataset import CELL_SIZE, PAD_TO, unfold_mirror
from microstructure_generation_2d.eval_nn_comparison import _decode_to_50
from microstructure_generation_2d.network.model_trainer import DiffusionModel
from rayleigh_cloak.materials import _get_converters
from .fem import (
    build_cloak_layout,
    build_pixel_objective,
    structure_cloaking_loss,
    tile_image,
)
from .load_pretrained import load_scalers, load_neural_field

_OFF = (PAD_TO - CELL_SIZE) // 2  # 7-pixel void border for legacy 64x64 mode


def moduli_in_diffusion_order(C, xp=np):
    """Read ``(C11, C22, C12, C66)`` from a full ``(..., 2, 2, 2, 2)`` stiffness
    tensor in the exact order the 2D diffusion was conditioned on.

    Mirrors ``dataset.cellular_chiral.bulk_stiffness._ortho_params_from_C`` (which
    reads the augmented-Voigt matrix): ``C11 = C[0,0,0,0]``, ``C22 = C[1,1,1,1]``,
    ``C12 = C[0,0,1,1]`` (lateral coupling, Voigt ``[0,1]``), ``C66 = C[0,1,0,1]``
    (engineering shear, Voigt ``[2,2]``). Going through the full tensor makes this
    independent of the neural field's flat layout (``n_C_params``); e.g. the flat4
    ``C_to_flatC`` order is ``[C11, C22, C66, C12]``, which this reorders correctly.

    ``xp`` is the array module — ``numpy`` for the eval path, ``jax.numpy`` inside
    the autodiff bridge.
    """
    return xp.stack([
        C[..., 0, 0, 0, 0],  # C11
        C[..., 1, 1, 1, 1],  # C22
        C[..., 0, 0, 1, 1],  # C12
        C[..., 0, 1, 0, 1],  # C66
    ], axis=-1)


@dataclass
class TrajectoryStep:
    """One step of the ``predict_structure`` trajectory, handed to ``on_step``.

    The optimisation loop emits this after each diffusion/Adam step so callers
    can log, plot, or checkpoint without the loop owning any I/O.

    Attributes
    ----------
    step : 0-based step index (== diffusion step == optimizer step).
    n_steps : total steps in the trajectory.
    loss : the full-structure FEM cloaking loss at this step (lower = better).
    canvas : (n_y*CELL_SIZE, n_x*CELL_SIZE) soft occupancy in [0, 1] — the tiled
        "combined topology" (cloak cells = generated microstructure, the rest
        solid). Threshold at 0.5 for the binary structure.
    theta : the MLP weights *after* this step's update (for checkpointing/resume).
    """
    step: int
    n_steps: int
    loss: float
    canvas: np.ndarray
    theta: list


# A per-step callback: receives the TrajectoryStep, returns nothing.
StepCallback = Callable[[TrajectoryStep], None]


def tile_image_torch(geoms: torch.Tensor, n_x: int, n_y: int) -> torch.Tensor:
    """Differentiable torch port of :func:`multiscale_generation.fem.tile_image`.

    Tiles ``(n_cells, H, W)`` into ``(n_y*H, n_x*W)`` with the same y-up,
    ``cell_idx = ix*n_y + iy`` layout (``row = n_y-1-iy``, ``col = ix``) via a
    flip + permute + reshape so gradients flow back to ``geoms``.
    """
    n_cells, H, W = geoms.shape
    assert n_cells == n_x * n_y, f"{n_cells} != {n_x}*{n_y}"
    g = geoms.reshape(n_x, n_y, H, W)        # [ix, iy, H, W]
    g = torch.flip(g, dims=(1,))             # [ix, n_y-1-iy, H, W]
    # → [iy_row, H, ix_col, W] → (n_y*H, n_x*W); block(row, col) = geoms[col*n_y + n_y-1-row]
    return g.permute(1, 2, 0, 3).reshape(n_y * H, n_x * W)


class MultiscaleDiffusionModel:
    def __init__(self, device,
                 diffusion_config,
                 cell_decomposition_config,
                 neural_field_config
                 ):
        self.diffusion_config = diffusion_config
        self.cell_decomposition_config = cell_decomposition_config
        self.nf_config = neural_field_config
        self.device = device

        # Diffusion condition scalers, in the model's order (C11, C22, C12, C66).
        (self.scaler_C11, self.scaler_C22,
         self.scaler_C12, self.scaler_C66) = load_scalers(self.diffusion_config.scaler_dir)

        diffusion = DiffusionModel.load_from_checkpoint(str(self.diffusion_config.ckpt), map_location=device).to(device)
        diffusion.eval()
        self.generator = diffusion.ema_model
        self.generator.eval()

    @property
    def compressed(self) -> bool:
        """Decode mode: explicit ``diffusion_config.compressed`` wins; if it is
        ``None`` (or absent), trust the checkpoint's own layout flag (a 25x25
        quadrant generator => compressed)."""
        cfg = getattr(self.diffusion_config, "compressed", None)
        if cfg is not None:
            return bool(cfg)
        return bool(getattr(self.generator, "compressed", False))

    @classmethod
    def from_config(cls, device, config):
        """Build from a :class:`multiscale_generation.config.MultiscaleConfig`."""
        return cls(device, config.diffusion, config.cell_decomposition, config.neural_field)

    def predict_with_neural_field(self):
        # Older single-cell numpy path (pre-bridge); not wired. The trajectory-
        # coupled optimisation lives in ``predict_structure``. ``load_neural_field``
        # now returns the JAX ``(reparam, theta)`` (see load_pretrained), so this
        # stub's single-cell extraction needs reworking before use.
        raise NotImplementedError(
            "predict_with_neural_field is a stub; use predict_structure for the "
            "torch<->JAX gradient bridge."
        )

        reparam, theta = load_neural_field(
            self.nf_config, self._cloak_layout().decomp, self._neural_field_init_params()
        )
        cell_C, cell_rho = reparam.decode(theta)  # (n_cells, n_C), (n_cells,)
        tf = self._cell_condition(cell_C[0], cell_rho[0])  # (5,) for one cell

        img = self.generator.sample_with_tensor(
            tensor_c=tf,
            batch_size=1,
            steps=self.diffusion_config.steps,
            tensor_w=self.diffusion_config.tensor_w,
            verbose=False,
        )
        arr = img.detach().cpu().numpy().squeeze()  # (H, H) with H in {64, 25}
        gen_cell = _decode_to_50((arr > 0).astype(np.uint8), compressed=self.compressed)

        return gen_cell
    
    def _scaler_params(self):
        """Cache (mean, scale) tensors for a differentiable StandardScaler transform.

        Pulls the fitted params straight out of the sklearn StandardScalers so the
        condition transform can run in torch and keep the graph (sklearn's
        `.transform` returns numpy and would detach the neural field).
        """
        if not hasattr(self, "_cond_mean"):
            scalers = (self.scaler_C11, self.scaler_C22, self.scaler_C12, self.scaler_C66)
            self._cond_mean = torch.tensor(
                [s.mean_[0] for s in scalers], dtype=torch.float32, device=self.device,
            )
            self._cond_scale = torch.tensor(
                [s.scale_[0] for s in scalers], dtype=torch.float32, device=self.device,
            )
        return self._cond_mean, self._cond_scale

    def _scaler_params_jnp(self):
        """JAX (mean, scale) for the differentiable condition transform.

        Same fitted StandardScaler params as :meth:`_scaler_params`, but as jnp
        arrays so ``conditions_fn`` (which feeds ``jax.vjp``) stays pure-jnp and
        keeps the graph to the neural-field weights.
        """
        if not hasattr(self, "_cond_mean_jnp"):
            import jax.numpy as jnp
            scalers = (self.scaler_C11, self.scaler_C22, self.scaler_C12, self.scaler_C66)
            self._cond_mean_jnp = jnp.asarray([s.mean_[0] for s in scalers], dtype=jnp.float32)
            self._cond_scale_jnp = jnp.asarray([s.scale_[0] for s in scalers], dtype=jnp.float32)
        return self._cond_mean_jnp, self._cond_scale_jnp

    def _jnp_to_torch(self, arr, dtype=torch.float32, requires_grad=False):
        """Move a jnp array to a torch tensor on ``self.device`` via host numpy.

        Host round-trip is device-agnostic (works whether JAX/torch are on CPU or
        the same GPU) and sidesteps dlpack placement pitfalls; the per-step
        transfers are tiny so the overhead is negligible vs the FEM solve.
        """
        t = torch.tensor(np.asarray(arr), device=self.device, dtype=dtype)
        if requires_grad:
            t.requires_grad_(True)
        return t

    def _torch_to_jnp(self, t):
        """Move a torch tensor to a jnp array (on JAX's default device) via host numpy."""
        import jax.numpy as jnp
        return jnp.asarray(t.detach().cpu().numpy())

    def _to_condition_torch(self, c11, c22, c12, c66, vol):
        """Differentiable version of the scaled diffusion condition.

        c11/c22/c12/c66/vol are 0-d torch tensors from the neural field (with
        grad), already in the diffusion order. The four moduli are z-scored with
        the fitted scalers; vol is passed through unscaled. Returns a
        ``(TENSOR_DIM,)`` = ``(5,)`` tensor whose graph reaches the field params.
        """
        mean, scale = self._scaler_params()
        moduli = (torch.stack([c11, c22, c12, c66]) - mean) / scale  # (4,)
        return torch.cat([moduli, vol.reshape(1)])                   # (5,)

    def _rayleigh_config(self):
        """Load + cache the rayleigh ``SimulationConfig`` describing the cloak."""
        if not hasattr(self, "_ray_cfg"):
            from rayleigh_cloak import load_config
            # TODO: confirm where the rayleigh config lives on cell_decomposition_config.
            self._ray_cfg = load_config(self.cell_decomposition_config.config_path)
        return self._ray_cfg

    def _cloak_layout(self):
        """Geometry + cell grid + cloak mask + bbox (cached)."""
        if not hasattr(self, "_layout"):
            self._layout = build_cloak_layout(self._rayleigh_config())
        return self._layout

    def _cell_condition(self, cell_C_flat, cell_rho):
        """Map per-cell ``(C_flat, rho)`` from the neural field to scaled diffusion
        conditions ``(C11, C22, C12, C66, vol)`` — the exact 5-dim order the model
        was trained on (``TENSOR_DIM``; see ``microstructure_generation_2d``).

        ``cell_C_flat`` is ``(n, n_C)`` (a single ``(n_C,)`` is also accepted) in
        the neural field's flat layout. It is reconstructed to the full stiffness
        tensor and the four moduli are read out in diffusion order (see
        :func:`moduli_in_diffusion_order`) **before** z-scoring — so this is correct
        for any ``n_C_params`` (e.g. flat4's ``[C11, C22, C66, C12]`` is reordered to
        ``[C11, C22, C12, C66]``). vol is the solid volume fraction
        ``rho_eff / rho_solid`` (homogenize_simp uses ``rho_eff = vol * rho_solid``),
        passed through unscaled. Returns ``(n, 5)`` (or ``(5,)`` for a single cell).
        """
        import jax  # the flat<->tensor converters are jnp-based
        import jax.numpy as jnp
        flat = np.atleast_2d(np.asarray(cell_C_flat, dtype=np.float64))  # (n, n_C)
        _, from_flat = _get_converters(flat.shape[1])
        C = np.asarray(jax.vmap(from_flat)(jnp.asarray(flat)))           # (n, 2,2,2,2)
        moduli = moduli_in_diffusion_order(C, xp=np)                     # (n, 4)

        scalers = (self.scaler_C11, self.scaler_C22, self.scaler_C12, self.scaler_C66)
        scaled = np.stack(
            [s.transform(moduli[:, i:i + 1])[:, 0] for i, s in enumerate(scalers)], axis=1
        )  # (n, 4)

        # vol = solid volume fraction; invert rho_eff = vol * rho_solid.
        rho_solid = self._cloak_layout().dp.rho0
        rho = np.atleast_1d(np.asarray(cell_rho, dtype=np.float64))
        vol = np.clip(rho / rho_solid, 0.0, 1.0)
        out = np.concatenate([scaled, vol[:, None]], axis=1).astype(np.float32)  # (n, 5)
        return out[0] if np.ndim(cell_C_flat) == 1 else out

    def _assemble_canvas(self, x_start, cloak_idx, n_cells, n_x, n_y):
        """Decode the per-cloak-cell predicted-clean images and tile them into the
        full structure (solid cement outside the cloak).

        Forward (no-grad) path: binarized `_decode_to_50` -> uint8 canvas. The
        torch<->JAX bridge will replace this with a differentiable soft-occupancy
        decode + tile so gradients reach `x_start`.
        """
        arrs = x_start.detach().cpu().numpy()[:, 0]  # (n_cloak, H, H)
        cloak_geoms = np.stack([
            _decode_to_50((a > 0).astype(np.uint8), compressed=self.compressed)
            for a in arrs
        ])  # (n_cloak, 50, 50)
        H = cloak_geoms.shape[1]
        geoms = np.ones((n_cells, H, H), dtype=np.uint8)
        geoms[cloak_idx] = cloak_geoms
        return tile_image(geoms, n_x, n_y)

    def _assemble_canvas_torch(self, x_start, cloak_idx, n_cells, n_x, n_y):
        """Differentiable decode + tile: ``x_start -> canvas`` (grad reaches x_start).

        Soft-occupancy counterpart of :meth:`_assemble_canvas` used inside the
        torch↔JAX bridge. ``occ = (x_start + 1)/2`` in [0, 1] is decoded to the
        50x50 cell (mirror-unfold for compressed, centre-crop for legacy), the
        cloak cells are scattered into an all-solid ``(n_cells, 50, 50)`` field
        (non-cloak cells are constant 1.0, no grad), and the field is tiled with
        :func:`tile_image_torch`.
        """
        occ = ((x_start + 1.0) / 2.0)[:, 0]  # (n_cloak, S, S) in [0, 1]
        if self.compressed:
            cells = unfold_mirror(occ)       # (n_cloak, 50, 50)
        else:
            cells = occ[..., _OFF:_OFF + CELL_SIZE, _OFF:_OFF + CELL_SIZE]
        H = cells.shape[-1]
        geoms = torch.ones((n_cells, H, H), device=self.device, dtype=cells.dtype)
        cloak_idx_t = torch.as_tensor(np.asarray(cloak_idx), device=self.device, dtype=torch.long)
        geoms = geoms.index_copy(0, cloak_idx_t, cells)  # out-of-place: grad flows to `cells`
        return tile_image_torch(geoms, n_x, n_y)

    def _neural_field_init_params(self):
        """Initial ``(cell_C_flat, cell_rho)`` for the cloak cells (from the
        continuous transformational push-forward), used to build the NeuralReparam.

        Mirrors the ``CellMaterial`` setup in ``solve_optimization_neural`` so the
        reparam reconstructed at inference matches the one optimised at train time.
        """
        from rayleigh_cloak.materials import C_iso, CellMaterial
        layout = self._cloak_layout()
        cfg = self._rayleigh_config()
        C0 = C_iso(layout.dp.lam, layout.dp.mu)
        cell_mat = CellMaterial(
            layout.geometry, C0, layout.dp.rho0, layout.decomp,
            n_C_params=cfg.cells.n_C_params,
            symmetrize_init=cfg.cells.symmetrize_init,
            init=cfg.cells.init,
            init_path=cfg.cells.init_path,
        )
        return cell_mat.get_initial_params()

    def _load_neural_field(self):
        """Build the JAX ``(reparam, theta)`` for the bridge (see load_pretrained)."""
        return load_neural_field(
            self.nf_config, self._cloak_layout().decomp, self._neural_field_init_params()
        )

    def predict_structure(self, X=None, lr=1e-3, refinement_factor=None,
                          void_ratio=1e-6, simp_p=3.0, binarize=False,
                          on_step: Optional[StepCallback] = None):
        """
        Optimize the neural field DURING a single diffusion sampling trajectory.

        One sampling run of `steps` denoise steps, batched over all cloak cells.
        After *each* step we decode the current predicted-clean microstructures,
        tile them into the full cloak structure, run the pixel-level full-structure
        FEM, compute the cloaking loss, and take one optimizer step on the (JAX)
        neural field — whose updated per-cell targets re-condition the next
        diffusion step. Hence #optimizer steps == #diffusion steps.

        The gradient ``loss → canvas → diffusion → conditions → θ`` crosses the
        torch/JAX boundary by chaining three VJPs per step (no jax.custom_vjp):

            θ (JAX) --vjp_θ--> conditions --[torch leaf]--> canvas (torch)
            canvas --[JAX]--> (loss, g_canvas)            (ad_wrapper pixel FEM)
            g_canvas --[torch.autograd.grad]--> g_conditions
            g_conditions --[JAX vjp_θ]--> g_θ ; adam_update(θ)

        NB: this runs the full-structure FEM once per diffusion step (expensive).

        ``on_step``, if given, is called once per step with a :class:`TrajectoryStep`
        (step index, loss, the soft tiled canvas, and the updated θ) — the hook the
        training driver (``multiscale_generation.optimize``) uses for loss logging,
        per-step structure images, and θ checkpoints. I/O lives in the callback,
        not in this loop.

        Returns ``(canvas, theta, loss_history)``: the final (binarized) tiled
        structure, the optimised MLP weights, and the per-step FEM loss.
        """
        import jax
        import jax.numpy as jnp
        from rayleigh_cloak.optimize import adam_init, adam_update

        layout = self._cloak_layout()
        decomp = layout.decomp
        cloak_idx = np.where(decomp.cloak_mask)[0]
        n_cells = decomp.cloak_mask.shape[0]
        n_cloak = cloak_idx.shape[0]
        n_x, n_y = layout.n_x, layout.n_y
        rho0 = layout.dp.rho0

        # ── (once) neural field + Adam + pure-jnp conditions function ──
        reparam, theta = self._load_neural_field()
        opt_state = adam_init(theta)

        cloak_idx_jnp = jnp.asarray(cloak_idx)
        cond_mean, cond_scale = self._scaler_params_jnp()
        # flat<->tensor converter for the field's layout, so the reorder below is
        # independent of n_C_params (flat4 = [C11, C22, C66, C12], etc.).
        _, from_flat = _get_converters(reparam.C_flat_init.shape[1])

        def conditions_fn(theta):
            """θ → (n_cloak, 5) scaled diffusion conditions (pure jnp).

            jnp counterpart of :meth:`_cell_condition`: reconstruct each cloak
            cell's full stiffness tensor, read ``(C11, C22, C12, C66)`` in the
            diffusion order (see :func:`moduli_in_diffusion_order`), z-score with the
            fitted scalers, and append ``vol = clip(rho/rho0, 0, 1)`` unscaled.
            """
            cell_C, cell_rho = reparam.decode(theta)       # (n_cells, n_C), (n_cells,)
            C_clk = jax.vmap(from_flat)(cell_C[cloak_idx_jnp])  # (n_cloak, 2,2,2,2)
            rho_clk = cell_rho[cloak_idx_jnp]                   # (n_cloak,)
            moduli = moduli_in_diffusion_order(C_clk, xp=jnp)   # (n_cloak, 4)
            moduli = (moduli - cond_mean) / cond_scale
            vol = jnp.clip(rho_clk / rho0, 0.0, 1.0)
            return jnp.concatenate([moduli, vol[:, None]], axis=1)  # (n_cloak, 5)

        # ── (once) differentiable pixel-FEM objective (canvas → loss, g_canvas) ──
        canvas_shape = (n_y * CELL_SIZE, n_x * CELL_SIZE)
        objective = build_pixel_objective(
            self._rayleigh_config(), layout.cloak_bbox, canvas_shape,
            refinement_factor=refinement_factor,
            void_ratio=void_ratio, simp_p=simp_p, binarize=binarize,
        )

        gen = self.generator
        tensor_w = self.diffusion_config.tensor_w
        n_steps = self.diffusion_config.steps
        img, tensor_zero, time_pairs = gen.prepare_sampling(
            batch_size=n_cloak, steps=n_steps
        )
        x_start = None
        canvas = None
        loss_history: list[float] = []

        for step, (time, time_next) in time_pairs:
            # (1) Per-cloak-cell conditions from the CURRENT neural field (JAX),
            #     capturing the VJP so we can pull g_conditions back to θ.
            conditions_jnp, vjp_theta = jax.vjp(conditions_fn, theta)
            tensor_cond = self._jnp_to_torch(conditions_jnp, requires_grad=True)  # torch leaf

            # (2) One diffusion step for all cloak cells (grad-enabled).
            img, x_start = gen.denoise_step(
                img, x_start, time, time_next, tensor_cond, tensor_zero, tensor_w
            )

            # (3) Differentiable decode + tile into the full structure (torch).
            canvas_torch = self._assemble_canvas_torch(x_start, cloak_idx, n_cells, n_x, n_y)

            # (4) Pixel-level full-structure FEM loss + grad w.r.t. canvas (JAX).
            loss, g_canvas_jnp = objective(self._torch_to_jnp(canvas_torch))
            loss_history.append(float(loss))

            # (5) Chain VJPs back to θ and take one Adam step.
            #     g_canvas --autograd.grad--> g_conditions --vjp_θ--> g_θ.
            g_canvas_torch = self._jnp_to_torch(g_canvas_jnp)
            g_cond_torch, = torch.autograd.grad(canvas_torch, tensor_cond, grad_outputs=g_canvas_torch)
            # The cotangent must match conditions_jnp's dtype (torch is f32; the NF
            # may decode to f64), so cast before crossing back into vjp_theta.
            g_cond_jnp = self._torch_to_jnp(g_cond_torch).astype(conditions_jnp.dtype)
            g_theta, = vjp_theta(g_cond_jnp)
            updates, opt_state = adam_update(g_theta, opt_state, lr=lr)
            theta = jax.tree.map(lambda p, u: p + u, theta, updates)

            # (6) Emit the step event (logging / plotting / checkpoint live here).
            if on_step is not None:
                on_step(TrajectoryStep(
                    step=step, n_steps=n_steps, loss=float(loss),
                    canvas=canvas_torch.detach().cpu().numpy(), theta=theta,
                ))

            # (7) Detach carryover so each step builds its own single-step graph.
            img = img.detach()
            x_start = x_start.detach()

        # Final (binarized) structure from the last predicted-clean microstructures.
        canvas = self._assemble_canvas(x_start, cloak_idx, n_cells, n_x, n_y)
        return canvas, theta, loss_history

    def compute_fem_loss(self, canvas, refinement_factor=None):
        """
        Run the pixel-level full-structure FEM on the tiled cloak ``canvas`` and
        return the cloaking loss (transmitted-displacement ratio; lower = better
        cloaking). See multiscale_generation/fem.py.

        Returns ``(loss, u_val, diag)``.
        """
        layout = self._cloak_layout()
        return structure_cloaking_loss(
            canvas,
            self._rayleigh_config(),
            layout.cloak_bbox,
            refinement_factor=refinement_factor,
            # TODO: expose void_ratio / simp_p / binarize via config if needed.
        )