import torch
import numpy as np

from microstructure_generation_2d.eval_nn_comparison import _decode_to_50
from microstructure_generation_2d.network.model_trainer import DiffusionModel
from .fem import build_cloak_layout, tile_image, structure_cloaking_loss
from .load_pretrained import load_scalers, load_neural_field


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

        self.scaler_C11, self.scaler_C12, self.scaler_C66 = load_scalers(self.diffusion_config.scaler_dir)
        diffusion = DiffusionModel.load_from_checkpoint(self.diffusion_config.ckpt, map_location=device).to(device)
        diffusion.eval()
        self.generator = diffusion.ema_model
        self.generator.eval()


        self.scaler_C11, self.scaler_C12, self.scaler_C66 = load_scalers(self.diffusion_config.scaler_dir)

    def predict_with_neural_field(self):

        neural_field = load_neural_field(self.nf_config)
        neural_field.eval().to(self.device)
        c11, c12, c66, vol = ... # TODO: extract from neural field prediction

        tf = np.array([
            float(self.scaler_C11.transform([[c11]])[0, 0]),
            float(self.scaler_C12.transform([[c12]])[0, 0]),
            float(self.scaler_C66.transform([[c66]])[0, 0]),
            vol,
        ], dtype=np.float32)


        img = self.generator.sample_with_tensor(
            tensor_c=tf,
            batch_size=1,
            steps=self.diffusion_config.steps,
            tensor_w=self.diffusion_config.tensor_w,
            verbose=False,
        )
        arr = img.detach().cpu().numpy().squeeze()  # (H, H) with H in {64, 25}
        gen_cell = _decode_to_50((arr > 0).astype(np.uint8), compressed=self.diffusion_config.compressed)

        return gen_cell
    
    def _scaler_params(self):
        """Cache (mean, scale) tensors for a differentiable StandardScaler transform.

        Pulls the fitted params straight out of the sklearn StandardScalers so the
        condition transform can run in torch and keep the graph (sklearn's
        `.transform` returns numpy and would detach the neural field).
        """
        if not hasattr(self, "_cond_mean"):
            self._cond_mean = torch.tensor(
                [self.scaler_C11.mean_[0], self.scaler_C12.mean_[0], self.scaler_C66.mean_[0]],
                dtype=torch.float32, device=self.device,
            )
            self._cond_scale = torch.tensor(
                [self.scaler_C11.scale_[0], self.scaler_C12.scale_[0], self.scaler_C66.scale_[0]],
                dtype=torch.float32, device=self.device,
            )
        return self._cond_mean, self._cond_scale

    def _to_condition_torch(self, c11, c12, c66, vol):
        """Differentiable version of the scaled diffusion condition.

        c11/c12/c66/vol are 0-d torch tensors from the neural field (with grad).
        vol is passed through unscaled, matching the numpy `predict_with_neural_field`
        path. Returns a (TENSOR_DIM,) tensor whose graph reaches the field params.
        """
        mean, scale = self._scaler_params()
        moduli = (torch.stack([c11, c12, c66]) - mean) / scale  # (3,)
        return torch.cat([moduli, vol.reshape(1)])              # (4,)

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
        """Map a cell's (C_flat, rho) from the neural field to a scaled diffusion
        condition (C11, C12, C66, vol).

        ``cell_C_flat`` carries the stiffness components (C11, C12, C66) directly
        — the same components the diffusion was conditioned on and the scalers
        were fit on, so they are scaled as-is (no lambda/mu detour).
        """
        c11, c12, c66 = (float(x) for x in np.asarray(cell_C_flat).ravel()[:3])
        # vol = solid volume fraction. homogenize_simp uses rho_eff = vol * rho_solid,
        # so invert: vol = rho_eff / rho_solid. (4th condition is passed unscaled.)
        rho_solid = self._cloak_layout().dp.rho0
        vol = float(np.clip(cell_rho / rho_solid, 0.0, 1.0))
        return np.array([
            float(self.scaler_C11.transform([[c11]])[0, 0]),
            float(self.scaler_C12.transform([[c12]])[0, 0]),
            float(self.scaler_C66.transform([[c66]])[0, 0]),
            vol,
        ], dtype=np.float32)

    def _assemble_canvas(self, x_start, cloak_idx, n_cells, n_x, n_y):
        """Decode the per-cloak-cell predicted-clean images and tile them into the
        full structure (solid cement outside the cloak).

        Forward (no-grad) path: binarized `_decode_to_50` -> uint8 canvas. The
        torch<->JAX bridge will replace this with a differentiable soft-occupancy
        decode + tile so gradients reach `x_start`.
        """
        arrs = x_start.detach().cpu().numpy()[:, 0]  # (n_cloak, H, H)
        cloak_geoms = np.stack([
            _decode_to_50((a > 0).astype(np.uint8), compressed=self.diffusion_config.compressed)
            for a in arrs
        ])  # (n_cloak, 50, 50)
        H = cloak_geoms.shape[1]
        geoms = np.ones((n_cells, H, H), dtype=np.uint8)
        geoms[cloak_idx] = cloak_geoms
        return tile_image(geoms, n_x, n_y)

    def predict_structure(self, X):
        """
        Optimize the neural field DURING a single diffusion sampling trajectory.

        One sampling run of `steps` denoise steps, batched over all cloak cells.
        After *each* step we decode the current predicted-clean microstructures,
        tile them into the full cloak structure, run the pixel-level full-structure
        FEM, compute the cloaking loss, and take one optimizer step on the (JAX)
        neural field — whose updated per-cell targets re-condition the next
        diffusion step. Hence #optimizer steps == #diffusion steps.

        NB: this runs the full-structure FEM once per diffusion step (expensive).
        """
        layout = self._cloak_layout()
        decomp = layout.decomp
        cloak_idx = np.where(decomp.cloak_mask)[0]
        n_cells = decomp.cloak_mask.shape[0]
        n_cloak = cloak_idx.shape[0]

        neural_field = load_neural_field(self.nf_config)
        # TODO: set up the JAX Adam state for the neural-field params here (once).

        gen = self.generator
        tensor_w = self.diffusion_config.tensor_w
        img, tensor_zero, time_pairs = gen.prepare_sampling(
            batch_size=n_cloak, steps=self.diffusion_config.steps
        )
        x_start = None
        canvas = None

        for step, (time, time_next) in time_pairs:
            # (1) Per-cloak-cell conditions from the CURRENT neural field.
            # TODO: reparam.decode(theta) -> (cell_C: (n_cells, n_C), cell_rho: (n_cells,)).
            cell_C, cell_rho = ...
            cell_C, cell_rho = np.asarray(cell_C), np.asarray(cell_rho)
            conds = np.stack([
                self._cell_condition(cell_C[i], float(cell_rho[i])) for i in cloak_idx
            ]).astype(np.float32)
            tensor_cond = torch.as_tensor(conds, device=self.device)  # (n_cloak, TENSOR_DIM)

            # (2) One diffusion step for all cloak cells.
            img, x_start = gen.denoise_step(
                img, x_start, time, time_next, tensor_cond, tensor_zero, tensor_w
            )

            # (3) Decode current microstructures + tile into the full structure.
            canvas = self._assemble_canvas(x_start, cloak_idx, n_cells, layout.n_x, layout.n_y)

            # (4) Pixel-level full-structure FEM cloaking loss.
            loss, u_val, diag = self.compute_fem_loss(canvas)

            # (5) One optimizer step on the JAX neural field via the torch<->JAX
            #     custom_vjp bridge (TODO): backprop loss -> canvas -> diffusion ->
            #     conditions -> neural field. The updated NF re-conditions step+1.

            # (6) Detach carryover so each step builds its own single-step graph.
            img = img.detach()
            x_start = x_start.detach()

        return canvas

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