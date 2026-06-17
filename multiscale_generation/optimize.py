"""Training driver for the multiscale neural-field bridge.

Wraps :meth:`MultiscaleDiffusionModel.predict_structure` — which optimises the
neural field *during* one diffusion sampling trajectory (one Adam step per
diffusion step) — with disk-backed **callbacks**, so the bridge loop stays free of
any I/O:

* :class:`LossLogger`       — append ``step,loss,ratio`` to ``loss_history.csv``.
* :class:`StructurePlotter` — save the tiled "combined topology" image each step.
* :class:`ThetaCheckpointer`— periodically ``save_theta`` for resume.

Callbacks are the right seam here: ``predict_structure`` already owns the loop and
emits a :class:`TrajectoryStep` per step (step index, FEM loss, the soft tiled
canvas, and the updated θ). Each callback consumes that event independently, so
logging / plotting / checkpointing compose without entangling the optimisation.

CLI — from a single YAML (see ``configs/multiscale_generation/``)::

    python -m multiscale_generation.optimize --yaml configs/multiscale_generation/v1.yaml

or with explicit per-field flags::

    python -m multiscale_generation.optimize \\
        --ckpt output/diffusion_ca_squared/v1_ortho/last.ckpt \\
        --config output/cell20_flat4_materialreg/config.yaml \\
        --init-weights output/cell20_flat4_materialreg/best_weights.npz \\
        --nf-hidden 512 --nf-layers 6 --nf-fourier 64 \\
        --steps 50 --lr 1e-3 --refinement-factor 2 \\
        --output-dir output/multiscale_run1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rayleigh_cloak.neural_reparam import save_theta

from .config import (
    CellDecompositionConfig, DiffusionConfig, MultiscaleConfig,
    NeuralFieldConfig, OptimizeConfig,
)
from .model import MultiscaleDiffusionModel, TrajectoryStep


# ── callbacks ────────────────────────────────────────────────────────


class LossLogger:
    """Append ``step,loss,ratio`` to a CSV (flushed each step) and keep the history.

    ``loss`` is the SIMP-soft cloaking loss the optimiser descends; ``ratio`` is the
    transmitted-displacement ratio of the binarized physical structure (1.0 =
    perfect cloak), the metric ``scripts/frequency_sweep.py`` reports.
    """

    def __init__(self, path):
        self.path = Path(path)
        self._f = open(self.path, "w", buffering=1)  # line-buffered
        self._f.write("step,loss,ratio\n")
        self.history: list[float] = []
        self.ratio_history: list[float] = []

    def __call__(self, s: TrajectoryStep) -> None:
        self.history.append(s.loss)
        self.ratio_history.append(s.ratio)
        self._f.write(f"{s.step},{s.loss:.8e},{s.ratio:.8e}\n")

    def close(self) -> None:
        self._f.close()


class StructurePlotter:
    """Save the tiled combined-topology canvas as a PNG every ``every`` steps.

    ``s.canvas`` is the soft occupancy in [0, 1] (cloak cells = generated
    microstructure, the rest solid). With ``binarize`` the canvas is thresholded
    at 0.5 first (the physical binary structure).
    """

    def __init__(self, frames_dir, every: int = 1, binarize: bool = False,
                 cmap: str = "gray_r"):
        self.frames_dir = Path(frames_dir)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.every = every
        self.binarize = binarize
        self.cmap = cmap

    def __call__(self, s: TrajectoryStep) -> None:
        if self.every <= 0:
            return
        is_last = s.step == s.n_steps - 1
        if (s.step % self.every != 0) and not is_last:
            return
        img = (s.canvas > 0.5).astype(float) if self.binarize else s.canvas
        save_canvas_png(
            img, self.frames_dir / f"step_{s.step:04d}.png",
            title=f"step {s.step + 1}/{s.n_steps}   loss={s.loss:.4e}   ratio={s.ratio:.4f}",
            cmap=self.cmap,
        )


class ThetaCheckpointer:
    """Save the MLP weights every ``every`` steps (0 = nothing here; final is
    written by :func:`run_optimization`)."""

    def __init__(self, out_dir, every: int = 0):
        self.out_dir = Path(out_dir)
        self.every = every

    def __call__(self, s: TrajectoryStep) -> None:
        if self.every <= 0:
            return
        if (s.step % self.every == 0) or (s.step == s.n_steps - 1):
            save_theta(s.theta, str(self.out_dir / f"theta_step_{s.step:04d}.npz"))


def compose(*callbacks):
    """Chain callbacks into one ``on_step``; ``None`` entries are skipped."""
    cbs = [c for c in callbacks if c is not None]

    def _run(s: TrajectoryStep) -> None:
        for c in cbs:
            c(s)

    return _run


# ── plotting helpers ─────────────────────────────────────────────────


def save_canvas_png(canvas, path, title=None, cmap="gray_r"):
    """Render a (H, W) occupancy canvas to ``path`` (headless / Agg)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = canvas.shape
    fig, ax = plt.subplots(figsize=(max(W / 120, 3.0), max(H / 120, 3.0)))
    ax.imshow(canvas, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=110, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def save_loss_curve(history, path):
    """Plot the per-step FEM cloaking loss to ``path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history)), history, marker="o", ms=3)
    ax.set_xlabel("step")
    ax.set_ylabel("FEM cloaking loss")
    if history and all(h > 0 for h in history):
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_ratio_curve(history, path):
    """Plot the per-step transmitted-displacement ratio (1.0 = perfect cloak)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(len(history)), history, marker="o", ms=3, color="C0")
    ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.6, label="perfect cloak")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\langle |u| \rangle \,/\, \langle |u_{\rm ref}| \rangle$")
    ax.set_title("Physical (binarized) cloaking ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def try_make_gif(frames_dir, out_path, fps: int = 5) -> bool:
    """Best-effort GIF of the per-step structure frames (needs Pillow)."""
    try:
        from PIL import Image
    except ImportError:
        return False
    files = sorted(Path(frames_dir).glob("step_*.png"))
    if not files:
        return False
    frames = [Image.open(f).convert("RGB") for f in files]
    size = frames[0].size  # tight bbox => sizes can vary; normalise to the first
    frames = [im if im.size == size else im.resize(size) for im in frames]
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / max(fps, 1)), loop=0)
    return True


# ── driver ───────────────────────────────────────────────────────────


def run_optimization(config: MultiscaleConfig, output_dir, device=None):
    """Run one ``predict_structure`` trajectory and persist all artifacts.

    Writes, under ``output_dir``: ``multiscale_config.json`` (provenance),
    ``loss_history.csv`` (``step,loss,ratio``) + ``loss_curve.png`` +
    ``ratio_curve.png``, ``frames/step_*.png`` (and a ``structure_evolution.gif``
    if Pillow is present), ``final_canvas.npy`` + ``final_structure.png``, and
    ``theta_final.npz`` (plus periodic ``theta_step_*.npz`` if
    ``optimize.ckpt_every`` > 0).

    Returns ``{"canvas", "theta", "loss_history", "ratio_history", "output_dir"}``.
    """
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "multiscale_config.json").write_text(config.model_dump_json(indent=2))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiscaleDiffusionModel.from_config(device, config)

    opt = config.optimize
    loss_logger = LossLogger(out / "loss_history.csv")
    on_step = compose(
        loss_logger,
        StructurePlotter(out / "frames", every=opt.plot_every, binarize=opt.binarize),
        ThetaCheckpointer(out, every=opt.ckpt_every),
    )

    try:
        canvas, theta, loss_history = model.predict_structure(
            lr=opt.lr, refinement_factor=opt.refinement_factor,
            void_ratio=opt.void_ratio, simp_p=opt.simp_p, binarize=opt.binarize,
            freeze=opt.freeze,
            on_step=on_step,
        )
    finally:
        loss_logger.close()

    ratio_history = loss_logger.ratio_history

    # Final artifacts.
    np.save(out / "final_canvas.npy", np.asarray(canvas))
    if loss_history and ratio_history:
        final_title = f"final   loss={loss_history[-1]:.4e}   ratio={ratio_history[-1]:.4f}"
    else:
        final_title = "final"
    save_canvas_png(
        np.asarray(canvas, dtype=float), out / "final_structure.png",
        title=final_title,
    )
    save_theta(theta, str(out / "theta_final.npz"))
    if loss_history:
        save_loss_curve(loss_history, out / "loss_curve.png")
    if ratio_history:
        save_ratio_curve(ratio_history, out / "ratio_curve.png")
    try_make_gif(out / "frames", out / "structure_evolution.gif")

    return {"canvas": canvas, "theta": theta,
            "loss_history": loss_history, "ratio_history": ratio_history,
            "output_dir": str(out)}


def _build_config(a: argparse.Namespace) -> MultiscaleConfig:
    return MultiscaleConfig(
        diffusion=DiffusionConfig(
            ckpt=a.ckpt, scaler_dir=a.scaler_dir, steps=a.steps,
            tensor_w=a.tensor_w, compressed=a.compressed,
        ),
        cell_decomposition=CellDecompositionConfig(config_path=a.config),
        neural_field=NeuralFieldConfig(
            hidden_size=a.nf_hidden, n_layers=a.nf_layers, n_fourier=a.nf_fourier,
            seed=a.nf_seed, output_scale=a.nf_output_scale, init_weights=a.init_weights,
        ),
        optimize=OptimizeConfig(
            lr=a.lr, refinement_factor=a.refinement_factor, void_ratio=a.void_ratio,
            simp_p=a.simp_p, binarize=a.binarize, freeze=a.freeze,
            plot_every=a.plot_every, ckpt_every=a.ckpt_every,
            output_dir=a.output_dir,
        ),
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Full-run config from a single YAML (see configs/multiscale_generation/). When
    # given, the per-field args below are ignored; --output-dir / --device still override.
    p.add_argument("--yaml", default=None,
                   help="MultiscaleConfig YAML; if set, the per-field args below are ignored")
    p.add_argument("--ckpt", help="diffusion .ckpt")
    p.add_argument("--config", help="rayleigh SimulationConfig YAML (cloak layout)")
    p.add_argument("--output-dir", default=None,
                   help="artifact dir (overrides optimize.output_dir from the YAML)")
    p.add_argument("--scaler-dir", default="microstructure_generation_2d")
    p.add_argument("--steps", type=int, default=50, help="diffusion == optimizer steps")
    p.add_argument("--tensor-w", type=float, default=2.0)
    p.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=None,
                   help="override the checkpoint's decode mode")
    # neural field (must match the warm-start weights, if any)
    p.add_argument("--nf-hidden", type=int, default=256)
    p.add_argument("--nf-layers", type=int, default=4)
    p.add_argument("--nf-fourier", type=int, default=32)
    p.add_argument("--nf-seed", type=int, default=42)
    p.add_argument("--nf-output-scale", type=float, default=0.1)
    p.add_argument("--init-weights", default=None, help="warm-start MLP weights .npz")
    # optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--refinement-factor", type=int, default=None)
    p.add_argument("--void-ratio", type=float, default=1e-6)
    p.add_argument("--simp-p", type=float, default=3.0)
    p.add_argument("--binarize", action="store_true")
    p.add_argument("--freeze", action="store_true",
                   help="disable optimisation: NF frozen, no grads, forward-only eval")
    p.add_argument("--plot-every", type=int, default=1)
    p.add_argument("--ckpt-every", type=int, default=0)
    p.add_argument("--device", default=None)
    a = p.parse_args()

    if a.yaml:
        config = MultiscaleConfig.from_yaml(a.yaml)
    else:
        if not a.ckpt or not a.config:
            p.error("--ckpt and --config are required unless --yaml is given")
        config = _build_config(a)

    # --output-dir wins; otherwise fall back to optimize.output_dir from the YAML.
    output_dir = a.output_dir or config.optimize.output_dir
    if output_dir is None:
        p.error("no output dir: pass --output-dir or set optimize.output_dir in the YAML")

    result = run_optimization(config, output_dir, device=a.device)
    lh = result["loss_history"]
    rh = result["ratio_history"]
    print(f"Done: {len(lh)} steps -> {result['output_dir']}")
    if lh:
        print(f"  loss[0]={lh[0]:.4e}  loss[-1]={lh[-1]:.4e}")
    if rh:
        print(f"  ratio[0]={rh[0]:.4f}  ratio[-1]={rh[-1]:.4f}  (1.0 = perfect cloak)")


if __name__ == "__main__":
    main()
