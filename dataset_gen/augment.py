"""Augment an existing surrogate dataset with best-sample frequency sweeps.

For each unique source frequency already present in the HDF5 file, find the
sample with the lowest loss and re-evaluate its (C, rho) at a grid of target
frequencies spanning the dataset's frequency range (same parameters, only
omega changes).  New entries are appended to the same HDF5 file.

Augmented entries use sample_type of the form ``augment_from_f{source:.3f}`` so
subsequent runs can skip them when picking best samples.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from pathlib import Path

import h5py
import jax.numpy as jnp
import numpy as np
from tqdm.auto import tqdm

from rayleigh_cloak.config import SimulationConfig

from dataset_gen.generate import (
    append_samples,
    build_fixed_context,
    build_freq_context,
    evaluate_loss,
)


@contextlib.contextmanager
def _silence_stdout():
    """Swallow stdout — used to suppress noisy jax-fem / solver prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _decode_sample_types(raw) -> np.ndarray:
    return np.array([
        s.decode() if isinstance(s, (bytes, np.bytes_)) else s for s in raw
    ])


def _existing_augment_pairs(h5_path: Path, round_decimals: int = 4) -> set[tuple[str, float]]:
    """Return {(sample_type, f_star_rounded)} for all augment_* rows already in the file."""
    with h5py.File(h5_path, "r") as f:
        types = _decode_sample_types(f["sample_type"][:])
        f_stars = np.round(f["f_star"][:], round_decimals)
    return {
        (t, float(fs))
        for t, fs in zip(types, f_stars)
        if t.startswith("augment_")
    }


def load_best_params_per_freq(
    h5_path: Path, top_k: int = 1, round_decimals: int = 4,
) -> list[dict]:
    """Return the top-k lowest-loss samples for each unique source frequency.

    f_star values are rounded to ``round_decimals`` before grouping to avoid
    treating float64 near-duplicates (e.g. 1.95 vs 1.9500000000000002) as
    distinct source frequencies.

    Samples whose ``sample_type`` begins with ``augment_`` are excluded so the
    function is idempotent when the dataset has already been augmented.
    """
    with h5py.File(h5_path, "r") as f:
        f_stars = np.round(f["f_star"][:], round_decimals)
        losses = f["loss"][:]
        sample_types = _decode_sample_types(f["sample_type"][:])
        cell_C_ds = f["cell_C_flat"]
        cell_rho_ds = f["cell_rho"]

        is_augment = np.array([st.startswith("augment_") for st in sample_types])
        valid = ~is_augment

        best = []
        for fs in np.unique(f_stars[valid]):
            mask = (f_stars == fs) & valid
            idxs = np.where(mask)[0]
            order = idxs[np.argsort(losses[idxs])[:top_k]]
            for idx in order:
                best.append({
                    "f_star_source": float(fs),
                    "cell_C_flat": np.asarray(cell_C_ds[idx]),
                    "cell_rho": np.asarray(cell_rho_ds[idx]),
                    "loss_source": float(losses[idx]),
                })
    return best


@dataclass
class AugmentConfig:
    """Controls for dataset augmentation."""

    n_target_freqs: int = 32
    f_min: float | None = None
    f_max: float | None = None
    top_k: int = 1


def run_dataset_augmentation(
    base_config: SimulationConfig,
    h5_path: Path,
    aug_config: AugmentConfig | None = None,
):
    """Augment an HDF5 dataset with frequency sweeps on per-freq best samples."""
    if aug_config is None:
        aug_config = AugmentConfig()

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"Dataset not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        existing_f_stars = f["f_star"][:]
    f_min = aug_config.f_min if aug_config.f_min is not None else float(existing_f_stars.min())
    f_max = aug_config.f_max if aug_config.f_max is not None else float(existing_f_stars.max())

    best_samples = load_best_params_per_freq(h5_path, top_k=aug_config.top_k)
    if not best_samples:
        raise RuntimeError(f"No non-augmented samples found in {h5_path}")

    target_f_stars = [
        round(float(f), 4)
        for f in np.linspace(f_min, f_max, aug_config.n_target_freqs)
    ]

    already_done = _existing_augment_pairs(h5_path)

    planned_pairs = {
        (f"augment_from_f{s['f_star_source']:.3f}", round(f_star, 4))
        for f_star in target_f_stars
        for s in best_samples
        if abs(s["f_star_source"] - f_star) >= 1e-6
    }
    total = len(planned_pairs)
    resumed = len(planned_pairs & already_done)

    ctx = build_fixed_context(base_config)

    tqdm.write(f"\n=== Augmentation plan ===")
    tqdm.write(f"  Source samples:  {len(best_samples)} (top-{aug_config.top_k} per freq)")
    tqdm.write(f"  Target freqs:    {len(target_f_stars)} "
               f"({target_f_stars[0]:.3f} → {target_f_stars[-1]:.3f})")
    tqdm.write(f"  Total evals:     {total}  (already done: {resumed})")
    tqdm.write(f"  Output:          {h5_path}\n")

    with tqdm(total=total, initial=resumed, desc="augment", unit="eval",
              dynamic_ncols=True) as pbar:
        for f_star in target_f_stars:
            # Skip the whole target freq if every src pair is already in the file.
            pending = [
                s for s in best_samples
                if abs(s["f_star_source"] - f_star) >= 1e-6
                and (f"augment_from_f{s['f_star_source']:.3f}", round(f_star, 4))
                    not in already_done
            ]
            if not pending:
                continue

            with _silence_stdout():
                fctx = build_freq_context(ctx, f_star)

            batch = []
            for s in pending:
                src = s["f_star_source"]
                params = (jnp.array(s["cell_C_flat"]), jnp.array(s["cell_rho"]))
                with _silence_stdout():
                    loss = evaluate_loss(fctx, params)
                batch.append({
                    "cell_C_flat": s["cell_C_flat"],
                    "cell_rho": s["cell_rho"],
                    "f_star": f_star,
                    "loss": loss,
                    "sample_type": f"augment_from_f{src:.3f}",
                })
                pbar.set_postfix_str(
                    f"target={f_star:.3f} src={src:.2f} loss={loss:.2e}"
                )
                pbar.update(1)

            with _silence_stdout():
                append_samples(h5_path, batch)

    with h5py.File(h5_path, "r") as f:
        n_total = f["loss"].shape[0]
    tqdm.write(f"\n=== Augmentation complete ===")
    tqdm.write(f"  New samples: {total}  |  Total in file: {n_total}")
    tqdm.write(f"  Output: {h5_path}")
