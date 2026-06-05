"""Generate 50x50 binary unit cells from a trained 2D diffusion checkpoint and
evaluate the realised stiffness via FEM round-trip.

Targets are read from a CSV with columns ``C11,C22,C12,C66,vol`` (header row).
Use ``-1`` (the CFG sentinel) to leave a channel unconstrained.

Usage:

    python -m microstructure_generation_2d.generate \\
        --ckpt output/diffusion_ca_squared/v1/last.ckpt \\
        --targets targets.csv \\
        --output-dir output/diffusion_ca_squared/v1/eval \\
        --num-per-target 4 --steps 50 --tensor-w 2.0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm

from dataset.cellular_chiral.diffusion_dataset import (
    CELL_SIZE, PAD_TO, QUADRANT_SIZE, CFG_SENTINEL, unfold_mirror,
)
from .network.model_trainer import DiffusionModel


def _crop_64_to_50(arr: np.ndarray) -> np.ndarray:
    """Inverse of dataset _pad_to_64; arr is (..., 64, 64). Legacy mode only."""
    off = (PAD_TO - CELL_SIZE) // 2
    return arr[..., off : off + CELL_SIZE, off : off + CELL_SIZE]


def _decode_to_50(arr: np.ndarray, compressed: bool) -> np.ndarray:
    """Decode the model's raw output (after sign threshold) into a (..., 50, 50)
    binary cell. Legacy v1-v3 models output 64x64 padded; v4+ outputs the 25x25
    NW mirror-quadrant that we tile via ``unfold_mirror``.
    """
    if compressed:
        return unfold_mirror(arr[..., :QUADRANT_SIZE, :QUADRANT_SIZE]).astype(np.uint8)
    return _crop_64_to_50(arr)


def _scale_targets(row: dict, scaler_dir: Path) -> np.ndarray:
    """Convert a {C11,C22,C12,C66,vol} dict into the scaled 5-dim tensor used by
    the model. Values equal to CFG_SENTINEL (-1) are passed through unchanged.
    """
    scalers = {
        "C11": joblib.load(scaler_dir / "scaler_C11"),
        "C22": joblib.load(scaler_dir / "scaler_C22"),
        "C12": joblib.load(scaler_dir / "scaler_C12"),
        "C66": joblib.load(scaler_dir / "scaler_C66"),
    }
    names = ("C11", "C22", "C12", "C66")
    tf = np.empty(len(names) + 1, dtype=np.float32)
    for i, name in enumerate(names):
        v = float(row[name])
        if v == CFG_SENTINEL:
            tf[i] = CFG_SENTINEL
        else:
            tf[i] = float(scalers[name].transform([[v]])[0, 0])
    tf[-1] = float(row["vol"])  # vol left raw; -1 means drop
    return tf


def _fem_roundtrip(cells_50: np.ndarray) -> list[dict]:
    """Run periodic homogenization on each (50,50) uint8 cell. Returns one
    dict per cell with realised (C11, C22, C12, C66, vf), or a non-None ``error``.

    Imported lazily because jax-fem prints a banner on import and is slow.
    """
    from dataset.cellular_chiral.bulk_stiffness import (
        _compute_stiffness, _ortho_params_from_C,
    )
    results = []
    for cell in tqdm(cells_50, desc="FEM eval", unit="cell"):
        try:
            C, rho, vf = _compute_stiffness(cell.astype(np.int8))
            C11, C22, C12, C66 = _ortho_params_from_C(C)
            results.append({"C11": C11, "C22": C22, "C12": C12, "C66": C66,
                            "vf": vf, "error": None})
        except Exception as exc:  # noqa: BLE001
            results.append({"C11": None, "C22": None, "C12": None, "C66": None,
                            "vf": None, "error": f"{type(exc).__name__}: {exc}"})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Lightning .ckpt path")
    parser.add_argument("--targets", required=True,
                        help="CSV with columns C11,C22,C12,C66,vol")
    parser.add_argument("--scaler-dir", default="microstructure_generation_2d")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-per-target", type=int, default=4)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tensor-w", type=float, default=2.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--ema", action="store_true", default=True)
    parser.add_argument("--no-fem", action="store_true",
                        help="Skip the FEM round-trip (just save cells)")
    parser.add_argument("--compressed", action=argparse.BooleanOptionalAction, default=None,
                        help="Override the checkpoint's compressed flag. v4+ "
                             "compressed=True decodes 25x25 -> 50x50 via mirror tile; "
                             "v1-v3 compressed=False decodes 64x64 -> 50x50 via crop. "
                             "Defaults to the checkpoint's saved hparam.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_dir = Path(args.scaler_dir)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    torch.manual_seed(args.seed)

    module = DiffusionModel.load_from_checkpoint(args.ckpt, map_location=device).to(device)
    module.eval()
    generator = module.ema_model if args.ema else module.model
    generator.eval()

    compressed = (
        args.compressed
        if args.compressed is not None
        else bool(module.hparams.get("compressed", False))
    )

    with open(args.targets) as f:
        rows = list(csv.DictReader(f))

    all_cells = []
    summary = []
    for tgt_i, row in enumerate(tqdm(rows, desc="targets")):
        tensor_c = _scale_targets(row, scaler_dir)
        img = generator.sample_with_tensor(
            tensor_c=tensor_c,
            batch_size=args.num_per_target,
            steps=args.steps,
            tensor_w=args.tensor_w,
            verbose=False,
        )
        arr = img.detach().cpu().numpy().squeeze(1)  # (B, H, H) with H in {64, 25}
        bin_raw = (arr > 0).astype(np.uint8)
        bin50 = _decode_to_50(bin_raw, compressed=compressed)  # (B, 50, 50)
        all_cells.append(bin50)

        target_record = {"target_idx": tgt_i, **{k: float(row[k]) for k in row}}
        summary.append(target_record)

    cells_arr = np.concatenate(all_cells, axis=0)  # (T*num_per_target, 50, 50)
    np.save(out_dir / "cells.npy", cells_arr)

    fem = None
    if not args.no_fem:
        fem = _fem_roundtrip(cells_arr)
        with open(out_dir / "fem_results.json", "w") as f:
            json.dump(fem, f, indent=2)

    # Per-target errors
    per_target = []
    n = args.num_per_target
    for tgt_i, row in enumerate(rows):
        block = fem[tgt_i * n : (tgt_i + 1) * n] if fem is not None else []
        per_target.append({"target": row, "fem": block})
    with open(out_dir / "report.json", "w") as f:
        json.dump({"targets": per_target,
                   "summary": _summary_stats(rows, fem, n) if fem else None}, f, indent=2)
    print(f"Wrote {out_dir}/cells.npy ({cells_arr.shape}) and report.json")


def _summary_stats(rows, fem, num_per_target):
    """Per-target relative error vs requested (C11, C22, C12, C66, vol)."""
    out = []
    for tgt_i, row in enumerate(rows):
        block = fem[tgt_i * num_per_target : (tgt_i + 1) * num_per_target]
        for k in ("C11", "C22", "C12", "C66"):
            tv = float(row[k])
            if tv == CFG_SENTINEL:
                continue
            vals = [b[k] for b in block if b["error"] is None]
            if not vals:
                continue
            mean = float(np.mean(vals))
            rel_err = abs(mean - tv) / max(abs(tv), 1.0)
            out.append({"target_idx": tgt_i, "channel": k, "target": tv,
                        "mean_realised": mean, "rel_err": rel_err})
        vol_tv = float(row["vol"])
        if vol_tv != CFG_SENTINEL:
            vols = [b["vf"] for b in block if b["error"] is None]
            if vols:
                mean = float(np.mean(vols))
                out.append({"target_idx": tgt_i, "channel": "vol",
                            "target": vol_tv, "mean_realised": mean,
                            "rel_err": abs(mean - vol_tv) / max(abs(vol_tv), 1e-6)})
    return out


if __name__ == "__main__":
    main()
