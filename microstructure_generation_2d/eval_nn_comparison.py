"""Compare diffusion generation vs nearest-neighbour retrieval on the full
validation set.

For every val sample the script:
  1. Generates one diffusion sample conditioned on (C11, C12, C66, vol).
  2. Runs FEM to get its realised stiffness.
  3. Finds the nearest neighbour in the training set (L2 in z-scored feature
     space: scaled C11/C12/C66 + z-scored vol).
  4. Saves a side-by-side PNG: [Val target | Generated | Nearest neighbour].

Outputs
-------
  <output_dir>/comparison.csv   columns:
      comparison_id, val_idx, nn_idx,
      C11, C11_generated, C11_dataset,
      C12, C12_generated, C12_dataset,
      C66, C66_generated, C66_dataset,
      vol, vol_generated, vol_dataset

  <output_dir>/images/          one PNG per comparison

The script is resumable: rows already present in comparison.csv are skipped.

Usage
-----
    python -m microstructure_generation_2d.eval_nn_comparison \\
        --ckpt output/diffusion_ca_squared/v2/last.ckpt \\
        --output-dir output/diffusion_ca_squared/v2/eval_nn

    # quick smoke-test on 20 samples
    python -m microstructure_generation_2d.eval_nn_comparison \\
        --ckpt output/diffusion_ca_squared/v2/last.ckpt \\
        --output-dir output/diffusion_ca_squared/v2/eval_nn_smoke \\
        --max-samples 20
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset.cellular_chiral.diffusion_dataset import (
    CELL_SIZE, PAD_TO, load_or_make_split,
)
from .network.model_trainer import DiffusionModel

CSV_FIELDS = [
    "comparison_id", "val_idx", "nn_idx",
    "C11", "C11_generated", "C11_dataset",
    "C12", "C12_generated", "C12_dataset",
    "C66", "C66_generated", "C66_dataset",
    "vol", "vol_generated", "vol_dataset",
]

VOL_MEAN = 0.478
VOL_STD  = 0.104


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop(arr: np.ndarray) -> np.ndarray:
    off = (PAD_TO - CELL_SIZE) // 2
    return arr[..., off : off + CELL_SIZE, off : off + CELL_SIZE]


def _run_fem(cell_uint8: np.ndarray) -> tuple[float, float, float, float]:
    from dataset.cellular_chiral.bulk_stiffness import (
        _compute_stiffness, _d4_params_from_C,
    )
    C, _, vf = _compute_stiffness(cell_uint8.astype(np.int8))
    C11, C12, C66 = _d4_params_from_C(C)
    return float(C11), float(C12), float(C66), float(vf)


def _scale_features(
    C11: np.ndarray,
    C12: np.ndarray,
    C66: np.ndarray,
    vol: np.ndarray,
    sc11, sc12, sc66,
) -> np.ndarray:
    """Return (N, 4) array with all features z-scored."""
    s11 = sc11.transform(C11.reshape(-1, 1)).ravel()
    s12 = sc12.transform(C12.reshape(-1, 1)).ravel()
    s66 = sc66.transform(C66.reshape(-1, 1)).ravel()
    sv  = (vol - VOL_MEAN) / VOL_STD
    return np.column_stack([s11, s12, s66, sv]).astype(np.float32)


def _load_train_features(
    h5_path: str,
    train_indices: list[int],
    sc11, sc12, sc66,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and scale training-set stiffness features.

    Returns (features (N,4), C11, C12, C66, vol) for all train indices,
    in the same order as train_indices.
    """
    idx = np.asarray(train_indices, dtype=np.int64)
    sort_ord = np.argsort(idx)
    inv_ord  = np.argsort(sort_ord)
    sorted_idx = idx[sort_ord]

    with h5py.File(h5_path, "r") as f:
        C11_all = np.asarray(f["C11"])
        C12_all = np.asarray(f["C12"])
        C66_all = np.asarray(f["C66"])
        vol_all = np.asarray(f["vol"])

    C11 = C11_all[sorted_idx][inv_ord]
    C12 = C12_all[sorted_idx][inv_ord]
    C66 = C66_all[sorted_idx][inv_ord]
    vol = vol_all[sorted_idx][inv_ord]

    feats = _scale_features(C11, C12, C66, vol, sc11, sc12, sc66)
    return feats, C11, C12, C66, vol


def _find_nn(query_feat: np.ndarray, train_feats: np.ndarray) -> int:
    """Return index into train_feats of the closest row."""
    dists = np.sum((train_feats - query_feat) ** 2, axis=1)
    return int(np.argmin(dists))


def _load_cell(h5: h5py.File, h_idx: int) -> np.ndarray:
    return np.asarray(h5["cells"][h_idx], dtype=np.uint8)


def _rel_err(pred: float, target: float) -> float:
    denom = max(abs(target), 1.0)
    return abs(pred - target) / denom


def _save_image(
    val_cell: np.ndarray,
    gen_cell: np.ndarray,
    nn_cell: np.ndarray,
    row: dict,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.8))
    for ax, cell, title in zip(
        axes,
        [val_cell, gen_cell, nn_cell],
        ["Val target", "Generated", "Nearest neighbour"],
    ):
        ax.imshow(cell, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    def fmt_errs(prefix: str) -> str:
        return (
            f"C11={_rel_err(row[f'C11_{prefix}'], row['C11']):.1%}  "
            f"C12={_rel_err(row[f'C12_{prefix}'], row['C12']):.1%}  "
            f"C66={_rel_err(row[f'C66_{prefix}'], row['C66']):.1%}  "
            f"vol={abs(row[f'vol_{prefix}'] - row['vol']):.3f}"
        )

    fig.suptitle(
        f"id={row['comparison_id']}  val_idx={row['val_idx']}  "
        f"C11={row['C11']:.2e}  C12={row['C12']:.2e}  "
        f"C66={row['C66']:.2e}  vol={row['vol']:.3f}\n"
        f"Gen  rel-err: {fmt_errs('generated')}\n"
        f"NN   rel-err: {fmt_errs('dataset')}",
        fontsize=7.5,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _load_done_ids(csv_path: Path) -> set[int]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                done.add(int(row["comparison_id"]))
            except (KeyError, ValueError):
                pass
    return done


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--h5", default="output/ca_bulk_squared/stiffness.h5")
    parser.add_argument("--scaler-dir", default="microstructure_generation_2d")
    parser.add_argument("--split-path", default="microstructure_generation_2d/split.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tensor-w", type=float, default=2.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of val samples (for smoke testing)")
    args = parser.parse_args()

    out_dir   = Path(args.output_dir)
    img_dir   = out_dir / "images"
    csv_path  = out_dir / "comparison.csv"
    scaler_dir = Path(args.scaler_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    # ---- scalers ----
    sc11 = joblib.load(scaler_dir / "scaler_C11")
    sc12 = joblib.load(scaler_dir / "scaler_C12")
    sc66 = joblib.load(scaler_dir / "scaler_C66")

    # ---- split ----
    with h5py.File(args.h5, "r") as f:
        n_total = f["cells"].shape[0]
    split = load_or_make_split(
        n=n_total, val_frac=0.02, seed=777, split_path=Path(args.split_path),
    )
    val_indices   = split["val"]
    train_indices = split["train"]
    if args.max_samples is not None:
        val_indices = val_indices[: args.max_samples]

    print(f"Val samples : {len(val_indices)}")
    print(f"Train samples: {len(train_indices)}")

    # ---- build NN feature matrix ----
    print("Loading train features for NN search...")
    train_feats, tr_C11, tr_C12, tr_C66, tr_vol = _load_train_features(
        args.h5, train_indices, sc11, sc12, sc66,
    )
    train_indices_arr = np.asarray(train_indices, dtype=np.int64)

    # ---- diffusion model ----
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    torch.manual_seed(args.seed)
    module = DiffusionModel.load_from_checkpoint(args.ckpt, map_location=device).to(device)
    module.eval()
    generator = module.ema_model if args.ema else module.model
    generator.eval()

    # ---- resume: skip already-done ----
    done_ids = _load_done_ids(csv_path)
    print(f"Already done : {len(done_ids)}")

    csv_file = open(csv_path, "a", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if not done_ids:
        writer.writeheader()

    # ---- main loop ----
    with h5py.File(args.h5, "r") as h5:
        for cmp_id, val_h5_idx in enumerate(tqdm(val_indices, desc="eval")):
            if cmp_id in done_ids:
                continue

            # -- load val sample --
            val_cell = _load_cell(h5, val_h5_idx)
            c11  = float(h5["C11"][val_h5_idx])
            c12  = float(h5["C12"][val_h5_idx])
            c66  = float(h5["C66"][val_h5_idx])
            vol  = float(h5["vol"][val_h5_idx])

            # -- scaled conditioning vector (same as training) --
            tf = np.array([
                float(sc11.transform([[c11]])[0, 0]),
                float(sc12.transform([[c12]])[0, 0]),
                float(sc66.transform([[c66]])[0, 0]),
                vol,
            ], dtype=np.float32)

            # -- generate --
            with torch.no_grad():
                img = generator.sample_with_tensor(
                    tensor_c=tf,
                    batch_size=1,
                    steps=args.steps,
                    tensor_w=args.tensor_w,
                    verbose=False,
                )
            arr64    = img.cpu().numpy().squeeze()
            gen_cell = _crop((arr64 > 0).astype(np.uint8))

            # -- FEM on generated --
            c11_gen, c12_gen, c66_gen, vol_gen = _run_fem(gen_cell)

            # -- nearest neighbour --
            query = np.array([
                float(sc11.transform([[c11]])[0, 0]),
                float(sc12.transform([[c12]])[0, 0]),
                float(sc66.transform([[c66]])[0, 0]),
                (vol - VOL_MEAN) / VOL_STD,
            ], dtype=np.float32)
            nn_pos     = _find_nn(query, train_feats)
            nn_h5_idx  = int(train_indices_arr[nn_pos])
            nn_cell    = _load_cell(h5, nn_h5_idx)
            c11_nn = float(tr_C11[nn_pos])
            c12_nn = float(tr_C12[nn_pos])
            c66_nn = float(tr_C66[nn_pos])
            vol_nn = float(tr_vol[nn_pos])

            # -- write CSV row --
            row = {
                "comparison_id": cmp_id,
                "val_idx":        val_h5_idx,
                "nn_idx":         nn_h5_idx,
                "C11":            c11,
                "C11_generated":  c11_gen,
                "C11_dataset":    c11_nn,
                "C12":            c12,
                "C12_generated":  c12_gen,
                "C12_dataset":    c12_nn,
                "C66":            c66,
                "C66_generated":  c66_gen,
                "C66_dataset":    c66_nn,
                "vol":            vol,
                "vol_generated":  vol_gen,
                "vol_dataset":    vol_nn,
            }
            writer.writerow(row)
            csv_file.flush()

            # -- save comparison image --
            _save_image(
                val_cell, gen_cell, nn_cell, row,
                img_dir / f"comparison_{cmp_id:06d}.png",
            )

    csv_file.close()
    print(f"\nDone. Results in {csv_path}")
    print(f"Images in       {img_dir}/")


if __name__ == "__main__":
    main()
