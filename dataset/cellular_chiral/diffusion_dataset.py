"""Torch Dataset over the ca_bulk_squared stiffness.h5 subset.

Two modes:

* ``compressed=False`` (legacy): 64x64 padded occupancy fields ({-1, +1}).
* ``compressed=True`` (v4+): the NW 25x25 mirror-quadrant of the cell. The
  full 50x50 dataset cell is exactly recovered by mirror tiling
  (``unfold_mirror``), the inverse of ``generator._assemble_squared``.

Both modes return a 5-dim conditioning vector (scaled C11/C22/C12/C66, raw vol)
with the 10/10/10% classifier-free dropout schedule from the 3D pipeline.

A deterministic train/val split (seeded shuffle) is materialised on first use
and persisted to JSON so independent processes / future runs see the same one.
"""
from __future__ import annotations

import json
from pathlib import Path
from random import random
from typing import Optional, Union

import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


CELL_SIZE = 50
QUADRANT_SIZE = CELL_SIZE // 2  # = 25
PAD_TO = 64
TENSOR_DIM = 5  # C11, C22, C12, C66, vol
CFG_SENTINEL = -1.0  # matches the 3D pipeline; see plan §2


def _pad_to_64(cell_pm1: np.ndarray) -> np.ndarray:
    """Pad a (50, 50) {-1, +1} float array to (64, 64) with -1 ("void" sign)."""
    out = -np.ones((PAD_TO, PAD_TO), dtype=np.float32)
    off = (PAD_TO - CELL_SIZE) // 2  # = 7
    out[off : off + CELL_SIZE, off : off + CELL_SIZE] = cell_pm1
    return out


def unfold_mirror(quadrant: Union[np.ndarray, torch.Tensor]):
    """Mirror-tile a (..., 25, 25) NW quadrant to (..., 50, 50).

    Layout matches ``dataset/cellular_chiral/generator.py:_assemble_squared``:

        TL = Q              TR = fliplr(Q)
        BL = flipud(Q)      BR = flipud(fliplr(Q))

    Works on numpy arrays or torch tensors of any leading batch shape.
    """
    if isinstance(quadrant, torch.Tensor):
        tl = quadrant
        tr = torch.flip(quadrant, dims=(-1,))
        bl = torch.flip(quadrant, dims=(-2,))
        br = torch.flip(quadrant, dims=(-2, -1))
        top = torch.cat([tl, tr], dim=-1)
        bot = torch.cat([bl, br], dim=-1)
        return torch.cat([top, bot], dim=-2)
    tl = quadrant
    tr = np.flip(quadrant, axis=-1)
    bl = np.flip(quadrant, axis=-2)
    br = np.flip(quadrant, axis=(-2, -1))
    top = np.concatenate([tl, tr], axis=-1)
    bot = np.concatenate([bl, br], axis=-1)
    return np.concatenate([top, bot], axis=-2)


def make_split(n: int, val_frac: float, seed: int, out_path: Path) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    n_val = int(round(n * val_frac))
    split = {"train": perm[n_val:], "val": perm[:n_val], "seed": seed,
             "n": n, "val_frac": val_frac}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(split, f)
    return split


def load_or_make_split(n: int, val_frac: float, seed: int, split_path: Path) -> dict:
    if split_path.exists():
        with open(split_path) as f:
            sp = json.load(f)
        if sp["n"] == n and sp["seed"] == seed and abs(sp["val_frac"] - val_frac) < 1e-9:
            return sp
        # n/seed/val_frac mismatch: regenerate
    return make_split(n, val_frac, seed, split_path)


def inverse_density_weights(
    values: np.ndarray, bins: int = 50, value_range: Optional[tuple] = None,
    clip: float = 0.0,
) -> np.ndarray:
    """Per-sample weights ∝ 1/histogram-density(values), normalized to mean 1.

    Flattens the (training) distribution along ``values`` (e.g. volume fraction):
    samples in sparse bins are up-weighted, samples in over-represented bins
    down-weighted, so each bin contributes equal total weight. ``clip`` (>0) caps
    the maximum per-sample weight at ``clip``× the mean to bound gradient variance
    from very sparse bins; the mean is renormalized to 1 afterwards.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.histogram_bin_edges(values, bins=bins, range=value_range)
    counts, _ = np.histogram(values, bins=edges)
    b = np.clip(np.digitize(values, edges) - 1, 0, len(counts) - 1)
    w = 1.0 / np.maximum(counts[b].astype(np.float64), 1.0)
    w *= values.size / w.sum()  # mean -> 1
    if clip and clip > 0:
        w = np.minimum(w, clip)
        w *= values.size / w.sum()  # renormalize mean -> 1
    return w.astype(np.float32)


class CABulkDiffusionDataset(Dataset):
    """Random-access dataset over `stiffness.h5["cells"]` (gzip-chunked HDF5).

    HDF5 random access is opened lazily per-worker (workers are forked after
    the parent constructs the Dataset, so we can't hold an open file handle
    across the fork boundary).
    """

    def __init__(
        self,
        h5_path: str,
        scaler_dir: str,
        indices: list[int],
        cfg_dropout: bool = True,
        seed: Optional[int] = None,
        compressed: bool = False,
        padded_size: int = QUADRANT_SIZE,
        sample_weights: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.scaler_dir = Path(scaler_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cfg_dropout = cfg_dropout
        self.compressed = compressed
        self.padded_size = padded_size
        # Full-length (n_h5,) per-row loss weights, indexed by h5 row; None -> 1.0.
        self.sample_weights = sample_weights
        self._h5: Optional[h5py.File] = None
        self._cells = None  # type: ignore
        self._C11 = None
        self._C22 = None
        self._C12 = None
        self._C66 = None
        self._vol = None

        self.scaler_C11 = joblib.load(self.scaler_dir / "scaler_C11")
        self.scaler_C22 = joblib.load(self.scaler_dir / "scaler_C22")
        self.scaler_C12 = joblib.load(self.scaler_dir / "scaler_C12")
        self.scaler_C66 = joblib.load(self.scaler_dir / "scaler_C66")

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._cells = self._h5["cells"]
            self._C11 = self._h5["C11"]
            self._C22 = self._h5["C22"]
            self._C12 = self._h5["C12"]
            self._C66 = self._h5["C66"]
            self._vol = self._h5["vol"]

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, idx: int) -> dict:
        self._ensure_open()
        h_idx = int(self.indices[idx])

        cell_u8 = np.asarray(self._cells[h_idx], dtype=np.uint8)  # (50, 50)
        cell_pm1 = (cell_u8.astype(np.float32) * 2.0 - 1.0)
        if self.compressed:
            q = cell_pm1[:QUADRANT_SIZE, :QUADRANT_SIZE]  # (25, 25)
            if self.padded_size > QUADRANT_SIZE:
                pad = self.padded_size - QUADRANT_SIZE
                q = np.pad(q, ((0, pad), (0, pad)), constant_values=-1.0)
            occ = q[None, :, :]  # (1, padded_size, padded_size)
        else:
            occ = _pad_to_64(cell_pm1)[None, :, :]  # (1, 64, 64)

        c11 = float(self._C11[h_idx])
        c22 = float(self._C22[h_idx])
        c12 = float(self._C12[h_idx])
        c66 = float(self._C66[h_idx])
        vol = float(self._vol[h_idx])

        c11s = float(self.scaler_C11.transform([[c11]])[0, 0])
        c22s = float(self.scaler_C22.transform([[c22]])[0, 0])
        c12s = float(self.scaler_C12.transform([[c12]])[0, 0])
        c66s = float(self.scaler_C66.transform([[c66]])[0, 0])

        tensor_feature = np.array([c11s, c22s, c12s, c66s, vol], dtype=np.float32)

        if self.cfg_dropout:
            r = random()
            if r < 0.1:
                tensor_feature[:] = CFG_SENTINEL  # drop everything (uncond)
            elif r < 0.2:
                tensor_feature[0:4] = CFG_SENTINEL  # drop stiffness, keep vol
            elif r < 0.3:
                tensor_feature[4] = CFG_SENTINEL    # drop vol, keep stiffness

        w = 1.0 if self.sample_weights is None else float(self.sample_weights[h_idx])

        return {
            "occupancy": torch.from_numpy(occ),
            "tensor_feature": torch.from_numpy(tensor_feature),
            "h5_index": h_idx,
            "weight": torch.tensor(w, dtype=torch.float32),
        }
