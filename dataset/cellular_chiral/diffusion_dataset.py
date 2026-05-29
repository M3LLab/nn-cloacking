"""Torch Dataset over the ca_bulk_squared stiffness.h5 subset.

Provides 64x64 padded occupancy fields ({-1, +1}) plus a 4-dim conditioning
vector (scaled C11, scaled C12, scaled C66, raw vol). Classifier-free guidance
dropout matches the 3D pipeline's 10/10/10% schedule (see
`microstructure_generation_3d/network/data_loader.py`).

A deterministic train/val split (seeded shuffle) is materialised on first use
and persisted to JSON so independent processes / future runs see the same one.
"""
from __future__ import annotations

import json
from pathlib import Path
from random import random
from typing import Optional

import h5py
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset


CELL_SIZE = 50
PAD_TO = 64
TENSOR_DIM = 4  # C11, C12, C66, vol
CFG_SENTINEL = -1.0  # matches the 3D pipeline; see plan §2


def _pad_to_64(cell_pm1: np.ndarray) -> np.ndarray:
    """Pad a (50, 50) {-1, +1} float array to (64, 64) with -1 ("void" sign)."""
    out = -np.ones((PAD_TO, PAD_TO), dtype=np.float32)
    off = (PAD_TO - CELL_SIZE) // 2  # = 7
    out[off : off + CELL_SIZE, off : off + CELL_SIZE] = cell_pm1
    return out


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
    ):
        super().__init__()
        self.h5_path = str(h5_path)
        self.scaler_dir = Path(scaler_dir)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.cfg_dropout = cfg_dropout
        self._h5: Optional[h5py.File] = None
        self._cells = None  # type: ignore
        self._C11 = None
        self._C12 = None
        self._C66 = None
        self._vol = None

        self.scaler_C11 = joblib.load(self.scaler_dir / "scaler_C11")
        self.scaler_C12 = joblib.load(self.scaler_dir / "scaler_C12")
        self.scaler_C66 = joblib.load(self.scaler_dir / "scaler_C66")

    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)
            self._cells = self._h5["cells"]
            self._C11 = self._h5["C11"]
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
        occ = _pad_to_64(cell_pm1)[None, :, :]  # (1, 64, 64)

        c11 = float(self._C11[h_idx])
        c12 = float(self._C12[h_idx])
        c66 = float(self._C66[h_idx])
        vol = float(self._vol[h_idx])

        c11s = float(self.scaler_C11.transform([[c11]])[0, 0])
        c12s = float(self.scaler_C12.transform([[c12]])[0, 0])
        c66s = float(self.scaler_C66.transform([[c66]])[0, 0])

        tensor_feature = np.array([c11s, c12s, c66s, vol], dtype=np.float32)

        if self.cfg_dropout:
            r = random()
            if r < 0.1:
                tensor_feature[:] = CFG_SENTINEL  # drop everything (uncond)
            elif r < 0.2:
                tensor_feature[0:3] = CFG_SENTINEL  # drop stiffness, keep vol
            elif r < 0.3:
                tensor_feature[3] = CFG_SENTINEL    # drop vol, keep stiffness

        return {
            "occupancy": torch.from_numpy(occ),
            "tensor_feature": torch.from_numpy(tensor_feature),
            "h5_index": h_idx,
        }
