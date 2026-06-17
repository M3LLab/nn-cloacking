"""Torch Dataset / DataLoader for the surrogate HDF5 files.

HDF5 layout (see dataset_gen/):
    cell_C_flat  (N, n_cells, n_C_params)  stiffness params per cell
    cell_rho     (N, n_cells)              density per cell
    f_star       (N,)                      normalised frequency
    loss         (N,)                      transmission loss  (target)
    sample_type  (N,)                      "init" / "random*" / "smooth*" / "opt*"
    cloak_mask   (n_cells,)                shared boolean mask of cloak cells

`n_cells` is a flattened (n_x, n_y) grid with layout
    idx = ix * n_y + iy                     (see rayleigh_cloak/cells.py)
so reshape(-1, n_x, n_y, ...) recovers the spatial axes.
"""
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

LossTransform = Literal["none", "log_clip"]


def apply_loss_transform(
    loss: np.ndarray,
    transform: LossTransform = "none",
    floor: float = 1e-8,
) -> np.ndarray:
    """Forward loss transform. Use on the hi-fi dataset (loss spans ~18 decades).

    ``log_clip``: ``log(max(loss, floor))``. Floor 1e-8 is well below the
    ~1% quantile (~1e-12) but clips the ~2% bottom tail that reaches 1e-18,
    which otherwise produces skew≈-1.2; with clipping skew≈-0.2, kurt≈0.2.
    """
    if transform == "none":
        return loss
    if transform == "log_clip":
        return np.log(np.clip(loss, floor, None))
    raise ValueError(f"Unknown loss_transform: {transform!r}")


def invert_loss_transform(
    y: np.ndarray | Tensor,
    transform: LossTransform = "none",
) -> np.ndarray | Tensor:
    """Inverse of apply_loss_transform, for reporting predictions in raw units."""
    if transform == "none":
        return y
    if transform == "log_clip":
        return torch.exp(y) if isinstance(y, Tensor) else np.exp(y)
    raise ValueError(f"Unknown loss_transform: {transform!r}")

# jaxtyping axis names:
#   X, Y -> n_x, n_y  (cell grid)
#   P    -> n_C_params
#   B    -> batch


class SurrogateSample(BaseModel):
    """A single (input, target) pair on the 2D cell grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    C:      Float[Tensor, "X Y P"]
    rho:    Float[Tensor, "X Y"]
    f_star: Float[Tensor, ""]
    loss:   Float[Tensor, ""]


class SurrogateBatch(BaseModel):
    """Stacked batch produced by the dataloader's collate_fn."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    C:      Float[Tensor, "B X Y P"]
    rho:    Float[Tensor, "B X Y"]
    f_star: Float[Tensor, "B"]
    loss:   Float[Tensor, "B"]

    def __len__(self) -> int:
        return self.loss.shape[0]

    def to(self, device: str | torch.device) -> "SurrogateBatch":
        return SurrogateBatch(
            C=self.C.to(device),
            rho=self.rho.to(device),
            f_star=self.f_star.to(device),
            loss=self.loss.to(device),
        )


def collate_surrogate(samples: list[SurrogateSample]) -> SurrogateBatch:
    return SurrogateBatch(
        C      = torch.stack([s.C      for s in samples]),
        rho    = torch.stack([s.rho    for s in samples]),
        f_star = torch.stack([s.f_star for s in samples]),
        loss   = torch.stack([s.loss   for s in samples]),
    )


def _resolve_grid(n_cells_full: int, n_x: int | None, n_y: int | None) -> tuple[int, int]:
    """Fill in missing axis sizes. Both None -> square grid via sqrt."""
    if n_x is None and n_y is None:
        side = int(round(n_cells_full ** 0.5))
        if side * side != n_cells_full:
            raise ValueError(
                f"Cannot infer square grid: n_cells={n_cells_full} is not a perfect square. "
                f"Provide n_x and/or n_y explicitly."
            )
        return side, side
    if n_x is None:
        n_x = n_cells_full // n_y
    elif n_y is None:
        n_y = n_cells_full // n_x
    if n_x * n_y != n_cells_full:
        raise ValueError(f"n_x * n_y = {n_x * n_y} does not match n_cells = {n_cells_full}.")
    return n_x, n_y


class SurrogateDataset(Dataset[SurrogateSample]):
    def __init__(
        self,
        path: str | Path,
        cloak_only: bool = True,
        sample_types: list[str] | None = None,
        n_x: int | None = None,
        n_y: int | None = None,
        dtype: torch.dtype = torch.float32,
        loss_transform: LossTransform = "none",
        loss_clip_floor: float = 1e-8,
    ):
        """
        Args:
            path:            HDF5 file produced by dataset_gen/.
            cloak_only:      zero out non-cloak cells in the returned grid.
                             The shape stays (n_x, n_y); only cloak cells carry values.
            sample_types:    optional category filter ("init" / "random" / "smooth" / "opt"),
                             matched via startswith on the stored string.
            n_x, n_y:        grid dims for reshape (ix varies slowest: idx = ix*n_y + iy).
                             If both None, inferred as square root of n_cells.
                             Only one can be provided; the other is derived.
            dtype:           torch dtype for the float tensors.
            loss_transform:  applied to ``loss`` before storing; use ``"log_clip"`` on
                             the hi-fi dataset. ``self.loss`` then holds transformed
                             values; use ``invert_loss_transform`` to recover raw.
            loss_clip_floor: floor for ``log_clip`` (ignored otherwise).
        """
        with h5py.File(path, "r") as f:
            C     = f["cell_C_flat"][:]                 # (N, C_full, P)
            rho   = f["cell_rho"][:]                     # (N, C_full)
            fstr  = f["f_star"][:]                       # (N,)
            loss  = f["loss"][:]                         # (N,)
            stype = f["sample_type"][:].astype(str)      # (N,)
            mask  = f["cloak_mask"][:].astype(bool)      # (C_full,)

        n_cells_full = C.shape[1]
        self.n_x, self.n_y = _resolve_grid(n_cells_full, n_x, n_y)

        if sample_types is not None:
            keep = np.zeros(len(stype), dtype=bool)
            for t in sample_types:
                keep |= np.char.startswith(stype, t)
            C, rho, fstr, loss, stype = C[keep], rho[keep], fstr[keep], loss[keep], stype[keep]

        if cloak_only:
            C   = C   * mask[None, :, None]          # zero non-cloak
            rho = rho * mask[None, :]

        N, _, P = C.shape
        C   = C.reshape(N, self.n_x, self.n_y, P)
        rho = rho.reshape(N, self.n_x, self.n_y)
        mask_grid = mask.reshape(self.n_x, self.n_y)

        loss_t = apply_loss_transform(loss, loss_transform, floor=loss_clip_floor)

        self.cloak_only      = cloak_only
        self.loss_transform  = loss_transform
        self.loss_clip_floor = loss_clip_floor
        self.C           = torch.from_numpy(np.ascontiguousarray(C)).to(dtype)
        self.rho         = torch.from_numpy(np.ascontiguousarray(rho)).to(dtype)
        self.f_star      = torch.from_numpy(fstr).to(dtype)
        self.loss        = torch.from_numpy(loss_t).to(dtype)
        self.sample_type = stype
        self.cloak_mask  = torch.from_numpy(np.ascontiguousarray(mask_grid))

    def __len__(self) -> int:
        return self.loss.shape[0]

    def __getitem__(self, i: int) -> SurrogateSample:
        return SurrogateSample(
            C      = self.C[i],
            rho    = self.rho[i],
            f_star = self.f_star[i],
            loss   = self.loss[i],
        )


def make_dataloader(
    path: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs,
) -> DataLoader:
    ds = SurrogateDataset(path, **dataset_kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_surrogate,
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "output/surrogate_dataset.h5"

    for label, kwargs in [("cloak_only (zeros outside)", dict(cloak_only=True)),
                          ("full grid",                  dict(cloak_only=False))]:
        loader = make_dataloader(path, batch_size=8, shuffle=False, **kwargs)
        ds = loader.dataset
        batch = next(iter(loader))
        nz = (batch.rho != 0).float().mean().item()
        print(f"[{label}]  N={len(ds)}  grid=({ds.n_x}, {ds.n_y})  "
              f"mask={tuple(ds.cloak_mask.shape)}  nonzero-frac(rho)={nz:.2f}")
        for name, t in [("C", batch.C), ("rho", batch.rho),
                        ("f_star", batch.f_star), ("loss", batch.loss)]:
            print(f"  {name:7s} {tuple(t.shape)}  {t.dtype}")
