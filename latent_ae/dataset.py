"""Dataset re-exports. Reuses surrogate/dataset.py verbatim."""
from surrogate.dataset import (
    LossTransform,
    SurrogateBatch,
    SurrogateDataset,
    SurrogateSample,
    apply_loss_transform,
    collate_surrogate,
    invert_loss_transform,
    make_dataloader,
)

__all__ = [
    "LossTransform",
    "SurrogateBatch",
    "SurrogateDataset",
    "SurrogateSample",
    "apply_loss_transform",
    "collate_surrogate",
    "invert_loss_transform",
    "make_dataloader",
]
