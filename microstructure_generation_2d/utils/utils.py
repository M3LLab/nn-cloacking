from __future__ import annotations

import os
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def set_requires_grad(model, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad = flag


def ensure_directory(directory: str) -> None:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def update_moving_average(ma_model, current_model, ema_updater) -> None:
    for cur, ma in zip(current_model.parameters(), ma_model.parameters()):
        ma.data = ema_updater.update_average(ma.data, cur.data)


def find_best_epoch(ckpt_folder: str):
    try:
        files = os.listdir(ckpt_folder)
        epochs = [int(fn.split(".")[0].split("=")[1]) for fn in files if "epoch=" in fn]
        return max(epochs) if epochs else 0
    except Exception:
        return None
