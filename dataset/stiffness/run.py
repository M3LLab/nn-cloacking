"""Compute effective stiffness for all CA dataset unit cells and store in HDF5.

Runs FEM homogenization (calc_fem.py) for every *.npy in the dataset directory,
writing results into a single HDF5 file suitable for ML training at scale (~200k+).

HDF5 layout:
    /geometry    (N, H, W)   int8    — binary pixel images
    /C_eff       (N, 4, 4)   float64 — augmented Voigt stiffness [σ11,σ22,σ12,σ21]
    /rho_eff     (N,)        float64 — effective density (kg/m³)
    /vf          (N,)        float64 — volume fraction
    /chirality   (N,)        float64 — |C1212 - C2121| / (C1212 + C2121)
    /names       (N,)        str     — sample name (stem of .npy file)

Supports --resume to skip already-computed samples.

Usage:
    python -m dataset.stiffness.run
    python -m dataset.stiffness.run --dataset output/ca_chiral/dataset --out results.h5
    python -m dataset.stiffness.run --resume
"""

from __future__ import annotations

import argparse
import contextlib
import io
import logging
import signal
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress jax-fem banner and logger output
with contextlib.redirect_stdout(io.StringIO()):
    from .calc_fem import compute_effective_stiffness
logging.getLogger("jax_fem").setLevel(logging.ERROR)

# ---------------------------------------------------------------------------
# Chirality metric
# ---------------------------------------------------------------------------

def chirality(C: np.ndarray) -> float:
    """χ = |C_1212 - C_2121| / (C_1212 + C_2121)."""
    c1212, c2121 = C[2, 2], C[3, 3]
    denom = c1212 + c2121
    return 0.0 if denom == 0.0 else abs(c1212 - c2121) / denom


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE = 256  # samples per chunk — good balance for sequential & random access
SAMPLE_TIMEOUT = 20  # seconds — skip samples that take longer than this


class _Timeout:
    """Context manager that raises TimeoutError after `seconds` on POSIX."""
    def __init__(self, seconds: int):
        self.seconds = seconds

    def _handler(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        self._old = signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        signal.alarm(0)
        signal.signal(signal.SIGALRM, self._old)
        return False

def _create_datasets(f: h5py.File, img_shape: tuple[int, int]):
    """Create resizable datasets in a fresh HDF5 file."""
    H, W = img_shape
    f.create_dataset("geometry", shape=(0, H, W), maxshape=(None, H, W),
                     dtype=np.int8, chunks=(CHUNK_SIZE, H, W),
                     compression="gzip", compression_opts=4)
    f.create_dataset("C_eff", shape=(0, 4, 4), maxshape=(None, 4, 4),
                     dtype=np.float64, chunks=(CHUNK_SIZE, 4, 4))
    f.create_dataset("rho_eff", shape=(0,), maxshape=(None,),
                     dtype=np.float64, chunks=(CHUNK_SIZE,))
    f.create_dataset("vf", shape=(0,), maxshape=(None,),
                     dtype=np.float64, chunks=(CHUNK_SIZE,))
    f.create_dataset("chirality", shape=(0,), maxshape=(None,),
                     dtype=np.float64, chunks=(CHUNK_SIZE,))
    dt = h5py.string_dtype()
    f.create_dataset("names", shape=(0,), maxshape=(None,), dtype=dt,
                     chunks=(CHUNK_SIZE,))


def _append_sample(f: h5py.File, name: str, pixel_image: np.ndarray,
                   C: np.ndarray, rho: float, vf: float, chi: float):
    """Append one sample to every dataset."""
    idx = f["C_eff"].shape[0]
    for key in ("geometry", "C_eff", "rho_eff", "vf", "chirality", "names"):
        f[key].resize(idx + 1, axis=0)
    f["geometry"][idx] = pixel_image.astype(np.int8)
    f["C_eff"][idx] = C
    f["rho_eff"][idx] = rho
    f["vf"][idx] = vf
    f["chirality"][idx] = chi
    f["names"][idx] = name


def _existing_names(f: h5py.File) -> set[str]:
    """Return set of sample names already in the file."""
    if "names" not in f:
        return set()
    return set(n.decode() if isinstance(n, bytes) else n for n in f["names"][:])


# ---------------------------------------------------------------------------
# Summary & plots
# ---------------------------------------------------------------------------

def _print_summary(f: h5py.File, out_dir: Path):
    n = f["C_eff"].shape[0]
    if n == 0:
        return
    chis = f["chirality"][:]
    c1212 = f["C_eff"][:, 2, 2]
    c2121 = f["C_eff"][:, 3, 3]
    names = [s.decode() if isinstance(s, bytes) else s for s in f["names"][:]]

    lines = [
        "=" * 72,
        f"{'Sample':<15}  {'χ':>10}  {'C_1212 (Pa)':>14}  {'C_2121 (Pa)':>14}",
        "-" * 72,
    ]
    for nm, chi, c12, c21 in zip(names, chis, c1212, c2121):
        lines.append(f"{nm:<15}  {chi:10.6f}  {c12:14.4e}  {c21:14.4e}")
    lines += [
        "-" * 72,
        f"{'mean':<15}  {chis.mean():10.6f}  {c1212.mean():14.4e}  {c2121.mean():14.4e}",
        f"{'std':<15}  {chis.std():10.6f}  {c1212.std():14.4e}  {c2121.std():14.4e}",
        f"{'min':<15}  {chis.min():10.6f}  {c1212.min():14.4e}  {c2121.min():14.4e}",
        f"{'max':<15}  {chis.max():10.6f}  {c1212.max():14.4e}  {c2121.max():14.4e}",
        "=" * 72,
    ]
    summary = "\n".join(lines)
    print(f"\nChirality summary ({n} samples):")
    print(summary)
    (out_dir / "chirality_summary.txt").write_text(summary + "\n")


def _plot(f: h5py.File, out_dir: Path):
    n = f["C_eff"].shape[0]
    if n == 0:
        return
    chis = f["chirality"][:]
    c1212 = f["C_eff"][:, 2, 2]
    c2121 = f["C_eff"][:, 3, 3]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(chis, bins=max(5, n // 3), color="steelblue", edgecolor="white",
            linewidth=0.8)
    ax.axvline(chis.mean(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"mean = {chis.mean():.4f}")
    ax.set_xlabel(r"Chirality  $\chi = \frac{|C_{1212} - C_{2121}|}{C_{1212} + C_{2121}}$",
                  fontsize=12)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Chirality distribution", fontsize=13)
    ax.legend(fontsize=10)

    ax2 = axes[1]
    x = np.arange(min(n, 100))  # cap bar plot at 100 samples for readability
    ax2.bar(x, c1212[:len(x)] / 1e9, label=r"$C_{1212}$", alpha=0.7, color="steelblue")
    ax2.bar(x, c2121[:len(x)] / 1e9, label=r"$C_{2121}$", alpha=0.7, color="darkorange")
    ax2.set_xlabel("Sample index", fontsize=11)
    ax2.set_ylabel("Stiffness (GPa)", fontsize=11)
    ax2.set_title(r"$C_{1212}$ and $C_{2121}$ per sample", fontsize=13)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    plot_path = out_dir / "chirality_distribution.png"
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="output/ca_chiral/dataset",
                        help="Directory containing *.npy pixel images")
    parser.add_argument("--out", default="dataset/stiffness/results/dataset.h5",
                        help="Output HDF5 path")
    parser.add_argument("--resume", action="store_true",
                        help="Skip samples already in the HDF5 file")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip summary plot generation")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    npy_files = sorted(dataset_dir.glob("*.npy"))
    if not npy_files:
        print(f"No *.npy files found in {dataset_dir}")
        sys.exit(1)

    # Peek at first image to get spatial dimensions
    first_img = np.load(npy_files[0])
    img_shape = first_img.shape
    assert len(img_shape) == 2, f"Expected 2D images, got shape {img_shape}"

    mode = "a" if args.resume and out_path.exists() else "w"
    with h5py.File(out_path, mode) as f:
        if "C_eff" not in f:
            _create_datasets(f, img_shape)

        done = _existing_names(f) if args.resume else set()
        todo = [p for p in npy_files if p.stem not in done]
        n_skip = len(npy_files) - len(todo)

        n_timeout = 0
        pbar = tqdm(todo, desc="FEM homogenization", unit="cell")
        for npy_path in pbar:
            name = npy_path.stem
            pbar.set_postfix_str(name)
            try:
                with _Timeout(SAMPLE_TIMEOUT):
                    C, rho = compute_effective_stiffness(npy_path, output_path=None, verbose=False)
            except TimeoutError:
                n_timeout += 1
                tqdm.write(f"  SKIP (>{SAMPLE_TIMEOUT}s): {name}")
                continue
            pixel_image = np.load(npy_path)
            vf = float(pixel_image.astype(float).mean())
            chi = chirality(C)
            _append_sample(f, name, pixel_image, C, rho, vf, chi)

            # Flush periodically so progress survives interrupts
            if f["C_eff"].shape[0] % CHUNK_SIZE == 0:
                f.flush()

        if n_skip:
            print(f"Skipped {n_skip} already-computed samples")
        if n_timeout:
            print(f"Timed out {n_timeout} samples (>{SAMPLE_TIMEOUT}s)")
        print(f"Total samples in {out_path}: {f['C_eff'].shape[0]}")

        _print_summary(f, out_path.parent)
        if not args.no_plot:
            _plot(f, out_path.parent)


if __name__ == "__main__":
    main()
