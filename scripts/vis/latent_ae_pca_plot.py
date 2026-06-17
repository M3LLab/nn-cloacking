"""PCA of the autoencoder latent on the hi-fi dataset.

Usage:
    python scripts/latent_ae_pca_plot.py [checkpoint_path]

Defaults to ``output/latent_ae_train_hi_fi/best.pt``. Writes two figures and
a CSV to ``{checkpoint_dir}/pca/``:
    pca_z_sh_by_f.png      — first 2 PCs of z_sh, colored by f_star
    pca_z_sh_by_loss.png   — first 2 PCs of z_sh, colored by log10(loss)
    pca_coords.csv         — (pc1, pc2, f_star, loss, sample_type) per sample
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import csv

# Ensure project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from latent_ae.evaluate import encode_dataset, load_model  # noqa: E402
from surrogate.dataset import SurrogateDataset  # noqa: E402


def main(checkpoint_path: str) -> None:
    ckpt_path = Path(checkpoint_path)
    cfg = json.loads((ckpt_path.parent / "config.json").read_text())

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Loading model from {ckpt_path} on {device} ...")
    model, _ = load_model(ckpt_path, device=device)

    print(f"Loading dataset {cfg['data_path']} ...")
    ds = SurrogateDataset(
        cfg["data_path"],
        loss_transform=cfg["loss_transform"],
        loss_clip_floor=cfg["loss_clip_floor"],
    )

    print(f"Encoding {len(ds)} samples ...")
    enc = encode_dataset(model, ds, batch_size=cfg["batch_size"], device=device)

    print("Fitting PCA(2) on z_sh ...")
    pca = PCA(n_components=2)
    pc = pca.fit_transform(enc["z_sh"])
    explained = pca.explained_variance_ratio_
    print(f"  explained variance: {explained[0]:.3f}, {explained[1]:.3f}")

    out_dir = ckpt_path.parent / "pca"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    with open(out_dir / "pca_coords.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pc1", "pc2", "f_star", "loss", "sample_type"])
        for r in zip(pc[:, 0], pc[:, 1], enc["f_star"], enc["loss"], enc["sample_type"]):
            w.writerow([float(r[0]), float(r[1]), float(r[2]), float(r[3]), str(r[4])])

    # ── Plot 1: colored by f_star ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(
        pc[:, 0], pc[:, 1],
        c=enc["f_star"], cmap="viridis",
        s=8, alpha=0.7, edgecolors="none",
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$f^\star$ (normalized frequency)")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(r"PCA of $z_{sh}$, colored by frequency")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_z_sh_by_f.png", dpi=140)
    plt.close(fig)

    # ── Plot 2: colored by log10(loss) ────────────────────────────────
    loss_log10 = np.log10(np.clip(enc["loss"], 1e-18, None))
    fig, ax = plt.subplots(figsize=(7, 6))
    # Use diverging colormap centered at the median; low-loss (blue) is what we care about
    vmin, vmax = np.quantile(loss_log10, [0.02, 0.98])
    sc = ax.scatter(
        pc[:, 0], pc[:, 1],
        c=loss_log10, cmap="coolwarm_r",
        vmin=vmin, vmax=vmax,
        s=8, alpha=0.7, edgecolors="none",
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(r"$\log_{10}$(loss)")
    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
    ax.set_title(r"PCA of $z_{sh}$, colored by loss")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_z_sh_by_loss.png", dpi=140)
    plt.close(fig)

    print(f"Wrote plots to {out_dir}/")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "output/latent_ae_train_hi_fi/best.pt"
    main(path)
