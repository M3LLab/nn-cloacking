"""Allow running as: python -m dataset.cellular_chiral"""
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .generator import CAConfig, generate_unit_cell, ASSEMBLY_MODES
from .visualize import plot_quadrant_and_cell


def generate_dataset(N: int, seed: int, assembly: str, out_dir: Path, plot: bool = True):
    dataset_dir = out_dir / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if plot:
        individual_dir = out_dir / "individual"
        individual_dir.mkdir(parents=True, exist_ok=True)

    config = CAConfig()

    for s in tqdm(range(seed, seed + N), desc="Generating", unit="cell"):
        uc, q = generate_unit_cell(config=config, seed=s, assembly=assembly)
        if plot:
            plot_quadrant_and_cell(q, uc, s, save_path=individual_dir / f"seed_{s}.png")
        np.save(dataset_dir / f"seed_{s}.npy", uc)


def main():
    parser = argparse.ArgumentParser(description="Generate cellular unit-cell dataset")
    parser.add_argument("-n", "--num", type=int, default=16, help="Number of samples")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Starting seed")
    parser.add_argument(
        "-a", "--assembly",
        choices=list(ASSEMBLY_MODES),
        default="chiral",
        help="Assembly mode: 'chiral' (rotational) or 'squared' (mirror/D4)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("output/ca_chiral"),
        help="Output directory",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip PNG plot generation",
    )
    args = parser.parse_args()
    generate_dataset(N=args.num, seed=args.seed, assembly=args.assembly,
                     out_dir=args.output, plot=not args.no_plot)


main()
