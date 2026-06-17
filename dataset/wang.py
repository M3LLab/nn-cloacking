"""
Loader for the 3D microstructure dataset from arXiv:2401.13570.

Each sample consists of:
  - *binary_voxel  : 64³ binary geometry (packed bits, Fortran order with block reversal)
  - *binary_C      : 6×6 Voigt stiffness tensor (36 float32 values, 144 bytes)
  - *vol           : volume fraction (1 float32, 4 bytes)

The 3D geometry exploits cubic symmetry: only 1/8 of the full 128³ cell is stored.
For 2D use, we extract the z=0 cross-section (64×64 binary image) and the plane-strain
stiffness submatrix (3×3, indices [0,1,5] of the 6×6 Voigt matrix).
"""

from pathlib import Path
import numpy as np


def load_voxel(path: str | Path) -> np.ndarray:
    """Load 64³ binary geometry as a float32 array with values in {0, 1}."""
    with open(path, "rb") as f:
        bits = np.unpackbits(np.fromfile(f, dtype=np.uint8))
    arr = np.reshape(bits, (64, 64, 64), order="F").astype(np.float32)
    # Block reversal applied by the original loader
    for i in range(8):
        arr[i * 8:(i + 1) * 8] = arr[i * 8:(i + 1) * 8][::-1]
    return arr  # shape (64, 64, 64), values {0.0, 1.0}


def load_C_voigt(path: str | Path) -> np.ndarray:
    """Load 6×6 Voigt stiffness matrix from binary_C file."""
    with open(path, "rb") as f:
        flat = np.fromfile(f, dtype=np.float32)  # 36 values
    return flat.reshape(6, 6)  # C[i,j] in Voigt notation


def load_vol(path: str | Path) -> float:
    """Load scalar volume fraction."""
    with open(path, "rb") as f:
        return float(np.fromfile(f, dtype=np.float32)[0])


def load_sample_2d(voxel_path: str | Path, z_slice: int = 0):
    """
    Load a single sample and return 2D geometry and stiffness.

    Parameters
    ----------
    voxel_path : path to *binary_voxel file
    z_slice    : which z-layer to use as the 2D cross-section (default 0)

    Returns
    -------
    geom2d : np.ndarray, shape (64, 64), float32, values {0.0, 1.0}
    C3x3   : np.ndarray, shape (3, 3), float32
        Plane-strain stiffness in 2D Voigt order [ε11, ε22, γ12]:
            [[C11, C12, C16],
             [C12, C22, C26],
             [C16, C26, C66]]
        For the cubic microstructures here C16=C26=0, C11=C22, so effectively
            [[C11, C12,  0 ],
             [C12, C11,  0 ],
             [0,    0,  C66]]
    vol    : float, volume fraction
    """
    voxel_path = Path(voxel_path)
    C_path = Path(str(voxel_path).replace("binary_voxel", "binary_C"))
    vol_path = Path(str(voxel_path).replace("binary_voxel", "vol"))

    voxel3d = load_voxel(voxel_path)
    geom2d = voxel3d[:, :, z_slice]          # (64, 64)

    C6x6 = load_C_voigt(C_path)
    idx = [0, 1, 5]                           # Voigt rows/cols for 2D plane strain
    C3x3 = C6x6[np.ix_(idx, idx)]

    vol = load_vol(vol_path)
    return geom2d, C3x3, vol


def iter_dataset(root: str | Path, pattern: str = "**/*binary_voxel", z_slice: int = 0):
    """
    Iterate over all samples under *root*, yielding (geom2d, C3x3, vol) tuples.

    Usage
    -----
    for geom, C, vol in iter_dataset("/home/david/Downloads/structures_dataset/3/rand8000"):
        ...
    """
    for voxel_path in sorted(Path(root).glob(pattern)):
        yield load_sample_2d(voxel_path, z_slice=z_slice)


def load_all(root: str | Path, pattern: str = "**/*binary_voxel", z_slice: int = 0):
    """
    Load all samples into arrays.

    Returns
    -------
    geoms : np.ndarray, shape (N, 64, 64)
    Cs    : np.ndarray, shape (N, 3, 3)
    vols  : np.ndarray, shape (N,)
    """
    geoms, Cs, vols = [], [], []
    for geom, C, vol in iter_dataset(root, pattern=pattern, z_slice=z_slice):
        geoms.append(geom)
        Cs.append(C)
        vols.append(vol)
    return np.stack(geoms), np.stack(Cs), np.array(vols, dtype=np.float32)
