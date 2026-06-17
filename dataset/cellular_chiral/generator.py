"""
Cellular automata-based chiral unit cell generator.

Implements the dataset generation method from:
  Nakarmi et al., "Predicting non-linear stress-strain response of mesostructured
  cellular materials using supervised autoencoder", CMAME 432 (2024) 117372.

Modified to produce chiral (asymmetric) structures by assembling four 90°-rotated
copies of a single quadrant instead of mirroring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class CAConfig:
    """Configuration for cellular automata generation."""

    grid_size: int = 25
    live_fraction: float = 0.49
    num_iterations: int = 7
    gate_width: int = 5
    bridge_width: int = 4
    birth_range: Tuple[int, ...] = (5, 6, 7, 8)
    death_range: Tuple[int, ...] = (0, 1, 2, 3)
    survive_count: int = 4


def _initialize_grid(
    size: int, live_fraction: float, gate_width: int, rng: np.random.Generator
) -> Tuple[NDArray[np.int8], NDArray[np.bool_]]:
    """Create initial grid with random interior, solid borders, and centered gates.

    Returns the grid and a boolean mask of frozen (gate) pixels.
    """
    grid = np.zeros((size, size), dtype=np.int8)

    # Random interior (~49% live)
    n_interior = (size - 2) ** 2
    n_live = int(round(live_fraction * size * size))
    interior_flat = np.zeros(n_interior, dtype=np.int8)
    interior_flat[: min(n_live, n_interior)] = 1
    rng.shuffle(interior_flat)
    grid[1 : size - 1, 1 : size - 1] = interior_flat.reshape(size - 2, size - 2)

    # Set all border pixels to live (1)
    grid[0, :] = 1
    grid[size - 1, :] = 1
    grid[:, 0] = 1
    grid[:, size - 1] = 1

    # Create centered gates (dead pixels on borders) — same position on all edges
    # so that gates align under 90° rotation
    gate_start = (size - gate_width) // 2
    gate_end = gate_start + gate_width

    frozen = np.zeros((size, size), dtype=bool)

    # Top edge gates
    grid[0, gate_start:gate_end] = 0
    frozen[0, gate_start:gate_end] = True

    # Bottom edge gates
    grid[size - 1, gate_start:gate_end] = 0
    frozen[size - 1, gate_start:gate_end] = True

    # Left edge gates
    grid[gate_start:gate_end, 0] = 0
    frozen[gate_start:gate_end, 0] = True

    # Right edge gates
    grid[gate_start:gate_end, size - 1] = 0
    frozen[gate_start:gate_end, size - 1] = True

    # Freeze all border pixels (they don't evolve)
    frozen[0, :] = True
    frozen[size - 1, :] = True
    frozen[:, 0] = True
    frozen[:, size - 1] = True

    return grid, frozen


def _count_neighbors(grid: NDArray[np.int8]) -> NDArray[np.int32]:
    """Count the 8-connected neighbors of each cell."""
    h, w = grid.shape
    counts = np.zeros((h, w), dtype=np.int32)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            shifted = np.roll(np.roll(grid, -di, axis=0), -dj, axis=1)
            counts += shifted
    # Zero out border neighbor counts (borders don't evolve)
    return counts


def _step_ca(
    grid: NDArray[np.int8],
    frozen: NDArray[np.bool_],
    config: CAConfig,
) -> NDArray[np.int8]:
    """Apply one step of cellular automata rules."""
    neighbors = _count_neighbors(grid)
    new_grid = grid.copy()

    # Birth: dead cell with 5-8 neighbors becomes alive
    birth = (grid == 0) & np.isin(neighbors, config.birth_range)
    new_grid[birth] = 1

    # Death: live cell with 0-3 neighbors dies
    death = (grid == 1) & np.isin(neighbors, config.death_range)
    new_grid[death] = 0

    # Survive: exactly 4 neighbors → keep current state (no change needed)

    # Frozen pixels (borders + gates) don't change
    new_grid[frozen] = grid[frozen]

    return new_grid


def _reverse_map(grid: NDArray[np.int8]) -> NDArray[np.int8]:
    """Invert the grid: dead→material(1), live→void(0)."""
    return 1 - grid


def _flood_fill(pattern: NDArray[np.int8]) -> List[NDArray[np.int64]]:
    """Identify disconnected regions of material (1) pixels.

    Returns a list of regions, where each region is an array of pixel
    indices (flattened) belonging to that region.

    Implements Algorithm 1 from Nakarmi et al. 2024.
    """
    h, w = pattern.shape
    visited = np.zeros_like(pattern, dtype=bool)
    regions: List[NDArray[np.int64]] = []

    material_positions = np.argwhere(pattern == 1)

    for pos in material_positions:
        r, c = pos
        if visited[r, c]:
            continue

        # BFS flood fill
        region_pixels = []
        queue = [(r, c)]
        visited[r, c] = True

        while queue:
            cr, cc = queue.pop(0)
            region_pixels.append(cr * w + cc)

            # 4-connected neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and pattern[nr, nc] == 1:
                    visited[nr, nc] = True
                    queue.append((nr, nc))

        regions.append(np.array(region_pixels, dtype=np.int64))

    return regions


def _build_connection_graph(
    regions: List[NDArray[np.int64]], width: int
) -> List[Tuple[int, int, int, int, int, float]]:
    """Build connection graph between all pairs of disconnected regions.

    Returns edges as (region_u, region_v, pixel_m_row, pixel_m_col,
    pixel_n_row, pixel_n_col, distance).

    For each pair of regions, finds the nearest pixels between them.
    """
    n_regions = len(regions)
    if n_regions <= 1:
        return []

    # Convert flat indices to (row, col) for each region
    region_coords = []
    for reg in regions:
        rows = reg // width
        cols = reg % width
        region_coords.append(np.column_stack([rows, cols]))

    edges = []
    for i in range(n_regions):
        for j in range(i + 1, n_regions):
            coords_i = region_coords[i]
            coords_j = region_coords[j]

            # Compute pairwise distances (Chebyshev/L-inf for grid, but
            # we use Euclidean to pick nearest pair, Manhattan for bridge cost)
            diff = coords_i[:, np.newaxis, :] - coords_j[np.newaxis, :, :]
            dists = np.sum(diff**2, axis=2)
            min_idx = np.unravel_index(np.argmin(dists), dists.shape)

            m = coords_i[min_idx[0]]
            n = coords_j[min_idx[1]]
            dist = np.sqrt(dists[min_idx[0], min_idx[1]])

            edges.append((i, j, m[0], m[1], n[0], n[1], dist))

    return edges


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> bool:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True


def _kruskal_mst(
    edges: List[Tuple[int, int, int, int, int, int, float]], n_regions: int
) -> List[Tuple[int, int, int, int, int, int, float]]:
    """Kruskal's minimum spanning tree algorithm.

    Returns the MST edges sorted by weight.
    Implements Algorithm 2 from Nakarmi et al. 2024.
    """
    # Sort edges by distance
    sorted_edges = sorted(edges, key=lambda e: e[6])

    uf = UnionFind(n_regions)
    mst = []

    for edge in sorted_edges:
        u, v = edge[0], edge[1]
        if uf.union(u, v):
            mst.append(edge)
            if len(mst) == n_regions - 1:
                break

    return mst


def _draw_bridge(
    pattern: NDArray[np.int8],
    r1: int, c1: int,
    r2: int, c2: int,
    width: int,
) -> None:
    """Draw a bridge of given width between two points using L-shaped path.

    The bridge goes horizontally first, then vertically (or vice versa),
    with the specified pixel width.
    """
    h, w_grid = pattern.shape
    half = width // 2

    # L-shaped path: horizontal from (r1,c1) to (r1,c2), then vertical to (r2,c2)
    # Draw horizontal segment
    r_lo = max(0, r1 - half)
    r_hi = min(h, r1 - half + width)
    c_min, c_max = min(c1, c2), max(c1, c2)
    pattern[r_lo:r_hi, c_min : c_max + 1] = 1

    # Draw vertical segment
    c_lo = max(0, c2 - half)
    c_hi = min(w_grid, c2 - half + width)
    r_min, r_max = min(r1, r2), max(r1, r2)
    pattern[r_min : r_max + 1, c_lo:c_hi] = 1


def _connect_regions(
    pattern: NDArray[np.int8], bridge_width: int
) -> NDArray[np.int8]:
    """Connect disconnected material regions using flood fill + Kruskal's MST."""
    h, w = pattern.shape
    result = pattern.copy()

    regions = _flood_fill(result)
    if len(regions) <= 1:
        return result

    edges = _build_connection_graph(regions, w)
    mst = _kruskal_mst(edges, len(regions))

    for edge in mst:
        _, _, r1, c1, r2, c2, _ = edge
        _draw_bridge(result, r1, c1, r2, c2, bridge_width)

    return result


def generate_quadrant(
    config: Optional[CAConfig] = None,
    seed: Optional[int] = None,
) -> NDArray[np.int8]:
    """Generate a single quadrant using cellular automata.

    Steps:
    1. Initialize random grid with borders and centered gates
    2. Run CA for num_iterations steps
    3. Reverse map (invert)
    4. Connect disconnected regions via flood fill + MST

    Returns an (N x N) binary array (1=material, 0=void).
    """
    if config is None:
        config = CAConfig()

    rng = np.random.default_rng(seed)
    N = config.grid_size

    # Step 1: Initialize
    grid, frozen = _initialize_grid(N, config.live_fraction, config.gate_width, rng)

    # Step 2: CA evolution
    for _ in range(config.num_iterations):
        grid = _step_ca(grid, frozen, config)

    # Step 3: Reverse map
    material = _reverse_map(grid)

    # Step 4: Connect disconnected regions
    material = _connect_regions(material, config.bridge_width)

    return material


def _assemble_squared(quadrant: NDArray[np.int8]) -> NDArray[np.int8]:
    """Assemble a unit cell with square (D4) symmetry from four mirrored copies.

    Layout (2x2):
        TL = quadrant            | TR = flipped left-right
        BL = flipped up-down     | BR = flipped both axes

    Mirror assembly produces reflection symmetry along both the horizontal
    and vertical center lines, yielding full square symmetry (no chirality).
    The result is a 2N x 2N grid.
    """
    N = quadrant.shape[0]

    tl = quadrant.copy()
    tr = np.fliplr(quadrant)        # mirror horizontally
    bl = np.flipud(quadrant)        # mirror vertically
    br = np.flipud(np.fliplr(quadrant))  # mirror both

    full = np.zeros((2 * N, 2 * N), dtype=np.int8)
    full[:N, :N] = tl
    full[:N, N:] = tr
    full[N:, :N] = bl
    full[N:, N:] = br

    return full


def _assemble_chiral(quadrant: NDArray[np.int8]) -> NDArray[np.int8]:
    """Assemble a chiral unit cell from four rotated copies of a quadrant.

    Layout (2x2):
        TL = quadrant (0°)     | TR = rot90 CW (270° CCW)
        BL = rot270 CW (90° CCW) | BR = rot180

    Gates are centered on each edge, so they align when rotated copies are joined.
    The result is a 2N x 2N grid (borders NOT overlapped, gate openings face each other).
    """
    N = quadrant.shape[0]

    tl = quadrant.copy()
    tr = np.rot90(quadrant, k=-1)   # 90° CW
    br = np.rot90(quadrant, k=2)    # 180°
    bl = np.rot90(quadrant, k=1)    # 90° CCW (= 270° CW)

    full = np.zeros((2 * N, 2 * N), dtype=np.int8)
    full[:N, :N] = tl
    full[:N, N:] = tr
    full[N:, :N] = bl
    full[N:, N:] = br

    return full


ASSEMBLY_MODES = {"chiral": _assemble_chiral, "squared": _assemble_squared}


def generate_unit_cell(
    config: Optional[CAConfig] = None,
    seed: Optional[int] = None,
    assembly: str = "chiral",
) -> Tuple[NDArray[np.int8], NDArray[np.int8]]:
    """Generate a unit cell by CA + assembly.

    Args:
        config: CA generation parameters.
        seed: Random seed for reproducibility.
        assembly: Assembly mode — "chiral" (rotational) or "squared" (mirror/D4).

    Returns:
        (unit_cell, quadrant): The full 2N×2N unit cell and the N×N quadrant.
    """
    if assembly not in ASSEMBLY_MODES:
        raise ValueError(f"Unknown assembly mode {assembly!r}, choose from {list(ASSEMBLY_MODES)}")
    quadrant = generate_quadrant(config=config, seed=seed)
    unit_cell = ASSEMBLY_MODES[assembly](quadrant)
    return unit_cell, quadrant


# Keep old name as alias for backwards compatibility
generate_chiral_unit_cell = generate_unit_cell
