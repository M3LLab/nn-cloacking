"""
chiral_ca_generator.py

Generate diverse 2D chiral metamaterial unit-cell binary images (material=1, void=0)
using a cellular automaton (CA) whose states are chiral building-block tiles.

Key features:
- Output size: 128×128
- Coarse-graining: CA grid is (128/k)×(128/k), each cell maps to k×k pixels, k∈{1,2,4,8,16}
- Tile states: tetrachiral ring-ligament, rotated inclusion, blocky chiral, hierarchical, freeform
- Precomputed or procedural tiles
- Periodic wrap in CA + periodic postprocessing
- Connectivity enforcement (single connected solid region on a torus) via repair or reject
- Minimum feature width ≥ 3 pixels
- Mirror partner generation + chirality score
- Saves masks (PNG+NPY), signed distance fields (NPY), metadata (JSON + JSONL)

Dependencies:
  numpy, scipy, scikit-image, pillow
Optional (for mosaics): matplotlib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.fft import fft2, ifft2
from PIL import Image
from scipy import ndimage
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial import cKDTree
from skimage import draw, measure, morphology, transform


# -----------------------------
# Utilities
# -----------------------------

FAMILIES = ["tetrachiral", "rotated_inclusion", "blocky", "hierarchical", "freeform"]


def _disk(radius_px: float) -> np.ndarray:
    """Binary disk structuring element."""
    r = int(np.ceil(radius_px))
    if r <= 0:
        return np.array([[True]], dtype=bool)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    m = x * x + y * y <= radius_px * radius_px
    return m.astype(bool)


def downsample_antialias(img_hi: np.ndarray, out_px: int) -> np.ndarray:
    """Downsample float [0,1] image to out_px×out_px with antialiasing."""
    if img_hi.shape[0] == out_px and img_hi.shape[1] == out_px:
        return img_hi.astype(np.float32)
    out = transform.resize(
        img_hi,
        (out_px, out_px),
        order=1,
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def binarize(img: np.ndarray, thresh: float = 0.5) -> np.ndarray:
    return (img >= thresh).astype(bool)


def mask_hash(mask: np.ndarray, pool: int = 32) -> str:
    """Hash of a pooled-down version of the mask for fast dedupe."""
    small = (
        transform.resize(
            mask.astype(np.float32),
            (pool, pool),
            order=0,
            anti_aliasing=False,
            preserve_range=True,
        )
        > 0.5
    )
    bits = np.packbits(small.astype(np.uint8).ravel())
    return hashlib.sha1(bits.tobytes()).hexdigest()


def signed_distance_field(mask: np.ndarray) -> np.ndarray:
    """Signed distance: positive inside material, negative in void."""
    m = mask.astype(bool)
    dist_in = ndimage.distance_transform_edt(m)
    dist_out = ndimage.distance_transform_edt(~m)
    return (dist_in - dist_out).astype(np.float32)


def chirality_score(mask: np.ndarray) -> float:
    """
    Chirality score in [0,1]. We compute the maximum cross-correlation
    between the mask and its left-right mirror across all periodic shifts
    (FFT-based). Score = 1 - max_corr.
    """
    a = mask.astype(np.float32)
    b = np.fliplr(mask).astype(np.float32)

    a = a - a.mean()
    b = b - b.mean()

    fa = fft2(a)
    fb = fft2(b)
    corr = np.real(ifft2(fa * np.conj(fb)))

    denom = np.sqrt((a * a).sum() * (b * b).sum()) + 1e-8
    max_corr = float(np.clip(corr.max() / denom, -1.0, 1.0))
    return float(np.clip(1.0 - max_corr, 0.0, 1.0))


def min_feature_width_px(mask: np.ndarray) -> float:
    """Estimate minimum solid feature width via skeleton + distance transform."""
    if mask.sum() == 0:
        return 0.0
    dist = ndimage.distance_transform_edt(mask.astype(bool))
    skel = morphology.skeletonize(mask.astype(bool))
    if skel.sum() == 0:
        return float(2.0 * dist[mask.astype(bool)].min())
    return float(2.0 * dist[skel].min())


# -----------------------------
# Primitive rendering (tile states)
# -----------------------------

def rasterize_segments(shape: Tuple[int, int],
                       segments: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                       thickness: int) -> np.ndarray:
    """Binary mask of thick line segments."""
    img = np.zeros(shape, dtype=bool)
    for (p0, p1) in segments:
        rr, cc = draw.line(int(round(p0[0])), int(round(p0[1])),
                           int(round(p1[0])), int(round(p1[1])))
        rr = np.clip(rr, 0, shape[0] - 1)
        cc = np.clip(cc, 0, shape[1] - 1)
        img[rr, cc] = True
    if thickness > 1:
        img = ndimage.binary_dilation(img, structure=_disk(thickness / 2.0))
    return img


def sample_params_for_family(family: str, rng: np.random.Generator, tile_px: int) -> Dict[str, Any]:
    """Sample parameters for a tile family (high-res tile size tile_px)."""
    min_hi = max(2, int(round(0.06 * tile_px)))
    max_hi = max(min_hi + 1, int(round(0.16 * tile_px)))

    if family == "tetrachiral":
        return {
            "r_outer": float(rng.uniform(0.22, 0.32)),
            "r_inner": float(rng.uniform(0.10, 0.20)),
            "alpha": float(rng.uniform(0.15, 0.75)),   # radians
            "lig_len": float(rng.uniform(0.75, 0.95)),
            "lig_thick": int(rng.integers(min_hi, max_hi + 1)),
        }

    if family == "rotated_inclusion":
        poly = str(rng.choice(["square", "tri"], p=[0.7, 0.3]))
        return {
            "poly": poly,
            "size": float(rng.uniform(0.38, 0.62)),
            "theta": float(rng.uniform(0.2, 1.0)),     # radians; multiplied by handedness
            "conn_thick": int(rng.integers(min_hi, max_hi + 1)),
            "connectors": True,
        }

    if family == "blocky":
        return {
            "block_frac": float(rng.uniform(0.35, 0.65)),
            "offset_frac": float(rng.uniform(0.25, 0.55)),
            "conn_thick": int(rng.integers(min_hi, max_hi + 1)),
        }

    if family == "hierarchical":
        base = sample_params_for_family("tetrachiral", rng, tile_px)
        base["inner_scale"] = float(rng.uniform(0.4, 0.7))
        return base

    if family == "freeform":
        return {
            "smooth_sigma": float(rng.uniform(1.2, 3.2)),
            "warp_strength": float(rng.uniform(1.5, 4.0)),
            "thresh": float(rng.uniform(-0.2, 0.2)),
            "thick": int(rng.integers(min_hi, max_hi + 1)),
        }

    raise ValueError(f"Unknown family: {family}")


def render_tetrachiral(tile_px: int, handedness: int, params: Dict[str, Any]) -> np.ndarray:
    """Ring + 4 ligaments with signed attachment angle."""
    S = tile_px
    cy = cx = (S - 1) / 2.0
    half = S / 2.0

    r_outer = params["r_outer"] * half
    r_inner = params["r_inner"] * half
    alpha = params["alpha"] * float(handedness)
    lig_len = params["lig_len"] * half
    lig_thick = int(params["lig_thick"])

    img = np.zeros((S, S), dtype=bool)

    rr_o, cc_o = draw.disk((cy, cx), r_outer, shape=img.shape)
    img[rr_o, cc_o] = True
    rr_i, cc_i = draw.disk((cy, cx), r_inner, shape=img.shape)
    img[rr_i, cc_i] = False

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    # East / West / North / South ligaments
    for base_phi, target in [
        (0.0, (cy, cx + lig_len)),
        (np.pi, (cy, cx - lig_len)),
        (np.pi / 2, (cy - lig_len, cx)),
        (3 * np.pi / 2, (cy + lig_len, cx)),
    ]:
        phi = base_phi + alpha
        p0 = (cy + r_outer * np.sin(phi), cx + r_outer * np.cos(phi))
        segments.append((p0, target))

    lig = rasterize_segments((S, S), segments, thickness=lig_thick)
    img |= lig
    return img.astype(np.float32)


def render_rotated_inclusion(tile_px: int, handedness: int, params: Dict[str, Any]) -> np.ndarray:
    """Rotated polygon inclusion plus connectors to edges."""
    S = tile_px
    cy = cx = (S - 1) / 2.0
    half = S / 2.0

    poly = params["poly"]
    size = params["size"] * half
    theta = params["theta"] * float(handedness)
    conn_thick = int(params["conn_thick"])
    add_conn = bool(params.get("connectors", True))

    img = np.zeros((S, S), dtype=bool)

    if poly == "tri":
        angles = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3], dtype=np.float32) + theta
    else:
        angles = np.array([np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4], dtype=np.float32) + theta

    verts = np.stack([cy + size * np.sin(angles), cx + size * np.cos(angles)], axis=1)

    rr, cc = draw.polygon(verts[:, 0], verts[:, 1], shape=img.shape)
    img[rr, cc] = True

    if add_conn:
        segments = []
        for (vy, vx) in verts:
            dy = float(vy - cy)
            dx = float(vx - cx)
            if abs(dx) > abs(dy):
                ex = 0 if dx < 0 else S - 1
                ey = float(vy)
            else:
                ey = 0 if dy < 0 else S - 1
                ex = float(vx)
            segments.append(((float(vy), float(vx)), (float(ey), float(ex))))
        conn = rasterize_segments((S, S), segments, thickness=conn_thick)
        img |= conn

    return img.astype(np.float32)


def render_blocky_chiral(tile_px: int, handedness: int, params: Dict[str, Any]) -> np.ndarray:
    """Two offset blocks + diagonal coupler + arms."""
    S = tile_px
    cy = cx = (S - 1) / 2.0
    half = S / 2.0

    b = params["block_frac"] * half
    off = params["offset_frac"] * half
    conn_thick = int(params["conn_thick"])

    img = np.zeros((S, S), dtype=bool)

    c1 = (cy - off, cx + float(handedness) * off * 0.6)
    c2 = (cy + off, cx - float(handedness) * off * 0.6)

    for (cY, cX) in [c1, c2]:
        y0 = int(round(cY - b / 2))
        y1 = int(round(cY + b / 2))
        x0 = int(round(cX - b / 2))
        x1 = int(round(cX + b / 2))
        y0 = max(y0, 0)
        x0 = max(x0, 0)
        y1 = min(y1, S - 1)
        x1 = min(x1, S - 1)
        img[y0 : y1 + 1, x0 : x1 + 1] = True

    img |= rasterize_segments((S, S), [(c1, c2)], thickness=conn_thick)

    arms = [
        ((cy, cx), (cy, 0)),
        ((cy, cx), (cy, S - 1)),
        ((cy, cx), (0, cx)),
        ((cy, cx), (S - 1, cx)),
    ]
    img |= rasterize_segments((S, S), arms, thickness=max(1, conn_thick - 2))

    return img.astype(np.float32)


def render_hierarchical(tile_px: int, handedness: int, params: Dict[str, Any]) -> np.ndarray:
    """Outer tetrachiral + scaled inner tetrachiral."""
    S = tile_px
    outer = render_tetrachiral(tile_px, handedness, params)

    inner_scale = float(params.get("inner_scale", 0.55))
    inner_S = int(round(S * inner_scale))
    if inner_S < 8:
        return outer.astype(np.float32)

    inner_params = dict(params)
    inner_params["lig_thick"] = max(2, int(round(params["lig_thick"] * 0.7)))
    inner_params["lig_len"] = float(params["lig_len"]) * 0.55

    inner = render_tetrachiral(inner_S, handedness, inner_params)

    y0 = (S - inner_S) // 2
    x0 = (S - inner_S) // 2
    out = outer.copy()
    out[y0 : y0 + inner_S, x0 : x0 + inner_S] = np.maximum(out[y0 : y0 + inner_S, x0 : x0 + inner_S], inner)
    return out.astype(np.float32)


def render_freeform_warp(tile_px: int, handedness: int, params: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    """
    Chiral freeform tile: smooth random field + signed angular warp, then threshold + thickening.
    """
    S = tile_px
    cy = cx = (S - 1) / 2.0

    smooth_sigma = float(params["smooth_sigma"])
    warp_strength = float(params["warp_strength"])
    thresh = float(params["thresh"])
    thick = int(params["thick"])

    noise = rng.normal(size=(S, S)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=smooth_sigma, mode="wrap")
    noise = (noise - noise.mean()) / (noise.std() + 1e-6)

    y, x = np.indices((S, S), dtype=np.float32)
    dy = y - cy
    dx = x - cx
    r = np.sqrt(dx * dx + dy * dy) / (S / 2.0 + 1e-6)
    theta = np.arctan2(dy, dx)
    theta2 = theta + float(handedness) * warp_strength * (r ** 2)

    yw = cy + r * (S / 2.0) * np.sin(theta2)
    xw = cx + r * (S / 2.0) * np.cos(theta2)

    coords = np.vstack([yw.ravel() % S, xw.ravel() % S])
    warped = map_coordinates(noise, coords, order=1, mode="wrap").reshape(S, S)

    img = warped > thresh
    img = ndimage.binary_opening(img, structure=_disk(1))
    img = ndimage.binary_closing(img, structure=_disk(1))
    if thick > 1:
        img = ndimage.binary_dilation(img, structure=_disk(thick / 2.0))

    # anchor + arms to promote connectivity across tiles
    rr, cc = draw.disk((cy, cx), radius=max(2, thick), shape=img.shape)
    img[rr, cc] = True
    arms = [
        ((cy, cx), (cy, 0)),
        ((cy, cx), (cy, S - 1)),
        ((cy, cx), (0, cx)),
        ((cy, cx), (S - 1, cx)),
    ]
    img |= rasterize_segments((S, S), arms, thickness=max(1, thick - 1))

    return img.astype(np.float32)


def render_family(tile_px: int, family: str, handedness: int, params: Dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    if family == "tetrachiral":
        return render_tetrachiral(tile_px, handedness, params)
    if family == "rotated_inclusion":
        return render_rotated_inclusion(tile_px, handedness, params)
    if family == "blocky":
        return render_blocky_chiral(tile_px, handedness, params)
    if family == "hierarchical":
        return render_hierarchical(tile_px, handedness, params)
    if family == "freeform":
        return render_freeform_warp(tile_px, handedness, params, rng)
    raise ValueError(family)


# -----------------------------
# Tile library
# -----------------------------

@dataclass
class TilePrototype:
    family: str
    handedness: int
    params: Optional[Dict[str, Any]]
    proto_id: str


@dataclass
class TileLibrary:
    mode: str = "precomputed"      # 'precomputed' or 'procedural'
    param_bins_per_family: int = 6
    seed: int = 0
    prototypes: List[TilePrototype] = field(default_factory=list)
    family_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    handedness: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    cache: Dict[Tuple[int, int], np.ndarray] = field(default_factory=dict)  # (tile_px, state_idx) -> mask

    def build(self, tile_px: int) -> None:
        rng = np.random.default_rng(self.seed)
        protos: List[TilePrototype] = []

        if self.mode == "precomputed":
            for fam in FAMILIES:
                for b in range(self.param_bins_per_family):
                    params = sample_params_for_family(fam, rng, tile_px)
                    for hand in (-1, +1):
                        pid = f"{fam[:3]}_b{b}_h{'L' if hand < 0 else 'R'}"
                        protos.append(TilePrototype(family=fam, handedness=hand, params=params, proto_id=pid))

        elif self.mode == "procedural":
            for fam in FAMILIES:
                for hand in (-1, +1):
                    pid = f"{fam[:3]}_proc_h{'L' if hand < 0 else 'R'}"
                    protos.append(TilePrototype(family=fam, handedness=hand, params=None, proto_id=pid))
        else:
            raise ValueError(self.mode)

        self.prototypes = protos
        self.family_ids = np.array([FAMILIES.index(p.family) for p in protos], dtype=np.int32)
        self.handedness = np.array([p.handedness for p in protos], dtype=np.int8)
        self.cache.clear()

    def render(self, state_idx: int, tile_px: int, rng: np.random.Generator) -> np.ndarray:
        key = (tile_px, state_idx)
        if self.mode == "precomputed" and key in self.cache:
            return self.cache[key]

        proto = self.prototypes[state_idx]
        if proto.params is None:
            params = sample_params_for_family(proto.family, rng, tile_px)
        else:
            params = proto.params

        img = render_family(tile_px, proto.family, proto.handedness, params, rng)

        if self.mode == "precomputed":
            self.cache[key] = img
        return img


# -----------------------------
# CA rules
# -----------------------------

def neighbor_offsets(neighborhood: str = "moore", radius: int = 1) -> List[Tuple[int, int]]:
    offs: List[Tuple[int, int]] = []
    if neighborhood == "moore":
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dy == 0 and dx == 0:
                    continue
                offs.append((dy, dx))
    elif neighborhood == "von_neumann":
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if abs(dx) + abs(dy) == 0:
                    continue
                if abs(dx) + abs(dy) <= radius:
                    offs.append((dy, dx))
    else:
        raise ValueError(neighborhood)
    return offs


def compute_neighbor_counts(states: np.ndarray, n_states: int, offsets: List[Tuple[int, int]]) -> np.ndarray:
    """Neighbor counts: (H*W, n_states)."""
    H, W = states.shape
    flat_idx = np.arange(H * W, dtype=np.int32)
    counts = np.zeros((H * W, n_states), dtype=np.int16)
    for dy, dx in offsets:
        shifted = np.roll(np.roll(states, dy, axis=0), dx, axis=1).ravel()
        np.add.at(counts, (flat_idx, shifted), 1)
    return counts


@dataclass
class CARule:
    rule_family: str = "weighted_totalistic"  # 'weighted_totalistic', 'cyclic', 'voter'
    neighborhood: str = "moore"
    radius: int = 1
    w_state: float = 1.0
    w_family: float = 0.35
    w_hand: float = 0.25
    w_keep: float = 0.5
    temperature: float = 0.4
    noise: float = 0.2
    hand_bias: float = 0.25
    bias_family: Optional[np.ndarray] = None
    bias_state: Optional[np.ndarray] = None
    cyclic_threshold: int = 3
    allowed_states: Optional[np.ndarray] = None  # boolean mask shape (n_states,)


def step_ca(states: np.ndarray,
            rule: CARule,
            state_family: np.ndarray,
            state_hand: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    H, W = states.shape
    n_states = len(state_family)
    n_families = int(state_family.max()) + 1
    offs = neighbor_offsets(rule.neighborhood, rule.radius)

    if rule.rule_family == "voter":
        dy, dx = offs[int(rng.integers(0, len(offs)))]
        new = np.roll(np.roll(states, dy, axis=0), dx, axis=1)
        keep_mask = rng.uniform(size=(H, W)) < 0.3
        new[keep_mask] = states[keep_mask]
        return new

    counts = compute_neighbor_counts(states, n_states, offs)

    if rule.rule_family == "cyclic":
        next_state = (states + 1) % n_states
        next_counts = counts[np.arange(H * W), next_state.ravel()]
        advance = next_counts >= rule.cyclic_threshold
        if rule.allowed_states is not None:
            advance = advance & rule.allowed_states[next_state.ravel()]
        new = states.ravel().copy()
        new[advance] = next_state.ravel()[advance]
        return new.reshape(H, W)

    # weighted_totalistic
    bias_state = np.zeros(n_states, dtype=np.float32) if rule.bias_state is None else rule.bias_state.astype(np.float32)
    bias_family = np.zeros(n_families, dtype=np.float32) if rule.bias_family is None else rule.bias_family.astype(np.float32)

    counts_family = np.zeros((H * W, n_families), dtype=np.int16)
    for f in range(n_families):
        counts_family[:, f] = counts[:, state_family == f].sum(axis=1)

    hand_bins = np.where(state_hand > 0, 1, 0)
    counts_hand = np.zeros((H * W, 2), dtype=np.int16)
    counts_hand[:, 0] = counts[:, hand_bins == 0].sum(axis=1)
    counts_hand[:, 1] = counts[:, hand_bins == 1].sum(axis=1)

    fam_of_state = state_family
    hand_of_state = state_hand
    hand_bin_state = np.where(hand_of_state > 0, 1, 0)

    scores = np.empty((H * W, n_states), dtype=np.float32)
    scores[:] = rule.w_state * counts.astype(np.float32)
    scores += rule.w_family * counts_family[:, fam_of_state].astype(np.float32)
    scores += rule.w_hand * counts_hand[:, hand_bin_state].astype(np.float32)
    scores += bias_state[None, :]
    scores += bias_family[fam_of_state][None, :]
    scores += (rule.hand_bias * hand_of_state.astype(np.float32))[None, :]

    if rule.w_keep != 0.0:
        scores[np.arange(H * W), states.ravel()] += rule.w_keep
    if rule.noise > 0.0:
        scores += rule.noise * rng.normal(size=scores.shape).astype(np.float32)
    if rule.allowed_states is not None:
        scores[:, ~rule.allowed_states] = -1e9

    if rule.temperature <= 0.0:
        new = np.argmax(scores, axis=1).astype(np.int32)
        return new.reshape(H, W)

    smax = scores / rule.temperature
    smax = smax - smax.max(axis=1, keepdims=True)
    probs = np.exp(smax)
    probs /= probs.sum(axis=1, keepdims=True)
    cum = np.cumsum(probs, axis=1)
    r = rng.random(size=cum.shape[0])[:, None]
    new = (cum < r).sum(axis=1).astype(np.int32)
    return new.reshape(H, W)


# -----------------------------
# Periodic connected components (fast torus labeling)
# -----------------------------

class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int8)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def periodic_connected_components(mask: np.ndarray, connectivity: int = 4) -> Tuple[np.ndarray, int]:
    """
    Fast connected components on a torus:
      - label planar components
      - union components touching opposite boundaries (and diagonals for 8-connectivity)
    Returns labels (int32, -1 for void) and component count.
    """
    m = mask.astype(bool)
    H, W = m.shape
    labels = measure.label(m, connectivity=1 if connectivity == 4 else 2)
    n = int(labels.max())
    if n == 0:
        return -np.ones_like(labels, dtype=np.int32), 0

    uf = UnionFind(n + 1)

    # Left-right wrap
    ys = np.where(m[:, 0] & m[:, W - 1])[0]
    for y in ys:
        uf.union(int(labels[y, 0]), int(labels[y, W - 1]))

    # Top-bottom wrap
    xs = np.where(m[0, :] & m[H - 1, :])[0]
    for x in xs:
        uf.union(int(labels[0, x]), int(labels[H - 1, x]))

    if connectivity == 8:
        # Diagonal wraps
        for y in range(H):
            if m[y, 0]:
                if m[(y - 1) % H, W - 1]:
                    uf.union(int(labels[y, 0]), int(labels[(y - 1) % H, W - 1]))
                if m[(y + 1) % H, W - 1]:
                    uf.union(int(labels[y, 0]), int(labels[(y + 1) % H, W - 1]))
        for x in range(W):
            if m[0, x]:
                if m[H - 1, (x - 1) % W]:
                    uf.union(int(labels[0, x]), int(labels[H - 1, (x - 1) % W]))
                if m[H - 1, (x + 1) % W]:
                    uf.union(int(labels[0, x]), int(labels[H - 1, (x + 1) % W]))

    roots = np.array([uf.find(i) for i in range(n + 1)], dtype=np.int32)
    root_to_new: Dict[int, int] = {}
    new_id = 0
    for i in range(1, n + 1):
        r = int(roots[i])
        if r not in root_to_new:
            root_to_new[r] = new_id
            new_id += 1

    lut = np.full(n + 1, -1, dtype=np.int32)
    for i in range(1, n + 1):
        lut[i] = root_to_new[int(roots[i])]
    mapped = lut[labels]
    return mapped, new_id


# -----------------------------
# Periodic morphology + connectivity repair
# -----------------------------

def periodic_morphology(mask: np.ndarray, op: str, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    se = _disk(radius)
    pad = se.shape[0] // 2
    padded = np.pad(mask.astype(bool), pad, mode="wrap")
    if op == "dilate":
        out = ndimage.binary_dilation(padded, structure=se)
    elif op == "erode":
        out = ndimage.binary_erosion(padded, structure=se)
    elif op == "open":
        out = ndimage.binary_opening(padded, structure=se)
    elif op == "close":
        out = ndimage.binary_closing(padded, structure=se)
    else:
        raise ValueError(op)
    return out[pad : pad + mask.shape[0], pad : pad + mask.shape[1]]


def draw_thick_line_periodic(mask: np.ndarray,
                             p0: Tuple[int, int],
                             p1_ext: Tuple[int, int],
                             thickness: int) -> None:
    """Draw periodic thick line between p0 (in-domain) and p1_ext (possibly outside), in-place."""
    H, W = mask.shape
    y0, x0 = p0
    y1, x1 = p1_ext
    rr, cc = draw.line(int(y0), int(x0), int(y1), int(x1))
    rr = np.mod(rr, H)
    cc = np.mod(cc, W)
    mask[rr, cc] = True
    if thickness > 1:
        mask[:] = periodic_morphology(mask, "dilate", max(1, thickness // 2))


def connect_components_periodic(mask: np.ndarray,
                               connectivity: int,
                               thickness: int,
                               max_bridges: int,
                               rng: np.random.Generator) -> Tuple[np.ndarray, bool, int]:
    """
    Ensure single connected component on torus by adding bridges (material).
    """
    labels, ncomp = periodic_connected_components(mask, connectivity)
    n_before = int(ncomp)
    if ncomp <= 1:
        return mask, True, n_before

    H, W = mask.shape
    shifts_y = [-H, 0, H]
    shifts_x = [-W, 0, W]

    bridges = 0
    while ncomp > 1 and bridges < max_bridges:
        labels, ncomp = periodic_connected_components(mask, connectivity)
        if ncomp <= 1:
            break

        # list components by size
        comps = []
        for c in range(ncomp):
            ys, xs = np.nonzero(labels == c)
            comps.append((c, int(len(ys)), ys, xs))
        comps.sort(key=lambda t: t[1], reverse=True)

        base = comps[0]
        others = comps[1:]
        base_coords = np.stack([base[2], base[3]], axis=1)

        tiled = []
        for dy in shifts_y:
            for dx in shifts_x:
                tiled.append(base_coords + np.array([dy, dx], dtype=np.int32))
        tiled = np.concatenate(tiled, axis=0)
        kdt = cKDTree(tiled)

        # pick an "other" component (weighted by size)
        w = np.array([t[1] for t in others], dtype=np.float64)
        w = w / (w.sum() + 1e-12)
        pick = int(rng.choice(len(others), p=w))
        c, _, ys, xs = others[pick]
        coords = np.stack([ys, xs], axis=1)

        # sample subset to speed
        if coords.shape[0] > 800:
            sel = rng.choice(coords.shape[0], size=800, replace=False)
            coords_s = coords[sel]
        else:
            coords_s = coords

        d, idx = kdt.query(coords_s, k=1)
        j = int(np.argmin(d))
        p_other = coords_s[j]
        p_base_ext = tiled[int(idx[j])]

        draw_thick_line_periodic(mask,
                                 (int(p_other[0]), int(p_other[1])),
                                 (int(p_base_ext[0]), int(p_base_ext[1])),
                                 thickness=thickness)
        bridges += 1

    _, n_final = periodic_connected_components(mask, connectivity)
    return mask, (n_final == 1), n_before


# -----------------------------
# Diversity descriptors (used for optional dedupe)
# -----------------------------

def radial_profile(arr: np.ndarray, nbins: int = 8) -> np.ndarray:
    H, W = arr.shape
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    y, x = np.indices((H, W))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.float32)
    bins = np.linspace(0, r.max(), nbins + 1)
    prof = np.zeros(nbins, dtype=np.float32)
    for i in range(nbins):
        m = (r >= bins[i]) & (r < bins[i + 1])
        prof[i] = float(arr[m].mean()) if m.any() else 0.0
    return prof


def two_point_corr_features(mask: np.ndarray, nbins: int = 8) -> np.ndarray:
    I = mask.astype(np.float32)
    F = fft2(I)
    ac = np.real(ifft2(F * np.conj(F))) / float(I.size)
    ac = np.fft.fftshift(ac)
    return radial_profile(ac, nbins=nbins)


def fourier_spectrum_features(mask: np.ndarray, nbins: int = 8) -> np.ndarray:
    I = mask.astype(np.float32) - float(mask.mean())
    P = np.abs(fft2(I)) ** 2
    P = np.fft.fftshift(P)
    P = P / (P.mean() + 1e-8)
    return np.log1p(radial_profile(P, nbins=nbins))


def skeleton_stats(mask: np.ndarray) -> Tuple[float, float, float]:
    sk = morphology.skeletonize(mask.astype(bool))
    if sk.sum() == 0:
        return 0.0, 0.0, 0.0
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.int32)
    nbr = ndimage.convolve(sk.astype(np.int32), kernel, mode="wrap")
    deg = nbr[sk] - 10
    endpoints = float(np.sum(deg == 1))
    junctions = float(np.sum(deg >= 3))
    length = float(sk.sum())
    return length, junctions, endpoints


def holes_count_planar(mask: np.ndarray) -> int:
    """Planar proxy hole count: void components not touching boundary."""
    void = ~mask.astype(bool)
    lab = measure.label(void, connectivity=1)
    n = int(lab.max())
    holes = 0
    H, W = mask.shape
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if ys.size == 0:
            continue
        if (ys == 0).any() or (ys == H - 1).any() or (xs == 0).any() or (xs == W - 1).any():
            continue
        holes += 1
    return holes


def descriptor_vector(mask: np.ndarray) -> np.ndarray:
    vf = float(mask.mean())
    chi = float(chirality_score(mask))
    _, ncomp = periodic_connected_components(mask, connectivity=4)
    holes = float(holes_count_planar(mask))
    sk_len, sk_junc, sk_end = skeleton_stats(mask)
    s2 = two_point_corr_features(mask, nbins=8)
    fs = fourier_spectrum_features(mask, nbins=8)
    return np.concatenate(
        [np.array([vf, chi, float(ncomp), holes, sk_len, sk_junc, sk_end], dtype=np.float32),
         s2.astype(np.float32),
         fs.astype(np.float32)]
    )


# -----------------------------
# Generation config + pipeline
# -----------------------------

@dataclass
class GeneratorConfig:
    N: int = 100                    # number of BASE samples (mirror doubles outputs if enabled)
    image_px: int = 128
    k: int = 8
    upsample: int = 8
    tile_mode: str = "precomputed"  # precomputed | procedural
    ca_steps: int = 10
    ca_radius: int = 1
    ca_neighborhood: str = "moore"
    ca_rule_family: str = "weighted_totalistic"
    global_handedness: int = 1      # +1, -1, or 0 (random per sample)
    temperature: float = 0.4
    noise: float = 0.2
    w_state: float = 1.0
    w_family: float = 0.35
    w_hand: float = 0.25
    w_keep: float = 0.5

    min_feature_px: int = 3
    connectivity: int = 4
    max_attempts_per_sample: int = 60

    close_radius: int = 2
    open_radius: int = 0
    bridge_thickness: int = 3
    max_bridges: int = 30

    dedupe: bool = True
    descriptor_distance_thresh: float = 0.08
    hash_pool: int = 32

    include_mirror_pairs: bool = True
    seed: int = 0
    output_dir: str = "out_chiral_ca"


def make_ca_rule(cfg: GeneratorConfig,
                 lib: TileLibrary,
                 rng: np.random.Generator,
                 allowed_states: Optional[np.ndarray]) -> CARule:
    n_states = len(lib.prototypes)
    bias_state = rng.normal(scale=0.05, size=n_states).astype(np.float32)
    bias_family = rng.normal(scale=0.05, size=len(FAMILIES)).astype(np.float32)
    hand_bias = float(cfg.global_handedness if cfg.global_handedness != 0 else 1) * float(rng.uniform(0.1, 0.45))

    return CARule(
        rule_family=cfg.ca_rule_family,
        neighborhood=cfg.ca_neighborhood,
        radius=cfg.ca_radius,
        w_state=cfg.w_state,
        w_family=cfg.w_family,
        w_hand=cfg.w_hand,
        w_keep=cfg.w_keep,
        temperature=cfg.temperature,
        noise=cfg.noise,
        hand_bias=hand_bias,
        bias_family=bias_family,
        bias_state=bias_state,
        allowed_states=allowed_states,
    )


def rasterize_tile_grid(states: np.ndarray, lib: TileLibrary,
                        k: int, upsample: int, img_px: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Render CA state grid into a high-res float image (img_px*upsample)."""
    Hc, Wc = states.shape
    tile_hi = k * upsample
    out_hi = img_px * upsample
    canvas = np.zeros((out_hi, out_hi), dtype=np.float32)

    for i in range(Hc):
        for j in range(Wc):
            s = int(states[i, j])
            tile = lib.render(s, tile_hi, rng)
            y0 = i * tile_hi
            x0 = j * tile_hi
            canvas[y0 : y0 + tile_hi, x0 : x0 + tile_hi] = np.maximum(
                canvas[y0 : y0 + tile_hi, x0 : x0 + tile_hi], tile
            )
    return canvas


def postprocess_mask(mask: np.ndarray, cfg: GeneratorConfig, rng: np.random.Generator) -> Tuple[np.ndarray, Dict[str, Any]]:
    info: Dict[str, Any] = {}

    m = mask.astype(bool)

    if cfg.close_radius > 0:
        m = periodic_morphology(m, "close", cfg.close_radius)
    if cfg.open_radius > 0:
        m = periodic_morphology(m, "open", cfg.open_radius)

    # planar small-object cleanup (after periodic morphology)
    m = morphology.remove_small_objects(m, min_size=cfg.min_feature_px * cfg.min_feature_px, connectivity=1)
    m = morphology.remove_small_holes(m, area_threshold=cfg.min_feature_px * cfg.min_feature_px)

    # connectivity repair on torus
    repaired = m.copy()
    repaired, ok, n_before = connect_components_periodic(
        repaired,
        connectivity=cfg.connectivity,
        thickness=cfg.bridge_thickness,
        max_bridges=cfg.max_bridges,
        rng=rng,
    )
    info["components_before_repair"] = int(n_before)
    info["repair_success"] = bool(ok)

    # enforce min feature width via minimal dilation if needed
    w = min_feature_width_px(repaired)
    if w < cfg.min_feature_px:
        need = int(np.ceil((cfg.min_feature_px - w) / 2.0))
        if need > 0:
            repaired = periodic_morphology(repaired, "dilate", need)

    info["min_feature_width_px_est"] = float(min_feature_width_px(repaired))
    return repaired.astype(bool), info


def generate_one(cfg: GeneratorConfig,
                 lib: TileLibrary,
                 rng: np.random.Generator,
                 sample_id: str,
                 dedupe_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    img_px = cfg.image_px
    k = cfg.k

    target_hand = cfg.global_handedness
    if target_hand == 0:
        target_hand = int(rng.choice([-1, +1]))

    if k == 1:
        # Pixel-level CA (simple binary CA via iterative neighborhood smoothing)
        # Start random and repeatedly apply majority-like updates.
        p = float(rng.uniform(0.35, 0.65))
        m = rng.random((img_px, img_px)) < p
        offs = neighbor_offsets(cfg.ca_neighborhood, radius=1)
        for _ in range(cfg.ca_steps):
            cnt = np.zeros_like(m, dtype=np.int16)
            for dy, dx in offs:
                cnt += np.roll(np.roll(m, dy, axis=0), dx, axis=1).astype(np.int16)
            # thresholds chosen to encourage blob-like connected structures
            born = (~m) & (cnt >= 5)
            surv = m & (cnt >= 3)
            m = born | surv
        mask0 = m
        states = None
        rule_meta = {"binary_CA": True, "target_hand": int(target_hand)}

    else:
        Hc = img_px // k

        allowed = (lib.handedness == target_hand)
        if not bool(allowed.any()):
            allowed = np.ones_like(lib.handedness, dtype=bool)
        allowed_idx = np.nonzero(allowed)[0]

        states = rng.choice(allowed_idx, size=(Hc, Hc), replace=True).astype(np.int32)

        rule = make_ca_rule(cfg, lib, rng, allowed_states=allowed)

        # symmetry-breaking sweep seed
        if rng.random() < 0.35:
            states = (states + np.fliplr(states)) // 2
            n_mut = max(1, int(0.02 * Hc * Hc))
            idxs = rng.choice(Hc * Hc, size=n_mut, replace=False)
            states.ravel()[idxs] = rng.choice(allowed_idx, size=n_mut)

        for _ in range(cfg.ca_steps):
            states = step_ca(states, rule, lib.family_ids, lib.handedness, rng)

        img_hi = rasterize_tile_grid(states, lib, k=k, upsample=cfg.upsample, img_px=img_px, rng=rng)
        img = downsample_antialias(img_hi, img_px)
        mask0 = binarize(img, thresh=0.5)

        rule_meta = {
            "rule_family": rule.rule_family,
            "neighborhood": rule.neighborhood,
            "radius": rule.radius,
            "w_state": rule.w_state,
            "w_family": rule.w_family,
            "w_hand": rule.w_hand,
            "w_keep": rule.w_keep,
            "temperature": rule.temperature,
            "noise": rule.noise,
            "hand_bias": rule.hand_bias,
            "target_hand": int(target_hand),
        }

    mask, post_info = postprocess_mask(mask0, cfg, rng)

    _, ncomp = periodic_connected_components(mask, connectivity=cfg.connectivity)
    if int(ncomp) != 1:
        return None
    if min_feature_width_px(mask) < cfg.min_feature_px:
        return None

    # optional dedupe
    if cfg.dedupe:
        h = mask_hash(mask, pool=cfg.hash_pool)
        if h in dedupe_state["hashes"]:
            return None
        vec = descriptor_vector(mask)
        if len(dedupe_state["descriptors"]) > 0:
            D = np.vstack(dedupe_state["descriptors"])
            mu = D.mean(axis=0)
            sig = D.std(axis=0) + 1e-6
            z = (vec - mu) / sig
            Z = (D - mu) / sig
            dist = float(np.linalg.norm(Z - z, axis=1).min())
            if dist < cfg.descriptor_distance_thresh:
                return None
        dedupe_state["hashes"].add(h)
        dedupe_state["descriptors"].append(vec)

    vf = float(mask.mean())
    chi = float(chirality_score(mask))

    family_counts: Dict[str, int] = {}
    if states is None:
        family_counts = {"pixel_CA": int(mask.sum())}
    else:
        fam_grid = lib.family_ids[states]
        for i, fam in enumerate(FAMILIES):
            family_counts[fam] = int(np.sum(fam_grid == i))

    meta: Dict[str, Any] = {
        "id": sample_id,
        "handedness": int(target_hand),
        "image_px": int(cfg.image_px),
        "k": int(cfg.k),
        "tile_mode": str(cfg.tile_mode),
        "volume_fraction": vf,
        "chirality_score": chi,
        "connectivity_components": int(ncomp),
        "min_feature_width_px_est": float(min_feature_width_px(mask)),
        "postprocess": post_info,
        "rule": rule_meta,
        "family_counts": family_counts,
        "timestamp_utc": float(time.time()),
    }

    sdf = signed_distance_field(mask)
    return {"mask": mask.astype(np.uint8), "sdf": sdf, "meta": meta}


def save_sample(out_dir: str, sample: Dict[str, Any]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    masks_dir = os.path.join(out_dir, "masks")
    sdf_dir = os.path.join(out_dir, "sdf")
    meta_dir = os.path.join(out_dir, "meta")
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    sid = sample["meta"]["id"]
    mask = sample["mask"].astype(np.uint8)
    sdf = sample["sdf"].astype(np.float32)

    Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(os.path.join(masks_dir, f"{sid}.png"))
    np.save(os.path.join(masks_dir, f"{sid}.npy"), mask)
    np.save(os.path.join(sdf_dir, f"{sid}.npy"), sdf)
    with open(os.path.join(meta_dir, f"{sid}.json"), "w", encoding="utf-8") as f:
        json.dump(sample["meta"], f, indent=2)


def generate_dataset(cfg: GeneratorConfig) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)

    # Build tile library at a representative tile size for precomputed params.
    # For caching, tiles are rendered at the actual rendering size anyway.
    rep_tile_px = max(16, cfg.k * cfg.upsample) if cfg.k > 1 else 16
    lib = TileLibrary(mode=cfg.tile_mode, param_bins_per_family=6, seed=cfg.seed)
    lib.build(tile_px=rep_tile_px)

    dedupe_state: Dict[str, Any] = {"hashes": set(), "descriptors": []}
    metas: List[Dict[str, Any]] = []

    base_count = 0
    attempts = 0
    max_attempts = cfg.N * cfg.max_attempts_per_sample

    while base_count < cfg.N and attempts < max_attempts:
        attempts += 1
        sid = f"sample_{base_count:06d}"
        res = generate_one(cfg, lib, rng, sid, dedupe_state)
        if res is None:
            continue

        save_sample(cfg.output_dir, res)
        metas.append(res["meta"])
        base_count += 1

        if cfg.include_mirror_pairs:
            sid_m = f"{sid}_mir"
            mask_m = np.fliplr(res["mask"].astype(bool)).astype(np.uint8)
            sdf_m = np.fliplr(res["sdf"]).astype(np.float32)

            meta_m = dict(res["meta"])
            meta_m["id"] = sid_m
            meta_m["mirror_of"] = sid
            meta_m["handedness"] = -int(res["meta"]["handedness"])
            meta_m["chirality_score"] = float(chirality_score(mask_m.astype(bool)))

            save_sample(cfg.output_dir, {"mask": mask_m, "sdf": sdf_m, "meta": meta_m})
            metas.append(meta_m)

    with open(os.path.join(cfg.output_dir, "metadata.jsonl"), "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")

    if base_count < cfg.N:
        print(f"[WARN] Only generated {base_count} base samples out of requested {cfg.N} after {attempts} attempts.")
    else:
        print(f"[OK] Generated {base_count} base samples ({len(metas)} total including mirrors) in {cfg.output_dir}.")


# -----------------------------
# Example batch runner
# -----------------------------

def run_examples(out_root: str) -> None:
    """
    Generate 100 base samples in both modes:
      - precomputed tiles (k=8)
      - procedural tiles (k=4)
      - pixel CA (k=1)
    Mirrors are enabled by default, so outputs are doubled.
    """
    cases = [
        ("precomputed_k8", GeneratorConfig(N=100, k=8, tile_mode="precomputed", output_dir=os.path.join(out_root, "precomputed_k8"),
                                           close_radius=2, bridge_thickness=3, max_bridges=30, seed=1)),
        ("procedural_k4", GeneratorConfig(N=100, k=4, tile_mode="procedural", output_dir=os.path.join(out_root, "procedural_k4"),
                                          close_radius=4, bridge_thickness=5, max_bridges=60, seed=2,
                                          dedupe=True, descriptor_distance_thresh=0.06)),
        ("pixel_k1", GeneratorConfig(N=100, k=1, tile_mode="procedural", output_dir=os.path.join(out_root, "pixel_k1"),
                                     close_radius=3, bridge_thickness=3, max_bridges=40, seed=3)),
    ]
    for name, cfg in cases:
        print(f"\n=== Running example case: {name} ===")
        generate_dataset(cfg)


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="out_chiral_ca", help="Output directory")
    p.add_argument("--N", type=int, default=100, help="Number of base samples (mirror doubles total if enabled)")
    p.add_argument("--k", type=int, default=16, choices=[1, 2, 4, 8, 16], help="Coarse-graining factor")
    p.add_argument("--tile-mode", type=str, default="precomputed", choices=["precomputed", "procedural"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-mirror", action="store_true", help="Disable mirror partner generation")
    p.add_argument("--no-dedupe", action="store_true", help="Disable descriptor-based dedupe")
    p.add_argument("--run-examples", action="store_true", help="Generate 100 samples in each example mode")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_examples:
        run_examples(args.out)
        return

    cfg = GeneratorConfig(
        N=args.N,
        k=args.k,
        tile_mode=args.tile_mode,
        seed=args.seed,
        output_dir=args.out,
        include_mirror_pairs=not args.no_mirror,
        dedupe=not args.no_dedupe,
    )
    generate_dataset(cfg)


if __name__ == "__main__":
    main()
