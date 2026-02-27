"""
Plot displacement fields from saved simulation results.

Usage:  python plot_results.py [path/to/results.npz]
        Defaults to output/results.npz if no argument given.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
import matplotlib.colors as mcolors


class SigmoidNorm(mcolors.Normalize):
    """Normalize using a sigmoid, stretching extremes and compressing the middle.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness : float
        Controls how sharp the transition is.  Higher = more contrast at
        the extremes, flatter middle.  Typical range: 5–30.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness=10.0, vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness = steepness
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # normalise data to [0, 1] linearly first
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # sigmoid centred on mid_norm
        s = 1.0 / (1.0 + np.exp(-self.steepness * (x - mid_norm)))
        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


class AsymSigmoidNorm(mcolors.Normalize):
    """Normalize using an asymmetric sigmoid with different steepness on each side.

    Parameters
    ----------
    mid : float
        Data value mapped to 0.5 in colour-space (the inflection point).
    steepness_left : float
        Steepness for values below *mid*.  Higher = sharper transition.
    steepness_right : float
        Steepness for values above *mid*.  Higher = sharper transition.
    vmin, vmax : float
        Data range (as usual for Normalize).
    """

    def __init__(self, mid, steepness_left=10.0, steepness_right=10.0,
                 vmin=None, vmax=None, clip=False):
        self.mid = mid
        self.steepness_left = steepness_left
        self.steepness_right = steepness_right
        super().__init__(vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x = np.ma.array(value, dtype=float)
        self.autoscale_None(x)
        x = (x - self.vmin) / (self.vmax - self.vmin)
        mid_norm = (self.mid - self.vmin) / (self.vmax - self.vmin)

        # pick steepness per element depending on which side of mid_norm
        k = np.where(x <= mid_norm, self.steepness_left, self.steepness_right)
        s = 1.0 / (1.0 + np.exp(-k * (x - mid_norm)))

        # rescale so that s(0)→0 and s(1)→1 exactly
        s0 = 1.0 / (1.0 + np.exp(-self.steepness_left * (0.0 - mid_norm)))
        s1 = 1.0 / (1.0 + np.exp(-self.steepness_right * (1.0 - mid_norm)))
        result = (s - s0) / (s1 - s0)
        return np.ma.array(np.clip(result, 0, 1))


from datetime import datetime
import os

# ── Load data ────────────────────────────────────────────────────────
data_path = sys.argv[1] if len(sys.argv) > 1 else "output/results.npz"
d = np.load(data_path)

u       = d['u']           # (num_nodes, 4)
pts_x   = d['pts_x']
pts_y   = d['pts_y']
x_src   = float(d['x_src'])
y_top   = float(d['y_top'])
x_off   = float(d['x_off'])
y_off   = float(d['y_off'])
W       = float(d['W'])
H       = float(d['H'])
x_src_phys = float(d['x_src_phys'])
f_star  = float(d['f_star'])

# ── Derived quantities ───────────────────────────────────────────────
ux_R, uy_R, ux_I, uy_I = u[:, 0], u[:, 1], u[:, 2], u[:, 3]
mag = np.sqrt(ux_R**2 + uy_R**2 + ux_I**2 + uy_I**2)

# ── Cloak geometry (must match boundaries.py) ──────────────────────
a_cloak = 0.0774 * H                       # inner triangle depth
b_cloak = 3 * a_cloak                      # outer triangle depth
c_cloak = 0.309 * H / 2.0                  # half-width at surface
x_c     = x_off + W / 2.0                  # cloak centre (extended coords)

os.makedirs("output", exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# ── Full-domain plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
norm = AsymSigmoidNorm(mid=0.25*mag.max(), steepness_left=2, steepness_right=30,
                       vmin=mag.min(), vmax=mag.max())
tc = ax.tricontourf(pts_x, pts_y, mag, levels=100, cmap='RdBu_r', norm=norm)
ax.plot(x_src, y_top, 'r*', markersize=16, label='source')

ax.axvline(x_off,       color='cyan', ls='--', lw=0.7, label='PML interface')
ax.axvline(x_off + W,   color='cyan', ls='--', lw=0.7)
ax.axhline(y_off,       color='cyan', ls='--', lw=0.7)

# Cloak outer boundary: (x_c-c, y_top) → (x_c, y_top-b) → (x_c+c, y_top)
ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
        [y_top, y_top - b_cloak, y_top],
        ls='--', color='yellow', lw=1.2)
# Cloak inner boundary: (x_c-c, y_top) → (x_c, y_top-a) → (x_c+c, y_top)
ax.plot([x_c - c_cloak, x_c, x_c + c_cloak],
        [y_top, y_top - a_cloak, y_top],
        ls='--', color='yellow', lw=1.2)

fig.colorbar(tc, ax=ax, shrink=0.8, label='|u|')
ax.set_title(f"Symmetrized Triangular Cloak with Absorbing Layers  (f*={f_star})")
ax.set_xlabel("x")
ax.set_ylabel("y")
# ax.legend(loc='upper right', fontsize=8)
ax.set_aspect('equal')

fname_full = f"output/cloak_abslay_full_{stamp}.png"
fig.savefig(fname_full, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f"Full-domain plot saved → {fname_full}")

# ── Physical-domain-only plot ────────────────────────────────────────
phys_mask = ((pts_x >= x_off - 1e-8) & (pts_x <= x_off + W + 1e-8) &
             (pts_y >= y_off - 1e-8))

fig2, ax2 = plt.subplots(figsize=(13, 4))
mag_phys = mag[phys_mask]
# norm2 = SigmoidNorm(mid=0.2*mag_phys.max(), steepness=14,
#                      vmin=mag_phys.min(), vmax=1*mag_phys.max())
norm2 = AsymSigmoidNorm(mid=0.25*mag_phys.max(), steepness_left=2, steepness_right=30,
                       vmin=mag_phys.min(), vmax=mag_phys.max())

tc2 = ax2.tricontourf(pts_x[phys_mask] - x_off,
                       pts_y[phys_mask] - y_off,
                       mag_phys,
                       levels=100, cmap='RdBu_r', norm=norm2)
ax2.plot(x_src_phys, H, 'r*', markersize=16)

# Cloak boundaries in physical coords (centre at W/2, surface at y=H)
x_c_phys = W / 2.0
ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
         [H, H - b_cloak, H],
         ls='--', color='yellow', lw=1.2)
ax2.plot([x_c_phys - c_cloak, x_c_phys, x_c_phys + c_cloak],
         [H, H - a_cloak, H],
         ls='--', color='yellow', lw=1.2)
# ax2.legend(loc='upper right', fontsize=8)

fig2.colorbar(tc2, ax=ax2, shrink=0.8, label='|u|')
ax2.set_title(f"Physical domain only  (f*={f_star})")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_aspect('equal')

fname_crop = f"output/cloak_abslay_phys_{stamp}.png"
# fname_crop = f"output/cloak_abslay_phys.png"
fig2.savefig(fname_crop, dpi=200, bbox_inches='tight')
plt.close(fig2)
print(f"Physical-domain plot saved → {fname_crop}")
