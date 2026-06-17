"""Plot mesh convergence of the outside-cloak loss for the continuous C_eff model."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: (mesh_resolution, n_outside, dist_outside %)
data = np.array([
    [80,  18851,  4.22],
    [120, 33481,  2.35],
    [160, 52218,  1.57],
    [200, 75213,  1.30],
    [250, 109510, 1.06],
])

mesh_res = data[:, 0]
n_outside = data[:, 1]
distortion = data[:, 2]

# Fit: D(n) = D_inf + A * n^(-p)  (power-law convergence in number of nodes)
def power_law(n, D_inf, A, p):
    return D_inf + A * n**(-p)

popt, pcov = curve_fit(power_law, n_outside, distortion,
                       p0=[0.5, 1000.0, 0.5],
                       bounds=([0, 0, 0.05], [2.0, 1e8, 3.0]))
D_inf, A, p = popt
perr = np.sqrt(np.diag(pcov))

print(f"Fit: D(n) = {D_inf:.3f} + {A:.1f} * n^(-{p:.2f})")
print(f"  D_inf = {D_inf:.3f} +/- {perr[0]:.3f} %")
print(f"  A     = {A:.1f} +/- {perr[1]:.1f}")
print(f"  p     = {p:.2f} +/- {perr[2]:.2f}")

# Estimate n_outside for finer meshes (quadratic in N)
# From data: n ~ 1.76 * N^2  (fit the relationship)
coeff = np.polyfit(mesh_res**2, n_outside, 1)
print(f"\nn_outside ~ {coeff[0]:.3f} * N^2 + {coeff[1]:.0f}")

for N in [300, 400, 500, 750, 1000]:
    n_est = coeff[0] * N**2 + coeff[1]
    pred = power_law(n_est, *popt)
    print(f"  N={N:4d}: n_outside ~ {n_est:.0f}, D = {pred:.3f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.5, 4))

# Smooth fit curve
n_smooth = np.linspace(10000, 1900000, 1000)
D_smooth = power_law(n_smooth, *popt)

ax.plot(n_smooth / 1000, D_smooth, '-', color='C0', linewidth=1.5,
        label=f'Fit: $\\mathcal{{D}}_\\infty + A\\,n^{{-p}}$\n'
              f'($\\mathcal{{D}}_\\infty$={D_inf:.2f}%, $p$={p:.2f})')

# Data points
ax.plot(n_outside / 1000, distortion, 'o', color='C0', markersize=7, zorder=5,
        label='Measured')

# Extrapolation region
n_max_data = n_outside[-1]
ax.axvspan(n_max_data / 1000, 1900, alpha=0.08, color='grey')
ax.text(900, 3.5, 'extrapolation', ha='center', fontsize=9, color='0.4', style='italic')

# Mark key predictions
for N, marker in [(500, 's'), (1000, 'D')]:
    n_est = coeff[0] * N**2 + coeff[1]
    pred = power_law(n_est, *popt)
    ax.plot(n_est / 1000, pred, marker, color='C3', markersize=8, zorder=5,
            label=f'Predicted $N$={N}: {pred:.2f}%')

# Asymptote
ax.axhline(D_inf, color='C2', linestyle='--', linewidth=1, alpha=0.7,
           label=f'Asymptote $\\mathcal{{D}}_\\infty$ = {D_inf:.2f}%')

ax.set_xlabel('Number of outside-cloak nodes ($\\times 10^3$)', fontsize=11)
ax.set_ylabel('Outside-cloak distortion $\\mathcal{D}_{\\mathrm{out}}$ [%]', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(0, 1900)
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)
fig.tight_layout()

outpath = 'docs/figures/continuous_ceff_outside_convergence.pdf'
fig.savefig(outpath, bbox_inches='tight')
print(f'\nSaved to {outpath}')
plt.show()
