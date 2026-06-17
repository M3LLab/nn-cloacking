"""Plot mesh convergence of the continuous C_eff model and extrapolate to finer meshes."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data: (mesh_resolution, right-boundary distortion %)
data = np.array([
    [80,  41.58],
    [120, 31.70],
    [160, 27.51],
    [200, 27.13],
    [250, 23.02],
])

mesh_res = data[:, 0]
distortion = data[:, 1]

# Characteristic mesh size h = 1/N (proportional to element size)
h = 1.0 / mesh_res

# Fit: D(h) = D_inf + A * h^p  (power-law convergence)
# In log space for (D - D_inf): log(D - D_inf) = log(A) + p * log(h)
# We fit all three parameters jointly.
def power_law(h, D_inf, A, p):
    return D_inf + A * h**p

# Initial guess: asymptote ~20%, power ~1
popt, pcov = curve_fit(power_law, h, distortion, p0=[15.0, 500.0, 1.0],
                       bounds=([0, 0, 0.1], [25, 1e6, 5.0]))
D_inf, A, p = popt
perr = np.sqrt(np.diag(pcov))

print(f"Fit: D(h) = {D_inf:.2f} + {A:.1f} * h^{p:.2f}")
print(f"  D_inf = {D_inf:.2f} +/- {perr[0]:.2f} %")
print(f"  A     = {A:.1f} +/- {perr[1]:.1f}")
print(f"  p     = {p:.2f} +/- {perr[2]:.2f}")

# Predictions
for N in [300, 400, 500, 750, 1000]:
    pred = power_law(1.0/N, *popt)
    print(f"  N={N:4d}: D = {pred:.2f}%")

# --- Plot ---
fig, ax = plt.subplots(figsize=(5.5, 4))

# Smooth fit curve
N_smooth = np.linspace(60, 1100, 500)
h_smooth = 1.0 / N_smooth
D_smooth = power_law(h_smooth, *popt)

ax.plot(N_smooth, D_smooth, '-', color='C0', linewidth=1.5,
        label=f'Fit: $\\mathcal{{D}}_\\infty + A\\,h^p$\n'
              f'($\\mathcal{{D}}_\\infty$={D_inf:.1f}%, $p$={p:.2f})')

# Data points
ax.plot(mesh_res, distortion, 'o', color='C0', markersize=7, zorder=5,
        label='Measured')

# Extrapolation region
ax.axvspan(250, 1100, alpha=0.08, color='grey')
ax.text(620, 39, 'extrapolation', ha='center', fontsize=9, color='0.4', style='italic')

# Mark key predictions
for N, marker in [(500, 's'), (1000, 'D')]:
    pred = power_law(1.0/N, *popt)
    ax.plot(N, pred, marker, color='C3', markersize=8, zorder=5,
            label=f'Predicted $N$={N}: {pred:.1f}%')

# Asymptote
ax.axhline(D_inf, color='C2', linestyle='--', linewidth=1, alpha=0.7,
           label=f'Asymptote $\\mathcal{{D}}_\\infty$ = {D_inf:.1f}%')

ax.set_xlabel('Mesh resolution $N$ ($N{\\times}N$ elements)', fontsize=11)
ax.set_ylabel('Right-boundary distortion $\\mathcal{D}$ [%]', fontsize=11)
ax.legend(fontsize=8.5, loc='upper right')
ax.set_xlim(50, 1100)
ax.set_ylim(0, 50)
ax.grid(True, alpha=0.3)
fig.tight_layout()

outpath = 'docs/figures/continuous_ceff_convergence.pdf'
fig.savefig(outpath, bbox_inches='tight')
print(f'\nSaved to {outpath}')
plt.show()
