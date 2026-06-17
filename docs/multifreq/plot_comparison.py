import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data_25 = np.genfromtxt('uratio_vs_fstar_2.5.csv', delimiter=',', skip_header=1)
data_30 = np.genfromtxt('uratio_vs_fstar_3.0.csv', delimiter=',', skip_header=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(data_25[:, 0], data_25[:, 1], 'o-', label=r'Optimized at $f^*=2.5$', linewidth=2, markersize=5)
ax.plot(data_30[:, 0], data_30[:, 1], 's-', label=r'Optimized at $f^*=3.0$', linewidth=2, markersize=5)
ax.set_xlabel(r'Frequency $f^*$', fontsize=12)
ax.set_ylabel('Displacement Ratio', fontsize=12)
ax.set_title('Single-Frequency Optimization: Frequency-Specific Overfitting', fontsize=13)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
ax.set_xlim(0.8, 4.2)
plt.tight_layout()
plt.savefig('single_freq_comparison.png', dpi=150, bbox_inches='tight')
print('saved single_freq_comparison.png')
