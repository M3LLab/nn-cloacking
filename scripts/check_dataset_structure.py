import h5py, numpy as np
import argparse


parser = argparse.ArgumentParser(description="Check dataset structure and value ranges.")
parser.add_argument("dataset_path", 
                    default=None,
                    help="Path to the dataset HDF5 file")
args = parser.parse_args()

dataset_path = 'output/dataset.h5' if args.dataset_path is None else args.dataset_path

with h5py.File(dataset_path, 'r') as f:
    vf = f['vf'][:]
    rho = f['rho_eff'][:]
    C = f['C_eff'][:]
    # Effective properties range
    lam = C[:, 0, 1]  # C_1122
    mu = 0.5*(C[:, 2, 2] + C[:, 2, 3])
    print('Volume fraction: min={:.3f}, max={:.3f}, mean={:.3f}'.format(vf.min(), vf.max(), vf.mean()))
    print('rho_eff: min={:.1f}, max={:.1f}'.format(rho.min(), rho.max()))
    print('lambda_eff: min={:.2e}, max={:.2e}'.format(lam.min(), lam.max()))
    print('mu_eff: min={:.2e}, max={:.2e}'.format(mu.min(), mu.max()))
    # Anisotropy check: C_1212 vs C_1221
    print('C_1212 range: [{:.2e}, {:.2e}]'.format(C[:,2,2].min(), C[:,2,2].max()))
    print('C_1221 range: [{:.2e}, {:.2e}]'.format(C[:,2,3].min(), C[:,2,3].max()))
    print('Max |C_1212-C_1221|/mean: {:.3f}'.format(np.max(np.abs(C[:,2,2]-C[:,2,3]))/np.mean(np.abs(C[:,2,2]))))

    print(f'lam/mu ratio:  [{ratio.min():.4f}, {ratio.max():.4f}], mean={ratio.mean():.4f}')

    print(f'\\nTarget needs: lam/mu in [0.61, 1.44], rho in [1154, 3634]')
    print(f'Dataset covers: lam/mu in [{ratio.min():.2f}, {ratio.max():.2f}], rho in [{rho.min():.0f}, {rho.max():.0f}]')
    
    # Check C_eff shape
    print(f'\\nC_eff shape: {C.shape}')  # (n_samples, 3, 4) or (n_samples, 4, 4)?
    if len(f.keys()) > 3:
        print(f'Dataset keys: {list(f.keys())}')
"
