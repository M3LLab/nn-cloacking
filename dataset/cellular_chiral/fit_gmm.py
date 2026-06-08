"""Fit a Gaussian Mixture Model on per-cell material features from a stiffness HDF5.

Saves a portable .npz that the optimisation loop can load on any machine to
apply a flat-top dataset-prior penalty: ``max(0, threshold - log p(C, rho))``.

Two feature sets are supported, selected with ``--feature-set``:

  * ``lame``  (default, D=3) → (λ, μ, ρ), matching the ``n_C_params=2`` (flat2)
    isotropic cell parameterisation.
  * ``flat4`` (D=5) → (C₁₁₁₁, C₂₂₂₂, C₁₂₁₂, C₁₁₂₂, ρ), matching the
    ``n_C_params=4`` (flat4) cell parameterisation. The four stiffness features
    are read from the dataset's orthotropic Voigt components
    (C11, C22, C66, C12) in exactly the flat4 column order.

The .npz contains (D = 3 for ``lame``, 5 for ``flat4``):
    weights              (K,)            mixture weights
    means                (K, D)          means in standardised feature space
    covariances          (K, D, D)       full covariances (standardised space)
    precisions_cholesky  (K, D, D)       lower-triangular Cholesky of precision
                                         (sklearn convention; lets the JAX
                                         log-prob skip an inverse)
    feature_mean         (D,)            standardisation mean (raw → std)
    feature_std          (D,)            standardisation std
    threshold            scalar          τ used in the flat-top penalty
    threshold_percentile scalar          quantile of dataset log p chosen as τ
    n_components         int             K
    n_samples            int             dataset size used for the fit
    feature_order        (D,) str        feature names, in order
    n_C_params           int             cell representation this prior matches
                                         (2 for ``lame``, 4 for ``flat4``)
    bic                  scalar          for K-selection sanity
    log_p_quantiles      (5,)            (q1, q5, q25, q50, q75) of dataset log p

Usage
-----
    python -m dataset.cellular_chiral.fit_gmm \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/gmm_lambda_mu_rho.npz \
        -K 16 \
        --threshold-percentile 0.25

    python -m dataset.cellular_chiral.fit_gmm \
        -i output/ca_bulk_squared/stiffness.h5 \
        -o output/ca_bulk_squared/gmm_flat4.npz \
        --feature-set flat4 -K 16 --threshold-percentile 0.25
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
from sklearn.mixture import GaussianMixture

# Concurrent reads of an actively-written h5 are fine if locking is off.
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")


# Feature-set definitions. Each maps to:
#   datasets     : HDF5 dataset names to stack as the raw feature columns
#   feature_order: human-readable names (saved to the .npz, in column order)
#   n_C_params   : the cell representation this prior is valid against
# For ``flat4`` the dataset's orthotropic Voigt components map to the flat4
# tensor columns [C1111, C2222, C1212, C1122] as: C11→C1111, C22→C2222,
# C66→C1212 (shear), C12→C1122 (normal coupling).
_FEATURE_SETS = {
    "lame": {
        "datasets": ["lambda_", "mu", "rho"],
        "feature_order": ["lambda", "mu", "rho"],
        "n_C_params": 2,
    },
    "flat4": {
        "datasets": ["C11", "C22", "C66", "C12", "rho"],
        "feature_order": ["C1111", "C2222", "C1212", "C1122", "rho"],
        "n_C_params": 4,
    },
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("-i", "--input", type=Path,
                    default=Path("output/ca_bulk_squared/stiffness.h5"),
                    help="Stiffness HDF5 produced by bulk_stiffness.py")
    ap.add_argument("-o", "--output", type=Path,
                    default=Path("output/ca_bulk_squared/gmm_lambda_mu_rho.npz"),
                    help="Where to save the fitted GMM artifact")
    ap.add_argument("--feature-set", default="lame", choices=list(_FEATURE_SETS),
                    help="Which per-cell features to fit the GMM on. "
                         "'lame' (D=3, λ/μ/ρ) matches n_C_params=2; "
                         "'flat4' (D=5, C1111/C2222/C1212/C1122/ρ) matches "
                         "n_C_params=4.")
    ap.add_argument("-K", "--n-components", type=int, default=16,
                    help="Number of GMM mixture components")
    ap.add_argument("--covariance-type", default="full",
                    choices=["full"],
                    help="Only 'full' is supported by the JAX-side log-prob.")
    ap.add_argument("--threshold-percentile", type=float, default=0.25,
                    help="Quantile of dataset log p below which the flat-top "
                         "penalty kicks in. 0.25 = bottom quartile, leaves a "
                         "comfortable margin so the optimiser is pushed away "
                         "from the borders of the manifold, not just its tails.")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bic-sweep", type=int, nargs="*", default=None,
                    help="Optional list of K values to evaluate BIC on, "
                         "before the final fit at -K. Just for diagnostics.")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"missing input: {args.input}")

    fset = _FEATURE_SETS[args.feature_set]
    with h5py.File(args.input, "r") as f:
        missing = [d for d in fset["datasets"] if d not in f]
        if missing:
            raise SystemExit(
                f"feature-set {args.feature_set!r} needs datasets {missing} "
                f"which are absent from {args.input}"
            )
        cols = [f[d][:] for d in fset["datasets"]]
    n = cols[0].size
    print(f"loaded {n} samples from {args.input} "
          f"(feature-set={args.feature_set}, features={fset['feature_order']})")

    X = np.column_stack(cols).astype(np.float64)
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    Xs = (X - feature_mean) / feature_std
    print(f"standardisation: mean={feature_mean}, std={feature_std}")

    if args.bic_sweep:
        print("\nBIC sweep:")
        for k in args.bic_sweep:
            g = GaussianMixture(
                n_components=k, covariance_type=args.covariance_type,
                max_iter=args.max_iter, random_state=args.seed,
            ).fit(Xs)
            print(f"  K={k:3d}  BIC={g.bic(Xs):.1f}  AIC={g.aic(Xs):.1f}")

    print(f"\nfitting GMM (K={args.n_components}, cov={args.covariance_type}) ...")
    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type=args.covariance_type,
        max_iter=args.max_iter,
        random_state=args.seed,
        verbose=2,
    )
    gmm.fit(Xs)
    bic = float(gmm.bic(Xs))

    log_p = gmm.score_samples(Xs)
    quantiles = np.quantile(log_p, [0.01, 0.05, 0.25, 0.50, 0.75])
    threshold = float(np.quantile(log_p, args.threshold_percentile))
    print(
        f"\nlog_p stats:\n"
        f"  min={log_p.min():.3f}  q1={quantiles[0]:.3f}  q5={quantiles[1]:.3f}  "
        f"q25={quantiles[2]:.3f}  q50={quantiles[3]:.3f}  q75={quantiles[4]:.3f}  "
        f"max={log_p.max():.3f}"
    )
    print(f"threshold τ (q={args.threshold_percentile}): {threshold:.4f}")
    print(f"BIC: {bic:.1f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        weights=gmm.weights_.astype(np.float64),
        means=gmm.means_.astype(np.float64),
        covariances=gmm.covariances_.astype(np.float64),
        precisions_cholesky=gmm.precisions_cholesky_.astype(np.float64),
        feature_mean=feature_mean.astype(np.float64),
        feature_std=feature_std.astype(np.float64),
        threshold=np.float64(threshold),
        threshold_percentile=np.float64(args.threshold_percentile),
        n_components=np.int64(args.n_components),
        n_samples=np.int64(n),
        feature_order=np.array(fset["feature_order"]),
        n_C_params=np.int64(fset["n_C_params"]),
        bic=np.float64(bic),
        log_p_quantiles=quantiles.astype(np.float64),
        covariance_type=np.array(args.covariance_type),
    )
    print(f"\nsaved GMM to {args.output}")


if __name__ == "__main__":
    main()
