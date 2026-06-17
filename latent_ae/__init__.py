"""Frequency-aware autoencoder and latent-space multi-frequency optimization.

Sibling of ``surrogate/``. The autoencoder uses a shared + frequency-residual
latent decomposition ``z = z_sh + z_f(f) + z_f_shift``; a performance head
predicts the cloaking loss from ``z``; a within-frequency pairwise margin
ranking loss organizes ``z_sh`` so good designs cluster together.
"""
