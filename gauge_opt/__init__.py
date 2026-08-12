"""Gauge optimization for transformation elasticity onto D2-cell reachable sets.

See ``theory.md`` for the formulas and ``README.md`` for the pipeline.
"""

import jax

# The gauge identities we rely on (exact minor symmetry of the A = c F^T gauge,
# agreement with the closed form of eq 4.5) hold to machine precision only in
# double.  In JAX's default float32 the residuals floor out around 1e-7, which
# is indistinguishable from a real modelling error.
jax.config.update("jax_enable_x64", True)
