"""Finite-difference diagonal Hessian for the Laplace E-step.

The simulated PEC likelihood is not analytically differentiable, so posterior curvature is
estimated numerically. Only the diagonal is needed for the diagonal-covariance model. Central
second differences are used; the objective should hold random seeds fixed (common random
numbers) across the perturbations so the curvature is not dominated by simulation noise.
"""

import numpy as np


def diagonal_hessian(func, z, step=1e-4, f0=None):
    """Diagonal of the Hessian of scalar ``func`` at ``z`` via central differences.

    H_kk = (f(z + h e_k) - 2 f(z) + f(z - h e_k)) / h^2
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    steps = np.full(n, step, dtype=float) if np.isscalar(step) else np.asarray(step, dtype=float)
    if f0 is None:
        f0 = func(z)
    diag = np.empty(n, dtype=float)
    for k in range(n):
        h = steps[k]
        zp = z.copy()
        zm = z.copy()
        zp[k] += h
        zm[k] -= h
        diag[k] = (func(zp) - 2.0 * f0 + func(zm)) / (h * h)
    return diag
