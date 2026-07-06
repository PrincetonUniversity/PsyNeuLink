"""Per-subject Laplace E-step: MAP optimization + diagonal posterior variance.

Finds ``z_hat = argmin neg_log_post(z)`` with a local optimizer warm-started from the group
prediction, then approximates the posterior as ``N(z_hat, V)`` with ``V`` the inverse of the
diagonal Hessian of ``neg_log_post`` at the mode.

The optimizer is composition-agnostic: it only sees the scalar objective, so any PEC model
(however complex) plugs in behind ``neg_log_post`` unchanged.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from hessian import diagonal_hessian


@dataclass
class SubjectPosterior:
    z_hat: np.ndarray          # MAP in unconstrained space
    variance: np.ndarray       # diagonal posterior variance V (z-space)
    neg_log_post: float        # objective at the mode
    curvature: np.ndarray      # diagonal Hessian of neg_log_post at the mode
    success: bool


def subject_map_estep(
    neg_log_post,
    z0,
    *,
    method="Nelder-Mead",
    hessian_step=1e-4,
    variance_floor=1e-6,
    prior_variance=None,
    simplex_scale=0.5,
    optimizer_options=None,
):
    """Optimize one subject's negative log posterior and return its Laplace summary."""
    z0 = np.asarray(z0, dtype=float)

    options = {}
    if method == "Nelder-Mead":
        # An explicit simplex avoids scipy's value-relative default, which collapses near z0 == 0.
        n = z0.size
        simplex = np.vstack([z0] + [z0 + simplex_scale * e for e in np.eye(n)])
        options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000, "initial_simplex": simplex}
    if optimizer_options:
        options.update(optimizer_options)

    res = minimize(neg_log_post, z0, method=method, options=options)
    z_hat = np.asarray(res.x, dtype=float)

    curvature = diagonal_hessian(neg_log_post, z_hat, step=hessian_step, f0=float(res.fun))
    # Non-positive curvature means the likelihood is locally uninformative; the posterior then
    # falls back to the prior, so cap the variance at the prior variance rather than collapsing it.
    with np.errstate(divide="ignore"):
        variance = np.where(curvature > 0, 1.0 / np.where(curvature > 0, curvature, 1.0), np.inf)
    if prior_variance is not None:
        variance = np.minimum(variance, np.asarray(prior_variance, dtype=float))
    variance = np.maximum(variance, variance_floor)

    return SubjectPosterior(
        z_hat=z_hat,
        variance=variance,
        neg_log_post=float(res.fun),
        curvature=curvature,
        success=bool(res.success),
    )
