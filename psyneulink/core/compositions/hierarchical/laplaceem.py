# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  Laplace EM  **************************************************************

"""Empirical-Bayes Laplace EM.

The E-step maximizes ``log_likelihood(theta(z)) + log N(z | mu, diag(sigma))`` over the unconstrained
`z` (see `transforms <transforms>`) and takes the inverse curvature at the mode as the posterior
covariance.  Only the diagonal is computed, matching the diagonal group covariance.  The M-step
updates `beta` by least squares on the modes and `sigma` from the posterior second moments.

Curvature comes from central finite differences, which requires the objective to be deterministic in
`theta`.  Participant models must therefore be built with common random numbers; without them the
differences measure simulation noise rather than curvature.

The likelihood is reached only through the E-step runner passed to `fit_laplace_em`, so the driver is
independent of how participants are fitted or where.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from psyneulink._typing import Mapping, Optional, Union

__all__ = [
    "EStepConfig",
    "SubjectPosterior",
    "diagonal_hessian",
    "log_gauss_diag",
    "subject_map_estep",
]

LOG_2PI = np.log(2.0 * np.pi)

#: Step size for the finite-difference curvature, as a fraction of the prior standard deviation.
#: See `EStepConfig.hessian_step` for why the step is scaled to the prior rather than fixed.
DEFAULT_HESSIAN_STEP_SCALE = 0.25


def log_gauss_diag(z, mean, var):
    """Log density of a diagonal Gaussian, summed over dimensions.

    This is the group-level prior term in a participant's MAP objective.
    """
    z = np.asarray(z, dtype=float)
    return float(-0.5 * np.sum((z - mean) ** 2 / var + LOG_2PI + np.log(var)))


def diagonal_hessian(func, z, step, f0=None):
    """Diagonal of the Hessian of scalar `func` at `z`, by central second differences.

    ``H_kk = (f(z + h e_k) - 2 f(z) + f(z - h e_k)) / h^2``

    Arguments
    ---------

    func : callable
        Scalar objective.  Must be deterministic in its argument; see the module docstring.

    z : array-like
        Point at which to evaluate the curvature, normally the mode.

    step : float or array-like
        Perturbation size, either shared or per dimension.

    f0 : float : default None
        Value of `func` at `z`, if already known, to save one evaluation.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    steps = np.full(n, float(step)) if np.isscalar(step) else np.asarray(step, dtype=float)
    if steps.shape != (n,):
        raise ValueError(f"step must be scalar or of shape {(n,)}; got shape {steps.shape}")
    if np.any(steps <= 0):
        raise ValueError("step must be positive")

    if f0 is None:
        f0 = func(z)
    diag = np.empty(n, dtype=float)
    for k in range(n):
        h = steps[k]
        zp, zm = z.copy(), z.copy()
        zp[k] += h
        zm[k] -= h
        diag[k] = (func(zp) - 2.0 * f0 + func(zm)) / (h * h)
    return diag


@dataclass(frozen=True)
class EStepConfig:
    """Settings shared by every participant's E-step.

    Collected into one object so that the same values reach both the per-participant optimization
    and the group-level update, rather than being passed separately to each and risking a mismatch.

    Attributes
    ----------

    method : str
        Any method accepted by `scipy.optimize.minimize`.  The default is derivative-free because a
        simulation-backed likelihood has no gradient.

    hessian_step : float or array-like or None
        Perturbation for the finite-difference curvature, in unconstrained units.  When None (the
        default) it is derived per dimension from the current group variance as
        ``DEFAULT_HESSIAN_STEP_SCALE * sqrt(sigma)``.

        Scaling rather than fixing it matters because, in unconstrained space, the width of a
        participant's posterior is set by the group standard deviation.  A step proportional to
        ``sqrt(sigma)`` is therefore dimensionless and independent of how wide the user's search
        range happens to be, whereas a fixed step has to be retuned whenever either changes.

        `DEFAULT_HESSIAN_STEP_SCALE` is a reasonable default rather than a tuned optimum.  The step
        actually used is recorded on `SubjectPosterior.hessian_step` so that a fit can be audited.

    variance_floor : float
        Smallest posterior variance to report.  Guards against a zero that would make the group
        update degenerate.

    optimizer_options : Mapping or None
        Passed through to `scipy.optimize.minimize`, overriding the defaults below.

    simplex_scale : float
        Size of the initial Nelder-Mead simplex.  Set explicitly because scipy's default is
        proportional to the starting point, which collapses when a coordinate starts at zero -- as
        it does whenever a participant starts at the group mean.
    """

    method: str = "Nelder-Mead"
    hessian_step: Optional[Union[float, np.ndarray]] = None
    variance_floor: float = 1e-6
    optimizer_options: Optional[Mapping] = None
    simplex_scale: float = 0.5

    def resolve_hessian_step(self, prior_variance):
        """Return the per-dimension finite-difference step to use for this prior variance."""
        prior_variance = np.asarray(prior_variance, dtype=float)
        if self.hessian_step is None:
            return DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(prior_variance)
        step = np.asarray(self.hessian_step, dtype=float)
        return np.broadcast_to(step, prior_variance.shape).astype(float, copy=True)


@dataclass
class SubjectPosterior:
    """One participant's Laplace posterior, plus enough detail to tell whether to trust it."""

    z_hat: np.ndarray          # mode, unconstrained
    variance: np.ndarray       # diagonal posterior variance, unconstrained
    curvature: np.ndarray      # diagonal Hessian of the objective at the mode
    neg_log_post: float        # objective value at the mode
    success: bool              # whether the optimizer reported convergence
    message: str               # the optimizer's own account of why it stopped
    hessian_step: np.ndarray   # the step actually used, so the choice is auditable


def subject_map_estep(neg_log_post, z0, prior_variance, config=None):
    """Find one participant's posterior mode and approximate the posterior around it.

    Arguments
    ---------

    neg_log_post : callable
        Negative log posterior for this participant, as a function of unconstrained `z`.  Must be
        deterministic; see the module docstring.

    z0 : array-like
        Starting point, normally the group's prediction for this participant, or their previous
        mode when warm-starting.

    prior_variance : array-like
        Current group variance, per parameter.  Used both to derive the finite-difference step and
        to bound the reported posterior variance from above.

    config : EStepConfig : default None
        Settings; a default-constructed `EStepConfig` if omitted.

    Returns
    -------

    A `SubjectPosterior`.
    """
    config = config if config is not None else EStepConfig()
    z0 = np.asarray(z0, dtype=float)
    prior_variance = np.asarray(prior_variance, dtype=float)

    options = {}
    if config.method == "Nelder-Mead":
        n = z0.size
        simplex = np.vstack([z0] + [z0 + config.simplex_scale * e for e in np.eye(n)])
        options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000, "initial_simplex": simplex}
    if config.optimizer_options:
        options.update(config.optimizer_options)

    result = minimize(neg_log_post, z0, method=config.method, options=options)
    z_hat = np.asarray(result.x, dtype=float)

    step = config.resolve_hessian_step(prior_variance)
    curvature = diagonal_hessian(neg_log_post, z_hat, step=step, f0=float(result.fun))

    # Non-positive curvature means the objective is locally flat or concave here, so the data say
    # nothing about this parameter for this participant. Falling back to the prior variance is the
    # honest answer: it reports "we learned nothing", rather than a spuriously tight interval from
    # a meaningless reciprocal.
    with np.errstate(divide="ignore"):
        variance = np.where(curvature > 0, 1.0 / np.where(curvature > 0, curvature, 1.0), np.inf)
    variance = np.minimum(variance, prior_variance)
    variance = np.maximum(variance, config.variance_floor)

    return SubjectPosterior(
        z_hat=z_hat,
        variance=variance,
        curvature=curvature,
        neg_log_post=float(result.fun),
        success=bool(result.success),
        message=str(getattr(result, "message", "")),
        hessian_step=step,
    )
