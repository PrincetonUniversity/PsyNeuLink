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

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from psyneulink._typing import Mapping, Optional, Union

__all__ = [
    "EStepConfig",
    "EStepResult",
    "HierarchicalEMWarning",
    "LaplaceEMResult",
    "SubjectPosterior",
    "diagonal_hessian",
    "fit_laplace_em",
    "log_gauss_diag",
    "make_inprocess_estep_runner",
    "subject_laplace_objective",
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


class HierarchicalEMWarning(UserWarning):
    """Raised when an EM iteration completes but something about it warrants attention."""


def subject_laplace_objective(neg_log_post, variance, n_params):
    """One participant's contribution to the Laplace marginal log-likelihood.

    The quantity EM is really maximizing: the log-likelihood of the participant's data with their
    parameters integrated out, under the Gaussian approximation to their posterior.
    """
    return -neg_log_post + 0.5 * n_params * LOG_2PI + 0.5 * float(np.sum(np.log(variance)))


@dataclass
class EStepResult:
    """Every participant's posterior for one EM iteration.

    Arrays are indexed by participant in a fixed order, so that a distributed E-step and an
    in-process one produce identical results rather than depending on completion order.
    """

    z_hat: np.ndarray             # (n_subjects, n_params) modes
    variance: np.ndarray          # (n_subjects, n_params) posterior variances
    curvature: np.ndarray         # (n_subjects, n_params)
    hessian_step: np.ndarray      # (n_subjects, n_params) steps used
    subject_objective: np.ndarray  # (n_subjects,) per-participant Laplace marginal
    success: np.ndarray           # (n_subjects,) bool
    messages: tuple               # (index, message) for participants that did not converge

    @property
    def objective(self):
        """Total Laplace marginal log-likelihood, summed in participant order."""
        return float(np.sum(self.subject_objective))


def make_inprocess_estep_runner(log_likelihood, transform, config=None):
    """Build an E-step that fits each participant in turn, in this process.

    Arguments
    ---------

    log_likelihood : callable
        ``log_likelihood(theta, subject_index) -> float``, the log-likelihood of one participant's
        data at parameters `theta`, in the model's own units.

    transform : BoundedTransform or IdentityTransform
        Maps between the unconstrained space the group model lives in and the model's units.

    config : EStepConfig : default None
        Settings for each participant's optimization.

    Returns
    -------

    A callable ``runner(mu, sigma, prev_z, warm_start) -> EStepResult``.
    """
    config = config if config is not None else EStepConfig()

    def runner(mu, sigma, prev_z, warm_start):
        n_subjects, n_params = mu.shape
        z_hat = np.empty((n_subjects, n_params))
        variance = np.empty((n_subjects, n_params))
        curvature = np.empty((n_subjects, n_params))
        steps = np.empty((n_subjects, n_params))
        subject_objective = np.empty(n_subjects)
        success = np.empty(n_subjects, dtype=bool)
        messages = []

        for s in range(n_subjects):
            mu_s = mu[s]

            def neg_log_post(z, s=s, mu_s=mu_s):
                theta = transform.to_natural(z)
                return -float(log_likelihood(theta, s)) - log_gauss_diag(z, mu_s, sigma)

            post = subject_map_estep(
                neg_log_post,
                z0=prev_z[s] if warm_start else mu_s,
                prior_variance=sigma,
                config=config,
            )
            z_hat[s] = post.z_hat
            variance[s] = post.variance
            curvature[s] = post.curvature
            steps[s] = post.hessian_step
            subject_objective[s] = subject_laplace_objective(
                post.neg_log_post, post.variance, n_params
            )
            success[s] = post.success
            if not post.success:
                messages.append((s, post.message))

        return EStepResult(
            z_hat=z_hat,
            variance=variance,
            curvature=curvature,
            hessian_step=steps,
            subject_objective=subject_objective,
            success=success,
            messages=tuple(messages),
        )

    return runner


@dataclass
class LaplaceEMResult:
    """Outcome of a hierarchical fit, in unconstrained units.

    Conversion to the model's own units belongs to the caller, which owns the transform.
    """

    beta: np.ndarray          # (n_predictors, n_params) group means
    sigma: np.ndarray         # (n_params,) group variances
    z_hat: np.ndarray         # (n_subjects, n_params) participant modes
    variance: np.ndarray      # (n_subjects, n_params) participant posterior variances
    objective: float          # Laplace marginal log-likelihood at the returned beta and sigma
    n_iter: int
    converged: bool
    subject_converged: np.ndarray  # (n_subjects,) bool, from the final E-step
    history: list             # one entry per iteration; see `fit_laplace_em`
    hessian_step: np.ndarray  # (n_subjects, n_params) steps used in the final E-step


def fit_laplace_em(
    estep_runner,
    n_subjects,
    n_params,
    *,
    design_matrix=None,
    estep_config=None,
    max_iterations=50,
    tol=1e-4,
    damping=0.0,
    init_beta=None,
    init_sigma=None,
    warm_start=True,
    final_estep=True,
):
    """Fit a hierarchical model by empirical-Bayes Laplace EM.

    Alternates between estimating each participant's posterior given the group (the E-step, supplied
    as `estep_runner`) and re-estimating the group from those posteriors (the M-step, here).  The
    group model is ``z_s ~ N(X_s beta, diag(sigma))``.

    The likelihood is reached only through `estep_runner`, so the same driver fits a closed-form test
    model and a simulation-backed one without change.

    Arguments
    ---------

    estep_runner : callable
        ``runner(mu, sigma, prev_z, warm_start) -> EStepResult``.

    n_subjects, n_params : int
        Shape of the problem.

    design_matrix : array-like : default None
        ``(n_subjects, n_predictors)`` of participant-level predictors.  Defaults to an intercept.

    estep_config : EStepConfig : default None
        Used here only for `variance_floor`; the runner holds its own copy for the E-step.

    max_iterations, tol : int, float
        Stop after this many iterations, or once no group parameter moves by more than `tol`.

    damping : float
        Fraction of the previous group estimate to retain each M-step.  Slows the fit but steadies
        it when participant posteriors are noisy.

    init_beta, init_sigma : array-like : default None
        Starting group estimates; zeros and ones respectively by default.

    warm_start : bool
        Start each participant from their previous mode rather than from the group prediction.

    final_estep : bool
        Run one more E-step at the returned group estimate, so that the participant-level results
        describe the group estimate actually reported.  Without it they lag by one M-step.

    Returns
    -------

    A `LaplaceEMResult`.  Each entry of its `history` records an iteration's group estimate together
    with the objective computed under *that* estimate, so the two can be read side by side.

    Convergence is judged by how far the group estimate moves, not by the objective, which is not
    monotone under an approximate E-step.
    """
    X = np.ones((n_subjects, 1)) if design_matrix is None else np.asarray(design_matrix, float)
    if X.shape[0] != n_subjects:
        raise ValueError(
            f"design_matrix must have one row per participant; got {X.shape[0]} rows "
            f"for {n_subjects} participants"
        )
    n_predictors = X.shape[1]
    variance_floor = (estep_config or EStepConfig()).variance_floor

    beta = np.zeros((n_predictors, n_params)) if init_beta is None else np.array(init_beta, float)
    sigma = np.ones(n_params) if init_sigma is None else np.array(init_sigma, float)

    prev_z = X @ beta
    history = []
    converged = False
    estep = None

    for iteration in range(max_iterations):
        mu = X @ beta
        estep = estep_runner(mu, sigma, prev_z, warm_start)
        prev_z = estep.z_hat

        if not np.all(estep.success):
            failed = np.flatnonzero(~estep.success)
            warnings.warn(
                f"EM iteration {iteration}: {failed.size} of {n_subjects} participants did not "
                f"converge (indices {failed[:3].tolist()}"
                f"{', ...' if failed.size > 3 else ''}). Their estimates still contribute to the "
                f"group update; check subject_converged on the result.",
                HierarchicalEMWarning,
                stacklevel=2,
            )

        # M-step: group means by least squares on the participant modes, group variances from the
        # posterior second moments. The posterior variances are added so that participants whose
        # parameters are poorly determined widen the group variance rather than shrinking it.
        beta_new = np.linalg.lstsq(X, estep.z_hat, rcond=None)[0]
        resid = estep.z_hat - X @ beta_new
        sigma_new = np.maximum(np.mean(resid ** 2 + estep.variance, axis=0), variance_floor)

        if damping > 0.0:
            beta_new = (1.0 - damping) * beta_new + damping * beta
            sigma_new = (1.0 - damping) * sigma_new + damping * sigma

        delta = max(
            float(np.max(np.abs(beta_new - beta))),
            float(np.max(np.abs(sigma_new - sigma))),
        )

        # The objective was computed under the group estimate that produced it, so record them
        # together; the update that follows belongs to the next entry.
        history.append({
            "iter": iteration,
            "objective": estep.objective,
            "beta": beta.copy(),
            "sigma": sigma.copy(),
            "delta": delta,
            "n_subject_failures": int(np.count_nonzero(~estep.success)),
        })

        beta, sigma = beta_new, sigma_new
        if delta < tol:
            converged = True
            break

    if final_estep and estep is not None:
        estep = estep_runner(X @ beta, sigma, prev_z, warm_start)

    return LaplaceEMResult(
        beta=beta,
        sigma=sigma,
        z_hat=estep.z_hat,
        variance=estep.variance,
        objective=estep.objective,
        n_iter=len(history),
        converged=converged,
        subject_converged=estep.success,
        history=history,
        hessian_step=estep.hessian_step,
    )
