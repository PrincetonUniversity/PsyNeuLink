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
covariance.  The M-step updates `beta` by least squares on the modes and `sigma` from the posterior
second moments.

How much of the curvature is measured is `EStepConfig.curvature`; the group covariance is
diagonal either way.

Curvature comes from central finite differences, which requires the objective to be deterministic in
`theta`: without that the differences measure simulation noise rather than curvature.  A simulated
model is made deterministic by `same_seed_for_all_parameter_combinations`, which
`check_scoring_is_deterministic <subjectlikelihood>` requires of every participant model before a
fit begins.

The likelihood is reached only through the E-step runner passed to `fit_laplace_em`, so the driver is
independent of how participants are fitted or where.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from psyneulink._typing import Mapping, Optional, Union

__all__ = [
    "CURVATURE_KINDS",
    "EStepConfig",
    "EStepResult",
    "HierarchicalEMError",
    "HierarchicalEMWarning",
    "LaplaceEMResult",
    "SubjectPosterior",
    "diagonal_hessian",
    "fit_laplace_em",
    "full_hessian",
    "log_gauss_diag",
    "make_inprocess_estep_runner",
    "subject_map_estep",
]

#: How much of the curvature at a participant's mode to measure.  See `EStepConfig.curvature`.
CURVATURE_KINDS = ("diagonal", "full")

LOG_2PI = np.log(2.0 * np.pi)

#: Step size for the finite-difference curvature, as a fraction of the prior standard deviation.
#: See `EStepConfig.hessian_step` for why the step is scaled to the prior rather than fixed.
DEFAULT_HESSIAN_STEP_SCALE = 0.25

#: How many times to halve a finite-difference step that reached a parameter value the model
#: rules out, before giving up on measuring that parameter's curvature.
MAX_HESSIAN_RETRIES = 3

#: Size of the initial Nelder-Mead simplex.  Set explicitly because scipy's default is proportional
#: to the starting point, which collapses when a coordinate starts at zero -- as it does whenever a
#: participant starts at the group mean.
SIMPLEX_SCALE = 0.5


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
    steps = _resolve_steps(step, n)

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


def _resolve_steps(step, n):
    """Broadcast and check a finite-difference step against a problem of `n` dimensions."""
    steps = np.full(n, float(step)) if np.isscalar(step) else np.asarray(step, dtype=float)
    if steps.shape != (n,):
        raise ValueError(f"step must be scalar or of shape {(n,)}; got shape {steps.shape}")
    if np.any(steps <= 0):
        raise ValueError("step must be positive")
    return steps


def full_hessian(func, z, step, f0=None):
    """Full Hessian of scalar `func` at `z`, by central second differences.

    The diagonal is `diagonal_hessian`; each off-diagonal entry is the four-point form

    ``H_jk = (f(z + h_j e_j + h_k e_k) - f(z + h_j e_j - h_k e_k)
              - f(z - h_j e_j + h_k e_k) + f(z - h_j e_j - h_k e_k)) / (4 h_j h_k)``

    computed once and mirrored, so the result is symmetric by construction.  Arguments are as for
    `diagonal_hessian`; the cost is ``2 n^2`` evaluations of `func` rather than ``2 n``.
    """
    z = np.asarray(z, dtype=float)
    n = z.size
    steps = _resolve_steps(step, n)
    if f0 is None:
        f0 = func(z)

    hessian = np.zeros((n, n))
    np.fill_diagonal(hessian, diagonal_hessian(func, z, steps, f0=f0))
    for j in range(n):
        for k in range(j + 1, n):
            corners = 0.0
            for sign_j in (1.0, -1.0):
                for sign_k in (1.0, -1.0):
                    shifted = z.copy()
                    shifted[j] += sign_j * steps[j]
                    shifted[k] += sign_k * steps[k]
                    corners += sign_j * sign_k * func(shifted)
            hessian[j, k] = hessian[k, j] = corners / (4.0 * steps[j] * steps[k])
    return hessian


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

    curvature : "diagonal" or "full"
        How much of the curvature at each participant's mode to measure.  ``"diagonal"`` measures
        one parameter at a time with the others held at the mode; ``"full"`` measures the whole
        matrix and inverts it, so a parameter's reported width accounts for the others being
        uncertain too.  Where parameters trade off the first is the smaller of the two, so it
        reports intervals that are too tight and group variances that are too low.  ``"full"``
        costs ``2 P^2`` evaluations of a participant's objective instead of ``2 P``.

    hessian_step : float or array-like or None
        Perturbation for the finite-difference curvature, in unconstrained units.  When None (the
        default) it is derived per dimension from the current group variance as
        ``DEFAULT_HESSIAN_STEP_SCALE * sqrt(sigma)``.

        In unconstrained space a participant's posterior width is set by the group standard
        deviation, so a step proportional to ``sqrt(sigma)`` is dimensionless and independent of
        the search range; a fixed step has to be retuned when either changes.  The step used is
        recorded on `SubjectPosterior.hessian_step`.

    variance_floor : float
        Smallest posterior variance to report.  Guards against a zero that would make the group
        update degenerate.

    optimizer_options : Mapping or None
        Passed through to `scipy.optimize.minimize`, overriding the defaults below.

    """

    method: str = "Nelder-Mead"
    curvature: str = "diagonal"
    hessian_step: Optional[Union[float, np.ndarray]] = None
    variance_floor: float = 1e-6
    optimizer_options: Optional[Mapping] = None

    def __post_init__(self):
        if self.curvature not in CURVATURE_KINDS:
            raise ValueError(
                f"curvature must be one of {list(CURVATURE_KINDS)}; got {self.curvature!r}"
            )

    def resolve_hessian_step(self, prior_variance):
        """Return the per-dimension finite-difference step to use for this prior variance."""
        prior_variance = np.asarray(prior_variance, dtype=float)
        if self.hessian_step is None:
            return DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(prior_variance)
        step = np.asarray(self.hessian_step, dtype=float)
        return np.broadcast_to(step, prior_variance.shape).astype(float, copy=True)


def subject_laplace_objective(neg_log_post, covariance, n_params):
    """One participant's contribution to the Laplace marginal log-likelihood.

    The quantity EM is really maximizing: the log-likelihood of the participant's data with their
    parameters integrated out, under the Gaussian approximation to their posterior.
    """
    _, log_det = np.linalg.slogdet(covariance)
    return -neg_log_post + 0.5 * n_params * LOG_2PI + 0.5 * float(log_det)


@dataclass
class SubjectPosterior:
    """One participant's Laplace posterior, plus enough detail to tell whether to trust it."""

    z_hat: np.ndarray          # mode, unconstrained
    covariance: np.ndarray     # (n_params, n_params) posterior covariance, unconstrained
    curvature: np.ndarray      # (n_params, n_params) Hessian of the objective at the mode
    neg_log_post: float        # objective value at the mode
    success: bool              # whether the optimizer reported convergence
    message: str               # the optimizer's own account of why it stopped
    hessian_step: np.ndarray   # the step actually used, so the choice is auditable
    laplace_objective: float   # this participant's marginal, from the curvature alone

    @property
    def variance(self):
        """Per-parameter posterior variance: the diagonal of `covariance`.

        What that diagonal means depends on `EStepConfig.curvature`.
        """
        return np.diag(self.covariance).copy()


def _floor_eigenvalues(covariance, floor):
    """Raise any eigenvalue below `floor`, so the result stays a covariance worth reporting."""
    symmetric = 0.5 * (covariance + covariance.T)
    values, directions = np.linalg.eigh(symmetric)
    if np.all(values >= floor):
        return symmetric
    return (directions * np.maximum(values, floor)) @ directions.T


def _posterior_covariance(curvature, prior_variance, variance_floor):
    """Turn the curvature at a participant's mode into a posterior covariance.

    Returns the covariance to report and the one the Laplace marginal is taken under; they differ
    only in the cap below.

    Both corrections are eigenvalue clips once the curvature is expressed in the prior's own
    scale.  The objective already includes the prior term, so in that scale the curvature is the
    identity where the data said nothing and larger where they said something: an eigenvalue of 1
    means "the prior alone".  With a diagonal curvature the eigenvectors are the parameters
    themselves and this reduces to ``min(1 / H_kk, prior_variance_k)``, one parameter at a time.
    """
    scale = np.sqrt(prior_variance)
    scaled = scale[:, None] * curvature * scale[None, :]
    scaled = 0.5 * (scaled + scaled.T)
    eigenvalues, directions = np.linalg.eigh(scaled)

    # Curvature that is zero or negative describes a fit that is flat or rises away from the mode,
    # which is not a width. The prior stands in and reports that this direction was not pinned
    # down, which is an eigenvalue of 1 in this scale.
    eigenvalues = np.where(eigenvalues > 0, eigenvalues, 1.0)

    def rebuild(values):
        inner = (directions * values) @ directions.T
        return _floor_eigenvalues(scale[:, None] * inner * scale[None, :], variance_floor)

    # Reported: never wider than the prior, which a Gaussian prior guarantees whenever the
    # likelihood is concave at the mode, and is the sane answer when curvature says otherwise.
    # For the marginal, the width of the Gaussian being integrated is set by the curvature alone;
    # capping it would report an integral over a narrower density than was approximated.
    return rebuild(1.0 / np.maximum(eigenvalues, 1.0)), rebuild(1.0 / eigenvalues)


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
        simplex = np.vstack([z0] + [z0 + SIMPLEX_SCALE * e for e in np.eye(n)])
        options = {"xatol": 1e-6, "fatol": 1e-6, "maxiter": 2000, "initial_simplex": simplex}
    if config.optimizer_options:
        options.update(config.optimizer_options)

    result = minimize(neg_log_post, z0, method=config.method, options=options)
    z_hat = np.asarray(result.x, dtype=float)

    # Curvature is measured as a difference from the value at the mode, so an objective that is
    # not finite there describes no posterior at all and every number taken from it would be
    # meaningless. Falling back to the prior is for a probe that missed, not for this.
    if not np.isfinite(result.fun):
        raise HierarchicalEMError(
            f"the objective is {result.fun} at the fitted point {z_hat.tolist()}, so this "
            f"participant has no posterior to summarize. This usually means their data are "
            f"impossible under their model everywhere the fit looked; check that the data and "
            f"the model the factory builds for them go together."
        )

    step = config.resolve_hessian_step(prior_variance)
    f0 = float(result.fun)

    def measure(step):
        if config.curvature == "full":
            return full_hessian(neg_log_post, z_hat, step=step, f0=f0)
        return np.diag(diagonal_hessian(neg_log_post, z_hat, step=step, f0=f0))

    curvature = measure(step)

    # A difference measures curvature only if every point it uses lands where the model allows; a
    # step reaching an impossible parameter value returns infinity, which describes that point and
    # not the peak. Halve the step in the dimensions involved until it fits, keeping the entries
    # that already came back.
    for _ in range(MAX_HESSIAN_RETRIES):
        unusable = ~np.isfinite(curvature)
        if not unusable.any():
            break
        step = np.where(unusable.any(axis=0) | unusable.any(axis=1), 0.5 * step, step)
        curvature = np.where(unusable, measure(step), curvature)

    # An entry that is still infinite describes a probe that never fit, so the prior stands in
    # there for the inversion below. `curvature` keeps the infinity, so the result still shows
    # that this part of it was never measured.
    usable = curvature
    if not np.isfinite(curvature).all():
        usable = np.where(np.isfinite(curvature), curvature, np.diag(1.0 / prior_variance))

    covariance, laplace_covariance = _posterior_covariance(
        usable, prior_variance, config.variance_floor
    )

    return SubjectPosterior(
        z_hat=z_hat,
        covariance=covariance,
        curvature=curvature,
        neg_log_post=float(result.fun),
        success=bool(result.success),
        message=str(getattr(result, "message", "")),
        hessian_step=step,
        laplace_objective=subject_laplace_objective(
            float(result.fun), laplace_covariance, z_hat.size
        ),
    )


class HierarchicalEMWarning(UserWarning):
    """Raised when an EM iteration completes but something about it warrants attention."""


class HierarchicalEMError(Exception):
    """Raised when a fit cannot proceed, rather than proceeding on meaningless numbers."""


@dataclass
class EStepResult:
    """Every participant's posterior for one EM iteration.

    Arrays are indexed by participant in a fixed order, so that a distributed E-step and an
    in-process one produce identical results rather than depending on completion order.
    """

    z_hat: np.ndarray             # (n_subjects, n_params) modes
    covariance: np.ndarray        # (n_subjects, n_params, n_params) posterior covariances
    curvature: np.ndarray         # (n_subjects, n_params, n_params)
    hessian_step: np.ndarray      # (n_subjects, n_params) steps used
    subject_objective: np.ndarray  # (n_subjects,) per-participant Laplace marginal
    success: np.ndarray           # (n_subjects,) bool
    messages: tuple               # (index, message) for participants that did not converge

    @property
    def variance(self):
        """Per-participant, per-parameter posterior variance: the diagonals of `covariance`.

        What the group update uses, since the group covariance is diagonal.
        """
        return np.diagonal(self.covariance, axis1=1, axis2=2).copy()

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
        posterior = np.empty((n_subjects, n_params, n_params))
        curvature = np.empty((n_subjects, n_params, n_params))
        steps = np.empty((n_subjects, n_params))
        subject_objective = np.empty(n_subjects)
        success = np.empty(n_subjects, dtype=bool)
        messages = []

        for s in range(n_subjects):
            mu_s = mu[s]

            def neg_log_post(z, s=s, mu_s=mu_s):
                theta = transform.to_natural(z)
                return -float(log_likelihood(theta, s)) - log_gauss_diag(z, mu_s, sigma)

            try:
                post = subject_map_estep(
                    neg_log_post,
                    z0=prev_z[s] if warm_start else mu_s,
                    prior_variance=sigma,
                    config=config,
                )
            except HierarchicalEMError as error:
                raise HierarchicalEMError(f"participant {s}: {error}") from None
            z_hat[s] = post.z_hat
            posterior[s] = post.covariance
            curvature[s] = post.curvature
            steps[s] = post.hessian_step
            subject_objective[s] = post.laplace_objective
            success[s] = post.success
            if not post.success:
                messages.append((s, post.message))

        return EStepResult(
            z_hat=z_hat,
            covariance=posterior,
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
    posterior_covariance: np.ndarray  # (n_subjects, n_params, n_params)
    objective: float          # Laplace marginal log-likelihood at the returned beta and sigma
    n_iter: int
    converged: bool
    subject_converged: np.ndarray  # (n_subjects,) bool, from the final E-step
    history: list             # one entry per iteration; see `fit_laplace_em`
    hessian_step: np.ndarray  # (n_subjects, n_params) steps used in the final E-step

    @property
    def variance(self):
        """Per-participant, per-parameter posterior variance."""
        return np.diagonal(self.posterior_covariance, axis1=1, axis2=2).copy()


def fit_laplace_em(
    estep_runner,
    n_subjects,
    n_params,
    *,
    design_matrix=None,
    estep_config=None,
    max_iterations=50,
    tol=1e-4,
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
    if max_iterations < 1:
        raise ValueError(f"max_iterations must be at least 1; got {max_iterations}")

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
        posterior_covariance=estep.covariance,
        objective=estep.objective,
        n_iter=len(history),
        converged=converged,
        subject_converged=estep.success,
        history=history,
        hessian_step=estep.hessian_step,
    )
