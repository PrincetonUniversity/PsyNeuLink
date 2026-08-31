"""Tests for hierarchical parameter estimation.

Layered fastest first.  This layer covers the parameter transforms and the per-participant E-step,
checked against a Gaussian model whose posterior is available in closed form.  It needs neither
PsyNeuLink nor a cluster, so it runs under a plain ``[dev]`` install.
"""

from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pandas as pd
import pytest

from psyneulink.core.compositions.hierarchical.laplaceem import (
    DEFAULT_HESSIAN_STEP_SCALE,
    EStepConfig,
    EStepResult,
    HierarchicalEMWarning,
    diagonal_hessian,
    fit_laplace_em,
    log_gauss_diag,
    make_inprocess_estep_runner,
    subject_map_estep,
)
from psyneulink.core.components.functions.nonstateful import fitfunctions
from psyneulink.core.compositions.hierarchical import distributedestep
from psyneulink.core.compositions.hierarchical.hierarchicalresults import (
    HierarchicalPECResults,
)
from psyneulink.core.compositions.hierarchical.subjectlikelihood import (
    PECFactorySubjectLikelihood,
    ParameterSchema,
    SubjectLikelihoodProvider,
    split_stacked_data,
)
from psyneulink.core.compositions.hierarchical.transforms import (
    BoundedTransform,
    IdentityTransform,
)


# ===========================================================================
# A Gaussian hierarchical model with closed-form answers.
#
# z_s     ~ N(beta_true, diag(sigma_true))        group random effect
# y_{s,i} ~ N(z_s, tau^2 I),  i = 1..n_obs        observations
#
# The per-participant likelihood is Gaussian in z, so the posterior under a
# Gaussian prior is conjugate and exact -- which is what lets the numerical
# E-step be checked against a known answer, with no PsyNeuLink involved.
# ===========================================================================
LOG_2PI = np.log(2.0 * np.pi)


@dataclass
class ToyHierarchicalModel:
    ybar: np.ndarray     # (n_subjects, n_params) per-participant means
    ss: np.ndarray       # (n_subjects,) within-participant sums of squares
    n_obs: int
    tau: float
    z_true: np.ndarray   # (n_subjects, n_params) ground truth

    @property
    def n_subjects(self):
        return self.ybar.shape[0]

    @property
    def n_params(self):
        return self.ybar.shape[1]

    @classmethod
    def generate(cls, n_subjects, n_params, beta_true, sigma_true, tau, n_obs, rng):
        beta_true = np.asarray(beta_true, float)
        sigma_true = np.asarray(sigma_true, float)
        z_true = rng.normal(beta_true, np.sqrt(sigma_true), size=(n_subjects, n_params))
        ybar = np.empty((n_subjects, n_params))
        ss = np.empty(n_subjects)
        for s in range(n_subjects):
            y = rng.normal(z_true[s], tau, size=(n_obs, n_params))
            ybar[s] = y.mean(axis=0)
            ss[s] = np.sum((y - ybar[s]) ** 2)
        return cls(ybar=ybar, ss=ss, n_obs=n_obs, tau=tau, z_true=z_true)

    def log_likelihood_s(self, theta, s):
        """Exact log-likelihood of participant ``s``'s data at ``theta`` (== z here)."""
        theta = np.asarray(theta, float)
        quad = self.ss[s] + self.n_obs * np.sum((theta - self.ybar[s]) ** 2)
        norm = 0.5 * self.n_obs * self.n_params * (LOG_2PI + 2.0 * np.log(self.tau))
        return -0.5 * quad / self.tau ** 2 - norm

    def closed_form_posterior(self, mu, sigma):
        """Exact posteriors ``N(z_hat, V)`` given prior means ``mu`` and variances ``sigma``."""
        mu = np.asarray(mu, float)
        sigma = np.asarray(sigma, float)
        lam_lik = self.n_obs / self.tau ** 2
        lam_prior = 1.0 / sigma
        lam_post = lam_lik + lam_prior
        z_hat = (lam_lik * self.ybar + lam_prior * mu) / lam_post
        variance = np.broadcast_to(1.0 / lam_post, self.ybar.shape).copy()
        return z_hat, variance


def _make_toy(seed=1, n_subjects=50, n_params=2, n_obs=25):
    rng = np.random.default_rng(seed)
    return ToyHierarchicalModel.generate(
        n_subjects, n_params, beta_true=[0.5, -1.0], sigma_true=[0.4, 0.9],
        tau=1.0, n_obs=n_obs, rng=rng,
    )


# ===========================================================================
# Transforms
# ===========================================================================
def test_bounded_transform_roundtrip():
    t = BoundedTransform(lower=[0.0, -2.0], upper=[1.0, 3.0])
    theta = np.array([0.2, 1.5])
    z = t.to_unconstrained(theta)
    assert np.allclose(t.to_natural(z), theta, atol=1e-10)


def test_bounded_transform_respects_bounds():
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    # Moderate z stays strictly interior; extreme z saturates within the closed box.
    interior = t.to_natural(np.array([-20.0, 0.0, 20.0]))
    assert np.all(interior > 0.0) and np.all(interior < 1.0)
    saturated = t.to_natural(np.array([-50.0, 50.0]))
    assert np.all(saturated >= 0.0) and np.all(saturated <= 1.0)
    assert saturated[0] < 1e-6 and saturated[1] > 1.0 - 1e-6


def test_bounded_transform_saturates_without_overflow():
    # Saturating inputs must reach the bounds without overflowing on the way.
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    with np.errstate(over="raise", under="raise"):
        out = t.to_natural(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 1.0


def test_dtheta_dz_matches_numerical():
    t = BoundedTransform(lower=[0.0, 1.0], upper=[2.0, 5.0])
    z = np.array([0.3, -0.7])
    h = 1e-6
    num = np.array([
        (t.to_natural(z + h * e)[k] - t.to_natural(z - h * e)[k]) / (2 * h)
        for k, e in enumerate(np.eye(2))
    ])
    assert np.allclose(t.dtheta_dz(z), num, rtol=1e-5)


def test_bounded_transform_rejects_bad_bounds():
    with pytest.raises(ValueError, match="upper bounds must exceed lower bounds"):
        BoundedTransform(lower=[0.0, 2.0], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        BoundedTransform(lower=[0.0, 1.0], upper=[1.0])


def test_identity_transform_is_the_identity():
    t = IdentityTransform()
    theta = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(t.to_natural(theta), theta)
    assert np.allclose(t.to_unconstrained(theta), theta)
    assert np.allclose(t.dtheta_dz(theta), np.ones_like(theta))


# ===========================================================================
# Finite-difference curvature
# ===========================================================================
def test_diagonal_hessian_of_quadratic():
    # f(z) = 0.5 * sum a_k (z_k - c_k)^2 has diagonal Hessian a.
    a = np.array([2.0, 5.0, 0.7])
    c = np.array([1.0, -3.0, 0.5])
    f = lambda z: 0.5 * np.sum(a * (z - c) ** 2)  # noqa: E731
    diag = diagonal_hessian(f, np.array([0.0, 0.0, 0.0]), step=1e-3)
    assert np.allclose(diag, a, rtol=1e-5)


def test_diagonal_hessian_accepts_per_dimension_step():
    a = np.array([2.0, 5.0])
    f = lambda z: 0.5 * np.sum(a * z ** 2)  # noqa: E731
    diag = diagonal_hessian(f, np.zeros(2), step=np.array([1e-3, 1e-2]))
    assert np.allclose(diag, a, rtol=1e-5)


def test_diagonal_hessian_rejects_bad_step():
    f = lambda z: float(np.sum(z ** 2))  # noqa: E731
    with pytest.raises(ValueError, match="step must be scalar or of shape"):
        diagonal_hessian(f, np.zeros(3), step=np.array([1e-3, 1e-3]))
    with pytest.raises(ValueError, match="step must be positive"):
        diagonal_hessian(f, np.zeros(2), step=0.0)


# ===========================================================================
# E-step against the closed-form posterior
# ===========================================================================
def test_estep_matches_closed_form_posterior():
    model = _make_toy()
    beta = np.array([0.3, -0.8])
    sigma = np.array([0.5, 0.7])
    s = 7

    def neg_log_post(z):
        return -model.log_likelihood_s(z, s) - log_gauss_diag(z, beta, sigma)

    post = subject_map_estep(neg_log_post, z0=beta, prior_variance=sigma)
    z_cf, v_cf = model.closed_form_posterior(np.tile(beta, (model.n_subjects, 1)), sigma)
    assert np.allclose(post.z_hat, z_cf[s], atol=1e-5)
    assert np.allclose(post.variance, v_cf[s], rtol=1e-4)


def test_estep_reports_optimizer_outcome():
    # The optimizer's verdict must reach the caller: a participant whose fit failed
    # cannot be allowed to feed the group update silently.
    model = _make_toy()
    sigma = np.array([0.5, 0.7])

    def neg_log_post(z):
        return -model.log_likelihood_s(z, 0) - log_gauss_diag(z, np.zeros(2), sigma)

    ok = subject_map_estep(neg_log_post, z0=np.zeros(2), prior_variance=sigma)
    assert ok.success is True
    assert isinstance(ok.message, str)

    capped = subject_map_estep(
        neg_log_post, z0=np.zeros(2), prior_variance=sigma,
        config=EStepConfig(optimizer_options={"maxiter": 1}),
    )
    assert capped.success is False


def test_estep_falls_back_to_prior_where_data_are_uninformative():
    # A flat objective has no curvature to invert; the posterior should report the
    # prior rather than a spuriously tight interval.
    sigma = np.array([0.25, 4.0])
    post = subject_map_estep(lambda z: 0.0, z0=np.zeros(2), prior_variance=sigma)
    assert np.allclose(post.variance, sigma)


# ===========================================================================
# The finite-difference step is derived, and recorded
# ===========================================================================
def test_hessian_step_is_derived_from_the_prior_variance():
    sigma = np.array([0.36, 0.04])
    step = EStepConfig().resolve_hessian_step(sigma)
    assert np.allclose(step, DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(sigma))
    # Pin a concrete value, so the rule is checked against arithmetic and not
    # merely against its own implementation.
    assert np.isclose(step[0], 0.15)


def test_explicit_hessian_step_overrides_the_derived_one():
    sigma = np.array([0.36, 0.04])
    assert np.allclose(EStepConfig(hessian_step=1e-3).resolve_hessian_step(sigma), 1e-3)
    per_dim = np.array([1e-3, 1e-2])
    assert np.allclose(EStepConfig(hessian_step=per_dim).resolve_hessian_step(sigma), per_dim)


def test_estep_records_the_step_it_used():
    # Recorded so that a fit can be audited, rather than the step being an
    # invisible choice made inside the E-step.
    sigma = np.array([0.36, 0.04])
    post = subject_map_estep(lambda z: float(np.sum(z ** 2)), z0=np.zeros(2), prior_variance=sigma)
    assert np.allclose(post.hessian_step, DEFAULT_HESSIAN_STEP_SCALE * np.sqrt(sigma))


def test_estep_config_is_immutable():
    # One config object is shared by every participant and by the group update, so it
    # must not be mutable from any of them.
    config = EStepConfig()
    with pytest.raises(FrozenInstanceError):
        config.variance_floor = 1.0


# ===========================================================================
# EM driver
# ===========================================================================
def _closed_form_em(model, max_iter=500, tol=1e-12):
    """Exact EM for the toy model, using its conjugate posterior instead of an optimizer."""
    beta = np.zeros(model.n_params)
    sigma = np.ones(model.n_params)
    for _ in range(max_iter):
        mu = np.tile(beta, (model.n_subjects, 1))
        z_hat, variance = model.closed_form_posterior(mu, sigma)
        beta_new = z_hat.mean(axis=0)
        sigma_new = np.mean((z_hat - beta_new) ** 2 + variance, axis=0)
        converged = max(
            np.max(np.abs(beta_new - beta)), np.max(np.abs(sigma_new - sigma))
        ) < tol
        beta, sigma = beta_new, sigma_new
        if converged:
            break
    return beta, sigma


def _fit_toy(model, **kwargs):
    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    kwargs.setdefault("max_iterations", 300)
    kwargs.setdefault("tol", 1e-8)
    return fit_laplace_em(runner, model.n_subjects, model.n_params, **kwargs)


def test_em_matches_closed_form_em():
    model = _make_toy(seed=3, n_subjects=120)
    beta_cf, sigma_cf = _closed_form_em(model)
    result = _fit_toy(model)
    assert np.allclose(result.beta.ravel(), beta_cf, atol=5e-3)
    assert np.allclose(result.sigma, sigma_cf, atol=5e-3)


def test_em_beta_equals_mean_ybar_at_convergence():
    # With an intercept-only design the group mean is the mean of the per-participant data means.
    model = _make_toy(seed=4, n_subjects=150)
    result = _fit_toy(model)
    assert np.allclose(result.beta.ravel(), model.ybar.mean(axis=0), atol=5e-3)


def test_em_recovers_ground_truth():
    model = _make_toy(seed=5, n_subjects=400, n_obs=40)
    result = _fit_toy(model, tol=1e-7)
    assert np.allclose(result.beta.ravel(), [0.5, -1.0], atol=0.15)
    assert np.allclose(result.sigma, [0.4, 0.9], atol=0.2)


def test_em_objective_nondecreasing():
    # Holds for an exact E-step; not guaranteed for an approximate one.
    model = _make_toy(seed=6, n_subjects=100)
    result = _fit_toy(model, max_iterations=100)
    obj = np.array([h["objective"] for h in result.history])
    assert np.all(np.diff(obj) > -1e-6)


def test_em_history_pairs_each_objective_with_the_estimate_that_produced_it():
    # A history entry must describe one consistent state: the group estimate used for
    # that iteration's E-step, and the objective that E-step returned. The first entry
    # therefore carries the initial estimate, not the result of the first update.
    model = _make_toy(seed=7, n_subjects=40)
    init_beta = np.array([[0.11, -0.22]])
    init_sigma = np.array([0.7, 1.3])
    result = _fit_toy(model, max_iterations=3, tol=0.0,
                      init_beta=init_beta, init_sigma=init_sigma)

    assert np.allclose(result.history[0]["beta"], init_beta)
    assert np.allclose(result.history[0]["sigma"], init_sigma)
    # Each subsequent entry carries the estimate the previous entry's update produced.
    assert not np.allclose(result.history[1]["beta"], init_beta)


def test_em_result_describes_the_group_estimate_it_returns():
    # The participant-level results must correspond to the returned group estimate,
    # not to the one from the iteration before it.
    model = _make_toy(seed=8, n_subjects=30)
    result = _fit_toy(model, max_iterations=4, tol=0.0)

    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    recomputed = runner(
        np.ones((model.n_subjects, 1)) @ result.beta, result.sigma, result.z_hat, True
    )
    assert np.allclose(recomputed.z_hat, result.z_hat, atol=1e-6)
    assert np.isclose(recomputed.objective, result.objective, rtol=1e-10)


def test_em_warns_and_records_when_a_participant_fails():
    model = _make_toy(seed=9, n_subjects=6)
    base = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())

    def failing_runner(mu, sigma, prev_z, warm_start):
        out = base(mu, sigma, prev_z, warm_start)
        out.success[1] = False
        out.messages = ((1, "did not converge"),)
        return out

    with pytest.warns(HierarchicalEMWarning, match="did not converge"):
        result = fit_laplace_em(failing_runner, model.n_subjects, model.n_params,
                                max_iterations=2, tol=0.0)

    assert result.subject_converged[1] is np.False_ or not result.subject_converged[1]
    assert result.history[0]["n_subject_failures"] == 1


def test_em_objective_is_summed_in_participant_order():
    # Independent of completion order, so an in-process and a distributed E-step agree.
    r = EStepResult(
        z_hat=np.zeros((3, 2)), variance=np.ones((3, 2)), curvature=np.ones((3, 2)),
        hessian_step=np.ones((3, 2)), subject_objective=np.array([1.5, -2.0, 0.25]),
        success=np.ones(3, dtype=bool), messages=(),
    )
    assert np.isclose(r.objective, float(np.sum(np.array([1.5, -2.0, 0.25]))))


def test_em_rejects_mismatched_design_matrix():
    model = _make_toy(seed=10, n_subjects=8)
    runner = make_inprocess_estep_runner(model.log_likelihood_s, IdentityTransform())
    with pytest.raises(ValueError, match="one row per participant"):
        fit_laplace_em(runner, model.n_subjects, model.n_params,
                       design_matrix=np.ones((3, 1)), max_iterations=1)


# ===========================================================================
# The seam between the driver and real models
# ===========================================================================
class _StubOptimizationFunction:
    def __init__(self, names, bounds):
        self.fit_param_names = list(names)
        self.fit_param_bounds = {n: (lo, hi, 0.1) for n, (lo, hi) in zip(names, bounds)}


class _StubController:
    def __init__(self, names, bounds):
        self.function = _StubOptimizationFunction(names, bounds)


class _StubPEC:
    """Stands in for a ParameterEstimationComposition: just enough surface for the seam."""

    def __init__(self, names, bounds, value=-1.0):
        self.controller = _StubController(names, bounds)
        self._value = value
        self.calls = []

    def log_likelihood(self, *theta, inputs=None):
        self.calls.append(np.asarray(theta, dtype=float))
        return self._value


def _stacked_frame():
    return pd.DataFrame({
        "subject": ["b", "b", "a", "a", "a", "c"],
        "decision": [1, 0, 1, 1, 0, 0],
        "response_time": [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    })


def test_split_orders_participants_by_first_appearance():
    # Not sorted: sorting would silently reorder results if participants were relabelled.
    split = split_stacked_data(_stacked_frame(), "subject")
    assert split.labels == ("b", "a", "c")
    assert split.n_subjects == 3


def test_split_gives_each_participant_their_own_rows():
    split = split_stacked_data(_stacked_frame(), "subject")
    assert [len(f) for f in split.frames] == [2, 3, 1]
    assert list(split.frames[1]["response_time"]) == [0.6, 0.7, 0.8]


def test_split_drops_the_participant_column_and_resets_the_index():
    split = split_stacked_data(_stacked_frame(), "subject")
    for frame in split.frames:
        assert "subject" not in frame.columns
        assert list(frame.index) == list(range(len(frame)))


def test_split_preserves_categorical_columns():
    data = _stacked_frame()
    data["decision"] = data["decision"].astype("category")
    split = split_stacked_data(data, "subject")
    assert all(isinstance(f["decision"].dtype, pd.CategoricalDtype) for f in split.frames)


def test_split_rejects_unusable_data():
    with pytest.raises(ValueError, match="pandas DataFrame"):
        split_stacked_data(np.zeros((4, 2)), "subject")
    with pytest.raises(ValueError, match="not a column of data"):
        split_stacked_data(_stacked_frame(), "participant")
    with pytest.raises(ValueError, match="at least two participants"):
        split_stacked_data(pd.DataFrame({"subject": ["a", "a"], "rt": [0.1, 0.2]}), "subject")


def test_split_rejects_trials_with_no_participant_identifier():
    # A missing identifier matches no row, so such a trial would belong to no participant and
    # leave the fit silently.
    data = pd.DataFrame({"subject": ["a", "a", np.nan, "b"], "rt": [0.1, 0.2, 0.3, 0.4]})
    with pytest.raises(ValueError, match="no participant identifier on 1 of 4 rows"):
        split_stacked_data(data, "subject")


def test_provider_reads_names_and_bounds_from_the_model():
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    factory = lambda data, subject_index=None: (  # noqa: E731
        _StubPEC(["rate", "threshold"], [(-1.5, 1.5), (0.3, 1.5)]), None
    )
    provider = PECFactorySubjectLikelihood(factory, frames)
    assert provider.fit_param_names == ("rate", "threshold")
    assert provider.n_params == 2 and provider.n_subjects == 3
    lower, upper = provider.bounds
    assert np.allclose(lower, [-1.5, 0.3]) and np.allclose(upper, [1.5, 1.5])
    assert np.allclose(provider.transform.to_natural([0.0, 0.0]), [0.0, 0.9])


def test_provider_builds_each_model_once_and_routes_by_participant():
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    built = []

    def factory(data, subject_index=None):
        pec = _StubPEC(["rate"], [(-1.0, 1.0)], value=float(subject_index))
        built.append((subject_index, len(data)))
        return pec, None

    provider = PECFactorySubjectLikelihood(factory, frames)
    for _ in range(3):
        assert provider.log_likelihood([0.1], 1) == 1.0
    assert provider.log_likelihood([0.1], 2) == 2.0
    # One build per participant used, regardless of how many evaluations they receive.
    assert sorted(built) == [(1, 3), (2, 1)]


def test_provider_passes_the_participants_own_rows_to_the_factory():
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    seen = {}

    def factory(data, subject_index=None):
        seen[subject_index] = list(data["response_time"])
        return _StubPEC(["rate"], [(-1.0, 1.0)]), None

    provider = PECFactorySubjectLikelihood(factory, frames)
    provider.log_likelihood([0.0], 1)
    assert seen[1] == [0.6, 0.7, 0.8]


def test_provider_rejects_models_that_disagree_about_the_search_range():
    # The group prior is defined in terms of these ranges, so they cannot vary by participant.
    frames = split_stacked_data(_stacked_frame(), "subject").frames

    def factory(data, subject_index=None):
        bounds = [(-1.0, 1.0)] if subject_index == 0 else [(-2.0, 2.0)]
        return _StubPEC(["rate"], bounds), None

    provider = PECFactorySubjectLikelihood(factory, frames)
    provider.log_likelihood([0.0], 0)
    with pytest.raises(ValueError, match="searches ranges"):
        provider.log_likelihood([0.0], 1)


def test_provider_tolerates_per_instance_parameter_names():
    # A model reports parameters as "<mechanism>.<parameter>", and the mechanism carries a
    # number assigned in construction order. Building one model per participant therefore
    # yields DDM-6.rate for one and DDM-7.rate for the next, for the same parameter.
    frames = split_stacked_data(_stacked_frame(), "subject").frames

    def factory(data, subject_index=None):
        prefix = f"DDM-{6 + subject_index}"
        names = [f"{prefix}.rate", f"{prefix}.threshold"]
        return _StubPEC(names, [(-1.5, 1.5), (0.3, 1.5)]), None

    provider = PECFactorySubjectLikelihood(factory, frames)
    for s in range(3):
        provider.log_likelihood([0.0, 0.9], s)
    # Reported without the instance qualifier, which is meaningless across participants.
    assert provider.fit_param_names == ("rate", "threshold")


def test_provider_rejects_models_that_disagree_about_the_parameters():
    frames = split_stacked_data(_stacked_frame(), "subject").frames

    def factory(data, subject_index=None):
        names = ["rate"] if subject_index == 0 else ["threshold"]
        return _StubPEC(names, [(-1.0, 1.0)]), None

    provider = PECFactorySubjectLikelihood(factory, frames)
    provider.log_likelihood([0.0], 0)
    with pytest.raises(ValueError, match="must fit the same"):
        provider.log_likelihood([0.0], 1)


def test_provider_keeps_full_names_when_stripping_them_would_collide():
    # Two mechanisms fitting the same parameter both strip to "rate", which would leave one name
    # for two parameters and lose one of them wherever results are keyed by name.
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    factory = lambda data, subject_index=None: (  # noqa: E731
        _StubPEC(["left.rate", "right.rate"], [(-1.0, 1.0), (-2.0, 2.0)]), None
    )
    provider = PECFactorySubjectLikelihood(factory, frames)
    assert provider.fit_param_names == ("left.rate", "right.rate")
    assert len(set(provider.fit_param_names)) == 2


def test_provider_holds_models_to_a_supplied_schema():
    # The model the fit was configured on is authoritative; a factory that fits something else is
    # rejected rather than quietly redefining what is being estimated.
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    schema = ParameterSchema.from_pec(
        _StubPEC(["rate"], [(-1.5, 1.5)]), source="the model given to the fit"
    )

    factory = lambda data, subject_index=None: (  # noqa: E731
        _StubPEC(["threshold"], [(10.0, 20.0)]), None
    )
    provider = PECFactorySubjectLikelihood(factory, frames, schema=schema)
    assert provider.fit_param_names == ("rate",)
    with pytest.raises(ValueError, match="the model given to the fit fits"):
        provider.log_likelihood([0.0], 0)


def test_provider_reports_a_supplied_schema_without_building_a_model():
    # Nothing needs to be built to know what is being fitted, which is what lets a distributed
    # fit build every model on a worker.
    frames = split_stacked_data(_stacked_frame(), "subject").frames
    schema = ParameterSchema.from_pec(
        _StubPEC(["rate", "threshold"], [(-1.5, 1.5), (0.3, 1.5)]), source="the fit"
    )

    def factory(data, subject_index=None):
        raise AssertionError("no model should be built here")

    provider = PECFactorySubjectLikelihood(factory, frames, schema=schema)
    assert provider.fit_param_names == ("rate", "threshold")
    assert provider.n_params == 2
    assert np.allclose(provider.bounds[0], [-1.5, 0.3])


def test_driver_fits_through_the_provider_interface_alone():
    # The driver reaches the model only through log_likelihood, so a provider backed by
    # something other than a composition drives it unchanged.
    model = _make_toy(seed=11, n_subjects=20)

    class _ToyProvider(SubjectLikelihoodProvider):
        n_subjects = model.n_subjects
        n_params = model.n_params
        fit_param_names = ("a", "b")

        def log_likelihood(self, theta, subject_index):
            return model.log_likelihood_s(theta, subject_index)

    provider = _ToyProvider()
    runner = make_inprocess_estep_runner(provider.log_likelihood, IdentityTransform())
    result = fit_laplace_em(runner, provider.n_subjects, provider.n_params, max_iterations=25)
    assert np.allclose(result.beta.ravel(), model.ybar.mean(axis=0), atol=0.05)


# ===========================================================================
# Results in the model's own units
# ===========================================================================
def _fit_toy_for_results(n_subjects=12, seed=21):
    model = _make_toy(seed=seed, n_subjects=n_subjects)
    transform = BoundedTransform(lower=[-4.0, -4.0], upper=[4.0, 4.0])
    runner = make_inprocess_estep_runner(model.log_likelihood_s, transform)
    em = fit_laplace_em(runner, model.n_subjects, model.n_params, max_iterations=15)
    labels = [f"S{i:02d}" for i in range(model.n_subjects)]
    return em, transform, labels


def test_results_shapes_and_labels():
    em, transform, labels = _fit_toy_for_results()
    res = HierarchicalPECResults.from_em(em, transform, ("a", "b"), labels)
    assert res.fit_param_names == ("a", "b")
    assert res.subject_labels == tuple(labels)
    assert res.subject_parameters.shape == (len(labels), 2)
    assert list(res.subject_parameters.index) == labels
    assert len(res.subject_posteriors) == len(labels) * 2
    assert list(res.group_parameters.index) == ["a", "b"]


def test_results_convert_estimates_into_the_models_units():
    em, transform, labels = _fit_toy_for_results()
    res = HierarchicalPECResults.from_em(em, transform, ("a", "b"), labels)
    expected = np.vstack([transform.to_natural(z) for z in res.z_hat])
    assert np.allclose(res.subject_parameters.to_numpy(), expected)
    # The group value is the transformed group mean, i.e. a median (see module docstring).
    assert np.allclose(res.group_parameters["value"].to_numpy(), transform.to_natural(res.beta[0]))


def test_results_uncertainty_uses_the_delta_method():
    em, transform, labels = _fit_toy_for_results()
    res = HierarchicalPECResults.from_em(em, transform, ("a", "b"), labels)
    slope = np.vstack([transform.dtheta_dz(z) for z in res.z_hat])
    expected = np.abs(slope) * np.sqrt(res.posterior_variance)
    got = res.subject_posteriors["theta_sd"].to_numpy().reshape(len(labels), 2)
    assert np.allclose(got, expected)


def test_results_history_pairs_objective_with_its_own_estimate():
    em, transform, labels = _fit_toy_for_results()
    res = HierarchicalPECResults.from_em(em, transform, ("a", "b"), labels)
    assert list(res.em_history["iter"]) == list(range(len(em.history)))
    assert {"objective", "delta", "n_subject_failures", "beta_a", "sigma_b"} <= set(
        res.em_history.columns
    )
    # The first row holds the starting estimate, not the one the first update produced.
    assert np.isclose(res.em_history["beta_a"].iloc[0], em.history[0]["beta"][0][0])


def test_results_record_the_transform_and_settings():
    em, transform, labels = _fit_toy_for_results()
    res = HierarchicalPECResults.from_em(
        em, transform, ("a", "b"), labels, settings={"max_iterations": 15}
    )
    assert res.transform_metadata["kind"] == "BoundedTransform"
    assert res.transform_metadata["lower"] == [-4.0, -4.0]
    assert res.settings["max_iterations"] == 15


def test_results_repr_surfaces_non_convergence():
    em, transform, labels = _fit_toy_for_results()
    em.converged = False
    em.subject_converged = np.zeros(len(labels), dtype=bool)
    res = HierarchicalPECResults.from_em(em, transform, ("a", "b"), labels)
    text = repr(res)
    assert "stopped at the iteration limit" in text
    assert "participant fit(s) did not converge" in text


# ===========================================================================
# The distributed E-step's worker task, exercised in this process.
#
# The task falls back to a module-level cache when there is no Dask worker
# context, so its caching, locking and thread setup can be checked without a
# cluster. The cluster itself is covered by the integration tests.
# ===========================================================================
@pytest.fixture
def clear_subject_cache():
    distributedestep._SUBJECT_FALLBACK_CACHE.clear()
    yield
    distributedestep._SUBJECT_FALLBACK_CACHE.clear()


def _dask_task(factory, subject_index, data, fit_id, worker_cores=None, sigma=None, schema=None):
    sigma = np.array([1.0]) if sigma is None else sigma
    if schema is None:
        schema = ParameterSchema.from_pec(_StubPEC(["rate"], [(-1.0, 1.0)]), source="the fit")
    return distributedestep._dask_subject_estep(
        factory, subject_index, data, np.zeros(1), sigma,
        schema, np.zeros(1), worker_cores, fit_id, EStepConfig(),
    )


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_builds_each_participant_once_per_fit():
    builds = []

    def factory(data, subject_index=None):
        builds.append(subject_index)
        return _StubPEC(["rate"], [(-1.0, 1.0)]), None

    frame = pd.DataFrame({"rt": [0.1]})
    for _ in range(3):
        _dask_task(factory, 0, frame, "fit-a")
    assert builds == [0]
    # A different participant is a different model.
    _dask_task(factory, 1, frame, "fit-a")
    assert builds == [0, 1]


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_rebuilds_for_a_new_fit():
    builds = []

    def factory(data, subject_index=None):
        builds.append(subject_index)
        return _StubPEC(["rate"], [(-1.0, 1.0)]), None

    frame = pd.DataFrame({"rt": [0.1]})
    _dask_task(factory, 0, frame, "fit-a")
    _dask_task(factory, 0, frame, "fit-a")
    _dask_task(factory, 0, frame, "fit-b")
    assert builds == [0, 0]


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_sets_thread_count_before_building(monkeypatch):
    seen = []
    import psyneulink.core.globals.threads as threads_module
    monkeypatch.setattr(threads_module, "set_num_threads", lambda n: seen.append(n))

    factory = lambda data, subject_index=None: (_StubPEC(["rate"], [(-1.0, 1.0)]), None)  # noqa: E731
    _dask_task(factory, 0, pd.DataFrame({"rt": [0.1]}), "fit-a", worker_cores=3)
    assert seen == [3]


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_holds_the_evaluation_lock_for_the_whole_call(monkeypatch):
    # Two compiled models driven at once in one process is what the lock prevents,
    # so it must cover the fit, not only the model construction.
    events = []

    class _RecordingLock:
        def __enter__(self):
            events.append("enter")

        def __exit__(self, *exc):
            events.append("exit")
            return False

    monkeypatch.setattr(fitfunctions, "_PEC_EVALUATION_LOCK", _RecordingLock())

    def factory(data, subject_index=None):
        events.append("build")
        return _StubPEC(["rate"], [(-1.0, 1.0)]), None

    _dask_task(factory, 0, pd.DataFrame({"rt": [0.1]}), "fit-a")
    assert events[0] == "enter" and events[-1] == "exit"
    assert "build" in events


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_returns_the_participant_index_and_posterior():
    factory = lambda data, subject_index=None: (_StubPEC(["rate"], [(-1.0, 1.0)]), None)  # noqa: E731
    index, post, address = _dask_task(factory, 4, pd.DataFrame({"rt": [0.1]}), "fit-a")
    assert index == 4
    assert isinstance(post.success, bool)
    assert post.z_hat.shape == (1,)
    # No worker context in-process, so no address to pin to.
    assert address is None


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_holds_its_model_to_the_schema():
    # The in-process runner checks each model as it builds it; a worker has to do the same, or a
    # participant fitting different parameters would be handed theta in someone else's order.
    frame = pd.DataFrame({"rt": [0.1]})
    factory = lambda data, subject_index=None: (  # noqa: E731
        _StubPEC(["threshold"], [(0.3, 1.5)]), None
    )
    with pytest.raises(ValueError, match="must fit the same"):
        _dask_task(factory, 2, frame, "fit-a")


@pytest.mark.usefixtures("clear_subject_cache")
def test_worker_checks_the_search_range_before_scoring():
    frame = pd.DataFrame({"rt": [0.1]})
    factory = lambda data, subject_index=None: (  # noqa: E731
        _StubPEC(["rate"], [(-9.0, 9.0)]), None
    )
    with pytest.raises(ValueError, match="searches ranges"):
        _dask_task(factory, 0, frame, "fit-a")


@pytest.mark.usefixtures("clear_subject_cache")
def test_releasing_a_fit_drops_only_that_fits_models():
    # Workers on a cluster the user supplied outlive the fit, so its models have to go explicitly.
    frame = pd.DataFrame({"rt": [0.1]})
    factory = lambda data, subject_index=None: (_StubPEC(["rate"], [(-1.0, 1.0)]), None)  # noqa: E731
    _dask_task(factory, 0, frame, "fit-a")
    _dask_task(factory, 1, frame, "fit-a")
    _dask_task(factory, 0, frame, "fit-b")
    assert len(distributedestep._SUBJECT_FALLBACK_CACHE) == 3

    distributedestep._release_fit_models("fit-a")
    assert list(distributedestep._SUBJECT_FALLBACK_CACHE) == [("fit-b", 0)]


# ===========================================================================
# Configuring a hierarchical fit on ParameterEstimationComposition
# ===========================================================================
def _group_frame(n_subjects=3, n_trials=4):
    rows = []
    for s in range(n_subjects):
        for t in range(n_trials):
            rows.append({"subject": f"S{s}", "decision": t % 2, "response_time": 0.4 + 0.01 * t})
    return pd.DataFrame(rows)


def _build_group_pec(**overrides):
    """A PEC configured for hierarchical fitting, without running it."""
    import psyneulink as pnl
    from psyneulink.core.components.functions.nonstateful.fitfunctions import (
        PECOptimizationFunction,
    )

    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=0.3, noise=1.0, threshold=0.6,
            non_decision_time=0.15, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    kwargs = dict(
        name="group_pec",
        nodes=[comp],
        parameters={("rate", decision): np.linspace(-1.5, 1.5, 1000)},
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=_group_frame(),
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1),
        fit_method="hierarchical",
        hierarchical_options={"subject_id": "subject"},
    )
    kwargs.update(overrides)
    return pnl.ParameterEstimationComposition(**kwargs)


@pytest.mark.composition
def test_pec_splits_participants_and_hides_the_column():
    pec = _build_group_pec()
    # The remaining columns are the outcome variables, which is what the existing
    # data validation expects.
    assert list(pec.data.columns) == ["decision", "response_time"]
    assert list(pec.hierarchical_data.columns) == ["subject", "decision", "response_time"]
    assert pec._subject_split.labels == ("S0", "S1", "S2")


@pytest.mark.composition
def test_pec_rejects_unknown_hierarchical_options():
    with pytest.raises(Exception, match="unknown hierarchical_options"):
        _build_group_pec(hierarchical_options={"subject_id": "subject", "max_iters": 5})


@pytest.mark.composition
def test_pec_requires_a_subject_column():
    with pytest.raises(Exception, match="subject_id"):
        _build_group_pec(hierarchical_options={})
    with pytest.raises(Exception, match="not a column of data"):
        _build_group_pec(hierarchical_options={"subject_id": "participant"})


@pytest.mark.composition
def test_pec_rejects_out_of_range_hierarchical_options():
    for opts, match in [
        ({"subject_id": "subject", "max_iterations": 0}, "max_iterations"),
        ({"subject_id": "subject", "tol": 0.0}, "tol"),
        ({"subject_id": "subject", "damping": 1.0}, "damping"),
        ({"subject_id": "subject", "variance_floor": 0.0}, "variance_floor"),
    ]:
        with pytest.raises(Exception, match=match):
            _build_group_pec(hierarchical_options=opts)


@pytest.mark.composition
def test_pec_rejects_a_likelihood_include_mask():
    # The mask indexes the stacked table, but each participant is scored by its own model built
    # from its own slice, so there is nowhere for it to be applied.
    mask = np.ones(len(_group_frame()), dtype=bool)
    mask[0] = False
    with pytest.raises(Exception, match="likelihood_include_mask is not supported"):
        _build_group_pec(likelihood_include_mask=mask)


@pytest.mark.composition
def test_pec_requires_a_factory_to_run():
    # A Composition cannot be copied, so each participant's model has to be built.
    pec = _build_group_pec()
    with pytest.raises(Exception, match="pec_factory"):
        pec.run()


@pytest.mark.composition
def test_pec_log_likelihood_refuses_in_hierarchical_mode():
    # Scoring the stacked table as one participant would silently pool it.
    pec = _build_group_pec()
    with pytest.raises(Exception, match="silently pool"):
        pec.log_likelihood(0.3)


@pytest.mark.composition
def test_pec_uses_the_cluster_when_distributed(monkeypatch):
    # `distributed` selects where participants are fitted; the group update is unaffected.
    pytest.importorskip("dask.distributed")
    from psyneulink.core.compositions import parameterestimationcomposition as pec_module

    calls = {"distributed": 0}
    real = pec_module.make_distributed_estep_runner

    def spy(*args, **kwargs):
        calls["distributed"] += 1
        return real(*args, **kwargs)

    # Patched on the module that calls it, which is where the name is resolved.
    monkeypatch.setattr(pec_module, "make_distributed_estep_runner", spy)

    def factory(data, subject_index=None):
        return _StubPEC(["DDM-1.rate"], [(-1.5, 1.5)], value=-1.0), None

    pec = _build_group_pec(
        distributed=True,
        distributed_options={"pec_factory": factory, "n_workers": 1},
        hierarchical_options={"subject_id": "subject", "max_iterations": 1},
    )
    pec.run()
    assert calls["distributed"] == 1


@pytest.mark.composition
def test_pec_stays_in_process_by_default():
    from psyneulink.core.compositions.hierarchical import distributedestep

    def factory(data, subject_index=None):
        return _StubPEC(["DDM-1.rate"], [(-1.5, 1.5)], value=-1.0), None

    pec = _build_group_pec(
        distributed_options={"pec_factory": factory},
        hierarchical_options={"subject_id": "subject", "max_iterations": 1},
    )
    called = []
    original = distributedestep.make_distributed_estep_runner
    distributedestep.make_distributed_estep_runner = lambda *a, **k: called.append(1)
    try:
        results = pec.run()
    finally:
        distributedestep.make_distributed_estep_runner = original
    assert called == []
    assert results.subject_parameters.shape[0] == 3
