"""Numerical contracts for the research-local CSI direct likelihood."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest
import torch


CSI_FIT_DIRECTORY = (
    Path(__file__).resolve().parents[3]
    / "Scripts"
    / "Debug"
    / "pec_batch_compile"
    / "csi_fit"
)
sys.path.insert(0, str(CSI_FIT_DIRECTORY))

from direct_likelihood import (  # noqa: E402
    CSITrialData,
    ContinuousCSILikelihood,
    ContinuousCSIParameters,
    EndpointCrossingDDMSolver,
    MovingBoundaryDDMSolver,
    SolverConfig,
    PrescribedDDMCase,
    simulate_sequential_trials,
    simulate_prescribed_first_passage,
    wiener_choice_density,
)
from direct_likelihood.solver import (  # noqa: E402
    differentiable_tridiagonal_solve,
    solve_tridiagonal_pcr,
)
from direct_likelihood.validation import pnl_lca_endpoint  # noqa: E402
from direct_likelihood.fit import _feasible_start, fit_lbfgsb  # noqa: E402
from direct_likelihood.model import ContinuousLCA, pnl_euler_lca_step  # noqa: E402
from direct_likelihood.model import parameter_bounds, parameter_names  # noqa: E402
from direct_likelihood.native import native_lca_available  # noqa: E402
from direct_likelihood.recovery_summary import (  # noqa: E402
    summarize_recovery_results,
)
import direct_likelihood.fit as fit_module  # noqa: E402


pytestmark = [pytest.mark.composition]


def _trials(*, first_rt=0.55, second_rt=0.62, first_include=True):
    dtype = torch.float64
    return CSITrialData(
        task=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=dtype),
        stimulus=torch.tensor(
            [[1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]], dtype=dtype
        ),
        correct_response=torch.tensor([1.0, -1.0], dtype=dtype),
        choice=torch.tensor([1.0, 0.0], dtype=dtype),
        response_time=torch.tensor([first_rt, second_rt], dtype=dtype),
        condition_index=torch.tensor([0, 1], dtype=torch.long),
        is_switch=torch.tensor([0.0, 1.0], dtype=dtype),
        include=torch.tensor([first_include, True]),
        row_id=torch.tensor([12, 13], dtype=torch.long),
        subject_nr=4,
        rt_resolution=0.004,
    )


def test_parallel_cyclic_reduction_matches_dense_solve():
    generator = torch.Generator().manual_seed(2)
    lower = 0.1 * torch.rand(4, 63, generator=generator, dtype=torch.float64)
    upper = 0.1 * torch.rand(4, 63, generator=generator, dtype=torch.float64)
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    diagonal = 2.0 * torch.ones_like(lower)
    rhs = torch.randn(4, 63, generator=generator, dtype=torch.float64)
    matrix = (
        torch.diag_embed(diagonal)
        + torch.diag_embed(lower[:, 1:], offset=-1)
        + torch.diag_embed(upper[:, :-1], offset=1)
    )
    expected = torch.linalg.solve(matrix, rhs[..., None])[..., 0]
    actual = solve_tridiagonal_pcr(lower, diagonal, upper, rhs)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


def test_implicit_tridiagonal_backward_passes_gradcheck():
    generator = torch.Generator().manual_seed(9)
    lower = 0.03 * torch.rand(2, 7, generator=generator, dtype=torch.float64)
    upper = 0.03 * torch.rand(2, 7, generator=generator, dtype=torch.float64)
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0
    arguments = tuple(
        value.requires_grad_()
        for value in (
            lower,
            1.5 * torch.ones_like(lower),
            upper,
            torch.randn(2, 7, generator=generator, dtype=torch.float64),
        )
    )
    assert torch.autograd.gradcheck(
        differentiable_tridiagonal_solve,
        arguments,
        rtol=1e-5,
        atol=1e-6,
    )


def test_fixed_boundary_flux_matches_wiener_density_and_conserves_mass():
    dtype = torch.float64
    time_step = 0.001
    observation_time = 0.4
    observation_width = 0.004
    steps = int((observation_time + observation_width / 2) / time_step) + 1
    solver = MovingBoundaryDDMSolver(
        time_step=time_step, spatial_points=65, noise=0.1
    )
    result = solver.solve_observation_batch(
        drift=torch.full((2, steps), 0.03, dtype=dtype),
        threshold=torch.full((2,), 0.12, dtype=dtype),
        collapse_rate=torch.zeros(2, dtype=dtype),
        interval_low=torch.full(
            (2,), observation_time - observation_width / 2, dtype=dtype
        ),
        interval_high=torch.full(
            (2,), observation_time + observation_width / 2, dtype=dtype
        ),
        choice=torch.tensor([1.0, 0.0], dtype=dtype),
    )
    expected = torch.stack(
        [
            wiener_choice_density(
                torch.tensor(observation_time, dtype=dtype),
                drift=0.03,
                boundary=0.12,
                noise=0.1,
                upper=upper,
            )
            for upper in (True, False)
        ]
    ) * observation_width
    torch.testing.assert_close(result.probability, expected, rtol=0.002, atol=0.0)
    assert torch.amax(result.mass_error) < 1e-10


def test_endpoint_crossing_solver_matches_one_step_gaussian_tails():
    dtype = torch.float64
    time_step = 0.01
    drift = 0.03
    threshold = 0.015
    noise = 0.1
    solver = EndpointCrossingDDMSolver(
        time_step=time_step,
        spatial_points=1023,
        noise=noise,
        evidence_domain=0.1,
    )
    result = solver.solve_observation_batch(
        drift=torch.tensor([[drift]], dtype=dtype),
        threshold=torch.tensor([threshold], dtype=dtype),
        collapse_rate=torch.zeros(1, dtype=dtype),
        interval_low=torch.zeros(1, dtype=dtype),
        interval_high=torch.full((1,), 0.02, dtype=dtype),
        choice=torch.ones(1, dtype=dtype),
    )
    normal = torch.distributions.Normal(
        torch.tensor(drift * time_step, dtype=dtype),
        torch.tensor(noise * np.sqrt(time_step), dtype=dtype),
    )
    expected_upper = 1.0 - normal.cdf(torch.tensor(threshold, dtype=dtype))
    expected_lower = normal.cdf(torch.tensor(-threshold, dtype=dtype))
    torch.testing.assert_close(
        result.upper_probability[0], expected_upper, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(
        result.lower_probability[0], expected_lower, rtol=2e-4, atol=2e-5
    )
    torch.testing.assert_close(result.probability[0], result.upper_probability[0])
    assert result.mass_error[0] < 1e-10


def test_endpoint_crossing_solver_ignores_bucket_padding():
    solver = EndpointCrossingDDMSolver(
        time_step=0.01,
        spatial_points=255,
        noise=0.1,
        evidence_domain=0.1,
    )
    common = {
        "threshold": torch.tensor([0.02], dtype=torch.float64),
        "collapse_rate": torch.tensor([-0.1], dtype=torch.float64),
        "interval_low": torch.tensor([0.0], dtype=torch.float64),
        "interval_high": torch.tensor([0.025], dtype=torch.float64),
        "choice": torch.tensor([1.0], dtype=torch.float64),
    }
    short = solver.solve_observation_batch(
        drift=torch.full((1, 2), 0.03, dtype=torch.float64), **common
    )
    padded = solver.solve_observation_batch(
        drift=torch.full((1, 20), 0.03, dtype=torch.float64), **common
    )
    torch.testing.assert_close(short.probability, padded.probability)
    torch.testing.assert_close(
        short.survival_probability, padded.survival_probability
    )
    assert not bool(padded.invalid_boundary[0])


def test_pnl_lca_is_exactly_the_euler_mirror():
    step_size = 0.01
    state = torch.zeros(2, dtype=torch.float64)
    gain = torch.tensor(10.0, dtype=torch.float64)
    for task in (
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
        torch.tensor([1.0, 0.0], dtype=torch.float64),
    ):
        state = pnl_euler_lca_step(state, task, gain, step_size=step_size)
    pnl_state = pnl_lca_endpoint(
        gain=10.0,
        task=(1.0, 0.0),
        iti_duration=0.02,
        active_duration=0.02,
        step_size=step_size,
    )
    np.testing.assert_allclose(pnl_state, state.numpy(), rtol=0.0, atol=1e-15)


def test_configured_euler_lca_matches_legacy_step_and_scheduler_order():
    lca = ContinuousLCA()
    state = torch.zeros((1, 2), dtype=torch.float64)
    task = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    gain = torch.tensor([10.0], dtype=torch.float64)
    expected = pnl_euler_lca_step(
        state, task, gain, step_size=0.01
    )
    actual = lca.integrate_euler(
        state, task, gain, 0.01, max_step=0.01
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    stimulus = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float64)
    correct = torch.tensor([1.0], dtype=torch.float64)
    _, final_state = lca.drift_path_euler(
        state,
        task,
        gain,
        stimulus,
        correct,
        steps=1,
        step_size=0.01,
    )
    torch.testing.assert_close(final_state, expected, rtol=0.0, atol=0.0)


def test_euler_likelihood_path_is_finite_and_bypasses_native_rk4_scan():
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.01,
            ddm_spatial_points=17,
            lca_max_step=0.01,
            lca_integration_method="euler",
            iti_duration=0.02,
            native_lca_scan=True,
        )
    )
    parameters = ContinuousCSIParameters.defaults(dtype=torch.float64)
    result = likelihood.score(parameters, _trials())
    assert torch.isfinite(result.log_likelihood)
    assert result.diagnostics["lca_integration_method"] == "euler"


def test_solver_config_rejects_unknown_lca_integrator():
    with pytest.raises(ValueError, match="lca_integration_method"):
        SolverConfig(lca_integration_method="bogus")


def test_reset_history_ablation_removes_previous_rt_effect():
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.005,
            ddm_spatial_points=17,
            lca_max_step=0.01,
            iti_duration=0.02,
            reset_lca_each_trial=True,
        )
    )
    parameters = ContinuousCSIParameters.defaults(dtype=torch.float64)
    short = likelihood.score(
        parameters, _trials(first_rt=0.35, first_include=False)
    )
    long = likelihood.score(
        parameters, _trials(first_rt=0.75, first_include=False)
    )
    torch.testing.assert_close(short.probability[1], long.probability[1])
    assert short.diagnostics["reset_lca_each_trial"]


def test_fixed_rt_bins_match_pnl_empirical_range_width():
    trials = _trials()
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.005,
            ddm_spatial_points=17,
            lca_max_step=0.01,
            iti_duration=0.02,
            rt_bin_count=10,
        )
    )
    result = likelihood.score(
        ContinuousCSIParameters.defaults(dtype=torch.float64), trials
    )
    expected_width = (
        float(torch.max(trials.response_time) - torch.min(trials.response_time))
        * 1.04
        * 1.000001
        / 10
    )
    assert float(result.diagnostics["rt_bin_width"]) == pytest.approx(
        expected_width
    )
    assert result.diagnostics["rt_bin_count"] == 10


def test_sequential_likelihood_is_finite_differentiable_and_updates_masked_rows():
    config = SolverConfig(
        ddm_time_step=0.002,
        ddm_spatial_points=33,
        lca_max_step=0.005,
        iti_duration=0.05,
    )
    likelihood = ContinuousCSILikelihood(config)
    vector = (
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
    )
    result = likelihood.score_vector(vector, _trials())
    assert torch.isfinite(result.log_likelihood)
    assert torch.all(result.probability > 0.0)
    assert result.diagnostics["maximum_mass_error"] < 1e-9
    (-result.log_likelihood).backward()
    assert vector.grad is not None
    assert torch.all(torch.isfinite(vector.grad))
    assert torch.count_nonzero(vector.grad) > 0

    short_masked = likelihood.score_vector(
        vector.detach(), _trials(first_rt=0.35, first_include=False)
    )
    long_masked = likelihood.score_vector(
        vector.detach(), _trials(first_rt=0.75, first_include=False)
    )
    assert short_masked.per_trial_log_likelihood[0] == 0.0
    assert long_masked.per_trial_log_likelihood[0] == 0.0
    assert not torch.allclose(
        short_masked.lca_state_after_trial[0],
        long_masked.lca_state_after_trial[0],
    )
    assert not torch.allclose(
        short_masked.probability[1], long_masked.probability[1]
    )


def test_direct_likelihood_threshold_gradient_matches_finite_difference():
    trials = _trials(first_include=False)
    trials = CSITrialData(
        **{
            **trials.__dict__,
            "include": torch.tensor([False, True]),
        }
    )
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.002,
            ddm_spatial_points=33,
            lca_max_step=0.005,
            iti_duration=0.05,
        )
    )
    vector = (
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
    )
    value = likelihood.score_vector(vector, trials).log_likelihood
    value.backward()
    analytic = vector.grad[5]
    epsilon = 1.0e-5
    plus = vector.detach().clone()
    minus = vector.detach().clone()
    plus[5] += epsilon
    minus[5] -= epsilon
    finite_difference = (
        likelihood.score_vector(plus, trials).log_likelihood
        - likelihood.score_vector(minus, trials).log_likelihood
    ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite_difference, rtol=1e-5, atol=1e-6)


def test_direct_likelihood_ndt_gradient_matches_finite_difference():
    # Avoid an exact DDM time-cell edge: the piecewise-constant flux integral
    # is continuous there, but its endpoint derivative has a legitimate kink.
    trials = _trials(first_rt=0.553, second_rt=0.627, first_include=False)
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.002,
            ddm_spatial_points=33,
            lca_max_step=0.005,
            iti_duration=0.05,
        )
    )
    vector = (
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
    )
    likelihood.score_vector(vector, trials).log_likelihood.backward()
    # The only included row is condition 1, whose NDT is vector element 11.
    analytic = vector.grad[11]
    epsilon = 1.0e-6
    plus = vector.detach().clone()
    minus = vector.detach().clone()
    plus[11] += epsilon
    minus[11] -= epsilon
    finite_difference = (
        likelihood.score_vector(plus, trials).log_likelihood
        - likelihood.score_vector(minus, trials).log_likelihood
    ) / (2.0 * epsilon)
    torch.testing.assert_close(analytic, finite_difference, rtol=1e-5, atol=1e-6)


def test_duration_buckets_preserve_likelihood_state_and_gradients():
    # Keep RT-bin endpoints away from DDM cell edges, where the piecewise
    # constant flux approximation has a legitimate derivative kink.
    trials = _trials(first_rt=0.553, second_rt=0.627)
    vectors = [
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
        for _ in range(2)
    ]
    configurations = [
        SolverConfig(
            ddm_time_step=0.002,
            ddm_spatial_points=33,
            lca_max_step=0.005,
            iti_duration=0.05,
            ddm_bucket_size=bucket_size,
        )
        for bucket_size in (0, 1)
    ]
    results = [
        ContinuousCSILikelihood(config).score_vector(vector, trials)
        for config, vector in zip(configurations, vectors, strict=True)
    ]
    for result, vector in zip(results, vectors, strict=True):
        result.log_likelihood.backward()
        assert vector.grad is not None

    torch.testing.assert_close(results[0].probability, results[1].probability)
    torch.testing.assert_close(
        results[0].lca_state_after_trial,
        results[1].lca_state_after_trial,
    )
    torch.testing.assert_close(vectors[0].grad, vectors[1].grad)


def test_custom_adjoints_preserve_likelihood_and_all_gradients():
    trials = _trials(first_rt=0.253, second_rt=0.277)
    vectors = [
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
        for _ in range(2)
    ]
    common = dict(
        ddm_time_step=0.02,
        ddm_spatial_points=9,
        lca_max_step=0.02,
        iti_duration=0.02,
        ddm_bucket_size=1,
    )
    configurations = [
        SolverConfig(**common),
        SolverConfig(
            **common,
            custom_ddm_adjoint=True,
            custom_lca_adjoint=True,
        ),
    ]
    results = [
        ContinuousCSILikelihood(config).score_vector(vector, trials)
        for config, vector in zip(configurations, vectors, strict=True)
    ]
    for result, vector in zip(results, vectors, strict=True):
        (-result.log_likelihood).backward()
        assert vector.grad is not None

    torch.testing.assert_close(results[0].probability, results[1].probability)
    torch.testing.assert_close(vectors[0].grad, vectors[1].grad)


@pytest.mark.skipif(
    not native_lca_available(),
    reason="The optional native LCA scan requires Ninja and a C++ compiler.",
)
def test_native_ddm_forward_and_adjoint_match_torch_oracle():
    dtype = torch.float64
    generator = torch.Generator().manual_seed(12)
    drift = 0.04 * torch.randn(
        3, 37, generator=generator, dtype=dtype
    )
    threshold = torch.tensor([0.08, 0.12, 0.17], dtype=dtype)
    collapse = torch.tensor([-0.02, 0.0, -0.05], dtype=dtype)
    interval_low = torch.tensor([0.0113, 0.0217, 0.0312], dtype=dtype)
    interval_high = interval_low + 0.004
    choice = torch.tensor([0.0, 1.0, 1.0], dtype=dtype)
    common = dict(time_step=0.001, spatial_points=9, noise=0.1)
    torch_solver = MovingBoundaryDDMSolver(**common)
    native_solver = MovingBoundaryDDMSolver(**common, native_forward=True)
    arguments = dict(
        drift=drift,
        threshold=threshold,
        collapse_rate=collapse,
        interval_low=interval_low,
        interval_high=interval_high,
        choice=choice,
        store_density_history=True,
    )
    torch_result, torch_history = torch_solver._solve_observation_batch_impl(
        **arguments
    )
    native_result, native_history = native_solver._solve_observation_batch_impl(
        **arguments
    )
    assert torch_history is not None
    assert native_history is not None
    for field in torch_result.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(torch_result, field), getattr(native_result, field)
        )
    torch.testing.assert_close(torch_history, native_history)

    probability_gradient = torch.tensor([1.3, -0.7, 0.2], dtype=dtype)
    adjoint_arguments = (
        native_history,
        drift,
        threshold,
        collapse,
        interval_low,
        interval_high,
        choice,
        native_result.invalid_boundary,
        probability_gradient,
    )
    torch_gradients = torch_solver._likelihood_adjoint(*adjoint_arguments)
    native_gradients = native_solver._likelihood_adjoint(*adjoint_arguments)
    for torch_gradient, native_gradient in zip(
        torch_gradients, native_gradients, strict=True
    ):
        torch.testing.assert_close(torch_gradient, native_gradient)


@pytest.mark.skipif(
    not native_lca_available(),
    reason="The optional native LCA scan requires Ninja and a C++ compiler.",
)
def test_native_lca_scans_preserve_likelihood_and_all_gradients():
    trials = _trials(first_rt=0.253, second_rt=0.277)
    vectors = [
        ContinuousCSIParameters.defaults(dtype=torch.float64)
        .vector()
        .clone()
        .requires_grad_()
        for _ in range(2)
    ]
    common = dict(
        ddm_time_step=0.02,
        ddm_spatial_points=9,
        lca_max_step=0.02,
        iti_duration=0.02,
        ddm_bucket_size=1,
        custom_ddm_adjoint=True,
        custom_lca_adjoint=True,
    )
    configurations = [
        SolverConfig(**common),
        SolverConfig(
            **common,
            native_lca_scan=True,
            native_ddm_forward=True,
        ),
    ]
    results = [
        ContinuousCSILikelihood(config).score_vector(vector, trials)
        for config, vector in zip(configurations, vectors, strict=True)
    ]
    for result, vector in zip(results, vectors, strict=True):
        (-result.log_likelihood).backward()
        assert vector.grad is not None

    torch.testing.assert_close(results[0].probability, results[1].probability)
    torch.testing.assert_close(
        results[0].lca_state_after_trial,
        results[1].lca_state_after_trial,
    )
    torch.testing.assert_close(vectors[0].grad, vectors[1].grad)


def test_legacy_parameter_units_round_trip():
    parameters = ContinuousCSIParameters.defaults(dtype=torch.float64)
    restored = ContinuousCSIParameters.from_legacy_row(parameters.as_legacy_dict())
    torch.testing.assert_close(restored.vector(), parameters.vector())
    restored_from_series = ContinuousCSIParameters.from_legacy_row(
        pd.Series(parameters.as_legacy_dict())
    )
    torch.testing.assert_close(restored_from_series.vector(), parameters.vector())


def test_lbfgsb_uses_default_start_and_unit_scaled_coordinates():
    lower, upper = parameter_bounds()
    scale = upper - lower
    target_scaled = np.linspace(0.2, 0.8, len(lower))

    class QuadraticLikelihood:
        @staticmethod
        def score_vector(vector, trials):
            del trials
            torch_lower = torch.as_tensor(lower, dtype=vector.dtype)
            torch_scale = torch.as_tensor(scale, dtype=vector.dtype)
            target = torch.as_tensor(target_scaled, dtype=vector.dtype)
            loss = torch.sum(((vector - torch_lower) / torch_scale - target) ** 2)
            return SimpleNamespace(
                probability=torch.exp(-loss).reshape(1),
                included_row_indices=torch.tensor([0]),
            )

    trials = SimpleNamespace(response_time=torch.zeros(1, dtype=torch.float64))
    result = fit_lbfgsb(
        QuadraticLikelihood(),
        trials,
        starts=1,
        max_iterations=50,
    )
    expected = lower + scale * target_scaled
    np.testing.assert_allclose(result.parameter_vector, expected, atol=1.0e-6)
    assert result.success
    assert result.method.endswith("/unit-scaled")
    assert result.projected_gradient_inf_norm < 1.0e-5


def test_lbfgsb_screens_random_candidate_pool_before_optimization():
    lower, upper = parameter_bounds()
    scale = upper - lower

    class QuadraticLikelihood:
        @staticmethod
        def score_vector(vector, trials):
            del trials
            target = torch.as_tensor(
                lower + 0.7 * scale, dtype=vector.dtype
            )
            loss = torch.sum(((vector - target) / torch.as_tensor(
                scale, dtype=vector.dtype
            )) ** 2)
            return SimpleNamespace(
                probability=torch.exp(-loss).reshape(1),
                included_row_indices=torch.tensor([0]),
            )

    trials = _trials()
    result = fit_lbfgsb(
        QuadraticLikelihood(),
        trials,
        starts=2,
        max_iterations=5,
        random_start_candidates=5,
        coordinate_polish=False,
        polish_restarts=0,
        seed=19,
    )
    start_rows = [
        row for row in result.run_results if row["phase"] == "start"
    ]
    assert start_rows[0]["source"] == "default"
    assert start_rows[1]["source"] == "screened-random"
    assert start_rows[1]["random_pool_rank"] == 1
    assert start_rows[1]["random_pool_valid_candidates"] == 5


def test_random_start_respects_rt_and_collapsing_boundary_constraints():
    trials = _trials()
    lower, upper = parameter_bounds()
    vector = _feasible_start(
        np.random.default_rng(14), trials, lower, upper
    )
    condition = trials.condition_index.numpy()
    decision_high = (
        trials.response_time.numpy()
        + 0.5 * trials.rt_resolution
        - vector[10 + condition]
        - trials.is_switch.numpy() * vector[3]
    )
    boundary = vector[4 + condition] + vector[7 + condition] * decision_high
    assert np.all(decision_high > 0.0)
    assert np.all(boundary >= 1.0e-4)


def test_recovery_simulation_is_seeded_and_scores_at_truth():
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.005,
            ddm_spatial_points=33,
            lca_max_step=0.01,
            iti_duration=0.05,
        )
    )
    vector = torch.tensor(
        [
            15.0, 15.0, 15.0, 0.04,
            0.06, 0.06, 0.06,
            -0.02, -0.02, -0.02,
            0.15, 0.15, 0.15,
        ],
        dtype=torch.float64,
    )
    parameters = ContinuousCSIParameters.from_vector(vector)
    simulations = [
        simulate_sequential_trials(
            likelihood,
            parameters,
            _trials(),
            seed=8,
            simulation_time_step=0.002,
            maximum_decision_time=maximum_decision_time,
        )
        for maximum_decision_time in (2.0, 2.0, 2.5)
    ]
    torch.testing.assert_close(
        simulations[0].trials.choice, simulations[1].trials.choice
    )
    torch.testing.assert_close(
        simulations[0].trials.response_time,
        simulations[1].trials.response_time,
    )
    # The safety horizon must not perturb any trial's random-number stream.
    torch.testing.assert_close(
        simulations[0].trials.choice, simulations[2].trials.choice
    )
    torch.testing.assert_close(
        simulations[0].trials.response_time,
        simulations[2].trials.response_time,
    )
    result = likelihood.score(parameters, simulations[0].trials)
    assert torch.isfinite(result.log_likelihood)
    assert len(result.diagnostics["invalid_included_rows"]) == 0
    assert len(result.diagnostics["zero_probability_included_rows"]) == 0


def test_prescribed_first_passage_sampler_is_seeded_and_detects_crossings():
    case = PrescribedDDMCase(
        name="test",
        threshold=0.08,
        collapse_rate=-0.01,
        drift_offset=0.03,
    )
    simulations = [
        simulate_prescribed_first_passage(
            case,
            paths=1_000,
            time_step=0.002,
            maximum_time=0.5,
            seed=72,
            bridge_correction=True,
            chunk_size=400,
        )
        for _ in range(2)
    ]
    np.testing.assert_array_equal(
        simulations[0].crossing_time, simulations[1].crossing_time
    )
    np.testing.assert_array_equal(simulations[0].choice, simulations[1].choice)
    assert np.any(simulations[0].choice == 0)
    assert np.any(simulations[0].choice == 1)
    finite = np.isfinite(simulations[0].crossing_time)
    assert np.all(simulations[0].crossing_time[finite] > 0.0)


def test_recovery_summary_reports_scaled_error_and_bound_hits():
    lower, upper = parameter_bounds()
    truth = lower + 0.5 * (upper - lower)
    recovered = truth.copy()
    recovered[0] = lower[0]
    payload = {
        "truth_label": "middle",
        "simulation_seed": 3,
        "parameter_names": parameter_names(),
        "truth_parameter_vector": truth,
        "recovered_parameter_vector": recovered,
        "truth_log_likelihood": -10.0,
        "recovered_log_likelihood": -9.5,
        "fit": {
            "success": True,
            "stationary": False,
            "coordinate_stationary": True,
            "evaluations": 12,
            "iterations": 7,
            "rejected_start_attempts": 1,
        },
    }
    summary = summarize_recovery_results([payload])
    run = summary["recoveries"][0]
    assert run["bound_hit_count"] == 1
    assert run["lower_bound_parameters"] == ["gain[NoInstruction]"]
    assert run["log_likelihood_gain_over_truth"] == 0.5
    assert summary["groups"][0]["coordinate_stationary_count"] == 1


def test_staged_fit_uses_coarse_default_and_fine_meshes(monkeypatch):
    calls = []
    vector = ContinuousCSIParameters.defaults(dtype=torch.float64).vector().numpy()

    def result(method, value):
        return fit_module.FitResult(
            method=method,
            parameter_vector=value.copy(),
            log_likelihood=-1.0,
            success=True,
            evaluations=1,
            iterations=1,
            projected_gradient_inf_norm=0.0,
            stationary=True,
            coordinate_stationary=True,
            rejected_start_attempts=0,
            run_results=(),
            message="test",
        )

    def fake_cmaes(likelihood, trials, **kwargs):
        del trials
        calls.append(("coarse", likelihood.config, kwargs))
        return result("cmaes", vector)

    def fake_lbfgsb(likelihood, trials, **kwargs):
        del trials
        calls.append(("lbfgsb", likelihood.config, kwargs))
        return result("lbfgsb", np.asarray(kwargs["initial_vectors"])[0])

    monkeypatch.setattr(fit_module, "fit_cmaes", fake_cmaes)
    monkeypatch.setattr(fit_module, "fit_lbfgsb", fake_lbfgsb)
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.004,
            ddm_spatial_points=17,
            lca_max_step=0.01,
        )
    )
    staged = fit_module.fit_staged(
        likelihood,
        _trials(),
        coarse_ddm_time_step=0.008,
        coarse_ddm_spatial_points=9,
        coarse_lca_max_step=0.02,
        fine_ddm_time_step=0.002,
        fine_ddm_spatial_points=33,
        fine_lca_max_step=0.005,
    )
    assert [call[0] for call in calls] == ["coarse", "lbfgsb", "lbfgsb"]
    assert calls[0][1].ddm_time_step == 0.008
    assert calls[1][1].ddm_time_step == 0.004
    assert calls[2][1].ddm_time_step == 0.002
    assert staged.final_mesh == "fine"
    np.testing.assert_array_equal(staged.parameter_vector, vector)
