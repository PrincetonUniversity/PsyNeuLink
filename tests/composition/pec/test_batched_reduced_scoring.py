"""Fused estimator reductions preserve samples' statistics and RNG identities."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import BatchedCompositionCompiler, LikelihoodEffectContract, StochasticSamplingError, batched_node_op, unregister_batched_instance_op
from psyneulink.core.batched.likelihood import histogram_likelihood
from test_batched_endpoints import _model, _node, _csi_drift_rate, _spec


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.fixture
def scoring_case(batched_backend):
    composition, inputs, outputs = _model(ddm_noise=.15)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        history = BatchedCompositionCompiler.compile_history_replay(composition, _spec(outputs), backend=batched_backend, max_steps=128)
        plan = history.compile_boundary_trajectories().compile_stochastic_sampler().compile_observation_sampler()
        data = history.simulate_reference(inputs, seed=12).observations[0]
        rows = [{}, {f"{_node(composition, 'DDM').name}.non_decision_time": .29},
                {f"{_node(composition, 'Task Activations [C1, C2]').name}.gain": 9.5}]
        yield plan, inputs, data, rows
    finally:
        unregister_batched_instance_op(drift.name)


@pytest.mark.parametrize("sigma,pseudocount", [(0., 0.), (.5, .1)])
def test_histogram_integer_counts_and_legacy_density_agree(scoring_case, sigma, pseudocount):
    observation, inputs, data, rows = scoring_case
    options = dict(categorical_dims=[0], bins=13, smoothing_sigma=sigma, pseudocount=pseudocount,
                   categorical_cardinalities=[2])
    plan = observation.compile_histogram_score(**options)
    kwargs = dict(num_estimates=37, seed=17, common_random_numbers=False)
    fused = plan.score(inputs, data, rows, **kwargs)
    reference = plan.score(inputs, data, rows, reference=True, **kwargs)
    np.testing.assert_array_equal(fused.bin_counts, reference.bin_counts)
    np.testing.assert_array_equal(fused.log_likelihood, reference.log_likelihood)
    samples = observation.sample(inputs, data, rows, **kwargs)
    legacy = histogram_likelihood(samples.values, data, **options)
    np.testing.assert_allclose(fused.densities, legacy, rtol=2e-6, atol=1e-10)
    assert not fused.bin_counts.flags.writeable


@pytest.mark.parametrize("common_random", [True, False])
def test_candidate_and_estimate_chunking_preserve_both_estimators(scoring_case, common_random):
    observation, inputs, data, rows = scoring_case
    kwargs = dict(num_estimates=37, seed=19, common_random_numbers=common_random)
    for plan, field in ((observation.compile_empirical_mass(), "successes"),
                        (observation.compile_histogram_score(categorical_dims=[0], bins=13, smoothing_sigma=.5, pseudocount=.1), "bin_counts")):
        whole = plan.score(inputs, data, rows, **kwargs)
        chunked = plan.score(inputs, data, rows, candidate_batch_size=1, estimate_batch_size=11, **kwargs)
        np.testing.assert_array_equal(getattr(whole, field), getattr(chunked, field))
        np.testing.assert_array_equal(whole.log_likelihood, chunked.log_likelihood)


def test_mass_fused_matches_materialized_execution(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_empirical_mass()
    kwargs = dict(num_estimates=37, seed=29)
    fused = plan.score(inputs, data, rows, **kwargs)
    sampled = plan.score(inputs, data, rows, execution="materialized", **kwargs)
    np.testing.assert_array_equal(fused.successes, sampled.successes)


def test_histogram_outside_range_floor_and_score_mask(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_histogram_score(categorical_dims=[0], bins=1, bin_range=[(10., 11.)])
    result = plan.score(inputs, data, rows, num_estimates=3, include_mask=[False, True])
    np.testing.assert_array_equal(result.bin_counts, 0)
    np.testing.assert_array_equal(result.log_likelihood, result.log_factors[:, 1])
    assert np.all(np.isfinite(result.log_likelihood))  # Explicit legacy floor, not exact mass.


def test_scoring_refuses_truncation_and_insufficient_path_budget(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_histogram_score(categorical_dims=[0])
    with pytest.raises(StochasticSamplingError) as error:
        plan.score(inputs, data, rows, num_estimates=3, horizon=1)
    assert error.value.code == "sampling.truncated"
    with pytest.raises(StochasticSamplingError) as error:
        plan.score(inputs, data, rows, num_estimates=3, max_buffer_bytes=1)
    assert error.value.code == "sampling.memory_budget"


def test_reduced_scoring_does_not_allocate_sample_sized_outputs(scoring_case):
    observation, inputs, data, rows = scoring_case
    if observation.sampler.path_plan.history_plan.simulation_plan.backend != "triton":
        pytest.skip("Memory scaling check uses compiled GPU sampling")
    plan = observation.compile_histogram_score(categorical_dims=[0])
    # This fits deterministic paths and compact reductions, not sample buffers.
    result = plan.score(inputs, data, rows, num_estimates=4097, max_buffer_bytes=256 * 1024)
    assert result.bin_counts.shape == (3, 2, 1)
    with pytest.raises(StochasticSamplingError) as error:
        plan.score(inputs, data, rows, num_estimates=4097, max_buffer_bytes=256 * 1024, reference=True)
    assert error.value.code == "sampling.memory_budget"


def test_memory_planner_chunks_candidates_without_changing_counts(scoring_case, monkeypatch):
    from psyneulink.core.batched.trajectories import BoundaryTrajectoryPlan

    observation, inputs, data, rows = scoring_case
    if observation.sampler.path_plan.history_plan.simulation_plan.backend != "triton":
        pytest.skip("Memory planner acceptance check uses GPU buffers")
    plan = observation.compile_histogram_score(categorical_dims=[0], bins=13)
    expected = plan.score(inputs, data, rows, num_estimates=37, common_random_numbers=False)
    original = BoundaryTrajectoryPlan.generate
    sizes = []

    def capture(path, inputs, data, candidates, **kwargs):
        sizes.append(len(candidates))
        return original(path, inputs, data, candidates, **kwargs)

    monkeypatch.setattr(BoundaryTrajectoryPlan, "generate", capture)
    result = plan.score(inputs, data, rows, num_estimates=37, common_random_numbers=False, max_buffer_bytes=16 * 1024)
    assert sizes == [1, 1, 1]
    np.testing.assert_array_equal(result.bin_counts, expected.bin_counts)


@pytest.mark.parametrize("execution", ["strict", "window"])
def test_nonfinite_observation_cannot_be_hidden_by_histogram_floor(scoring_case, monkeypatch, execution):
    from psyneulink.core.batched.backend.triton.scoring import ReducedObservationEmitter

    observation, inputs, data, rows = scoring_case
    original = ReducedObservationEmitter._emit_sample_outputs

    def inject_nonfinite(emitter, raw_vars):
        original(emitter, raw_vars)
        emitter.builder.line('observation_0 = tl.full((BLOCK,), float("inf"), tl.float32)')

    monkeypatch.setattr(ReducedObservationEmitter, "_emit_sample_outputs", inject_nonfinite)
    plan = observation.compile_histogram_score(categorical_dims=[0], pseudocount=.1)
    with pytest.raises(StochasticSamplingError) as error:
        plan.score(inputs, data, rows, num_estimates=3, execution=execution)
    assert error.value.code == "sampling.nonfinite"


def test_invalid_histogram_configuration_and_forged_readouts(scoring_case):
    observation, inputs, data, rows = scoring_case
    for options in (dict(bins=0), dict(smoothing_sigma=-1), dict(pseudocount=np.inf), dict(bin_range=[(1, 0)])):
        with pytest.raises(ValueError):
            observation.compile_histogram_score(categorical_dims=[0], **options)
    with pytest.raises(StochasticSamplingError):
        observation.compile_histogram_score(categorical_dims=[0, 1])
    plan = observation.compile_histogram_score(categorical_dims=[0])
    forged = replace(observation, witness=replace(observation.witness, readouts=tuple(reversed(observation.witness.readouts))))
    with pytest.raises(StochasticSamplingError):
        replace(plan, observation_plan=forged).score(inputs, data, rows, num_estimates=1)


def test_gpu_launch_geometry_preserves_integer_counts(scoring_case):
    observation, inputs, data, rows = scoring_case
    if observation.sampler.path_plan.history_plan.simulation_plan.backend != "triton":
        pytest.skip("Launch geometry is a GPU check")
    plan = observation.compile_histogram_score(categorical_dims=[0], smoothing_sigma=.5)
    kwargs = dict(num_estimates=257, seed=17, common_random_numbers=False)
    default = plan.score(inputs, data, rows, **kwargs)
    small = plan.score(inputs, data, rows, triton_launch_options={"block_size": 32, "num_warps": 1}, **kwargs)
    np.testing.assert_array_equal(default.bin_counts, small.bin_counts)


@pytest.mark.parametrize("common_random", [True, False])
def test_score_only_and_window_preserve_masked_counts_and_rng(scoring_case, common_random):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_histogram_score(categorical_dims=[0], bins=31, smoothing_sigma=.5, pseudocount=.1)
    kwargs = dict(num_estimates=137, seed=19, common_random_numbers=common_random, include_mask=[False, True])
    strict = plan.score(inputs, data, rows, **kwargs)
    for execution in ("score_only", "window"):
        result = plan.score(inputs, data, rows, execution=execution,
                            candidate_batch_size=1, estimate_batch_size=47, **kwargs)
        np.testing.assert_array_equal(result.bin_counts[:, 1], strict.bin_counts[:, 1])
        np.testing.assert_array_equal(result.log_likelihood, strict.log_likelihood)
        np.testing.assert_array_equal(result.sampled_trials, [False, True])
        np.testing.assert_array_equal(result.bin_counts[:, 0], -1)
        assert np.all(np.isnan(result.log_factors[:, 0]))
        assert np.all(np.isnan(result.densities[:, 0]))
        assert result.execution == execution
        if execution == "window":
            assert np.sum(result.window_stopped) > 0
        else:
            np.testing.assert_array_equal(result.window_stopped, 0)


def test_window_keeps_denominator_and_does_not_score_unfinished_readouts(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_histogram_score(categorical_dims=[0], bins=13, smoothing_sigma=.5, pseudocount=.1)
    kwargs = dict(num_estimates=137, seed=19)
    result = plan.score(inputs, data, rows, execution="window", **kwargs)
    reference = plan.score(inputs, data, rows, reference=True, **kwargs)
    np.testing.assert_array_equal(result.bin_counts, reference.bin_counts)
    np.testing.assert_array_equal(result.densities, reference.densities)
    assert np.sum(result.window_stopped) > 0
    assert result.num_estimates == kwargs["num_estimates"]
    with pytest.raises(StochasticSamplingError) as error:
        plan.score(inputs, data, rows, execution="window", horizon=1, **kwargs)
    assert error.value.code == "sampling.truncated"


def test_empty_include_mask_still_checks_observed_history(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.compile_histogram_score(categorical_dims=[0])
    for execution in ("score_only", "window"):
        result = plan.score(inputs, data, rows, execution=execution, include_mask=[False, False], num_estimates=3, horizon=1)
        np.testing.assert_array_equal(result.log_likelihood, 0)
        np.testing.assert_array_equal(result.bin_counts, -1)
        bad = data.copy()
        bad[0, 1] = -10.
        with pytest.raises(ValueError):
            plan.score(inputs, bad, rows, execution=execution, include_mask=[False, False], num_estimates=3)
    with pytest.raises(ValueError, match="oracle"):
        plan.score(inputs, data, rows, execution="window", reference=True)
    with pytest.raises(ValueError, match="execution"):
        plan.score(inputs, data, rows, execution="unknown")


def test_zero_count_cutoff_and_unrecognized_numeric_field(scoring_case):
    observation, inputs, data, rows = scoring_case
    empty = observation.compile_histogram_score(categorical_dims=[0], bins=1, bin_range=[(10., 11.)])
    result = empty.score(inputs, data, rows, num_estimates=3, execution="window")
    strict = empty.score(inputs, data, rows, num_estimates=3)
    np.testing.assert_array_equal(result.bin_counts, strict.bin_counts)
    np.testing.assert_array_equal(result.window_stopped, 3)
    # Scoring a non-time output numerically must not borrow the RT certificate.
    unknown = observation.compile_histogram_score(categorical_dims=[1], bins=3)
    result = unknown.score(inputs, data, rows, num_estimates=3, execution="window")
    strict = unknown.score(inputs, data, rows, num_estimates=3)
    np.testing.assert_array_equal(result.bin_counts, strict.bin_counts)
    np.testing.assert_array_equal(result.window_stopped, 0)
