"""Count-only scoring must preserve the ordinary stateful simulator's objective."""

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.batched.errors import BatchedNumericalError
from psyneulink.core.batched import likelihood


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _model(backend, max_steps=32):
    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=1.1),
        leak=.3, competition=.2, self_excitation=0.,
        noise=pnl.NormalDist(standard_deviation=.15, seed=11),
        time_step_size=.1, termination_threshold=.65,
        execute_until_finished=False, reset_stateful_function_when=pnl.Never(),
        output_ports=[pnl.RESULT, pnl.DECISION_INDEX, pnl.DECISION_TIME],
    )
    readout = pnl.ProcessingMechanism()
    comp = pnl.Composition()
    comp.add_nodes([lca, readout])
    comp.add_projection(sender=lca.output_ports[pnl.DECISION_TIME], receiver=readout)
    comp.scheduler.add_condition(lca, pnl.Always())
    comp.scheduler.add_condition(readout, pnl.WhenFinished(lca))
    plan = BatchedCompositionCompiler.compile(comp, backend=backend, max_steps=max_steps,
                                             outputs=list(lca.output_ports))
    return plan, lca


@pytest.mark.parametrize("common_random", [True, False])
def test_fused_matches_materialized_persistent_sequences(batched_backend, common_random, monkeypatch):
    plan, lca = _model(batched_backend)
    inputs = {lca: [[3., 1.], [1., 3.], [2., 1.], [1., 3.], [3., 1.], [1., 2.]]}
    candidates = [{}, {f"{lca.name}.gain": BatchedTrialParameter(np.linspace(.9, 1.2, 6))}]
    densities = []
    original = likelihood._sum_histogram_log_likelihood

    def capture(values, include_mask):
        densities.append(values.copy())
        return original(values, include_mask)

    monkeypatch.setattr(likelihood, "_sum_histogram_log_likelihood", capture)
    options = dict(data=np.array([[.2, 0], [.3, 1], [.2, 0]]), categorical_dims=[1],
                   outcome_indices=[3, 2], bins=7, bin_range=[(0., 2.)], pseudocount=.3,
                   categorical_cardinalities=[2], include_mask=[True, False, True],
                   subject_slices=[slice(0, 3), slice(3, 6)], seed=29,
                   common_random_numbers=common_random, strict_truncation=True)
    actual = plan.log_likelihood(inputs, candidates, 37, **options)
    expected = plan.log_likelihood(inputs, candidates, 37, fused=False, **options)
    np.testing.assert_array_equal(densities[0], densities[1])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, plan.log_likelihood(inputs, candidates, 37, **options))


@pytest.mark.parametrize("categorical_dims", [None, [True, True]])
def test_fused_vector_outputs_and_bin_edges(batched_backend, categorical_dims):
    plan, lca = _model(batched_backend)
    inputs = {lca: [[3., 1.], [1., 3.]]}
    values = plan.run(inputs, [{}], 5, seed=29).values[0, 0, :, 0, :2]
    # Includes exact simulated observations and edges, with both purely
    # categorical scoring and a two-dimensional continuous histogram.
    options = dict(data=values, outcome_indices=[0, 1], categorical_dims=categorical_dims,
                   bins=1, bin_range=[(values[:, i].min(), values[:, i].max()) for i in range(2)],
                   seed=29, pseudocount=1.)
    actual = plan.log_likelihood(inputs, [{}, {}], 17, **options)
    expected = plan.log_likelihood(inputs, [{}, {}], 17, fused=False, **options)
    np.testing.assert_array_equal(actual, expected)


def test_fused_checks_masked_trials_and_unscored_outputs(batched_backend):
    plan, lca = _model(batched_backend, max_steps=1)
    options = dict(data=[[.1]], categorical_dims=None, outcome_indices=[3],
                   bin_range=[(0., 1.)], include_mask=[False], strict_truncation=True)
    with pytest.raises(BatchedTruncationError):
        plan.log_likelihood({lca: [[0., 0.]]}, [{}], 5, **options)
    # A selected finite readout must not hide NaNs in another plan output.
    options.update(outcome_indices=[2], data=[[0.], [0.]], include_mask=[False, False])
    with pytest.raises(BatchedNumericalError):
        plan.log_likelihood({lca: [[3., 1.], [3., 1.]]}, [{f"{lca.name}.time_step_size": 3e38}], 5, **options)


def test_fused_exact_interior_edge_belongs_to_lower_bin(batched_backend):
    plan, lca = _model(batched_backend)
    inputs = {lca: [[3., 1.]]}
    rt = float(plan.run(inputs, [{}], 1, seed=29).values[0, 0, 0, 0, 3])
    # Undo the estimator's upper-edge padding so its middle FP32 edge is
    # exactly an attainable response time. Torch bucketize uses right=False.
    options = dict(data=[[rt]], outcome_indices=[3], bins=2,
                   bin_range=[(0., 2 * rt / (1 + 1e-6))], seed=29)
    actual = plan.log_likelihood(inputs, [{}], 37, **options)
    expected = plan.log_likelihood(inputs, [{}], 37, fused=False, **options)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("smoothing", [0., .5])
def test_atomic_stateful_histogram_and_smoothed_fallback(batched_backend, smoothing):
    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=1.3),
        leak=.3, competition=.4, self_excitation=.2, noise=.125,
        time_step_size=.2, termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3, reset_stateful_function_when=pnl.Never(),
    )
    plan = BatchedCompositionCompiler.compile(pnl.Composition(pathways=lca),
                                             backend=batched_backend, max_steps=16)
    inputs = {lca: [[1., -.25], [-.6, .85]]}
    options = dict(data=[[.5, .6], [.7, .8]], bins=5, bin_range=[(0., 1.), (0., 1.)],
                   smoothing_sigma=smoothing, pseudocount=.1)
    actual = plan.log_likelihood(inputs, [{}, {}], 17, **options)
    expected = plan.log_likelihood(inputs, [{}, {}], 17, fused=False, **options)
    np.testing.assert_array_equal(actual, expected)
