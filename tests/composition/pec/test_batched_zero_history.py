"""Projected zero-event histories preserve the counted prelude, not a fake DDM step."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import ObservationField, ObservationSpec, StochasticSamplingError
from psyneulink.core.batched.history import HistoryReplayError, validate_history_witness
from test_batched_csi_coevolving_acceptance import (
    _model, _compile_csi, registered_csi_drift_rate, _node,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("iti,csi", [(0, 0), (0, 2), (3, 0), (3, 2)])
def test_zero_and_positive_histories_match_oracle(registered_csi_drift_rate, batched_backend, iti, csi):
    if batched_backend != "triton":
        pytest.skip("The independent handwritten oracle requires CUDA")
    comp, inputs, outputs = _model(iti=iti, csi_switch=csi, csi_repeat=0,
                                   cue_values=[[0.], [1.], [0.], [1.]], ddm_noise=.1)
    plan = _compile_csi(comp, backend=batched_backend, outputs=outputs, max_steps=128)
    spec = ObservationSpec((ObservationField(outputs[0], "counting"),
                            ObservationField(outputs[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp")))
    history = plan.compile_history_replay(spec)
    rows = [{f"{_node(comp, 'DDM').name}.non_decision_time": .5},
            {f"{_node(comp, 'DDM').name}.non_decision_time": .2}]
    data = np.array([[1., .4], [1., .75], [1., .49], [1., .78]])
    trace = history.reconstruct(inputs, data, rows)
    assert trace.event_counts[0, 0] == trace.event_counts[0, 2] == 0
    assert np.all(trace.event_counts[1] > 0)
    _, oracle = plan.deterministic_history_log_likelihood(
        inputs, rows, 7, data, [0], return_debug=True, implementation="handwritten", seed=17)
    np.testing.assert_array_equal(trace.event_counts, oracle["observed_steps"])
    legacy_states = oracle["history_states"].cpu().numpy()
    # The custom kernel uses zero placeholders for never-initialized output
    # activities; the source IR stores the declared logistic initial output.
    # With initialized=0 both kernels replace these on their first active step.
    np.testing.assert_allclose(trace.end_states[..., [0, 1, 4]], legacy_states[..., [0, 1, 4]], atol=2e-6, rtol=2e-5)
    initialized = trace.end_states[..., 4] != 0
    np.testing.assert_allclose(trace.end_states[initialized], legacy_states[initialized], atol=2e-6, rtol=2e-5)
    if iti == 0:
        np.testing.assert_array_equal(trace.end_states[0, 0], trace.start_states[0, 0])
        gate = history.witness.component_ids.index(history.witness.zero_step_gate_component)
        assert trace.execution_counts[0, 0, gate] == 0
    paths = history.compile_boundary_trajectories().generate_device(inputs, data, rows)
    np.testing.assert_allclose(paths.values[..., 0].cpu(), oracle["drift_paths"].cpu(), atol=2e-6, rtol=2e-5)
    scorer = history.compile_boundary_trajectories().compile_stochastic_sampler().compile_observation_sampler().compile_histogram_score(
        categorical_dims=[0], bins=13, pseudocount=.1)
    fast = scorer.score(inputs, data, rows, num_estimates=37, execution="window", include_mask=[False, True, False, True])
    ref = scorer.score(inputs, data, rows, num_estimates=37, reference=True, include_mask=[False, True, False, True])
    np.testing.assert_array_equal(fast.bin_counts[:, [1, 3]], ref.bin_counts[:, [1, 3]])
    np.testing.assert_array_equal(fast.log_likelihood, ref.log_likelihood)
    with pytest.raises(HistoryReplayError, match="frozen source"):
        validate_history_witness(plan, spec, replace(history.witness, zero_step_gate_component=None))


@pytest.mark.triton_gpu
def test_legacy_named_api_defaults_to_generated(registered_csi_drift_rate, monkeypatch):
    from psyneulink.core.batched.backend.triton import csi_deterministic

    def forbidden(*args, **kwargs):
        pytest.fail("The default API executed the handwritten CSI kernel")

    monkeypatch.setattr(csi_deterministic, "run_csi_deterministic_history_likelihood", forbidden)
    comp, inputs, outputs = _model(ddm_noise=.1)
    plan = _compile_csi(comp, backend="triton", outputs=outputs, max_steps=128)
    data = np.array([[1., .4], [1., .75]])
    rows = [{f"{_node(comp, 'DDM').name}.non_decision_time": .5}]
    score = plan.deterministic_history_log_likelihood(inputs, rows, 7, data, [0])
    assert type(score) is float
    assert np.all(np.isfinite(score))
    cached = plan._default_history_observation_plan
    np.testing.assert_array_equal(score, plan.deterministic_history_log_likelihood(inputs, rows, 7, data, [0]))
    assert plan._default_history_observation_plan is cached
    with pytest.raises(ValueError, match="return_debug"):
        plan.deterministic_history_log_likelihood(inputs, rows, 7, data, [0], return_debug=True)
    with pytest.raises(StochasticSamplingError) as error:
        plan.deterministic_history_log_likelihood(inputs, rows, 7, data, [0], max_buffer_bytes=1)
    assert error.value.code == "sampling.memory_budget"


def test_empty_prelude_and_event_preserve_source_state(registered_csi_drift_rate, batched_backend):
    comp, inputs, outputs = _model(iti=0, csi_switch=0, csi_repeat=0, cue_values=[[0.], [0.]])
    plan = _compile_csi(comp, backend=batched_backend, outputs=outputs, max_steps=16)
    spec = ObservationSpec((ObservationField(outputs[0], "counting"),
                            ObservationField(outputs[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp")))
    history = plan.compile_history_replay(spec)
    trace = history.reconstruct(inputs, [[1., .1], [1., .2]])
    np.testing.assert_array_equal(trace.event_counts, 0)
    np.testing.assert_array_equal(trace.start_states, trace.end_states)
    gate = history.witness.component_ids.index(history.witness.zero_step_gate_component)
    np.testing.assert_array_equal(trace.execution_counts[..., gate], 0)
