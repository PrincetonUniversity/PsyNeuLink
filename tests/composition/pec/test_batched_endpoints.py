"""Event reconstruction tests against independent readouts and coupled execution."""

from dataclasses import replace
import json

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedTrialParameter,
    EndpointExpression,
    EndpointReconstructionError,
    LikelihoodEffectContract,
    ObservationField,
    ObservationSpec,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import registry
from psyneulink.core.batched.endpoints import validate_endpoint_witness
from psyneulink.core.batched.specs import BatchedOpSpecError, register_batched_op
from test_batched_csi_coevolving_acceptance import _csi_drift_rate, _model, _node


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _spec(outputs):
    return ObservationSpec((
        ObservationField(outputs[0], "counting"),
        ObservationField(outputs[1], "counting", role="event_time"),
    ))


def _ddm_plan(*, wrapper=None, cap=32):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            non_decision_time=0.2, time_step_size=0.01, threshold=0.05, noise=0.0,
        ), output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
    )
    composition = pnl.Composition(pathways=decision)
    output = decision.output_ports[1]
    if wrapper is not None:
        gate = pnl.ProcessingMechanism(function=wrapper)
        composition.add_node(gate)
        composition.add_projection(sender=output, receiver=gate)
        output = gate.output_port
    observations = _spec((decision.output_ports[0], output))
    plan = BatchedCompositionCompiler.compile_observed_endpoints(
        composition, observations, max_steps=cap,
    )
    return plan, decision


@pytest.fixture
def coupled():
    composition, inputs, outputs = _model(ddm_noise=0.0)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        yield composition, inputs, outputs
    finally:
        unregister_batched_instance_op(drift.name)


def test_direct_readout_reconstructs_counts_and_parameter_dependent_offsets():
    plan, decision = _ddm_plan()
    data = np.array([[1.0, 0.25], [1.0, 0.27]])
    counts = plan.reconstruct(
        {decision: [[1.0], [1.0]]}, data,
        parameter_sets=[{}, {"non_decision_time": 0.22}, {
            "non_decision_time": BatchedTrialParameter([0.21, 0.23]),
        }],
    )
    np.testing.assert_array_equal(counts[..., 0], [[5, 7], [3, 5], [4, 4]])
    assert not counts.flags.writeable


@pytest.mark.parametrize("wrapper, observed", [
    (lambda: pnl.Linear(slope=2.0, intercept=0.1, scale=3.0, offset=0.04), 1.84),
    (lambda: pnl.Linear(slope=-2.0, intercept=1.0), 0.5),
])
def test_affine_gate_is_derived_from_the_supplied_model(wrapper, observed):
    plan, decision = _ddm_plan(wrapper=wrapper())
    counts = plan.reconstruct({decision: [[1.0]]}, [[1.0, observed]])
    np.testing.assert_array_equal(counts, [[[5]]])
    assert plan.witnesses[0].projection_ids


def test_nonlinear_output_is_not_assumed_to_be_an_affine_event_readout():
    with pytest.raises(EndpointReconstructionError) as error:
        _ddm_plan(wrapper=pnl.Logistic())
    assert error.value.code == "endpoint.readout_rule_missing"


@pytest.mark.parametrize("observed, parameters, code", [
    (0.255, {}, "endpoint.count_incompatible"),
    (0.2, {}, "endpoint.count_incompatible"),
    (0.60, {}, "endpoint.count_incompatible"),
    (np.nan, {}, "endpoint.observation_nonfinite"),
    (np.inf, {}, "endpoint.observation_nonfinite"),
    (0.25, {"time_step_size": 0.0}, "endpoint.step_nonpositive"),
    (float(2**24), {"non_decision_time": float(2**24)}, "endpoint.count_ambiguous"),
])
def test_runtime_guards_do_not_ceil_or_choose_an_ambiguous_count(observed, parameters, code):
    plan, decision = _ddm_plan()
    with pytest.raises(EndpointReconstructionError) as error:
        plan.reconstruct({decision: [[1.0]]}, [[1.0, observed]], parameters)
    assert error.value.code == code


def test_zero_gate_scale_does_not_identify_a_counter():
    plan, decision = _ddm_plan(wrapper=pnl.Linear(slope=0.0, intercept=1.0))
    with pytest.raises(EndpointReconstructionError) as error:
        plan.reconstruct({decision: [[1.0]]}, [[1.0, 1.0]])
    assert error.value.code == "endpoint.count_ambiguous"


def test_nonfinite_parameters_use_shared_parameter_validation():
    plan, decision = _ddm_plan()
    with pytest.raises(ValueError, match="finite"):
        plan.reconstruct({decision: [[1.0]]}, [[1.0, 0.25]], {"non_decision_time": np.nan})


def test_timestep_change_reconstructs_a_different_count():
    plan, decision = _ddm_plan(cap=128)
    counts = plan.reconstruct(
        {decision: [[1.0]]}, [[1.0, 0.25]], [{}, {"time_step_size": 0.001}],
    )
    np.testing.assert_array_equal(counts[..., 0], [[5], [50]])


def test_contracted_and_separate_fp32_readouts_resolve_the_same_counts():
    plan, decision = _ddm_plan(cap=64)
    counts = np.arange(1, 65, dtype=np.float32)
    dt, ndt = np.float32(0.003), np.float32(0.237)
    separated = np.float32(ndt + np.float32(counts * dt))
    contracted = np.float32(float(ndt) + counts.astype(np.float64) * float(dt))
    parameters = {"time_step_size": float(dt), "non_decision_time": float(ndt)}
    inputs = {decision: np.ones((64, 1))}
    for values in (separated, contracted):
        reconstructed = plan.reconstruct(inputs, np.column_stack((np.ones(64), values)), parameters)
        np.testing.assert_array_equal(reconstructed[0, :, 0], counts)


@pytest.mark.parametrize("mutation", [
    lambda witness: replace(witness, clock_component_id=witness.clock_component_id + 1),
    lambda witness: replace(witness, minimum_count=0),
    lambda witness: replace(witness, expression=EndpointExpression("constant", value=0.25)),
    lambda witness: replace(witness, parameter_ids=()),
    lambda witness: replace(witness, guarantee="proved"),
])
def test_forged_witness_does_not_authorize_reconstruction(mutation):
    plan, decision = _ddm_plan()
    forged = mutation(plan.witnesses[0])
    with pytest.raises(EndpointReconstructionError) as error:
        validate_endpoint_witness(plan.simulation_plan.kernel_ir, forged)
    assert error.value.code == "endpoint.witness_mismatch"
    with pytest.raises(EndpointReconstructionError):
        replace(plan, witnesses=(forged,)).reconstruct({decision: [[1.0]]}, [[1.0, 0.25]])


def test_forged_observation_column_is_rejected():
    plan, decision = _ddm_plan()
    witness = plan.witnesses[0]
    forged = replace(witness, observation=replace(witness.observation, column_start=0))
    with pytest.raises(EndpointReconstructionError) as error:
        replace(plan, witnesses=(forged,)).reconstruct({decision: [[1.0]]}, [[1.0, 0.25]])
    assert error.value.code == "endpoint.witness_mismatch"


def test_event_label_on_choice_does_not_resolve_a_clock():
    plan, decision = _ddm_plan()
    wrong = ObservationSpec((ObservationField(decision.output_ports[0], "counting", role="event_time"),))
    with pytest.raises(EndpointReconstructionError) as error:
        plan.simulation_plan.compile_observed_endpoints(wrong)
    assert error.value.code == "endpoint.readout_rule_missing"


@pytest.mark.parametrize("options", [
    {"recording": "rounded", "precision": 0.01},
    {"availability": "may_be_missing"},
    {"recording": "censored"},
])
def test_recording_ambiguity_still_requires_an_observation_operator(options):
    plan, decision = _ddm_plan()
    observations = ObservationSpec((ObservationField(
        decision.output_ports[1], "counting", role="event_time", **options,
    ),))
    with pytest.raises(EndpointReconstructionError) as error:
        plan.simulation_plan.compile_observed_endpoints(observations)
    assert error.value.code == "endpoint.recording_unsupported"


def test_frozen_readout_does_not_follow_registry_replacement():
    plan, decision = _ddm_plan()
    witness = plan.witnesses[0]
    original = plan.simulation_plan.kernel_ir.op_specs.lookup_spec(witness.clock_spec_key)
    try:
        register_batched_op(replace(original, likelihood_contract=None))
        counts = plan.reconstruct({decision: [[1.0]]}, [[1.0, 0.25]])
        np.testing.assert_array_equal(counts, [[[5]]])
    finally:
        register_batched_op(original)


@pytest.mark.parametrize("attribute, value", [
    ("counter_state", "not_a_counter"),
    ("output_port", "not_an_output"),
    ("step_parameter", "not_a_parameter"),
    ("offset_parameter", "not_a_parameter"),
])
def test_event_readout_registration_authenticates_primitive_bindings(attribute, value):
    plan, _ = _ddm_plan()
    witness = plan.witnesses[0]
    original = plan.simulation_plan.kernel_ir.op_specs.lookup_spec(witness.clock_spec_key)
    contract = original.likelihood_contract
    invalid = replace(contract.event_readout, **{attribute: value})
    with pytest.raises(BatchedOpSpecError, match="Event readout must bind"):
        register_batched_op(replace(original, likelihood_contract=replace(contract, event_readout=invalid)))


def test_premature_rt_gate_is_rejected_by_endpoint_publication_validation(coupled):
    composition, _, outputs = coupled
    composition.scheduler.add_condition(outputs[1].owner, pnl.Always())
    # Continuous readouts are executable, but do not establish the publication
    # ordering required for reconstruction from an observed finishing event.
    simulation = BatchedCompositionCompiler.compile(composition, outputs=outputs, max_steps=128)
    assert simulation.kernel_ir.executable
    with pytest.raises(EndpointReconstructionError) as error:
        BatchedCompositionCompiler.compile_observed_endpoints(composition, _spec(outputs), max_steps=128)
    assert error.value.code == "endpoint.publication_unproven"


def test_endpoint_plan_does_not_require_a_cuda_device(monkeypatch):
    monkeypatch.setattr(registry, "_backend_availability", lambda backend: (False, []))
    plan, decision = _ddm_plan()
    assert not plan.simulation_plan.capability_report.backend_available
    np.testing.assert_array_equal(plan.reconstruct({decision: [[1.0]]}, [[1.0, 0.25]]), [[[5]]])


@pytest.mark.triton_interpreter
def test_coupled_csi_readout_matches_actual_simulation(coupled):
    composition, inputs, outputs = coupled
    simulation = BatchedCompositionCompiler.compile(composition, outputs=outputs, max_steps=128)
    simulated = simulation.run(inputs, parameter_sets=[{}], num_estimates=1, strict_truncation=True)
    data = simulated.values[0, 0, :, 0, :]
    endpoints = simulation.compile_observed_endpoints(_spec(outputs))
    counts = endpoints.reconstruct(inputs, data)
    cue = _node(composition, "Cue Stimulus Interval")
    expected = np.rint((data[:, 1] - 0.3 - np.asarray(inputs[cue]).reshape(-1) * 0.01) / 0.01)
    np.testing.assert_array_equal(counts[0, :, 0], expected)
    report = simulation.diagnose_likelihood(_spec(outputs))
    assert report.history_kind == "deterministic_history_candidate"
    assert not report.codegen_ready
    assert not report.can_execute
    assert report.endpoint_witnesses
    assert "likelihood.endpoint_reconstruction_required" not in {d.code for d in report.obligations}
    assert "likelihood.history_reconstruction_required" in {d.code for d in report.obligations}
    json.dumps(report.to_dict())


@pytest.mark.triton_interpreter
def test_renamed_csi_with_changed_rt_projection_and_gate(coupled):
    composition, inputs, outputs = coupled
    response = outputs[1].owner
    response.function.parameters.slope.set(2.0)
    response.function.parameters.intercept.set(0.1)
    decision = _node(composition, "DDM")
    cue = _node(composition, "Cue Stimulus Interval")
    for projection in response.path_afferents:
        if projection.sender is decision.output_ports[1]:
            projection.parameters.matrix.set([[3.0]])
    drift = _node(composition, "Drift Rate Value")
    old_name = drift.name
    for index, node in enumerate(composition.nodes):
        node.name = f"renamed endpoint node {index}"
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    try:
        simulation = BatchedCompositionCompiler.compile(composition, outputs=outputs, max_steps=128)
        data = simulation.run(inputs, [{}], num_estimates=1, strict_truncation=True).values[0, 0, :, 0, :]
        endpoints = simulation.compile_observed_endpoints(_spec(outputs))
        counts = endpoints.reconstruct(inputs, data)[0, :, 0]
        cue_times = np.asarray(inputs[cue]).reshape(-1) * 0.01
        expected = np.rint((((data[:, 1] - 0.1) / 2.0 - cue_times) / 3.0 - 0.3) / 0.01)
        np.testing.assert_array_equal(counts, expected)
    finally:
        unregister_batched_instance_op(old_name)
        unregister_batched_instance_op(drift.name)
