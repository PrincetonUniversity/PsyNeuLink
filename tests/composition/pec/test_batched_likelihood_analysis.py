"""Likelihood diagnosis must not authorize unproved history transformations."""

from dataclasses import FrozenInstanceError, replace
import json

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    LikelihoodEffectContract,
    ObservationField,
    ObservationSpec,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import registry
from psyneulink.core.batched.bindings import BatchedComponentBindings
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source
from psyneulink.core.batched.observation import resolve_observations
from psyneulink.core.batched.specs import (
    BatchedOpSpecError,
    MechanismOpSpec,
    RngDecl,
    register_batched_instance_op,
)
from test_batched_csi_coevolving_acceptance import _csi_drift_rate, _model, _node


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _codes(report):
    return {item.code for item in report.diagnostics}


def _obligation_codes(report):
    return {item.code for item in report.obligations}


def _ddm(*, execute_until_finished=True):
    node = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(noise=0.2),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="arbitrary stochastic integrator",
        execute_until_finished=execute_until_finished,
    )
    return pnl.Composition(pathways=node), node


def _observations(outputs, **time_options):
    return ObservationSpec((
        ObservationField(outputs[0], "counting"),
        ObservationField(outputs[1], "counting", role="event_time", **time_options),
    ))


def _diagnose(composition, observations):
    return BatchedCompositionCompiler.diagnose_likelihood(
        composition, observations, max_steps=64,
    )


@pytest.fixture
def csi():
    composition, _, outputs = _model(ddm_noise=0.1)
    drift = _node(composition, "Drift Rate Value")
    batched_node_op(
        drift.name, likelihood_contract=LikelihoodEffectContract(),
    )(_csi_drift_rate)
    try:
        yield composition, outputs
    finally:
        unregister_batched_instance_op(drift.name)


def test_reset_ddm_factorization_is_separate_from_likelihood_codegen():
    composition, node = _ddm()
    report = _diagnose(composition, _observations(node.output_ports))

    assert report.simulation.is_supported
    assert report.history_kind == "independent_trials"
    assert report.factorization_status == "eligible"
    assert report.candidate_strategy == "independent_trials"
    assert not report.retained_state_ids
    assert not report.held_parameter_ids
    assert not report.diagnostics
    assert not report.codegen_ready
    assert not report.can_execute
    assert report.contracts
    assert all(item.trust == "registered_contract" for item in report.contracts)
    json.dumps(report.to_dict())


def test_diagnosis_works_without_cuda(monkeypatch):
    composition, node = _ddm()
    monkeypatch.setattr(registry, "_backend_availability", lambda backend: (False, []))
    report = BatchedCompositionCompiler.diagnose_likelihood(
        composition, _observations(node.output_ports), backend="triton",
    )
    assert not report.simulation.backend_available
    assert report.history_kind == "independent_trials"
    assert not report.can_execute


def test_float_outputs_do_not_establish_a_continuous_density():
    composition, node = _ddm()
    observations = ObservationSpec((
        ObservationField(node.output_ports[1], "lebesgue", role="event_time"),
    ))
    report = _diagnose(composition, observations)
    assert report.factorization_status == "eligible"
    assert "observation.density_not_established" in _obligation_codes(report)


def test_csi_is_a_candidate_with_explicit_scheduler_obligations(csi):
    composition, outputs = csi
    plan = BatchedCompositionCompiler.compile(composition, outputs=outputs, max_steps=64)
    original_source = triton_graph_kernel_source(plan.kernel_ir)
    original_dependencies = plan.kernel_ir.metadata["axis_dependencies"]
    report = plan.diagnose_likelihood(_observations(outputs))

    assert report.history_kind == "deterministic_history_candidate", report
    assert report.factorization_status == "candidate"
    assert report.candidate_strategy == "deterministic_history"
    assert len(report.retained_state_ids) == 3
    assert report.held_parameter_ids  # includes held controls, not only LCA state
    assert {
        "likelihood.endpoint_runtime_guard_required",
        "likelihood.history_reconstruction_required",
        "likelihood.trajectory_partition_required",
    } <= _obligation_codes(report)
    assert len(report.endpoint_witnesses) == 1
    assert report.endpoint_witnesses[0].counter_state == "steps"
    assert any(edge[2] == "schedule_termination_control" for edge in original_dependencies["edges"])
    assert not report.can_execute
    assert plan.kernel_ir.metadata["axis_dependencies"] == original_dependencies
    assert triton_graph_kernel_source(plan.kernel_ir) == original_source


def test_csi_diagnosis_does_not_match_model_names(csi):
    composition, outputs = csi
    original = _diagnose(composition, _observations(outputs))
    drift = _node(composition, "Drift Rate Value")
    old_name = drift.name
    for index, node in enumerate(composition.nodes):
        node.name = f"anonymous component {index}"
    batched_node_op(
        drift.name, likelihood_contract=LikelihoodEffectContract(),
    )(_csi_drift_rate)
    try:
        renamed = _diagnose(composition, _observations(outputs))
        assert renamed.history_kind == original.history_kind
        assert renamed.factorization_status == original.factorization_status
        assert _obligation_codes(renamed) == _obligation_codes(original)
    finally:
        unregister_batched_instance_op(old_name)
        unregister_batched_instance_op(drift.name)


@pytest.mark.parametrize("options", [
    {"recording": "rounded", "precision": 0.01},
    {"recording": "noisy"},
    {"recording": "censored"},
    {"availability": "may_be_missing"},
    {"condition_history": False},
])
def test_uncertain_or_unused_endpoint_does_not_enable_history(csi, options):
    composition, outputs = csi
    report = _diagnose(composition, _observations(outputs, **options))
    assert report.factorization_status == "blocked"
    assert report.candidate_strategy is None
    assert "observation.endpoint_ambiguous" in _codes(report)


def test_no_event_observation_reports_scheduler_dependency(csi):
    composition, outputs = csi
    observations = ObservationSpec(tuple(ObservationField(p, "counting") for p in outputs))
    report = _diagnose(composition, observations)
    assert "observation.endpoint_required" in _codes(report)
    assert report.diagnostics[0].dependency_path


def test_unscored_endpoint_still_conditions_history(csi):
    composition, outputs = csi
    report = _diagnose(composition, _observations(outputs, score=False))
    assert report.history_kind == "deterministic_history_candidate"
    assert report.observations[1].condition_history
    assert not report.observations[1].score
    assert "observation.selected_score" in _obligation_codes(report)


def test_numeric_lca_noise_is_not_misclassified_as_random(csi):
    composition, outputs = csi
    lca = _node(composition, "Task Activations [C1, C2]")
    lca.parameters.noise.set(0.1)
    lca.integrator_function.parameters.noise.set(0.1)
    report = _diagnose(composition, _observations(outputs))
    assert report.history_kind == "deterministic_history_candidate", report


def test_latent_initial_conditions_are_not_silently_initialized(csi):
    composition, outputs = csi
    observations = replace(_observations(outputs), initial_state="latent")
    report = _diagnose(composition, observations)
    assert report.factorization_status == "blocked"
    assert "likelihood.initial_state_unresolved" in _codes(report)


@pytest.mark.parametrize("reset", [False, True])
def test_nondecision_deterministic_history_and_reset(reset):
    node = pnl.LCAMechanism(
        input_shapes=2, noise=0.0, function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL, termination_threshold=3,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart() if reset else pnl.Never(),
    )
    readout = pnl.TransferMechanism(input_shapes=2)
    composition = pnl.Composition(pathways=[node, readout])
    composition.scheduler.add_condition(node, pnl.Always())
    composition.scheduler.add_condition(readout, pnl.WhenFinished(node))
    observations = ObservationSpec((ObservationField(readout.output_port, "counting"),))
    report = _diagnose(composition, observations)
    assert report.history_kind == (
        "independent_trials" if reset else "deterministic_history_candidate"
    ), report
    assert "likelihood.endpoint_reconstruction_required" not in _obligation_codes(report)


def test_stochastic_state_dependency_is_reported_with_a_witness():
    composition, decision = _ddm()
    state = pnl.LCAMechanism(
        input_shapes=2, noise=0.0, function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL, termination_threshold=3,
        reset_stateful_function_when=pnl.Never(),
    )
    composition.add_node(state)
    composition.add_projection(
        sender=decision.output_ports[0], receiver=state,
        projection=pnl.MappingProjection(matrix=[[1.0, -1.0]]),
    )
    observations = ObservationSpec((
        ObservationField(state.output_port, "counting"),
        ObservationField(decision.output_ports[1], "counting", role="event_time"),
    ))
    report = _diagnose(composition, observations)
    assert report.simulation.is_supported, report.simulation.unsupported_reasons
    assert report.history_kind == "unresolved_stochastic_history"
    assert report.factorization_status == "blocked"
    diagnostic = next(
        item for item in report.diagnostics
        if item.code == "likelihood.stochastic_history_unresolved"
    )
    assert diagnostic.dependency_path
    assert any(edge.kind == "projection" for edge in diagnostic.dependency_path)


def test_port_resolution_is_object_free_and_rejects_foreign_or_duplicate_ports():
    composition, node = _ddm()
    observations = _observations(node.output_ports)
    plan = BatchedCompositionCompiler.compile(composition, outputs=node.output_ports)
    report = plan.diagnose_likelihood(observations)
    assert report.observations[0].port_id != report.observations[1].port_id
    assert report.observations[1].column_start == 1
    with pytest.raises(FrozenInstanceError):
        report.observations[0].port_id = 123
    _, foreign = _ddm()
    with pytest.raises(ValueError, match="exact output"):
        plan.diagnose_likelihood(_observations(foreign.output_ports))
    duplicate = ObservationSpec((observations.fields[0], observations.fields[0]))
    with pytest.raises(ValueError, match="observed twice"):
        resolve_observations(duplicate, plan.ir.graph, plan.component_bindings)
    # A missing binding must not authenticate a None port by .get() fallback.
    with pytest.raises(ValueError, match="exact output"):
        resolve_observations(
            ObservationSpec((ObservationField(None, "counting"),)),
            replace(plan.ir.graph, outputs=(plan.ir.graph.outputs[0],)),
            BatchedComponentBindings(),
        )


def _identity(x0):
    return x0


def test_opaque_udf_requires_contract_and_contract_is_frozen():
    node = pnl.ProcessingMechanism(
        function=pnl.UserDefinedFunction(custom_function=lambda variable: variable),
        name="likelihood contract snapshot",
    )
    composition = pnl.Composition(pathways=node)
    observations = ObservationSpec((ObservationField(node.output_port, "counting"),))
    try:
        batched_node_op(node.name)(_identity)
        old = BatchedCompositionCompiler.compile(composition)
        assert "likelihood.effects_undeclared" in _codes(old.diagnose_likelihood(observations))
        batched_node_op(node.name, likelihood_contract=LikelihoodEffectContract())(_identity)
        new = BatchedCompositionCompiler.compile(composition)
        assert new.diagnose_likelihood(observations).factorization_status == "eligible"
        unregister_batched_instance_op(node.name)
        assert new.diagnose_likelihood(observations).factorization_status == "eligible"
        assert "likelihood.effects_undeclared" in _codes(old.diagnose_likelihood(observations))
    finally:
        unregister_batched_instance_op(node.name)


def test_effect_contract_cannot_deny_declared_rng():
    spec = MechanismOpSpec(
        mechanism_class=None, rng=(RngDecl("rng"),),
        likelihood_contract=LikelihoodEffectContract(),
    )
    with pytest.raises(BatchedOpSpecError, match="agree with declared RNG"):
        register_batched_instance_op("invalid effect declaration", spec)


@pytest.mark.parametrize("kwargs", [
    {"measure": "continuous"},
    {"recording": "rounded"},
    {"recording": "rounded", "precision": float("nan")},
    {"recording": "rounded", "precision": True},
    {"recording": "rounded", "precision": 0.1, "measure": "lebesgue"},
    {"precision": 0.1},
    {"score": False, "condition_history": False},
    {"score": 1},
])
def test_observation_contract_rejects_ambiguous_declarations(kwargs):
    with pytest.raises(ValueError):
        ObservationField(object(), **{"measure": "counting", **kwargs})


def test_unsupported_composition_exposes_simulation_blockers():
    node = pnl.ProcessingMechanism(
        function=pnl.UserDefinedFunction(custom_function=lambda variable: variable),
    )
    composition = pnl.Composition(pathways=node)
    report = _diagnose(
        composition, ObservationSpec((ObservationField(node.output_port, "counting"),)),
    )
    assert report.factorization_status == "blocked"
    assert "likelihood.simulation_ir_unavailable" in _codes(report)
    assert report.simulation.diagnostics
