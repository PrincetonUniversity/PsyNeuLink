"""Structural contract for declaration-only batched scheduler lowering."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION, lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.scheduling.condition import Condition


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _kernel_ir(lowering):
    graph = lowering.graph
    assert graph is not None
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _assert_scheduler_data_only(value):
    """Scheduler declarations must never retain live PNL graph objects."""

    if is_dataclass(value) and not isinstance(value, type):
        assert type(value).__module__.startswith("psyneulink.core.batched")
        for item in fields(value):
            _assert_scheduler_data_only(getattr(value, item.name))
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_scheduler_data_only(key)
            _assert_scheduler_data_only(item)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_scheduler_data_only(item)
        return
    assert value is None or isinstance(value, (str, int, float, bool))


def _static_scheduler_model():
    source = pnl.TransferMechanism(input_shapes=1, name="a source")
    trial_start = pnl.TransferMechanism(input_shapes=1, name="b trial start")
    middle = pnl.TransferMechanism(input_shapes=1, name="c middle")
    target = pnl.TransferMechanism(input_shapes=1, name="d target")
    composition = pnl.Composition()
    # Deliberately neither dependency nor scheduler-condition order.
    composition.add_nodes([target, middle, trial_start, source])
    composition.add_projection(sender=source, receiver=middle)
    composition.add_projection(sender=middle, receiver=target)
    composition.scheduler.add_condition(target, pnl.WhenFinished(middle))
    composition.scheduler.add_condition(trial_start, pnl.AtTrialStart())
    composition.scheduler.add_condition(source, pnl.AtPass(0))
    composition.scheduler.add_condition(middle, pnl.Always())
    return composition


def test_static_scheduler_declarations_preserve_regions_sets_and_finished_ids():
    lowering = lower_composition(_static_scheduler_model())

    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    graph = lowering.graph
    assert graph is not None
    assert tuple((region.name, region.parent) for region in graph.schedule_regions) == (
        ("trial", ""),
        ("pass", "trial"),
    )
    assert tuple(
        (item.consideration_set_id, item.nodes, item.component_ids, item.inputs_frozen)
        for item in graph.consideration_sets
    ) == (
        (0, ("a source", "b trial start"), (0, 1), True),
        (1, ("c middle",), (2,), True),
        (2, ("d target",), (3,), True),
    )

    scheduler = {condition.node: condition for condition in graph.scheduler}
    assert tuple(condition.node for condition in graph.scheduler) == (
        "a source",
        "b trial start",
        "c middle",
        "d target",
    )
    assert scheduler["a source"].condition_type == "AtPass"
    assert scheduler["a source"].attrs == {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert scheduler["a source"].consideration_set_id == 0
    assert scheduler["b trial start"].condition_type == "AtTrialStart"
    assert scheduler["b trial start"].region == "pass"
    assert scheduler["b trial start"].attrs == {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert scheduler["c middle"].condition_type == "Always"

    finished = graph.finished_values
    assert len(finished) == 1
    assert (
        finished[0].node,
        finished[0].component_id,
        finished[0].value_id,
        finished[0].producer_consideration_set_id,
    ) == ("c middle", 2, 0, 1)
    assert finished[0].storage == "combinational"
    assert scheduler["d target"].dependencies == ("c middle",)
    assert scheduler["d target"].dependency_component_ids == (2,)
    assert scheduler["d target"].finished_value_ids == (0,)
    assert scheduler["d target"].consideration_set_id == 2

    kernel = _kernel_ir(lowering)
    assert kernel.scheduler == graph.scheduler
    assert kernel.schedule_regions == graph.schedule_regions
    assert kernel.consideration_sets == graph.consideration_sets
    assert kernel.finished_values == graph.finished_values
    # These predicates are equivalent to the existing one-pass topological
    # execution, so accepted static graphs retain their exact old op shape.
    assert all(op.kind != "ForPasses" for op in kernel.ops)
    _assert_scheduler_data_only(graph.scheduler)
    _assert_scheduler_data_only(graph.schedule_regions)
    _assert_scheduler_data_only(graph.consideration_sets)
    _assert_scheduler_data_only(graph.finished_values)


def test_delayed_at_pass_is_declared_but_execution_remains_fail_closed():
    mechanism = pnl.TransferMechanism(input_shapes=1, name="delayed")
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, pnl.AtPass(3))

    lowering = lower_composition(composition)
    assert lowering.graph is not None
    assert len(lowering.rejected_conditions) == 1
    condition = lowering.graph.scheduler[0]
    assert condition.condition_type == "AtPass"
    assert condition.attrs == {
        "pass_index": 3,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert not lowering.graph.metadata["scheduler_executable"]
    kernel = _kernel_ir(lowering)
    assert tuple(op.kind for op in kernel.ops) == ("ForPasses",)
    pass_op = kernel.ops[0]
    assert pass_op.attrs["declaration_only"] is True
    assert pass_op.attrs["conditions"] == lowering.graph.scheduler
    assert pass_op.attrs["consideration_sets"] == lowering.graph.consideration_sets
    _assert_scheduler_data_only(
        {
            key: value
            for key, value in pass_op.attrs.items()
            if key != "body"
        }
    )

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    assert any("AtPass" in reason for reason in report.unsupported_reasons)
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


def test_triton_emitter_enforces_kernel_executability_independently_of_ops():
    kernel = _kernel_ir(lower_composition(_static_scheduler_model()))
    assert kernel.executable
    assert all(op.kind != "ForPasses" for op in kernel.ops)

    with pytest.raises(ValueError, match="declaration-only, non-executable KernelIR"):
        triton_graph_kernel_source(replace(kernel, executable=False))


@pytest.mark.parametrize(
    "condition_factory, expected_type",
    (
        (lambda source: pnl.AtPass(0), "AtPass"),
        (lambda source: pnl.WhenFinished(source), "WhenFinished"),
    ),
)
def test_control_conditions_are_retained_even_when_control_execution_rejects(
    condition_factory,
    expected_type,
):
    source = pnl.TransferMechanism(input_shapes=1, name="control source")
    target = pnl.TransferMechanism(input_shapes=1, name="control target")
    controller = pnl.ControlMechanism(
        function=pnl.Identity(),
        monitor_for_control=source,
        control_signals=[(pnl.SLOPE, target)],
        modulation=pnl.OVERRIDE,
        name="generic controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([target, controller, source])
    composition.add_projection(sender=source, receiver=target)
    composition.scheduler.add_condition(
        controller,
        condition_factory(source),
    )

    lowering = lower_composition(composition)
    assert lowering.graph is not None
    assert any(
        diagnostic.component == controller.name
        for diagnostic in lowering.rejected_nodes
    )
    controller_spec = lowering.graph.node(controller.name)
    condition = next(
        item
        for item in lowering.graph.scheduler
        if item.component_id == controller_spec.component_id
    )
    assert condition.node == controller.name
    assert condition.condition_type == expected_type
    owning_set = next(
        item
        for item in lowering.graph.consideration_sets
        if controller_spec.component_id in item.component_ids
    )
    assert condition.consideration_set_id == owning_set.consideration_set_id
    if expected_type == "WhenFinished":
        source_spec = lowering.graph.node(source.name)
        assert condition.dependencies == (source.name,)
        assert condition.dependency_component_ids == (source_spec.component_id,)
        assert condition.finished_value_ids == (0,)
    _assert_scheduler_data_only(condition)
    assert not lowering.graph.executable
    kernel = _kernel_ir(lowering)
    assert not kernel.executable
    assert kernel.ops[0].kind == "ForPasses"
    with pytest.raises(ValueError, match="declaration-only, non-executable KernelIR"):
        triton_graph_kernel_source(kernel)

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


@pytest.mark.parametrize(
    "producer_name, dependent_name",
    (
        ("A DDM", "Z dependent"),
        ("Z DDM", "A dependent"),
    ),
)
def test_same_consideration_set_when_finished_rejects_independent_of_names(
    producer_name,
    dependent_name,
):
    producer = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name=producer_name,
    )
    dependent = pnl.TransferMechanism(input_shapes=1, name=dependent_name)
    composition = pnl.Composition()
    composition.add_nodes([producer, dependent])
    composition.scheduler.add_condition(dependent, pnl.WhenFinished(producer))

    lowering = lower_composition(composition)
    graph = lowering.graph
    assert graph is not None
    assert len(graph.consideration_sets) == 1
    condition = next(item for item in graph.scheduler if item.node == dependent.name)
    assert condition.consideration_set_id == 0
    assert graph.finished_values[0].producer_consideration_set_id == 0
    assert len(lowering.rejected_conditions) == 1
    assert lowering.schedule_kind == "dynamic_lane_local"
    assert not graph.executable
    assert _kernel_ir(lowering).ops[0].kind == "ForPasses"

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    assert any("WhenFinished" in reason for reason in report.unsupported_reasons)


def test_condition_name_impostor_is_not_a_typed_scheduler_predicate():
    impostor_always = type("Always", (Condition,), {})(lambda: False)
    mechanism = pnl.TransferMechanism(input_shapes=1, name="impostor target")
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, impostor_always)

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert len(lowering.rejected_conditions) == 1
    assert lowering.rejected_conditions[0].detail == "Always"
    assert lowering.supported_conditions == ()

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    assert not any(
        diagnostic.code == BatchedDiagnosticCode.MODEL_TOPOLOGY_UNSUPPORTED
        for diagnostic in report.model_diagnostics
    )
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


@pytest.mark.parametrize(
    "invalid_pass",
    (None, "bad", False, 0.5, -1, float("inf"), float("-inf")),
)
def test_malformed_at_pass_rejects_without_lowering_crash(invalid_pass):
    mechanism = pnl.TransferMechanism(input_shapes=1, name="malformed onset")
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, pnl.AtPass(invalid_pass))

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert len(lowering.rejected_conditions) == 1
    diagnostic = lowering.rejected_conditions[0]
    assert diagnostic.reason == "unsupported scheduler condition for static batched graph"
    assert "requires one non-negative non-bool integer index" in diagnostic.detail

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


def test_nondefault_at_pass_clock_is_declaration_incomplete():
    mechanism = pnl.TransferMechanism(input_shapes=1, name="wrong clock")
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(
        mechanism,
        pnl.AtPass(0, time_scale=pnl.TimeScale.ENVIRONMENT_SEQUENCE),
    )

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert len(lowering.rejected_conditions) == 1
    assert "time_scale=ENVIRONMENT_SEQUENCE" in lowering.rejected_conditions[0].detail


@pytest.mark.parametrize(
    "condition_owner, condition_factory, expected_detail",
    (
        ("third", lambda first, second, third: pnl.BeforeNode(first), "BeforeNode"),
        ("second", lambda first, second, third: pnl.AfterNodes(third), "AfterNodes"),
    ),
)
def test_structural_scheduler_conditions_fail_closed(
    condition_owner,
    condition_factory,
    expected_detail,
):
    first = pnl.TransferMechanism(input_shapes=1, name="structural first")
    second = pnl.TransferMechanism(input_shapes=1, name="structural second")
    third = pnl.TransferMechanism(input_shapes=1, name="structural third")
    composition = pnl.Composition(pathways=[[first, second, third]])
    owners = {"first": first, "second": second, "third": third}
    owner = owners[condition_owner]
    composition.scheduler.add_condition(
        owner,
        condition_factory(first, second, third),
    )

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert lowering.schedule_kind == "unsupported"
    assert lowering.supported_conditions == ()
    assert len(lowering.rejected_conditions) == 1
    diagnostic = lowering.rejected_conditions[0]
    assert diagnostic.component == owner.name
    assert diagnostic.reason == (
        "unsupported structural scheduler condition for batched v2"
    )
    assert diagnostic.detail == expected_detail

    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    assert report.model_diagnostics[0].code == (
        BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED
    )
    assert not any(
        item.code == BatchedDiagnosticCode.MODEL_TOPOLOGY_UNSUPPORTED
        for item in report.model_diagnostics
    )
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


def test_mixed_delayed_graph_declares_scheduler_effective_implicit_defaults():
    delayed = pnl.TransferMechanism(input_shapes=1, name="delayed source")
    downstream = pnl.TransferMechanism(input_shapes=1, name="implicit downstream")
    composition = pnl.Composition(pathways=[[delayed, downstream]])
    composition.scheduler.add_condition(delayed, pnl.AtPass(3))

    lowering = lower_composition(composition)
    graph = lowering.graph
    assert graph is not None
    assert not graph.executable
    scheduler = {condition.node: condition for condition in graph.scheduler}
    assert scheduler[delayed.name].condition_type == "AtPass"
    assert scheduler[downstream.name].condition_type == "EveryNCalls"
    assert scheduler[downstream.name].dependencies == (delayed.name,)
    assert scheduler[downstream.name].dependency_component_ids == (
        graph.node(delayed.name).component_id,
    )
    assert scheduler[downstream.name].attrs == {
        "implicit": True,
        "calls": 1,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    assert scheduler[downstream.name].consideration_set_id == 1
    assert {condition.component_id for condition in graph.scheduler} == {
        node.component_id for node in graph.nodes
    }


def test_implicit_multi_parent_condition_declares_all_dependency_call_counts():
    left = pnl.TransferMechanism(input_shapes=1, name="left parent")
    right = pnl.TransferMechanism(input_shapes=1, name="right parent")
    child = pnl.TransferMechanism(input_shapes=1, name="multi-parent child")
    composition = pnl.Composition()
    composition.add_nodes([child, right, left])
    composition.add_projection(sender=left, receiver=child)
    composition.add_projection(sender=right, receiver=child)

    graph = lower_composition(composition).graph
    assert graph is not None
    condition = next(item for item in graph.scheduler if item.node == child.name)
    assert condition.condition_type == "AllEveryNCalls"
    assert condition.dependencies == (left.name, right.name)
    assert condition.dependency_component_ids == (
        graph.node(left.name).component_id,
        graph.node(right.name).component_id,
    )
    assert condition.attrs == {
        "implicit": True,
        "calls": 1,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }


def test_materialized_implicit_conditions_do_not_change_lowering():
    source = pnl.TransferMechanism(input_shapes=1, name="implicit source")
    target = pnl.TransferMechanism(input_shapes=1, name="implicit target")
    composition = pnl.Composition(pathways=[[source, target]])

    before = lower_composition(composition)
    assert before.graph is not None
    assert before.schedule_kind == "static_graph"
    assert not before.rejected_conditions
    before_scheduler = before.graph.scheduler

    # graph-scheduler creates its implicit Always/EveryNCalls objects lazily.
    # This is execution history, not a semantic change to the Composition.
    list(composition.scheduler.run())
    assert composition.scheduler.conditions.conditions_basic

    after = lower_composition(composition)
    assert after.graph is not None
    assert after.schedule_kind == before.schedule_kind
    assert not after.rejected_conditions
    assert after.graph.scheduler == before_scheduler
    assert after.graph.executable


def _coevolving_model():
    stepper = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=2.0),
        leak=1.0,
        competition=0.5,
        self_excitation=0.0,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        time_step_size=0.01,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.Never(),
        name="persistent stepper",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.02,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        name="lane terminator",
    )
    composition = pnl.Composition()
    composition.add_nodes([terminator, stepper])
    composition.add_projection(
        sender=stepper,
        receiver=terminator,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(stepper, pnl.Always())
    composition.scheduler.add_condition(terminator, pnl.WhenFinished(stepper))
    return composition, stepper, terminator


def test_coevolution_declares_nested_pass_and_finished_transition():
    composition, stepper, terminator = _coevolving_model()
    lowering = lower_composition(
        composition,
        outputs=tuple(terminator.output_ports),
    )

    graph = lowering.graph
    assert graph is not None
    assert graph.fusion_kind == COEVOLVING_GRAPH_FUSION
    assert len(lowering.rejected_conditions) == 1
    assert tuple(item.nodes for item in graph.consideration_sets) == (
        (stepper.name,),
        (terminator.name,),
    )
    terminator_condition = next(
        item for item in graph.scheduler if item.node == terminator.name
    )
    assert terminator_condition.consideration_set_id == 1
    assert graph.finished_values[0].producer_consideration_set_id == 0

    kernel = _kernel_ir(lowering)
    assert tuple(op.kind for op in kernel.ops) == ("InitializeState", "ForTrials")
    trial_body = kernel.ops[1].attrs["body"]
    assert len(trial_body) == 1
    assert trial_body[0].kind == "ForPasses"
    assert trial_body[0].attrs["declaration_only"] is True


def _lca_with_reset(reset_condition):
    return pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        reset_stateful_function_when=reset_condition,
        name="retained state",
    )


def test_retained_state_reset_policy_is_typed_or_rejected():
    persistent = _lca_with_reset(pnl.Never())
    lowering = lower_composition(pnl.Composition(pathways=persistent))

    graph = lowering.graph
    assert graph is not None
    assert len(graph.resets) == 1
    reset = graph.resets[0]
    assert reset.node == persistent.name
    assert reset.component_id == graph.node(persistent.name).component_id
    assert reset.condition_type == "Never"
    assert reset.state_ids == tuple(
        state.state_id
        for state in graph.states
        if state.component_id == reset.component_id
    )
    assert _kernel_ir(lowering).resets == graph.resets
    _assert_scheduler_data_only(graph.resets)

    reset_each_trial = _lca_with_reset(pnl.AtTrialStart())
    rejected = lower_composition(pnl.Composition(pathways=reset_each_trial))
    assert rejected.graph is None
    assert any(
        diagnostic.component == reset_each_trial.name
        and diagnostic.reason == "unsupported LCA reset policy for batched v2"
        and diagnostic.detail == "AtTrialStart"
        for diagnostic in rejected.rejected_nodes
    )


@pytest.mark.parametrize(
    "reset_condition, expected_detail",
    (
        (pnl.Never(), "Never"),
        (pnl.AtPass(0), "AtPass"),
        (Condition(lambda: False), "Condition"),
    ),
)
def test_emitter_private_ddm_state_rejects_unmodeled_reset_policy(
    reset_condition,
    expected_detail,
):
    ddm = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            noise=0.0,
            threshold=0.05,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        reset_stateful_function_when=reset_condition,
        name="reset-sensitive DDM",
    )
    composition = pnl.Composition(pathways=ddm)

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert any(
        diagnostic.component == ddm.name
        and diagnostic.reason == "unsupported DDM reset policy for batched v2"
        and diagnostic.detail == expected_detail
        for diagnostic in lowering.rejected_nodes
    )
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


def test_stepwise_ddm_rejects_without_typed_coevolution_pair():
    ddm = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            noise=0.0,
            threshold=0.05,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        execute_until_finished=False,
        name="orphan stepwise DDM",
    )
    composition = pnl.Composition(pathways=ddm)

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert any(
        diagnostic.component == ddm.name
        and diagnostic.reason == "unsupported DDM execution mode for batched v2"
        and "Always/WhenFinished" in diagnostic.detail
        for diagnostic in lowering.rejected_nodes
    )
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert not report.is_supported
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")


def test_integrating_transfer_rejects_reset_condition_name_impostor():
    impostor_reset = type("AtTrialStart", (Condition,), {})(lambda: False)
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(),
        integrator_mode=True,
        integration_rate=1.0,
        reset_stateful_function_when=impostor_reset,
        name="stateful transfer",
    )
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, pnl.AtPass(0))

    lowering = lower_composition(composition)
    assert lowering.graph is None
    assert any(
        diagnostic.component == mechanism.name
        and diagnostic.reason
        == "unsupported stateful transfer (integrator_mode) for batched v2"
        for diagnostic in lowering.rejected_nodes
    )
