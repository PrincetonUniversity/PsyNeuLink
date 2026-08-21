"""KernelIR contract for typed, precomputed stateless schedule traces."""

from dataclasses import replace
import re

import pytest

import psyneulink as pnl
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import (
    BatchedCompositionIR,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.kernel_ir import (
    _component_trial_body_ops,
    _compose_component_trial_body_ops,
    _kernel_op_sequences_match_exactly,
    _trial_body_ops,
    _trial_output_ops,
    iter_kernel_ops,
    lower_to_kernel_ir,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _scheduled_lowering():
    source = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=2.0),
        name="always source",
    )
    receiver = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=-3.0),
        name="delayed receiver",
    )
    composition = pnl.Composition(pathways=[[source, receiver]])
    composition.scheduler.add_condition(source, pnl.Always())
    composition.scheduler.add_condition(receiver, pnl.AtPass(3))
    lowering = lower_composition(
        composition,
        outputs=(receiver.output_port,),
    )
    assert lowering.graph is not None
    return lowering, source, receiver


def _default_termination(graph):
    component_ids = tuple(sorted(condition.component_id for condition in graph.scheduler))
    return (
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_STATE_UPDATE",
            condition_type="AllHaveRun",
            dependency_component_ids=component_ids,
        ),
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_SEQUENCE",
            condition_type="Never",
        ),
    )


def _kernel(lowering, graph):
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _future_executable_graph(lowering, **metadata):
    graph = lowering.graph
    assert graph is not None
    return replace(
        graph,
        executable=True,
        termination=_default_termination(graph),
        metadata={**graph.metadata, **metadata},
    )


def test_component_builders_compose_to_the_canonical_trial_program():
    lowering, _, _ = _scheduled_lowering()
    graph = lowering.graph
    assert graph is not None

    component_ops = []
    diagnostic_slot = 0
    for node_name in graph.execution_order:
        body = _component_trial_body_ops(
            graph,
            node_name,
            diagnostic_slot=diagnostic_slot,
        )
        assert body
        assert all(op.kind != "StoreOutput" for op in body)
        component_ops.extend(body)
        diagnostic_slot += sum(op.kind == "StoreFlag" for op in body)

    ordered_ops = _compose_component_trial_body_ops(
        graph,
        graph.execution_order,
    )
    assert _kernel_op_sequences_match_exactly(tuple(component_ops), ordered_ops)
    assert _kernel_op_sequences_match_exactly(
        (*ordered_ops, *_trial_output_ops(graph)),
        _trial_body_ops(graph),
    )


def test_precomputed_trace_has_typed_nested_component_bodies_and_one_epilogue():
    lowering, source, receiver = _scheduled_lowering()
    graph = _future_executable_graph(lowering)
    kernel = _kernel(lowering, graph)

    assert kernel.termination == graph.termination
    assert kernel.schedule_trace is not None
    assert kernel.schedule_trace.component_execution_count == 5
    assert tuple(op.kind for op in kernel.ops) == ("ForPasses", "StoreOutput")

    pass_op, store = kernel.ops
    assert pass_op.attrs["declaration_only"] is False
    assert pass_op.attrs["trace_kind"] == "precomputed"
    assert store.target == f"{receiver.name}.RESULT"

    executions = pass_op.attrs["body"]
    assert tuple(
        (
            op.attrs["pass_index"],
            op.attrs["consideration_set_id"],
            op.attrs["component_ids"],
        )
        for op in executions
    ) == (
        (0, 0, (0,)),
        (1, 0, (0,)),
        (2, 0, (0,)),
        (3, 0, (0,)),
        (3, 1, (1,)),
    )
    assert all(op.kind == "ExecuteConsiderationSet" for op in executions)

    source_calls = tuple(
        child
        for execution in executions
        for child in execution.attrs["body"]
        if child.kind == "CallFunction" and child.target == source.name
    )
    assert len(source_calls) == 4
    assert all("onset_step" not in op.attrs for op in source_calls)
    assert all(
        "onset_step" not in op.attrs
        for op in iter_kernel_ops(kernel)
        if op.kind == "CallFunction"
    )
    assert sum(op.kind == "StoreOutput" for op in iter_kernel_ops(kernel)) == 1
    assert all(
        child.kind not in {"StoreOutput", "StoreFlag"}
        for execution in executions
        for child in execution.attrs["body"]
    )


def test_precomputed_trace_emits_directly_without_a_runtime_step_symbol():
    lowering, _, _ = _scheduled_lowering()
    kernel = _kernel(lowering, _future_executable_graph(lowering))

    source = triton_graph_kernel_source(kernel)

    assert re.search(r"\bstep\b", source) is None
    assert source.count("# precomputed scheduler pass") == 5
    assert source.count("tl.store(out + lane_out") == 1


def test_explicit_nonexecutable_graph_stays_declaration_only_with_termination():
    lowering, _, _ = _scheduled_lowering()
    graph = lowering.graph
    assert graph is not None
    graph = replace(
        graph,
        executable=False,
        termination=_default_termination(graph),
    )

    assert not graph.executable
    kernel = _kernel(lowering, graph)
    assert not kernel.executable
    assert kernel.schedule_trace is None
    assert tuple(op.kind for op in kernel.ops) == ("ForPasses",)
    assert kernel.ops[0].attrs["declaration_only"] is True
    with pytest.raises(
        ValueError,
        match="declaration-only, non-executable KernelIR",
    ):
        triton_graph_kernel_source(kernel)


def test_precomputed_trace_weighted_op_expansion_is_bounded_before_unrolling():
    lowering, _, _ = _scheduled_lowering()
    graph = _future_executable_graph(
        lowering,
        schedule_trace_weighted_op_budget=1,
    )

    with pytest.raises(
        ValueError,
        match=r"weighted op expansion \d+ exceeds budget 1",
    ):
        _kernel(lowering, graph)


def test_stateful_graph_does_not_enter_the_stateless_precomputed_path():
    mechanism = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        time_step_size=0.4,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=1,
        reset_stateful_function_when=pnl.Never(),
        name="stateful delayed mechanism",
    )
    composition = pnl.Composition(pathways=mechanism)
    composition.scheduler.add_condition(mechanism, pnl.AtPass(3))
    lowering = lower_composition(composition)
    graph = lowering.graph
    assert graph is not None
    graph = replace(
        graph,
        executable=True,
        termination=_default_termination(graph),
    )

    kernel = _kernel(lowering, graph)

    assert kernel.schedule_trace is None
    assert not kernel.executable
    assert tuple(op.kind for op in kernel.ops) == ("InitializeState", "ForTrials")
    trial_body = kernel.ops[1].attrs["body"]
    assert len(trial_body) == 1
    assert trial_body[0].kind == "ForPasses"
    assert trial_body[0].attrs["declaration_only"] is True


def test_missing_executable_component_body_remains_declaration_only():
    source = pnl.TransferMechanism(input_shapes=1, name="control source")
    target = pnl.TransferMechanism(input_shapes=1, name="control target")
    controller = pnl.ControlMechanism(
        function=pnl.Identity(),
        monitor_for_control=source,
        control_signals=[(pnl.SLOPE, target)],
        modulation=pnl.OVERRIDE,
        name="omitted controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([target, controller, source])
    composition.add_projection(sender=source, receiver=target)
    composition.scheduler.add_condition(controller, pnl.AtPass(0))
    lowering = lower_composition(composition)
    graph = lowering.graph
    assert graph is not None
    graph = replace(
        graph,
        executable=True,
        termination=_default_termination(graph),
        metadata={**graph.metadata, "schedule_kind": "precomputed_trace"},
    )

    kernel = _kernel(lowering, graph)

    assert kernel.schedule_trace is None
    assert not kernel.executable
    assert kernel.ops[0].kind == "ForPasses"
    assert kernel.ops[0].attrs["declaration_only"] is True
