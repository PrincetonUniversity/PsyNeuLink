"""Frozen GraphIR contract for scheduler-visible finished hooks."""

from dataclasses import replace

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    register_batched_instance_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import registry as batched_registry
from psyneulink.core.batched import specs
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _counted_finished_model():
    producer = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        leak=0.0,
        competition=0.0,
        self_excitation=0.0,
        noise=0.0,
        time_step_size=0.5,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.Never(),
        name="counting finished producer",
    )
    follower = pnl.TransferMechanism(
        input_shapes=1,
        name="counting finished follower",
    )
    composition = pnl.Composition(name="counting finished hook snapshot")
    composition.add_nodes([producer, follower])
    composition.add_projection(
        sender=producer,
        receiver=follower,
        projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
    )
    composition.scheduler.add_condition(producer, pnl.Always())
    composition.scheduler.add_condition(follower, pnl.WhenFinished(producer))
    return composition, producer, follower


def _kernel(lowering):
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


def test_finished_count_hook_runs_once_per_graph_snapshot(monkeypatch):
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda _backend: (True, ()),
    )
    composition, producer, follower = _counted_finished_model()
    specs.ensure_builtin_specs()
    original_spec = specs.mechanism_spec_for(producer)
    assert original_spec is not None
    hook_calls = []

    def counting_finished_hook(node, owner_composition):
        hook_calls.append((node, owner_composition))
        return 3

    register_batched_instance_op(
        producer.name,
        replace(
            original_spec,
            finished_after_execution_count=counting_finished_hook,
        ),
    )
    try:
        lowering = lower_composition(
            composition,
            outputs=(follower.output_port,),
        )
        assert hook_calls == [(producer, composition)]
        graph = lowering.graph
        assert graph is not None
        assert graph.executable
        assert graph.finished_values[0].attrs == {"count": 3}
        kernel = _kernel(lowering)
        assert kernel.schedule_trace is not None
        assert sum(
            op.kind == "StepMechanism" and op.target == producer.name
            for op in iter_kernel_ops(kernel)
        ) == 3
        assert hook_calls == [(producer, composition)]

        report = BatchedCompositionCompiler.diagnose(
            composition,
            backend="triton_cpu",
            outputs=(follower.output_port,),
        )
        assert report.can_execute
        assert len(hook_calls) == 2

        plan = BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
            outputs=(follower.output_port,),
        )
        assert plan.kernel_ir.finished_values[0].attrs == {"count": 3}
        assert plan.kernel_ir.schedule_trace is not None
        assert plan.kernel_ir.schedule_trace.component_execution_count == 4
        assert len(hook_calls) == 3
        assert all(
            node is producer and owner is composition
            for node, owner in hook_calls
        )
    finally:
        unregister_batched_instance_op(producer.name)
