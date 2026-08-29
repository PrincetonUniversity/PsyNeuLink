"""Tests for general batched execution-axis dependency analysis."""

import numpy as np

import psyneulink as pnl
from psyneulink.core.batched.compiler import BatchedCompositionCompiler
from psyneulink.core.batched.dependency import (
    ESTIMATE_AXIS,
    PARAMETER_SET_AXIS,
    SUBJECT_AXIS,
    TRIAL_AXIS,
    analyze_axis_dependencies,
)
from psyneulink.core.batched.ir import (
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedNodeSpec,
    BatchedParamSpec,
    BatchedProjectionSpec,
    BatchedRngStreamSpec,
    BatchedSchedulerSpec,
    BatchedStateSpec,
)


def _node(component_id, name, *, params=None):
    return BatchedNodeSpec(
        name,
        "test",
        "test",
        1,
        1,
        params={} if params is None else params,
        component_id=component_id,
    )


def _projection(producer, consumer, projection_id):
    return BatchedProjectionSpec(
        sender=f"node-{producer}",
        sender_port="RESULT",
        receiver=f"node-{consumer}",
        receiver_port="InputPort-0",
        matrix=np.ones((1, 1)),
        projection_id=projection_id,
        sender_component_id=producer,
        sender_port_id=producer * 2,
        receiver_component_id=consumer,
        receiver_port_id=consumer * 2 + 1,
    )


def _graph(
    nodes,
    *,
    inputs=(),
    projections=(),
    states=(),
    rng_streams=(),
    scheduler=(),
):
    return BatchedGraphIR(
        nodes=tuple(nodes),
        inputs=tuple(inputs),
        projections=tuple(projections),
        outputs=(),
        states=tuple(states),
        scheduler=tuple(scheduler),
        ops=(),
        execution_order=tuple(node.name for node in nodes),
        rng_streams=tuple(rng_streams),
        executable=False,
    )


def test_estimate_dependency_propagates_from_rng_at_stochastic_frontier():
    nodes = (
        _node(0, "node-0"),
        _node(1, "node-1", params={"gain": "gain"}),
        _node(2, "node-2"),
        _node(3, "node-3"),
    )
    graph = _graph(
        nodes,
        inputs=(BatchedInputSpec("stimulus", "node-0", 1, 0, 1),),
        projections=(
            _projection(0, 1, 0),
            _projection(1, 2, 1),
            _projection(2, 3, 2),
        ),
        states=(BatchedStateSpec("control.state", "node-1", 1, (0.0,), 1, 0),),
        rng_streams=(BatchedRngStreamSpec("decision.rng", "node-2", 1, "MAX_STEPS", 2, 0),),
    )
    params = (
        BatchedParamSpec(
            "gain",
            1.0,
            parameter_id=0,
            owner_component_id=1,
            owner_scope="node",
        ),
    )

    result = analyze_axis_dependencies(graph, params)

    assert result.node(0).axes == (SUBJECT_AXIS, TRIAL_AXIS)
    assert result.node(1).axes == (
        PARAMETER_SET_AXIS,
        SUBJECT_AXIS,
        TRIAL_AXIS,
    )
    assert result.node(2).axes == (
        PARAMETER_SET_AXIS,
        SUBJECT_AXIS,
        TRIAL_AXIS,
        ESTIMATE_AXIS,
    )
    assert result.node(3).estimate_dependent
    assert result.estimate_invariant_component_ids == (0, 1)
    assert result.estimate_dependent_component_ids == (2, 3)
    assert result.stochastic_root_component_ids == (2,)
    assert tuple(
        (
            edge.producer_component_id,
            edge.consumer_component_id,
            edge.kind,
        )
        for edge in result.estimate_frontier_edges
    ) == ((1, 2, "projection"),)


def test_scheduler_dependencies_propagate_estimate_axis_without_data_edge():
    nodes = tuple(_node(component_id, f"node-{component_id}") for component_id in range(3))
    graph = _graph(
        nodes,
        rng_streams=(BatchedRngStreamSpec("source.rng", "node-0", 1, "MAX_STEPS", 0, 0),),
        scheduler=(
            BatchedSchedulerSpec(
                node="node-1",
                condition_type="WhenFinished",
                dependencies=("node-0",),
                component_id=1,
                dependency_component_ids=(0,),
            ),
            BatchedSchedulerSpec(
                node="node-2",
                condition_type="EveryNCalls",
                dependencies=("node-1",),
                component_id=2,
                dependency_component_ids=(1,),
            ),
        ),
    )

    result = analyze_axis_dependencies(graph)

    assert result.estimate_dependent_component_ids == (0, 1, 2)
    assert result.estimate_frontier_edges == ()


def test_compiler_exposes_axis_analysis_without_changing_execution_plan():
    stimulus = pnl.TransferMechanism(input_shapes=1, name="axis-stimulus")
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.2,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="axis-decision",
    )
    composition = pnl.Composition(pathways=[[stimulus, decision]])

    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton_cpu",
        outputs=tuple(decision.output_ports),
        max_steps=32,
    )
    metadata = plan.kernel_ir.metadata["axis_dependencies"]
    nodes = {
        name: axes
        for _, name, _, axes in metadata["nodes"]
    }

    assert ESTIMATE_AXIS not in nodes[stimulus.name]
    assert ESTIMATE_AXIS in nodes[decision.name]
    assert metadata["estimate_frontier_edges"]
    assert (
        plan.capability_report.metadata["axis_dependencies"]
        == metadata
    )
