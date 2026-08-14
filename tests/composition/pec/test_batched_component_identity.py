import psyneulink as pnl

from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir


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


def test_component_ids_make_sanitized_value_names_distinct_and_stable():
    dashed = pnl.TransferMechanism(input_shapes=1, name="identity-collision")
    underscored = pnl.TransferMechanism(input_shapes=1, name="identity_collision")
    receiver = pnl.TransferMechanism(input_shapes=1, name="identity-result")
    composition = pnl.Composition()
    # Deliberately use neither dependency order nor lexical order.
    composition.add_nodes([receiver, underscored, dashed])
    composition.add_projection(sender=dashed, receiver=receiver)
    composition.add_projection(sender=underscored, receiver=receiver)

    first = lower_composition(composition)
    second = lower_composition(composition)
    assert first.graph is not None
    assert second.graph is not None

    first_ids = {node.name: node.component_id for node in first.graph.nodes}
    second_ids = {node.name: node.component_id for node in second.graph.nodes}
    assert first_ids == second_ids
    assert tuple(
        first.graph.node(name).component_id
        for name in first.graph.execution_order
    ) == (0, 1, 2)

    kernel = _kernel_ir(first)
    function_outputs = {
        op.target: op.outputs[0].name
        for op in iter_kernel_ops(kernel)
        if op.kind == "CallFunction"
    }
    assert function_outputs[dashed.name] == "n0:output:0"
    assert function_outputs[underscored.name] == "n1:output:0"
    assert function_outputs[dashed.name] != function_outputs[underscored.name]

    source = triton_graph_kernel_source(kernel)
    assert "n0_output_0_0" in source
    assert "n1_output_0_0" in source
    # Display names continue to own the external semantic contract.
    assert kernel.graph.node(dashed.name).name == dashed.name
    assert kernel.graph.node(underscored.name).name == underscored.name
    assert kernel.output_names == (f"{receiver.name}.RESULT",)


def test_state_and_diagnostic_symbols_use_component_ids():
    stateful = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        name="state-collision",
    )
    terminator = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="state_collision",
    )
    composition = pnl.Composition()
    composition.add_nodes([terminator, stateful])
    composition.scheduler.add_condition(stateful, pnl.Always())

    lowering = lower_composition(composition)
    assert lowering.graph is not None
    assert lowering.graph.node(stateful.name).component_id == 0
    assert lowering.graph.node(terminator.name).component_id == 1
    assert {
        state.component_id
        for state in lowering.graph.states
        if state.node == stateful.name
    } == {0}

    kernel = _kernel_ir(lowering)
    initialize = kernel.ops[0]
    assert initialize.kind == "InitializeState"
    assert tuple(value.name for value in initialize.outputs) == (
        "n0:state:0",
        "n0:state:1",
    )
    store_flag = next(op for op in iter_kernel_ops(kernel) if op.kind == "StoreFlag")
    assert store_flag.inputs[0].name == "n1:diagnostic:0"

    source = triton_graph_kernel_source(kernel)
    assert "n0_state_0_0" in source
    assert "n0_state_1_0" in source
    assert "n0_lca_steps" in source
    assert "n1_diagnostic_0_0" in source
