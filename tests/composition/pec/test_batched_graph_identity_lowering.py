"""Composition-lowering tests for the stable numeric identity schema.

``test_batched_ir_identity_schema.py`` pins direct dataclass construction.  The
tests here pin the other half of the contract: lowering a live Composition must
populate those fields and the live-object binding sidecar consistently, without
making display names or target-language sanitization the source of identity.
"""

from dataclasses import dataclass
import re

import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, lower_to_kernel_ir


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@dataclass(frozen=True)
class _IdentityModel:
    composition: pnl.Composition
    nodes: dict[str, object]
    projections: dict[tuple[str, str], object]
    outputs: tuple


def _make_identity_model(prefix, *, reverse_insertion=False):
    # These two names deliberately collide after ordinary code-generator
    # sanitization.  Numeric IDs, not either spelling, must distinguish them.
    dashed = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(
            slope=2.0,
            intercept=0.25,
            scale=1.5,
            offset=-0.5,
        ),
        name=f"{prefix}-origin-a",
    )
    underscored = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(
            slope=-1.0,
            intercept=0.75,
            scale=0.5,
            offset=0.125,
        ),
        name=f"{prefix}-origin_a",
    )
    stateful = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=2.0),
        leak=1.0,
        competition=0.5,
        self_excitation=0.1,
        noise=0.0,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        time_step_size=0.01,
        name=f"{prefix}-stateful-target",
    )

    composition = pnl.Composition()
    insertion_order = (
        (stateful, underscored, dashed)
        if reverse_insertion
        else (dashed, underscored, stateful)
    )
    composition.add_nodes(list(insertion_order))

    dashed_projection = pnl.MappingProjection(
        matrix=[[1.0, 0.0], [0.0, 1.0]],
    )
    underscored_projection = pnl.MappingProjection(
        matrix=[[0.5, -1.0]],
    )
    projection_order = (
        (
            (underscored_projection, underscored),
            (dashed_projection, dashed),
        )
        if reverse_insertion
        else (
            (dashed_projection, dashed),
            (underscored_projection, underscored),
        )
    )
    for projection, sender in projection_order:
        composition.add_projection(
            projection=projection,
            sender=sender,
            receiver=stateful,
        )

    return _IdentityModel(
        composition=composition,
        nodes={
            "dashed": dashed,
            "underscored": underscored,
            "stateful": stateful,
        },
        projections={
            ("dashed", "stateful"): dashed_projection,
            ("underscored", "stateful"): underscored_projection,
        },
        # Reordered, mixed-width outputs make flattened ABI positions visible.
        outputs=(
            underscored.output_port,
            stateful.output_port,
            dashed.output_port,
        ),
    )


def _lower(model):
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert lowering.graph is not None
    return lowering


def _assert_contiguous_ids(values):
    values = tuple(values)
    assert values
    assert all(isinstance(value, int) and value >= 0 for value in values)
    assert len(values) == len(set(values))
    assert sorted(values) == list(range(len(values)))


def _safe_identifier(name):
    return re.sub(r"[^0-9A-Za-z_]", "_", name)


@pytest.mark.parametrize("reverse_insertion", [False, True])
def test_component_ids_are_deterministic_unique_and_sanitization_safe(
    reverse_insertion,
):
    model = _make_identity_model(
        "component-id",
        reverse_insertion=reverse_insertion,
    )
    first = _lower(model)
    second = _lower(model)

    first_ids = {node.name: node.component_id for node in first.graph.nodes}
    second_ids = {node.name: node.component_id for node in second.graph.nodes}
    assert first_ids == second_ids
    _assert_contiguous_ids(first_ids.values())

    dashed = model.nodes["dashed"]
    underscored = model.nodes["underscored"]
    assert _safe_identifier(dashed.name) == _safe_identifier(underscored.name)
    assert first_ids[dashed.name] != first_ids[underscored.name]


def test_parameter_ids_are_unique_and_bind_live_parameters():
    model = _make_identity_model("parameter-id")
    lowering = _lower(model)
    parameter_by_name = {parameter.name: parameter for parameter in lowering.params}
    _assert_contiguous_ids(parameter.parameter_id for parameter in lowering.params)
    assert set(lowering.bindings.parameters_by_id) == {
        parameter.parameter_id for parameter in lowering.params
    }

    for node_spec in lowering.graph.nodes:
        node = lowering.bindings.node(node_spec.name)
        owners = (
            getattr(node, "function", None),
            node,
            getattr(node, "integrator_function", None),
        )
        for argument, public_name in node_spec.params.items():
            parameter_spec = parameter_by_name[public_name]
            live_parameter = lowering.bindings.parameter_by_id(
                parameter_spec.parameter_id
            )
            candidates = [
                getattr(getattr(owner, "parameters", None), argument, None)
                for owner in owners
            ]
            assert any(live_parameter is candidate for candidate in candidates)


def test_port_ids_preserve_exact_live_port_identity():
    model = _make_identity_model("port-id")
    lowering = _lower(model)
    graph = lowering.graph
    bindings = lowering.bindings
    node_by_name = {node.name: node for node in model.nodes.values()}

    referenced = []
    for input_spec in graph.inputs:
        node = node_by_name[input_spec.node]
        port = node.input_ports[0]
        assert input_spec.component_id == graph.node(node.name).component_id
        assert input_spec.port == port.name
        assert bindings.port_by_id(input_spec.port_id) is port
        referenced.append((input_spec.port_id, port))

    for output_spec, output_port in zip(graph.outputs, model.outputs):
        assert output_spec.node == output_port.owner.name
        assert output_spec.port == output_port.name
        assert output_spec.component_id == graph.node(output_port.owner.name).component_id
        assert bindings.port_by_id(output_spec.port_id) is output_port
        referenced.append((output_spec.port_id, output_port))

    for projection_spec in graph.projections:
        projection = bindings.projection(
            projection_spec.sender,
            projection_spec.sender_port,
            projection_spec.receiver,
            projection_spec.receiver_port,
        )
        assert projection_spec.sender_port == projection.sender.name
        assert projection_spec.receiver_port == projection.receiver.name
        assert bindings.port_by_id(projection_spec.sender_port_id) is projection.sender
        assert bindings.port_by_id(projection_spec.receiver_port_id) is projection.receiver
        referenced.extend(
            (
                (projection_spec.sender_port_id, projection.sender),
                (projection_spec.receiver_port_id, projection.receiver),
            )
        )

    ids_by_live_port = {}
    for port_id, port in referenced:
        assert isinstance(port_id, int) and port_id >= 0
        ids_by_live_port.setdefault(id(port), set()).add(port_id)
    assert all(len(port_ids) == 1 for port_ids in ids_by_live_port.values())
    assert len({next(iter(ids)) for ids in ids_by_live_port.values()}) == len(
        ids_by_live_port
    )
    _assert_contiguous_ids(lowering.bindings.ports_by_id)


def test_projection_ids_and_endpoint_ids_bind_the_exact_live_edges():
    model = _make_identity_model("projection-id", reverse_insertion=True)
    lowering = _lower(model)
    graph = lowering.graph
    _assert_contiguous_ids(
        projection.projection_id for projection in graph.projections
    )
    assert set(lowering.bindings.projections_by_id) == {
        projection.projection_id for projection in graph.projections
    }

    node_roles = {node.name: role for role, node in model.nodes.items()}
    for projection_spec in graph.projections:
        route = (
            node_roles[projection_spec.sender],
            node_roles[projection_spec.receiver],
        )
        projection = model.projections[route]
        assert lowering.bindings.projection_by_id(
            projection_spec.projection_id
        ) is projection
        assert projection_spec.sender_component_id == graph.node(
            projection.sender.owner.name
        ).component_id
        assert projection_spec.receiver_component_id == graph.node(
            projection.receiver.owner.name
        ).component_id
        assert lowering.bindings.port_by_id(
            projection_spec.sender_port_id
        ) is projection.sender
        assert lowering.bindings.port_by_id(
            projection_spec.receiver_port_id
        ) is projection.receiver


def test_state_and_rng_ids_are_unique_and_owned_by_the_stateful_component():
    model = _make_identity_model("state-rng-id")
    lowering = _lower(model)
    graph = lowering.graph
    stateful = model.nodes["stateful"]
    component_id = graph.node(stateful.name).component_id

    assert {state.name.removeprefix(f"{stateful.name}.") for state in graph.states} == {
        "act",
        "pre",
    }
    _assert_contiguous_ids(state.state_id for state in graph.states)
    assert all(state.component_id == component_id for state in graph.states)

    assert len(graph.rng_streams) == 1
    _assert_contiguous_ids(stream.stream_id for stream in graph.rng_streams)
    stream = graph.rng_streams[0]
    assert stream.node == stateful.name
    assert stream.component_id == component_id
    assert stream.width == 2
    assert stream.step_extent == "LCA_MAX_STEPS"


def test_output_flat_slices_follow_explicit_mixed_width_order():
    model = _make_identity_model("output-slice")
    lowering = _lower(model)
    expected = (
        (model.nodes["underscored"].output_port, 0, 1),
        (model.nodes["stateful"].output_port, 1, 3),
        (model.nodes["dashed"].output_port, 3, 5),
    )

    assert len(lowering.graph.outputs) == len(expected)
    for output_spec, (port, start, stop) in zip(lowering.graph.outputs, expected):
        assert output_spec.node == port.owner.name
        assert output_spec.port == port.name
        assert output_spec.width == stop - start
        assert output_spec.flat_start == start
        assert output_spec.flat_stop == stop
        assert output_spec.flat_slice == slice(start, stop)


def test_kernel_ir_operations_retain_graph_identity():
    model = _make_identity_model("kernel-identity")
    lowering = _lower(model)
    graph = lowering.graph
    kernel = lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )
    operations = iter_kernel_ops(kernel)

    projection_by_id = {
        projection.projection_id: projection for projection in graph.projections
    }
    projection_ops = [op for op in operations if op.kind == "CallProjection"]
    assert len(projection_ops) == len(projection_by_id)
    for operation in projection_ops:
        projection = projection_by_id[operation.attrs["projection_id"]]
        assert operation.attrs["sender_component_id"] == projection.sender_component_id
        assert operation.attrs["sender_port_id"] == projection.sender_port_id
        assert operation.attrs["receiver_component_id"] == projection.receiver_component_id
        assert operation.attrs["receiver_port_id"] == projection.receiver_port_id

    input_by_component = {
        input_spec.component_id: input_spec for input_spec in graph.inputs
    }
    for operation in (op for op in operations if op.kind == "LoadInput"):
        input_spec = input_by_component[operation.attrs["component_id"]]
        assert operation.attrs["port_id"] == input_spec.port_id

    output_ops = [op for op in operations if op.kind == "StoreOutput"]
    assert [op.attrs["flat_start"] for op in output_ops] == [0, 1, 3]
    assert [op.attrs["flat_stop"] for op in output_ops] == [1, 3, 5]
    assert [op.attrs["port_id"] for op in output_ops] == [
        output.port_id for output in graph.outputs
    ]
    assert [stream.stream_id for stream in kernel.rng_streams] == [0]


def test_numeric_node_and_function_bindings_are_live_and_complete():
    model = _make_identity_model("component-binding")
    lowering = _lower(model)
    component_ids = {node.component_id for node in lowering.graph.nodes}
    assert set(lowering.bindings.nodes_by_id) == component_ids
    assert set(lowering.bindings.functions_by_id) == component_ids

    for node_spec in lowering.graph.nodes:
        live_node = next(
            node for node in model.nodes.values() if node.name == node_spec.name
        )
        assert lowering.bindings.node_by_id(node_spec.component_id) is live_node
        assert lowering.bindings.function_by_id(
            node_spec.component_id
        ) is live_node.function


def _normalized_identity_signature(model, lowering):
    graph = lowering.graph
    role_by_name = {node.name: role for role, node in model.nodes.items()}
    parameter_by_name = {parameter.name: parameter for parameter in lowering.params}
    signature = {
        "components": {
            role_by_name[node.name]: node.component_id for node in graph.nodes
        },
        "parameters": {
            (role_by_name[node.name], argument): parameter_by_name[public_name].parameter_id
            for node in graph.nodes
            for argument, public_name in node.params.items()
        },
        "inputs": {
            role_by_name[input_spec.node]: (
                input_spec.component_id,
                input_spec.port_id,
            )
            for input_spec in graph.inputs
        },
        "outputs": tuple(
            (
                role_by_name[output.node],
                output.component_id,
                output.port_id,
                output.flat_start,
                output.flat_stop,
            )
            for output in graph.outputs
        ),
        "projections": {
            (role_by_name[projection.sender], role_by_name[projection.receiver]): (
                projection.projection_id,
                projection.sender_component_id,
                projection.sender_port_id,
                projection.receiver_component_id,
                projection.receiver_port_id,
            )
            for projection in graph.projections
        },
        "states": {
            state.name.removeprefix(f"{state.node}."): (
                state.component_id,
                state.state_id,
            )
            for state in graph.states
        },
        "rng": {
            stream.name.removeprefix(f"{stream.node}."): (
                stream.component_id,
                stream.stream_id,
            )
            for stream in graph.rng_streams
        },
    }
    return signature


def test_identity_assignment_is_insertion_and_display_name_invariant():
    ordinary = _make_identity_model("ordinary", reverse_insertion=False)
    renamed_reversed = _make_identity_model("renamed", reverse_insertion=True)

    assert _normalized_identity_signature(ordinary, _lower(ordinary)) == (
        _normalized_identity_signature(renamed_reversed, _lower(renamed_reversed))
    )
