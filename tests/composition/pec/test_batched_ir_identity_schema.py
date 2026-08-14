import numpy as np
import pytest

from psyneulink.core.batched import (
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedNodeSpec,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedProjectionSpec,
    BatchedRngStreamSpec,
    BatchedStateSpec,
)
from psyneulink.core.batched.bindings import BatchedComponentBindings
from psyneulink.core.batched.kernel_ir import KernelIR, KernelLaneLayout
from psyneulink.core.batched.specs import BatchedOpSpecSnapshot


pytestmark = pytest.mark.batched


def test_identity_fields_preserve_legacy_direct_construction():
    parameter = BatchedParamSpec("gain", 1.0, ("node.gain",))
    input_spec = BatchedInputSpec("node", "node", 2)
    output = BatchedOutputSpec("node.RESULT", "node", "RESULT", 2)
    projection = BatchedProjectionSpec(
        "source",
        "RESULT",
        "node",
        "InputPort-0",
        np.eye(2),
    )
    state = BatchedStateSpec("node.state", "node", 2, (0.0, 0.0))
    graph = BatchedGraphIR(
        (),
        (input_spec,),
        (projection,),
        (output,),
        (state,),
        (),
        (),
        (),
    )
    bindings = BatchedComponentBindings({"node": object()}, {}, {})

    assert parameter.parameter_id == -1
    assert (input_spec.component_id, input_spec.port_id, input_spec.port) == (-1, -1, "")
    assert (output.component_id, output.port_id) == (-1, -1)
    assert (output.flat_start, output.flat_stop) == (-1, -1)
    with pytest.raises(ValueError, match="no flattened slice assignment"):
        output.flat_slice
    assert projection.projection_id == -1
    assert projection.sender_component_id == -1
    assert projection.receiver_port_id == -1
    assert (state.component_id, state.state_id) == (-1, -1)
    assert graph.rng_streams == ()
    assert bindings.nodes_by_id == {}


def test_numeric_identity_and_output_slice_survive_direct_kernel_structure():
    parameter = BatchedParamSpec("gain", 1.0, parameter_id=20)
    input_spec = BatchedInputSpec(
        "node.InputPort-0",
        "node",
        2,
        component_id=10,
        port_id=30,
        port="InputPort-0",
    )
    output = BatchedOutputSpec(
        "node.RESULT",
        "node",
        "RESULT",
        2,
        component_id=10,
        port_id=31,
        flat_start=3,
        flat_stop=5,
    )
    projection = BatchedProjectionSpec(
        sender="source",
        sender_port="RESULT",
        receiver="node",
        receiver_port="InputPort-0",
        matrix=np.eye(2),
        projection_id=40,
        sender_component_id=9,
        sender_port_id=29,
        receiver_component_id=10,
        receiver_port_id=30,
    )
    state = BatchedStateSpec(
        "node.state",
        "node",
        2,
        (0.0, 0.0),
        component_id=10,
        state_id=50,
    )
    rng_stream = BatchedRngStreamSpec(
        "node.noise",
        "node",
        2,
        "MAX_STEPS",
        component_id=10,
        stream_id=60,
    )
    node = BatchedNodeSpec("node", "TransferMechanism", "Linear", 2, 2, component_id=10)
    graph = BatchedGraphIR(
        nodes=(node,),
        inputs=(input_spec,),
        projections=(projection,),
        outputs=(output,),
        states=(state,),
        scheduler=(),
        ops=(),
        execution_order=("node",),
        rng_streams=(rng_stream,),
    )
    kernel = KernelIR(
        model_kind="graph",
        fusion_kind=None,
        lane_layout=KernelLaneLayout("trial", ("trial",)),
        inputs=graph.inputs,
        params=(parameter,),
        states=graph.states,
        outputs=graph.outputs,
        rng_streams=(),
        ops=(),
        output_names=(output.name,),
        max_steps=1,
        graph=graph,
        op_specs=BatchedOpSpecSnapshot({}),
    )

    assert kernel.params[0].parameter_id == 20
    assert (kernel.inputs[0].component_id, kernel.inputs[0].port_id) == (10, 30)
    assert kernel.inputs[0].port == "InputPort-0"
    assert kernel.outputs[0].flat_slice == slice(3, 5)
    assert kernel.graph.projections[0].projection_id == 40
    assert kernel.states[0].state_id == 50
    assert kernel.graph.rng_streams[0].stream_id == 60


def test_output_slice_validation_rejects_partial_reversed_and_wrong_width_bounds():
    with pytest.raises(ValueError, match="requires both flattened bounds"):
        BatchedOutputSpec("out", "node", "RESULT", 1, flat_start=0)
    with pytest.raises(ValueError, match="flat_stop 1 before flat_start 2"):
        BatchedOutputSpec("out", "node", "RESULT", 1, flat_start=2, flat_stop=1)
    with pytest.raises(ValueError, match="does not match output width 2"):
        BatchedOutputSpec("out", "node", "RESULT", 2, flat_start=0, flat_stop=1)


def test_numeric_component_bindings_are_additive_to_name_bindings():
    node = object()
    function = object()
    parameter = object()
    port = object()
    projection = object()
    bindings = BatchedComponentBindings(
        nodes={"node": node},
        functions={"node": function},
        projections={"source.RESULT->node.InputPort-0": projection},
        nodes_by_id={10: node},
        functions_by_id={10: function},
        parameters_by_id={20: parameter},
        ports_by_id={30: port},
        projections_by_id={40: projection},
    )

    assert bindings.node("node") is bindings.node_by_id(10) is node
    assert bindings.function("node") is bindings.function_by_id(10) is function
    assert bindings.parameter_by_id(20) is parameter
    assert bindings.port_by_id(30) is port
    assert bindings.projection_by_id(40) is projection
    assert bindings.projection("source", "RESULT", "node", "InputPort-0") is projection
