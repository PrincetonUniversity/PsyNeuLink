"""Complete-KernelIR identity validation, including executable IO effects."""

from dataclasses import replace

import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import (
    BatchedCompositionIR,
    BatchedConsiderationSetSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedNodeSpec,
    BatchedParamSpec,
    BatchedPortSpec,
    BatchedSchedulerSpec,
)
from psyneulink.core.batched.kernel_ir import (
    KernelIR,
    KernelLaneLayout,
    KernelValue,
    lower_to_kernel_ir,
)
from psyneulink.core.batched.specs import BatchedOpSpecSnapshot


pytestmark = pytest.mark.batched


def _identity_kernel() -> KernelIR:
    producer = BatchedNodeSpec(
        name="producer",
        component_type="TransferMechanism",
        function_type="Linear",
        input_width=1,
        output_width=1,
        component_id=0,
        input_port_ids=(0,),
        output_port_ids=(1,),
    )
    follower = BatchedNodeSpec(
        name="follower",
        component_type="TransferMechanism",
        function_type="Linear",
        input_width=1,
        output_width=1,
        component_id=1,
        input_port_ids=(2,),
        output_port_ids=(3,),
    )
    ports = (
        BatchedPortSpec(0, "InputPort-0", "producer", 0, "InputPort", 1),
        BatchedPortSpec(1, "RESULT", "producer", 0, "OutputPort", 1),
        BatchedPortSpec(2, "InputPort-0", "follower", 1, "InputPort", 1),
        BatchedPortSpec(3, "RESULT", "follower", 1, "OutputPort", 1),
    )
    consideration_sets = (
        BatchedConsiderationSetSpec(0, ("producer",), (0,)),
        BatchedConsiderationSetSpec(1, ("follower",), (1,)),
    )
    finished_values = (
        BatchedFinishedValueSpec(
            name="producer.is_finished",
            node="producer",
            component_id=0,
            value_id=0,
            producer_consideration_set_id=0,
        ),
    )
    scheduler = (
        BatchedSchedulerSpec(
            node="producer",
            condition_type="Always",
            component_id=0,
            consideration_set_id=0,
        ),
        BatchedSchedulerSpec(
            node="follower",
            condition_type="WhenFinished",
            dependencies=("producer",),
            attrs={"predicate": "is_finished"},
            component_id=1,
            dependency_component_ids=(0,),
            finished_value_ids=(0,),
            consideration_set_id=1,
        ),
    )
    graph = BatchedGraphIR(
        nodes=(producer, follower),
        inputs=(),
        projections=(),
        outputs=(),
        states=(),
        scheduler=scheduler,
        ops=(),
        execution_order=("producer", "follower"),
        executable=False,
        ports=ports,
        consideration_sets=consideration_sets,
        finished_values=finished_values,
    )
    return KernelIR(
        model_kind="graph",
        fusion_kind=None,
        lane_layout=KernelLaneLayout(
            "trial",
            ("parameter_set", "subject", "trial", "estimate"),
        ),
        inputs=(),
        params=(
            BatchedParamSpec(
                "producer.slope",
                1.0,
                parameter_id=0,
                owner_component_id=0,
                owner_scope="function",
            ),
            BatchedParamSpec(
                "follower.slope",
                1.0,
                parameter_id=1,
                owner_component_id=1,
                owner_scope="function",
            ),
        ),
        states=(),
        outputs=(),
        rng_streams=(),
        ops=(),
        output_names=(),
        max_steps=1,
        graph=graph,
        op_specs=BatchedOpSpecSnapshot({}),
        executable=False,
        ports=ports,
        scheduler=scheduler,
        consideration_sets=consideration_sets,
        finished_values=finished_values,
    )


def _replace_graph_declarations(kernel, field, declarations):
    graph = replace(kernel.graph, **{field: declarations})
    return replace(kernel, graph=graph, **{field: declarations})


def _executable_io_kernel() -> KernelIR:
    mechanism = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(),
        name="global io identity fixture",
    )
    composition = pnl.Composition(pathways=[mechanism])
    lowering = lower_composition(
        composition,
        outputs=(mechanism.output_port,),
    )
    assert lowering.graph is not None
    graph = lowering.graph
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _multi_node_output_kernel() -> KernelIR:
    first = pnl.TransferMechanism(input_shapes=1, name="first output owner")
    second = pnl.TransferMechanism(input_shapes=1, name="second output owner")
    composition = pnl.Composition()
    composition.add_nodes([first, second])
    lowering = lower_composition(composition, outputs=(first.output_port,))
    assert lowering.graph is not None
    graph = lowering.graph
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _multi_port_output_kernel() -> KernelIR:
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.0,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="multi-port output owner",
    )
    composition = pnl.Composition(pathways=decision)
    lowering = lower_composition(
        composition,
        outputs=(decision.output_ports[pnl.DECISION_OUTCOME],),
    )
    assert lowering.graph is not None
    graph = lowering.graph
    return lower_to_kernel_ir(
        BatchedCompositionIR(
            model_kind=lowering.model_kind,
            node_names=tuple(node.name for node in graph.nodes),
            params=lowering.params,
            output_names=tuple(output.name for output in graph.outputs),
            graph=graph,
        )
    )


def _replace_io_declarations(kernel, field, declarations):
    graph = replace(kernel.graph, **{field: declarations})
    return replace(kernel, graph=graph, **{field: declarations})


def _replace_top_level_op(kernel, kind, replacement):
    matching = tuple(index for index, op in enumerate(kernel.ops) if op.kind == kind)
    assert len(matching) == 1
    ops = list(kernel.ops)
    ops[matching[0]] = replacement
    return replace(kernel, ops=tuple(ops))


def _top_level_op(kernel, kind):
    matching = tuple(op for op in kernel.ops if op.kind == kind)
    assert len(matching) == 1
    return matching[0]


def test_global_identity_inventory_accepts_one_consistent_kernel():
    kernel = _identity_kernel()

    assert tuple(parameter.parameter_id for parameter in kernel.params) == (0, 1)
    assert tuple(port.port_id for port in kernel.ports) == (0, 1, 2, 3)
    assert kernel.scheduler[1].finished_value_ids == (0,)


@pytest.mark.parametrize(
    "index, changes, message",
    [
        (0, {"parameter_id": False}, "exact non-bool integers"),
        (1, {"parameter_id": 0}, "unique, contiguous"),
        (1, {"parameter_id": 2}, "unique, contiguous"),
        (1, {"name": "producer.slope"}, "canonical names must be unique"),
        (1, {"owner_component_id": True}, "owner component IDs"),
        (1, {"owner_component_id": 9}, "resolve to exactly one"),
        (1, {"owner_scope": ""}, "owner_scope"),
        (1, {"owner_scope": "port"}, "owner_scope"),
    ],
)
def test_global_parameter_inventory_rejects_forged_identity(
    index,
    changes,
    message,
):
    kernel = _identity_kernel()
    params = list(kernel.params)
    params[index] = replace(params[index], **changes)

    with pytest.raises(ValueError, match=message):
        replace(kernel, params=tuple(params))


def test_global_port_inventory_rejects_duplicate_target_port_label():
    kernel = _identity_kernel()
    ports = list(kernel.ports)
    ports[3] = replace(
        ports[3],
        name=ports[2].name,
        kind=ports[2].kind,
    )

    with pytest.raises(ValueError, match="unique .*owner component, kind, name"):
        _replace_graph_declarations(kernel, "ports", tuple(ports))


@pytest.mark.parametrize(
    "index, changes, message",
    [
        (0, {"port_id": False}, "exact non-bool integers"),
        (1, {"port_id": 0}, "unique, contiguous"),
        (3, {"owner_component_id": 0}, "owner name and component id"),
        (3, {"owner": "producer"}, "owner name and component id"),
        (3, {"kind": "MysteryPort"}, "supported port kind"),
    ],
)
def test_global_port_inventory_rejects_forged_identity(
    index,
    changes,
    message,
):
    kernel = _identity_kernel()
    ports = list(kernel.ports)
    if type(changes.get("port_id")) is bool:
        # Exercise the complete-IR boundary even though ordinary dataclass
        # construction also rejects this value.
        object.__setattr__(ports[index], "port_id", False)
    else:
        ports[index] = replace(ports[index], **changes)

    with pytest.raises(ValueError, match=message):
        _replace_graph_declarations(kernel, "ports", tuple(ports))


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"value_id": False}, "exact non-bool integers"),
        ({"value_id": 1}, "unique, contiguous"),
        ({"name": "forged"}, "exact node/component/set identity"),
        ({"node": "follower"}, "exact node/component/set identity"),
        ({"component_id": 1}, "exact GraphIR node and producer"),
        ({"producer_consideration_set_id": 1}, "exact GraphIR node and producer"),
        ({"width": 2}, "scalar bool combinational"),
        ({"dtype": "float32"}, "scalar bool combinational"),
        ({"storage": "lane_persistent"}, "scalar bool combinational"),
    ],
)
def test_global_finished_inventory_rejects_forged_identity(changes, message):
    kernel = _identity_kernel()
    finished = replace(kernel.finished_values[0], **changes)

    with pytest.raises(ValueError, match=message):
        _replace_graph_declarations(kernel, "finished_values", (finished,))


def test_global_finished_inventory_rejects_duplicate_value_id():
    kernel = _identity_kernel()
    duplicate = replace(
        kernel.finished_values[0],
        name="follower.is_finished",
        node="follower",
        component_id=1,
        producer_consideration_set_id=1,
    )

    with pytest.raises(ValueError, match="unique, contiguous"):
        _replace_graph_declarations(
            kernel,
            "finished_values",
            (*kernel.finished_values, duplicate),
        )


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"finished_value_ids": (False,)}, "exact non-bool"),
        ({"finished_value_ids": (1,)}, "does not match"),
        ({"finished_value_ids": ()}, "parallel unique tuples"),
        ({"dependencies": ("follower",)}, "does not match component id"),
        ({"dependency_component_ids": (1,)}, "does not match component id"),
    ],
)
def test_when_finished_rejects_forged_dependency_reference(changes, message):
    kernel = _identity_kernel()
    scheduler = (
        kernel.scheduler[0],
        replace(kernel.scheduler[1], **changes),
    )

    with pytest.raises(ValueError, match=message):
        _replace_graph_declarations(kernel, "scheduler", scheduler)


def test_finished_declaration_cannot_be_orphaned_from_scheduler():
    kernel = _identity_kernel()
    scheduler = (
        kernel.scheduler[0],
        replace(
            kernel.scheduler[1],
            condition_type="Always",
            dependencies=(),
            dependency_component_ids=(),
            finished_value_ids=(),
        ),
    )

    with pytest.raises(ValueError, match="exact component-wise bijection"):
        _replace_graph_declarations(kernel, "scheduler", scheduler)


def test_executable_io_inventory_accepts_exact_declarations_and_effects():
    kernel = _executable_io_kernel()

    assert kernel.executable
    assert tuple(op.kind for op in kernel.ops).count("LoadInput") == 1
    assert tuple(op.kind for op in kernel.ops).count("StoreOutput") == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "width",
        "component_id",
        "node",
        "port_id",
        "port",
        "name",
    ],
)
def test_executable_input_declaration_must_match_typed_port(mutation):
    kernel = _executable_io_kernel()
    input_spec = kernel.inputs[0]
    changes = {
        "width": {"width": input_spec.width + 1},
        "component_id": {"component_id": input_spec.component_id + 100},
        "node": {"node": f"forged {input_spec.node}"},
        "port_id": {"port_id": kernel.outputs[0].port_id},
        "port": {"port": f"forged {input_spec.port}"},
        "name": {"name": f"forged {input_spec.name}"},
    }[mutation]

    with pytest.raises(ValueError, match="exact typed GraphIR node and external"):
        _replace_io_declarations(
            kernel,
            "inputs",
            (replace(input_spec, **changes),),
        )


def test_executable_input_and_load_cannot_be_erased_together():
    kernel = _executable_io_kernel()
    graph = replace(kernel.graph, inputs=())
    ops = tuple(op for op in kernel.ops if op.kind != "LoadInput")

    with pytest.raises(ValueError, match="cover every external typed InputPort"):
        replace(kernel, graph=graph, inputs=(), ops=ops)


@pytest.mark.parametrize(
    "mutation",
    [
        "width",
        "component_id",
        "node",
        "port_id",
        "port",
        "name",
    ],
)
def test_executable_output_declaration_must_match_typed_port(mutation):
    kernel = _executable_io_kernel()
    output = kernel.outputs[0]
    changes = {
        "width": {"width": 1, "flat_stop": 1},
        "component_id": {"component_id": output.component_id + 100},
        "node": {"node": f"forged {output.node}"},
        "port_id": {"port_id": kernel.inputs[0].port_id},
        "port": {"port": f"forged {output.port}"},
        "name": {"name": f"forged {output.name}"},
    }[mutation]

    with pytest.raises(ValueError, match="exact typed GraphIR node and OutputPort"):
        _replace_io_declarations(
            kernel,
            "outputs",
            (replace(output, **changes),),
        )


def test_executable_output_declaration_requires_exact_flat_layout():
    kernel = _executable_io_kernel()
    output = kernel.outputs[0]

    with pytest.raises(ValueError, match="flattened slices must be contiguous"):
        _replace_io_declarations(
            kernel,
            "outputs",
            (
                replace(
                    output,
                    flat_start=output.flat_start + output.width,
                    flat_stop=output.flat_stop + output.width,
                ),
            ),
        )


def test_executable_output_names_require_exact_declaration_order():
    kernel = _executable_io_kernel()

    with pytest.raises(ValueError, match="output_names must exactly match"):
        replace(kernel, output_names=(f"forged {kernel.output_names[0]}",))


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "value_name",
        "value_width",
        "value_dtype",
        "attr_component_id",
        "attr_port_id",
        "attr_node",
        "attr_port",
        "attr_name",
        "attr_width",
        "attr_slice",
        "extra_attr",
    ],
)
def test_executable_load_input_must_match_declaration_exactly(mutation):
    kernel = _executable_io_kernel()
    load = _top_level_op(kernel, "LoadInput")
    value = load.outputs[0]
    attrs = dict(load.attrs)
    changes = {}
    if mutation == "target":
        changes["target"] = f"forged {load.target}"
    elif mutation == "value_name":
        changes["outputs"] = (replace(value, name=f"forged {value.name}"),)
    elif mutation == "value_width":
        changes["outputs"] = (replace(value, width=value.width + 1),)
    elif mutation == "value_dtype":
        changes["outputs"] = (replace(value, dtype="float64"),)
    elif mutation == "attr_component_id":
        attrs["component_id"] += 100
    elif mutation == "attr_port_id":
        attrs["port_id"] = kernel.outputs[0].port_id
    elif mutation == "attr_node":
        attrs["node"] = f"forged {attrs['node']}"
    elif mutation == "attr_port":
        attrs["port"] = f"forged {attrs['port']}"
    elif mutation == "attr_name":
        attrs["input_name"] = f"forged {attrs['input_name']}"
    elif mutation == "attr_width":
        attrs["width"] += 1
    elif mutation == "attr_slice":
        attrs["flat_start"] += 1
    else:
        attrs["forged"] = True
    if mutation.startswith("attr_") or mutation == "extra_attr":
        changes["attrs"] = attrs

    with pytest.raises(
        ValueError,
        match=(
            "LoadInput (does not exactly match|references no)|"
            "defined by a dominating operation"
        ),
    ):
        _replace_top_level_op(kernel, "LoadInput", replace(load, **changes))


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_executable_load_input_count_must_match_schedule(mutation):
    kernel = _executable_io_kernel()
    load = _top_level_op(kernel, "LoadInput")
    if mutation == "missing":
        ops = tuple(op for op in kernel.ops if op is not load)
    else:
        ops = (load, *kernel.ops)

    with pytest.raises(
        ValueError,
        match="exact scheduled LoadInput count|defined by a dominating operation",
    ):
        replace(kernel, ops=ops)


@pytest.mark.parametrize(
    "mutation",
    [
        "target",
        "value_name",
        "value_width",
        "value_dtype",
        "attr_component_id",
        "attr_port_id",
        "attr_node",
        "attr_port",
        "attr_width",
        "attr_slice",
        "extra_attr",
    ],
)
def test_executable_store_output_must_match_declaration_exactly(mutation):
    kernel = _executable_io_kernel()
    store = _top_level_op(kernel, "StoreOutput")
    value = store.inputs[0]
    attrs = dict(store.attrs)
    changes = {}
    if mutation == "target":
        changes["target"] = f"forged {store.target}"
    elif mutation == "value_name":
        changes["inputs"] = (replace(value, name=f"forged {value.name}"),)
    elif mutation == "value_width":
        changes["inputs"] = (replace(value, width=value.width + 1),)
    elif mutation == "value_dtype":
        changes["inputs"] = (replace(value, dtype="float64"),)
    elif mutation == "attr_component_id":
        attrs["component_id"] += 100
    elif mutation == "attr_port_id":
        attrs["port_id"] = kernel.inputs[0].port_id
    elif mutation == "attr_node":
        attrs["node"] = f"forged {attrs['node']}"
    elif mutation == "attr_port":
        attrs["port"] = f"forged {attrs['port']}"
    elif mutation == "attr_width":
        attrs["width"] += 1
    elif mutation == "attr_slice":
        attrs["flat_stop"] += 1
    else:
        attrs["forged"] = True
    if mutation.startswith("attr_") or mutation == "extra_attr":
        changes["attrs"] = attrs

    with pytest.raises(ValueError, match="StoreOutput does not exactly match"):
        _replace_top_level_op(kernel, "StoreOutput", replace(store, **changes))


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_executable_store_output_count_must_match_declarations(mutation):
    kernel = _executable_io_kernel()
    store = _top_level_op(kernel, "StoreOutput")
    if mutation == "missing":
        ops = tuple(op for op in kernel.ops if op is not store)
    else:
        ops = (*kernel.ops, store)

    with pytest.raises(ValueError, match="exactly one StoreOutput"):
        replace(kernel, ops=ops)


@pytest.mark.parametrize("reset_declaration_ids", [False, True])
def test_typed_io_inventory_cannot_be_erased_to_enter_legacy_validation(
    reset_declaration_ids,
):
    kernel = _executable_io_kernel()
    nodes = tuple(
        replace(
            node,
            input_port_ids=(),
            output_port_ids=(),
            parameter_port_ids=(),
        )
        for node in kernel.graph.nodes
    )
    inputs = kernel.inputs
    outputs = kernel.outputs
    if reset_declaration_ids:
        inputs = (replace(inputs[0], port_id=-1),)
        outputs = (replace(outputs[0], port_id=-1),)
    graph = replace(
        kernel.graph,
        nodes=nodes,
        ports=(),
        inputs=inputs,
        outputs=outputs,
    )

    with pytest.raises(ValueError, match="exact typed GraphIR"):
        replace(
            kernel,
            graph=graph,
            ports=(),
            inputs=inputs,
            outputs=outputs,
        )


def test_executable_ops_require_dominating_input_definitions():
    kernel = _executable_io_kernel()
    load = _top_level_op(kernel, "LoadInput")
    call = _top_level_op(kernel, "CallFunction")
    store = _top_level_op(kernel, "StoreOutput")

    with pytest.raises(ValueError, match="defined by a dominating operation"):
        replace(kernel, ops=(call, load, store))


def test_store_output_rejects_value_from_another_component():
    kernel = _multi_node_output_kernel()
    store = _top_level_op(kernel, "StoreOutput")
    other_value = next(
        op.outputs[0]
        for op in kernel.ops
        if op.kind == "CallFunction" and op.target != kernel.outputs[0].node
    )

    with pytest.raises(ValueError, match="StoreOutput does not exactly match"):
        _replace_top_level_op(
            kernel,
            "StoreOutput",
            replace(store, inputs=(other_value,)),
        )


def test_store_output_rejects_another_port_from_the_same_component():
    kernel = _multi_port_output_kernel()
    store = _top_level_op(kernel, "StoreOutput")
    mechanism_call = _top_level_op(kernel, "CallMechanism")
    response_time = mechanism_call.outputs[1]

    with pytest.raises(ValueError, match="StoreOutput does not exactly match"):
        _replace_top_level_op(
            kernel,
            "StoreOutput",
            replace(store, inputs=(response_time,)),
        )


def test_store_output_rejects_bool_value_width():
    kernel = _executable_io_kernel()
    store = _top_level_op(kernel, "StoreOutput")
    forged = replace(store.inputs[0], width=True)

    with pytest.raises(ValueError, match="positive non-bool widths"):
        _replace_top_level_op(
            kernel,
            "StoreOutput",
            replace(store, inputs=(forged,)),
        )
