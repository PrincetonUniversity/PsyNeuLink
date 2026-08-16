"""Complete-KernelIR identity validation independent of executable shape."""

from dataclasses import replace

import pytest

from psyneulink.core.batched.ir import (
    BatchedConsiderationSetSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedNodeSpec,
    BatchedParamSpec,
    BatchedPortSpec,
    BatchedSchedulerSpec,
)
from psyneulink.core.batched.kernel_ir import KernelIR, KernelLaneLayout
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
        lane_layout=KernelLaneLayout("trial", ("trial",)),
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
