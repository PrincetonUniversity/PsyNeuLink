import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition


def _reasons(composition, outputs=None):
    result = lower_composition(composition, outputs=outputs)
    return "; ".join(
        f"{diagnostic.reason} {diagnostic.detail}" for diagnostic in result.rejected_nodes
    )


@pytest.mark.composition
@pytest.mark.parametrize(
    "function, parameter",
    [
        (pnl.Linear(scale=2.0), "scale"),
        (pnl.Linear(offset=0.25), "offset"),
        (pnl.Logistic(bias=0.25), "bias"),
        (pnl.Logistic(x_0=0.25), "x_0"),
        (pnl.Logistic(scale=2.0), "scale"),
        (pnl.Logistic(offset=0.25), "offset"),
    ],
)
def test_rejects_unlowered_transfer_function_parameters(function, parameter):
    node = pnl.TransferMechanism(input_shapes=1, function=function, name="node")
    reasons = _reasons(pnl.Composition(pathways=node))

    assert f"unsupported {type(function).__name__} parameter" in reasons
    assert parameter in reasons


@pytest.mark.composition
@pytest.mark.parametrize("kwargs, parameter", [({"noise": 0.1}, "noise"), ({"clip": (0, 1)}, "clip")])
def test_rejects_unlowered_transfer_mechanism_semantics(kwargs, parameter):
    node = pnl.TransferMechanism(input_shapes=1, name="node", **kwargs)

    assert parameter in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_rejects_generic_control_mechanism_and_control_projection():
    monitor = pnl.TransferMechanism(input_shapes=1, name="monitor")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    controller = pnl.ControlMechanism(
        monitor_for_control=monitor,
        control_signals=[("slope", target)],
        modulation=pnl.OVERRIDE,
        name="controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([monitor, target, controller])

    reasons = _reasons(composition)
    assert "unsupported generic ControlMechanism" in reasons
    assert "target.slope" in reasons


@pytest.mark.composition
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "termination measure"),
        (
            {
                "termination_measure": pnl.TimeScale.TRIAL,
                "termination_threshold": 2,
                "initial_value": [0.1, 0.0],
            },
            "initial_value",
        ),
        (
            {
                "termination_measure": pnl.TimeScale.TRIAL,
                "termination_threshold": 2,
                "reset_stateful_function_when": pnl.AtTrialStart(),
            },
            "reset policy",
        ),
    ],
)
def test_rejects_lca_configurations_not_represented_by_width2_op(kwargs, expected):
    lca = pnl.LCAMechanism(input_shapes=2, name="lca", **kwargs)

    assert expected in _reasons(pnl.Composition(pathways=lca))


@pytest.mark.composition
def test_rejects_multi_input_port_mechanism():
    node = pnl.TransferMechanism(input_ports=["left", "right"], name="mimo")

    assert "multi-port input routing" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_rejects_multi_output_port_mechanism_without_lowered_port_ops():
    node = pnl.TransferMechanism(
        input_shapes=2,
        output_ports=[pnl.RESULT, pnl.MEAN],
        name="multi-output",
    )

    assert "multi-port output routing" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_rejects_non_primary_output_port_routing():
    source = pnl.TransferMechanism(input_ports=["left", "right"], name="source")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    composition = pnl.Composition()
    composition.add_nodes([source, target])
    composition.add_projection(sender=source.output_ports[1], receiver=target)

    reasons = _reasons(composition)
    assert "multi-port input routing" in reasons
    assert "multi-port projection routing" in reasons


@pytest.mark.composition
def test_exact_scalar_lca_termination_override_remains_explicitly_absorbed():
    task = pnl.TransferMechanism(input_shapes=2, name="task")
    cue = pnl.TransferMechanism(input_shapes=1, name="cue")
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(),
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=8,
        name="lca",
    )
    controller = pnl.ControlMechanism(
        monitor_for_control=cue,
        control_signals=[(pnl.TERMINATION_THRESHOLD, lca)],
        modulation=pnl.OVERRIDE,
        name="controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([task, cue, lca, controller])
    composition.add_projection(sender=task, receiver=lca)

    result = lower_composition(composition)
    assert not result.rejected_nodes
    control_spec = result.graph.node("controller")
    assert control_spec.attrs["absorbed_control"] == {
        "source": "cue",
        "target": "lca",
        "parameter": "termination_threshold",
        "modulation": "OVERRIDE",
    }
    assert result.graph.node("lca").attrs["termination_input_node"] == "cue"
    assert all(projection.receiver != "controller" for projection in result.graph.projections)
