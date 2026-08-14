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
def test_accepts_fully_lowered_transfer_function_parameters(function, parameter):
    node = pnl.TransferMechanism(input_shapes=1, function=function, name="node")
    result = lower_composition(pnl.Composition(pathways=node))

    assert not result.rejected_nodes
    assert parameter in result.graph.node(node.name).params


@pytest.mark.composition
def test_accepts_broadcast_scalar_transfer_function_parameter():
    node = pnl.TransferMechanism(
        input_shapes=3,
        function=pnl.Linear(slope=[2.0, 2.0, 2.0]),
        name="node",
    )
    result = lower_composition(pnl.Composition(pathways=node))

    assert not result.rejected_nodes


@pytest.mark.composition
def test_rejects_elementwise_transfer_function_parameter_until_vector_abi_lands():
    node = pnl.TransferMechanism(
        input_shapes=3,
        function=pnl.Linear(slope=[1.0, 2.0, 3.0]),
        name="node",
    )
    result = lower_composition(pnl.Composition(pathways=node))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported non-scalar Linear parameter for batched v2"
    )
    assert diagnostic.component == node.name
    assert diagnostic.detail == "slope shape=(3,)"


@pytest.mark.composition
def test_accepts_constant_transfer_noise_and_clip():
    node = pnl.TransferMechanism(
        input_shapes=3,
        name="node",
        noise=[0.25, 0.25, 0.25],
        clip=(-1.0, 2.0),
    )
    result = lower_composition(pnl.Composition(pathways=node))

    assert not result.rejected_nodes
    assert result.graph.node(node.name).attrs["noise"] == 0.25
    assert result.graph.node(node.name).attrs["clip"] == (-1.0, 2.0)


@pytest.mark.composition
@pytest.mark.parametrize(
    "noise",
    [
        pytest.param([0.1, 0.2, 0.3], id="heterogeneous"),
        pytest.param(lambda: 0.1, id="callable"),
        pytest.param(pnl.NormalDist(), id="distribution"),
        pytest.param(float("inf"), id="nonfinite"),
    ],
)
def test_rejects_untyped_or_stochastic_transfer_noise(noise):
    node = pnl.TransferMechanism(input_shapes=3, name="node", noise=noise)

    assert "TransferMechanism noise" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_rejects_nonfinite_transfer_clip():
    node = pnl.TransferMechanism(
        input_shapes=1,
        name="node",
        clip=(float("-inf"), float("inf")),
    )

    assert "TransferMechanism clip" in _reasons(pnl.Composition(pathways=node))


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
@pytest.mark.parametrize("parameter", ["bias", "x_0", "scale", "offset"])
def test_full_logistic_transfer_support_does_not_relax_lca_contract(parameter):
    defaults = {"bias": 0.0, "x_0": 0.0, "scale": 1.0, "offset": 0.0}
    defaults[parameter] += 0.25
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(**defaults),
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=2,
        name="lca",
    )

    reasons = _reasons(pnl.Composition(pathways=lca))
    assert "unsupported LCA Logistic parameter" in reasons
    assert parameter in reasons


@pytest.mark.composition
def test_rejects_multi_input_port_mechanism():
    node = pnl.TransferMechanism(input_ports=["left", "right"], name="mimo")

    assert "external multi-port input binding" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_stale_external_afferents_do_not_hide_multi_port_runtime_inputs():
    left = pnl.TransferMechanism(input_shapes=1, name="left")
    right = pnl.TransferMechanism(input_shapes=1, name="right")
    receiver = pnl.TransferMechanism(input_ports=["left-in", "right-in"], name="receiver")
    first_composition = pnl.Composition()
    first_composition.add_nodes([left, right, receiver])
    first_composition.add_projection(sender=left, receiver=receiver.input_ports[0])
    first_composition.add_projection(sender=right, receiver=receiver.input_ports[1])

    # Reusing the receiver preserves its live path_afferents, but neither
    # projection belongs to this singleton Composition.  Both ports therefore
    # require external values in the graph being lowered.
    result = lower_composition(pnl.Composition(pathways=receiver))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported external multi-port input binding for batched v2"
    )
    assert diagnostic.component == receiver.name
    assert diagnostic.detail == "lowered external ports=['left-in', 'right-in']"
    assert result.graph is None


@pytest.mark.composition
def test_rejects_duplicate_live_node_names_until_graph_lookup_is_id_native():
    source = pnl.TransferMechanism(input_shapes=1, name="source")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    composition = pnl.Composition(pathways=[[source, target]])
    target.name = source.name

    result = lower_composition(composition)

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "duplicate live node names are unsupported for batched v2"
    )
    assert diagnostic.component == composition.name
    assert diagnostic.detail == f"name={source.name!r}"
    assert result.graph is None


@pytest.mark.composition
def test_rejects_duplicate_output_port_names_until_lookup_is_id_native():
    left = pnl.TransferMechanism(input_shapes=1, name="left")
    right = pnl.TransferMechanism(input_shapes=1, name="right")
    receiver = pnl.TransferMechanism(input_ports=["left-in", "right-in"], name="receiver")
    composition = pnl.Composition()
    composition.add_nodes([left, right, receiver])
    composition.add_projection(sender=left, receiver=receiver.input_ports[0])
    composition.add_projection(sender=right, receiver=receiver.input_ports[1])
    receiver.output_ports[1].name = receiver.output_ports[0].name

    result = lower_composition(composition)

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "duplicate OutputPort names are unsupported for batched v2"
    )
    assert diagnostic.component == receiver.name
    assert diagnostic.detail == f"duplicates={[receiver.output_ports[0].name]!r}"
    assert result.graph is None


@pytest.mark.composition
def test_rejects_default_variable_input_instead_of_inventing_runtime_input():
    node = pnl.TransferMechanism(
        input_ports={
            pnl.VARIABLE: [2.0],
            pnl.PARAMS: {pnl.DEFAULT_INPUT: pnl.DEFAULT_VARIABLE},
        },
        name="constant-input",
    )
    result = lower_composition(pnl.Composition(pathways=node))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported InputPort default/internal binding for batched v2"
    )
    assert diagnostic.component == node.name
    assert "default_input='default_variable'" in diagnostic.detail
    assert "internal_only=True" in diagnostic.detail
    assert result.graph is None


@pytest.mark.composition
def test_rejects_multi_output_port_mechanism_without_lowered_port_ops():
    node = pnl.TransferMechanism(
        input_shapes=2,
        output_ports=[pnl.RESULT, pnl.MEAN],
        name="multi-output",
    )

    reasons = _reasons(pnl.Composition(pathways=node))
    assert "unsupported OutputPort function" in reasons
    assert pnl.MEAN in reasons


@pytest.mark.composition
def test_rejects_non_owner_value_output_selector_even_when_index_property_matches():
    node = pnl.TransferMechanism(
        input_shapes=1,
        output_ports=[
            pnl.OutputPort(name="execution-count", variable=("num_executions", 0))
        ],
        name="counter-output",
    )
    result = lower_composition(pnl.Composition(pathways=node))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason == "unsupported OutputPort function for batched v2"
    )
    assert diagnostic.component == node.name
    assert "identity OWNER_VALUE slice" in diagnostic.detail
    assert result.graph is None


@pytest.mark.composition
def test_rejects_custom_function_on_declared_ddm_output_name():
    ddm = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(noise=0.0),
        output_ports=[
            {
                pnl.NAME: pnl.DECISION_OUTCOME,
                pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                pnl.FUNCTION: pnl.Linear(slope=0.0, intercept=7.0),
            },
            pnl.RESPONSE_TIME,
        ],
        name="custom-output-ddm",
    )
    result = lower_composition(pnl.Composition(pathways=ddm))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported mechanism output-port function for batched v2"
    )
    assert diagnostic.component == ddm.name
    assert diagnostic.detail.startswith("DECISION_OUTCOME: Linear")
    assert result.graph is None


@pytest.mark.composition
def test_rejects_custom_selector_on_declared_ddm_output_name():
    ddm = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(noise=0.0),
        output_ports=[
            {
                pnl.NAME: pnl.DECISION_OUTCOME,
                pnl.VARIABLE: (pnl.OWNER_VALUE, 1),
            },
            pnl.RESPONSE_TIME,
        ],
        name="custom-selector-ddm",
    )
    result = lower_composition(pnl.Composition(pathways=ddm))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported mechanism output-port selector for batched v2"
    )
    assert diagnostic.component == ddm.name
    assert "requires ('OWNER_VALUE', 0)" in diagnostic.detail
    assert result.graph is None


@pytest.mark.composition
def test_rejects_custom_width_on_declared_ddm_output_name():
    def duplicate_value(value):
        return np.asarray([value[0], value[0]])

    ddm = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(noise=0.0),
        output_ports=[
            {
                pnl.NAME: pnl.DECISION_OUTCOME,
                pnl.VARIABLE: (pnl.OWNER_VALUE, 0),
                pnl.FUNCTION: pnl.UserDefinedFunction(
                    custom_function=duplicate_value,
                ),
            },
            pnl.RESPONSE_TIME,
        ],
        name="custom-width-ddm",
    )
    result = lower_composition(pnl.Composition(pathways=ddm))

    diagnostic = next(
        diagnostic
        for diagnostic in result.rejected_nodes
        if diagnostic.reason
        == "unsupported mechanism output-port width for batched v2"
    )
    assert diagnostic.component == ddm.name
    assert diagnostic.detail == "DECISION_OUTCOME: width=2, requires 1"
    assert result.graph is None


@pytest.mark.composition
def test_near_identity_output_parameter_is_not_treated_as_exact():
    node = pnl.TransferMechanism(
        input_shapes=1,
        output_ports=[
            pnl.OutputPort(
                name="near-identity",
                variable=(pnl.OWNER_VALUE, 0),
                function=pnl.Linear(slope=1.000009),
            )
        ],
        name="node",
    )

    assert "OutputPort function" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_near_default_input_combine_parameter_is_not_treated_as_exact():
    node = pnl.TransferMechanism(
        input_ports={
            pnl.VARIABLE: [0.0],
            pnl.FUNCTION: pnl.LinearCombination(scale=1.000009),
        },
        name="node",
    )

    assert "InputPort function" in _reasons(pnl.Composition(pathways=node))


@pytest.mark.composition
def test_rejects_non_primary_output_port_routing():
    source = pnl.TransferMechanism(input_ports=["left", "right"], name="source")
    target = pnl.TransferMechanism(input_shapes=1, name="target")
    composition = pnl.Composition()
    composition.add_nodes([source, target])
    composition.add_projection(sender=source.output_ports[1], receiver=target)

    reasons = _reasons(composition)
    assert "external multi-port input binding" in reasons


@pytest.mark.composition
@pytest.mark.parametrize(
    "function, expected",
    [
        (pnl.LinearCombination(operation=pnl.CROSS_ENTROPY), "cross-entropy"),
        (pnl.LinearCombination(operation=pnl.SUM, scale=2.0), "scale"),
        (pnl.LinearCombination(operation=pnl.PRODUCT, offset=1.0), "offset"),
        (pnl.Linear(), "Linear"),
    ],
)
def test_rejects_unmodeled_input_port_functions(function, expected):
    left = pnl.TransferMechanism(input_shapes=1, name="left")
    right = pnl.TransferMechanism(input_shapes=1, name="right")
    receiver = pnl.TransferMechanism(
        input_ports={pnl.INPUT_SHAPES: 1, pnl.FUNCTION: function},
        name="receiver",
    )
    composition = pnl.Composition()
    composition.add_nodes([left, right, receiver])
    composition.add_projection(sender=left, receiver=receiver)
    composition.add_projection(sender=right, receiver=receiver)

    reasons = _reasons(composition)
    assert "unsupported InputPort function" in reasons
    assert expected in reasons


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
