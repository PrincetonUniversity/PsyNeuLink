"""Exact external InputPort binding for partially projected mechanisms."""

import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.prep import prepare_inputs

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _mixed_internal_external_port_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        source = pnl.TransferMechanism(input_shapes=1, name=f"port-source-{index}")
        receiver = pnl.TransferMechanism(
            input_ports=[
                {pnl.NAME: "projected-input", pnl.INPUT_SHAPES: 1},
                {pnl.NAME: "external-input", pnl.INPUT_SHAPES: 1},
            ],
            name=f"mixed-port-receiver-{index}",
        )
        composition = pnl.Composition()
        composition.add_nodes([source, receiver])
        composition.add_projection(sender=source, receiver=receiver.input_ports[0])
        # PNL accepts direct input for the otherwise-unprojected second port
        # when the partially projected receiver is explicitly an INPUT node.
        composition.require_node_roles(receiver, pnl.NodeRole.INPUT)
        return SemanticModel(
            composition=composition,
            inputs={
                source: np.asarray([[2.0], [-1.0]]),
                receiver.input_ports[1]: np.asarray([[5.0], [7.0]]),
            },
            outputs=tuple(receiver.output_ports),
        )

    return SemanticCase(
        name="mixed_projected_and_external_input_ports",
        build=build,
        provenance=(
            "tests/composition/test_composition.py mixed INPUT-role routing; "
            "external values bind to their exact InputPort"
        ),
    )


MIXED_PORT_CASE = _mixed_internal_external_port_case()


def _composition_ir(lowering):
    graph = lowering.graph
    assert graph is not None
    return BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        graph=graph,
    )


def test_external_input_binding_uses_exact_port_identity():
    model = MIXED_PORT_CASE.build()
    receiver = model.outputs[0].owner
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None
    external_spec = next(
        spec for spec in lowering.graph.inputs if spec.component_id == lowering.graph.node(receiver.name).component_id
    )
    assert lowering.bindings.port_by_id(external_spec.port_id) is receiver.input_ports[1]

    correct = prepare_inputs(
        _composition_ir(lowering),
        model.inputs,
        component_bindings=lowering.bindings,
    )
    np.testing.assert_array_equal(correct[receiver.name][0], [5.0, 7.0])

    wrong_port_inputs = dict(model.inputs)
    wrong_port_inputs[receiver.input_ports[0]] = wrong_port_inputs.pop(receiver.input_ports[1])
    with pytest.raises(KeyError, match="external-input"):
        prepare_inputs(
            _composition_ir(lowering),
            wrong_port_inputs,
            component_bindings=lowering.bindings,
        )


def test_mixed_internal_external_port_matches_python(batched_backend):
    assert_matches_python(MIXED_PORT_CASE, backend=batched_backend)
