import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.prep import prepare_inputs
from psyneulink.core.batched.registry import analyze_composition


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _two_origin_composition():
    short = pnl.TransferMechanism(input_shapes=1, name="A")
    long = pnl.TransferMechanism(input_shapes=1, name="AB")
    receiver = pnl.TransferMechanism(input_shapes=1, name="result")
    composition = pnl.Composition()
    composition.add_nodes([short, long, receiver])
    composition.add_projection(sender=short, receiver=receiver)
    composition.add_projection(sender=long, receiver=receiver)
    report, ir, bindings = analyze_composition(composition)
    assert report.model_supported
    return short, long, ir, bindings


def test_prepare_inputs_resolves_live_nodes_by_identity_not_mapping_order():
    short, long, ir, bindings = _two_origin_composition()

    prepared = prepare_inputs(
        ir,
        {
            long: np.asarray([[10.0], [20.0]]),
            short: np.asarray([[1.0], [2.0]]),
        },
        component_bindings=bindings,
    )

    np.testing.assert_array_equal(prepared[short.name], [[1.0, 2.0]])
    np.testing.assert_array_equal(prepared[long.name], [[10.0, 20.0]])


def test_prepare_inputs_accepts_only_complete_string_names():
    short, long, ir, bindings = _two_origin_composition()

    prepared = prepare_inputs(
        ir,
        {
            long.name: np.asarray([[10.0], [20.0]]),
            short.name: np.asarray([[1.0], [2.0]]),
        },
        component_bindings=bindings,
    )

    np.testing.assert_array_equal(prepared[short.name], [[1.0, 2.0]])
    np.testing.assert_array_equal(prepared[long.name], [[10.0, 20.0]])


def test_prepare_inputs_rejects_two_keys_for_one_bound_input():
    short, long, ir, bindings = _two_origin_composition()

    with pytest.raises(ValueError, match="Multiple input entries"):
        prepare_inputs(
            ir,
            {
                short: np.asarray([[1.0]]),
                short.name: np.asarray([[2.0]]),
                long: np.asarray([[3.0]]),
            },
            component_bindings=bindings,
        )
