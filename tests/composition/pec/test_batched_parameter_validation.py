"""Validation for scalar values crossing the fp32 parameter-row ABI."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR
from psyneulink.core.batched.prep import normalize_parameter_sets


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _linear_ir():
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(),
        name="parameter target",
    )
    composition = pnl.Composition(pathways=mechanism)
    lowering = lower_composition(composition)
    assert lowering.graph is not None
    graph = lowering.graph
    ir = BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        graph=graph,
    )
    return ir, graph.node(mechanism.name).params["slope"]


def _constrained_linear_ir():
    ir, slope = _linear_ir()
    constrained = tuple(
        replace(
            spec,
            minimum=0.0,
            minimum_inclusive=False,
        )
        if spec.name == slope
        else spec
        for spec in ir.params
    )
    return replace(ir, params=constrained), slope


@pytest.mark.parametrize(
    "value, message",
    [
        (1.0 + 1.0j, "real numeric scalars"),
        (np.nan, "finite"),
        (np.inf, "finite"),
        (1e40, "representable as float32"),
        ([1.0, 2.0], "must be scalar"),
    ],
    ids=("complex", "nan", "infinite", "float32-overflow", "vector-in-row"),
)
def test_invalid_runtime_parameter_value_is_rejected(value, message):
    ir, slope = _linear_ir()

    with pytest.raises(ValueError, match=message):
        normalize_parameter_sets([{slope: value}], ir)


def test_vectorized_mapping_remains_an_explicit_parameter_lane_shorthand():
    ir, slope = _linear_ir()

    rows = normalize_parameter_sets({slope: np.asarray([1.0, 2.0])}, ir)

    assert [row[slope] for row in rows] == [1.0, 2.0]


@pytest.mark.parametrize("value", (0.0, -0.01))
def test_runtime_parameter_respects_declared_exclusive_minimum(value):
    ir, parameter = _constrained_linear_ir()

    with pytest.raises(ValueError, match=r"must be > 0\.0"):
        normalize_parameter_sets([{parameter: value}], ir)
