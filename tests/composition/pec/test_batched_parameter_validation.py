"""Validation for scalar values crossing the fp32 parameter-row ABI."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedTrialParameter
from psyneulink.core.batched.prep import (
    normalize_parameter_sets,
    prepare_parameter_values,
)


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


def test_explicit_trial_parameter_preserves_one_parameter_lane():
    ir, slope = _linear_ir()
    rows = normalize_parameter_sets(
        {slope: BatchedTrialParameter([1.0, 2.0, 3.0])},
        ir,
    )

    assert len(rows) == 1
    assert isinstance(rows[0][slope], BatchedTrialParameter)

    buffers, strides = prepare_parameter_values(
        ir,
        rows,
        num_subjects=1,
        num_trials=3,
    )
    parameter_index = next(
        index for index, spec in enumerate(ir.params) if spec.name == slope
    )
    assert buffers[parameter_index].shape == (1, 1, 3)
    assert buffers[parameter_index][0, 0].tolist() == [1.0, 2.0, 3.0]
    assert strides[2 * parameter_index : 2 * parameter_index + 2] == (3, 1)

    scalar_index = next(
        index for index, spec in enumerate(ir.params) if spec.name != slope
    )
    assert buffers[scalar_index].shape == (1,)
    assert strides[2 * scalar_index : 2 * scalar_index + 2] == (1, 0)


def test_trial_parameter_shape_must_match_prepared_trials():
    ir, slope = _linear_ir()
    rows = normalize_parameter_sets(
        [{slope: BatchedTrialParameter([1.0, 2.0])}],
        ir,
    )

    with pytest.raises(ValueError, match="expected a flat trial vector"):
        prepare_parameter_values(
            ir,
            rows,
            num_subjects=1,
            num_trials=3,
        )


def test_flat_trial_parameter_uses_subject_slices_and_padding():
    ir, slope = _linear_ir()
    rows = normalize_parameter_sets(
        [{slope: BatchedTrialParameter([1.0, 2.0, 3.0, 4.0, 5.0])}],
        ir,
    )
    buffers, strides = prepare_parameter_values(
        ir,
        rows,
        num_subjects=2,
        num_trials=3,
        subject_slices=(slice(0, 2), slice(2, 5)),
    )
    parameter_index = next(
        index for index, spec in enumerate(ir.params) if spec.name == slope
    )

    assert buffers[parameter_index][0].tolist() == [
        [1.0, 2.0, 0.0],
        [3.0, 4.0, 5.0],
    ]
    assert strides[2 * parameter_index : 2 * parameter_index + 2] == (6, 1)


def test_ambiguous_parameter_alias_is_rejected():
    ir, _ = _linear_ir()
    shared_alias = "shared.parameter"
    params = tuple(
        replace(spec, aliases=(*spec.aliases, shared_alias))
        if index < 2
        else spec
        for index, spec in enumerate(ir.params)
    )
    forged = replace(ir, params=params)

    assert normalize_parameter_sets([{}], forged) == [dict(forged.param_defaults)]

    with pytest.raises(ValueError, match="Ambiguous batched parameter"):
        normalize_parameter_sets(
            [{shared_alias: 2.0}],
            forged,
        )

    exact_name = params[0].name
    exact_wins = replace(
        ir,
        params=(
            params[0],
            replace(params[1], aliases=(*params[1].aliases, exact_name)),
            *params[2:],
        ),
    )
    normalized = normalize_parameter_sets([{exact_name: 2.0}], exact_wins)[0]
    assert normalized[exact_name] == 2.0
    assert normalized[params[1].name] == params[1].default


@pytest.mark.parametrize("value", (0.0, -0.01))
def test_runtime_parameter_respects_declared_exclusive_minimum(value):
    ir, parameter = _constrained_linear_ir()

    with pytest.raises(ValueError, match=r"must be > 0\.0"):
        normalize_parameter_sets([{parameter: value}], ir)


def test_trial_parameter_respects_declared_exclusive_minimum():
    ir, parameter = _constrained_linear_ir()

    with pytest.raises(ValueError, match=r"must be > 0\.0"):
        normalize_parameter_sets(
            [{parameter: BatchedTrialParameter([1.0, 0.0])}],
            ir,
        )
