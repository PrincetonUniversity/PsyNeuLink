import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    batched_node_op,
    unregister_batched_instance_op,
)

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
]


_DRIFT_NODE_NAME = "Drift Rate Value"

_STIMULUS_TO_DRIFT = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ]
)
_CONTROL_TO_DRIFT = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    ]
)
_RESPONSE_TO_DRIFT = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

_STIMULUS_INPUTS = np.array(
    [
        [2.0, -1.0, 1.5, -0.5],
        [-0.75, 1.25, -1.5, 0.5],
        [0.2, 0.2, -0.4, -0.4],
        [4.0, -3.0, -2.0, 2.5],
    ]
)
_CONTROL_INPUTS = np.array(
    [
        [1.0, 0.1],
        [0.2, 1.0],
        [0.5, 0.5],
        [0.8, 0.35],
    ]
)
_RESPONSE_INPUTS = np.array([[1.0], [-1.0], [1.0], [-1.0]])


def _python_drift_rate(variable):
    x0, x1, x2, x3, x4, x5, x6 = np.asarray(variable, dtype=float).reshape(-1)
    a = 1.0 / (1.0 + np.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + np.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + np.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + np.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + np.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + np.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


def _make_case():
    build_count = 0

    def build():
        nonlocal build_count
        # Only the batched copy needs the registered name.  Giving the Python
        # oracle a generic name also prevents PsyNeuLink's live-name registry
        # from suffixing the decorator target on the second, fresh model.
        drift_name = "scalar sink" if build_count == 0 else _DRIFT_NODE_NAME
        build_count += 1

        four_wide_source = pnl.ProcessingMechanism(
            input_shapes=4,
            name="four-wide source",
        )
        two_wide_source = pnl.ProcessingMechanism(
            input_shapes=2,
            name="two-wide source",
        )
        scalar_source = pnl.ProcessingMechanism(
            input_shapes=1,
            name="scalar source",
        )
        drift = pnl.ProcessingMechanism(
            name=drift_name,
            function=_python_drift_rate,
            input_ports=[
                {
                    pnl.NAME: "seven-wide input",
                    pnl.INPUT_SHAPES: 7,
                    pnl.COMBINE: pnl.SUM,
                }
            ],
        )

        composition = pnl.Composition()
        composition.add_nodes(
            [four_wide_source, two_wide_source, scalar_source, drift]
        )
        composition.add_projection(
            sender=four_wide_source,
            receiver=drift.input_port,
            projection=pnl.MappingProjection(matrix=_STIMULUS_TO_DRIFT.copy()),
        )
        composition.add_projection(
            sender=two_wide_source,
            receiver=drift.input_port,
            projection=pnl.MappingProjection(matrix=_CONTROL_TO_DRIFT.copy()),
        )
        composition.add_projection(
            sender=scalar_source,
            receiver=drift.input_port,
            projection=pnl.MappingProjection(matrix=_RESPONSE_TO_DRIFT.copy()),
        )
        return SemanticModel(
            composition=composition,
            inputs={
                four_wide_source: _STIMULUS_INPUTS.copy(),
                two_wide_source: _CONTROL_INPUTS.copy(),
                scalar_source: _RESPONSE_INPUTS.copy(),
            },
            outputs=(drift.output_port,),
        )

    return SemanticCase(
        name="nested_logistic_three_source_fan_in",
        build=build,
        provenance=(
            "Scripts/Debug/pec_batch_compile/csi_model_surrogate.py:162-169,228-248"
        ),
        atol=1e-6,
        rtol=1e-5,
    )


def test_nested_logistic_three_source_fan_in_matches_python(batched_backend):
    try:
        @batched_node_op(_DRIFT_NODE_NAME)
        def drift_rate(x0, x1, x2, x3, x4, x5, x6):
            a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
            b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
            c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
            d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
            positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
            negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
            return (positive - negative) * x6

        comparison = assert_matches_python(_make_case(), backend=batched_backend)

        assert comparison.python_values.shape == (len(_STIMULUS_INPUTS), 1)
        assert comparison.batched_values.shape == (
            1,
            1,
            len(_STIMULUS_INPUTS),
            1,
            1,
        )
    finally:
        unregister_batched_instance_op(_DRIFT_NODE_NAME)
