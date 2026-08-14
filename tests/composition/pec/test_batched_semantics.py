import numpy as np
import pytest

import psyneulink as pnl

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


def _transfer_case(name, provenance, function_factory, inputs):
    def build():
        mechanism = pnl.TransferMechanism(
            input_shapes=2,
            function=function_factory(),
            name=name,
        )
        composition = pnl.Composition(pathways=mechanism)
        return SemanticModel(
            composition=composition,
            inputs={mechanism: np.asarray(inputs, dtype=float)},
            outputs=(mechanism.output_port,),
        )

    return SemanticCase(
        name=name,
        build=build,
        provenance=provenance,
    )


CASES = (
    _transfer_case(
        "linear_nondefault",
        "tests/functions/test_transfer.py::test_execute[LINEAR]",
        lambda: pnl.Linear(slope=2.0, intercept=1.0),
        [[1.0, 2.0], [3.0, 4.0]],
    ),
    _transfer_case(
        "logistic_nondefault_gain",
        "tests/functions/test_transfer.py::test_execute[LOGISTIC]",
        lambda: pnl.Logistic(gain=2.0),
        [[0.5, -1.0], [0.0, 2.0]],
    ),
)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_stateless_transfer_matches_python(case, batched_backend):
    assert_matches_python(case, backend=batched_backend)
