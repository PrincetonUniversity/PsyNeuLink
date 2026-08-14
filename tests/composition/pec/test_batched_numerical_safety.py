"""Runtime rejection of non-finite batched simulation outcomes."""

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    BatchedNumericalError,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
]


def _overflowing_linear():
    maximum = float(np.finfo(np.float32).max)
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=maximum),
        name="numerically extreme linear",
    )
    composition = pnl.Composition(pathways=mechanism)
    inputs = {mechanism: np.asarray([[maximum]])}
    return composition, mechanism, inputs


def test_nonfinite_outcome_raises_before_result_is_returned(batched_backend):
    composition, mechanism, inputs = _overflowing_linear()
    python_result = composition.run(
        inputs=inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    assert np.all(np.isfinite(python_result))

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend=batched_backend,
        outputs=(mechanism.output_port,),
    )
    assert report.can_execute, report.to_dict()

    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=batched_backend,
        outputs=(mechanism.output_port,),
    )
    with pytest.raises(
        BatchedNumericalError,
        match=r"NaN or infinite outcome value\(s\)",
    ):
        plan.run(
            inputs=inputs,
            parameter_sets=[{}],
            num_estimates=1,
            seed=0,
        )
