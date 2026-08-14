"""Truncation visibility for bounded-loop batched ops (DDM integrator).

Bounded loops (the DDM integrator's ``max_steps`` cap) can stop before a lane
reaches threshold.  The compiler surfaces this per lane through a diagnostic
buffer; the runtime aggregates it into ``result.metadata["truncation"]`` (the
fraction of truncated lanes per node), warns by default, and raises under
``strict_truncation``.  These run on the ``triton_cpu`` (interpret) backend.

The DDM here is deterministic (noise=0): with rate=1, threshold=0.05,
time_step_size=0.01 and input 1.0 the particle gains 0.01/step, so it crosses in
5 steps.  ``max_steps=2`` therefore truncates every lane; ``max_steps=64`` none.
"""

import warnings

import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
    pytest.mark.triton_interpreter,
]


def _ddm_plan(max_steps):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.0, threshold=0.05,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=max_steps)
    inputs = {decision: np.array([[1.0], [1.0], [1.0], [1.0]])}
    return plan, inputs


def _run(plan, inputs, **kwargs):
    return plan.run(inputs=inputs, parameter_sets=[{}], num_estimates=1, seed=0, **kwargs)


def test_truncation_detected_and_warns_when_max_steps_too_low():
    plan, inputs = _ddm_plan(max_steps=2)
    with pytest.warns(UserWarning, match="truncated bounded loops"):
        result = _run(plan, inputs)
    assert result.metadata["truncation"]["DDM"] == pytest.approx(1.0)


def test_no_truncation_when_max_steps_sufficient():
    plan, inputs = _ddm_plan(max_steps=64)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any truncation warning would fail the test
        result = _run(plan, inputs)
    assert result.metadata["truncation"]["DDM"] == pytest.approx(0.0)


def test_strict_truncation_raises():
    plan, inputs = _ddm_plan(max_steps=2)
    with pytest.raises(BatchedTruncationError, match="max_steps=2"):
        _run(plan, inputs, strict_truncation=True)


def test_stateless_graph_has_no_truncation_diagnostics():
    source = pnl.TransferMechanism(
        input_shapes=1, function=pnl.Linear(slope=2.0), name="source"
    )
    comp = pnl.Composition(pathways=source)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu")
    result = _run(plan, {source: np.array([[1.0], [2.0]])})
    assert result.metadata["truncation"] == {}
