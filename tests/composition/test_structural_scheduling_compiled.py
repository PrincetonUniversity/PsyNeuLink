"""Graph ordering conditions must not be compiled as runtime predicates."""

import numpy as np
import pytest

import psyneulink as pnl


@pytest.mark.composition
@pytest.mark.parametrize("execution_mode", [pnl.ExecutionMode.Python, pnl.ExecutionMode.LLVMRun])
def test_structural_order_with_basic_and_implicit_conditions(execution_mode):
    source = pnl.ProcessingMechanism(name="delayed source")
    target = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator, name="counted target")
    composition = pnl.Composition(nodes=[source, target])
    composition.scheduler.add_condition(source, pnl.AtPass(1))
    composition.scheduler.add_condition(source, pnl.AddEdgeTo(target))
    composition.run(inputs={source: [[1.]], target: [[2.]]}, execution_mode=execution_mode)
    # The target waits for the delayed source and executes once. Treating its
    # implicit predicate as Always would integrate twice and return 4 instead.
    np.testing.assert_allclose(target.parameters.value.get(composition), [[2.]])
    np.testing.assert_allclose(composition.results[-1], [[1.], [2.]])
