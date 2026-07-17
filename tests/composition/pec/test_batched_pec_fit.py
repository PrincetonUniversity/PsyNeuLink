"""Batched routing of the PEC data-fitting objective (roadmap step 10).

The full optimizer loop is far too slow under Triton's CPU interpreter, so these
tests exercise the objective **closure** that ``PECOptimizationFunction`` builds
when ``batched_backend`` is set: it compiles the model, feeds the raw stimulus as
batched inputs, supplies the fitting parameters as parameter sets, and scores the
data with the on-device histogram likelihood.  Correctness of the recovered
parameters at PEC scale is validated separately on GPU.
"""

import importlib.util

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
    OptimizationFunctionError,
)

requires_triton = pytest.mark.skipif(
    importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None,
    reason="torch + triton are required for batched CPU (interpret) execution",
)

pytestmark = [pytest.mark.composition, requires_triton]


def _make_ddm_pec(batched_backend, num_estimates=150, threshold=0.2):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.5, threshold=threshold,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)

    rng = np.random.default_rng(0)
    n_trials = 5
    trial_inputs = np.zeros((n_trials, 1))
    comp.run(inputs={decision: trial_inputs})
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")

    fit_parameters = {("threshold", comp.nodes["DDM"]): np.linspace(0.1, 0.4, 100)}
    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            comp.nodes["DDM"].output_ports[pnl.DECISION_OUTCOME],
            comp.nodes["DDM"].output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
            batched_backend=batched_backend,
            batched_max_steps=400,
            batched_bins=30,
            batched_seed=7,
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )
    # Populate the node-keyed stimulus cache the batched path reads (normally done
    # by pec.run via set_pec_inputs_cache).
    pec.controller._pec_input_values_by_node = {comp.nodes["DDM"]: trial_inputs}
    return pec, comp


def test_batched_objective_scores_true_threshold_highest():
    """The batched data-fitting objective peaks near the data-generating threshold."""
    pec, comp = _make_ddm_pec("triton_cpu", num_estimates=200, threshold=0.2)
    opt_func = pec.controller.function

    objfunc = opt_func._make_objective_func()
    # objfunc returns total log-likelihood (higher = better fit).
    lls = {t: objfunc(t) for t in (0.15, 0.2, 0.3)}
    best = max(lls, key=lls.get)
    assert best == 0.2, lls


def test_batched_backend_none_uses_default_path():
    """Without batched_backend the objective is the standard (non-batched) closure."""
    pec, comp = _make_ddm_pec(None)
    opt_func = pec.controller.function
    assert opt_func.batched_backend is None
    # No plan is compiled for the default path.
    assert opt_func._batched_plan is None


def test_batched_unsupported_model_raises_no_silent_fallback(monkeypatch):
    """A model the batched compiler rejects raises a clear error (never falls back)."""
    import psyneulink.core.batched as batched

    pec, comp = _make_ddm_pec("triton_cpu", num_estimates=50, threshold=0.2)
    opt_func = pec.controller.function

    def _reject(*args, **kwargs):
        raise batched.BatchedCompileError("unsupported: made-up reason")

    monkeypatch.setattr(batched.BatchedCompositionCompiler, "compile", staticmethod(_reject))

    objfunc = opt_func._make_objective_func()
    with pytest.raises(OptimizationFunctionError, match="cannot be compiled"):
        objfunc(0.2)
