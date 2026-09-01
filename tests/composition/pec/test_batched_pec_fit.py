"""Batched routing of the PEC data-fitting objective (roadmap step 10).

The full optimizer loop is far too slow under Triton's CPU interpreter, so these
tests exercise the objective **closure** that ``PECOptimizationFunction`` builds
when ``batched_backend`` is set: it compiles the model, feeds the raw stimulus as
batched inputs, supplies the fitting parameters as parameter sets, and scores the
data with the on-device histogram likelihood.  Correctness of the recovered
parameters at PEC scale is validated separately on GPU.
"""

import numpy as np
import optuna
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedTrialParameter
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
    OptimizationFunctionError,
)

pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


def _make_ddm_pec(batched_backend, num_estimates=150, threshold=0.2):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.5, threshold=threshold,
            non_decision_time=0.0, time_step_size=0.01,
            # Seed the *data-generating* run.  Without this the experimental
            # data is redrawn on every test run, so any assertion about which
            # parameter the objective prefers is really an assertion about one
            # random 5-trial sample -- it passes or fails by luck.
            seed=1234,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)

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


@pytest.mark.triton_interpreter
def test_batched_objective_scores_true_threshold_highest():
    """The batched data-fitting objective peaks near the data-generating threshold.

    This is a statistical assertion, so it has to be given enough signal to be
    about the objective rather than about a lucky draw.  Two things matter, and
    the fixture now pins both: the experimental data is seeded (see
    `_make_ddm_pec`), and the density is estimated from enough simulations that
    its noise sits well under the real gap between candidate thresholds.

    For this 5-trial dataset the true log-likelihood gap between 0.1 and 0.2 is
    only ~2 units (5 trials do not distinguish them sharply), while 0.4 is far
    worse.  At 500 estimates the estimator noise is a fraction of that gap, and
    0.2 won for every simulation seed tried.
    """

    pec, comp = _make_ddm_pec("triton_cpu", num_estimates=500, threshold=0.2)
    opt_func = pec.controller.function

    objfunc = opt_func._make_objective_func()
    # objfunc returns total log-likelihood (higher = better fit).
    lls = {t: objfunc(t) for t in (0.1, 0.2, 0.4)}
    best = max(lls, key=lls.get)
    assert best == 0.2, lls


@pytest.mark.triton_interpreter
def test_public_log_likelihood_reuses_batched_objective():
    """Fixed-parameter rescoring follows the configured batched backend."""

    pec, comp = _make_ddm_pec("triton_cpu", num_estimates=100, threshold=0.2)
    opt_func = pec.controller.function
    inputs = {comp.nodes["DDM"]: np.zeros((5, 1))}

    expected = opt_func._make_objective_func()(0.2)
    actual = pec.log_likelihood(0.2, inputs=inputs)

    assert actual == pytest.approx(expected, rel=1e-6, abs=1e-5)


def test_batched_backend_none_uses_default_path():
    """Without batched_backend the objective is the standard (non-batched) closure."""
    pec, comp = _make_ddm_pec(None)
    opt_func = pec.controller.function
    assert opt_func.batched_backend is None
    # No plan is compiled for the default path.
    assert opt_func._batched_plan is None


def test_parameter_batching_requires_local_batched_backend():
    with pytest.raises(ValueError, match="requires a batched_backend"):
        PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(popsize=2),
            batched_parameter_batch_size=2,
        )

    with pytest.raises(ValueError, match="cannot be combined"):
        PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(popsize=2),
            batched_backend="triton",
            batched_parameter_batch_size=2,
            distributed=True,
        )


def test_batched_likelihood_smoothing_options_are_validated():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_backend="triton",
            batched_smoothing_sigma=-0.5,
        )

    with pytest.raises(ValueError, match="finite and nonnegative"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_backend="triton",
            batched_pseudocount=-0.5,
        )

    with pytest.raises(ValueError, match="require a batched_backend"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_smoothing_sigma=0.5,
        )


def test_triton_launch_options_require_compiled_gpu_backend():
    with pytest.raises(ValueError, match="requires batched_backend='triton'"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_backend="triton_cpu",
            batched_triton_launch_options={"block_size": 64},
        )

    options = {"block_size": 64, "num_warps": 2, "maxnreg": 96}
    function = PECOptimizationFunction(
        method="differential_evolution",
        batched_backend="triton",
        batched_triton_launch_options=options,
    )
    options["block_size"] = 256
    assert function.batched_triton_launch_options == {
        "block_size": 64,
        "num_warps": 2,
        "maxnreg": 96,
    }


def test_deterministic_history_requires_exclusive_gpu_mode():
    with pytest.raises(ValueError, match="requires batched_backend='triton'"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_backend="triton_cpu",
            deterministic_history_likelihood=True,
        )

    with pytest.raises(ValueError, match="cannot both be True"):
        PECOptimizationFunction(
            method="differential_evolution",
            batched_backend="triton",
            conditioned_likelihood=True,
            deterministic_history_likelihood=True,
        )


@pytest.mark.triton_interpreter
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


def test_batched_conditional_parameter_is_selected_per_trial(batched_backend):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            rate=1.0,
            noise=0.0,
            threshold=0.2,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="conditional DDM",
    )
    comp = pnl.Composition(pathways=decision)
    data = pd.DataFrame(
        {
            "decision": [1.0, 1.0, 1.0, 1.0],
            "response_time": [0.1, 0.4, 0.1, 0.4],
            "condition": ["low", "high", "low", "high"],
        }
    )
    data["decision"] = data["decision"].astype("category")
    data["condition"] = data["condition"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={
            ("threshold", decision): np.linspace(0.1, 0.4, 4),
            ("rate", decision): np.linspace(0.5, 1.5, 3),
        },
        depends_on={("threshold", decision): "condition"},
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(
                popsize=2,
                seed=1,
            ),
            max_iterations=5,
            batched_backend=batched_backend,
            batched_max_steps=100,
            batched_bins=10,
            batched_seed=1,
            batched_parameter_batch_size=2,
        ),
        num_estimates=2,
    )
    pec.controller._pec_input_values_by_node = {decision: np.ones((4, 1))}

    opt_func = pec.controller.function
    parameter_set = opt_func._batched_parameter_set((0.1, 0.4, 1.0))
    threshold = parameter_set[f"{decision.name}.threshold"]
    assert isinstance(threshold, BatchedTrialParameter)
    assert threshold.values.tolist() == [0.1, 0.4, 0.1, 0.4]
    assert parameter_set[f"{decision.name}.rate"] == 1.0

    objective = opt_func._make_objective_func()
    single_value = objective(0.1, 0.4, 1.0)
    batch_values = objective._batched_parameter_sets(
        [(0.1, 0.4, 1.0), (0.4, 0.1, 1.0)]
    )
    assert batch_values.shape == (2,)
    assert batch_values[0] == pytest.approx(single_value, rel=1e-6, abs=1e-5)
    assert np.all(np.isfinite(batch_values))

    fit_result = opt_func._fit_optuna(
        objective,
        opt_func.method,
        display_iter=False,
    )
    assert opt_func.num_evals == 5
    assert np.isfinite(fit_result["optimal_value"])
