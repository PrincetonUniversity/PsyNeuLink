"""DDM model provider: a single drift-diffusion mechanism, 3 fit parameters.

The reference benchmark workload (cheap simulation, KDE-bound). See the package
docstring for the provider contract.
"""

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.globals.utilities import set_global_seed

# Ground-truth generative parameters (also defines the model's fixed params).
TRUE_PARAMS = dict(
    starting_value=0.0, rate=0.3, noise=1.0,
    threshold=0.6, non_decision_time=0.15, time_step_size=0.01,
)

# Fit parameters: name -> (low, high). Order defines the log_likelihood arg order.
FIT_BOUNDS = {
    "rate": (-0.5, 0.5),
    "threshold": (0.5, 1.0),
    "non_decision_time": (0.0, 1.0),
}

# True value for each FIT parameter (for recovery %). Same keys as FIT_BOUNDS.
# (Kept separate from TRUE_PARAMS because a model's fit-param name may differ
# from its generative-param name -- e.g. stab-flex's "slope" vs "automaticity".)
TRUE_FIT_VALUES = {name: TRUE_PARAMS[name] for name in FIT_BOUNDS}

INITIAL_SEED = 42       # PEC simulation seed (with CRN -> reproducible)
_INPUT_SEED = 12345     # trial-input sequence seed (deterministic data/inputs)


def build_comp():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(**TRUE_PARAMS),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision)


def make_inputs(comp, num_trials):
    """Node-keyed inputs; deterministic in num_trials (so data and worker match)."""
    decision = comp.nodes[0]
    rng = np.random.default_rng(_INPUT_SEED)
    trial_inputs = rng.choice([5.0, -5.0], size=(num_trials, 1), p=[0.1, 0.9])
    return {decision: trial_inputs}


def make_data(num_trials):
    set_global_seed(0)  # reproducible synthetic data
    comp = build_comp()
    comp.run(inputs=make_inputs(comp, num_trials))
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results)), columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def build_pec(comp, data, num_estimates, optimization_function=None):
    decision = comp.nodes[0]
    fit_parameters = {
        (name, decision): np.linspace(lo, hi, 1000)
        for name, (lo, hi) in FIT_BOUNDS.items()
    }
    if optimization_function is None:
        # Unused for likelihood-only evaluation, but required by the PEC API.
        optimization_function = pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        )
    pec = pnl.ParameterEstimationComposition(
        name="pec_ddm",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=optimization_function,
        num_estimates=num_estimates,
        initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec
