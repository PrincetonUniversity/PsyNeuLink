"""Distributed PEC fitting of the Stability-Flexibility model.

This is a larger example than ``Scripts/Debug/pec_dask/study.py``: a full
Stability-Flexibility composition with an LCA control module feeding a DDM. The
Dask-specific settings are ``distributed=True`` and ``pec_factory`` in
``distributed_options``.

Run it two ways:

  Single node::

      python stability_flexibility_dask.py

  Multiple nodes::

      srun -n <workers+2> python -m psyneulink.dask_run stability_flexibility_dask.py

  rank 0 is the scheduler, rank 1 is the driver, and ranks 2+ are workers.
  Requires the ``psyneulink[dask]`` extra and LLVM execution.

The model itself is reused from ``stability_flexibility.py`` (same directory).
The SLURM script puts this directory on ``PYTHONPATH`` so workers can import
``make_stab_flex`` and ``generate_trial_sequence``.
"""

import os
import sys

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)

# Make the co-located model importable on the driver. Workers rely on PYTHONPATH
# (set by the SLURM script) since they import this module by name, not from here.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from stability_flexibility import make_stab_flex, generate_trial_sequence  # noqa: E402

# Ground-truth parameters used to synthesize the "observed" data (replace `data`
# with real subject data for a true fit), the fitted-parameter search bounds, and
# the run-size knobs. num_estimates and the CMA-ES budget below dominate runtime.
TIME_STEP_SIZE = 0.01
SF_PARAMS = dict(
    gain=3.0, leak=3.0, competition=2.0, lca_time_step_size=TIME_STEP_SIZE,
    non_decision_time=0.2, automaticity=0.01, starting_value=0.0, threshold=0.1,
    ddm_noise=0.1, lca_noise=0.0, scale=0.2, ddm_time_step_size=TIME_STEP_SIZE,
)
# Fitted parameters: (suggest-name, ground-truth key). "slope" recovers automaticity.
TRUE = {"gain": 3.0, "slope": 0.01, "threshold": 0.1, "non_decision_time": 0.2}
FIT_BOUNDS = {
    "gain": (1.0, 10.0),
    "slope": (0.0, 0.1),
    "threshold": (0.01, 0.5),
    "non_decision_time": (0.1, 0.4),
}
NUM_TRIALS = 200
NUM_ESTIMATES = 10000
INITIAL_SEED = 42
TRIAL_SEQ_SEED = 0

# CMA-ES population per generation == candidates dispatched per ask/tell round.
MAX_CONCURRENT = 16
NUM_GENERATIONS = 60


def get_node(comp, name):
    """Composition node whose name starts with ``name``.

    Tolerant of PsyNeuLink's dedup suffixes (e.g. ``DDM`` vs ``DDM-1``) when more
    than one Stability-Flexibility composition is built in the same process.
    """
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    raise KeyError(name)


def trial_sequence():
    """Deterministic (seeded) trial sequence -- rebuilt identically on every worker."""
    taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(
        240, 0.5, seed=TRIAL_SEQ_SEED
    )
    taskTrain = taskTrain[:NUM_TRIALS]
    stimulusTrain = stimulusTrain[:NUM_TRIALS]
    # CSI is in time steps; scale by ten because the sequence is authored for a
    # 0.001 step size but the model runs at 0.01.
    cueTrain = [c / 10.0 for c in cueTrain[:NUM_TRIALS]]
    correctResponse = correctResponse[:NUM_TRIALS]
    return taskTrain, stimulusTrain, cueTrain, correctResponse


def input_dict(comp, taskTrain, stimulusTrain, cueTrain, correctResponse):
    """Map the trial sequence onto the composition's four origin nodes."""
    return {
        get_node(comp, "Task Input [I1, I2]"): [[np.array(taskTrain[i])] for i in range(NUM_TRIALS)],
        get_node(comp, "Stimulus Input [S1, S2]"): [[np.array(stimulusTrain[i])] for i in range(NUM_TRIALS)],
        get_node(comp, "Cue-Stimulus Interval"): [[np.array([cueTrain[i]])] for i in range(NUM_TRIALS)],
        get_node(comp, "Correct Response Info"): [[np.array([correctResponse[i]])] for i in range(NUM_TRIALS)],
    }


def fit_parameters(comp):
    """Fitted parameters keyed to the composition's mechanisms."""
    controlModule = get_node(comp, "Task Activations [Act1, Act2]")
    congruenceWeighting = get_node(comp, "Automaticity-weighted Stimulus Input [w*S1, w*S2]")
    decisionMaker = get_node(comp, "DDM")
    return {
        ("gain", controlModule): np.linspace(*FIT_BOUNDS["gain"], 1000),
        ("slope", congruenceWeighting): np.linspace(*FIT_BOUNDS["slope"], 1000),
        ("threshold", decisionMaker): np.linspace(*FIT_BOUNDS["threshold"], 1000),
        ("non_decision_time", decisionMaker): np.linspace(*FIT_BOUNDS["non_decision_time"], 1000),
    }


def outcome_variables(comp):
    return [
        get_node(comp, "DECISION_GATE").output_ports[0],
        get_node(comp, "RESPONSE_GATE").output_ports[0],
    ]


def pec_factory(data):
    """Worker recipe: rebuild a serial PEC and inputs from observed data."""
    comp = make_stab_flex(**SF_PARAMS)
    pec = pnl.ParameterEstimationComposition(
        name="stabflex_worker", nodes=comp,
        parameters=fit_parameters(comp),
        outcome_variables=outcome_variables(comp),
        data=data,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=NUM_ESTIMATES, initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, input_dict(comp, *trial_sequence())


def make_observed_data(comp, inputs):
    """Synthesize data from SF_PARAMS by running the model once (replace with real data)."""
    comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results))[:, 1:], columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def main():
    import optuna

    # One composition serves both data synthesis and the driver PEC (as in
    # stability_flexibility_pec_fit.py), so the fitted-parameter names stay unsuffixed.
    comp = make_stab_flex(**SF_PARAMS)
    inputs = input_dict(comp, *trial_sequence())
    data = make_observed_data(comp, inputs)

    # CMA-ES with popsize pinned to the per-round batch; max_concurrent_evaluations
    # fixes the batch so the trajectory does not depend on the live worker count.
    optimizer = PECOptimizationFunction(
        method=optuna.samplers.CmaEsSampler(seed=0, popsize=MAX_CONCURRENT),
        max_iterations=MAX_CONCURRENT * NUM_GENERATIONS,
        distributed=True,
        distributed_options={
            "pec_factory": pec_factory,
            "max_concurrent_evaluations": MAX_CONCURRENT,
        },
    )

    pec = pnl.ParameterEstimationComposition(
        name="stabflex_mle", nodes=comp,
        parameters=fit_parameters(comp),
        outcome_variables=outcome_variables(comp),
        data=data,
        optimization_function=optimizer,
        num_estimates=NUM_ESTIMATES, initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")

    pec.run(inputs=inputs)

    print("optimal log-likelihood:", pec.optimal_value)
    for name, val in pec.optimized_parameter_values.items():
        true = TRUE[name.split(".")[-1].split("[")[0]]
        print(f"  {name}: recovered={val:.4f}  true={true}")


if __name__ == "__main__":
    main()
