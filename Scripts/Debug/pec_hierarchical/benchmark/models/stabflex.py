"""Stability-Flexibility model provider: a 12-mechanism task-switching network
(recurrent LCA control + DDM), 4 fit parameters.

The "heavy simulation" workload -- each trial runs the LCA for up to ~1200 steps
plus the DDM -- so estimate-level threading scales far better than on the DDM.
Mirrors Scripts/Debug/stability_flexibility/stability_flexibility_pec_fit.py.
"""

import os
import sys

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.globals.utilities import set_global_seed

# Make the stability_flexibility model importable (it lives a few dirs up).
_SF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "stability_flexibility")
)
if _SF_DIR not in sys.path:
    sys.path.insert(0, _SF_DIR)
from stability_flexibility import make_stab_flex, generate_trial_sequence  # noqa: E402


def _node(comp, base):
    """Look up a node by name, tolerating PNL's "-N" suffixing.

    PNL's global name registry renames duplicate-named mechanisms (e.g. when the
    comp is rebuilt in the same process: make_data then run_regular) to
    "<name>-1", "<name>-2", ..., so an exact-name lookup on the second build
    fails. Match the exact name first, else a "<name>-<int>" rebuild duplicate.
    """
    for n in comp.nodes:
        if n.name == base:
            return n
    for n in comp.nodes:
        if n.name.startswith(base + "-") and n.name[len(base) + 1:].isdigit():
            return n
    raise KeyError(f"{base!r} not found among {[n.name for n in comp.nodes]}")

# Full generative parameters (the model's fixed config + ground truth).
TRUE_PARAMS = dict(
    gain=3.0, leak=3.0, competition=2.0, lca_time_step_size=0.01,
    non_decision_time=0.2, automaticity=0.01, starting_value=0.0,
    threshold=0.1, ddm_noise=0.1, lca_noise=0.0, scale=0.2,
    ddm_time_step_size=0.01,
)

# Fit parameters: name -> (low, high). Order defines the log_likelihood arg order.
FIT_BOUNDS = {
    "gain": (1.0, 10.0),
    "slope": (0.0, 0.1),               # automaticity weight (congruenceWeighting.slope)
    "threshold": (0.01, 0.5),
    "non_decision_time": (0.1, 0.4),
}

# True value for each FIT parameter (note "slope" maps to "automaticity").
TRUE_FIT_VALUES = {
    "gain": TRUE_PARAMS["gain"],
    "slope": TRUE_PARAMS["automaticity"],
    "threshold": TRUE_PARAMS["threshold"],
    "non_decision_time": TRUE_PARAMS["non_decision_time"],
}

INITIAL_SEED = 42       # PEC simulation seed (with CRN -> reproducible)
_SEQ_SEED = 0           # trial-sequence seed (deterministic data/inputs)


def build_comp():
    return make_stab_flex(**TRUE_PARAMS)


def make_inputs(comp, num_trials):
    """Node-keyed 4-layer inputs; deterministic in num_trials."""
    taskLayer = comp.nodes["Task Input [I1, I2]"]
    stimulusInfo = comp.nodes["Stimulus Input [S1, S2]"]
    cueInterval = comp.nodes["Cue-Stimulus Interval"]
    correctInfo = comp.nodes["Correct Response Info"]

    # Generate enough counterbalanced trials (multiple of 16), then take num_trials.
    seq_n = max(240, ((num_trials + 15) // 16) * 16)
    taskTrain, stimulusTrain, cueTrain, correctResponse = generate_trial_sequence(
        seq_n, 0.5, seed=_SEQ_SEED
    )
    taskTrain = taskTrain[:num_trials]
    stimulusTrain = stimulusTrain[:num_trials]
    # CSI is in time steps; /10 because the original was tuned for timestep 0.001.
    cueTrain = [c / 10.0 for c in cueTrain[:num_trials]]
    correctResponse = correctResponse[:num_trials]

    return {
        taskLayer: [[np.array(taskTrain[i])] for i in range(num_trials)],
        stimulusInfo: [[np.array(stimulusTrain[i])] for i in range(num_trials)],
        cueInterval: [[np.array([cueTrain[i]])] for i in range(num_trials)],
        correctInfo: [[np.array([correctResponse[i]])] for i in range(num_trials)],
    }


def make_data(num_trials):
    set_global_seed(0)  # reproducible synthetic data
    comp = build_comp()
    inputs = make_inputs(comp, num_trials)
    comp.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
    # Per-trial outputs: column 0 dropped; columns 1:2 are decision + response_time.
    data = pd.DataFrame(
        np.squeeze(np.array(comp.results))[:, 1:], columns=["decision", "response_time"]
    )
    data["decision"] = data["decision"].astype("category")
    return data


def build_pec(comp, data, num_estimates, optimization_function=None):
    controlModule = comp.nodes["Task Activations [Act1, Act2]"]
    congruenceWeighting = comp.nodes["Automaticity-weighted Stimulus Input [w*S1, w*S2]"]
    decisionMaker = comp.nodes["DDM"]
    decisionGate = comp.nodes["DECISION_GATE"]
    responseGate = comp.nodes["RESPONSE_GATE"]

    fit_parameters = {
        ("gain", controlModule): np.linspace(*FIT_BOUNDS["gain"], 1000),
        ("slope", congruenceWeighting): np.linspace(*FIT_BOUNDS["slope"], 1000),
        ("threshold", decisionMaker): np.linspace(*FIT_BOUNDS["threshold"], 1000),
        ("non_decision_time", decisionMaker): np.linspace(*FIT_BOUNDS["non_decision_time"], 1000),
    }
    if optimization_function is None:
        optimization_function = pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        )
    pec = pnl.ParameterEstimationComposition(
        name="pec_stabflex",
        nodes=comp,
        parameters=fit_parameters,
        outcome_variables=[
            decisionGate.output_ports[0],
            responseGate.output_ports[0],
        ],
        data=data,
        optimization_function=optimization_function,
        num_estimates=num_estimates,
        initial_seed=INITIAL_SEED,
        same_seed_for_all_parameter_combinations=True,  # common random numbers
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec
