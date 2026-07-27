"""Synthetic multi-subject Stability-Flexibility data and per-subject PEC construction.

The composition-agnostic hierarchical EM machinery (laplace_em, dask_estep, pec_likelihood) is
reused unchanged; only the per-subject model and data generation differ from the DDM case. This is
the generality test: a 4-parameter LCA->DDM composition with a simulation-only likelihood.

Fitted parameters: gain (control LCA), slope (automaticity weighting), threshold and
non_decision_time (DDM).
"""

import os
import sys

import numpy as np
import pandas as pd
import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SF_DIR = os.path.join(os.path.dirname(_PARENT), "stability_flexibility")
for _path in (_SF_DIR, os.path.join(_PARENT, "core")):
    if _path not in sys.path:
        sys.path.append(_path)
from stability_flexibility import make_stab_flex, generate_trial_sequence  # noqa: E402

from transforms import BoundedTransform  # noqa: E402

TIME_STEP_SIZE = 0.01
SF_FIXED = dict(leak=3.0, competition=2.0, lca_time_step_size=TIME_STEP_SIZE,
                starting_value=0.0, ddm_noise=0.1, lca_noise=0.0, scale=0.2,
                ddm_time_step_size=TIME_STEP_SIZE)
FIT_PARAMS = ("gain", "slope", "threshold", "non_decision_time")
FIT_RANGES = {"gain": (1.0, 10.0), "slope": (0.0, 0.1),
              "threshold": (0.01, 0.5), "non_decision_time": (0.1, 0.4)}
TRIAL_SEQ_SEED = 0


def fit_bounds():
    lo = np.array([FIT_RANGES[p][0] for p in FIT_PARAMS])
    hi = np.array([FIT_RANGES[p][1] for p in FIT_PARAMS])
    return lo, hi


def transform():
    lo, hi = fit_bounds()
    return BoundedTransform(lo, hi)


def get_node(comp, name):
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    raise KeyError(name)


def _trial_sequence(num_trials):
    # n = 240 is kept for backward-compatible sequences; longer requests round up to a multiple
    # of 8 because generate_trial_sequence under-fills its condition lists otherwise.
    n = max(240, int(np.ceil(num_trials / 8)) * 8)
    task, stim, cue, correct = generate_trial_sequence(n, 0.5, seed=TRIAL_SEQ_SEED)
    cue = [c / 10.0 for c in cue]  # sequence authored for 0.001 step, model runs at 0.01
    return task[:num_trials], stim[:num_trials], cue[:num_trials], correct[:num_trials]


def input_dict(comp, num_trials, cue_steps=None):
    task, stim, cue, correct = _trial_sequence(num_trials)
    if cue_steps is not None:
        cue = list(np.asarray(cue_steps, float))  # per-trial CSI override (in model time steps)
    return {
        get_node(comp, "Task Input [I1, I2]"): [[np.array(task[i])] for i in range(num_trials)],
        get_node(comp, "Stimulus Input [S1, S2]"): [[np.array(stim[i])] for i in range(num_trials)],
        get_node(comp, "Cue-Stimulus Interval"): [[np.array([cue[i]])] for i in range(num_trials)],
        get_node(comp, "Correct Response Info"): [[np.array([correct[i]])] for i in range(num_trials)],
    }


def _make_comp(theta, rng_seed=None):
    gain, slope, threshold, ndt = (float(x) for x in theta)
    return make_stab_flex(gain=gain, automaticity=slope, threshold=threshold,
                          non_decision_time=ndt, rng_seed=rng_seed, **SF_FIXED)


def _fit_parameters(comp):
    control = get_node(comp, "Task Activations [Act1, Act2]")
    congruence = get_node(comp, "Automaticity-weighted Stimulus Input [w*S1, w*S2]")
    ddm = get_node(comp, "DDM")
    return {
        ("gain", control): np.linspace(*FIT_RANGES["gain"], 1000),
        ("slope", congruence): np.linspace(*FIT_RANGES["slope"], 1000),
        ("threshold", ddm): np.linspace(*FIT_RANGES["threshold"], 1000),
        ("non_decision_time", ddm): np.linspace(*FIT_RANGES["non_decision_time"], 1000),
    }


def _outcome_variables(comp):
    return [get_node(comp, "DECISION_GATE").output_ports[0],
            get_node(comp, "RESPONSE_GATE").output_ports[0]]


def simulate_subject(theta, num_trials, seed):
    """Run the SF model at natural parameters ``theta`` -> DataFrame[decision, response_time]."""
    comp = _make_comp(theta, rng_seed=seed)
    comp.run(input_dict(comp, num_trials), execution_mode=pnl.ExecutionMode.LLVMRun)
    res = np.squeeze(np.array(comp.results))[:, 1:]
    data = pd.DataFrame(res, columns=["decision", "response_time"])
    data["decision"] = data["decision"].astype("category")
    return data


def make_subject_pec(data, num_estimates, initial_seed):
    comp = _make_comp([3.0, 0.01, 0.1, 0.2])  # values are overwritten during fitting
    pec = pnl.ParameterEstimationComposition(
        name="stabflex_subject", nodes=comp,
        parameters=_fit_parameters(comp),
        outcome_variables=_outcome_variables(comp),
        data=data,
        optimization_function=PECOptimizationFunction(method="differential_evolution", max_iterations=1),
        num_estimates=num_estimates, initial_seed=initial_seed,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp


def stabflex_pec_factory(payload):
    """Top-level, picklable factory: subject payload -> (pec, inputs)."""
    pec, comp = make_subject_pec(payload["data"], payload["num_estimates"], payload["seed"])
    return pec, input_dict(comp, payload["num_trials"])


def build_serial_subjects(payloads):
    return [dict(zip(("pec", "inputs"), stabflex_pec_factory(p))) for p in payloads]


def generate_group_data(n_subjects, beta_z, sigma_z, num_trials, rng, num_estimates):
    """Draw subject parameters from the group, simulate, return serializable payloads."""
    tf = transform()
    beta_z = np.asarray(beta_z, float)
    sigma_z = np.asarray(sigma_z, float)
    z_true = rng.normal(beta_z, np.sqrt(sigma_z), size=(n_subjects, len(FIT_PARAMS)))
    theta_true = np.array([tf.to_natural(z_true[s]) for s in range(n_subjects)])
    payloads = []
    for s in range(n_subjects):
        data = simulate_subject(theta_true[s], num_trials, seed=2000 + s)
        payloads.append({"data": data, "seed": 100 + s,
                         "num_estimates": num_estimates, "num_trials": num_trials})
    return {"payloads": payloads, "z_true": z_true, "theta_true": theta_true,
            "beta_z": beta_z, "sigma_z": sigma_z}
