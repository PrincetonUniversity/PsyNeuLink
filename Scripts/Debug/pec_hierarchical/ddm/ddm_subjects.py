"""Synthetic multi-subject DDM data and per-subject PEC construction (M2).

Each subject's true parameters are drawn from a group Gaussian in unconstrained space and mapped
to natural DDM parameters through the shared bounded transform. One PEC is built per subject,
holding only that subject's simulated data, mirroring the per-worker ``pec_factory`` pattern that
the distributed E-step will use in M3.
"""

import os
import sys

import numpy as np
import pandas as pd

import psyneulink as pnl

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core",):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from transforms import BoundedTransform

# Fit two parameters; the rest of the DDM is fixed. Bounds are wide enough that a group centered
# near the middle stays in the interior, where the logit transform is near-linear and z-space
# variance is identifiable.
FIT_PARAMS = ("rate", "threshold")
FIT_RANGES = {"rate": (-1.5, 1.5), "threshold": (0.3, 1.5)}
FIXED_DDM = dict(starting_value=0.0, noise=1.0, non_decision_time=0.15, time_step_size=0.01)


def fit_bounds():
    lower = np.array([FIT_RANGES[p][0] for p in FIT_PARAMS])
    upper = np.array([FIT_RANGES[p][1] for p in FIT_PARAMS])
    return lower, upper


def transform():
    lower, upper = fit_bounds()
    return BoundedTransform(lower, upper)


def _build_ddm(rate, threshold, *, seed=None):
    fixed = dict(FIXED_DDM)
    if seed is not None:
        fixed["seed"] = int(seed)
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(rate=rate, threshold=threshold, **fixed),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision)


def simulate_subject(theta, trial_inputs, seed, *, return_composition=False):
    """Simulate one subject's DDM data at natural parameters ``theta`` -> DataFrame."""
    rate, threshold = float(theta[0]), float(theta[1])
    seed = int(seed) % (2 ** 32)
    comp = _build_ddm(rate, threshold, seed=seed)
    comp.run(inputs={comp.nodes[0]: trial_inputs}, context=f"sim_{seed}")
    data = pd.DataFrame(np.squeeze(np.array(comp.results)), columns=["decision", "response_time"])
    data["decision"] = data["decision"].astype("category")
    if return_composition:
        return data, comp
    return data


def make_subject_pec(
    data,
    num_estimates=100,
    initial_seed=0,
    likelihood_estimator="kde",
    likelihood_estimator_kwargs=None,
):
    """Build a serial LLVM PEC that scores ``data`` for the two fit parameters."""
    comp = _build_ddm(rate=0.0, threshold=0.9)
    decision = comp.nodes[0]
    pec = pnl.ParameterEstimationComposition(
        name="pec_subject",
        nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*FIT_RANGES["rate"], 1000),
            ("threshold", decision): np.linspace(*FIT_RANGES["threshold"], 1000),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=num_estimates,
        initial_seed=initial_seed,
        same_seed_for_all_parameter_combinations=True,
        likelihood_estimator=likelihood_estimator,
        likelihood_estimator_kwargs=likelihood_estimator_kwargs,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp


# Multiple signed coherences keep the drift rate identifiable across its whole range: a single
# large magnitude saturates choices for high rates (uninformative), while a single small one
# leaves low rates weakly constrained. Easy and hard trials together pin both.
COHERENCES = np.array([0.5, 1.0, 2.0])


def trial_inputs_for(num_trials):
    """Signed multi-coherence stimuli (drift = rate * input)."""
    idx = np.arange(num_trials)
    mag = COHERENCES[idx % len(COHERENCES)]
    sign = np.where((idx // len(COHERENCES)) % 2 == 0, 1.0, -1.0)
    return (sign * mag).reshape(-1, 1)


def ddm_pec_factory(payload):
    """Top-level, picklable factory: subject payload -> (pec, inputs) for a worker."""
    pec, comp = make_subject_pec(
        payload["data"],
        num_estimates=payload["num_estimates"],
        initial_seed=payload["seed"],
        likelihood_estimator=payload.get("likelihood_estimator", "kde"),
        likelihood_estimator_kwargs=payload.get("likelihood_estimator_kwargs"),
    )
    return pec, {comp: trial_inputs_for(payload["num_trials"])}


def build_serial_subjects(payloads):
    """Build per-subject (pec, inputs) pairs in this process (serial path)."""
    return [dict(zip(("pec", "inputs"), ddm_pec_factory(p))) for p in payloads]


def generate_group_data(
    n_subjects,
    beta_z,
    sigma_z,
    num_trials,
    rng,
    num_estimates=100,
    likelihood_estimator="kde",
    likelihood_estimator_kwargs=None,
):
    """Draw subject parameters from the group and simulate per-subject data.

    Returns serializable per-subject payloads (no PECs built here) plus the true z/theta.
    """
    tf = transform()
    beta_z = np.asarray(beta_z, float)
    sigma_z = np.asarray(sigma_z, float)
    z_true = rng.normal(beta_z, np.sqrt(sigma_z), size=(n_subjects, len(FIT_PARAMS)))
    theta_true = np.array([tf.to_natural(z_true[s]) for s in range(n_subjects)])

    trial_inputs = trial_inputs_for(num_trials)
    payloads = []
    for s in range(n_subjects):
        data = simulate_subject(theta_true[s], trial_inputs, seed=1000 + s)
        payloads.append({
            "data": data, "seed": 100 + s,
            "num_estimates": num_estimates, "num_trials": num_trials,
            "likelihood_estimator": likelihood_estimator,
            "likelihood_estimator_kwargs": (
                None
                if likelihood_estimator_kwargs is None
                else dict(likelihood_estimator_kwargs)
            ),
        })

    return {
        "payloads": payloads,
        "z_true": z_true,
        "theta_true": theta_true,
        "trial_inputs": trial_inputs,
        "beta_z": beta_z,
        "sigma_z": sigma_z,
    }
