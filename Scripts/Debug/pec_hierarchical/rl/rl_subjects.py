"""PsyNeuLink composition and per-subject PECs for the decay-Q bandit + DDM model.

Feedforward LLVM-compilable chain: the reward signal from trial t-1 enters as trial t's input
(zeros on the first trial), so the AdaptiveIntegrator's output at trial t is exactly the
pre-trial-t Q vector of rl_model.q_path — same model, no feedback projections.

    reward input (2)  ->  Q (AdaptiveIntegrator, rate=alpha, never reset)
                      ->  [-1, +1] matrix -> drift scale (Linear, slope=beta)
                      ->  DDM (threshold, ndt; resets each trial) -> decision/response gates
"""

import os
import sys

import numpy as np
import pandas as pd

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CORE = os.path.join(_PARENT, "core")
if _CORE not in sys.path:
    sys.path.append(_CORE)

from rl_model import FIT_RANGES, fit_bounds, transform  # noqa: E402,F401

TIME_STEP_SIZE = 0.01
DDM_NOISE = 1.0


def get_node(comp, name):
    for node in comp.nodes:
        if node.name.startswith(name):
            return node
    raise KeyError(name)


def _make_comp(theta):
    alpha, beta, thresh, ndt = (float(x) for x in theta)
    rewardIn = pnl.ProcessingMechanism(input_shapes=2, name="Reward Signal [r*c0, r*c1]")
    qMech = pnl.IntegratorMechanism(
        input_shapes=2,
        function=pnl.AdaptiveIntegrator(rate=alpha, initializer=[0.0, 0.0]),
        name="Q Values [Q0, Q1]",
    )
    driftScale = pnl.ProcessingMechanism(
        input_shapes=1, function=pnl.Linear(slope=beta), name="Drift Scale [beta*(Q1-Q0)]"
    )
    decisionMaker = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, threshold=thresh, noise=DDM_NOISE,
            time_step_size=TIME_STEP_SIZE, non_decision_time=ndt,
        ),
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    comp = pnl.Composition(name="decay_q_ddm")
    comp.add_node(rewardIn)
    comp.add_node(qMech)
    comp.add_node(driftScale)
    comp.add_node(decisionMaker)
    comp.add_projection(pnl.MappingProjection(sender=rewardIn, receiver=qMech))
    comp.add_projection(pnl.MappingProjection(sender=qMech, receiver=driftScale,
                                              matrix=np.array([[-1.0], [1.0]])))
    comp.add_projection(pnl.MappingProjection(sender=driftScale, receiver=decisionMaker))

    decisionGate = pnl.ProcessingMechanism(input_shapes=1, name="DECISION_GATE")
    responseGate = pnl.ProcessingMechanism(input_shapes=1, name="RESPONSE_GATE")
    comp.add_node(decisionGate)
    comp.add_node(responseGate)
    comp.add_projection(sender=decisionMaker.output_ports[0], receiver=decisionGate)
    comp.add_projection(sender=decisionMaker.output_ports[1], receiver=responseGate)
    comp.scheduler.add_condition(decisionGate, pnl.WhenFinished(decisionMaker))
    comp.scheduler.add_condition(responseGate, pnl.WhenFinished(decisionMaker))
    return comp


def reward_inputs(choices, rewards):
    """Trial inputs: previous trial's one-hot reward signal (zeros on trial 0)."""
    n = len(choices)
    sig = np.zeros((n, 2))
    idx = np.arange(n - 1)
    sig[idx + 1, np.asarray(choices[:-1], int)] = rewards[:-1]
    return sig


def input_dict(comp, signals):
    return {get_node(comp, "Reward Signal"): [[sig] for sig in signals]}


def make_subject_pec(data, signals, num_estimates, initial_seed):
    comp = _make_comp([0.3, 3.0, 0.7, 0.2])  # values are overwritten during fitting
    qMech = get_node(comp, "Q Values")
    driftScale = get_node(comp, "Drift Scale")
    ddm = get_node(comp, "DDM")
    pec = pnl.ParameterEstimationComposition(
        name="pec_rl_subject", nodes=comp,
        parameters={
            ("rate", qMech): np.linspace(*FIT_RANGES["alpha"], 1000),
            ("slope", driftScale): np.linspace(*FIT_RANGES["beta"], 1000),
            ("threshold", ddm): np.linspace(*FIT_RANGES["threshold"], 1000),
            ("non_decision_time", ddm): np.linspace(*FIT_RANGES["ndt"], 1000),
        },
        outcome_variables=[
            get_node(comp, "DECISION_GATE").output_ports[0],
            get_node(comp, "RESPONSE_GATE").output_ports[0],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(method="differential_evolution", max_iterations=1),
        num_estimates=num_estimates, initial_seed=initial_seed,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, comp


def subject_payloads(group, num_estimates):
    """Serializable per-subject payloads from rl_model.generate_group_data output."""
    payloads = []
    for s, sub in enumerate(group["subjects"]):
        df = pd.DataFrame({"decision": sub["choices"].astype(float), "response_time": sub["rts"]})
        df["decision"] = df["decision"].astype("category")
        payloads.append({
            "data": df, "signals": reward_inputs(sub["choices"], sub["rewards"]),
            "seed": 100 + s, "num_estimates": num_estimates,
        })
    return payloads


def rl_pec_factory(payload):
    """Top-level, picklable factory: subject payload -> (pec, inputs) for a worker."""
    pec, comp = make_subject_pec(payload["data"], payload["signals"],
                                 payload["num_estimates"], payload["seed"])
    return pec, input_dict(comp, payload["signals"])
