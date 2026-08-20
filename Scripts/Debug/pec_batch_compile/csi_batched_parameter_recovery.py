"""GPU parameter recovery for the CSI surrogate via the batched fit path.

This drives a full maximum-likelihood parameter-recovery loop for the co-evolving
CSI stability-flexibility surrogate on the GPU, using the experimental batched
Triton simulator + histogram likelihood (roadmap steps 9 & 10) instead of PNL's
LLVM ``grid_evaluate``:

  1. pick "true" parameter values;
  2. generate an experimental data set (one realization per trial) by running the
     batched simulator at the true parameters;
  3. fit the model back to that data with a ``ParameterEstimationComposition``
     whose ``PECOptimizationFunction`` is configured with
     ``batched_backend="triton"`` (optuna CMA-ES over the fitting parameters,
     each objective evaluation = ``num_estimates`` x ``trials`` GPU simulations
     scored by the on-device histogram likelihood);
  4. report recovered vs. true parameters.

The recovery uses the historical five-parameter surface: ``gain`` (LCA control
gain), ``csi_switch`` (Cue Stimulus Interval slope), ``threshold`` (the
Threshold Mechanism's Linear intercept), ``threshold_collapse`` (its
SimpleIntegrator ``offset-integrator_function``), and ``non_decision_time``
(DDM). The batched compiler folds the Threshold Mechanism into the DDM kernel
while retaining both public PEC parameter names as runtime lane aliases.

Usage (requires CUDA + triton):

    .venv/bin/python Scripts/Debug/pec_batch_compile/csi_batched_parameter_recovery.py \
        --trials 128 --estimates 10000 --max-iterations 200 --seed 1

Use ``--backend triton_cpu`` only for a tiny smoke test (interpret mode is slow).

Note on identifiability: the five parameters have overlapping effects on choice
and response-time distributions. Tiny smoke-test settings only verify the code
path; meaningful recovery requires enough trials, estimates, and optimizer
iterations. The histogram bins are anchored to the experimental data, so the
objective remains comparable as ``--estimates`` is varied.
"""
import argparse
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="graph_scheduler.condition")
warnings.filterwarnings("ignore", message=r"The following arg\(s\) were not specified.*")

import optuna  # noqa: E402

import psyneulink as pnl  # noqa: E402
from psyneulink.core.batched import batched_node_op, unregister_batched_instance_op  # noqa: E402

from csi_model_surrogate import make_stab_flex, generate_mixed_task_sequence  # noqa: E402


# The CSI drift-rate node is a UserDefinedFunction (nested logistic, 7 -> 1); it
# needs an instance-level batched op registered before the model can compile.
@batched_node_op("Drift Rate Value")
def _drift_rate(x0, x1, x2, x3, x4, x5, x6):
    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (pos - neg) * x6


# Structural parameters held fixed across data generation and fitting.
FIXED = dict(
    iti=0, csi_repeat=0, leak=7.0, competition=3.0,
    ddm_noise=0.1, lca_noise=0.0,
    lca_time_step_size=0.01, ddm_time_step_size=0.01,
)


def _node(comp, base):
    return next(n for n in comp.nodes if re.sub(r"-\d+$", "", n.name) == base)


def _inputs(comp, tasks, stimuli, correct):
    """Node-keyed CSI inputs for `comp` (``tasks``/``stimuli`` are already one-hot
    vectors from ``generate_mixed_task_sequence``; the Threshold Mechanism input is
    a no-op the batched compiler absorbs)."""
    n = len(tasks)
    csi = [[0] if (t == 0 or list(tasks[t]) == list(tasks[t - 1])) else [1] for t in range(n)]
    return {
        _node(comp, "Task Input"): np.array([list(v) for v in tasks], dtype=float),
        _node(comp, "Stimulus Input"): np.array([list(v) for v in stimuli], dtype=float),
        _node(comp, "Correct Response"): np.array(
            [[float(np.asarray(c).reshape(-1)[0])] for c in correct]),
        _node(comp, "Cue Stimulus Interval"): np.array(csi, dtype=float),
        _node(comp, "Threshold Mechanism"): np.zeros((n, 1)),
    }


def generate_data(true, trials, backend, max_steps, seed, subject):
    """One realization per trial from the batched simulator at the true parameters."""
    from psyneulink.core.batched import BatchedCompositionCompiler

    comp = make_stab_flex(
        gain=true["gain"],
        csi_switch=true["csi_switch"],
        threshold=true["threshold"],
        threshold_collapse=true["threshold_collapse"],
        non_decision_time=true["non_decision_time"],
        **FIXED,
    )
    task, stim, correct, _ = generate_mixed_task_sequence(trials, 0.5, 0.5, subject)
    inputs = _inputs(comp, task, stim, correct)
    plan = BatchedCompositionCompiler.compile(comp, backend=backend, max_steps=max_steps)
    res = plan.run(inputs=inputs, parameter_sets=[{}], num_estimates=1, seed=seed)
    outcomes = res.values[0, 0, :, 0, :]  # [trial, (decision, response_time)]
    data = pd.DataFrame({"decision": outcomes[:, 0], "response_time": outcomes[:, 1]})
    data["decision"] = data["decision"].astype("category")
    return data, (task, stim, correct)


def fit(data, seq, true, trials, estimates, backend, max_steps, max_iterations, seed, subject):
    task, stim, correct = seq
    comp = make_stab_flex(
        gain=true["gain"],
        csi_switch=true["csi_switch"],
        threshold=true["threshold"],
        threshold_collapse=true["threshold_collapse"],
        non_decision_time=true["non_decision_time"],
        **FIXED,
    )
    controlExecution = _node(comp, "Task Activations [C1, C2]")
    cueStimulusInterval = _node(comp, "Cue Stimulus Interval")
    thresholdMechanism = _node(comp, "Threshold Mechanism")
    decisionMaker = _node(comp, "DDM")
    decisionGate = _node(comp, "DECISION_GATE")
    responseGate = _node(comp, "RESPONSE_GATE")

    fit_parameters = {
        ("gain", controlExecution): np.linspace(5.0, 20.0, 151),
        ("slope", cueStimulusInterval): np.linspace(0.0, 50.0, 51),
        ("intercept", thresholdMechanism): np.linspace(0.08, 0.25, 341),
        ("offset-integrator_function", thresholdMechanism): np.linspace(
            -0.003,
            0.0,
            301,
        ),
        ("non_decision_time", decisionMaker): np.linspace(0.0, 0.6, 601),
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec_csi_recovery",
        nodes=comp,
        parameters=fit_parameters,
        outcome_variables=[decisionGate.output_ports[0], responseGate.output_ports[0]],
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method=optuna.samplers.CmaEsSampler(seed=seed),
            max_iterations=max_iterations,
            batched_backend=backend,
            batched_max_steps=max_steps,
            batched_bins=60,
            batched_seed=seed,
        ),
        num_estimates=estimates,
        initial_seed=seed,
    )
    inputs = _inputs(comp, task, stim, correct)
    t0 = time.perf_counter()
    pec.run(inputs=inputs)
    elapsed = time.perf_counter() - t0
    return pec.optimized_parameter_values, pec.optimal_value, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=128)
    ap.add_argument("--estimates", type=int, default=10000)
    ap.add_argument("--max-iterations", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--backend", default="triton", choices=["triton", "triton_cpu"])
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    true = dict(
        gain=float(rng.uniform(5.0, 20.0)),
        csi_switch=float(rng.integers(0, 51)),
        threshold=float(rng.uniform(0.08, 0.25)),
        threshold_collapse=float(rng.uniform(-0.003, 0.0)),
        non_decision_time=float(rng.uniform(0.0, 0.6)),
    )

    print(f"CSI batched parameter recovery  (backend={args.backend}, "
          f"{args.trials} trials x {args.estimates} estimates, "
          f"max_iterations={args.max_iterations})")
    print("True parameters:")
    for k, v in true.items():
        print(f"  {k:>18} = {v:.4f}")

    data, seq = generate_data(true, args.trials, args.backend, args.max_steps,
                              args.seed + 1000, args.subject)
    print(f"Generated {len(data)} trials of experimental data "
          f"(P(decision=1)={np.mean(np.asarray(data['decision'], dtype=float) > 0):.2f}, "
          f"mean RT={data['response_time'].mean():.3f})")

    print("Fitting (batched GPU objective; optuna CMA-ES) ...")
    recovered, ll, elapsed = fit(data, seq, true, args.trials, args.estimates,
                                 args.backend, args.max_steps, args.max_iterations,
                                 args.seed, args.subject)

    name_map = {
        "gain": "gain",
        "slope": "csi_switch",
        "intercept": "threshold",
        "offset-integrator_function": "threshold_collapse",
        "non_decision_time": "non_decision_time",
    }
    print(f"\nDone in {elapsed:.1f}s.  Recovered vs true (log-likelihood={ll:.2f}):")
    print(f"  {'parameter':>18}  {'true':>10}  {'recovered':>10}  {'abs err':>10}")
    for rec_name, value in recovered.items():
        base = rec_name.split(".")[-1] if "." in str(rec_name) else str(rec_name)
        key = name_map.get(base, base)
        tv = true.get(key)
        value = float(np.asarray(value).reshape(-1)[0])
        if tv is None:
            print(f"  {str(rec_name):>18}  {'-':>10}  {value:>10.4f}  {'-':>10}")
        else:
            print(f"  {key:>18}  {tv:>10.4f}  {value:>10.4f}  {abs(value - tv):>10.4f}")


if __name__ == "__main__":
    try:
        main()
    finally:
        unregister_batched_instance_op("Drift Rate Value")
