"""Compare the batched co-evolving CSI surrogate (Triton GPU) against the current
PEC grid_evaluate LLVM baseline, on the same workload.

Both run ``param_evals`` parameter values x ``estimates`` x ``trials`` independent
stochastic simulations of the CSI surrogate (LCA co-evolving with a collapsing-
threshold DDM).  The swept parameter is ``non_decision_time`` (the DDM
``threshold`` is already controlled by the model, so PEC cannot also modulate it).

    .venv/bin/python Scripts/Debug/pec_batch_compile/csi_triton_vs_llvm.py \
        --trials 64 --estimates 256 --param-evals 4

Requires CUDA + triton for the Triton path; LLVM is CPU.
"""
import argparse
import re
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="graph_scheduler.condition")
warnings.filterwarnings("ignore", message=r"The following arg\(s\) were not specified.*")

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler, batched_node_op, unregister_batched_instance_op,
)
from csi_model_surrogate import make_stab_flex


@batched_node_op("Drift Rate Value")
def _drift_rate(x0, x1, x2, x3, x4, x5, x6):
    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (pos - neg) * x6


def _node(comp, base):
    return next(n for n in comp.nodes if re.sub(r"-\d+$", "", n.name) == base)


def _csi_inputs(comp, trials):
    half = (trials + 1) // 2
    return {
        _node(comp, "Stimulus Input"): np.tile([[1, 0, 1, 0], [0, 1, 0, 1]], (half, 1))[:trials],
        _node(comp, "Task Input"): np.tile([[1, 0], [0, 1]], (half, 1))[:trials],
        _node(comp, "Correct Response"): np.tile([[1], [-1]], (half, 1))[:trials],
        _node(comp, "Cue Stimulus Interval"): np.zeros((trials, 1)),
    }


def _ndt_values(param_evals):
    return list(np.linspace(0.25, 0.35, param_evals))


def run_triton(trials, estimates, param_evals, max_steps, seed, noise=0.1):
    comp = make_stab_flex(iti=0, csi_repeat=0, csi_switch=0, threshold_collapse=-0.001,
                          ddm_noise=noise, lca_noise=0.0)
    inputs = _csi_inputs(comp, trials)
    param_sets = [{"DDM.non_decision_time": float(v)} for v in _ndt_values(param_evals)]
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=max_steps)
    # warm up (compile + GPU)
    plan.run(inputs=inputs, parameter_sets=param_sets, num_estimates=estimates, seed=seed)
    import torch
    torch.cuda.synchronize()
    t = time.perf_counter()
    res = plan.run(inputs=inputs, parameter_sets=param_sets, num_estimates=estimates, seed=seed)
    torch.cuda.synchronize()
    return time.perf_counter() - t, float(np.sum(res.values))


def run_llvm(trials, estimates, param_evals, seed, noise=0.1):
    comp = make_stab_flex(iti=0, csi_repeat=0, csi_switch=0, threshold_collapse=-0.001,
                          ddm_noise=noise, lca_noise=0.0)
    inputs = _csi_inputs(comp, trials)
    decision, dg, rg = _node(comp, "DDM"), _node(comp, "DECISION_GATE"), _node(comp, "RESPONSE_GATE")
    data = pd.DataFrame({"decision": np.ones(trials, dtype=np.float32),
                         "response_time": np.full(trials, 0.5, dtype=np.float32)})
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec_csi", nodes=comp,
        parameters={("non_decision_time", decision): np.array([0.1, 0.5])},
        outcome_variables=[dg.output_ports[0], rg.output_ports[0]], data=data,
        optimization_function=pnl.PECOptimizationFunction(method="differential_evolution", max_iterations=1),
        num_estimates=estimates, initial_seed=seed)
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.set_pec_objective_function(lambda s: float(np.sum(s)))
    # PEC expands the raw per-node stimulus itself (`_parse_input_dict` +
    # `set_parameters_in_inputs`), so pass it through unexpanded.  It does
    # require an entry for *every* model INPUT node, including the Threshold
    # Mechanism -- the batched compiler absorbs that one into the DDM boundary
    # and so does not need it, but the LLVM baseline runs the real model.
    pec_inputs = dict(inputs)
    pec_inputs[_node(comp, "Threshold Mechanism")] = np.zeros((trials, 1))
    t = time.perf_counter()
    total = 0.0
    for v in _ndt_values(param_evals):
        pec.controller.function._ll_func = None
        _, sim = pec.log_likelihood(float(v), inputs=pec_inputs, return_sim_data=True)
        total += float(np.sum(np.asarray(sim)))
    return time.perf_counter() - t, total


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--estimates", type=int, default=256)
    ap.add_argument("--param-evals", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--skip-llvm", action="store_true")
    ap.add_argument("--skip-triton", action="store_true")
    args = ap.parse_args()

    sims = args.param_evals * args.estimates * args.trials
    print(f"CSI surrogate: {args.param_evals} params x {args.estimates} estimates x "
          f"{args.trials} trials = {sims:,} simulations")
    try:
        t_tri = t_llvm = None
        if not args.skip_triton:
            t_tri, chk_tri = run_triton(args.trials, args.estimates, args.param_evals, args.max_steps, args.seed, args.noise)
            print(f"  triton GPU (coevolving): {t_tri*1000:9.1f} ms  "
                  f"({sims/t_tri:,.0f} sims/s)  checksum={chk_tri:.1f}")
        if not args.skip_llvm:
            t_llvm, chk_llvm = run_llvm(args.trials, args.estimates, args.param_evals, args.seed, args.noise)
            print(f"  LLVM PEC grid_evaluate : {t_llvm*1000:9.1f} ms  "
                  f"({sims/t_llvm:,.0f} sims/s)  checksum={chk_llvm:.1f}")
        if t_tri and t_llvm:
            print(f"  -> speedup: {t_llvm/t_tri:.1f}x")
    finally:
        unregister_batched_instance_op("Drift Rate Value")


if __name__ == "__main__":
    main()
