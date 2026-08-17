"""Compare deterministic CSI co-evolution on Triton GPU and the LLVM PEC path.

Both run ``param_evals`` parameter values x ``estimates`` x ``trials`` independent
simulations of the CSI surrogate (LCA co-evolving with a collapsing-threshold
DDM).  This benchmark uses the compiler's current exact executable boundary:
``iti=0``, ``csi_repeat=0``, ``csi_switch=1``, ``ddm_noise=0``, and
``lca_noise=0``.  The swept parameter is ``non_decision_time`` (the DDM
``threshold`` is already controlled by the model, so PEC cannot also modulate
it).

Both sides are measured the same way -- compilation separately from steady state
-- because they cost about the same to compile and mixing the two is misleading:

* **steady-state** (warm vs warm) is the number that matters for fitting, where
  one compilation is amortised over hundreds of objective evaluations;
* **cold one-shot** includes compilation for both, and is close to a wash.

An earlier version of this script timed triton warm but LLVM cold, which
overstated the speedup roughly threefold.

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

import psyneulink as pnl  # noqa: E402
from psyneulink.core.batched import (  # noqa: E402
    BatchedCompositionCompiler, batched_node_op, unregister_batched_instance_op,
)
from csi_model_surrogate import make_stab_flex  # noqa: E402


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
        _node(comp, "Cue Stimulus Interval"): np.tile([[1], [3]], (half, 1))[:trials],
    }


def _ndt_values(param_evals):
    return list(np.linspace(0.25, 0.35, param_evals))


def run_triton(trials, estimates, param_evals, max_steps, seed, noise=0.0, repeats=5):
    comp = make_stab_flex(iti=0, csi_repeat=0, csi_switch=1, threshold_collapse=-0.001,
                          ddm_noise=noise, lca_noise=0.0)
    inputs = _csi_inputs(comp, trials)
    param_sets = [{"DDM.non_decision_time": float(v)} for v in _ndt_values(param_evals)]
    import torch

    # Cold: lowering plus the first launch, which is what triggers the Triton JIT.
    # Note this depends on Triton's on-disk kernel cache -- point TRITON_CACHE_DIR
    # at an empty directory for a genuinely cold number.
    t = time.perf_counter()
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=max_steps)
    res = plan.run(inputs=inputs, parameter_sets=param_sets, num_estimates=estimates, seed=seed)
    torch.cuda.synchronize()
    cold = time.perf_counter() - t

    # Warm: the same sweep with everything compiled and the GPU up to clock.
    runs = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t = time.perf_counter()
        res = plan.run(inputs=inputs, parameter_sets=param_sets, num_estimates=estimates, seed=seed)
        torch.cuda.synchronize()
        runs.append(time.perf_counter() - t)
    warm = float(np.median(runs))
    values = np.asarray(res.values)
    return {
        "cold": cold,
        "warm": warm,
        "compile": cold - warm,
        "checksum": float(np.sum(values)),
        "checksum_after_trial0": float(np.sum(values[:, :, 1:, :, :])),
    }


def run_llvm(trials, estimates, param_evals, seed, noise=0.0, repeats=5):
    comp = make_stab_flex(iti=0, csi_repeat=0, csi_switch=1, threshold_collapse=-0.001,
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

    def sweep():
        """One pass over the parameter values; return time and smoke checks."""
        t = time.perf_counter()
        total = 0.0
        after_trial0 = 0.0
        for v in _ndt_values(param_evals):
            pec.controller.function._ll_func = None
            _, sim = pec.log_likelihood(float(v), inputs=pec_inputs, return_sim_data=True)
            values = np.asarray(sim)
            total += float(np.sum(values))
            after_trial0 += float(np.sum(values[1:, ...]))
        return time.perf_counter() - t, total, after_trial0

    # The first pass pays LLVM's compilation.  Median repeated warm passes just
    # as on the Triton side; timing only LLVM's first pass used to overstate the
    # GPU speedup materially.
    cold, checksum, checksum_after_trial0 = sweep()
    warm_runs = [sweep()[0] for _ in range(repeats)]
    warm = float(np.median(warm_runs))
    return {
        "cold": cold,
        "warm": warm,
        "compile": cold - warm,
        "checksum": checksum,
        "checksum_after_trial0": checksum_after_trial0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=64)
    ap.add_argument("--estimates", type=int, default=256)
    ap.add_argument("--param-evals", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=512)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="DDM noise; the current typed CSI boundary requires 0",
    )
    ap.add_argument("--repeats", type=int, default=5,
                    help="warm sweeps per backend to median over")
    ap.add_argument("--skip-llvm", action="store_true")
    ap.add_argument("--skip-triton", action="store_true")
    args = ap.parse_args()
    if args.noise != 0.0:
        ap.error("the current typed CSI benchmark requires --noise 0")

    sims = args.param_evals * args.estimates * args.trials
    print(f"CSI surrogate: {args.param_evals} params x {args.estimates} estimates x "
          f"{args.trials} trials = {sims:,} simulations\n")
    try:
        tri = llvm = None
        header = f"  {'':<24}{'compile':>12}{'warm sweep':>14}{'cold total':>13}   {'warm sims/s':>14}"
        print(header)
        if not args.skip_triton:
            tri = run_triton(args.trials, args.estimates, args.param_evals,
                             args.max_steps, args.seed, args.noise, args.repeats)
            print(f"  {'triton GPU (coevolving)':<24}{tri['compile']*1000:>11.1f}ms"
                  f"{tri['warm']*1000:>13.1f}ms{tri['cold']*1000:>12.1f}ms"
                  f"   {sims/tri['warm']:>14,.0f}   checksum={tri['checksum']:.1f}"
                  f" ({tri['checksum_after_trial0']:.1f} excluding trial 0)")
        if not args.skip_llvm:
            llvm = run_llvm(
                args.trials,
                args.estimates,
                args.param_evals,
                args.seed,
                args.noise,
                args.repeats,
            )
            print(f"  {'LLVM PEC grid_evaluate':<24}{llvm['compile']*1000:>11.1f}ms"
                  f"{llvm['warm']*1000:>13.1f}ms{llvm['cold']*1000:>12.1f}ms"
                  f"   {sims/llvm['warm']:>14,.0f}   checksum={llvm['checksum']:.1f}"
                  f" ({llvm['checksum_after_trial0']:.1f} excluding trial 0)")
        if tri and llvm:
            print(f"\n  -> steady-state speedup: {llvm['warm']/tri['warm']:6.1f}x   "
                  "(warm vs warm; what a fitting loop sees, compilation amortised "
                  "over many objective evaluations)")
            print(f"  -> cold one-shot speedup:{llvm['cold']/tri['cold']:6.1f}x   "
                  "(compile + one sweep; the two compile costs are comparable, so "
                  "a single ad-hoc run is close to a wash)")
            print("\n  Note: the checksum sums decisions and response times together, so "
                  "errors in one\n  can cancel the other -- it is a smoke test, not an "
                  "accuracy measure.  The executable support gate is\n  "
                  "test_batched_csi_coevolving_acceptance.py, which compares "
                  "fresh Python,\n  Triton interpreter, and compiled GPU output.")
    finally:
        unregister_batched_instance_op("Drift Rate Value")


if __name__ == "__main__":
    main()
