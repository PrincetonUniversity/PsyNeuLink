import argparse
import copy
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="graph_scheduler.condition")
warnings.filterwarnings("ignore", message=r"The following arg\(s\) were not specified.*aggregation_function.*")

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler

HELPER_DIR = Path(__file__).resolve().parents[3] / "tests" / "composition" / "pec"
sys.path.insert(0, str(HELPER_DIR))

from test_stab_flex_pec_fit import generate_trial_sequence, get_node, make_input_dict, make_stab_flex  # noqa: E402

from Scripts.Debug.pec_batch_compile.gpu_batch_compile_benchmark import (  # noqa: E402
    _PECGridDDMPlan,
    _PECGridStabilityFlexibilityPlan,
    _ddm_inputs,
    _ddm_parameter_sets,
    _make_ddm_comp,
    _make_stability_flexibility_comp,
    _stability_flexibility_inputs,
    _stability_flexibility_parameter_sets,
)


def _make_pec_grid_ddm(noise, trials, estimates):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=noise,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    data = pd.DataFrame(
        {
            "decision": np.ones(trials, dtype=np.float32),
            "response_time": np.full(trials, 0.06, dtype=np.float32),
        }
    )
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec_grid_ddm_correctness",
        nodes=[comp],
        parameters={
            ("rate", decision): np.array([0.1, 2.0]),
            ("threshold", decision): np.array([0.01, 0.1]),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
        ),
        num_estimates=estimates,
        initial_seed=42,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.set_pec_objective_function(lambda sim_data: (float(np.sum(sim_data)), sim_data))
    return pec, comp, decision


def _pec_grid_ddm_values(noise, trials, estimates, stimulus, rate, threshold):
    pec, comp, _ = _make_pec_grid_ddm(noise, trials, estimates)
    pec.controller.function._ll_func = None
    _, sim_data = pec.log_likelihood(rate, threshold, inputs={comp: stimulus})
    return np.asarray(sim_data, dtype=np.float32)[None, None, :, :, :]


def _triton_ddm_values(noise, trials, estimates, stimulus, params, max_steps=64):
    comp, decision = _make_ddm_comp()
    decision.function.parameters.noise.set(noise, None)
    return BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=max_steps).run(
        {decision: stimulus},
        params,
        estimates,
        seed=11,
    ).values


def _make_pec_grid_sf(noise, trials, estimates):
    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=noise,
        lca_noise=noise,
    )
    decision = get_node(comp, "DDM")
    decision_gate = get_node(comp, "DECISION_GATE")
    response_gate = get_node(comp, "RESPONSE_GATE")
    data = pd.DataFrame(
        {
            "decision": np.ones(trials, dtype=np.float32),
            "response_time": np.full(trials, 0.5, dtype=np.float32),
        }
    )
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec_grid_sf_correctness",
        nodes=comp,
        parameters={("threshold", decision): np.array([0.01, 0.1])},
        outcome_variables=[decision_gate.output_ports[0], response_gate.output_ports[0]],
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
        ),
        num_estimates=estimates,
        initial_seed=42,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.set_pec_objective_function(lambda sim_data: (float(np.sum(sim_data)), sim_data))
    return pec, comp


def _sf_inputs(comp, trials, seed=4):
    source_trials = max(16, ((trials + 15) // 16) * 16)
    task, stimulus, cue, correct = generate_trial_sequence(source_trials, 0.5, seed=seed)
    return make_input_dict(comp, task[:trials], stimulus[:trials], cue[:trials], correct[:trials])


def _pec_grid_sf_values(noise, trials, estimates, threshold):
    pec, comp = _make_pec_grid_sf(noise, trials, estimates)
    inputs = _sf_inputs(comp, trials)
    input_copy = {key: copy.deepcopy(value) for key, value in inputs.items()}
    pec_inputs, _ = comp._parse_input_dict(input_copy, pnl.Context(composition=pec))
    pec.controller.set_parameters_in_inputs([0.01], pec_inputs)
    pec.controller.function._ll_func = None
    _, sim_data = pec.log_likelihood(threshold, inputs=pec_inputs)
    return np.asarray(sim_data, dtype=np.float32)[None, None, :, :, :]


def _triton_sf_values(noise, trials, estimates, threshold, max_steps=256):
    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=noise,
        lca_noise=noise,
    )
    inputs = _sf_inputs(comp, trials)
    params = [{"threshold": threshold, "ddm_noise": noise, "lca_noise": noise}]
    return BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=max_steps).run(
        inputs,
        params,
        estimates,
        seed=11,
    ).values


def _summary(values):
    return {
        "decision_mean": float(np.mean(values[..., 0])),
        "rt_mean": float(np.mean(values[..., 1])),
        "decision_std": float(np.std(values[..., 0])),
        "rt_std": float(np.std(values[..., 1])),
        "sum": float(np.sum(values)),
    }


def _print_compare(name, llvm, triton, status, detail):
    print(f"{name}: {status}")
    print(f"  shape llvm={llvm.shape} triton={triton.shape}")
    print(f"  llvm={_summary(llvm)}")
    print(f"  triton={_summary(triton)}")
    print(f"  {detail}")


def _allclose_case(name, llvm, triton, atol=1e-6, rtol=1e-6):
    max_abs = float(np.max(np.abs(llvm - triton)))
    ok = np.allclose(llvm, triton, atol=atol, rtol=rtol)
    _print_compare(name, llvm, triton, "PASS" if ok else "FAIL", f"max_abs={max_abs:.8g}")
    return ok


def _summary_case(name, llvm, triton, decision_tol, rt_tol):
    llvm_summary = _summary(llvm)
    triton_summary = _summary(triton)
    decision_diff = abs(triton_summary["decision_mean"] - llvm_summary["decision_mean"])
    rt_diff = abs(triton_summary["rt_mean"] - llvm_summary["rt_mean"])
    ok = decision_diff <= decision_tol and rt_diff <= rt_tol
    _print_compare(
        name,
        llvm,
        triton,
        "PASS" if ok else "FAIL",
        f"decision_mean_abs_diff={decision_diff:.8g}, rt_mean_abs_diff={rt_diff:.8g}",
    )
    return ok


def run_checks():
    results = []

    stimulus = np.array([[1.0], [-1.0], [1.0], [-1.0]], dtype=np.float32)
    ddm_params = [{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}]
    results.append(
        _allclose_case(
            "ddm_deterministic_elementwise",
            _pec_grid_ddm_values(0.0, 4, 3, stimulus, 1.0, 0.05),
            _triton_ddm_values(0.0, 4, 3, stimulus, ddm_params),
        )
    )

    trials = 32
    estimates = 1024
    params = _ddm_parameter_sets(1)
    llvm_plan = _PECGridDDMPlan("LLVM", trials, estimates)
    llvm = llvm_plan.run(_ddm_inputs(llvm_plan.comp, trials), params, estimates, seed=11).values
    triton_comp, triton_decision = _make_ddm_comp()
    triton = BatchedCompositionCompiler.compile(triton_comp, backend="triton", max_steps=64).run(
        _ddm_inputs(triton_decision, trials),
        params,
        estimates,
        seed=11,
    ).values
    results.append(_summary_case("ddm_stochastic_benchmark_summary", llvm, triton, 0.02, 0.005))

    results.append(
        _allclose_case(
            "stability_flexibility_deterministic_elementwise",
            _pec_grid_sf_values(0.0, 1, 1, 0.05),
            _triton_sf_values(0.0, 1, 1, 0.05),
        )
    )

    trials = 16
    estimates = 1024
    params = _stability_flexibility_parameter_sets(1)
    llvm_plan = _PECGridStabilityFlexibilityPlan("LLVM", trials, estimates)
    llvm = llvm_plan.run(_stability_flexibility_inputs(llvm_plan.comp, trials), params, estimates, seed=11).values
    triton_comp = _make_stability_flexibility_comp()
    triton = BatchedCompositionCompiler.compile(triton_comp, backend="triton", max_steps=256).run(
        _stability_flexibility_inputs(triton_comp, trials),
        params,
        estimates,
        seed=11,
    ).values
    results.append(_summary_case("stability_flexibility_stochastic_benchmark_summary", llvm, triton, 0.05, 0.02))

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare batched Triton kernels with PEC LLVM grid_evaluate outputs.")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any check fails.")
    args = parser.parse_args()
    results = run_checks()
    failures = len([result for result in results if not result])
    print(f"summary: {len(results) - failures} passed, {failures} failed")
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
