import argparse
import copy
import gc
import statistics
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="graph_scheduler.condition")
warnings.filterwarnings("ignore", message=r"The following arg\(s\) were not specified.*aggregation_function.*")

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler


HELPER_DIR = Path(__file__).resolve().parents[3] / "tests" / "composition" / "pec"
sys.path.insert(0, str(HELPER_DIR))

from test_stab_flex_pec_fit import generate_trial_sequence, get_node, make_input_dict, make_stab_flex  # noqa: E402


_PEC_GRID_MODES = {
    "llvm": "LLVM",
    "ptx": "PTX",
}
_MONOLITHIC_SF_FUSION = "stability_flexibility"


class _BatchedResultAdapter:
    def __init__(self, values):
        self.values = values


def _with_graph_fusion_kind(plan, fusion_kind: str):
    graph = replace(plan.ir.graph, fusion_kind=fusion_kind)
    metadata = dict(plan.ir.metadata)
    metadata["fusion_kind"] = fusion_kind
    ir = replace(plan.ir, graph=graph, metadata=metadata)
    report_metadata = dict(plan.capability_report.metadata)
    report_metadata["fusion_kind"] = fusion_kind
    report = replace(plan.capability_report, metadata=report_metadata)
    return replace(plan, ir=ir, capability_report=report)


class _PECGridDDMPlan:
    def __init__(self, execution_mode, num_trials, num_estimates):
        self.comp, self.decision = _make_ddm_comp()
        self.execution_mode = execution_mode
        data = pd.DataFrame(
            {
                "decision": np.ones(num_trials, dtype=np.float32),
                "response_time": np.full(num_trials, 0.06, dtype=np.float32),
            }
        )
        data["decision"] = data["decision"].astype("category")
        self.pec = pnl.ParameterEstimationComposition(
            name="pec_grid_ddm_benchmark",
            nodes=[self.comp],
            parameters={
                ("rate", self.decision): np.array([0.1, 2.0]),
                ("threshold", self.decision): np.array([0.01, 0.1]),
            },
            outcome_variables=[
                self.decision.output_ports[pnl.DECISION_OUTCOME],
                self.decision.output_ports[pnl.RESPONSE_TIME],
            ],
            data=data,
            optimization_function=pnl.PECOptimizationFunction(
                method="differential_evolution",
                max_iterations=1,
            ),
            num_estimates=num_estimates,
            initial_seed=42,
        )
        self.pec.controller.parameters.comp_execution_mode.set(execution_mode)
        self.pec.controller.function.set_pec_objective_function(
            lambda sim_data: (float(np.sum(sim_data)), sim_data)
        )

    def run(
        self,
        inputs,
        parameter_sets,
        num_estimates,
        subject_slices=None,
        seed=None,
    ):
        if subject_slices is not None:
            raise ValueError("PEC grid DDM benchmark currently supports one unsliced subject.")

        stimulus = np.asarray(next(iter(inputs.values())), dtype=float).reshape(-1, 1)
        pec_inputs = {self.comp: stimulus}
        parameter_values = []

        for param in parameter_sets:
            self.pec.controller.function._ll_func = None
            _, sim_data = self.pec.log_likelihood(
                param["rate"],
                param["threshold"],
                inputs=pec_inputs,
            )
            parameter_values.append(np.asarray(sim_data, dtype=np.float32))

        return _BatchedResultAdapter(np.asarray(parameter_values, dtype=np.float32)[:, None, :, :, :])


class _PECGridDDMGraphPlan:
    def __init__(self, execution_mode, num_trials, num_estimates):
        self.comp, self.source, self.decision = _make_ddm_graph_comp()
        self.execution_mode = execution_mode
        data = pd.DataFrame(
            {
                "decision": np.ones(num_trials, dtype=np.float32),
                "response_time": np.full(num_trials, 0.06, dtype=np.float32),
            }
        )
        data["decision"] = data["decision"].astype("category")
        self.pec = pnl.ParameterEstimationComposition(
            name="pec_grid_ddm_graph_benchmark",
            nodes=[self.comp],
            parameters={
                ("rate", self.decision): np.array([0.1, 2.0]),
                ("threshold", self.decision): np.array([0.01, 0.1]),
            },
            outcome_variables=[
                self.decision.output_ports[pnl.DECISION_OUTCOME],
                self.decision.output_ports[pnl.RESPONSE_TIME],
            ],
            data=data,
            optimization_function=pnl.PECOptimizationFunction(
                method="differential_evolution",
                max_iterations=1,
            ),
            num_estimates=num_estimates,
            initial_seed=42,
        )
        self.pec.controller.parameters.comp_execution_mode.set(execution_mode)
        self.pec.controller.function.set_pec_objective_function(
            lambda sim_data: (float(np.sum(sim_data)), sim_data)
        )

    def run(
        self,
        inputs,
        parameter_sets,
        num_estimates,
        subject_slices=None,
        seed=None,
    ):
        if subject_slices is not None:
            raise ValueError("PEC grid DDM graph benchmark currently supports one unsliced subject.")

        stimulus = np.asarray(next(iter(inputs.values())), dtype=float).reshape(-1, 1)
        pec_inputs = {self.comp: stimulus}
        parameter_values = []

        for param in parameter_sets:
            self.pec.controller.function._ll_func = None
            _, sim_data = self.pec.log_likelihood(
                param["rate"],
                param["threshold"],
                inputs=pec_inputs,
            )
            parameter_values.append(np.asarray(sim_data, dtype=np.float32))

        return _BatchedResultAdapter(np.asarray(parameter_values, dtype=np.float32)[:, None, :, :, :])


class _PECGridStabilityFlexibilityPlan:
    def __init__(self, execution_mode, num_trials, num_estimates):
        self.comp = _make_stability_flexibility_comp()
        self.execution_mode = execution_mode
        decision = get_node(self.comp, "DDM")
        decision_gate = get_node(self.comp, "DECISION_GATE")
        response_gate = get_node(self.comp, "RESPONSE_GATE")
        data = pd.DataFrame(
            {
                "decision": np.ones(num_trials, dtype=np.float32),
                "response_time": np.full(num_trials, 0.5, dtype=np.float32),
            }
        )
        data["decision"] = data["decision"].astype("category")
        self.pec = pnl.ParameterEstimationComposition(
            name="pec_grid_stability_flexibility_benchmark",
            nodes=self.comp,
            parameters={
                ("threshold", decision): np.array([0.01, 0.1]),
            },
            outcome_variables=[
                decision_gate.output_ports[0],
                response_gate.output_ports[0],
            ],
            data=data,
            optimization_function=pnl.PECOptimizationFunction(
                method="differential_evolution",
                max_iterations=1,
            ),
            num_estimates=num_estimates,
            initial_seed=42,
        )
        self.pec.controller.parameters.comp_execution_mode.set(execution_mode)
        self.pec.controller.function.set_pec_objective_function(
            lambda sim_data: (float(np.sum(sim_data)), sim_data)
        )

    def run(
        self,
        inputs,
        parameter_sets,
        num_estimates,
        subject_slices=None,
        seed=None,
    ):
        if subject_slices is not None:
            raise ValueError("PEC grid stability-flexibility benchmark supports one unsliced subject.")

        input_copy = {key: copy.deepcopy(value) for key, value in inputs.items()}
        pec_inputs, _ = self.comp._parse_input_dict(input_copy, pnl.Context(composition=self.pec))
        dummy_params = [value[0] for value in self.pec.controller.function.fit_param_bounds.values()]
        self.pec.controller.set_parameters_in_inputs(dummy_params, pec_inputs)

        parameter_values = []
        for param in parameter_sets:
            self.pec.controller.function._ll_func = None
            _, sim_data = self.pec.log_likelihood(param["threshold"], inputs=pec_inputs)
            parameter_values.append(np.asarray(sim_data, dtype=np.float32))

        return _BatchedResultAdapter(np.asarray(parameter_values, dtype=np.float32)[:, None, :, :, :])


def _make_ddm_comp():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.2,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def _make_ddm_graph_comp():
    source = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(slope=1.0, intercept=0.0),
        name="DRIFT_PREP",
    )
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=0.2,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=[[source, decision]]), source, decision


def _ddm_inputs(decision, trials):
    stimulus = np.ones((trials, 1), dtype=np.float32)
    stimulus[1::2, 0] = -1.0
    return {decision: stimulus}


def _ddm_parameter_sets(count):
    rates = np.linspace(0.7, 1.3, count, dtype=np.float32)
    thresholds = np.linspace(0.04, 0.08, count, dtype=np.float32)
    return [
        {
            "rate": float(rate),
            "threshold": float(threshold),
            "noise": 0.2,
            "time_step_size": 0.01,
            "non_decision_time": 0.0,
        }
        for rate, threshold in zip(rates, thresholds)
    ]


def _make_stability_flexibility_comp():
    return make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.1,
        lca_noise=0.05,
    )


def _stability_flexibility_inputs(comp, trials):
    source_trials = max(16, ((trials + 15) // 16) * 16)
    task, stimulus, cue, correct = generate_trial_sequence(source_trials, 0.5, seed=3)
    return make_input_dict(comp, task[:trials], stimulus[:trials], cue[:trials], correct[:trials])


def _stability_flexibility_parameter_sets(count):
    thresholds = np.linspace(0.04, 0.08, count, dtype=np.float32)
    return [
        {
            "threshold": float(threshold),
            "ddm_noise": 0.1,
            "lca_noise": 0.05,
        }
        for threshold in thresholds
    ]


def _sync_cuda():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_peak_mb():
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _reset_gpu_peak():
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _time_run(plan, inputs, params, estimates, seed, repeats, warmups):
    def run_once():
        _sync_cuda()
        start = time.perf_counter()
        result = plan.run(
            inputs=inputs,
            parameter_sets=params,
            num_estimates=estimates,
            seed=seed,
        )
        _sync_cuda()
        return time.perf_counter() - start, result

    gc.collect()
    _reset_gpu_peak()
    first_duration, result = run_once()
    for _ in range(warmups):
        run_once()

    durations = []
    for _ in range(repeats):
        duration, result = run_once()
        durations.append(duration)

    values = result.values
    return {
        "first_ms": first_duration * 1000,
        "median_ms": statistics.median(durations) * 1000,
        "min_ms": min(durations) * 1000,
        "max_ms": max(durations) * 1000,
        "peak_mb": _gpu_peak_mb(),
        "checksum": float(np.sum(values)),
        "shape": values.shape,
    }


def _print_header():
    print(
        "model,backend,params,subjects,trials,estimates,lanes,max_steps,"
        "first_ms,median_ms,min_ms,max_ms,peak_gpu_mb,checksum,shape"
    )


def _print_row(model, backend, params, subjects, trials, estimates, max_steps, metrics):
    lanes = params * subjects * trials * estimates
    peak = "" if metrics["peak_mb"] is None else f"{metrics['peak_mb']:.1f}"
    print(
        f"{model},{backend},{params},{subjects},{trials},{estimates},{lanes},{max_steps},"
        f"{metrics['first_ms']:.3f},{metrics['median_ms']:.3f},"
        f"{metrics['min_ms']:.3f},{metrics['max_ms']:.3f},{peak},"
        f"{metrics['checksum']:.6g},{metrics['shape']}"
    )


def _run_ddm(args):
    comp, decision = _make_ddm_comp()
    for params_count, trials, estimates in args.ddm_cases:
        inputs = _ddm_inputs(decision, trials)
        params = _ddm_parameter_sets(params_count)
        for backend in args.backends:
            if backend == "triton_fused":
                print("ddm,triton_fused,SKIP,monolithic SF fusion is only available for stability-flexibility")
                continue
            if backend in _PEC_GRID_MODES:
                plan = _PECGridDDMPlan(_PEC_GRID_MODES[backend], trials, estimates)
                compiled_inputs = _ddm_inputs(plan.comp, trials)
            else:
                report = BatchedCompositionCompiler.diagnose(comp, backend=backend, max_steps=args.ddm_max_steps)
                if not report.is_supported or not report.backend_available:
                    print(f"ddm,{backend},SKIP,{'; '.join(report.unsupported_reasons)}")
                    continue
                plan = BatchedCompositionCompiler.compile(comp, backend=backend, max_steps=args.ddm_max_steps)
                compiled_inputs = inputs

            try:
                metrics = _time_run(plan, compiled_inputs, params, estimates, args.seed, args.repeats, args.warmups)
            except Exception as error:
                print(f"ddm,{backend},SKIP,{type(error).__name__}: {error}")
                continue
            _print_row("ddm", backend, params_count, 1, trials, estimates, args.ddm_max_steps, metrics)


def _run_ddm_graph(args):
    comp, source, _ = _make_ddm_graph_comp()
    for params_count, trials, estimates in args.ddm_graph_cases:
        inputs = _ddm_inputs(source, trials)
        params = _ddm_parameter_sets(params_count)
        for backend in args.backends:
            if backend == "triton_fused":
                print("ddm_graph,triton_fused,SKIP,monolithic SF fusion is only available for stability-flexibility")
                continue
            if backend in _PEC_GRID_MODES:
                plan = _PECGridDDMGraphPlan(_PEC_GRID_MODES[backend], trials, estimates)
                compiled_inputs = _ddm_inputs(plan.comp, trials)
            else:
                report = BatchedCompositionCompiler.diagnose(comp, backend=backend, max_steps=args.ddm_max_steps)
                if not report.is_supported or not report.backend_available:
                    print(f"ddm_graph,{backend},SKIP,{'; '.join(report.unsupported_reasons)}")
                    continue
                if backend == "triton" and report.metadata.get("fusion_kind") != "ddm_graph":
                    print(
                        "ddm_graph,triton,SKIP,"
                        f"expected fusion_kind ddm_graph, got {report.metadata.get('fusion_kind')}"
                    )
                    continue
                plan = BatchedCompositionCompiler.compile(comp, backend=backend, max_steps=args.ddm_max_steps)
                compiled_inputs = inputs

            try:
                metrics = _time_run(plan, compiled_inputs, params, estimates, args.seed, args.repeats, args.warmups)
            except Exception as error:
                print(f"ddm_graph,{backend},SKIP,{type(error).__name__}: {error}")
                continue
            _print_row("ddm_graph", backend, params_count, 1, trials, estimates, args.ddm_max_steps, metrics)


def _run_stability_flexibility(args):
    for params_count, trials, estimates in args.sf_cases:
        for backend in args.backends:
            if backend in _PEC_GRID_MODES:
                plan = _PECGridStabilityFlexibilityPlan(_PEC_GRID_MODES[backend], trials, estimates)
                inputs = _stability_flexibility_inputs(plan.comp, trials)
            else:
                comp = _make_stability_flexibility_comp()
                inputs = _stability_flexibility_inputs(comp, trials)
                diagnose_backend = "triton" if backend == "triton_fused" else backend
                report = BatchedCompositionCompiler.diagnose(comp, backend=diagnose_backend, max_steps=args.sf_max_steps)
                if not report.is_supported or not report.backend_available:
                    print(f"stability_flexibility,{backend},SKIP,{'; '.join(report.unsupported_reasons)}")
                    continue
                plan = BatchedCompositionCompiler.compile(comp, backend=diagnose_backend, max_steps=args.sf_max_steps)
                if backend == "triton_fused":
                    plan = _with_graph_fusion_kind(plan, _MONOLITHIC_SF_FUSION)

            params = _stability_flexibility_parameter_sets(params_count)
            try:
                metrics = _time_run(plan, inputs, params, estimates, args.seed, args.repeats, args.warmups)
            except Exception as error:
                print(f"stability_flexibility,{backend},SKIP,{type(error).__name__}: {error}")
                continue
            _print_row(
                "stability_flexibility",
                backend,
                params_count,
                1,
                trials,
                estimates,
                args.sf_max_steps,
                metrics,
            )


def _case(value):
    fields = value.split("x")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("cases must have the form paramsxtrialsxestimates")
    return tuple(int(field) for field in fields)


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the experimental batched simulator against current PEC "
            "grid_evaluate compiled baselines."
        )
    )
    parser.add_argument(
        "--backend",
        action="append",
        choices=("reference", "triton", "triton_fused", "llvm", "ptx"),
        dest="backends",
        help="'llvm' and 'ptx' run current PEC grid_evaluate; 'triton' runs the batched simulator.",
    )
    parser.add_argument("--ddm-case", action="append", type=_case, dest="ddm_cases")
    parser.add_argument("--ddm-graph-case", action="append", type=_case, dest="ddm_graph_cases")
    parser.add_argument("--sf-case", action="append", type=_case, dest="sf_cases")
    parser.add_argument("--ddm-max-steps", type=int, default=64)
    parser.add_argument("--sf-max-steps", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-ddm", action="store_true")
    parser.add_argument("--skip-ddm-graph", action="store_true")
    parser.add_argument("--skip-sf", action="store_true")
    args = parser.parse_args()
    args.backends = args.backends or ["triton"]
    args.ddm_cases = args.ddm_cases or [(1, 128, 1024), (8, 128, 1024), (16, 128, 4096)]
    args.ddm_graph_cases = args.ddm_graph_cases or [(1, 128, 1024), (8, 128, 1024), (16, 128, 4096)]
    args.sf_cases = args.sf_cases or [(1, 1, 1024), (8, 1, 1024)]
    return args


def main():
    args = _parse_args()
    _print_header()
    if not args.skip_ddm:
        _run_ddm(args)
    if not args.skip_ddm_graph:
        _run_ddm_graph(args)
    if not args.skip_sf:
        _run_stability_flexibility(args)


if __name__ == "__main__":
    main()
