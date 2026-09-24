"""Batch the local DAWA LC/LCA model using the ordinary PEC compiler.

The shared model builder and this driver both default to recurrent scheduling.
See dawa_batched_README.md for the corrected one-pass control dependency issue.
"""

import argparse
import importlib.util
import json
import re
from pathlib import Path
import time

import numpy as np
import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler


SOURCE = Path(__file__).with_name("dawa_lca_model") / "full_lca_model_lc.py"
DEFAULTS = dict(
    c_gain=10., c_leak=7., c_competition=3., c_bias=0., c_w=4.,
    s_gain=5., s_leak=8., s_competition=8., s_bias=-.45,
    d_gain=5., d_leak=8., d_competition=8., d_bias=-.45, d_noise=0.,
    r_gain=5., r_leak=8., r_competition=8., r_bias=-.45, r_noise=.1,
    r_threshold=.3, non_decision_time=.2, time_step_size=.01,
    w1=1., w2=1.2, sdr_bias=-.45, lc_base_gain=5., lc_scaling=1.,
    lc_mode=.9, lc_input=.3, lc_threshold=.5,
)


def source_module(path=SOURCE):
    spec = importlib.util.spec_from_file_location("dawa_lca_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def node(composition, name):
    matches = [item for item in composition.nodes if re.sub(r"-\d+$", "", item.name) == name]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name!r} node, found {len(matches)}")
    return matches[0]


def build_model(*, trials=4, schedule="recurrent", deterministic=False, seed=3, source=SOURCE):
    module = source_module(source)
    parameters = dict(DEFAULTS)
    if deterministic:
        parameters["r_noise"] = 0.
    composition = module.make_lca_model(**parameters)
    if schedule == "recurrent":
        # Also apply the default to older/custom source builders. The bundled
        # builder already specifies these conditions for all callers.
        for name in (
            "Control Units\n[Color, Location]", "Stimulus Units\n[Red, Blue, Left, Right]",
            "Decision Units\n[Left, Right]", "Response Units\n[Left, Right]",
            "Weighted Color Input", "Weighted Location Input",
            "Weighted Color Stimulus", "Weighted Location Stimulus",
        ):
            composition.scheduler.add_condition(node(composition, name), pnl.Always())
    tasks, stimuli = module.conflict_task_sequence(trials, seed=seed)
    inputs = {node(composition, "Task Input"): tasks, node(composition, "Stimulus Input"): stimuli}
    inputs.update({node(composition, name): np.zeros((trials, 1))
                   for name in ("Bias Mechanism", "w1 Mechanism", "w2 Mechanism")})
    outputs = tuple(node(composition, name).output_port for name in ("DECISION_GATE", "RT_GATE"))
    return composition, inputs, outputs


def fit_surface(composition):
    return {
        ("termination_threshold", node(composition, "Response Units\n[Left, Right]")): (.25, .7, .3),
        ("intercept", node(composition, "RT_GATE")): (.1, .3, .2),
        ("intercept", node(composition, "Bias Mechanism")): (-.5, 0., -.45),
        ("gain", node(composition, "Control Units\n[Color, Location]")): (5., 20., 10.),
        ("mode", node(composition, "LC")): (.1, .9, .9),
        ("slope", node(composition, "LC")): (1., 4., 1.),
        ("intercept", node(composition, "LC")): (3., 10., 5.),
    }


def pec_smoke(composition, inputs, outputs, observed, *, backend, max_steps, estimates, seed):
    """Evaluate two candidates through PEC, including conditional fit lanes."""
    import pandas as pd

    surface = fit_surface(composition)
    data = pd.DataFrame(observed, columns=["decision", "response_time"])
    data["decision"] = pd.Categorical(data["decision"], categories=[0., 1.])
    data["subject_nr"] = pd.Categorical(np.repeat(np.arange(2), (len(data) + 1) // 2)[:len(data)])
    data["PrevCongruency"] = pd.Categorical(np.arange(len(data)) % 2)
    depends = {key: "subject_nr" for key in (
        ("termination_threshold", node(composition, "Response Units\n[Left, Right]")),
        ("gain", node(composition, "Control Units\n[Color, Location]")),
        ("intercept", node(composition, "RT_GATE")),
        ("slope", node(composition, "LC")),
    )}
    depends[("mode", node(composition, "LC"))] = "PrevCongruency"
    pec = pnl.ParameterEstimationComposition(
        nodes=composition, parameters={key: np.array(values[:2]) for key, values in surface.items()},
        depends_on=depends, outcome_variables=list(outputs), data=data,
        likelihood_include_mask=np.ones(len(data), dtype=bool),
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1, batched_backend=backend,
            batched_max_steps=max_steps, batched_seed=seed, batched_bins=20,
            batched_bin_range=[(0., max_steps * .01 + .3)], batched_pseudocount=1.,
        ), num_estimates=estimates, initial_seed=seed,
    )
    pec.controller._pec_input_values_by_node = inputs
    values = []
    for key in pec.fit_parameters:
        if key == ("mode", node(composition, "LC")):
            mode_coordinate = len(values)
        levels = len(pec.cond_levels[key]) if key in pec.cond_levels else 1
        values.extend([surface[key][2]] * levels)
    objective = pec.controller.function._make_objective_func()
    candidates = np.asarray([values, values])
    candidates[1, 0] += .025
    candidates[1, mode_coordinate] = .7
    scores = objective._batched_parameter_sets(candidates)
    if not np.all(np.isfinite(scores)):
        raise AssertionError(f"PEC produced nonfinite scores: {scores}")
    return {"candidate_scores": np.asarray(scores).tolist(),
            "fit_coordinates": pec.controller.function.fit_param_names}


def reference_results(composition, inputs, outputs, mode):
    if mode == "python":
        rows = []

        def collect():
            rows.append([float(port.parameters.value.get(composition).ravel()[0]) for port in outputs])

        composition.run(inputs, call_after_trial=collect)
        return np.asarray(rows)
    composition.run(inputs, execution_mode=pnl.ExecutionMode.LLVMRun)
    indices = []
    for output in outputs:
        matches = [i for i, port in enumerate(composition.output_CIM.input_ports)
                   if any(projection.sender is output for projection in port.path_afferents)]
        if len(matches) != 1:
            raise AssertionError(f"Ambiguous reference output: {output}")
        indices.append(matches[0])
    return np.asarray([[float(np.asarray(row[i]).ravel()[0]) for i in indices] for row in composition.results])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--schedule", choices=("source", "recurrent"), default="recurrent",
                        help="Recurrent processing is the default; source uses the loaded builder's own conditions")
    parser.add_argument("--backend", choices=("triton", "triton_cpu"), default="triton")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--estimates", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--reference", choices=("none", "python", "llvm"), default="none")
    parser.add_argument("--pec-smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="Optional NPZ samples; a neighboring JSON stores the report")
    args = parser.parse_args()
    if args.reference != "none" and not args.deterministic:
        parser.error("--reference requires --deterministic; GPU and LLVM use different RNG streams")
    if args.trials < 2 or args.estimates < 1:
        parser.error("Use at least two trials and one estimate")
    composition, inputs, outputs = build_model(trials=args.trials, schedule=args.schedule,
                                                deterministic=args.deterministic, source=args.source)
    plan = BatchedCompositionCompiler.compile(composition, backend=args.backend, outputs=outputs, max_steps=args.max_steps)
    candidates = [{f"{node.name}.{parameter}": values[2] for (parameter, node), values in fit_surface(composition).items()}]
    candidates.append({**candidates[0], f"{node(composition, 'LC').name}.mode": .7})
    start = time.perf_counter()
    result = plan.run(inputs, candidates, args.estimates, seed=args.seed, strict_truncation=True)
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    replay = plan.run(inputs, candidates, args.estimates, seed=args.seed, strict_truncation=True)
    warm_seconds = time.perf_counter() - start
    np.testing.assert_array_equal(result.values, replay.values)
    report = {
        "schedule": args.schedule, "backend": args.backend, "shape": list(result.values.shape),
        "first_run_seconds": first_seconds, "warm_run_seconds": warm_seconds,
        "trial_estimates_per_second": 2 * args.trials * args.estimates / warm_seconds,
        "mean_rt_by_candidate": result.values[..., 1].mean(axis=(1, 2, 3)).tolist(),
        "reproducible": True,
    }
    if args.reference != "none":
        reference = reference_results(composition, inputs, outputs, args.reference)
        actual = result.values[0, 0, :, 0]
        np.testing.assert_allclose(actual, reference, rtol=1e-5, atol=2e-6)
        report["reference"] = args.reference
        report["max_reference_error"] = float(np.max(np.abs(actual - reference)))
    if args.pec_smoke:
        # Use a fresh graph: a native reference run leaves live scheduler state.
        composition, inputs, outputs = build_model(trials=args.trials, schedule=args.schedule,
                                                    deterministic=args.deterministic, source=args.source)
        report["pec"] = pec_smoke(composition, inputs, outputs, result.values[0, 0, :, 0],
                                  backend=args.backend, max_steps=args.max_steps, estimates=args.estimates, seed=args.seed)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, values=result.values)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
