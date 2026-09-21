#!/usr/bin/env python3
"""Local audit of the original CSI compiled histogram and direct likelihood.

This rescoring experiment does not fit parameters or change model semantics.
All history rows are retained, and only likelihood_include_mask rows are scored.
The direct interval probability is divided by its 1 ms observation resolution.
Raw fitted CSI values and common-grid (nearest 1 ms) values are both recorded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd
import torch

from csi_likelihood_surface_comparison import (
    DEFAULT_DATA, HERE, _build_gpu_problem, _fit_values, _rescale_legacy_time_step,
)
from direct_likelihood import (
    CONDITIONS, CSITrialData, ContinuousCSIParameters, ContinuousCSILikelihood,
    SolverConfig,
)
from direct_likelihood.model import parameter_names
from psyneulink.core.batched import (
    ObservationField, ObservationSpec,
)

FIT_ROOT = HERE / "data fitting/audit/direct-all-subjects-local-pilot"
EXPANDED_FIT = HERE / "data fitting/audit/threshold-schedule-fix-20260916/direct_fit.json"
LOWER = np.array([5., 5., 5., 0., .05, .05, .05, -.3, -.3, -.3, .1, .1, .1])
UPPER = np.array([120., 120., 120., .3, .3, .3, .3, 0., 0., 0., .5, .5, .5])
LAUNCH = {"block_size": 32, "num_warps": 1}


def parameters(vector):
    return ContinuousCSIParameters.from_vector(torch.tensor(vector, dtype=torch.float64))


class GPUProblem:
    def __init__(self, data, subject, vectors):
        self.pec, inputs = _build_gpu_problem(data, subject, 32, 100, .5, .1, 11, 101, 12000, .001)
        self.ff = self.pec.controller.function
        self.ff.batched_observations = ObservationSpec((
            ObservationField(self.pec.outcome_variables[0], "counting"),
            ObservationField(self.pec.outcome_variables[1], "lebesgue", role="event_time",
                             history_timing="ceil_fp32_8ulp"),
        ))
        self.ff.batched_triton_launch_options = LAUNCH
        values = self.values(vectors)
        self.pec.log_likelihood(*values[0], inputs=inputs)
        self.plan = self.ff._compile_batched_plan()
        self.compiled = self.ff._compile_batched_histogram_plan(self.plan)
        self.stimulus = self.ff._batched_stimulus_inputs()
        self.observed = np.asarray(self.pec._data_numpy, dtype=float)
        self.rows = [self.ff._batched_parameter_set(value) for value in values]

    def values(self, vectors):
        frame = pd.DataFrame([parameters(v).as_legacy_dict() for v in vectors])
        frame = _rescale_legacy_time_step(frame, .001)
        frame["Cue Stimulus Interval.slope"] = np.rint(frame["Cue Stimulus Interval.slope"])
        return _fit_values(frame, self.ff.fit_param_names)

    def score(self, include, estimates, seed, rows=None, execution="window"):
        return self.compiled.score(
            self.stimulus, self.observed, self.rows if rows is None else rows,
            num_estimates=estimates, seed=seed, include_mask=include, execution=execution,
            candidate_batch_size=4, max_buffer_bytes=512 * 1024**2,
            triton_launch_options=LAUNCH,
        )

def candidates(base, trials, deep, seed):
    names = list(parameter_names())
    snapped = np.array(base, dtype=float)
    snapped[3] = np.rint(snapped[3] / .001) * .001
    vectors, kinds = [snapped], ["anchor"]
    seen = {tuple(snapped)}

    def add(vector, kind):
        vector = np.clip(vector, LOWER, UPPER)
        vector[3] = np.rint(vector[3] / .001) * .001
        key = tuple(vector)
        if key not in seen:
            vectors.append(vector)
            kinds.append(kind)
            seen.add(key)

    if deep:
        deltas = np.r_[np.maximum(.1 * snapped[:3], 1.), .01,
                       .05 * snapped[4:7], np.maximum(.005, .1 * abs(snapped[7:10])),
                       [.01] * 3]
        for i, delta in enumerate(deltas):
            for sign in (-1, 1):
                v = snapped.copy()
                v[i] += sign * delta
                add(v, f"local:{names[i]}:{sign:+d}")
        # Joint points span the search ranges, with timing and boundary ranges
        # restricted so observations remain inside the mathematical support.
        rng = np.random.default_rng(seed)
        for j in range(8):
            v = rng.uniform(LOWER, UPPER)
            min_included_rt = float(trials.response_time[trials.include].min())
            v[3] = np.rint(rng.uniform(0, max(0., min(.3, min_included_rt - .12))) / .001) * .001
            for c in range(3):
                mask = trials.include.numpy() & (trials.condition_index.numpy() == c)
                available = trials.response_time.numpy()[mask] - v[3] * trials.is_switch.numpy()[mask]
                v[10+c] = rng.uniform(.1, max(.100001, min(.5, available.min() - .01)))
                max_decision = max(.001, float((available - v[10+c]).max()))
                v[7+c] = -rng.uniform(0, min(.3, .9 * v[4+c] / max_decision))
            add(v, f"joint_support_constrained:{j}")
        for kind, indices, value in (("gain_min", slice(0, 3), 5.),
                                     ("gain_max", slice(0, 3), 120.),
                                     ("constant_boundary", slice(7, 10), 0.),
                                     ("fast_collapse", slice(7, 10), -.3),
                                     ("ndt_max", slice(10, 13), .5)):
            v = snapped.copy()
            v[indices] = value
            add(v, "stress:" + kind)
    return np.array(vectors), kinds


def direct_score(likelihood, trials, vector):
    with torch.no_grad():
        result = likelihood.score(parameters(vector), trials, collect_timings=True)
    with np.errstate(divide="ignore"):
        logs = np.log(result.probability.numpy() / trials.rt_resolution)
    include = trials.include.numpy()
    return logs, {
        "direct_log_density": float(logs[include].sum()),
        "direct_zero_rows": int((np.isneginf(logs) & include).sum()),
        "direct_invalid_rows": int(result.diagnostics["invalid_included_rows"].numel()),
        "direct_mass_error": float(result.diagnostics["maximum_mass_error"]),
        "direct_seconds": result.timings["total_seconds"],
    }


def direct_convergence(source, output):
    """Refine PDE and LCA meshes at anchors without changing the source model."""
    destination = output / "direct_convergence.csv"
    records = pd.read_csv(destination).to_dict("records") if destination.exists() else []
    completed = {(row["dataset"], row["configuration"]) for row in records}
    baseline = SolverConfig(ddm_time_step=.001, ddm_spatial_points=65,
                            lca_max_step=.01, native_lca_scan=True, native_ddm_forward=True)
    configs = {
        "spatial129": replace(baseline, ddm_spatial_points=129),
        "spatial257": replace(baseline, ddm_spatial_points=257),
        "spatial257_dt_half_ms": replace(baseline, ddm_spatial_points=257, ddm_time_step=.0005),
        "lca_rk4_one_ms": replace(baseline, lca_max_step=.001),
    }
    for table in sorted(source.glob("*/summary.csv")):
        frame = pd.read_csv(table.parent / "data.csv")
        subject = int(frame.subject_nr.iloc[0])
        trials = CSITrialData.from_csv(table.parent / "data.csv", subject, dtype=torch.float64, device="cpu")
        original = np.load(table.parent / "trial_scores.npz")
        base = original["direct"][0]
        include = trials.include.numpy()
        for name, config in configs.items():
            if (table.parent.name, name) in completed:
                continue
            logs, record = direct_score(ContinuousCSILikelihood(config), trials, original["vectors"][0])
            finite = include & np.isfinite(base) & np.isfinite(logs)
            record.update({"dataset": table.parent.name, "configuration": name,
                           "delta_log_density": float((logs[include]-base[include]).sum()),
                           "finite_trial_mae_change": float(abs(logs[finite]-base[finite]).mean())})
            records.append(record)
            print(f"{table.parent.name} {name}: delta={record['delta_log_density']:.6g}", flush=True)
    pd.DataFrame(records).to_csv(destination, index=False)
    (output / "direct_convergence_configs.json").write_text(json.dumps({k: asdict(v) for k, v in configs.items()}, indent=2) + "\n")


def run_dataset(args, data, subject, label, base, deep, source, selected=None):
    output = args.output / label
    if output.exists():
        if (output / "summary.csv").exists():
            print(f"{label}: already complete", flush=True)
            return
        # Preserve partial output for inspection while allowing a resumed run.
        output.rename(output.with_name(output.name + f".incomplete-{time.time_ns()}"))
    output.mkdir(parents=True)
    started = time.perf_counter()
    trials = CSITrialData.from_csv(data, subject, dtype=torch.float64, device="cpu")
    include = trials.include.numpy()
    frame = pd.read_csv(data)
    frame = frame[(frame.subject_nr == subject) & frame.sequence.isin(CONDITIONS)].reset_index(drop=True)
    frame.to_csv(output / "data.csv", index=False)
    vectors, kinds = candidates(base, trials, deep, args.seed + subject) if selected is None else selected
    pd.DataFrame(vectors, columns=parameter_names()).assign(candidate_kind=kinds).to_csv(output / "parameters.csv", index=False)
    config = SolverConfig(ddm_time_step=.001, ddm_spatial_points=args.spatial_points,
                          lca_max_step=.01, native_lca_scan=True, native_ddm_forward=True)
    likelihood = ContinuousCSILikelihood(config)
    _, exact = direct_score(likelihood, trials, base)
    records, direct_logs = [], []
    for i, vector in enumerate(vectors):
        logs, record = direct_score(likelihood, trials, vector)
        direct_logs.append(logs)
        records.append({"dataset": label, "candidate": i, "kind": kinds[i],
                        "included_rows": int(include.sum()), **record})
    direct_logs = np.array(direct_logs)
    print(f"{label}: {len(vectors)} direct points scored in {time.perf_counter()-started:.1f}s; preparing GPU", flush=True)
    gpu = GPUProblem(data, subject, vectors)
    gpu_logs, empty_counts, elapsed = [], [], []
    for seed in range(args.seed, args.seed + args.repeats):
        t = time.perf_counter()
        scored = gpu.score(include, args.estimates, seed)
        gpu_logs.append(scored.log_factors)
        empty_counts.append((scored.bin_counts.sum(-1) == 0) & include[None, :])
        elapsed.append(time.perf_counter() - t)
        np.testing.assert_allclose(scored.log_likelihood, scored.log_factors[:, include].sum(-1), atol=.005, rtol=1e-5)
        print(f"{label}: GPU seed {seed}, {args.estimates} estimates, {elapsed[-1]:.2f}s", flush=True)
    if args.check_strict:
        strict = gpu.score(include, 512, args.seed, rows=gpu.rows[:1], execution="strict")
        window = gpu.score(include, 512, args.seed, rows=gpu.rows[:1])
        np.testing.assert_allclose(strict.log_factors[:, include], window.log_factors[:, include], atol=1e-5, rtol=1e-5)
    del gpu
    gc.collect()
    gpu_logs = np.array(gpu_logs)
    empty_counts = np.array(empty_counts)
    gpu_mean = gpu_logs.mean(0)
    totals = gpu_logs[:, :, include].sum(-1)
    for i, record in enumerate(records):
        finite = include & np.isfinite(direct_logs[i])
        gap = gpu_mean[i, finite] - direct_logs[i, finite]
        record.update({
            "gpu_log_density_mean": float(totals[:, i].mean()),
            "gpu_log_density_sd": float(totals[:, i].std(ddof=1)) if args.repeats > 1 else float("nan"),
            "gap_total": float(totals[:, i].mean() - record["direct_log_density"]),
            "finite_rows": int(finite.sum()), "finite_gap_mean": float(gap.mean()),
            "finite_gap_mae": float(abs(gap).mean()),
            "finite_gap_p95": float(np.quantile(abs(gap), .95)),
            "finite_fraction_within_half_log_unit": float((abs(gap) < .5).mean()),
            "gpu_empty_rows_mean": float(empty_counts[:, i].sum(-1).mean()),
            "estimates": args.estimates, "repeats": args.repeats,
        })
    np.savez_compressed(output / "trial_scores.npz", direct=direct_logs, gpu=gpu_logs,
                        empty_gpu=empty_counts, include=include, vectors=vectors,
                        conditions=trials.condition_index.numpy(), choice=trials.choice.numpy(),
                        switch=trials.is_switch.numpy(), rt=trials.response_time.numpy())
    metadata = {"source": source, "exact_anchor_vector": list(base), "exact_anchor_direct": exact,
                "common_csi_seconds": vectors[0, 3], "csi_rounding_seconds": vectors[0, 3]-base[3],
                "config": asdict(config), "gpu_seed_seconds": elapsed,
                "total_seconds": time.perf_counter()-started,
                "strict_window_checked": args.check_strict}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    pd.DataFrame(records).to_csv(output / "summary.csv", index=False)
    print(f"{label}: complete in {metadata['total_seconds']:.1f}s; anchor gap {records[0]['gap_total']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects", default="1,4,7,42,71,81", help="Comma-separated subjects, or all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--fits", type=Path, default=FIT_ROOT)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--expanded-subject-one", action="store_true")
    parser.add_argument("--expanded-fit", type=Path, default=EXPANDED_FIT)
    parser.add_argument("--estimates", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--spatial-points", type=int, default=65)
    parser.add_argument("--check-strict", action="store_true")
    parser.add_argument("--synthetic", action="store_true", help="Also generate and audit two original-GPU datasets")
    parser.add_argument("--refine-from", type=Path,
                        help="Recheck anchors and largest finite local gaps from an existing sweep")
    parser.add_argument("--replay-from", type=Path,
                        help="Rescore every archived input/candidate exactly, bypassing candidate generation")
    parser.add_argument("--direct-convergence-from", type=Path,
                        help="Refine direct solver grids at saved sweep anchors; no new GPU scoring")
    args = parser.parse_args()
    if args.estimates < 1 or args.repeats < 1:
        parser.error("estimates and repeats must be positive")
    if sum(x is not None for x in (args.refine_from, args.replay_from, args.direct_convergence_from)) > 1:
        parser.error("choose only one of --refine-from, --replay-from, or --direct-convergence-from")
    torch.set_num_threads(8)
    args.output.mkdir(parents=True, exist_ok=True)
    source_files = [Path(__file__), HERE / "data fitting/expectation_model_study2_study3.py",
                    HERE / "direct_likelihood/likelihood.py", HERE / "direct_likelihood/model.py"]
    manifest = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                "torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "cpu_threads": torch.get_num_threads(),
                "histogram": {"bins": 100, "smoothing_sigma": .5, "pseudocount": .1,
                              "dt": .001, "max_steps": 12000, "execution": "window"}}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        manifest_path = args.output / f"manifest-{time.time_ns()}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.direct_convergence_from:
        direct_convergence(args.direct_convergence_from, args.output)
        return
    if args.refine_from or args.replay_from:
        source = args.refine_from or args.replay_from
        for table in sorted(source.glob("*/summary.csv")):
            original = pd.read_csv(table)
            eligible = original[(original.direct_zero_rows == 0)
                                & original.kind.str.startswith("local:")]
            selected_ids = [0]
            if len(eligible):
                selected_ids += [int(eligible.gap_total.abs().idxmax()),
                                 int(eligible.gpu_log_density_mean.idxmax())]
            joint = original[(original.direct_zero_rows == 0)
                             & original.kind.str.startswith("joint_support_constrained:")]
            if len(joint):
                selected_ids.append(int(joint.gap_total.abs().idxmax()))
            selected_ids = list(dict.fromkeys(selected_ids))
            saved = pd.read_csv(table.parent / "parameters.csv")
            if args.replay_from:
                selected_ids = list(range(len(saved)))
            vectors = saved.loc[selected_ids, list(parameter_names())].to_numpy()
            kinds = [saved.loc[i, "candidate_kind"] if args.replay_from else
                     f"refine:{i}:{saved.loc[i, 'candidate_kind']}" for i in selected_ids]
            frame = pd.read_csv(table.parent / "data.csv")
            subject = int(frame.subject_nr.iloc[0])
            run_dataset(args, table.parent / "data.csv", subject, table.parent.name,
                        vectors[0], False, str(table.parent), selected=(vectors, kinds))
        frames = [pd.read_csv(p) for p in sorted(args.output.glob("*/summary.csv"))]
        pd.concat(frames, ignore_index=True).to_csv(args.output / "summary.csv", index=False)
        return
    subjects = range(1, 98) if args.subjects == "all" else [int(s) for s in args.subjects.split(",")]
    for subject in subjects:
        path = args.expanded_fit if subject == 1 and args.expanded_subject_one else args.fits / f"subject-{subject}/fit.json"
        payload = json.loads(path.read_text())
        run_dataset(args, args.data, subject, f"subject-{subject}", payload["parameter_vector"], args.deep, str(path))
    if args.synthetic:
        truths = {
            "synthetic_moderate": [10., 10., 10., .05, .12, .12, .12, -.05, -.05, -.05, .2, .2, .2],
            "synthetic_contrasting": [25., 80., 18., .12, .16, .08, .14, -.08, -.04, -.07, .22, .3, .2],
        }
        for number, (label, truth) in enumerate(truths.items()):
            data = args.output / (label + ".csv")
            generation_seed = args.seed + 10000 + number
            if not data.exists():
                gpu = GPUProblem(args.data, 1, [truth])
                generated = gpu.plan.run(gpu.stimulus, gpu.rows, num_estimates=1,
                                         seed=generation_seed, strict_truncation=True,
                                         triton_launch_options=LAUNCH)
                values = generated.values[0, 0, :, 0]
                frame = pd.read_csv(args.data)
                frame = frame[(frame.subject_nr == 1) & frame.sequence.isin(CONDITIONS)].reset_index(drop=True)
                if values.shape != (len(frame), 2):
                    raise ValueError(f"Unexpected synthetic output shape: {values.shape}")
                frame["decision"] = values[:, 0].astype(int)
                frame["response_time"] = values[:, 1]
                # Retain the original design and mask, replace only observed outputs.
                frame.to_csv(data, index=False)
                del gpu
                gc.collect()
            run_dataset(args, data, 1, label, truth, args.deep,
                        {"generator": "original composition / Triton full sequential simulation",
                         "seed": generation_seed, "truth": truth, "design_subject": 1,
                         "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest()})
    frames = [pd.read_csv(p) for p in sorted(args.output.glob("*/summary.csv"))]
    pd.concat(frames, ignore_index=True).to_csv(args.output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
