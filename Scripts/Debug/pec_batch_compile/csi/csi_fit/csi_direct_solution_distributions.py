#!/usr/bin/env python3
"""Compare conditional CSI RT distributions at a saved direct-fit solution.

The production GPU curve uses the generated observation sampler. A separate
Brownian-bridge diagnostic samples prescribed continuous-model drift paths;
it does not change the production model or perform another fit.

The original LLVM model defines source semantics. Its discrete LCA history
and endpoint crossings need not match the continuous direct approximation;
see csi_gpu_llvm_simulation_audit.py for independent model-output validation.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time
from psyneulink.core.batched.likelihood import histogram_likelihood

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import triton
import triton.language as tl

from csi_likelihood_surface_comparison import (
    DEFAULT_DATA, _build_gpu_problem, _fit_values, _rescale_legacy_time_step,
)
from csi_rt_density_overlay import _representative_trials
from direct_likelihood import (
    CONDITIONS, CSITrialData, ContinuousCSIParameters, ContinuousCSILikelihood,
    SolverConfig,
)
from psyneulink.core.batched import ObservationField, ObservationSpec


@triton.jit
def _bridge_kernel(DRIFT, THRESHOLD, COLLAPSE, HORIZON, OUT_TIME, OUT_CHOICE,
                   N: tl.constexpr, STEPS: tl.constexpr, DT: tl.constexpr,
                   SEED: tl.constexpr, BRIDGE: tl.constexpr, BLOCK: tl.constexpr):
    trial = tl.program_id(0)
    lane = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = lane < N
    active = valid
    evidence = tl.full((BLOCK,), 0., tl.float32)
    crossing = tl.full((BLOCK,), float("inf"), tl.float32)
    choice = tl.full((BLOCK,), -1, tl.int32)
    a = tl.load(THRESHOLD + trial)
    rate = tl.load(COLLAPSE + trial)
    horizon = tl.load(HORIZON + trial)
    seed = SEED + trial * 1009
    step = 0
    while (step < horizon) & (tl.sum(active.to(tl.int32), 0) > 0):
        offset = (step * N + lane).to(tl.uint32)
        drift = tl.load(DRIFT + trial * STEPS + step)
        new = evidence + drift * DT + (0.1 * tl.sqrt(DT)) * tl.randn(seed, offset)
        old_boundary = a + rate * (step * DT)
        boundary = a + rate * ((step + 1) * DT)
        upper = new >= boundary
        lower = new <= -boundary
        if BRIDGE:
            p_upper = tl.exp(-2. * (old_boundary - evidence) * (boundary - new) / (0.01 * DT))
            p_lower = tl.exp(-2. * (old_boundary + evidence) * (boundary + new) / (0.01 * DT))
            total = p_upper + p_lower
            crossed_between = (~upper & ~lower) & (tl.rand(seed + 100003, offset) < tl.minimum(1., total))
            bridge_upper = tl.rand(seed + 200003, offset) * total < p_upper
            upper = upper | (crossed_between & bridge_upper)
            lower = lower | (crossed_between & ~bridge_upper)
        crossed = active & (upper | lower)
        crossing = tl.where(crossed, (step + 1) * DT, crossing)
        choice = tl.where(crossed, upper.to(tl.int32), choice)
        active = active & ~crossed
        evidence = new
        step += 1
    tl.store(OUT_TIME + trial * N + lane, crossing, valid)
    tl.store(OUT_CHOICE + trial * N + lane, choice, valid)


def bridge_samples(drift, threshold, collapse, horizon_steps, dt, estimates, seed):
    tensors = [torch.as_tensor(x, device="cuda", dtype=torch.float32).contiguous()
               for x in (drift, threshold, collapse)]
    horizon = torch.as_tensor(horizon_steps, device="cuda", dtype=torch.int32)
    times = torch.empty((len(threshold), estimates), device="cuda", dtype=torch.float32)
    choices = torch.empty_like(times, dtype=torch.int32)
    _bridge_kernel[(len(threshold), triton.cdiv(estimates, 128))](
        *tensors, horizon, times, choices, estimates, drift.shape[1], dt, seed, True, 128,
        num_warps=4,
    )
    return times.cpu().numpy(), choices.cpu().numpy()


def direct_flux(likelihood, parameters, trials, indices):
    """Recover both boundary fluxes from the existing native PDE density history."""
    with torch.no_grad():
        timing = likelihood._prepare_trial_timing(parameters, trials)
        history = likelihood._scan_lca_history(trials, timing)
        positions = [history.included_indices.tolist().index(i) for i in indices]
        onset = history.decision_onset[positions]
        condition = trials.condition_index[indices]
        threshold = parameters.threshold[condition]
        collapse = parameters.collapse_rate[condition]
        dt = likelihood.config.ddm_time_step
        # Stop just before the mathematical collapse; report residual mass.
        safe_horizon = torch.where(collapse < 0, (threshold - 2e-5) / (-collapse),
                                   torch.full_like(threshold, 12.)).clamp(max=12.)
        end_steps = torch.floor(safe_horizon / dt).to(torch.long)
        end_times = end_steps.to(dtype=threshold.dtype) * dt
        steps = int(end_steps.max())
        drift, _ = likelihood._lca_drift_path(
            onset, trials.task[indices], timing.gain[indices],
            trials.stimulus[indices], trials.correct_response[indices], steps=steps,
        )
        result, density = likelihood.ddm._solve_observation_batch_impl(
            drift=drift, threshold=threshold, collapse_rate=collapse,
            interval_low=torch.zeros_like(threshold), interval_high=end_times,
            choice=torch.ones_like(threshold), store_density_history=True,
        )
        assert not result.invalid_boundary.any()
        t = torch.arange(steps, dtype=torch.float64) * dt
        boundary = threshold[None, :] + (t[:, None] + 0.5 * dt) * collapse[None, :]
        active = t[:, None] < end_times[None, :]
        safe = torch.where(active, boundary, torch.ones_like(boundary))
        _, _, _, terms = likelihood.ddm._operator(
            drift.T.reshape(-1), safe.reshape(-1),
            collapse[None, :].expand(steps, -1).reshape(-1),
        )
        lower, upper = likelihood.ddm._boundary_flux(
            (0.5 * (density[:-1] + density[1:])).reshape(-1, density.shape[-1]), terms,
        )
        flux = torch.stack((lower.reshape(steps, -1), upper.reshape(steps, -1)), -1)
        flux = flux.clamp(min=0) * active[..., None]
        masses = flux.sum(0) * dt
        np.testing.assert_allclose(masses[:, 0], result.lower_probability, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(masses[:, 1], result.upper_probability, atol=1e-10, rtol=1e-10)
        if float(result.survival_probability.max()) > 1e-5:
            raise RuntimeError("Non-negligible unplotted survival mass at the PDE horizon")
    return flux.numpy(), {
        "maximum_mass_error": float(result.mass_error.max()),
        "maximum_survival": float(result.survival_probability.max()),
        "flux_total_probabilities": masses.tolist(),
    }


def interpolate_cdf(flux, rt_edges, shift, dt):
    cumulative = np.r_[0., np.cumsum(flux) * dt]
    return np.interp(rt_edges - shift, np.arange(len(cumulative)) * dt,
                     cumulative, left=0., right=cumulative[-1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, required=True, help="Direct fit JSON, in physical units")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--estimates", type=int, default=100000)
    parser.add_argument("--chunk-estimates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--score-repeats", type=int, default=1,
                        help="Score with this many consecutive seeds to measure Monte Carlo variation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.estimates <= 0 or args.chunk_estimates <= 0 or args.score_repeats <= 0:
        parser.error("Estimate and score-repeat counts must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    payload = json.loads(args.parameters.read_text())
    (args.output / "direct_fit.json").write_text(json.dumps(payload, indent=2) + "\n")
    vector = np.asarray(payload["parameter_vector"], dtype=float)
    params = ContinuousCSIParameters.from_vector(torch.tensor(vector, dtype=torch.float64))
    trials = CSITrialData.from_csv(args.data, args.subject, dtype=torch.float64, device="cpu")
    frame = pd.read_csv(args.data)
    frame = frame[(frame.subject_nr == args.subject) & frame.sequence.isin(CONDITIONS)].reset_index(drop=True)
    include = trials.include.numpy()
    primary = _representative_trials(trials)
    config = SolverConfig(ddm_time_step=.001, ddm_spatial_points=65, lca_max_step=.01,
                          native_lca_scan=True, native_ddm_forward=True)
    likelihood = ContinuousCSILikelihood(config)
    with torch.no_grad():
        direct = likelihood.score(params, trials)
    np.testing.assert_allclose(float(direct.log_likelihood), payload["log_likelihood"], atol=1e-7, rtol=0)
    print("Direct fitted score reproduced", float(direct.log_likelihood), flush=True)

    dt = .001
    legacy = pd.DataFrame([params.as_legacy_dict()])
    gpu_parameters = _rescale_legacy_time_step(legacy, dt)
    gpu_parameters["Cue Stimulus Interval.slope"] = np.round(gpu_parameters["Cue Stimulus Interval.slope"])
    gpu_parameters.to_csv(args.output / "gpu_parameters.csv", index=False)
    pec, inputs = _build_gpu_problem(args.data, args.subject, 64, 100, .5, .1, 11, args.seed, 12000, dt)
    ff = pec.controller.function
    ff.batched_observations = ObservationSpec((
        ObservationField(pec.outcome_variables[0], "counting"),
        ObservationField(pec.outcome_variables[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp"),
    ))
    ff.batched_triton_launch_options = {"block_size":32,"num_warps":1}
    values = _fit_values(gpu_parameters, ff.fit_param_names)[0]
    pec.log_likelihood(*values, inputs=inputs)
    plan = ff._compile_batched_plan()
    compiled = ff._compile_batched_histogram_plan(plan)
    stimulus = ff._batched_stimulus_inputs()
    rows = [ff._batched_parameter_set(values)]
    observations = np.asarray(pec._data_numpy, dtype=float)
    print("Scoring all included trials with the generated GPU likelihood", flush=True)
    scored = compiled.score(stimulus, observations, rows, num_estimates=args.estimates,
                            seed=args.seed, include_mask=include, execution="window",
                            max_buffer_bytes=1024**3, triton_launch_options=ff.batched_triton_launch_options)
    seed_scores = [{"seed": args.seed, "log_density": float(scored.log_likelihood[0])}]
    for seed in range(args.seed + 1, args.seed + args.score_repeats):
        repeat = compiled.score(stimulus, observations, rows, num_estimates=args.estimates,
                                seed=seed, include_mask=include, execution="window",
                                max_buffer_bytes=1024**3, triton_launch_options=ff.batched_triton_launch_options)
        seed_scores.append({"seed": seed, "log_density": float(repeat.log_likelihood[0])})
    direct_density = direct.probability.numpy() / trials.rt_resolution
    direct_logs = np.log(direct_density)
    gpu_logs = scored.log_factors[0]
    worst = []
    for condition in range(3):
        selected = np.flatnonzero(include & (trials.condition_index.numpy() == condition))
        worst.append(int(selected[np.argmax(np.abs(gpu_logs[selected] - direct_logs[selected]))]))
    indices = list(dict.fromkeys(primary + worst))
    print("Representative trials:", primary, "largest discrepancies:", worst, flush=True)
    inspected_paths = compiled.observation_plan.sampler.path_plan.generate(stimulus, observations, rows, max_buffer_bytes=1024**3)
    threshold_field = next(field.column_start for field in inspected_paths.fields if field.kind == "held_modulation")
    first_thresholds = inspected_paths.values[0, :, 0, threshold_field].copy()
    np.savez_compressed(args.output/"selected_boundary_paths.npz", values=inspected_paths.values[0,indices,:20],
                        start_states=inspected_paths.history.start_states[0,indices],indices=indices)
    (args.output/"boundary_fields.txt").write_text(repr(inspected_paths.fields))
    print("First boundary values:", [(i, inspected_paths.values[0,i,:2].tolist()) for i in indices], flush=True)
    del inspected_paths
    print("Sampling the retained handwritten GPU reference from the earlier comparison", flush=True)
    reference_score, reference_debug = plan.deterministic_history_log_likelihood(
        stimulus, rows, implementation="handwritten", num_estimates=args.estimates,
        data=observations, categorical_dims=pec.data_categorical_dims,
        outcome_indices=ff._batched_outcome_indices(plan), bins=100,
        smoothing_sigma=.5, pseudocount=.1, categorical_cardinalities=[2],
        include_mask=include, seed=args.seed, common_random_numbers=True,
        strict_truncation=True, triton_launch_options=ff.batched_triton_launch_options,
        return_debug=True,
    )
    reference = reference_debug["values"][0, indices].detach().cpu().numpy()
    reference_densities = histogram_likelihood(
        reference_debug["values"], observations, [0], bins=100, smoothing_sigma=.5,
        pseudocount=.1, categorical_cardinalities=[2],
    )[0]
    if reference_debug["truncation_fraction"] != 0:
        raise RuntimeError("Reference GPU samples were truncated")
    del reference_debug
    collected = []
    for begin in range(0, args.estimates, args.chunk_estimates):
        count = min(args.chunk_estimates, args.estimates - begin)
        print(f"Generated full-trajectory samples {begin + count}/{args.estimates}", flush=True)
        samples = compiled.observation_plan.sample(
            stimulus, observations, rows, num_estimates=count,
            seed=args.seed + begin // args.chunk_estimates, strict_truncation=True,
            max_buffer_bytes=1024**3,
        )
        if np.any(samples.truncated):
            raise RuntimeError("Production samples were truncated")
        if begin == 0:
            sampled_density = histogram_likelihood(samples.values, observations, [0], bins=100,
                                                  smoothing_sigma=.5, pseudocount=.1,
                                                  categorical_cardinalities=[2])
            check = compiled.score(stimulus, observations, rows, num_estimates=count,
                                   seed=args.seed, include_mask=include, execution="strict",
                                   max_buffer_bytes=1024**3, triton_launch_options=ff.batched_triton_launch_options)
            pd.DataFrame({"sampled":sampled_density[0], "fused":check.densities[0]}).to_csv(args.output/"sampler_check.csv",index=False)
            np.testing.assert_allclose(sampled_density[0], check.densities[0], atol=1e-5, rtol=1e-5)
            np.savez_compressed(args.output/"sample_check.npz", values=samples.values[0,indices],
                                counts=samples.event_counts[0,indices],indices=indices)
            (args.output/"sampler_source.py").write_text(compiled.observation_plan.source())
        collected.append(np.asarray(samples.values)[0, indices].copy())
        del samples
    gpu = np.concatenate(collected, axis=1)
    print("Recovering continuous PDE flux curves", flush=True)
    flux, mass_checks = direct_flux(likelihood, params, trials, indices)
    for position, index in enumerate(indices):
        condition = int(trials.condition_index[index])
        shift = float(params.non_decision_time[condition] + params.csi_duration * trials.is_switch[index])
        interval = float(trials.response_time[index]) + np.array([-.0005, .0005])
        cdf = interpolate_cdf(flux[:,position,int(trials.choice[index])], interval, shift, dt)
        np.testing.assert_allclose(np.diff(cdf)[0], float(direct.probability[index]), atol=1e-12, rtol=1e-6)

    # Independent, finer bridge sampler using the direct model's own LCA history.
    bridge_dt = .0005
    bridge_model = ContinuousCSILikelihood(replace(config, ddm_time_step=bridge_dt))
    with torch.no_grad():
        timing = bridge_model._prepare_trial_timing(params, trials)
        history = bridge_model._scan_lca_history(trials, timing)
        relative = [history.included_indices.tolist().index(i) for i in indices]
        condition = trials.condition_index[indices]
        a = params.threshold[condition].numpy()
        rates = params.collapse_rate[condition].numpy()
        ends = np.floor(np.minimum(np.divide(a-2e-5, -rates,
                        out=np.full_like(a, 12.), where=rates<0),12.)/bridge_dt).astype(int)
        drift, _ = bridge_model._lca_drift_path(
            history.decision_onset[relative], trials.task[indices], timing.gain[indices],
            trials.stimulus[indices], trials.correct_response[indices], steps=int(ends.max()),
        )
    print("Sampling independent Brownian-bridge diagnostic on GPU", flush=True)
    bridge_time, bridge_choice = bridge_samples(drift.numpy(), a, rates, ends, bridge_dt, args.estimates, args.seed+1000)
    if np.any(bridge_choice < 0):
        raise RuntimeError("Bridge diagnostic has uncompleted samples")
    shifts = params.non_decision_time[condition].numpy() + float(params.csi_duration) * trials.is_switch[indices].numpy()
    bridge_rt = bridge_time + shifts[:,None]
    curves, metrics = [], []
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})

    def plot_trials(selected, name, title, show_generated=True):
        fig, axes = plt.subplots(3, 2, figsize=(12, 10), constrained_layout=True)
        for row, index in enumerate(selected):
            position = indices.index(index)
            cond = CONDITIONS[int(trials.condition_index[index])]
            limit = max(np.quantile(gpu[position,:,1], .9995), np.quantile(reference[position,:,1], .9995), np.quantile(bridge_rt[position], .9995)) + .03
            bin_width = .01
            edges = shifts[position] + np.arange(-.5, np.ceil((limit-shifts[position])/bin_width)+1) * bin_width
            for choice in (1, 0):
                ax = axes[row, 1-choice]
                pde_cdf = interpolate_cdf(flux[:,position,choice], edges, shifts[position], dt)
                pde_density = np.diff(pde_cdf) / bin_width
                gpu_counts = np.histogram(gpu[position, gpu[position,:,0]==choice, 1], bins=edges)[0]
                bridge_counts = np.histogram(bridge_rt[position,bridge_choice[position]==choice], bins=edges)[0]
                reference_counts = np.histogram(reference[position,reference[position,:,0]==choice,1], bins=edges)[0]
                gpu_density = gpu_counts / args.estimates / bin_width
                bridge_density = bridge_counts / args.estimates / bin_width
                reference_density = reference_counts / args.estimates / bin_width
                centers = .5*(edges[:-1]+edges[1:])
                if show_generated:
                    ax.stairs(gpu_density, edges, color="#d47a1f", linewidth=1.4, label="Current generated GPU: endpoint, 1 ms")
                ax.plot(centers, reference_density, color="#a459a5", linewidth=1.3, label="Earlier handwritten GPU: endpoint, 1 ms")
                ax.plot(centers, pde_density, color="#225b95", linewidth=2, label="Direct PDE: continuous crossing")
                ax.plot(centers, bridge_density, color="#22816a", linestyle="--", linewidth=1.4, label="Bridge diagnostic: direct drift, 0.5 ms")
                if int(trials.choice[index]) == choice:
                    ax.axvline(float(trials.response_time[index]), color="#444444", linestyle=":", linewidth=1.3, label="Observed RT")
                transition = "switch" if trials.is_switch[index] else "repeat"
                ax.set_title(f"{cond} · trial {index} ({transition}) · choice {choice}", fontsize=11)
                ax.set_ylabel("Joint choice/RT density (s⁻¹)")
                ax.set_xlim(max(0, edges[0]-.01), edges[-1])
                ax.set_ylim(bottom=0)
                ax.grid(alpha=.15)
                if show_generated and first_thresholds[index] < 0:
                    ax.text(.98,.95,f"First GPU threshold: {first_thresholds[index]:.4f}\nGPU samples all stop on step 1",ha="right",va="top",transform=ax.transAxes,fontsize=8,
                            bbox={"facecolor":"white","alpha":.85,"edgecolor":"none"})
                if row == 2:
                    ax.set_xlabel("Response time (seconds)")
                for j in range(len(centers)):
                    curves.append(dict(panel=name,trial=index,condition=cond,choice=choice,rt=centers[j],
                                       direct=pde_density[j],gpu=gpu_density[j],reference=reference_density[j],bridge=bridge_density[j]))
                full_gpu_cdf=np.asarray([np.mean((gpu[position,:,0]==choice)&(gpu[position,:,1] <= edge)) for edge in edges])
                full_bridge_cdf=np.asarray([np.mean((bridge_choice[position]==choice)&(bridge_rt[position] <= edge)) for edge in edges])
                full_reference_cdf=np.asarray([np.mean((reference[position,:,0]==choice)&(reference[position,:,1] <= edge)) for edge in edges])
                metrics.append(dict(panel=name,trial=index,condition=cond,choice=choice,
                                    gpu_max_cdf_error=float(np.max(np.abs(full_gpu_cdf-pde_cdf))),
                                    reference_max_cdf_error=float(np.max(np.abs(full_reference_cdf-pde_cdf))),
                                    bridge_max_cdf_error=float(np.max(np.abs(full_bridge_cdf-pde_cdf)))))
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False, fontsize=10)
        fig.suptitle(title + f"\nSubject {args.subject} · direct-fit parameters · {args.estimates:,} samples/trial · 10 ms display bins", fontsize=14)
        fig.savefig(args.output / f"{name}.png", dpi=180)
        fig.savefig(args.output / f"{name}.pdf")
        plt.close(fig)

    plot_trials(primary, "representative_distributions", "RT likelihoods at the direct solution: median-RT trials")
    plot_trials(primary, "reference_distributions", "Earlier GPU implementation and bridge diagnostic versus direct", show_generated=False)
    plot_trials(worst, "largest_discrepancies", "RT likelihoods at the direct solution: largest observed-score gaps")
    pd.DataFrame(curves).to_csv(args.output / "density_curves.csv", index=False)
    trial_table = frame[["sequence","decision","response_time","likelihood_include_mask"]].copy()
    trial_table["direct_log_density"] = direct_logs
    trial_table["gpu_log_density"] = gpu_logs
    trial_table["gpu_minus_direct"] = gpu_logs-direct_logs
    trial_table["reference_log_density"] = np.log(reference_densities)
    trial_table["generated_first_threshold"] = first_thresholds
    trial_table.to_csv(args.output / "trial_scores.csv", index_label="trial")
    fig, ax = plt.subplots(figsize=(7,6), constrained_layout=True)
    for c, color in zip(CONDITIONS, ["#225b95", "#d47a1f", "#22816a"]):
        mask = include & (frame.sequence.to_numpy()==c)
        ax.scatter(direct_logs[mask], gpu_logs[mask], s=22, alpha=.6, label=c, color=color)
    limits=[min(direct_logs[include].min(),gpu_logs[include].min())-.5,
            max(direct_logs[include].max(),gpu_logs[include].max())+.5]
    ax.plot(limits,limits,"--",color="#777777",linewidth=1)
    ax.set(xlim=limits,ylim=limits,xlabel="Direct log density (1 ms recording interval)",
           ylabel="GPU log density (100 bins, σ=0.5, pseudocount=0.1)",title=f"Same direct-fit parameters · {include.sum()} scored trials")
    ax.legend(frameon=False)
    fig.savefig(args.output/"trial_likelihoods.png",dpi=180)
    fig.savefig(args.output/"trial_likelihoods.pdf")
    plt.close(fig)
    manifest = dict(parameters=str(args.parameters.resolve()),data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),
                    git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                    diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model_source_sha256=hashlib.sha256((Path(__file__).parent / "data fitting" / "expectation_model_study2_study3.py").read_bytes()).hexdigest(),
                    parameter_vector=vector.tolist(),gpu_csi_seconds=float(gpu_parameters["Cue Stimulus Interval.slope"].iloc[0])*dt,
                    exact_csi_seconds=float(params.csi_duration),representative_trials=primary,largest_gap_trials=worst,
                    estimates=args.estimates,seed=args.seed,chunk_estimates=args.chunk_estimates,
                    direct_log_interval_probability=float(direct.log_likelihood),
                    direct_log_density=float(direct_logs[include].sum()),gpu_log_density=float(scored.log_likelihood[0]),
                    gpu_seed_scores=seed_scores,
                    reference_gpu_log_density=float(reference_score),
                    negative_initial_threshold_trials=np.flatnonzero(include & (first_thresholds < 0)).tolist(),
                    mass_checks=mass_checks,cdf_checks=metrics,elapsed_seconds=time.perf_counter()-started,
                    gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,
                    caveats=["GPU CSI alone is rounded to the nearest 1 ms; all other direct-fit parameters are unchanged.",
                             "All observed trial history, including masked rows, is retained; curves are conditional single-trial distributions, not marginal posterior predictions.",
                             "Bridge sampler is an independent diagnostic using direct-model drift and history, not a modification of the production GPU sampler.",
                             "Plots use unsmoothed 10 ms display bins and both choices retain their joint probability mass. Production scores use the actual smoothed fitting estimator.",
                             "The largest_discrepancies plot deliberately selects extreme score discrepancies; representative trials are selected independently by condition median observed RT.",
                             "The source model now orders the threshold update after LCA readiness and guards output gates with a fresh DDM call. Historical results before this scheduling fix can contain stale negative first-step thresholds.",
                             "The earlier handwritten GPU is retained only as a historical diagnostic. The current source composition executed with LLVM defines compiler ground truth."])
    (args.output / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == "__main__":
    main()
