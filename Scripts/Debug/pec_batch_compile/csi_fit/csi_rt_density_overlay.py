#!/usr/bin/env python3
"""Overlay continuous and GPU-sampled CSI likelihoods over response time."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from csi_likelihood_surface_comparison import (
    DEFAULT_DATA,
    DEFAULT_PARAMETERS,
    _build_gpu_problem,
    _fit_values,
    _rescale_legacy_time_step,
)
from direct_likelihood import (
    CONDITIONS,
    CSITrialData,
    ContinuousCSILikelihood,
    ContinuousCSIParameters,
    SolverConfig,
)
from direct_likelihood.native import native_kernels_available


def _representative_trials(trials):
    """Choose the included observation nearest each condition's median RT."""

    result = []
    response_time = trials.response_time.detach().cpu().numpy()
    condition = trials.condition_index.detach().cpu().numpy()
    include = trials.include.detach().cpu().numpy()
    for condition_index, name in enumerate(CONDITIONS):
        indices = np.flatnonzero(include & (condition == condition_index))
        if not len(indices):
            raise ValueError(f"Subject has no included trials for {name}.")
        median = np.median(response_time[indices])
        selected = indices[np.argmin(np.abs(response_time[indices] - median))]
        result.append(int(selected))
    return result


def _sample_gpu_outcomes(
    parameter_frame,
    *,
    data,
    subject,
    estimates,
    seed,
    max_steps,
    time_step,
    trial_indices,
):
    """Materialize one parameter lane for a diagnostic RT histogram."""

    pec, inputs = _build_gpu_problem(
        data,
        subject,
        estimates,
        100,
        0.0,
        0.0,
        11,
        seed,
        max_steps,
        time_step,
    )
    fit_function = pec.controller.function
    gpu_parameter_frame = _rescale_legacy_time_step(parameter_frame, time_step)
    values = _fit_values(gpu_parameter_frame, fit_function.fit_param_names)[0]
    fit_function.batched_seed = seed
    # Cache the node-keyed inputs and compile the plan through the public path.
    pec.log_likelihood(*values, inputs=inputs)
    plan = fit_function._compile_batched_plan()
    _, debug = plan.deterministic_history_log_likelihood(
        fit_function._batched_stimulus_inputs(),
        [fit_function._batched_parameter_set(values)],
        num_estimates=estimates,
        data=np.asarray(pec._data_numpy, dtype=float),
        categorical_dims=pec.data_categorical_dims,
        outcome_indices=fit_function._batched_outcome_indices(plan),
        bins=100,
        include_mask=pec.likelihood_include_mask,
        seed=seed,
        common_random_numbers=True,
        strict_truncation=True,
        triton_launch_options=fit_function.batched_triton_launch_options,
        return_debug=True,
    )
    outcomes = debug["values"][0, trial_indices].detach().cpu().numpy()
    return outcomes, float(debug["truncation_fraction"])


def _continuous_density_curve(likelihood, parameters, trials, trial_index, rt):
    """Evaluate the current trial's joint choice/RT density on an RT grid."""

    density = []
    started = time.perf_counter()
    with torch.no_grad():
        for value in rt:
            response_time = trials.response_time.clone()
            response_time[trial_index] = float(value)
            varied_trials = replace(trials, response_time=response_time)
            result = likelihood.score(parameters, varied_trials)
            density.append(
                float(result.probability[trial_index]) / trials.rt_resolution
            )
    return np.asarray(density), time.perf_counter() - started


def _gpu_histogram(
    trial,
    observed_choice,
    observed_rt,
    *,
    display_bin_width,
    display_anchor,
):
    selected = trial[np.isclose(trial[:, 0], observed_choice), 1]
    if not len(selected):
        raise RuntimeError(
            f"No GPU simulations made observed choice {observed_choice:g} "
            "on the selected trial."
        )
    low, high = np.quantile(selected, [0.001, 0.999])
    low = min(float(low), observed_rt) - 0.04
    high = max(float(high), observed_rt) + 0.04
    # Use the physical decision-onset time as a common phase reference. This
    # puts simulations with different dt values in identical display bins and
    # avoids splitting a coarse endpoint lattice at fp32 bin boundaries.
    first_center = display_anchor + np.floor(
        (low - display_anchor) / display_bin_width
    ) * display_bin_width
    last_center = display_anchor + np.ceil(
        (high - display_anchor) / display_bin_width
    ) * display_bin_width
    edges = np.arange(
        first_center - 0.5 * display_bin_width,
        last_center + display_bin_width,
        display_bin_width,
    )
    counts, _ = np.histogram(selected, bins=edges)
    # Divide by every estimate, not only estimates making this choice: this is
    # the joint density p(choice, RT), matching the direct likelihood.
    density = counts / (len(trial) * display_bin_width)
    return edges, density, len(selected) / len(trial)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--parameters", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--estimates", type=int, default=100_000)
    parser.add_argument("--gpu-seed", type=int, default=21)
    parser.add_argument(
        "--gpu-time-steps",
        nargs="+",
        type=float,
        default=[0.01],
        help="Synchronized LCA/DDM time steps to sample in seconds.",
    )
    parser.add_argument("--max-time", type=float, default=12.0)
    parser.add_argument("--display-bin-width", type=float, default=0.01)
    parser.add_argument("--curve-points", type=int, default=181)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parameter_frame = pd.read_csv(args.parameters)
    if len(parameter_frame) != 1:
        raise ValueError("--parameters must contain exactly one row.")
    parameters = ContinuousCSIParameters.from_legacy_row(
        parameter_frame.iloc[0], dtype=torch.float64, device="cpu"
    )
    trials = CSITrialData.from_csv(
        args.data, args.subject, dtype=torch.float64, device="cpu"
    )
    trial_indices = _representative_trials(trials)

    gpu_runs = []
    for time_step in args.gpu_time_steps:
        max_steps = int(np.ceil(args.max_time / time_step))
        print(
            f"Sampling {args.estimates:,} GPU outcomes for subject {args.subject}, "
            f"seed {args.gpu_seed}, dt={time_step:g}, max_steps={max_steps}..."
        )
        outcomes, truncation_fraction = _sample_gpu_outcomes(
            parameter_frame,
            data=args.data,
            subject=args.subject,
            estimates=args.estimates,
            seed=args.gpu_seed,
            max_steps=max_steps,
            time_step=time_step,
            trial_indices=trial_indices,
        )
        if truncation_fraction:
            raise RuntimeError(
                f"GPU dt={time_step:g} samples have truncation fraction "
                f"{truncation_fraction:.6g}."
            )
        gpu_runs.append((time_step, outcomes))

    native = native_kernels_available()
    likelihood = ContinuousCSILikelihood(
        SolverConfig(
            ddm_time_step=0.001,
            ddm_spatial_points=65,
            lca_max_step=0.01,
            native_lca_scan=native,
            native_ddm_forward=native,
        )
    )
    choices = trials.choice.detach().cpu().numpy()
    response_times = trials.response_time.detach().cpu().numpy()
    rows = []
    histogram_rows = []
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 11.5), constrained_layout=True)
    gpu_colors = ("#E68613", "#3A9256", "#8C5AAE", "#B64E4E")
    for panel_index, (axis, condition, trial_index) in enumerate(
        zip(axes, CONDITIONS, trial_indices)
    ):
        observed_choice = float(choices[trial_index])
        observed_rt = float(response_times[trial_index])
        condition_index = int(trials.condition_index[trial_index])
        decision_onset = float(parameters.non_decision_time[condition_index])
        if bool(trials.is_switch[trial_index]):
            decision_onset += float(parameters.csi_duration)
        histograms = []
        for time_step, outcomes in gpu_runs:
            edges, gpu_density, gpu_choice_probability = _gpu_histogram(
                outcomes[panel_index],
                observed_choice,
                observed_rt,
                display_bin_width=args.display_bin_width,
                display_anchor=decision_onset,
            )
            histograms.append(
                (time_step, edges, gpu_density, gpu_choice_probability)
            )
        plot_low = min(edges[0] for _, edges, _, _ in histograms)
        plot_high = max(edges[-1] for _, edges, _, _ in histograms)
        rt = np.linspace(plot_low, plot_high, args.curve_points)
        print(
            f"Evaluating direct curve for {condition}, trial {trial_index}, "
            f"choice {observed_choice:g}..."
        )
        direct_density, elapsed = _continuous_density_curve(
            likelihood, parameters, trials, trial_index, rt
        )

        for gpu_index, (
            time_step,
            edges,
            gpu_density,
            gpu_choice_probability,
        ) in enumerate(histograms):
            milliseconds = 1000.0 * time_step
            axis.stairs(
                gpu_density,
                edges,
                fill=gpu_index == 0,
                color=gpu_colors[gpu_index % len(gpu_colors)],
                alpha=0.28 if gpu_index == 0 else 0.95,
                linewidth=1.4,
                label=(
                    f"GPU endpoint samples ({args.estimates:,}; "
                    f"dt={milliseconds:g} ms)"
                ),
            )
            histogram_rows.extend(
                {
                    "condition": condition,
                    "trial_index": trial_index,
                    "observed_choice": observed_choice,
                    "observed_rt": observed_rt,
                    "gpu_time_step": time_step,
                    "display_bin_width": args.display_bin_width,
                    "rt_bin_low": low,
                    "rt_bin_high": high,
                    "gpu_density": density,
                    "gpu_choice_probability": gpu_choice_probability,
                }
                for low, high, density in zip(
                    edges[:-1], edges[1:], gpu_density
                )
            )
        axis.plot(
            rt,
            direct_density,
            color="#214F86",
            linewidth=2.2,
            label="Continuous direct likelihood (1 ms RT interval)",
        )
        axis.axvline(
            observed_rt,
            color="#222222",
            linestyle="--",
            linewidth=1.5,
            label=f"Observed RT = {observed_rt:.3f} s",
        )
        axis.set_title(
            f"{condition}: trial {trial_index}, observed choice {int(observed_choice)}"
        )
        axis.set_xlim(plot_low, plot_high)
        axis.set_ylabel("Joint density  p(choice, RT)")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=9)
        rows.extend(
            {
                "condition": condition,
                "trial_index": trial_index,
                "observed_choice": observed_choice,
                "observed_rt": observed_rt,
                "rt": value,
                "direct_density": density,
                "direct_curve_seconds": elapsed,
            }
            for value, density in zip(rt, direct_density)
        )
    axes[-1].set_xlabel("Response time (seconds)")
    fig.suptitle(
        "CSI response-time likelihood: continuous direct vs GPU endpoint sampling\n"
        f"Subject {args.subject}; fixed physical parameters; GPU seed {args.gpu_seed}; "
        f"{1000 * args.display_bin_width:g} ms display bins",
        fontsize=14,
    )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)
    curve_output = output.with_suffix(".curves.csv")
    histogram_output = output.with_suffix(".gpu_histograms.csv")
    pd.DataFrame(rows).to_csv(curve_output, index=False)
    pd.DataFrame(histogram_rows).to_csv(histogram_output, index=False)
    print(output)
    print(curve_output)
    print(histogram_output)


if __name__ == "__main__":
    main()
