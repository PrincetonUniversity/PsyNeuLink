"""Synthetic-data generation for direct-likelihood parameter recovery."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from .likelihood import ContinuousCSILikelihood
from .model import CSITrialData, ContinuousCSIParameters


@dataclass(frozen=True)
class RecoverySimulationResult:
    """A reproducible synthetic CSI sequence and generation diagnostics."""

    trials: CSITrialData
    maximum_decision_time: float
    mean_decision_time: float
    bridge_correction: bool


def simulate_sequential_trials(
    likelihood: ContinuousCSILikelihood,
    parameters: ContinuousCSIParameters,
    template: CSITrialData,
    *,
    seed: int = 1,
    simulation_time_step: float = 0.0005,
    maximum_decision_time: float = 3.0,
    bridge_correction: bool = True,
) -> RecoverySimulationResult:
    """Generate choices and RTs while retaining a template's ordered inputs.

    The deterministic LCA follows the configured equations and integrator used
    by the likelihood. A seeded Euler--Maruyama path supplies the DDM noise and stops
    at the first moving-boundary crossing. By default a Brownian-bridge test
    also detects paths that cross and return between Euler endpoints. The finer
    simulation step and independent path sampler keep the generator distinct
    from, but convergent with, the likelihood PDE mesh.
    """
    if simulation_time_step <= 0.0 or maximum_decision_time <= 0.0:
        raise ValueError("Simulation time settings must be positive.")
    vector = parameters.vector()
    if vector.device.type != "cpu" or template.response_time.device.type != "cpu":
        raise ValueError("Recovery simulation currently requires CPU tensors.")
    if vector.dtype != template.response_time.dtype:
        raise ValueError("Parameters and template must use the same dtype.")

    dtype = vector.dtype
    state = torch.zeros(2, dtype=dtype)
    zero_task = torch.zeros_like(state)
    choices = []
    response_times = []
    decision_times = []
    maximum_steps = int(math.ceil(maximum_decision_time / simulation_time_step))
    iti_steps = int(
        math.ceil(likelihood.config.iti_duration / likelihood.config.lca_max_step)
    )

    with torch.no_grad():
        for trial_index in range(len(template)):
            if likelihood.config.reset_lca_each_trial:
                state = torch.zeros_like(state)
            condition = int(template.condition_index[trial_index])
            gain = parameters.gain[condition]
            task = template.task[trial_index]
            csi_duration = parameters.csi_duration * template.is_switch[trial_index]
            csi_steps = int(
                math.ceil(float(csi_duration) / likelihood.config.lca_max_step)
            )

            state = likelihood._integrate_lca(
                state,
                zero_task,
                gain,
                likelihood.config.iti_duration,
                steps=iti_steps,
            )
            state = likelihood._integrate_lca(
                state, task, gain, csi_duration, steps=csi_steps
            )
            drift_arguments = (
                state.unsqueeze(0),
                task.unsqueeze(0),
                gain.unsqueeze(0),
                template.stimulus[trial_index].unsqueeze(0),
                template.correct_response[trial_index].unsqueeze(0),
            )
            if (
                likelihood.config.native_lca_scan
                and likelihood.config.lca_integration_method == "rk4"
            ):
                from .native import native_lca_drift_path

                def drift_path(steps):
                    return native_lca_drift_path(
                        *drift_arguments,
                        steps=steps,
                        step_size=simulation_time_step,
                        leak=likelihood.config.lca_leak,
                        competition=likelihood.config.lca_competition,
                    )
            else:
                def drift_path(steps):
                    path_function = (
                        likelihood.lca.drift_path_euler
                        if likelihood.config.lca_integration_method == "euler"
                        else likelihood.lca.drift_path
                    )
                    return path_function(
                        *drift_arguments,
                        steps=steps,
                        step_size=simulation_time_step,
                    )

            drift, _ = drift_path(maximum_steps)
            drift_values = drift[0].detach().cpu().numpy()
            # Independent per-trial streams make all sampled prefixes invariant
            # to the arbitrary maximum-decision-time safety horizon.
            noise_seed, crossing_seed, choice_seed = np.random.SeedSequence(
                [seed, trial_index]
            ).spawn(3)
            noise = np.random.default_rng(noise_seed).standard_normal(
                maximum_steps
            )
            bridge_crossing_draw = np.random.default_rng(
                crossing_seed
            ).random(maximum_steps)
            bridge_choice_draw = np.random.default_rng(choice_seed).random(
                maximum_steps
            )
            evidence = 0.0
            crossing_step = None
            choice = None
            threshold = float(parameters.threshold[condition])
            collapse = float(parameters.collapse_rate[condition])
            noise_scale = likelihood.config.ddm_noise * math.sqrt(
                simulation_time_step
            )
            variance = likelihood.config.ddm_noise ** 2 * simulation_time_step
            for step_index in range(maximum_steps):
                start_time = step_index * simulation_time_step
                end_time = (step_index + 1) * simulation_time_step
                boundary_start = threshold + collapse * start_time
                boundary_end = threshold + collapse * end_time
                if boundary_end <= likelihood.config.boundary_floor:
                    raise RuntimeError(
                        "Synthetic boundary collapsed before a decision on "
                        f"trial {trial_index}."
                    )
                previous_evidence = evidence
                evidence = previous_evidence + (
                    drift_values[step_index] * simulation_time_step
                    + noise_scale * noise[step_index]
                )
                if evidence >= boundary_end:
                    crossing_step = step_index + 1
                    choice = 1.0
                    break
                if evidence <= -boundary_end:
                    crossing_step = step_index + 1
                    choice = 0.0
                    break
                if bridge_correction:
                    upper_probability = math.exp(
                        -2.0
                        * max(0.0, boundary_start - previous_evidence)
                        * max(0.0, boundary_end - evidence)
                        / variance
                    )
                    lower_probability = math.exp(
                        -2.0
                        * max(0.0, boundary_start + previous_evidence)
                        * max(0.0, boundary_end + evidence)
                        / variance
                    )
                    total_probability = upper_probability + lower_probability
                    if bridge_crossing_draw[step_index] < min(
                        1.0, total_probability
                    ):
                        crossing_step = step_index + 1
                        choice = float(
                            bridge_choice_draw[step_index] * total_probability
                            >= lower_probability
                        )
                        break
            if crossing_step is None or choice is None:
                raise RuntimeError(
                    "Synthetic DDM did not cross a boundary within "
                    f"{maximum_decision_time:g} seconds on trial {trial_index}."
                )

            decision_time = crossing_step * simulation_time_step
            _, final_state = drift_path(crossing_step)
            state = final_state[0]
            response_time = (
                float(parameters.non_decision_time[condition])
                + float(csi_duration)
                + decision_time
            )
            choices.append(choice)
            response_times.append(response_time)
            decision_times.append(decision_time)

    return RecoverySimulationResult(
        trials=CSITrialData(
            task=template.task.clone(),
            stimulus=template.stimulus.clone(),
            correct_response=template.correct_response.clone(),
            choice=torch.tensor(choices, dtype=dtype),
            response_time=torch.tensor(response_times, dtype=dtype),
            condition_index=template.condition_index.clone(),
            is_switch=template.is_switch.clone(),
            include=template.include.clone(),
            row_id=template.row_id.clone(),
            subject_nr=template.subject_nr,
            rt_resolution=template.rt_resolution,
        ),
        maximum_decision_time=max(decision_times),
        mean_decision_time=float(np.mean(decision_times)),
        bridge_correction=bridge_correction,
    )
