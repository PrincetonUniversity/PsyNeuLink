"""Sequential continuous-time likelihood for the complete CSI model."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time
from typing import Any

import torch

from .model import CSITrialData, ContinuousCSIParameters, ContinuousLCA
from .solver import DDMBatchResult, MovingBoundaryDDMSolver


@dataclass(frozen=True)
class SolverConfig:
    """Numerical and fixed-model settings for the prototype."""

    ddm_time_step: float = 0.001
    ddm_spatial_points: int = 65
    ddm_process: str = "continuous"
    endpoint_evidence_domain: float = 0.35
    lca_max_step: float = 0.01
    lca_integration_method: str = "rk4"
    reset_lca_each_trial: bool = False
    rt_bin_count: int | None = None
    iti_duration: float = 1.0
    ddm_noise: float = 0.1
    lca_leak: float = 12.0
    lca_competition: float = 3.0
    boundary_floor: float = 1.0e-5
    rannacher_steps: int = 2
    ddm_checkpoint_steps: int = 32
    checkpoint_lca: bool = True
    ddm_bucket_size: int = 256
    compile_ddm_steps: bool = False
    custom_ddm_adjoint: bool = False
    custom_lca_adjoint: bool = False
    compile_lca_adjoint: bool = False
    native_lca_scan: bool = False
    native_ddm_forward: bool = False

    def __post_init__(self) -> None:
        if self.ddm_time_step <= 0.0:
            raise ValueError("ddm_time_step must be positive.")
        if self.ddm_process not in {"continuous", "endpoint"}:
            raise ValueError("ddm_process must be 'continuous' or 'endpoint'.")
        if self.endpoint_evidence_domain <= 0.0:
            raise ValueError("endpoint_evidence_domain must be positive.")
        if self.lca_max_step <= 0.0:
            raise ValueError("lca_max_step must be positive.")
        if self.lca_integration_method not in {"rk4", "euler"}:
            raise ValueError(
                "lca_integration_method must be either 'rk4' or 'euler'."
            )
        if self.iti_duration < 0.0:
            raise ValueError("iti_duration cannot be negative.")
        if self.ddm_bucket_size < 0:
            raise ValueError("ddm_bucket_size cannot be negative.")
        if self.rt_bin_count is not None and self.rt_bin_count < 1:
            raise ValueError("rt_bin_count must be positive when specified.")


@dataclass(frozen=True)
class LikelihoodResult:
    """Likelihood values, latent trajectory, and numerical diagnostics."""

    log_likelihood: torch.Tensor
    per_trial_log_likelihood: torch.Tensor
    probability: torch.Tensor
    decision_time: torch.Tensor
    lca_state_after_trial: torch.Tensor
    included_row_indices: torch.Tensor
    ddm: DDMBatchResult | None
    diagnostics: dict[str, Any]
    timings: dict[str, float]


@dataclass(frozen=True)
class _TrialTiming:
    """Parameter-dependent trial timing and fixed numerical topology."""

    gain: torch.Tensor
    csi_duration: torch.Tensor
    decision_time: torch.Tensor
    interval_low: torch.Tensor
    interval_high: torch.Tensor
    state_duration: torch.Tensor
    lca_steps: torch.Tensor
    ddm_steps: tuple[int, ...]
    iti_steps: int
    rt_bin_width: torch.Tensor | float


@dataclass(frozen=True)
class _LCAHistory:
    """Persistent LCA states required by the batched DDM likelihood."""

    decision_onset: torch.Tensor
    state_after_trial: torch.Tensor
    included_indices: torch.Tensor
    included_ddm_steps: tuple[int, ...]


@dataclass(frozen=True)
class _DDMEvaluation:
    """Bucketed DDM result plus execution diagnostics."""

    result: DDMBatchResult | None
    drift_seconds: float
    ddm_seconds: float
    bucket_count: int
    drift_cells_required: int
    drift_cells_computed: int


class ContinuousCSILikelihood:
    """Evaluate the sequential choice/RT likelihood of one participant.

    The LCA is a deterministic continuous ODE whose state persists between
    trials.  On each trial it first decays for the one-second ITI, processes
    the task during a switch-only CSI, and then co-evolves with the DDM until
    the observed decision time.  The DDM likelihood is computed from selected
    absorbing-boundary flux over the observation's RT-resolution interval.
    """

    def __init__(self, config: SolverConfig | None = None):
        self.config = config or SolverConfig()
        self.lca = ContinuousLCA(
            leak=self.config.lca_leak,
            competition=self.config.lca_competition,
            custom_adjoint=self.config.custom_lca_adjoint,
            compile_adjoint=self.config.compile_lca_adjoint,
        )
        if self.config.ddm_process == "endpoint":
            from .discrete_solver import EndpointCrossingDDMSolver

            self.ddm = EndpointCrossingDDMSolver(
                time_step=self.config.ddm_time_step,
                spatial_points=self.config.ddm_spatial_points,
                noise=self.config.ddm_noise,
                boundary_floor=self.config.boundary_floor,
                evidence_domain=self.config.endpoint_evidence_domain,
            )
        else:
            self.ddm = MovingBoundaryDDMSolver(
                time_step=self.config.ddm_time_step,
                spatial_points=self.config.ddm_spatial_points,
                noise=self.config.ddm_noise,
                boundary_floor=self.config.boundary_floor,
                rannacher_steps=self.config.rannacher_steps,
                checkpoint_steps=self.config.ddm_checkpoint_steps,
                compile_steps=self.config.compile_ddm_steps,
                custom_adjoint=self.config.custom_ddm_adjoint,
                native_forward=self.config.native_ddm_forward,
            )

    def _integrate_lca(self, state, task, gain, duration, *, steps=None):
        if steps == 0:
            return state
        duration_tensor = torch.as_tensor(
            duration, dtype=state.dtype, device=state.device
        )
        needs_checkpoint = (
            self.config.checkpoint_lca
            and not self.config.custom_lca_adjoint
            and torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (state, task, gain, duration_tensor)
            )
        )
        integration = (
            self.lca.integrate_euler
            if self.config.lca_integration_method == "euler"
            else self.lca.integrate
        )
        if not needs_checkpoint:
            return integration(
                state,
                task,
                gain,
                duration_tensor,
                max_step=self.config.lca_max_step,
                steps=steps,
            )
        from torch.utils.checkpoint import checkpoint

        def integrate_block(block_state, block_task, block_gain, block_duration):
            return integration(
                block_state,
                block_task,
                block_gain,
                block_duration,
                max_step=self.config.lca_max_step,
                steps=steps,
            )

        return checkpoint(
            integrate_block,
            state,
            task,
            gain,
            duration_tensor,
            use_reentrant=True,
        )

    def _lca_drift_path(
        self,
        state,
        task,
        gain,
        stimulus,
        correct_response,
        *,
        steps,
    ):
        if (
            self.config.native_lca_scan
            and self.config.lca_integration_method == "rk4"
        ):
            from .native import native_lca_drift_path

            return native_lca_drift_path(
                state,
                task,
                gain,
                stimulus,
                correct_response,
                steps=steps,
                step_size=self.config.ddm_time_step,
                leak=self.config.lca_leak,
                competition=self.config.lca_competition,
            )
        needs_checkpoint = (
            self.config.checkpoint_lca
            and not self.config.custom_lca_adjoint
            and torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (state, task, gain, stimulus, correct_response)
            )
        )

        def integrate_path(
            block_state,
            block_task,
            block_gain,
            block_stimulus,
            block_correct_response,
        ):
            drift_path = (
                self.lca.drift_path_euler
                if self.config.lca_integration_method == "euler"
                else self.lca.drift_path
            )
            return drift_path(
                block_state,
                block_task,
                block_gain,
                block_stimulus,
                block_correct_response,
                steps=steps,
                step_size=self.config.ddm_time_step,
            )

        arguments = (state, task, gain, stimulus, correct_response)
        if not needs_checkpoint:
            return integrate_path(*arguments)
        from torch.utils.checkpoint import checkpoint

        return checkpoint(integrate_path, *arguments, use_reentrant=True)

    def _prepare_trial_timing(
        self,
        parameters: ContinuousCSIParameters,
        trials: CSITrialData,
    ) -> _TrialTiming:
        """Translate observed RTs into LCA durations and DDM intervals."""
        condition = trials.condition_index
        gain = parameters.gain[condition]
        non_decision_time = parameters.non_decision_time[condition]
        csi_duration = parameters.csi_duration * trials.is_switch
        decision_time = (
            trials.response_time - non_decision_time - csi_duration
        )

        if self.config.rt_bin_count is None:
            interval_low = decision_time - 0.5 * trials.rt_resolution
            interval_high = decision_time + 0.5 * trials.rt_resolution
            rt_bin_width: torch.Tensor | float = trials.rt_resolution
        else:
            rt_min = torch.amin(trials.response_time)
            rt_max = torch.amax(trials.response_time)
            margin = torch.where(
                rt_max > rt_min,
                (rt_max - rt_min) * 0.02,
                torch.ones_like(rt_min),
            )
            edge_low = rt_min - margin
            edge_high = rt_max + margin
            edge_high = edge_high + (edge_high - edge_low) * 1.0e-6
            rt_edges = torch.linspace(
                edge_low,
                edge_high,
                self.config.rt_bin_count + 1,
                dtype=decision_time.dtype,
                device=decision_time.device,
            )
            rt_bins = torch.bucketize(trials.response_time, rt_edges[1:-1])
            interval_low = (
                rt_edges[rt_bins] - non_decision_time - csi_duration
            )
            interval_high = (
                rt_edges[rt_bins + 1] - non_decision_time - csi_duration
            )
            rt_bin_width = rt_edges[1] - rt_edges[0]

        state_duration = torch.clamp(decision_time, min=0.0)
        duration_mesh = torch.stack((csi_duration, state_duration), dim=1)
        lca_steps = (
            torch.ceil(
                torch.clamp(duration_mesh, min=0.0)
                / self.config.lca_max_step
            )
            .to(torch.long)
            .detach()
            .cpu()
        )
        ddm_steps = tuple(
            torch.clamp(
                torch.ceil(
                    torch.clamp(interval_high, min=0.0)
                    / self.config.ddm_time_step
                ).to(torch.long),
                min=1,
            )
            .detach()
            .cpu()
            .tolist()
        )
        iti_steps = int(
            math.ceil(self.config.iti_duration / self.config.lca_max_step)
        )
        return _TrialTiming(
            gain=gain,
            csi_duration=csi_duration,
            decision_time=decision_time,
            interval_low=interval_low,
            interval_high=interval_high,
            state_duration=state_duration,
            lca_steps=lca_steps,
            ddm_steps=ddm_steps,
            iti_steps=iti_steps,
            rt_bin_width=rt_bin_width,
        )

    def _scan_lca_history(
        self,
        trials: CSITrialData,
        timing: _TrialTiming,
    ) -> _LCAHistory:
        """Propagate the deterministic LCA through the observed trial history."""
        device = trials.response_time.device
        included_indices = torch.nonzero(
            trials.include, as_tuple=False
        ).reshape(-1)
        included_rows = included_indices.detach().cpu().tolist()
        included_ddm_steps = tuple(
            timing.ddm_steps[index] for index in included_rows
        )

        if (
            self.config.native_lca_scan
            and self.config.lca_integration_method == "rk4"
            and not self.config.reset_lca_each_trial
        ):
            from .native import native_lca_subject_scan

            decision_onset, state_after_trial = native_lca_subject_scan(
                trials.task,
                timing.gain,
                timing.csi_duration,
                timing.state_duration,
                timing.lca_steps[:, 0].contiguous(),
                timing.lca_steps[:, 1].contiguous(),
                iti_duration=self.config.iti_duration,
                iti_steps=timing.iti_steps,
                leak=self.config.lca_leak,
                competition=self.config.lca_competition,
            )
            return _LCAHistory(
                decision_onset=decision_onset[included_indices],
                state_after_trial=state_after_trial,
                included_indices=included_indices,
                included_ddm_steps=included_ddm_steps,
            )

        state = torch.zeros(2, dtype=trials.task.dtype, device=device)
        zero_task = torch.zeros_like(state)
        states_after_trial = []
        included_onsets = []
        included = trials.include.detach().cpu().tolist()
        for trial_index in range(len(trials)):
            if self.config.reset_lca_each_trial:
                state = torch.zeros_like(state)
            gain = timing.gain[trial_index]
            task = trials.task[trial_index]
            state = self._integrate_lca(
                state,
                zero_task,
                gain,
                self.config.iti_duration,
                steps=timing.iti_steps,
            )
            state = self._integrate_lca(
                state,
                task,
                gain,
                timing.csi_duration[trial_index],
                steps=int(timing.lca_steps[trial_index, 0]),
            )
            if included[trial_index]:
                included_onsets.append(state)

            # Masked observations still contribute their observed-duration LCA
            # evolution to later trials. Negative durations are clamped to zero.
            state = self._integrate_lca(
                state,
                task,
                gain,
                timing.state_duration[trial_index],
                steps=int(timing.lca_steps[trial_index, 1]),
            )
            states_after_trial.append(state)

        decision_onset = (
            torch.stack(included_onsets)
            if included_onsets
            else torch.empty(
                (0, 2), dtype=trials.task.dtype, device=device
            )
        )
        return _LCAHistory(
            decision_onset=decision_onset,
            state_after_trial=torch.stack(states_after_trial),
            included_indices=included_indices,
            included_ddm_steps=included_ddm_steps,
        )

    def _solve_ddm_buckets(
        self,
        parameters: ContinuousCSIParameters,
        trials: CSITrialData,
        timing: _TrialTiming,
        history: _LCAHistory,
        timestamp: Callable[[], float],
    ) -> _DDMEvaluation:
        """Evaluate included trials in duration-sorted, padded DDM batches."""
        included_count = int(history.included_indices.numel())
        drift_cells_required = sum(history.included_ddm_steps)
        if included_count == 0:
            return _DDMEvaluation(None, 0.0, 0.0, 0, 0, 0)

        bucket_size = self.config.ddm_bucket_size or included_count
        order = sorted(
            range(included_count),
            key=history.included_ddm_steps.__getitem__,
        )
        dtype, device = trials.response_time.dtype, trials.response_time.device
        combined_probability = torch.zeros(
            included_count, dtype=dtype, device=device
        )
        combined_upper = torch.zeros_like(combined_probability)
        combined_lower = torch.zeros_like(combined_probability)
        combined_survival = torch.zeros_like(combined_probability)
        combined_mass_error = torch.zeros_like(combined_probability)
        combined_minimum_density = torch.zeros_like(combined_probability)
        combined_invalid = torch.zeros(
            included_count, dtype=torch.bool, device=device
        )
        drift_seconds = 0.0
        ddm_seconds = 0.0
        bucket_count = 0
        drift_cells_computed = 0

        for bucket_start in range(0, included_count, bucket_size):
            positions = order[bucket_start:bucket_start + bucket_size]
            relative_index = torch.as_tensor(
                positions, dtype=torch.long, device=device
            )
            trial_index = history.included_indices[relative_index]
            bucket_steps = max(
                history.included_ddm_steps[position]
                for position in positions
            )
            drift_cells_computed += bucket_steps * len(positions)
            bucket_count += 1

            drift_start = timestamp()
            drift, _ = self._lca_drift_path(
                history.decision_onset[relative_index],
                trials.task[trial_index],
                timing.gain[trial_index],
                trials.stimulus[trial_index],
                trials.correct_response[trial_index],
                steps=bucket_steps,
            )
            drift_seconds += timestamp() - drift_start

            ddm_start = timestamp()
            condition = trials.condition_index[trial_index]
            bucket_result = self.ddm.solve_observation_batch(
                drift=drift,
                threshold=parameters.threshold[condition],
                collapse_rate=parameters.collapse_rate[condition],
                interval_low=timing.interval_low[trial_index],
                interval_high=timing.interval_high[trial_index],
                choice=trials.choice[trial_index],
            )
            ddm_seconds += timestamp() - ddm_start
            combined_probability = combined_probability.index_copy(
                0, relative_index, bucket_result.probability
            )
            combined_upper = combined_upper.index_copy(
                0, relative_index, bucket_result.upper_probability
            )
            combined_lower = combined_lower.index_copy(
                0, relative_index, bucket_result.lower_probability
            )
            combined_survival = combined_survival.index_copy(
                0, relative_index, bucket_result.survival_probability
            )
            combined_mass_error = combined_mass_error.index_copy(
                0, relative_index, bucket_result.mass_error
            )
            combined_minimum_density = combined_minimum_density.index_copy(
                0, relative_index, bucket_result.minimum_density
            )
            combined_invalid = combined_invalid.index_copy(
                0, relative_index, bucket_result.invalid_boundary
            )

        return _DDMEvaluation(
            result=DDMBatchResult(
                probability=combined_probability,
                upper_probability=combined_upper,
                lower_probability=combined_lower,
                survival_probability=combined_survival,
                mass_error=combined_mass_error,
                minimum_density=combined_minimum_density,
                invalid_boundary=combined_invalid,
            ),
            drift_seconds=drift_seconds,
            ddm_seconds=ddm_seconds,
            bucket_count=bucket_count,
            drift_cells_required=drift_cells_required,
            drift_cells_computed=drift_cells_computed,
        )

    def score_vector(
        self,
        vector: torch.Tensor,
        trials: CSITrialData,
        *,
        collect_timings: bool = False,
    ) -> LikelihoodResult:
        return self.score(
            ContinuousCSIParameters.from_vector(vector),
            trials,
            collect_timings=collect_timings,
        )

    def score(
        self,
        parameters: ContinuousCSIParameters,
        trials: CSITrialData,
        *,
        collect_timings: bool = False,
    ) -> LikelihoodResult:
        vector = parameters.vector()
        dtype, device = vector.dtype, vector.device
        if trials.response_time.dtype != dtype or trials.response_time.device != device:
            raise ValueError("Parameters and trial data must use the same dtype and device.")

        def timestamp() -> float:
            if collect_timings and device.type == "cuda":
                torch.cuda.synchronize(device)
            return time.perf_counter()

        total_start = timestamp()
        timing = self._prepare_trial_timing(parameters, trials)
        serial_start = timestamp()
        history = self._scan_lca_history(trials, timing)
        serial_seconds = timestamp() - serial_start

        ddm = self._solve_ddm_buckets(
            parameters, trials, timing, history, timestamp
        )
        probability = torch.ones(len(trials), dtype=dtype, device=device)
        per_trial_log_likelihood = torch.zeros_like(probability)
        if ddm.result is not None:
            ddm_result = ddm.result
            probability = probability.index_copy(
                0, history.included_indices, ddm_result.probability
            )
            included_log = torch.log(ddm_result.probability)
            per_trial_log_likelihood = per_trial_log_likelihood.index_copy(
                0, history.included_indices, included_log
            )
            log_likelihood = torch.sum(included_log)
            maximum_mass_error = torch.amax(ddm_result.mass_error)
            minimum_density = torch.amin(ddm_result.minimum_density)
            invalid_rows = history.included_indices[
                ddm_result.invalid_boundary
            ]
            zero_probability_rows = history.included_indices[
                ddm_result.probability <= 0.0
            ]
        else:
            ddm_result = None
            log_likelihood = torch.zeros((), dtype=dtype, device=device)
            maximum_mass_error = torch.zeros((), dtype=dtype, device=device)
            minimum_density = torch.zeros((), dtype=dtype, device=device)
            invalid_rows = torch.empty(0, dtype=torch.long, device=device)
            zero_probability_rows = torch.empty(
                0, dtype=torch.long, device=device
            )

        masked_negative_decision_rows = torch.nonzero(
            (~trials.include) & (timing.decision_time < 0.0), as_tuple=False
        ).reshape(-1)
        total_seconds = timestamp() - total_start

        return LikelihoodResult(
            log_likelihood=log_likelihood,
            per_trial_log_likelihood=per_trial_log_likelihood,
            probability=probability,
            decision_time=timing.decision_time,
            lca_state_after_trial=history.state_after_trial,
            included_row_indices=history.included_indices,
            ddm=ddm_result,
            diagnostics={
                "lca_integration_method": self.config.lca_integration_method,
                "ddm_process": self.config.ddm_process,
                "reset_lca_each_trial": self.config.reset_lca_each_trial,
                "rt_bin_count": self.config.rt_bin_count,
                "rt_bin_width": timing.rt_bin_width,
                "maximum_mass_error": maximum_mass_error,
                "minimum_density": minimum_density,
                "invalid_included_rows": invalid_rows,
                "zero_probability_included_rows": zero_probability_rows,
                "masked_negative_decision_rows": masked_negative_decision_rows,
                "ddm_bucket_count": ddm.bucket_count,
                "drift_cells_required": ddm.drift_cells_required,
                "drift_cells_computed": ddm.drift_cells_computed,
            },
            timings={
                "serial_lca_seconds": serial_seconds,
                "batched_drift_seconds": ddm.drift_seconds,
                "ddm_seconds": ddm.ddm_seconds,
                "total_seconds": total_seconds,
            } if collect_timings else {},
        )
