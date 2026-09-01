"""Data and continuous LCA definitions for the CSI direct likelihood."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch


CONDITIONS = ("NoInstruction", "RealRare", "RealFrequent")
CONDITION_INDEX = {name: index for index, name in enumerate(CONDITIONS)}


@torch.jit.script
def _scripted_lca_rhs(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    leak: float,
    competition: float,
) -> torch.Tensor:
    while gain.dim() < state.dim():
        gain = gain.unsqueeze(-1)
    activity = torch.sigmoid(gain * state)
    recurrent = torch.stack(
        (-competition * activity[..., 1], -competition * activity[..., 0]),
        dim=-1,
    )
    return -leak * state + task + recurrent


@torch.jit.script
def _scripted_lca_integrate(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    duration: torch.Tensor,
    steps: int,
    leak: float,
    competition: float,
) -> torch.Tensor:
    if steps == 0:
        return state
    step = duration / float(steps)
    for _ in range(steps):
        k1 = _scripted_lca_rhs(state, task, gain, leak, competition)
        k2 = _scripted_lca_rhs(
            state + 0.5 * step * k1, task, gain, leak, competition
        )
        k3 = _scripted_lca_rhs(
            state + 0.5 * step * k2, task, gain, leak, competition
        )
        k4 = _scripted_lca_rhs(
            state + step * k3, task, gain, leak, competition
        )
        state = state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return state


@torch.jit.script
def _scripted_lca_integrate_euler(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    duration: torch.Tensor,
    steps: int,
    leak: float,
    competition: float,
) -> torch.Tensor:
    """Euler mirror used to isolate the legacy PNL integration semantics."""
    if steps == 0:
        return state
    step = duration / float(steps)
    for _ in range(steps):
        state = state + step * _scripted_lca_rhs(
            state, task, gain, leak, competition
        )
    return state


@torch.jit.script
def _scripted_lca_integrate_history(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    duration: torch.Tensor,
    steps: int,
    leak: float,
    competition: float,
) -> torch.Tensor:
    states = torch.jit.annotate(list[torch.Tensor], [])
    states.append(state)
    if steps == 0:
        return torch.stack(states)
    step = duration / float(steps)
    for _ in range(steps):
        k1 = _scripted_lca_rhs(state, task, gain, leak, competition)
        k2 = _scripted_lca_rhs(
            state + 0.5 * step * k1, task, gain, leak, competition
        )
        k3 = _scripted_lca_rhs(
            state + 0.5 * step * k2, task, gain, leak, competition
        )
        k4 = _scripted_lca_rhs(
            state + step * k3, task, gain, leak, competition
        )
        state = state + (step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        states.append(state)
    return torch.stack(states)


def _functional_lca_rhs(state, task, gain, leak, competition):
    while gain.ndim < state.ndim:
        gain = gain.unsqueeze(-1)
    activity = torch.sigmoid(gain * state)
    recurrent = torch.stack(
        (-competition * activity[..., 1], -competition * activity[..., 0]),
        dim=-1,
    )
    return -leak * state + task + recurrent


def _functional_lca_rk4_step(state, task, gain, step, leak, competition):
    k1 = _functional_lca_rhs(state, task, gain, leak, competition)
    k2 = _functional_lca_rhs(
        state + 0.5 * step * k1, task, gain, leak, competition
    )
    k3 = _functional_lca_rhs(
        state + 0.5 * step * k2, task, gain, leak, competition
    )
    k4 = _functional_lca_rhs(
        state + step * k3, task, gain, leak, competition
    )
    return state + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _lca_rk4_vjp(
    state, task, gain, step, leak, competition, gradient_state
):
    def step_function(local_state, local_task, local_gain, local_step):
        return _functional_lca_rk4_step(
            local_state,
            local_task,
            local_gain,
            local_step,
            leak,
            competition,
        )

    _, pullback = torch.func.vjp(step_function, state, task, gain, step)
    return pullback(gradient_state)


def _functional_lca_drift_step(
    state,
    task,
    gain,
    stimulus,
    correct_response,
    step,
    leak,
    competition,
):
    midpoint = _functional_lca_rk4_step(
        state, task, gain, 0.5 * step, leak, competition
    )
    activity_gain = gain
    while activity_gain.ndim < midpoint.ndim:
        activity_gain = activity_gain.unsqueeze(-1)
    activity = torch.sigmoid(activity_gain * midpoint)
    drift = csi_drift_rate(stimulus, activity, correct_response)
    final_state = _functional_lca_rk4_step(
        midpoint, task, gain, 0.5 * step, leak, competition
    )
    return final_state, drift


def _lca_drift_step_vjp(
    state,
    task,
    gain,
    stimulus,
    correct_response,
    step,
    leak,
    competition,
    gradient_state,
    gradient_drift,
):
    def step_function(
        local_state,
        local_task,
        local_gain,
        local_stimulus,
        local_correct_response,
    ):
        return _functional_lca_drift_step(
            local_state,
            local_task,
            local_gain,
            local_stimulus,
            local_correct_response,
            step,
            leak,
            competition,
        )

    _, pullback = torch.func.vjp(
        step_function,
        state,
        task,
        gain,
        stimulus,
        correct_response,
    )
    return pullback((gradient_state, gradient_drift))


_COMPILED_LCA_RK4_VJP = torch.compile(
    _lca_rk4_vjp, fullgraph=True, dynamic=True
)
_COMPILED_LCA_DRIFT_STEP_VJP = torch.compile(
    _lca_drift_step_vjp, fullgraph=True, dynamic=True
)


def _as_tensor(value, *, dtype, device):
    # DataFrame categorical columns can expose read-only NumPy arrays.  The
    # model owns its tensors, so copy here and avoid undefined write behavior.
    return torch.tensor(value, dtype=dtype, device=device)


@dataclass(frozen=True)
class CSITrialData:
    """One participant's ordered CSI observations and model inputs."""

    task: torch.Tensor
    stimulus: torch.Tensor
    correct_response: torch.Tensor
    choice: torch.Tensor
    response_time: torch.Tensor
    condition_index: torch.Tensor
    is_switch: torch.Tensor
    include: torch.Tensor
    row_id: torch.Tensor
    subject_nr: int
    rt_resolution: float = 0.001

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        subject_nr: int,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> CSITrialData:
        frame = pd.read_csv(Path(path).expanduser())
        required = {
            "subject_nr", "sequence", "T1", "T2", "S1", "S2", "S3", "S4",
            "correct_response", "decision", "response_time",
            "likelihood_include_mask",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"CSI data are missing required columns: {missing}")

        frame = frame[
            (frame["subject_nr"] == subject_nr)
            & frame["sequence"].isin(CONDITIONS)
        ].reset_index(drop=True)
        if frame.empty:
            available = sorted(pd.unique(pd.read_csv(path, usecols=["subject_nr"])["subject_nr"]))
            raise ValueError(
                f"subject_nr={subject_nr} is not present in the selected CSI data; "
                f"available subjects begin {available[:10]}"
            )

        unknown = sorted(set(frame["sequence"]).difference(CONDITION_INDEX))
        if unknown:
            raise ValueError(f"Unsupported CSI sequence conditions: {unknown}")

        task = frame[["T1", "T2"]].to_numpy(dtype=float)
        # Match the existing fit driver: trial zero is compared with the final
        # retained task rather than inventing a new block-reset convention.
        previous_task = np.roll(task, 1, axis=0)
        is_switch = np.any(task != previous_task, axis=1).astype(float)
        choice = frame["decision"].to_numpy(dtype=float)
        if not set(np.unique(choice)).issubset({0.0, 1.0}):
            raise ValueError("CSI choices must be coded as 0 (lower) or 1 (upper).")
        response_time = frame["response_time"].to_numpy(dtype=float)
        if not np.all(np.isfinite(response_time)) or np.any(response_time <= 0):
            raise ValueError("CSI response times must be finite and positive.")

        row_id = (
            frame["row_id"].to_numpy(dtype=int)
            if "row_id" in frame
            else np.arange(len(frame), dtype=int)
        )
        return cls(
            task=_as_tensor(task, dtype=dtype, device=device),
            stimulus=_as_tensor(
                frame[["S1", "S2", "S3", "S4"]].to_numpy(dtype=float),
                dtype=dtype,
                device=device,
            ),
            correct_response=_as_tensor(
                frame["correct_response"].to_numpy(dtype=float),
                dtype=dtype,
                device=device,
            ),
            choice=_as_tensor(choice, dtype=dtype, device=device),
            response_time=_as_tensor(response_time, dtype=dtype, device=device),
            condition_index=torch.as_tensor(
                [CONDITION_INDEX[name] for name in frame["sequence"]],
                dtype=torch.long,
                device=device,
            ),
            is_switch=_as_tensor(is_switch, dtype=dtype, device=device),
            include=torch.as_tensor(
                frame["likelihood_include_mask"].to_numpy(
                    dtype=bool, copy=True
                ),
                dtype=torch.bool,
                device=device,
            ),
            row_id=torch.tensor(row_id, dtype=torch.long, device=device),
            subject_nr=int(subject_nr),
        )

    def __len__(self) -> int:
        return int(self.response_time.numel())


@dataclass(frozen=True)
class ContinuousCSIParameters:
    """The 13 fitted parameters of the continuous CSI prototype.

    The condition vectors follow :data:`CONDITIONS`.  ``collapse_rate`` is in
    boundary units per second, unlike the legacy fit CSV's per-10-ms offset.
    """

    gain: torch.Tensor
    csi_duration: torch.Tensor
    threshold: torch.Tensor
    collapse_rate: torch.Tensor
    non_decision_time: torch.Tensor

    @classmethod
    def from_vector(cls, vector: torch.Tensor) -> ContinuousCSIParameters:
        if vector.ndim != 1 or vector.numel() != 13:
            raise ValueError("A continuous CSI parameter vector must have 13 values.")
        return cls(
            gain=vector[0:3],
            csi_duration=vector[3],
            threshold=vector[4:7],
            collapse_rate=vector[7:10],
            non_decision_time=vector[10:13],
        )

    @classmethod
    def defaults(
        cls,
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> ContinuousCSIParameters:
        return cls.from_vector(
            torch.tensor(
                [10.0, 10.0, 10.0, 0.05, 0.12, 0.12, 0.12,
                 0.0, 0.0, 0.0, 0.20, 0.20, 0.20],
                dtype=dtype,
                device=device,
            )
        )

    @classmethod
    def from_legacy_row(
        cls,
        row: Mapping[str, object],
        *,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> ContinuousCSIParameters:
        def value(base: str, condition: str | None = None) -> float:
            wanted = f"{base}[{condition}]" if condition is not None else base
            matches = [
                key
                for key in row.keys()
                if str(key) == wanted or str(key).endswith(wanted)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one legacy fit column matching {wanted!r}; found {matches}"
                )
            return float(row[matches[0]])

        gains = [value("Task Activations [C1, C2].gain", cond) for cond in CONDITIONS]
        csi_seconds = value("Cue Stimulus Interval.slope") * 0.01
        thresholds = [value("Threshold Mechanism.intercept", cond) for cond in CONDITIONS]
        collapse_rates = [
            value("Threshold Mechanism.offset-integrator_function", cond) / 0.01
            for cond in CONDITIONS
        ]
        ndts = [value("DDM.non_decision_time", cond) for cond in CONDITIONS]
        return cls.from_vector(
            torch.tensor(
                [*gains, csi_seconds, *thresholds, *collapse_rates, *ndts],
                dtype=dtype,
                device=device,
            )
        )

    def vector(self) -> torch.Tensor:
        return torch.cat(
            (
                self.gain.reshape(-1),
                self.csi_duration.reshape(1),
                self.threshold.reshape(-1),
                self.collapse_rate.reshape(-1),
                self.non_decision_time.reshape(-1),
            )
        )

    def as_legacy_dict(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for index, condition in enumerate(CONDITIONS):
            result[f"Task Activations [C1, C2].gain[{condition}]"] = float(self.gain[index])
        result["Cue Stimulus Interval.slope"] = float(self.csi_duration / 0.01)
        for index, condition in enumerate(CONDITIONS):
            result[f"Threshold Mechanism.intercept[{condition}]"] = float(self.threshold[index])
            result[
                f"Threshold Mechanism.offset-integrator_function[{condition}]"
            ] = float(self.collapse_rate[index] * 0.01)
            result[f"DDM.non_decision_time[{condition}]"] = float(
                self.non_decision_time[index]
            )
        return result


@dataclass(frozen=True)
class ContinuousLCA:
    leak: float = 12.0
    competition: float = 3.0
    custom_adjoint: bool = False
    compile_adjoint: bool = False

    def rhs(self, state: torch.Tensor, task: torch.Tensor, gain: torch.Tensor) -> torch.Tensor:
        while gain.ndim < state.ndim:
            gain = gain.unsqueeze(-1)
        activity = torch.sigmoid(gain * state)
        recurrent = torch.stack(
            (-self.competition * activity[..., 1], -self.competition * activity[..., 0]),
            dim=-1,
        )
        return -self.leak * state + task + recurrent

    def rk4_step(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        step: torch.Tensor | float,
    ) -> torch.Tensor:
        h = torch.as_tensor(step, dtype=state.dtype, device=state.device)
        k1 = self.rhs(state, task, gain)
        k2 = self.rhs(state + 0.5 * h * k1, task, gain)
        k3 = self.rhs(state + 0.5 * h * k2, task, gain)
        k4 = self.rhs(state + h * k3, task, gain)
        return state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def euler_step(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        step: torch.Tensor | float,
    ) -> torch.Tensor:
        """Advance one explicit-Euler step, matching PNL's LCA integrator."""
        h = torch.as_tensor(step, dtype=state.dtype, device=state.device)
        return state + h * self.rhs(state, task, gain)

    def integrate(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        duration: torch.Tensor | float,
        *,
        max_step: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        duration_tensor = torch.as_tensor(
            duration, dtype=state.dtype, device=state.device
        )
        if steps is None:
            detached_duration = max(0.0, float(duration_tensor.detach().cpu()))
            if detached_duration == 0.0:
                return state
            steps = max(1, int(math.ceil(detached_duration / max_step)))
        elif steps < 0:
            raise ValueError("steps cannot be negative.")
        elif steps == 0:
            return state
        if (
            self.custom_adjoint
            and torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (state, task, gain, duration_tensor)
            )
        ):
            return _DifferentiableLCAIntegration.apply(
                state,
                task,
                gain,
                duration_tensor,
                steps,
                self,
            )
        return _scripted_lca_integrate(
            state,
            task,
            gain,
            duration_tensor,
            steps,
            self.leak,
            self.competition,
        )

    def integrate_euler(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        duration: torch.Tensor | float,
        *,
        max_step: float,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Integrate over an exact duration with explicit Euler substeps.

        At a 10 ms step and a duration divisible by 10 ms this is the original
        PNL update.  For a non-divisible observed duration the final step is
        shortened, preserving the direct likelihood's continuous-time RT
        conditioning rather than introducing an additional clock-rounding
        change.
        """
        duration_tensor = torch.as_tensor(
            duration, dtype=state.dtype, device=state.device
        )
        if steps is None:
            detached_duration = max(0.0, float(duration_tensor.detach().cpu()))
            if detached_duration == 0.0:
                return state
            steps = max(1, int(math.ceil(detached_duration / max_step)))
        elif steps < 0:
            raise ValueError("steps cannot be negative.")
        elif steps == 0:
            return state
        return _scripted_lca_integrate_euler(
            state,
            task,
            gain,
            duration_tensor,
            steps,
            self.leak,
            self.competition,
        )

    def drift_path(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        stimulus: torch.Tensor,
        correct_response: torch.Tensor,
        *,
        steps: int,
        step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return midpoint DDM drifts and the LCA state after ``steps``."""
        if (
            self.custom_adjoint
            and torch.is_grad_enabled()
            and any(
                value.requires_grad
                for value in (state, task, gain, stimulus, correct_response)
            )
        ):
            return _DifferentiableLCADriftPath.apply(
                state,
                task,
                gain,
                stimulus,
                correct_response,
                steps,
                step_size,
                self,
            )
        drifts = []
        half_step = step_size / 2.0
        for _ in range(steps):
            midpoint = self.rk4_step(state, task, gain, half_step)
            # The LCA integrator state is transformed by its Logistic function
            # before being projected to the drift-rate mechanism in PNL.
            activity_gain = gain
            while activity_gain.ndim < midpoint.ndim:
                activity_gain = activity_gain.unsqueeze(-1)
            activity = torch.sigmoid(activity_gain * midpoint)
            drifts.append(csi_drift_rate(stimulus, activity, correct_response))
            state = self.rk4_step(midpoint, task, gain, half_step)
        return torch.stack(drifts, dim=-1), state

    def drift_path_euler(
        self,
        state: torch.Tensor,
        task: torch.Tensor,
        gain: torch.Tensor,
        stimulus: torch.Tensor,
        correct_response: torch.Tensor,
        *,
        steps: int,
        step_size: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return post-Euler-step drifts, mirroring PNL scheduler order.

        The legacy scheduler advances the LCA and then evaluates the logistic
        output used by the drift mechanism.  Therefore this uses the activity
        at the end of each Euler cell, unlike the RK4 path's midpoint value.
        """
        drifts = []
        for _ in range(steps):
            state = self.euler_step(state, task, gain, step_size)
            activity_gain = gain
            while activity_gain.ndim < state.ndim:
                activity_gain = activity_gain.unsqueeze(-1)
            activity = torch.sigmoid(activity_gain * state)
            drifts.append(csi_drift_rate(stimulus, activity, correct_response))
        return torch.stack(drifts, dim=-1), state

    def rk4_vjp(self, *arguments):
        function = (
            _COMPILED_LCA_RK4_VJP
            if self.compile_adjoint
            else _lca_rk4_vjp
        )
        state, task, gain, step, gradient_state = arguments
        return function(
            state,
            task,
            gain,
            step,
            self.leak,
            self.competition,
            gradient_state,
        )

    def drift_step_vjp(self, *arguments):
        function = (
            _COMPILED_LCA_DRIFT_STEP_VJP
            if self.compile_adjoint
            else _lca_drift_step_vjp
        )
        state, task, gain, stimulus, correct_response, step, gradients = (
            arguments
        )
        gradient_state, gradient_drift = gradients
        return function(
            state,
            task,
            gain,
            stimulus,
            correct_response,
            step,
            self.leak,
            self.competition,
            gradient_state,
            gradient_drift,
        )


class _DifferentiableLCAIntegration(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, task, gain, duration, steps, lca):
        history = _scripted_lca_integrate_history(
            state,
            task,
            gain,
            duration,
            steps,
            lca.leak,
            lca.competition,
        )
        ctx.steps = steps
        ctx.lca = lca
        ctx.save_for_backward(history, task, gain, duration)
        return history[-1]

    @staticmethod
    def backward(ctx, gradient_state):
        history, task, gain, duration = ctx.saved_tensors
        step = duration / float(ctx.steps)
        gradient_task = torch.zeros_like(task)
        gradient_gain = torch.zeros_like(gain)
        gradient_duration = torch.zeros_like(duration)
        for step_index in range(ctx.steps - 1, -1, -1):
            (
                gradient_state,
                step_task,
                step_gain,
                gradient_step,
            ) = ctx.lca.rk4_vjp(
                history[step_index],
                task,
                gain,
                step,
                gradient_state,
            )
            gradient_task = gradient_task + step_task
            gradient_gain = gradient_gain + step_gain
            gradient_duration = (
                gradient_duration + gradient_step / float(ctx.steps)
            )
        return (
            gradient_state,
            gradient_task,
            gradient_gain,
            gradient_duration,
            None,
            None,
        )


class _DifferentiableLCADriftPath(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        state,
        task,
        gain,
        stimulus,
        correct_response,
        steps,
        step_size,
        lca,
    ):
        states = [state]
        drifts = []
        step = torch.as_tensor(
            step_size, dtype=state.dtype, device=state.device
        )
        for _ in range(steps):
            state, drift = _functional_lca_drift_step(
                state,
                task,
                gain,
                stimulus,
                correct_response,
                step,
                lca.leak,
                lca.competition,
            )
            states.append(state)
            drifts.append(drift)
        drift_path = torch.stack(drifts, dim=-1)
        history = torch.stack(states)
        ctx.steps = steps
        ctx.lca = lca
        ctx.save_for_backward(
            history,
            task,
            gain,
            stimulus,
            correct_response,
            step,
        )
        return drift_path, state

    @staticmethod
    def backward(ctx, gradient_drift, gradient_final_state):
        (
            history,
            task,
            gain,
            stimulus,
            correct_response,
            step,
        ) = ctx.saved_tensors
        gradient_state = gradient_final_state
        gradient_task = torch.zeros_like(task)
        gradient_gain = torch.zeros_like(gain)
        gradient_stimulus = torch.zeros_like(stimulus)
        gradient_correct_response = torch.zeros_like(correct_response)
        for step_index in range(ctx.steps - 1, -1, -1):
            step_gradients = ctx.lca.drift_step_vjp(
                history[step_index],
                task,
                gain,
                stimulus,
                correct_response,
                step,
                (gradient_state, gradient_drift[..., step_index]),
            )
            (
                gradient_state,
                step_task,
                step_gain,
                step_stimulus,
                step_correct_response,
            ) = step_gradients
            gradient_task = gradient_task + step_task
            gradient_gain = gradient_gain + step_gain
            gradient_stimulus = gradient_stimulus + step_stimulus
            gradient_correct_response = (
                gradient_correct_response + step_correct_response
            )
        return (
            gradient_state,
            gradient_task,
            gradient_gain,
            gradient_stimulus,
            gradient_correct_response,
            None,
            None,
            None,
        )

def csi_drift_rate(
    stimulus: torch.Tensor,
    control_activity: torch.Tensor,
    correct_response: torch.Tensor,
) -> torch.Tensor:
    """Readable Torch transcription of the production seven-input CSI UDF."""
    x0, x1, x2, x3 = stimulus.unbind(dim=-1)
    c0, c1 = control_activity.unbind(dim=-1)
    a = torch.sigmoid((x0 - x1) + 4.0 * c0 - 4.0)
    b = torch.sigmoid((x1 - x0) + 4.0 * c0 - 4.0)
    c = torch.sigmoid((x2 - x3) + 4.0 * c1 - 4.0)
    d = torch.sigmoid((x3 - x2) + 4.0 * c1 - 4.0)
    positive = torch.sigmoid(a - b + c - d)
    negative = torch.sigmoid(-a + b - c + d)
    return (positive - negative) * correct_response


def pnl_euler_lca_step(
    state: torch.Tensor,
    task: torch.Tensor,
    gain: torch.Tensor,
    *,
    leak: float = 12.0,
    competition: float = 3.0,
    step_size: float = 0.01,
) -> torch.Tensor:
    """One Euler step with the equation implemented by PNL's CSI LCA."""
    activity = torch.sigmoid(gain * state)
    recurrent = torch.stack(
        (-competition * activity[..., 1], -competition * activity[..., 0]),
        dim=-1,
    )
    return state + step_size * (-leak * state + task + recurrent)


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(
        [5.0, 5.0, 5.0, 0.0, 0.05, 0.05, 0.05,
         -0.3, -0.3, -0.3, 0.1, 0.1, 0.1],
        dtype=float,
    )
    upper = np.asarray(
        [35.0, 35.0, 35.0, 0.30, 0.25, 0.25, 0.25,
         0.0, 0.0, 0.0, 0.4, 0.4, 0.4],
        dtype=float,
    )
    return lower, upper


def parameter_names() -> Sequence[str]:
    names = [f"gain[{condition}]" for condition in CONDITIONS]
    names.append("csi_duration")
    names.extend(f"threshold[{condition}]" for condition in CONDITIONS)
    names.extend(f"collapse_rate[{condition}]" for condition in CONDITIONS)
    names.extend(f"non_decision_time[{condition}]" for condition in CONDITIONS)
    return tuple(names)
