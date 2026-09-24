"""Differentiable replay of the deterministic part of the DAWA fitting model.

This mirrors the shared builder's corrected schedule, including the first-pass
gain allocation of 1 and the distinction between held and sampled gains at
trial resets. Constants match dawa_batched_simulation.DEFAULTS. Only response
noise is supported; changing the source model requires a new parity audit.
"""

from dataclasses import dataclass
import math

import torch


PARAMETER_NAMES = ("threshold", "non_decision_time", "bias", "control_gain",
                   "lc_mode", "lc_scaling", "lc_base_gain")
DEFAULT_PARAMETERS = (.3, .2, -.45, 10., .9, 1., 5.)
PARAMETER_BOUNDS = ((.25, .7), (.1, .3), (-.5, 0.), (5., 20.), (.1, .9), (1., 4.), (3., 10.))
DT = .01


@dataclass(frozen=True)
class History:
    control_pre: torch.Tensor
    control_activity: torch.Tensor
    held_gain: torch.Tensor
    sampled_sd_gain: torch.Tensor
    sampled_response_gain: torch.Tensor
    sampled_bias: torch.Tensor


@dataclass(frozen=True)
class ResponsePath:
    inputs: torch.Tensor
    gain: torch.Tensor
    initial_activity: torch.Tensor
    history: tuple[History, ...]
    # Independent native-model parity checks use these internal trajectories.
    control: torch.Tensor
    stimulus: torch.Tensor
    decision: torch.Tensor
    lc: torch.Tensor
    time_step: float = DT


def initial_history(parameters):
    zero = parameters.new_zeros(2)
    return History(zero, zero + .5, parameters.new_tensor(1.),
                   parameters.new_tensor(5.), parameters.new_tensor(5.), parameters.new_tensor(-.45))


def validate_parameters(parameters):
    if parameters.shape != (7,) or not bool(torch.isfinite(parameters).all()):
        raise ValueError("Expected seven finite DAWA parameters.")
    threshold, ndt, _, control_gain, mode, scaling, base_gain = parameters.detach().tolist()
    if not (0 < threshold < 1 and ndt >= 0 and control_gain > 0 and 0 <= mode <= 1
            and scaling >= 0 and base_gain > 0):
        raise ValueError("Invalid threshold, nondecision time, gain, or LC parameters.")


def response_path(parameters, task, stimulus, steps, history=None, *, time_step=DT, lc_clock_ratio=20., lc_substeps=10):
    """Replay a fixed number of scheduler passes (10 ms by default).

    ``history[k]`` is the state to carry into the next trial if this trial ends
    after k+1 passes. No response noise, choice, or unobserved response state is
    needed to compute it. All carried tensors retain their autograd history.
    """
    validate_parameters(parameters)
    if not isinstance(steps, int) or steps < 1:
        raise ValueError("steps must be a positive integer.")
    if any(not math.isfinite(x) or x <= 0 for x in (time_step, lc_clock_ratio)):
        raise ValueError("Time step and LC clock ratio must be finite and positive.")
    if not isinstance(lc_substeps, int) or lc_substeps < 1:
        raise ValueError("LC substeps must be a positive integer.")
    lc_step = lc_clock_ratio * time_step / lc_substeps
    task = torch.as_tensor(task, dtype=parameters.dtype, device=parameters.device)
    stimulus = torch.as_tensor(stimulus, dtype=parameters.dtype, device=parameters.device)
    if task.shape != (2,) or stimulus.shape != (4,) or not bool(torch.isfinite(torch.cat((task, stimulus))).all()):
        raise ValueError("Expected finite task[2] and stimulus[4] inputs.")
    history = initial_history(parameters) if history is None else history
    bias, control_gain, mode, scaling, base_gain = parameters[2:]
    c_pre, c_act = history.control_pre, history.control_activity
    s_pre, d_pre = parameters.new_zeros(4), parameters.new_zeros(2)
    s_act = torch.sigmoid(history.sampled_sd_gain * history.sampled_bias).expand(4)
    d_act = s_act[:2]
    r_initial = torch.sigmoid(history.sampled_response_gain * history.sampled_bias).expand(2)
    v, w = parameters.new_zeros(()), parameters.new_zeros(())
    held_gain = history.held_gain
    raw = torch.stack((stimulus[0] - stimulus[1], stimulus[1] - stimulus[0],
                       1.2 * (stimulus[2] - stimulus[3]), 1.2 * (stimulus[3] - stimulus[2])))
    inputs, gains, histories, controls, stimuli, decisions, lcs = [], [], [], [], [], [], []
    for _ in range(steps):
        c_pre = c_pre + time_step * (task - 3. * c_act.flip(0) - 7. * c_pre)
        c_act = torch.sigmoid(control_gain * c_pre)
        s_pre = s_pre + time_step * (raw + 4. * c_act.repeat_interleave(2)
                              - 8. * s_act.reshape(2, 2).flip(1).reshape(4) - 8. * s_pre)
        s_act = torch.sigmoid(held_gain * (s_pre + bias))
        contrast = s_act[0] - s_act[1] + 1.2 * (s_act[2] - s_act[3])
        d_pre = d_pre + time_step * (4. * c_act.sum() + torch.stack((contrast, -contrast))
                              - 8. * d_act.flip(0) - 8. * d_pre)
        d_act = torch.sigmoid(held_gain * (d_pre + bias))
        drive = .3 * d_act.sum()
        for _ in range(lc_substeps):
            dv = (-v**3 + 1.5 * v**2 - .5 * v - w + drive) / .05
            dw = (mode * v - w + (1. - mode) * .5) / 5.
            v, w = v + lc_step * dv, w + lc_step * dw
        gain = base_gain + scaling * w
        contrast = d_act[0] - d_act[1]
        inputs.append(4. * c_act.sum() + torch.stack((contrast, -contrast)))
        gains.append(gain)
        histories.append(History(c_pre, c_act, gain, held_gain, gain, bias))
        controls.append(c_act)
        stimuli.append(s_act)
        decisions.append(d_act)
        lcs.append(torch.stack((v, w)))
        held_gain = gain
    return ResponsePath(torch.stack(inputs), torch.stack(gains), r_initial, tuple(histories),
                        torch.stack(controls), torch.stack(stimuli), torch.stack(decisions), torch.stack(lcs), time_step)
