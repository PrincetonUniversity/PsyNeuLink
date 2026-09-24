"""A coupled continuous-time extension of the DAWA fitting configuration.

All modulation is instantaneous. C carries its integrator state across trials;
S, D, R and LC reset to zero. LC's own clock runs 20 times faster than the LCA
clock, preserving 10 * .02 internal units per .01-second source-model pass.
This explicitly defined continuous model removes finite-pass gain-allocation
and sample/hold artifacts. It is not the old scheduler with a smaller step.
"""

from dataclasses import dataclass
import math

import torch
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from .model import validate_parameters


def _rhs(state: torch.Tensor, p: torch.Tensor, task: torch.Tensor, stimulus: torch.Tensor, clock_ratio: float):
    c, s, d = state[:2], state[2:6], state[6:8]
    v, w = state[8], state[9]
    gain = p[6] + p[5] * w
    ca = torch.sigmoid(p[3] * c)
    sa = torch.sigmoid(gain * (s + p[2]))
    da = torch.sigmoid(gain * (d + p[2]))
    dc = task - 7. * c - 3. * ca.flip(0)
    raw = torch.stack((stimulus[0] - stimulus[1], stimulus[1] - stimulus[0],
                       1.2 * (stimulus[2] - stimulus[3]), 1.2 * (stimulus[3] - stimulus[2])))
    ds = raw + 4. * ca.repeat_interleave(2) - 8. * s - 8. * sa.reshape(2, 2).flip(1).reshape(4)
    contrast = sa[0] - sa[1] + 1.2 * (sa[2] - sa[3])
    dd = 4. * ca.sum() + torch.stack((contrast, -contrast)) - 8. * d - 8. * da.flip(0)
    dv = clock_ratio * (-v**3 + 1.5 * v**2 - .5 * v - w + .3 * da.sum()) / .05
    dw = clock_ratio * (p[4] * v - w + (1. - p[4]) * .5) / 5.
    return torch.cat((dc, ds, dd, torch.stack((dv, dw))))


def _scan(state: torch.Tensor, p: torch.Tensor, task: torch.Tensor, stimulus: torch.Tensor,
          dt: float, steps: int, substeps: int, clock_ratio: float):
    rows = [state]
    h = dt / float(substeps)
    for _ in range(steps):
        for _ in range(substeps):
            k1 = _rhs(state, p, task, stimulus, clock_ratio)
            k2 = _rhs(state + h * .5 * k1, p, task, stimulus, clock_ratio)
            k3 = _rhs(state + h * .5 * k2, p, task, stimulus, clock_ratio)
            k4 = _rhs(state + h * k3, p, task, stimulus, clock_ratio)
            state = state + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
        rows.append(state)
    return torch.stack(rows)


def _control_rhs(c: torch.Tensor, task: torch.Tensor, gain: torch.Tensor):
    return task - 7. * c - 3. * torch.sigmoid(gain * c).flip(0)


def _control_scan(c: torch.Tensor, task: torch.Tensor, gain: torch.Tensor, duration: torch.Tensor, steps: int):
    h = duration / float(steps)
    for _ in range(steps):
        k1 = _control_rhs(c, task, gain)
        k2 = _control_rhs(c + .5 * h * k1, task, gain)
        k3 = _control_rhs(c + .5 * h * k2, task, gain)
        k4 = _control_rhs(c + h * k3, task, gain)
        c = c + h / 6. * (k1 + 2. * k2 + 2. * k3 + k4)
    return c


@dataclass(frozen=True)
class ContinuousPath:
    inputs: torch.Tensor
    gain: torch.Tensor
    gain_rate: torch.Tensor
    states: torch.Tensor
    time_step: float


def advance_control(control, parameters, task, duration, *, max_step=.002):
    """Replay the actual candidate-dependent duration, including its gradient."""
    validate_parameters(parameters)
    if not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("The control integration step must be finite and positive.")
    if not bool(torch.isfinite(duration)) or float(duration.detach()) <= 0:
        raise ValueError("Observed decision duration must be finite and positive.")
    device = parameters.device
    p, c = parameters.to("cpu"), control.to("cpu")
    task = torch.as_tensor(task, dtype=p.dtype, device="cpu")
    if task.shape != (2,) or c.shape != (2,) or not bool(torch.isfinite(torch.cat((task, c))).all()):
        raise ValueError("Expected finite task[2] and control[2].")
    duration = duration.to("cpu")
    steps = max(1, math.ceil(float(duration.detach()) / max_step))
    if torch.is_grad_enabled() and (p.requires_grad or c.requires_grad or duration.requires_grad):
        with set_checkpoint_early_stop(False):
            c = checkpoint(_control_scan, c, task, p[3], duration, steps, use_reentrant=False)
    else:
        c = _control_scan(c, task, p[3], duration, steps)
    return c.to(device)


def continuous_path(parameters, task, stimulus, *, steps, time_step=.001, ode_step=.0005,
                    clock_ratio=20., control=None, checkpoint_steps=32):
    validate_parameters(parameters)
    if not isinstance(steps, int) or steps < 1 or checkpoint_steps < 0 or any(
            not math.isfinite(x) or x <= 0 for x in (time_step, ode_step, clock_ratio)):
        raise ValueError("Positive integration settings are required.")
    device = parameters.device
    # Tiny ODE systems run on CPU; differentiable copies connect to GPU density
    # solves without launching hundreds of tiny CUDA operations per ODE step.
    p = parameters.to("cpu")
    task = torch.as_tensor(task, dtype=p.dtype, device="cpu")
    stimulus = torch.as_tensor(stimulus, dtype=p.dtype, device="cpu")
    if task.shape != (2,) or stimulus.shape != (4,) or not bool(torch.isfinite(torch.cat((task, stimulus))).all()):
        raise ValueError("Expected finite task[2] and stimulus[4].")
    c = p.new_zeros(2) if control is None else control.to("cpu")
    if c.shape != (2,) or not bool(torch.isfinite(c).all()):
        raise ValueError("Expected finite control[2].")
    state = torch.cat((c, p.new_zeros(8)))
    parts = [state[None]]
    substeps = max(1, math.ceil(time_step / ode_step))
    block = checkpoint_steps or steps
    for start in range(0, steps, block):
        args = (state, p, task, stimulus, time_step, min(block, steps - start), substeps, clock_ratio)
        if checkpoint_steps and torch.is_grad_enabled() and (p.requires_grad or state.requires_grad):
            with set_checkpoint_early_stop(False):
                states = checkpoint(_scan, *args, use_reentrant=False)
        else:
            states = _scan(*args)
        parts.append(states[1:])
        state = states[-1]
    states = torch.cat(parts)
    gain = p[6] + p[5] * states[:, 9]
    ca = torch.sigmoid(p[3] * states[:, :2])
    da = torch.sigmoid(gain[:, None] * (states[:, 6:8] + p[2]))
    contrast = da[:, 0] - da[:, 1]
    inputs = 4. * ca.sum(dim=1, keepdim=True) + torch.stack((contrast, -contrast), dim=1)
    gain_rate = p[5] * clock_ratio * (p[4] * states[:, 8] - states[:, 9] + (1. - p[4]) * .5) / 5.
    return ContinuousPath(inputs.to(device), gain.to(device), gain_rate.to(device), states.to(device), time_step)
