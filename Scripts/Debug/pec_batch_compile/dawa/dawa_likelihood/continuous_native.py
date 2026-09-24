"""DAWA equations using the general generated RK4 phase/adjoint backend.

Only the equation declaration is model-specific. Code generation, integration,
and reverse differentiation use psyneulink.core.batched.numerical.dynamics.
The phase API samples at cell midpoints. A half-cell prefix aligns those samples
with the endpoints required by the finite-volume solver; the unused final half
cell has no influence on returned states or gradients.
"""

from functools import lru_cache
import math

import sympy as sp
import torch

from psyneulink.core.batched.continuous_ir import ContinuousDynamics, Sigmoid
from psyneulink.core.batched.numerical.dynamics import compile_continuous_phase

from .continuous_model import ContinuousPath
from .model import validate_parameters


def equations(*, control_only=False):
    c0, c1, s0, s1, s2, s3, d0, d1, v, w = sp.symbols("c0 c1 s0 s1 s2 s3 d0 d1 v w", real=True)
    t0, t1, u0, u1, u2, u3 = sp.symbols("task0 task1 stimulus0 stimulus1 stimulus2 stimulus3", real=True)
    threshold, ndt, bias, cg, mode, scale, base, clock = sp.symbols("threshold ndt bias cg mode scale base clock", real=True)
    ca = (Sigmoid(cg * c0), Sigmoid(cg * c1))
    dc = (t0 - 7 * c0 - 3 * ca[1], t1 - 7 * c1 - 3 * ca[0])
    if control_only:
        return ContinuousDynamics(states=(c0, c1), inputs=(t0, t1), parameters=(cg,), drift=dc, readouts=())
    g = base + scale * w
    sa = tuple(Sigmoid(g * (s + bias)) for s in (s0, s1, s2, s3))
    da = (Sigmoid(g * (d0 + bias)), Sigmoid(g * (d1 + bias)))
    r = sp.Rational(6, 5)
    raw = (u0 - u1, u1 - u0, r * (u2 - u3), r * (u3 - u2))
    ds = tuple(raw[i] + 4 * ca[i // 2] - 8 * s - 8 * sa[i ^ 1] for i, s in enumerate((s0, s1, s2, s3)))
    contrast = sa[0] - sa[1] + r * (sa[2] - sa[3])
    dd = (4 * sum(ca) + contrast - 8 * d0 - 8 * da[1], 4 * sum(ca) - contrast - 8 * d1 - 8 * da[0])
    dv = clock * (-v**3 + sp.Rational(3, 2) * v**2 - v / 2 - w + sp.Rational(3, 10) * sum(da)) * 20
    dw = clock * (mode * v - w + (1 - mode) / 2) / 5
    states = (c0, c1, s0, s1, s2, s3, d0, d1, v, w)
    return ContinuousDynamics(states=states, inputs=(t0, t1, u0, u1, u2, u3),
                              parameters=(threshold, ndt, bias, cg, mode, scale, base, clock),
                              drift=dc + ds + dd + (dv, dw), readouts=tuple((str(s), s) for s in states))


@lru_cache(maxsize=2)
def phase(control_only=False):
    return compile_continuous_phase(equations(control_only=control_only))


def _integrate(plan, state, inputs, parameters, duration, count, start=0.):
    duration = torch.as_tensor(duration, dtype=torch.float64, device="cpu").reshape(1)
    return plan.integrate(state=state.reshape(1, -1), inputs=inputs.reshape(1, -1), parameters=parameters.reshape(1, -1),
                          duration=duration, start_time=duration.new_tensor([start]), steps=torch.tensor([count], dtype=torch.int64))


def continuous_path_native(parameters, task, stimulus, *, steps, time_step=.001, ode_step=.0005,
                           clock_ratio=20., control=None):
    validate_parameters(parameters)
    if parameters.dtype != torch.float64:
        raise ValueError("The generated dynamics backend requires float64 parameters.")
    if not isinstance(steps, int) or steps < 1 or any(not math.isfinite(x) or x <= 0 for x in (time_step, ode_step, clock_ratio)):
        raise ValueError("Positive integration settings are required.")
    device, p = parameters.device, parameters.to("cpu")
    task = torch.as_tensor(task, dtype=p.dtype, device="cpu")
    stimulus = torch.as_tensor(stimulus, dtype=p.dtype, device="cpu")
    if task.shape != (2,) or stimulus.shape != (4,):
        raise ValueError("Expected task[2] and stimulus[4].")
    c = p.new_zeros(2) if control is None else control.to("cpu")
    if c.shape != (2,):
        raise ValueError("Expected control[2].")
    initial = torch.cat((c, p.new_zeros(8)))
    inputs, parameters = torch.cat((task, stimulus)), torch.cat((p, p.new_tensor([clock_ratio])))
    count = max(1, math.ceil(time_step / (2. * ode_step)))
    h = time_step / count
    plan = phase()
    prefix = _integrate(plan, initial, inputs, parameters, .5 * h, 1).final_state[0]
    result = _integrate(plan, prefix, inputs, parameters, steps * time_step, steps * count, .5 * h)
    states = torch.cat((initial[None], result.readouts[0, count - 1::count]))
    gain = p[6] + p[5] * states[:, 9]
    ca = torch.sigmoid(p[3] * states[:, :2])
    da = torch.sigmoid(gain[:, None] * (states[:, 6:8] + p[2]))
    contrast = da[:, 0] - da[:, 1]
    drive = 4. * ca.sum(dim=1, keepdim=True) + torch.stack((contrast, -contrast), dim=1)
    gain_rate = p[5] * clock_ratio * (p[4] * states[:, 8] - states[:, 9] + (1. - p[4]) * .5) / 5.
    return ContinuousPath(drive.to(device), gain.to(device), gain_rate.to(device), states.to(device), time_step)


def advance_control_native(control, parameters, task, duration, *, max_step=.002):
    validate_parameters(parameters)
    if parameters.dtype != torch.float64:
        raise ValueError("The generated dynamics backend requires float64 parameters.")
    if not bool(torch.isfinite(duration)) or float(duration.detach()) <= 0 or not math.isfinite(max_step) or max_step <= 0:
        raise ValueError("Positive finite duration and step are required.")
    p = parameters.to("cpu")
    task = torch.as_tensor(task, dtype=p.dtype, device="cpu")
    result = _integrate(phase(True), control.to("cpu"), task, p[3:4], duration.to("cpu"),
                        max(1, math.ceil(float(duration.detach()) / (2. * max_step))))
    return result.final_state[0].to(parameters.device)
