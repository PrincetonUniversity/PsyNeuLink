"""Independent Euler SDE validation with Brownian-bridge crossing correction.

The bridge freezes local drift/diffusion and linearizes the boundary over a
microstep. Within-step crossing times use endpoint interpolation or a midpoint
for hidden bridge crossings. Refining the microstep audits these approximations;
this sampler is not used to evaluate the direct likelihood or its gradients.
"""

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _sample(Inputs, Gain, Bound, Output, N: tl.constexpr, STEPS: tl.constexpr,
            DT: tl.constexpr, NOISE: tl.constexpr, BIAS: tl.constexpr, SEED: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = lane < N
    x = tl.full((BLOCK,), 0., tl.float32)
    y = tl.full((BLOCK,), 0., tl.float32)
    choice = tl.full((BLOCK,), -1., tl.float32)
    finish = tl.full((BLOCK,), STEPS * DT, tl.float32)
    k = 0
    while (k < STEPS) & (tl.sum(active.to(tl.int32), 0) > 0):
        gain = tl.load(Gain + k)
        old_bound, new_bound = tl.load(Bound + k), tl.load(Bound + k + 1)
        offset = lane.to(tl.uint32) * (4 * STEPS) + 4 * k
        next_x = x + DT * (tl.load(Inputs + 2 * k) - 8. * x - 8. / (1. + tl.exp(-gain * (y + BIAS))))
        next_y = y + DT * (tl.load(Inputs + 2 * k + 1) - 8. * y - 8. / (1. + tl.exp(-gain * (x + BIAS))))
        next_x += NOISE * tl.sqrt(DT) * tl.randn(SEED, offset)
        next_y += NOISE * tl.sqrt(DT) * tl.randn(SEED, offset + 1)
        dx0, dx1 = old_bound - x, new_bound - next_x
        dy0, dy1 = old_bound - y, new_bound - next_y
        cross_x = (dx1 <= 0.) | (tl.rand(SEED, offset + 2) < tl.exp(tl.minimum(0., -2. * dx0 * dx1 / (NOISE * NOISE * DT))))
        cross_y = (dy1 <= 0.) | (tl.rand(SEED, offset + 3) < tl.exp(tl.minimum(0., -2. * dy0 * dy1 / (NOISE * NOISE * DT))))
        tx = tl.where(dx1 <= 0., dx0 / tl.maximum(dx0 - dx1, 1.e-20), .5)
        ty = tl.where(dy1 <= 0., dy0 / tl.maximum(dy0 - dy1, 1.e-20), .5)
        left = cross_x & ((~cross_y) | (tx < ty) | ((tx == ty) & (next_x >= next_y)))
        ended = active & (cross_x | cross_y)
        choice = tl.where(ended, tl.where(left, 0., 1.), choice)
        finish = tl.where(ended, (k + tl.where(left, tx, ty)) * DT, finish)
        active = active & (~ended)
        x, y = next_x, next_y
        k += 1
    tl.store(Output + 2 * lane, choice, lane < N)
    tl.store(Output + 2 * lane + 1, finish, lane < N)


def simulate_continuous(path, parameters, *, estimates=50000, seed=43, noise=.1):
    if not torch.cuda.is_available():
        raise RuntimeError("Continuous Monte Carlo validation requires CUDA.")
    if estimates < 1:
        raise ValueError("estimates must be positive.")
    steps = len(path.gain) - 1
    if 4 * steps * estimates >= 2**32:
        raise ValueError("Requested Monte Carlo workload exceeds unique 32-bit RNG offsets.")
    inputs = path.inputs.detach().to(device="cuda", dtype=torch.float32).contiguous()
    gain = path.gain.detach().to(device="cuda", dtype=torch.float32).contiguous()
    bound = (torch.logit(parameters[0]) / path.gain - parameters[2]).detach().to(device="cuda", dtype=torch.float32).contiguous()
    output = torch.empty((estimates, 2), dtype=torch.float32, device="cuda")
    _sample[(triton.cdiv(estimates, 256),)](inputs, gain, bound, output, estimates, steps, path.time_step,
                                         noise, float(parameters[2].detach()), seed, 256)
    result = output.cpu().numpy()
    if not np.all(np.isfinite(result)):
        raise AssertionError("Nonfinite Monte Carlo output.")
    return result
