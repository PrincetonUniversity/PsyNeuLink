"""Independent CUDA first-passage sampler for prescribed continuous CSI drifts.

Research benchmark/validation helper, not the scheduled PEC fitting sampler.
Uses float32 Euler updates, local Brownian-bridge crossing corrections, endpoint
interpolation for visible crossings, and midpoint times for hidden crossings.
The deterministic nonlinear LCA drift is supplied by the existing model.
"""

import math

import numpy as np
import torch
import triton
import triton.language as tl


@triton.jit
def _sample(Drift, Threshold, Collapse, Output, N: tl.constexpr, STEPS: tl.constexpr,
            DT: tl.constexpr, NOISE: tl.constexpr, SEED: tl.constexpr, BLOCK: tl.constexpr):
    lane = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    trial = tl.program_id(1)
    active = lane < N
    evidence = tl.full((BLOCK,), 0., tl.float32)
    choice = tl.full((BLOCK,), -1., tl.float32)
    finish = tl.full((BLOCK,), STEPS * DT, tl.float32)
    threshold, collapse = tl.load(Threshold + trial), tl.load(Collapse + trial)
    seed = (SEED + trial * 104729).to(tl.uint32)
    k = 0
    while (k < STEPS) & (tl.sum(active.to(tl.int32), 0) > 0):
        old_bound, new_bound = threshold + collapse * (k * DT), threshold + collapse * ((k + 1) * DT)
        offset = lane.to(tl.uint32) * (3 * STEPS) + 3 * k
        next_value = evidence + DT * tl.load(Drift + trial * STEPS + k) + NOISE * tl.sqrt(DT) * tl.randn(seed, offset)
        upper, lower = next_value >= new_bound, next_value <= -new_bound
        pu = tl.exp(tl.minimum(0., -2. * (old_bound - evidence) * (new_bound - next_value) / (NOISE * NOISE * DT)))
        pl = tl.exp(tl.minimum(0., -2. * (old_bound + evidence) * (new_bound + next_value) / (NOISE * NOISE * DT)))
        hidden = (~upper) & (~lower) & (tl.rand(seed, offset + 1) < tl.minimum(1., pu + pl))
        hidden_upper = tl.rand(seed, offset + 2) * (pu + pl) < pu
        ended = active & (upper | lower | hidden)
        fraction = tl.where(upper, (old_bound - evidence) / tl.maximum(old_bound - evidence - new_bound + next_value, 1.e-20),
                            tl.where(lower, (old_bound + evidence) / tl.maximum(old_bound + evidence - new_bound - next_value, 1.e-20), .5))
        choice = tl.where(ended, tl.where(upper | (hidden & hidden_upper), 1., 0.), choice)
        finish = tl.where(ended, (k + fraction) * DT, finish)
        active = active & (~ended)
        evidence = next_value
        k += 1
    tl.store(Output + (trial * N + lane) * 2, choice, lane < N)
    tl.store(Output + (trial * N + lane) * 2 + 1, finish, lane < N)


def simulate_continuous_drift(drift, threshold, collapse, *, time_step, estimates=100000, noise=.1, seed=43):
    """Return [trial, estimate, (choice, decision time)], including censored paths.

    Choice -1 denotes right censoring at the horizon; it stays in the probability
    denominator. Transfers and output synchronization happen inside this call.
    """
    if drift.ndim != 2 or drift.shape[1] < 1 or threshold.shape != (len(drift),) or collapse.shape != threshold.shape:
        raise ValueError("Expected drift[trial,time], threshold[trial], collapse[trial].")
    if not isinstance(estimates, int) or estimates < 1 or any(not math.isfinite(x) or x <= 0 for x in (time_step, noise)):
        raise ValueError("Positive estimates, time step, and noise are required.")
    if not all(bool(torch.isfinite(v).all()) for v in (drift, threshold, collapse)):
        raise ValueError("Nonfinite coefficients.")
    steps = drift.shape[1]
    if bool((threshold <= 0).any()) or bool((threshold + collapse * steps * time_step <= 0).any()):
        raise ValueError("The symmetric boundary must stay positive throughout the horizon.")
    if 3 * steps * estimates >= 2**32:
        raise ValueError("Requested workload exceeds unique 32-bit RNG offsets; split estimates into independent batches.")
    values = [v.detach().to(device="cuda", dtype=torch.float32).contiguous() for v in (drift, threshold, collapse)]
    output = torch.empty((len(drift), estimates, 2), device="cuda", dtype=torch.float32)
    _sample[(triton.cdiv(estimates, 256), len(drift))](*values, output, estimates, steps, time_step, noise, seed, 256)
    result = output.cpu().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite Monte Carlo output.")
    return result
