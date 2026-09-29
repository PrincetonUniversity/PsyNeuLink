"""Reusable CUDA observation weighting and particle-state gathering.

These operations consume the generic observation tables and flat state ABI.
They do not recognize model names or change the simulator, observation law,
random streams, or double-precision cumulative resampling weights.
"""

import math

from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module


_SOURCE = '''
import triton
import triton.language as tl

@triton.jit(do_not_specialize=["total", "trial"])
def observation_contributions(sim, observed, edges, lookups, output, total, trial,
                              WIDTH: tl.constexpr, CATS: tl.constexpr, CONS: tl.constexpr,
                              BINS: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = index < total
    weight = tl.full((BLOCK,), 1., tl.float32)
    for c in tl.static_range(len(CATS)):
        value = tl.load(sim + index * WIDTH + CATS[c], active, other=0.)
        target = tl.load(observed + trial * len(CATS) + c)
        weight = weight * (tl.abs(value - target) <= 1.e-6).to(tl.float32)
    for c in tl.static_range(len(CONS)):
        value = tl.load(sim + index * WIDTH + CONS[c], active, other=0.)
        lo = tl.zeros((BLOCK,), tl.int32)
        hi = tl.full((BLOCK,), BINS - 1, tl.int32)
        for iteration in range(STEPS):
            mid = (lo + hi) // 2
            edge = tl.load(edges[c] + mid + 1)
            right = (lo < hi) & (edge < value)
            lo = tl.where(right, mid + 1, lo)
            hi = tl.where(~right, mid, hi)
        contribution = tl.load(lookups[c] + trial * BINS + lo)
        inside = (value >= tl.load(edges[c])) & (value <= tl.load(edges[c] + BINS))
        weight = weight * contribution * inside.to(tl.float32)
    tl.store(output + index, weight, active)

@triton.jit(do_not_specialize=["estimates"])
def resample_states(cumulative, positions, terminal, output, estimates,
                    WIDTH: tl.constexpr, STEPS: tl.constexpr,
                    PARTICLES: tl.constexpr, FIELDS: tl.constexpr):
    row = tl.program_id(1)
    particle = tl.program_id(0) * PARTICLES + tl.arange(0, PARTICLES)
    active = particle < estimates
    position = tl.load(positions + row * estimates + particle, active, other=0.)
    lo = tl.zeros((PARTICLES,), tl.int32)
    hi = tl.full((PARTICLES,), estimates, tl.int32)
    for iteration in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(cumulative + row * estimates + mid, active & (mid < estimates), other=float("inf"))
        right = (lo < hi) & (value <= position)
        lo = tl.where(right, mid + 1, lo)
        hi = tl.where(~right, mid, hi)
    ancestor = tl.minimum(lo, estimates - 1)
    field = tl.arange(0, FIELDS)
    mask = active[:, None] & (field[None, :] < WIDTH)
    state = tl.load(terminal + (row * estimates + ancestor[:, None]) * WIDTH + field[None, :], mask, other=0.)
    tl.store(output + (row * estimates + particle[:, None]) * WIDTH + field[None, :], state, mask)
'''


def _module():
    return load_triton_kernel_module(_SOURCE, "conditioned_ops", "generic", interpret=False)


class FusedObservationContributions:
    """Match Torch's per-particle FP32 contributions, before sum/contamination.

    Retain the existing Torch reductions to preserve score rounding and ancestry.
    Returned storage is reused; consume it before the next invocation.
    """

    def __init__(self, tables):
        with interpret_scope(False):
            self.kernel = _module().observation_contributions
        self.observed_cat = tables.observed_cat
        self.edges = tuple(tables.edges)
        self.lookups = tuple(tables.bin_weights)
        self.width = tables.num_outcomes
        self.cats = tables.cat_indices
        self.cons = tuple(int(i) for i in tables.con_indices)
        self.bins = tables.bins
        self.steps = math.ceil(math.log2(self.bins))
        self.output = None

    def __call__(self, sim, trial):
        import torch

        sim = sim.contiguous()
        shape = sim.shape[:-1]
        if self.output is None or self.output.shape != shape:
            self.output = torch.empty(shape, device=sim.device, dtype=sim.dtype)
        total = self.output.numel()
        with interpret_scope(False):
            self.kernel[((total + 255) // 256,)](
                sim, self.observed_cat, self.edges, self.lookups,
                self.output, total, trial, WIDTH=self.width, CATS=self.cats, CONS=self.cons,
                BINS=self.bins, STEPS=self.steps, BLOCK=256,
                num_warps=4, enable_fp_fusion=False,
            )
        return self.output


class SystematicStateResampler:
    """Use the reference FP64 CDF/positions, fusing ancestor lookup and gather.

    State/output is contiguous FP32; rows correspond to independent filters.
    The output allocation is reused after the next trial has consumed it.
    """

    def __init__(self, estimates, device):
        import torch

        from psyneulink.core.batched.compiler import _systematic_resampling_inputs

        self.prepare = _systematic_resampling_inputs
        self.indices = torch.arange(estimates, dtype=torch.float64, device=device)
        with interpret_scope(False):
            self.kernel = _module().resample_states
        self.output = None

    def __call__(self, weights, terminal, *, generator, shared_first_axis=False):
        import torch
        import triton

        cumulative, positions = self.prepare(
            weights, generator=generator, shared_first_axis=shared_first_axis, indices=self.indices,
        )
        estimates, width = terminal.shape[-2:]
        rows = terminal.numel() // (estimates * width)
        if self.output is None:
            self.output = torch.empty_like(terminal)
        with interpret_scope(False):
            self.kernel[(triton.cdiv(estimates, 32), rows)](
                cumulative, positions, terminal, self.output, estimates,
                WIDTH=width, STEPS=math.ceil(math.log2(estimates + 1)), PARTICLES=32,
                FIELDS=triton.next_power_of_2(width), num_warps=4, enable_fp_fusion=False,
            )
        return self.output
