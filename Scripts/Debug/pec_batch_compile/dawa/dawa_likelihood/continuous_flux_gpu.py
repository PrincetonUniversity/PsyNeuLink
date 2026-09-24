"""CUDA float64 finite-volume rates, SSP-RK2 propagation, and exact discrete VJPs.

Triton kernels retain the native CPU discretization. CUDA graphs amortize kernel
launch overhead over a checkpoint block; graph outputs are copied before reuse.
Time steps stay ordered. No model state or trial history is reset by this backend.
"""

from collections import OrderedDict
import math
from threading import RLock

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _bernoulli(z):
    small = z < .1
    a = tl.where(small, 1., z)
    e = libdevice.exp(-a)
    denominator = 1. - e
    z2 = z * z
    value = 1. - z / 2. + z2 * (1. / 12. + z2 * (-1. / 720. + z2 * (1. / 30240. + z2 * (-1. / 1209600. + z2 / 47900160.))))
    derivative = -.5 + z * (1. / 6. + z2 * (-1. / 180. + z2 * (1. / 5040. + z2 * (-1. / 151200. + z2 / 4790016.))))
    return tl.where(small, value, a * e / denominator), tl.where(small, derivative, e / denominator * (1. - a / denominator))


@triton.jit
def _coefficients(Inputs, Gain, Bias, Boundary, Speed, Rates, Adjoint, Parts,
                  N: tl.constexpr, LOW: tl.constexpr, NOISE: tl.constexpr,
                  LEAK: tl.constexpr, COMPETITION: tl.constexpr,
                  REVERSE: tl.constexpr, BLOCK: tl.constexpr):
    tile, k, axis = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index = tile * BLOCK + tl.arange(0, BLOCK)
    valid = index < N * (N + 1)
    face_index, other = index // N, index % N
    face, center = face_index.to(tl.float64) / N, (other.to(tl.float64) + .5) / N
    width = tl.load(Boundary + k) - LOW
    gain, bias, speed = tl.load(Gain + k), tl.load(Bias), tl.load(Speed + k)
    argument = gain * (LOW + width * center + bias)
    e = libdevice.exp(-tl.abs(argument))
    activation = tl.where(argument >= 0., 1. / (1. + e), e / (1. + e))
    derivative = e / ((1. + e) * (1. + e))
    edge = (face_index == 0) | (face_index == N)
    diffusion: tl.constexpr = .5 * NOISE * NOISE
    factor = tl.where(edge, .5, 1.).to(tl.float64) / (N * diffusion)
    drift = tl.load(Inputs + 2 * k + axis) - LEAK * (LOW + width * face) - face * speed - COMPETITION * activation
    q = factor * drift * width
    b, db = _bernoulli(tl.abs(q))
    down, up = tl.where(q >= 0., b, b - q), tl.where(q >= 0., b + q, b)
    prefactor = tl.where(edge, 2., 1.).to(tl.float64) * diffusion * N * N / (width * width)
    di = (k * 4 + 2 * axis) * N * N + tl.where(axis == 0, face_index * N + other, other * N + face_index)
    ui = (k * 4 + 2 * axis + 1) * N * N + tl.where(axis == 0, (face_index - 1) * N + other, other * N + face_index - 1)
    if not REVERSE:
        tl.store(Rates + di, prefactor * down, valid & (face_index < N))
        tl.store(Rates + ui, prefactor * up, valid & (face_index > 0))
    else:
        adown = tl.load(Adjoint + di, valid & (face_index < N), other=0.)
        aup = tl.load(Adjoint + ui, valid & (face_index > 0), other=0.)
        ddown, dup = tl.where(q >= 0., db, -db - 1.), tl.where(q >= 0., db + 1., -db)
        weighted = prefactor * (adown * down + aup * up)
        dq = prefactor * (adown * ddown + aup * dup) * factor
        dr = dq * width
        gg = -dr * COMPETITION * derivative * (LOW + width * center + bias)
        gb = -dr * COMPETITION * derivative * gain
        ga = -2. * weighted / width + dq * drift + dr * (-LEAK * face - COMPETITION * derivative * gain * center)
        gad = -dr * face
        blocks: tl.constexpr = triton.cdiv(N * (N + 1), BLOCK)
        base = ((k * 2 + axis) * blocks + tile) * 5
        tl.store(Parts + base, tl.sum(tl.where(valid, dr, 0.)))
        tl.store(Parts + base + 1, tl.sum(tl.where(valid, gg, 0.)))
        tl.store(Parts + base + 2, tl.sum(tl.where(valid, gb, 0.)))
        tl.store(Parts + base + 3, tl.sum(tl.where(valid, ga, 0.)))
        tl.store(Parts + base + 4, tl.sum(tl.where(valid, gad, 0.)))


@triton.jit(do_not_specialize=["k", "j"])
def _step(Mass, Rates, Original, Output, ExitParts, StageFlux, Minimum,
          k, j, SUBSTEPS: tl.constexpr, N: tl.constexpr, DT: tl.constexpr, FINISH: tl.constexpr,
          COLLECT: tl.constexpr, FIRST: tl.constexpr, RESET_FLUX: tl.constexpr, BLOCK: tl.constexpr):
    tile = tl.program_id(0)
    i = tile * BLOCK + tl.arange(0, BLOCK)
    valid = i < N * N
    row, col = i // N, i % N
    mass = tl.load(Mass + i, valid, other=0.)
    fraction = j.to(tl.float64) / SUBSTEPS
    a, b = DT * (1. - fraction), DT * fraction
    sum0 = tl.full((BLOCK,), 0., tl.float64)
    sum1 = tl.full((BLOCK,), 0., tl.float64)
    incoming = tl.full((BLOCK,), 0., tl.float64)
    f0, f1, lost = sum0, sum0, sum0
    for d in tl.static_range(4):
        r0 = tl.load(Rates + (k * 4 + d) * N * N + i, valid, other=0.)
        r1 = tl.load(Rates + ((k + 1) * 4 + d) * N * N + i, valid, other=0.)
        sum0, sum1 = sum0 + r0, sum1 + r1
        if d == 0:
            neighbor, inside = i - N, row > 0
        elif d == 1:
            neighbor, inside = i + N, row < N - 1
        elif d == 2:
            neighbor, inside = i - 1, col > 0
        else:
            neighbor, inside = i + 1, col < N - 1
        nr0 = tl.load(Rates + (k * 4 + (d ^ 1)) * N * N + neighbor, valid & inside, other=0.)
        nr1 = tl.load(Rates + ((k + 1) * 4 + (d ^ 1)) * N * N + neighbor, valid & inside, other=0.)
        nm = tl.load(Mass + neighbor, valid & inside, other=0.)
        incoming += (a * nr0 + b * nr1) * nm
        if COLLECT:
            outgoing = tl.where(valid & ~inside, (a * r0 + b * r1) * mass, 0.)
            if d == 1:
                f0 += outgoing
            elif d == 3:
                f1 += outgoing
            else:
                lost += outgoing
    value = mass * (1. - a * sum0 - b * sum1) + incoming
    if FINISH:
        original = tl.load(Original + i, valid, other=0.)
        value = .5 * (original + value)
    tl.store(Output + i, value, valid)
    if COLLECT:
        blocks: tl.constexpr = triton.cdiv(N * N, BLOCK)
        for c in tl.static_range(3):
            local = tl.sum(f0 if c == 0 else f1 if c == 1 else lost)
            if FINISH:
                increment = .5 * (tl.load(StageFlux + c * blocks + tile) + local)
                offset = (k * 3 + c) * blocks + tile
                if not RESET_FLUX:
                    increment += tl.load(ExitParts + offset)
                tl.store(ExitParts + offset, increment)
            else:
                tl.store(StageFlux + c * blocks + tile, local)
        if FINISH:
            smallest = tl.min(tl.where(valid, value, float("inf")))
            if FIRST:
                smallest = tl.minimum(smallest, tl.min(tl.where(valid, original, float("inf"))))
            else:
                smallest = tl.minimum(smallest, tl.load(Minimum + tile))
            tl.store(Minimum + tile, smallest)


@triton.jit(do_not_specialize=["k", "j"])
def _vjp(Mass, Rates, Adjoint, AdjExits, SkipAdjoint, GradMass, GradRates,
         k, j, SUBSTEPS: tl.constexpr, N: tl.constexpr, DT: tl.constexpr, SCALE: tl.constexpr,
         ADD_SKIP: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N * N
    row, col = i // N, i % N
    mass = tl.load(Mass + i, valid, other=0.)
    adj = SCALE * tl.load(Adjoint + i, valid, other=0.)
    result = adj
    fraction = j.to(tl.float64) / SUBSTEPS
    for d in tl.static_range(4):
        if d == 0:
            neighbor, inside = i - N, row > 0
        elif d == 1:
            neighbor, inside = i + N, row < N - 1
        elif d == 2:
            neighbor, inside = i - 1, col > 0
        else:
            neighbor, inside = i + 1, col < N - 1
        destination = SCALE * tl.load(Adjoint + neighbor, valid & inside, other=0.)
        exit_adjoint = .5 * tl.load(AdjExits + 3 * k + (0 if d == 1 else 1 if d == 3 else 2))
        destination = tl.where(inside, destination, exit_adjoint)
        difference = DT * (destination - adj)
        i0, i1 = (k * 4 + d) * N * N + i, ((k + 1) * 4 + d) * N * N + i
        r0, r1 = tl.load(Rates + i0, valid, other=0.), tl.load(Rates + i1, valid, other=0.)
        result += ((1. - fraction) * r0 + fraction * r1) * difference
        g0, g1 = tl.load(GradRates + i0, valid, other=0.), tl.load(GradRates + i1, valid, other=0.)
        tl.store(GradRates + i0, g0 + (1. - fraction) * mass * difference, valid)
        tl.store(GradRates + i1, g1 + fraction * mass * difference, valid)
    if ADD_SKIP:
        result += .5 * tl.load(SkipAdjoint + i, valid, other=0.)
    tl.store(GradMass + i, result, valid)


def _check(*values):
    if any(v.device.type != "cuda" or v.dtype != torch.float64 for v in values):
        raise ValueError("The Triton flux backend requires CUDA float64 tensors.")
    if any(v.device != values[0].device for v in values):
        raise ValueError("All flux tensors must be on the same CUDA device.")


def _rates(inputs, gain, bias, boundary, speed, settings, adjoint=None):
    n, low, noise, leak, competition = settings
    _check(inputs, gain, bias, boundary, speed)
    if gain.ndim != 1 or inputs.shape != (len(gain), 2) or bias.numel() != 1 \
            or boundary.shape != gain.shape or speed.shape != gain.shape:
        raise ValueError("Invalid GPU coefficient shapes.")
    count, block = len(gain), 128
    blocks = triton.cdiv(n * (n + 1), block)
    if adjoint is None:
        rates = gain.new_empty((count, 4, n, n))
        parts = rates  # Unused by the forward specialization.
        adjoint = rates
        reverse = False
    else:
        _check(adjoint)
        if adjoint.shape != (count, 4, n, n):
            raise ValueError("Invalid GPU coefficient adjoint shape.")
        rates = adjoint
        parts = gain.new_empty((count, 2, blocks, 5))
        reverse = True
    _coefficients[(blocks, count, 2)](inputs, gain, bias, boundary, speed, rates, adjoint, parts,
                                    n, low, noise, leak, competition, reverse, block, enable_fp_fusion=False)
    if not reverse:
        return rates
    reduced = parts.sum(dim=2)
    return (reduced[:, :, 0].contiguous(), reduced[:, :, 1].sum(1), reduced[:, :, 2].sum().reshape_as(bias),
            reduced[:, :, 3].sum(1), reduced[:, :, 4].sum(1))


class _Rates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gain, bias, boundary, speed, settings):
        ctx.save_for_backward(inputs, gain, bias, boundary, speed)
        ctx.settings = settings
        return _rates(inputs, gain, bias, boundary, speed, settings)

    @staticmethod
    def backward(ctx, adjoint):
        if torch.is_grad_enabled():
            raise RuntimeError("The Triton coefficient adjoint supports first-order derivatives only.")
        values = ctx.saved_tensors
        with torch.cuda.device(values[0].device):
            return (*_rates(*values, ctx.settings, adjoint.contiguous()), None)


def gpu_rates(inputs, gain, bias, boundary, speed, config):
    values = tuple(v.contiguous() for v in (inputs, gain, bias, boundary, speed))
    _check(*values)
    settings = (config.points, config.lower_bound, config.noise, config.leak, config.competition)
    with torch.cuda.device(inputs.device):
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            return _Rates.apply(*values, settings)
        return _rates(*values, settings)


def _forward(mass, rates, *, substeps, dt):
    n, count, block = mass.shape[0], rates.shape[0] - 1, 128
    blocks = triton.cdiv(n * n, block)
    result, predicted = mass.clone(), torch.empty_like(mass)
    partials = mass.new_empty((count, 3, blocks))
    stage, minimum = mass.new_empty((3, blocks)), mass.new_empty(blocks)
    for k in range(count):
        for j in range(substeps):
            _step[(blocks,)](result, rates, result, predicted, partials, stage, minimum, k, j, substeps,
                             n, dt / substeps, False, True, False, False, block, enable_fp_fusion=False)
            _step[(blocks,)](predicted, rates, result, result, partials, stage, minimum, k, j + 1, substeps,
                             n, dt / substeps, True, True, k == 0 and j == 0, j == 0, block, enable_fp_fusion=False)
    return result, partials.sum(-1), minimum.min()


def _backward(mass, rates, adjoint, adj_exits, *, substeps, dt):
    n, count, block = mass.shape[0], rates.shape[0] - 1, 128
    blocks, total = triton.cdiv(n * n, block), count * substeps
    history = mass.new_empty((2 * total + 1, n, n))
    history[0].copy_(mass)
    for s in range(total):
        k, j = divmod(s, substeps)
        _step[(blocks,)](history[2 * s], rates, history[2 * s], history[2 * s + 1], mass, mass, mass,
                         k, j, substeps, n, dt / substeps, False, False, False, False, block, enable_fp_fusion=False)
        _step[(blocks,)](history[2 * s + 1], rates, history[2 * s], history[2 * s + 2], mass, mass, mass,
                         k, j + 1, substeps, n, dt / substeps, True, False, False, False, block, enable_fp_fusion=False)
    gm, gr, gp = adjoint.clone(), torch.zeros_like(rates), torch.empty_like(mass)
    for s in reversed(range(total)):
        k, j = divmod(s, substeps)
        _vjp[(blocks,)](history[2 * s + 1], rates, gm, adj_exits, gm, gp, gr, k, j + 1, substeps,
                        n, dt / substeps, .5, False, block, enable_fp_fusion=False)
        _vjp[(blocks,)](history[2 * s], rates, gp, adj_exits, gm, gm, gr, k, j, substeps,
                        n, dt / substeps, 1., True, block, enable_fp_fusion=False)
    return gm, gr


_GRAPHS = OrderedDict()
_GRAPH_LOCK = RLock()


def clear_gpu_graph_cache():
    """Release cached execution graphs (for benchmark configuration changes)."""
    with _GRAPH_LOCK:
        _GRAPHS.clear()


def _execute(function, values, substeps, dt, graphs):
    if not graphs:
        return function(*values, substeps=substeps, dt=dt)
    device = values[0].device
    current = torch.cuda.current_stream(device)
    key = (function.__name__, device, current.cuda_stream, tuple(v.shape for v in values), substeps, dt)
    with _GRAPH_LOCK:
        if key not in _GRAPHS:
            static = tuple(torch.empty_like(v) for v in values)
            for target, source in zip(static, values):
                target.copy_(source)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(current)
            with torch.cuda.stream(stream):
                function(*static, substeps=substeps, dt=dt)  # Compile and warm allocations before capture.
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = function(*static, substeps=substeps, dt=dt)
            _GRAPHS[key] = (graph, static, outputs)
            if len(_GRAPHS) > 8:
                _GRAPHS.popitem(last=False)
        _GRAPHS.move_to_end(key)
        graph, static, outputs = _GRAPHS[key]
        for target, source in zip(static, values):
            target.copy_(source)
        graph.replay()
        return tuple(v.clone() for v in outputs)


class _FluxBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mass, rates, substeps, dt, graphs):
        ctx.save_for_backward(mass, rates)
        ctx.substeps, ctx.dt, ctx.graphs = substeps, dt, graphs
        final, flux, minimum = _execute(_forward, (mass, rates), substeps, dt, graphs)
        ctx.mark_non_differentiable(minimum)
        return final, flux, minimum

    @staticmethod
    def backward(ctx, grad_mass, grad_flux, grad_minimum):
        if torch.is_grad_enabled():
            raise RuntimeError("The Triton flux adjoint supports first-order derivatives only.")
        mass, rates = ctx.saved_tensors
        if grad_mass is None:
            grad_mass = torch.zeros_like(mass)
        if grad_flux is None:
            grad_flux = mass.new_zeros((len(rates) - 1, 3))
        with torch.cuda.device(mass.device):
            result = _execute(_backward, (mass, rates, grad_mass.contiguous(), grad_flux.contiguous()),
                              ctx.substeps, ctx.dt, ctx.graphs)
        return *result, None, None, None


def gpu_flux_block(mass, rates, substeps, dt, *, graphs=True):
    _check(mass, rates)
    if mass.ndim != 2 or mass.shape[0] != mass.shape[1] or rates.ndim != 4 \
            or rates.shape[0] < 2 or rates.shape[1:] != (4, *mass.shape):
        raise ValueError("Expected square mass and matching rates[time,4,n,n].")
    if substeps < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("Positive substeps and finite dt required.")
    mass, rates = mass.contiguous(), rates.contiguous()
    with torch.cuda.device(mass.device):
        if torch.is_grad_enabled() and (mass.requires_grad or rates.requires_grad):
            return _FluxBlock.apply(mass, rates, substeps, dt, graphs)
        return _execute(_forward, (mass, rates), substeps, dt, graphs)
