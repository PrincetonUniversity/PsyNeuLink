"""Native CPU finite-volume coefficients, absorbing time loop, and adjoints."""

from functools import lru_cache
import hashlib
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def module():
    from torch.utils.cpp_extension import load_inline
    source = Path(__file__).with_name("continuous_flux_cpu.cpp").read_text()
    return load_inline(name="pnl_absorbing_rk2_" + hashlib.sha256(source.encode()).hexdigest()[:16],
                       cpp_sources=source, extra_cflags=["-O3", "-DNDEBUG", "-fopenmp"],
                       extra_ldflags=["-fopenmp"], with_cuda=False, verbose=False)


class _FluxBlock(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mass, rates, substeps, dt, threads):
        ctx.save_for_backward(mass, rates)
        ctx.substeps, ctx.dt, ctx.threads = substeps, dt, threads
        final, flux, minimum = module().forward(mass, rates, substeps, dt, threads)
        ctx.mark_non_differentiable(minimum)
        return final, flux, minimum

    @staticmethod
    def backward(ctx, grad_mass, grad_flux, grad_minimum):
        if torch.is_grad_enabled():
            raise RuntimeError("The native flux adjoint supports first-order derivatives only.")
        mass, rates = ctx.saved_tensors
        if grad_mass is None:
            grad_mass = torch.zeros_like(mass)
        if grad_flux is None:
            grad_flux = mass.new_zeros((len(rates) - 1, 3))
        gm, gr = module().backward(mass, rates, grad_mass.contiguous(), grad_flux.contiguous(), ctx.substeps, ctx.dt, ctx.threads)
        return gm, gr, None, None, None


def native_flux_block(mass, rates, substeps, dt, threads=1):
    if mass.device.type != "cpu" or mass.dtype != torch.float64:
        raise ValueError("The native flux backend requires CPU float64 tensors.")
    mass, rates = mass.contiguous(), rates.contiguous()
    if torch.is_grad_enabled() and (mass.requires_grad or rates.requires_grad):
        return _FluxBlock.apply(mass, rates, substeps, dt, threads)
    return tuple(module().forward(mass, rates, substeps, dt, threads))


class _Rates(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, gain, bias, boundary, boundary_rate, settings):
        ctx.save_for_backward(inputs, gain, bias, boundary, boundary_rate)
        ctx.settings = settings
        return module().coefficients(inputs, gain, bias, boundary, boundary_rate, *settings)

    @staticmethod
    def backward(ctx, adjoint):
        if torch.is_grad_enabled():
            raise RuntimeError("The native coefficient adjoint supports first-order derivatives only.")
        result = module().coefficients_vjp(*ctx.saved_tensors, *ctx.settings, adjoint.contiguous())
        return (*result, None)


def native_rates(inputs, gain, bias, boundary, boundary_rate, config):
    values = tuple(v.contiguous() for v in (inputs, gain, bias, boundary, boundary_rate))
    settings = (config.points, config.lower_bound, config.noise, config.leak, config.competition, config.cpu_threads)
    if torch.is_grad_enabled() and any(v.requires_grad for v in values):
        return _Rates.apply(*values, settings)
    return module().coefficients(*values, *settings)
