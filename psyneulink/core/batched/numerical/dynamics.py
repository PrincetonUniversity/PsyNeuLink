"""Generated deterministic continuous phases with a reusable native RK4 adjoint.

This API is below Composition admission: explicit initial states, conditioned
inputs, physical phase start times, durations and integration counts are supplied
by the caller. It does not implement trial resets or event/history inference.
"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import torch
import sympy as sp

from psyneulink.core.batched.continuous_ir import ContinuousDynamics, analyze_continuous_dynamics
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.numerical.dynamics_codegen import generate_dynamics_source


@lru_cache(maxsize=32)
def _load_module(source, header_digest):
    from torch.utils.cpp_extension import load_inline

    digest = hashlib.sha256((source + header_digest).encode()).hexdigest()[:20]
    return load_inline(
        name="pnl_continuous_" + digest, cpp_sources=source,
        extra_include_paths=[str(Path(__file__).resolve().parent)],
        extra_cflags=["-O3", "-DNDEBUG", "-fopenmp"], extra_ldflags=["-fopenmp"],
        with_cuda=False, verbose=False,
    )


@dataclass(frozen=True)
class ContinuousPhaseResult:
    readouts: torch.Tensor
    final_state: torch.Tensor


class _PhaseAdjoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, inputs, parameters, duration, start_time, steps, plan):
        module = plan._module()
        readouts, final_state, history = module.forward(state, inputs, parameters, duration, start_time, steps, True)
        ctx.module = module
        ctx.readout_shape = readouts.shape
        ctx.save_for_backward(history, state, inputs, parameters, duration, start_time, steps)
        return readouts, final_state

    @staticmethod
    def backward(ctx, grad_readouts, grad_final):
        if torch.is_grad_enabled():
            raise RuntimeError("Generated phase adjoints support first-order derivatives only; use create_graph=False.")
        history, state, inputs, parameters, duration, start_time, steps = ctx.saved_tensors
        if grad_readouts is None:
            grad_readouts = state.new_zeros(ctx.readout_shape)
        if grad_final is None:
            grad_final = torch.zeros_like(state)
        gradients = tuple(ctx.module.backward(history, state, inputs, parameters, duration, start_time, steps,
                                              grad_readouts.contiguous(), grad_final.contiguous()))
        if any(not bool(torch.isfinite(g).all()) for g in gradients):
            raise FloatingPointError("Nonfinite generated dynamics adjoint; check equation domains and integration resolution.")
        return (*gradients, None, None)


@dataclass(frozen=True)
class CompiledContinuousPhase:
    dynamics: ContinuousDynamics
    source: str
    header_digest: str

    def _module(self):
        return _load_module(self.source, self.header_digest)

    def explain(self):
        return dict(
            backend="cpp_cpu", process="explicit_continuous_equations",
            symbolic_backend=f"sympy {sp.__version__}",
            states=tuple(map(str, self.dynamics.states)), inputs=tuple(map(str, self.dynamics.inputs)),
            parameters=tuple(map(str, self.dynamics.parameters)),
            readouts=tuple(n for n, _ in self.dynamics.readouts),
            clock="physical seconds; explicit start_time and duration per lane",
            integrator="two RK4 half-steps per cell, midpoint readouts",
            gradient="generated equation VJPs plus reusable first-order discrete RK4 adjoint",
            gradient_inputs=("state", "inputs", "parameters", "duration", "start_time"),
            limitations=("No inferred source/continuous equivalence, resets, stopping events, or observed-history conditioning.",
                         "Domains/dependencies apply to retained expressions; caller simplifications cannot be reversed.",
                         "Inputs and parameters are constant within a phase; step counts are nondifferentiable.",
                         "Zero-count phases require zero duration, preserve state, and return zero-padded readouts.",
                         "No formal integration error certificate or positivity guarantee."),
        )

    def integrate(self, *, state, inputs, parameters, duration, start_time, steps):
        if not isinstance(state, torch.Tensor) or state.ndim != 2:
            raise ValueError("state must have shape [batch, state_dimension].")
        batch = state.shape[0]
        values = (state, inputs, parameters, duration, start_time)
        shapes = ((batch, len(self.dynamics.states)), (batch, len(self.dynamics.inputs)),
                  (batch, len(self.dynamics.parameters)), (batch,), (batch,))
        for value, shape in zip(values, shapes, strict=True):
            if (not isinstance(value, torch.Tensor) or value.shape != shape
                    or value.dtype != torch.float64 or value.device.type != "cpu"):
                raise ValueError(f"Expected CPU float64 tensor of shape {shape}.")
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Continuous phase inputs must be finite.")
        if (not isinstance(steps, torch.Tensor) or steps.shape != (batch,)
                or steps.device.type != "cpu" or steps.dtype != torch.int64):
            raise ValueError("steps must be a CPU int64 vector with one count per lane.")
        if bool((steps < 0).any()) or bool((duration < 0).any()) or bool(((steps == 0) & (duration != 0)).any()):
            raise ValueError("Nonnegative steps/durations are required; zero steps require zero duration.")
        # A contiguous batch copy is allowed; never a host callback per time cell.
        values = tuple(v.contiguous() for v in values)
        steps = steps.contiguous()
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            readouts, final = _PhaseAdjoint.apply(*values, steps, self)
        else:
            readouts, final, _ = self._module().forward(*values, steps, False)
        if not bool(torch.isfinite(readouts).all()) or not bool(torch.isfinite(final).all()):
            raise FloatingPointError("Nonfinite generated dynamics; check equation domains and integration resolution.")
        return ContinuousPhaseResult(readouts, final)


def compile_continuous_phase(dynamics):
    """Generate a deterministic phase from a checked equation schema, not a PNL graph."""
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("dynamics must be ContinuousDynamics.")
    report = analyze_continuous_dynamics(dynamics)
    if report.stochastic_states or report.stochastic_readouts or dynamics.latent_inputs:
        raise LikelihoodPlanningError("continuous.stochastic_phase", "The deterministic RK4 backend cannot ignore diffusion or unresolved latent history/inputs.")
    expressions = dynamics.drift + tuple(e for _, e in dynamics.readouts)
    if len(dynamics.states) > 256 or len(set().union(*(sp.preorder_traversal(e) for e in expressions))) > 4096:
        raise LikelihoodPlanningError("continuous.codegen_size", "This backend supports at most 256 state coordinates and 4096 equation DAG nodes.")
    # Freeze the integrator body too: later template edits must not silently
    # change the semantics of an already-created phase plan before first use.
    header = Path(__file__).with_name("rk4_cpu.h").read_text()
    source = generate_dynamics_source(dynamics).replace('#include "rk4_cpu.h"', header)
    header_digest = hashlib.sha256(header.encode()).hexdigest()
    return CompiledContinuousPhase(dynamics, source, header_digest)
