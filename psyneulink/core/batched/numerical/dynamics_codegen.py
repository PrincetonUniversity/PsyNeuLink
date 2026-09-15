"""Library symbolic differentiation/CSE/printing; reusable C++ RK4 adjoint.

Retained denominator domains are checked BEFORE optimized equations execute.
Source endpoint arithmetic is deliberately outside this mathematical backend.
"""

from functools import lru_cache

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
from sympy.printing.pytorch import TorchPrinter
import torch

from psyneulink.core.batched.continuous_ir import ContinuousDynamics, analyze_continuous_dynamics


class _Printer(CXX11CodePrinter):
    def __init__(self, bindings):
        super().__init__()
        self.bindings = bindings

    def _print_Symbol(self, expr):
        return self.bindings.get(expr, super()._print_Symbol(expr))

    _print_Dummy = _print_Symbol

    def _print_Sigmoid(self, expr):
        return f"pnl_continuous::sigmoid({self._print(expr.args[0])})"


def _denominators(expressions):
    return tuple(sorted({n.base for e in expressions for n in sp.preorder_traversal(e)
                         if n.func == sp.Pow and n.exp < 0}, key=sp.default_sort_key))


def _function(dynamics, expressions, name, reverse):
    bindings = {s: f"{axis}[{i}]" for axis, symbols in (("x", dynamics.states), ("u", dynamics.inputs), ("p", dynamics.parameters))
                for i, s in enumerate(symbols)}
    bindings[dynamics.time] = "t"
    # Capture domains from ORIGINAL expressions, including canceled divisions.
    denominators = _denominators(expressions)
    printer = _Printer(bindings)
    signature = "const double* x, const double* u, const double* p, double t"
    if reverse:
        adjoints = tuple(sp.Dummy(real=True) for _ in expressions)
        printer.bindings = {**bindings, **{a: f"adj[{i}]" for i, a in enumerate(adjoints)}}
        objective = sp.Add(*(a * e for a, e in zip(adjoints, expressions, strict=True)))
        expressions = tuple(sp.diff(objective, s) for s in bindings)
        targets = tuple("*gt" if v == "t" else "g" + v for v in bindings.values())
        signature += ", const double* adj, double* gx, double* gu, double* gp, double* gt"
    else:
        targets = tuple(f"output[{i}]" for i in range(len(expressions)))
        signature += ", double* output"
    lines = [f"static inline void {name}({signature}) {{"]
    for i, denominator in enumerate(denominators):
        lines += [f"    const double _domain{i} = {printer.doprint(denominator)};",
                  f"    if (!std::isfinite(_domain{i}) || _domain{i} == 0.) {{",
                  *(f"        {target} = NAN;" for target in targets), "        return;", "    }"]
    # Safe compiler symbols prevent names/Dummy counters entering cache keys.
    safe = {s: sp.Symbol(f"_arg{i}", real=True) for i, s in enumerate(printer.bindings)}
    printer.bindings = {safe[s]: v for s, v in printer.bindings.items()}
    replacements, reduced = sp.cse(tuple(e.xreplace(safe) for e in expressions), symbols=sp.numbered_symbols("_v"))
    lines.extend(f"    const double {s} = {printer.doprint(e)};" for s, e in replacements)
    lines.extend(f"    {target} {'+=' if reverse else '='} {printer.doprint(e)};"
                 for target, e in zip(targets, reduced, strict=True) if not reverse or e != 0)
    return "\n".join([*lines, "}"])


def generate_dynamics_source(dynamics):
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("Expected ContinuousDynamics.")
    report = analyze_continuous_dynamics(dynamics)
    if report.stochastic_states or report.stochastic_readouts or dynamics.latent_inputs:
        raise ValueError("Deterministic code generation cannot discard latent or stochastic equations.")
    methods = [_function(dynamics, exprs, name + suffix, reverse)
               for name, exprs in (("rhs", dynamics.drift), ("readout", tuple(e for _, e in dynamics.readouts)))
               for suffix, reverse in (("", False), ("_vjp", True))]
    return "\n".join([
        '#include "rk4_cpu.h"', 'struct GeneratedDynamics {',
        f'static constexpr int S = {len(dynamics.states)}, I = {len(dynamics.inputs)}, P = {len(dynamics.parameters)}, R = {len(dynamics.readouts)};',
        *methods, '};', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {',
        'module.def("forward", &pnl_continuous::forward<GeneratedDynamics>);',
        'module.def("backward", &pnl_continuous::backward<GeneratedDynamics>);', '}',
    ])


class _TorchPrinter(TorchPrinter):
    def _print_Sigmoid(self, expr):
        return "{}({}({}, dtype={}))".format(
            self._module_format("torch.sigmoid"), self._module_format("torch.as_tensor"),
            self._print(expr.args[0]), self._module_format("torch.float64"))


@lru_cache(maxsize=32)
def _torch_equations(expressions, symbols):
    return sp.lambdify(symbols, expressions, modules="torch", printer=_TorchPrinter(), dummify=True, cse=False)


def evaluate_equations(expressions, dynamics, state, inputs, parameters, time):
    """Library-printed Torch oracle; not the production time loop."""
    symbols = dynamics.states + dynamics.inputs + dynamics.parameters + (dynamics.time,)
    values = (*state.unbind(-1), *inputs.unbind(-1), *parameters.unbind(-1), time)
    domains = _denominators(expressions)
    result = _torch_equations(tuple(expressions) + domains, symbols)(*values)
    result = [torch.as_tensor(v, dtype=state.dtype, device=state.device) + torch.zeros_like(state[..., 0]) for v in result]
    valid = torch.ones_like(state[..., 0], dtype=torch.bool)
    for value in result[len(expressions):]:
        valid = valid & torch.isfinite(value) & (value != 0)
    return torch.where(valid[..., None], torch.stack(result[:len(expressions)], dim=-1), float("nan")) if expressions else state[..., :0]
