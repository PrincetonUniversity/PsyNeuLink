"""Disposable SymPy code-generation experiment, not a second compiler backend.

Run from the repository root with its Python environment. --check reruns the
existing continuous-phase tests with only equation code generation substituted.
The default benchmark compares both generators using the SAME C++ RK4 runtime.
No production registration, fit default, dependency declaration, or file output
is changed. The Equation adapter is experimental scaffolding, not a proposed
permanent second expression representation.
"""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import statistics
import sys
import time

import sympy as sp
from sympy.printing.cxx import CXX11CodePrinter
import torch

from psyneulink.core.batched.continuous_ir import (
    ContinuousDynamics, analyze_continuous_dynamics, constant, equation_nodes, state,
)
from psyneulink.core.batched.numerical import dynamics as runtime


class Sigmoid(sp.Function):
    """Keep the stable numerical primitive instead of expanding to exp(-x)."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ValueError("Sigmoid has one argument.")
        return self * (1 - self)


class _Printer(CXX11CodePrinter):
    def __init__(self, bindings):
        super().__init__()
        self.bindings = bindings

    def _print_Symbol(self, expr):
        return self.bindings.get(expr, super()._print_Symbol(expr))

    def _print_Sigmoid(self, expr):
        return f"pnl_continuous::sigmoid({self._print(expr.args[0])})"


def _adapt(dynamics):
    """Temporary bridge: check dependencies BEFORE SymPy can cancel expressions."""
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("Expected explicit ContinuousDynamics.")
    report = analyze_continuous_dynamics(dynamics)
    if report.stochastic_states or report.stochastic_readouts or dynamics.latent_inputs:
        raise ValueError("Deterministic code generation cannot discard latent or stochastic dynamics.")
    symbols, bindings = {}, {}
    for kind, axis, names in (("state", "x", dynamics.states), ("input", "u", dynamics.inputs),
                              ("parameter", "p", dynamics.parameters)):
        for index, name in enumerate(names):
            symbol = sp.Symbol(f"_{axis}{index}", real=True)
            symbols[kind, name] = symbol
            bindings[symbol] = f"{axis}[{index}]"
    t = sp.Symbol("_t", real=True)
    bindings[t] = "t"
    values = {}
    expressions = dynamics.drift + tuple(e for _, e in dynamics.readouts)
    operations = dict(add=lambda a, b: a + b, sub=lambda a, b: a - b,
                      mul=lambda a, b: a * b, div=lambda a, b: a / b,
                      neg=lambda a: -a, exp=sp.exp, tanh=sp.tanh, sigmoid=Sigmoid)
    for expr in equation_nodes(expressions):
        if expr.op == "constant":
            value = sp.Float(expr.value, 17)
        elif expr.op == "time":
            value = t
        elif (expr.op, expr.name) in symbols:
            value = symbols[expr.op, expr.name]
        else:
            value = operations[expr.op](*(values[a] for a in expr.arguments))
        values[expr] = value
    return tuple(values[e] for e in expressions), bindings


def _emit(expressions, bindings, name, reverse):
    """Symbolic weighted-output derivatives, explicit CSE, and library printing."""
    printer = _Printer(bindings)
    signature = "const double* x, const double* u, const double* p, double t"
    if reverse:
        adjoints = sp.symbols(f"_adj0:{len(expressions)}", real=True)
        printer.bindings = {**bindings, **{a: f"adj[{i}]" for i, a in enumerate(adjoints)}}
        objective = sp.Add(*(a * e for a, e in zip(adjoints, expressions, strict=True)))
        expressions = tuple(sp.diff(objective, variable) for variable in bindings)
        destinations = tuple("*gt" if value == "t" else "g" + value for value in bindings.values())
        signature += ", const double* adj, double* gx, double* gu, double* gp, double* gt"
    else:
        destinations = tuple(f"output[{i}]" for i in range(len(expressions)))
        signature += ", double* output"
    replacements, reduced = sp.cse(expressions, symbols=sp.numbered_symbols("_v"))
    lines = [f"static inline void {name}({signature}) {{"]
    lines.extend(f"    const double {symbol} = {printer.doprint(expr)};" for symbol, expr in replacements)
    lines.extend(f"    {target} {'+=' if reverse else '='} {printer.doprint(expr)};"
                 for target, expr in zip(destinations, reduced, strict=True) if not reverse or expr != 0)
    return "\n".join([*lines, "}"])


def generate_sympy_source(dynamics):
    expressions, bindings = _adapt(dynamics)
    count = len(dynamics.states)
    methods = [_emit(exprs, bindings, name + suffix, reverse)
               for name, exprs in (("rhs", expressions[:count]), ("readout", expressions[count:]))
               for suffix, reverse in (("", False), ("_vjp", True))]
    return "\n".join([
        '#include "rk4_cpu.h"', 'struct GeneratedDynamics {',
        f'static constexpr int S = {count}, I = {len(dynamics.inputs)}, P = {len(dynamics.parameters)}, R = {len(dynamics.readouts)};',
        *methods, '};', 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {',
        'module.def("forward", &pnl_continuous::forward<GeneratedDynamics>);',
        'module.def("backward", &pnl_continuous::backward<GeneratedDynamics>);', '}',
    ])


@contextmanager
def substitute_generator():
    original = runtime.generate_dynamics_source
    runtime.generate_dynamics_source = generate_sympy_source
    try:
        yield
    finally:
        runtime.generate_dynamics_source = original


def check():
    import pytest

    class Substitute:
        @pytest.fixture(autouse=True)
        def generator(self, monkeypatch):
            monkeypatch.setattr(runtime, "generate_dynamics_source", generate_sympy_source)
            monkeypatch.setattr("psyneulink.core.batched.numerical.dynamics_codegen.generate_dynamics_source", generate_sympy_source)

    # Saturated logistic values/derivatives must not overflow after symbolic AD.
    d = ContinuousDynamics(("x",), (), (), (constant(0),), (("sigmoid", state("x").sigmoid()),))
    with substitute_generator():
        plan = runtime.compile_continuous_phase(d)
    x = torch.tensor([[-1000.], [-20.], [0.], [20.], [1000.]], dtype=torch.float64, requires_grad=True)
    result = plan.integrate(state=x, inputs=x[:, :0], parameters=x[:, :0], duration=x.new_ones(5),
                            start_time=x.new_zeros(5), steps=torch.ones(5, dtype=torch.int64))
    reference = x.sigmoid()
    torch.testing.assert_close(result.readouts[:, 0], reference)
    torch.testing.assert_close(torch.autograd.grad(result.readouts.sum(), x)[0], reference * (1 - reference))
    return pytest.main(["-n", "0", "-q", "tests/composition/pec/test_batched_continuous_dynamics.py"], plugins=[Substitute()])


def benchmark(args):
    from benchmark_continuous_dynamics import csi_equations

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    d = csi_equations()
    plans, generation, build, source_sizes = {}, {}, {}, {}
    for name in ("custom", "sympy"):
        start = time.perf_counter()
        if name == "sympy":
            with substitute_generator():
                plans[name] = runtime.compile_continuous_phase(d)
        else:
            plans[name] = runtime.compile_continuous_phase(d)
        generation[name] = time.perf_counter() - start
        source_sizes[name] = dict(bytes=len(plans[name].source.encode()), lines=len(plans[name].source.splitlines()))
        start = time.perf_counter()
        plans[name]._module()
        build[name] = time.perf_counter() - start
    x = torch.linspace(-.06, .04, args.trials * 2, dtype=torch.float64).reshape(args.trials, 2).requires_grad_()
    u = torch.tensor([[1., 0., 1., 0., .2, .8, 1.], [0., 1., .2, .8, .9, .1, -1.]], dtype=torch.float64)
    u = u.repeat((args.trials + 1) // 2, 1)[:args.trials].contiguous().requires_grad_()
    p = torch.linspace(9, 24, args.trials, dtype=torch.float64)[:, None].requires_grad_()
    inputs = dict(state=x, inputs=u, parameters=p,
                  duration=torch.full((args.trials,), args.steps * .001, dtype=torch.float64),
                  start_time=torch.zeros(args.trials, dtype=torch.float64),
                  steps=torch.full((args.trials,), args.steps, dtype=torch.int64))

    def run(name, grad):
        begin = time.perf_counter()
        with torch.set_grad_enabled(grad):
            result = plans[name].integrate(**inputs)
            values = result.readouts, result.final_state
            gradients = torch.autograd.grad(sum(v.sum() for v in values), (x, u, p)) if grad else ()
        return time.perf_counter() - begin, tuple(v.detach() for v in (*values, *gradients))

    output = dict(trials=args.trials, steps=args.steps, threads=args.threads, dt=.001, dtype="float64",
                  sympy_version=sp.__version__, torch_version=torch.__version__, generation_seconds=generation,
                  build_or_cache_load_seconds=build, generated_source_including_header=source_sizes,
                  scope="equation generation and deterministic phase only, not subject fitting")
    for grad in (False, True):
        timings = {name: [] for name in plans}
        for name in plans:
            for _ in range(2):
                run(name, grad)
        last = {}
        for repetition in range(args.repeats):
            for name in (tuple(plans) if repetition % 2 == 0 else tuple(reversed(plans))):
                elapsed, last[name] = run(name, grad)
                timings[name].append(elapsed)
        differences = []
        for actual, reference in zip(last["sympy"], last["custom"], strict=True):
            torch.testing.assert_close(actual, reference, rtol=2e-9, atol=2e-10)
            differences.append(float((actual - reference).abs().max()))
        output["gradient" if grad else "forward"] = dict(
            seconds={name: statistics.median(runs) for name, runs in timings.items()},
            runs=timings, max_absolute_differences=differences,
        )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--trials", type=int, default=480)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    if min(args.trials, args.steps, args.threads, args.repeats) <= 0:
        parser.error("All counts must be positive.")
    if not Path("psyneulink/core/batched").is_dir():
        parser.error("Run from the repository root.")
    sys.exit(check() if args.check else benchmark(args))
