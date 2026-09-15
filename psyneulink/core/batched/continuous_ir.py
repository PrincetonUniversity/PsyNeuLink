"""Continuous phase metadata over SymPy expressions, not a second algebra DSL.

Use ``with sympy.evaluate(False)`` when constructing equations to retain source
dependencies and denominator domains. Analysis precedes algebraic optimization;
information already erased by a caller's simplification cannot be recovered.
These declarations are not an inferred PNL scheduler or history contract.
"""

from dataclasses import dataclass
import math

import sympy as sp


class Sigmoid(sp.Function):
    """Stable numerical primitive with a symbolic derivative, not exp expansion."""

    nargs = 1

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ValueError("Sigmoid has one argument.")
        return self * (1 - self)


def validate_equations(expressions, symbols):
    """Shared scalar algebra admission for primitive rules and phase equations."""
    if any(not isinstance(s, sp.Symbol) or s.is_real is not True for s in symbols):
        raise ValueError("Equation bindings must be real SymPy symbols.")
    for expr in expressions:
        if not isinstance(expr, sp.Expr):
            raise TypeError("Equations must be SymPy expressions; use SymPy numbers for constants.")
        if not expr.free_symbols <= set(symbols):
            raise ValueError("Unbound symbols in continuous equation.")
        for node in sp.preorder_traversal(expr):
            if isinstance(node, sp.Number):
                if node.is_finite is not True or node.is_real is not True or not math.isfinite(float(node)):
                    raise ValueError("Equation constants must be finite and real.")
            elif not isinstance(node, sp.Symbol) and node.func not in (sp.Add, sp.Mul, sp.Pow, sp.exp, sp.tanh, Sigmoid):
                raise ValueError("Unsupported symbolic equation operation.")
            if node.func == sp.Pow and not isinstance(node.exp, sp.Integer):
                raise ValueError("Only integer powers are supported; other powers need additional domain rules.")


@dataclass(frozen=True)
class DiffusionTerm:
    """Itô amplitude on a state; equal driver names denote shared noise."""

    state: sp.Symbol
    driver: str
    coefficient: sp.Expr

    def __post_init__(self):
        if not isinstance(self.state, sp.Symbol) or type(self.driver) is not str or not self.driver:
            raise ValueError("Diffusion requires a state symbol and nonempty driver name.")
        if not isinstance(self.coefficient, sp.Expr):
            raise TypeError("Diffusion coefficients must be SymPy expressions.")


@dataclass(frozen=True)
class ContinuousDynamics:
    """One phase; symbols bind to tensor columns, never executable identifiers.

    Initial states, times, and durations are caller supplied. No reset, event,
    schedule, or history conditioning is implicit. Retain unevaluated equations
    for conservative dependency/domain analysis; optimization is backend-local.
    """

    states: tuple[sp.Symbol, ...]
    inputs: tuple[sp.Symbol, ...]
    parameters: tuple[sp.Symbol, ...]
    drift: tuple[sp.Expr, ...]
    readouts: tuple[tuple[str, sp.Expr], ...] = ()
    diffusion: tuple[DiffusionTerm, ...] = ()
    latent_inputs: tuple[sp.Symbol, ...] = ()
    latent_initial_states: tuple[sp.Symbol, ...] = ()
    time: sp.Symbol = sp.Symbol("t", real=True)
    clock_unit: str = "seconds"

    def __post_init__(self):
        for name in ("states", "inputs", "parameters", "drift", "diffusion", "latent_inputs", "latent_initial_states"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "readouts", tuple(tuple(pair) for pair in self.readouts))
        symbols = self.states + self.inputs + self.parameters + (self.time,)
        if any(not isinstance(s, sp.Symbol) or s.is_real is not True for s in symbols) or len(set(symbols)) != len(symbols):
            raise ValueError("Bindings require distinct real SymPy symbols across all axes and time.")
        if not self.states or len(self.drift) != len(self.states):
            raise ValueError("One drift expression is required per state.")
        if self.clock_unit != "seconds":
            raise ValueError("Only physical seconds are supported; scheduler clocks need an explicit interpretation.")
        if not set(self.latent_inputs) <= set(self.inputs) or not set(self.latent_initial_states) <= set(self.states):
            raise ValueError("Latent bindings must refer to declared inputs/states.")
        if any(len(p) != 2 or type(p[0]) is not str or not p[0] for p in self.readouts):
            raise ValueError("Readouts require (name, expression) pairs.")
        if len({n for n, _ in self.readouts}) != len(self.readouts):
            raise ValueError("Readout names must be unique.")
        pairs = set()
        for term in self.diffusion:
            if type(term) is not DiffusionTerm or term.state not in self.states or (term.state, term.driver) in pairs:
                raise ValueError("Diffusion requires unique declared state/driver pairs.")
            pairs.add((term.state, term.driver))
        validate_equations(self.drift + tuple(e for _, e in self.readouts) + tuple(d.coefficient for d in self.diffusion), symbols)


@dataclass(frozen=True)
class ContinuousDependencyReport:
    deterministic_states: tuple[sp.Symbol, ...]
    stochastic_states: tuple[sp.Symbol, ...]
    stochastic_readouts: tuple[str, ...]
    noise_drivers: tuple[str, ...]
    state_dependencies: tuple[tuple[sp.Symbol, tuple[sp.Symbol, ...]], ...]
    guarantee: str = "conservative_dependency_closure_of_declared_equations"


def analyze_continuous_dynamics(dynamics):
    """Conservative closure over retained expressions, NOT simplified equations.

    Does not infer minimal stochastic dimension or sequential factorization.
    Already simplified caller expressions do not retain their former effects.
    """
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("dynamics must be ContinuousDynamics.")
    deps = {s: set(e.free_symbols) for s, e in zip(dynamics.states, dynamics.drift, strict=True)}
    random = set(dynamics.latent_initial_states) | set(dynamics.latent_inputs)
    drivers = set()
    for term in dynamics.diffusion:
        deps[term.state].update(term.coefficient.free_symbols)
        if not (term.coefficient.is_Number and term.coefficient.is_zero):
            random.add(term.state)
            drivers.add(term.driver)
    while True:
        new = {s for s, dependencies in deps.items() if dependencies & random} - random
        if not new:
            break
        random |= new
    return ContinuousDependencyReport(
        tuple(s for s in dynamics.states if s not in random), tuple(s for s in dynamics.states if s in random),
        tuple(n for n, e in dynamics.readouts if e.free_symbols & random), tuple(sorted(drivers)),
        tuple((s, tuple(v for v in dynamics.states if v in deps[s])) for s in dynamics.states),
    )


@dataclass(frozen=True)
class DeterministicSubsystem:
    dynamics: ContinuousDynamics
    state_indices: tuple[int, ...]
    input_indices: tuple[int, ...]
    parameter_indices: tuple[int, ...]
    readout_indices: tuple[int, ...]
    guarantee: str = "checked_dependency_slice_of_declared_continuous_equations"


def extract_deterministic_subsystem(dynamics):
    """Retain a closed deterministic equation slice, not a history-replay proof."""
    report = analyze_continuous_dynamics(dynamics)
    if not report.deterministic_states:
        return None
    states = tuple(i for i, s in enumerate(dynamics.states) if s in report.deterministic_states)
    readouts = tuple(i for i, (n, _) in enumerate(dynamics.readouts) if n not in report.stochastic_readouts)
    drift = tuple(dynamics.drift[i] for i in states)
    selected = tuple(dynamics.readouts[i] for i in readouts)
    refs = set().union(*(e.free_symbols for e in drift + tuple(e for _, e in selected)))
    inputs = tuple(i for i, s in enumerate(dynamics.inputs) if s in refs)
    parameters = tuple(i for i, s in enumerate(dynamics.parameters) if s in refs)
    reduced = ContinuousDynamics(
        tuple(dynamics.states[i] for i in states), tuple(dynamics.inputs[i] for i in inputs),
        tuple(dynamics.parameters[i] for i in parameters), drift, selected, time=dynamics.time,
    )
    return DeterministicSubsystem(reduced, states, inputs, parameters, readouts)
