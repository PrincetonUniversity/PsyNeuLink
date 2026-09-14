"""Explicit continuous equations, independent of source schedulers and solvers.

These are caller-supplied mathematical declarations, not equations inferred from
a simulator body. Input values are constant during one integration phase. Clock
time is physical time in seconds; its origin and the phase duration are explicit
runtime inputs, never inferred from scheduler pass counts or a source Euler dt.
"""

from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class Equation:
    """Small scalar expression DAG with explicit state/input/parameter bindings."""

    op: str
    arguments: tuple["Equation", ...] = ()
    name: str = ""
    value: float = 0.

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if type(self.name) is not str:
            raise TypeError("Equation binding names must be strings.")
        arities = {"constant": 0, "state": 0, "input": 0, "parameter": 0, "time": 0,
                   "add": 2, "sub": 2, "mul": 2, "div": 2, "neg": 1, "exp": 1, "tanh": 1, "sigmoid": 1}
        if self.op not in arities or len(self.arguments) != arities[self.op]:
            raise ValueError("Unknown equation operation or incorrect arity.")
        if any(type(a) is not Equation for a in self.arguments):
            raise TypeError("Equation arguments must be Equation instances.")
        if self.op in ("state", "input", "parameter"):
            if type(self.name) is not str or not self.name:
                raise ValueError("Equation bindings require a nonempty name.")
        elif self.name:
            raise ValueError("Only binding expressions carry names.")
        if isinstance(self.value, bool) or not isinstance(self.value, Real) or not math.isfinite(self.value):
            raise ValueError("Equation constants must be finite real numbers.")
        if self.op != "constant" and self.value != 0:
            raise ValueError("Only constant expressions carry values.")
        object.__setattr__(self, "value", float(self.value))

    def __add__(self, other):
        return Equation("add", (self, constant(other)))

    def __radd__(self, other):
        return constant(other) + self

    def __sub__(self, other):
        return Equation("sub", (self, constant(other)))

    def __rsub__(self, other):
        return constant(other) - self

    def __mul__(self, other):
        return Equation("mul", (self, constant(other)))

    def __rmul__(self, other):
        return constant(other) * self

    def __truediv__(self, other):
        return Equation("div", (self, constant(other)))

    def __rtruediv__(self, other):
        return constant(other) / self

    def __neg__(self):
        return Equation("neg", (self,))

    def exp(self):
        return Equation("exp", (self,))

    def tanh(self):
        return Equation("tanh", (self,))

    def sigmoid(self):
        return Equation("sigmoid", (self,))


def constant(value):
    return value if type(value) is Equation else Equation("constant", value=value)


def state(name):
    return Equation("state", name=name)


def parameter(name):
    return Equation("parameter", name=name)


def input_value(name):
    return Equation("input", name=name)


def physical_time():
    return Equation("time")


def equation_nodes(expressions):
    """Deterministic topological order, sharing common subexpressions."""
    nodes, seen = [], set()

    def visit(expr):
        if expr not in seen:
            for arg in expr.arguments:
                visit(arg)
            seen.add(expr)
            nodes.append(expr)

    for expr in expressions:
        visit(expr)
    return tuple(nodes)


@dataclass(frozen=True)
class DiffusionTerm:
    """An Itô coefficient multiplying one named independent Wiener driver.

    Reusing a driver name represents shared noise, not independent draws.
    Coefficients are standard-deviation amplitudes, not variances.
    """

    state: str
    driver: str
    coefficient: Equation

    def __post_init__(self):
        if any(type(n) is not str or not n for n in (self.state, self.driver)):
            raise ValueError("Diffusion state and driver names must be nonempty strings.")
        if type(self.coefficient) is not Equation:
            raise TypeError("Diffusion coefficients must be Equation expressions.")


@dataclass(frozen=True)
class ContinuousDynamics:
    """One continuous phase, with explicit full state and algebraic readouts.

    Drift entries follow states order. Diffusion is sparse in state/driver axes.
    Initial states and phase times are supplied by the caller; no reset, stopping
    event, phase schedule, or cross-trial conditioning is implicitly performed.
    latent_inputs/latent_initial_states record unresolved randomness for analysis.
    """

    states: tuple[str, ...]
    inputs: tuple[str, ...]
    parameters: tuple[str, ...]
    drift: tuple[Equation, ...]
    readouts: tuple[tuple[str, Equation], ...] = ()
    diffusion: tuple[DiffusionTerm, ...] = ()
    latent_inputs: tuple[str, ...] = ()
    latent_initial_states: tuple[str, ...] = ()
    clock_unit: str = "seconds"

    def __post_init__(self):
        for name in ("states", "inputs", "parameters", "drift", "diffusion", "latent_inputs", "latent_initial_states"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "readouts", tuple(tuple(pair) for pair in self.readouts))
        for values in (self.states, self.inputs, self.parameters, self.latent_inputs, self.latent_initial_states):
            if any(type(n) is not str or not n for n in values) or len(set(values)) != len(values):
                raise ValueError("Binding names must be nonempty unique strings within each axis.")
        if not self.states or len(self.drift) != len(self.states):
            raise ValueError("One drift expression is required per state.")
        if type(self.clock_unit) is not str or self.clock_unit != "seconds":
            raise ValueError("Only explicit physical seconds are supported; scheduler clocks require a separate interpretation.")
        if not set(self.latent_inputs) <= set(self.inputs) or not set(self.latent_initial_states) <= set(self.states):
            raise ValueError("Latent bindings must refer to declared inputs/states.")
        if any(len(pair) != 2 or type(pair[0]) is not str or not pair[0] for pair in self.readouts):
            raise ValueError("Readouts require (name, expression) pairs.")
        if len({pair[0] for pair in self.readouts}) != len(self.readouts):
            raise ValueError("Readout names must be unique.")
        pairs = set()
        for term in self.diffusion:
            if (type(term) is not DiffusionTerm or term.state not in self.states
                    or type(term.driver) is not str or not term.driver or (term.state, term.driver) in pairs):
                raise ValueError("Diffusion requires unique declared state/driver pairs.")
            pairs.add((term.state, term.driver))
        expressions = self.drift + tuple(e for _, e in self.readouts) + tuple(t.coefficient for t in self.diffusion)
        if any(type(e) is not Equation for e in expressions):
            raise TypeError("Dynamics must contain Equation expressions.")
        bindings = {"state": self.states, "input": self.inputs, "parameter": self.parameters}
        for expr in equation_nodes(expressions):
            if expr.op in bindings and expr.name not in bindings[expr.op]:
                raise ValueError(f"Unbound {expr.op}: {expr.name}.")


@dataclass(frozen=True)
class ContinuousDependencyReport:
    deterministic_states: tuple[str, ...]
    stochastic_states: tuple[str, ...]
    stochastic_readouts: tuple[str, ...]
    noise_drivers: tuple[str, ...]
    state_dependencies: tuple[tuple[str, tuple[str, ...]], ...]
    guarantee: str = "conservative_dependency_closure_of_declared_equations"


def analyze_continuous_dynamics(dynamics):
    """Propagate randomness through dynamics, not just through noise coefficients.

    This is a conservative closure, not minimal-state discovery, a density-rank
    proof, or permission to factor a sequential likelihood. Algebraic cancellation
    is deliberately not assumed (even 0*x retains a dependency on x).
    """
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("dynamics must be ContinuousDynamics.")

    def dependencies(expr):
        return {(e.op, e.name) for e in equation_nodes((expr,)) if e.op in ("state", "input")}

    state_deps = {name: dependencies(expr) for name, expr in zip(dynamics.states, dynamics.drift, strict=True)}
    random_states = set(dynamics.latent_initial_states)
    drivers = set()
    for term in dynamics.diffusion:
        state_deps[term.state].update(dependencies(term.coefficient))
        if term.coefficient != constant(0):
            random_states.add(term.state)
            drivers.add(term.driver)
    random_refs = {("input", n) for n in dynamics.latent_inputs} | {("state", n) for n in random_states}
    changed = True
    while changed:
        new = {("state", n) for n, deps in state_deps.items() if deps & random_refs}
        changed = not new <= random_refs
        random_refs |= new
    stochastic = tuple(n for n in dynamics.states if ("state", n) in random_refs)
    return ContinuousDependencyReport(
        tuple(n for n in dynamics.states if n not in stochastic), stochastic,
        tuple(n for n, expr in dynamics.readouts if dependencies(expr) & random_refs),
        tuple(sorted(drivers)),
        tuple((n, tuple(s for s in dynamics.states if ("state", s) in deps)) for n, deps in state_deps.items()),
    )


@dataclass(frozen=True)
class DeterministicSubsystem:
    """Equation-level dependency slice, not a cross-trial likelihood witness."""

    dynamics: ContinuousDynamics
    state_indices: tuple[int, ...]
    input_indices: tuple[int, ...]
    parameter_indices: tuple[int, ...]
    readout_indices: tuple[int, ...]
    guarantee: str = "checked_dependency_slice_of_declared_continuous_equations"


def extract_deterministic_subsystem(dynamics):
    """Automatically retain the closed deterministic upstream equation block.

    Feedback from stochastic states excludes affected coordinates. Unused latent
    inputs are dropped, not relabeled as observed. Known initial states must
    still be supplied, and stopping-time/recorded-history uncertainty is outside
    this phase-local reduction. None means there are no deterministic states.
    """
    report = analyze_continuous_dynamics(dynamics)
    if not report.deterministic_states:
        return None
    states = tuple(i for i, n in enumerate(dynamics.states) if n in report.deterministic_states)
    readouts = tuple(i for i, (n, _) in enumerate(dynamics.readouts) if n not in report.stochastic_readouts)
    drift = tuple(dynamics.drift[i] for i in states)
    selected_readouts = tuple(dynamics.readouts[i] for i in readouts)
    refs = {(e.op, e.name) for e in equation_nodes(drift + tuple(e for _, e in selected_readouts))}
    inputs = tuple(i for i, n in enumerate(dynamics.inputs) if ("input", n) in refs)
    parameters = tuple(i for i, n in enumerate(dynamics.parameters) if ("parameter", n) in refs)
    reduced = ContinuousDynamics(
        states=tuple(dynamics.states[i] for i in states), inputs=tuple(dynamics.inputs[i] for i in inputs),
        parameters=tuple(dynamics.parameters[i] for i in parameters), drift=drift, readouts=selected_readouts,
    )
    return DeterministicSubsystem(reduced, states, inputs, parameters, readouts)
