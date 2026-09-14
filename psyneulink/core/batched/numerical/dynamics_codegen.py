"""Generate fused scalar equation DAGs and reverse derivatives for C++ RK4.

Only the closed Equation operation set is emitted; caller-supplied names never
become C++ identifiers or code. Model derivatives are generated, not hand-coded.
The integration-method adjoint is implemented once in rk4_cpu.h.
"""

from psyneulink.core.batched.continuous_ir import ContinuousDynamics, analyze_continuous_dynamics, equation_nodes


def _function(dynamics, expressions, name, reverse):
    nodes = equation_nodes(expressions)
    indices = {e: i for i, e in enumerate(nodes)}
    binding_axes = {"state": ("x", dynamics.states), "input": ("u", dynamics.inputs), "parameter": ("p", dynamics.parameters)}
    signature = "const double* x, const double* u, const double* p, double t"
    signature += ", const double* adj, double* gx, double* gu, double* gp, double* gt" if reverse else ", double* output"
    lines = [f"static inline void {name}({signature}) {{"]
    for i, expr in enumerate(nodes):
        args = [f"v{indices[a]}" for a in expr.arguments]
        if expr.op in binding_axes:
            symbol, names = binding_axes[expr.op]
            value = f"{symbol}[{names.index(expr.name)}]"
        elif expr.op == "constant":
            value = expr.value.hex()
        elif expr.op == "time":
            value = "t"
        elif expr.op in ("add", "sub", "mul", "div"):
            value = f"({args[0]} {dict(add='+', sub='-', mul='*', div='/')[expr.op]} {args[1]})"
        elif expr.op == "neg":
            value = f"(-{args[0]})"
        else:
            function = "pnl_continuous::sigmoid" if expr.op == "sigmoid" else "std::" + expr.op
            value = f"{function}({args[0]})"
        lines.append(f"    const double v{i} = {value};")
    if not reverse:
        lines.extend(f"    output[{j}] = v{indices[e]};" for j, e in enumerate(expressions))
    else:
        lines.extend(f"    double a{i} = 0.;" for i in range(len(nodes)))
        lines.extend(f"    a{indices[e]} += adj[{j}];" for j, e in enumerate(expressions))
        for i in range(len(nodes) - 1, -1, -1):
            expr = nodes[i]
            children = [indices[a] for a in expr.arguments]
            if expr.op in binding_axes:
                symbol, names = binding_axes[expr.op]
                lines.append(f"    g{symbol}[{names.index(expr.name)}] += a{i};")
            elif expr.op == "time":
                lines.append(f"    *gt += a{i};")
            elif expr.op in ("add", "sub", "mul", "div"):
                a, b = children
                weights = dict(add=("1.", "1."), sub=("1.", "-1."), mul=(f"v{b}", f"v{a}"),
                               div=(f"(1. / v{b})", f"(-v{a} / (v{b} * v{b}))"))[expr.op]
                lines.extend(f"    a{child} += a{i} * {weight};" for child, weight in zip(children, weights, strict=True))
            elif expr.op != "constant":
                weight = dict(neg="-1.", exp=f"v{i}", tanh=f"(1. - v{i} * v{i})", sigmoid=f"(v{i} * (1. - v{i}))")[expr.op]
                lines.append(f"    a{children[0]} += a{i} * {weight};")
    lines.append("}")
    return "\n".join(lines)


def generate_dynamics_source(dynamics):
    """Generate source without building/loading a module or using model names."""
    if type(dynamics) is not ContinuousDynamics:
        raise TypeError("dynamics must be ContinuousDynamics.")
    report = analyze_continuous_dynamics(dynamics)
    if report.stochastic_states or report.stochastic_readouts or dynamics.latent_inputs:
        raise ValueError("Deterministic source generation cannot discard stochastic equations or latent inputs.")
    methods = []
    for name, expressions in (("rhs", dynamics.drift), ("readout", tuple(e for _, e in dynamics.readouts))):
        methods.extend((_function(dynamics, expressions, name, False), _function(dynamics, expressions, name + "_vjp", True)))
    return '\n'.join([
        '#include "rk4_cpu.h"',
        'struct GeneratedDynamics {',
        f'    static constexpr int S = {len(dynamics.states)}, I = {len(dynamics.inputs)}, P = {len(dynamics.parameters)}, R = {len(dynamics.readouts)};',
        *methods, '};',
        'PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {',
        '    module.def("forward", &pnl_continuous::forward<GeneratedDynamics>);',
        '    module.def("backward", &pnl_continuous::backward<GeneratedDynamics>);',
        '}',
    ])


def evaluate_equations(expressions, dynamics, state, inputs, parameters, time):
    """Ordinary Torch reference, not the production integration execution path."""
    import torch

    values = {}
    axes = {"state": (state, dynamics.states), "input": (inputs, dynamics.inputs), "parameter": (parameters, dynamics.parameters)}
    for expr in equation_nodes(expressions):
        args = [values[a] for a in expr.arguments]
        if expr.op in axes:
            value, names = axes[expr.op]
            value = value[..., names.index(expr.name)]
        elif expr.op == "constant":
            value = torch.full(state.shape[:-1], expr.value, dtype=state.dtype, device=state.device)
        elif expr.op == "time":
            value = time + torch.zeros(state.shape[:-1], dtype=state.dtype, device=state.device)
        elif expr.op == "add":
            value = args[0] + args[1]
        elif expr.op == "sub":
            value = args[0] - args[1]
        elif expr.op == "mul":
            value = args[0] * args[1]
        elif expr.op == "div":
            value = args[0] / args[1]
        elif expr.op == "neg":
            value = -args[0]
        else:
            value = getattr(torch, expr.op)(args[0])
        values[expr] = value
    if not expressions:
        return state[..., :0]
    return torch.stack([values[e] for e in expressions], dim=-1)
