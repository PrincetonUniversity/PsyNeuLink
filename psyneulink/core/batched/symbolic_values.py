"""Derive mathematical value expressions from the existing frozen graph.

This is an ideal-real interpretation of a checked stateless feedforward slice,
not a new graph/scheduler, a likelihood, or observed-history reconstruction.
The result can supply readouts to the existing continuous numerical backend.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from psyneulink.core.batched.ir import BatchedInputSpec, BatchedOutputSpec, BatchedParamSpec
from psyneulink.core.batched.likelihood_analysis import stateless_value_schedule
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.specs import ElementwiseFunctionSpec, _validate_likelihood_contract


@dataclass(frozen=True)
class SymbolicValues:
    """Flattened expressions with existing source identities, not another model.

    Input symbols follow inputs/coordinate order; parameter symbols follow the
    frozen canonical parameter order; expressions follow outputs/coordinate
    order. Substitute under sympy.evaluate(False) to retain domain dependencies.
    Initial states and a physical-time/history interpretation remain external.
    """

    inputs: tuple[BatchedInputSpec, ...]
    input_symbols: tuple[sp.Symbol, ...]
    parameters: tuple[BatchedParamSpec, ...]
    parameter_symbols: tuple[sp.Symbol, ...]
    outputs: tuple[BatchedOutputSpec, ...]
    expressions: tuple[sp.Expr, ...]
    source_spec_keys: tuple[str, ...]

    def explain(self):
        return dict(
            process="registered_ideal_real_values", guarantee="checked_stateless_feedforward_publication",
            input_ports=tuple(i.port_id for i in self.inputs), parameter_ids=tuple(p.parameter_id for p in self.parameters),
            output_ports=tuple(o.port_id for o in self.outputs), source_spec_keys=self.source_spec_keys,
            assumptions=("Primitive mathematical rules are trusted declarations in the frozen implementation snapshot.",
                         "Projection constants retain frozen source precision; arithmetic is interpreted over real numbers."),
            limitations=("Not a probability law, continuous-time graph interpretation, or history-factorization witness.",
                         "No state, noise, controls, dynamic schedules, clipping, or non-primary output transforms."),
        )


def derive_symbolic_values(kernel):
    """Lower supported value algebra without consulting live nodes or registries."""
    schedule = stateless_value_schedule(kernel, prefix="symbolic")
    graph = kernel.graph

    def reject(code, detail):
        raise LikelihoodPlanningError("symbolic." + code, detail)

    input_symbols = tuple(sp.Symbol(f"input_{i.port_id}_{j}", real=True) for i in graph.inputs for j in range(i.width))
    parameter_symbols = tuple(sp.Symbol(f"parameter_{p.parameter_id}", real=True) for p in kernel.params)
    params = {p.name: s for p, s in zip(kernel.params, parameter_symbols, strict=True)}
    external, offset = {}, 0
    for item in graph.inputs:
        external[item.port_id] = input_symbols[offset:offset + item.width]
        offset += item.width
    values = {}
    with sp.evaluate(False):
        for name in graph.execution_order:
            node = graph.node(name)
            if (len(node.input_port_ids) != 1 or len(node.output_port_ids) != 1 or node.input_width != node.output_width
                    or node.combine != "sum" or any(k in node.attrs for k in ("clip", "noise", "integrator_pre"))):
                reject("node", "Only unmodified, sum-combined, primary-output elementwise nodes are supported.")
            spec = kernel.op_specs.lookup_spec(node.attrs["spec_key"])
            rule = spec.likelihood_contract
            if not isinstance(spec, ElementwiseFunctionSpec) or rule is None or rule.symbolic_value is None:
                reject("rule_missing", "Every value primitive requires a registered symbolic rule; no class-name inference is used.")
            _validate_likelihood_contract(spec)
            inp, out = node.input_port_ids[0], node.output_port_ids[0]
            combined = list(external.get(inp, (sp.S.Zero,) * node.input_width))
            for edge in graph.projections:
                if edge.receiver_component_id != node.component_id:
                    continue
                projection = kernel.op_specs.lookup_spec(edge.spec_key).likelihood_contract
                matrix = np.asarray(edge.matrix)
                sender = values.get(edge.sender_port_id)
                if (projection is None or projection.value_rule != "dense_projection" or projection.randomness != "none"
                        or sender is None or edge.receiver_port_id != inp or matrix.shape != (len(sender), node.input_width)
                        or not np.isfinite(matrix).all()):
                    reject("projection", "Only finite registered dense projections from already-published value ports are supported.")
                if schedule[edge.sender_component_id].consideration_set_id >= schedule[node.component_id].consideration_set_id:
                    reject("publication", "A source must publish in an earlier consideration set, not a delayed/recurrent read.")
                for j in range(node.input_width):
                    combined[j] += sp.Add(*(sp.Float(float(matrix[i, j])) * value for i, value in enumerate(sender)))
            law = rule.symbolic_value
            try:
                bound = {s: params[node.params[s.name]] for s in law.variables[1:]}
            except KeyError:
                reject("parameters", "A symbolic primitive argument lacks a frozen parameter binding.")
            # xreplace under evaluate(False) retains canceled denominator/zero
            # dependencies until the numerical backend's separate optimization.
            values[out] = tuple(law.expr.xreplace({**bound, law.variables[0]: value}) for value in combined)
    if any(o.port_id not in values or len(values[o.port_id]) != o.width for o in graph.outputs):
        reject("output", "Requested output is not the admitted primary value publication.")
    return SymbolicValues(graph.inputs, input_symbols, kernel.params, parameter_symbols, graph.outputs,
                          tuple(e for o in graph.outputs for e in values[o.port_id]), tuple(sorted(kernel.op_specs.specs_by_key)))
