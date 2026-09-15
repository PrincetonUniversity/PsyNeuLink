"""Derive mathematical value expressions from the existing frozen graph.

This is an ideal-real interpretation of a checked stateless feedforward slice,
not a new graph/scheduler, a likelihood, or observed-history reconstruction.
The result can supply readouts to the existing continuous numerical backend.
"""

from dataclasses import dataclass

import numpy as np
import sympy as sp

from psyneulink.core.batched.dependency import NodeAxisDependency, analyze_axis_dependencies
from psyneulink.core.batched.ir import BatchedInputSpec, BatchedOutputSpec, BatchedParamSpec, BatchedPortSpec, BatchedSchedulerSpec
from psyneulink.core.batched.kernel_ir import validate_kernel_ir
from psyneulink.core.batched.likelihood_analysis import single_pass_value_schedule, stateless_value_schedule
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.specs import ElementwiseFunctionSpec, _validate_likelihood_contract


@dataclass(frozen=True)
class SymbolicValues:
    """Flattened expressions with existing source identities, not another model.

    Input symbols follow inputs/coordinate order; parameter symbols follow the
    frozen canonical parameter order (restricted to region owners for readouts);
    expressions follow outputs/coordinate order. Boundary symbols follow their
    port/coordinate order, separately from external inputs. Substitute under
    sympy.evaluate(False) to retain domains. Parameter derivatives hold boundary
    publications fixed; chaining boundary derivatives through dynamics/history
    and any marginalization remain the caller's responsibility.
    """

    inputs: tuple[BatchedInputSpec, ...]
    input_symbols: tuple[sp.Symbol, ...]
    parameters: tuple[BatchedParamSpec, ...]
    parameter_symbols: tuple[sp.Symbol, ...]
    outputs: tuple[BatchedOutputSpec, ...]
    expressions: tuple[sp.Expr, ...]
    source_spec_keys: tuple[str, ...]
    scope: str = "graph"
    boundary_ports: tuple[BatchedPortSpec, ...] = ()
    boundary_symbols: tuple[sp.Symbol, ...] = ()
    boundary_reasons: tuple[tuple[int, str], ...] = ()
    boundary_dependencies: tuple[NodeAxisDependency, ...] = ()
    component_ids: tuple[int, ...] = ()
    publication_schedule: tuple[BatchedSchedulerSpec, ...] = ()

    def explain(self):
        return dict(
            process="registered_ideal_real_values", scope=self.scope,
            guarantee="checked_stateless_feedforward_publication" if self.scope == "graph" else "conditional_readout_at_publication",
            input_ports=tuple(i.port_id for i in self.inputs), parameter_ids=tuple(p.parameter_id for p in self.parameters),
            output_ports=tuple(o.port_id for o in self.outputs), source_spec_keys=self.source_spec_keys,
            component_ids=self.component_ids, boundary_ports=tuple(p.port_id for p in self.boundary_ports),
            boundary_reasons=self.boundary_reasons,
            boundary_axes=tuple((n.component_id, n.axes) for n in self.boundary_dependencies),
            publication_schedule=tuple((s.component_id, s.condition_type, s.consideration_set_id, s.dependency_component_ids)
                                       for s in self.publication_schedule),
            assumptions=("Primitive mathematical rules are trusted declarations in the frozen implementation snapshot.",
                         "Projection constants retain frozen source precision; arithmetic is interpreted over real numbers."),
            limitations=("Not a probability law, continuous-time graph interpretation, or history-factorization witness.",
                         "No state, noise, controls, clipping, or non-primary output transforms inside the value region.",
                         "Boundary values may be latent or history-dependent; they are not declared observed or deterministic.",
                         "Readout scope describes an execution, not a trial-end value; physical clocks and boundary paths remain unbound."),
        )


def _value_region(kernel):
    """Cut at non-algebraic or conditionally published producers in source IR.

    Always ancestors and matching stable pass/finished gates may be substituted.
    Other conditional values are held at the boundary, never recomputed from
    newer inputs. Delayed/same-set edges fail the common publication check below.
    """
    graph = kernel.graph
    schedule = {s.component_id: s for s in graph.scheduler}
    single_pass = graph.metadata.get("schedule_kind") == "static_graph"
    if single_pass:
        single_pass_value_schedule(kernel, prefix="symbolic")
    nodes = {n.component_id: n for n in graph.nodes}
    roots = {o.component_id for o in graph.outputs}
    if len(roots) != 1:
        raise LikelihoodPlanningError("symbolic.region_outputs", "A readout region requires outputs from one execution owner.")
    affected = {s.component_id for s in (*graph.states, *graph.rng_streams, *graph.finished_values)}
    affected.update(s.target_component_id for s in (*graph.effective_parameters, *graph.folded_affine_controls, *graph.modulations))
    affected.update(s.receiver_component_id for s in graph.absorbed_projections)

    def publishes_with(sender, receiver):
        a, b = schedule[sender], schedule[receiver]
        if single_pass or a.condition_type == "Always":
            return True
        if (a.condition_type not in ("AtPass", "AtTrialStart", "WhenFinished")
                or (a.condition_type, a.attrs, a.dependency_component_ids, a.finished_value_ids)
                != (b.condition_type, b.attrs, b.dependency_component_ids, b.finished_value_ids)):
            return False
        # Finished flags are published by their owners. With matching gates,
        # no owner may update between the two reads of that flag.
        return all(schedule[d].consideration_set_id < a.consideration_set_id for d in a.dependency_component_ids)

    def reason(node):
        if node.component_id in affected or any(k in node.attrs for k in ("clip", "noise", "integrator_pre")):
            return "state_noise_or_other_effect"
        if (len(node.input_port_ids) != 1 or len(node.output_port_ids) != 1
                or node.input_width != node.output_width or node.combine != "sum"):
            return "unsupported_value_shape"
        key = node.attrs.get("spec_key")
        if not key:
            return "symbolic_rule_missing"
        spec = kernel.op_specs.lookup_spec(key)
        if not isinstance(spec, ElementwiseFunctionSpec) or spec.likelihood_contract is None or spec.likelihood_contract.symbolic_value is None:
            return "symbolic_rule_missing"
        return None

    selected, boundary, pending = set(), {}, list(roots)
    while pending:
        component = pending.pop()
        if component in selected:
            continue
        why = reason(nodes[component])
        if why:
            raise LikelihoodPlanningError("symbolic.region_root", "Requested execution cannot be lowered: " + why)
        selected.add(component)
        for edge in graph.projections:
            if edge.receiver_component_id != component:
                continue
            sender = nodes[edge.sender_component_id]
            why = reason(sender)
            if why is None and not publishes_with(sender.component_id, component):
                why = "held_conditional_publication"
            if why:
                boundary[edge.sender_port_id] = why
            else:
                pending.append(sender.component_id)
    # One port cannot mean both a held publication and a freshly expanded value.
    if any(p.owner_component_id in selected for p in graph.ports if p.port_id in boundary):
        raise LikelihoodPlanningError("symbolic.publication", "A readout mixes held and recomputed publications from the same producer.")
    return selected, boundary


def derive_symbolic_values(kernel, *, scope="graph"):
    """Lower supported value algebra without consulting live nodes or registries."""
    if scope not in ("graph", "readout"):
        raise ValueError("Symbolic scope must be 'graph' or 'readout'.")
    graph = kernel.graph
    if scope == "graph":
        schedule = stateless_value_schedule(kernel, prefix="symbolic")
        selected, boundary = {n.component_id for n in graph.nodes}, {}
    else:
        validate_kernel_ir(kernel)
        schedule = {s.component_id: s for s in graph.scheduler}
        selected, boundary = _value_region(kernel)

    def reject(code, detail):
        raise LikelihoodPlanningError("symbolic." + code, detail)

    inputs = tuple(i for i in graph.inputs if i.component_id in selected)
    input_symbols = tuple(sp.Symbol(f"input_{i.port_id}_{j}", real=True) for i in inputs for j in range(i.width))
    boundary_ports = tuple(p for p in graph.ports if p.port_id in boundary)
    boundary_symbols = tuple(sp.Symbol(f"boundary_{p.port_id}_{j}", real=True) for p in boundary_ports for j in range(p.width))
    names = {p for node in graph.nodes if node.component_id in selected for p in node.params.values()}
    parameters = kernel.params if scope == "graph" else tuple(p for p in kernel.params if p.name in names)
    parameter_symbols = tuple(sp.Symbol(f"parameter_{p.parameter_id}", real=True) for p in parameters)
    params = {p.name: s for p, s in zip(parameters, parameter_symbols, strict=True)}
    external, offset = {}, 0
    for item in inputs:
        external[item.port_id] = input_symbols[offset:offset + item.width]
        offset += item.width
    values = {}
    offset = 0
    for port in boundary_ports:
        values[port.port_id] = boundary_symbols[offset:offset + port.width]
        offset += port.width
    with sp.evaluate(False):
        for name in graph.execution_order:
            node = graph.node(name)
            if node.component_id not in selected:
                continue
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
    axes = analyze_axis_dependencies(graph, kernel.params) if boundary else None
    owners = selected | {p.owner_component_id for p in boundary_ports}
    return SymbolicValues(inputs, input_symbols, parameters, parameter_symbols, graph.outputs,
                          tuple(e for o in graph.outputs for e in values[o.port_id]), tuple(sorted(kernel.op_specs.specs_by_key)),
                          scope, boundary_ports, boundary_symbols, tuple(sorted(boundary.items())),
                          tuple(axes.node(p.owner_component_id) for p in boundary_ports), tuple(sorted(selected)),
                          tuple(s for s in graph.scheduler if s.component_id in owners))
