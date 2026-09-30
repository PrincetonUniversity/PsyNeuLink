"""Conservative detection of observation-relevant state carried between trials."""

from dataclasses import dataclass

import networkx as nx

from psyneulink.core.components.functions.stateful.statefulfunction import StatefulFunction
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import ControlMechanism
from psyneulink.core.scheduling.condition import Always, AtPass, All, Any


@dataclass(frozen=True)
class LikelihoodHistory:
    """Explain automatic likelihood selection; persistence need not be stochastic.

    ``requires_conditioning`` means trial independence could not be established
    by this structural analysis. Deterministic carried state can conservatively
    select filtering too: its dependence on random within-trial duration is not
    generally visible from a function's noise parameter alone.
    """

    reasons: tuple

    @property
    def requires_conditioning(self):
        return bool(self.reasons)


def _resets_each_trial(condition):
    # These conditions are evaluated at the start of each compiled trial, when
    # its pass counter is zero. Do not execute arbitrary user condition code.
    if type(condition) is Always:
        return True
    if isinstance(condition, AtPass):
        return condition.args[0] == 0
    if type(condition) is All:
        return all(_resets_each_trial(c) for c in condition.args)
    if type(condition) is Any:
        return any(_resets_each_trial(c) for c in condition.args)
    return False


def analyze_likelihood_history(model, outcome_variables, *, parameter_controls=()):
    """Inspect relevant mechanisms, ports, feedback, and nested compositions.

    A declaration of reset-on-every-trial establishes independence only for
    the mechanism's resettable functions and self-recurrence. Stateful ports,
    other feedback cycles, held control modulation, and custom functions are
    handled conservatively. RNG state and ordinary execution counters alone
    do not make an independent-trial model require filtering.
    """
    graph = nx.DiGraph()
    mechanisms = []
    parameter_controls = set(parameter_controls)
    termination_dependencies = set()
    feedback = []
    execution_conditions = {}

    def dependencies(value):
        if hasattr(value, "output_ports"):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from dependencies(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from dependencies(item)
        elif hasattr(value, "args"):
            yield from dependencies(value.args)

    def visit(composition):
        composition._analyze_graph()
        for node in composition.nodes:
            if hasattr(node, "nodes"):
                visit(node)
                # A condition on the nested composition also gates its inputs.
                graph.add_edge(node, node.input_CIM)
            else:
                mechanisms.append(node)
                graph.add_node(node)
        controller = composition.controller
        if controller is not None and composition.enable_controller:
            mechanisms.append(controller)
        for projection in composition.projections:
            if projection.sender is not None and projection.receiver is not None:
                graph.add_edge(projection.sender.owner, projection.receiver.owner)
        for node in composition.nodes:
            _, condition = composition._get_processing_basic_condition(node)
            execution_conditions[node] = condition
            for dependency in dependencies(condition):
                graph.add_edge(dependency, node)
        # A hidden stopping mechanism can affect how long observed mechanisms
        # integrate without having a projection to any observed output.
        termination_dependencies.update(dependencies(composition.termination_processing))
        feedback.extend(composition.feedback_projections)

    visit(model)
    relevant = set()
    for output in outcome_variables:
        node = output.owner if hasattr(output, "owner") else output
        relevant.add(node)
        if node in graph:
            relevant.update(nx.ancestors(graph, node))
    for node in termination_dependencies:
        relevant.add(node)
        if node in graph:
            relevant.update(nx.ancestors(graph, node))
    reasons = []
    for node, condition in execution_conditions.items():
        if node in relevant and node not in parameter_controls and not _resets_each_trial(condition):
            reasons.append(f"{node.name}: conditional execution can retain an output from a previous trial")
    for node in dict.fromkeys(mechanisms):
        if node not in relevant or node in parameter_controls:
            continue
        reset = _resets_each_trial(getattr(node, "reset_stateful_function_when", None))
        functions = [node.function]
        if getattr(node, "integrator_mode", False):
            functions.append(node.integrator_function)
        if not reset and any(isinstance(f, StatefulFunction) for f in functions):
            reasons.append(f"{node.name}: stateful function is not unconditionally reset each trial")
        if getattr(node, "recurrent_projection", None) is not None and not reset:
            reasons.append(f"{node.name}: recurrent output persists between trials")
        if isinstance(node, ControlMechanism):
            reasons.append(f"{node.name}: control modulation can retain values sampled on a previous trial")
        if any(isinstance(f, UserDefinedFunction) for f in functions):
            reasons.append(f"{node.name}: custom function's trial independence is not declared")
        for port in (*node.input_ports, *node.output_ports, *node.parameter_ports):
            if isinstance(port.function, StatefulFunction):
                reasons.append(f"{node.name}.{port.name}: stateful port is not covered by the mechanism reset")
    # Multi-node feedback can read last-trial outputs even when individual
    # mechanisms have stateless functions. Self-recurrence is checked above.
    for group in nx.strongly_connected_components(graph.subgraph(relevant)):
        if len(group) > 1:
            names = ", ".join(sorted(n.name for n in group))
            reasons.append(f"feedback among {names} can carry outputs across trials")
    for projection in feedback:
        if projection.receiver.owner in relevant and projection.sender.owner is not projection.receiver.owner:
            reasons.append(f"{projection.name}: feedback can read an output from the preceding trial")
    return LikelihoodHistory(tuple(dict.fromkeys(reasons)))
