"""Execution-axis dependency analysis for batched graph compilation.

The batched runtime exposes several independent dimensions of work: parameter
sets, subjects, trials, and stochastic estimates.  A value need not vary over
every one of those dimensions.  Recording that fact explicitly is the first
step toward general loop-invariant-code motion and partitioning at stochastic
frontiers.

This module deliberately performs analysis only.  It does not authorize a
kernel split or change execution semantics.  Later transformations can consume
the result while retaining the existing fully fused kernel as a conservative
fallback whenever their additional legality checks cannot be proven.
"""

from __future__ import annotations

from dataclasses import dataclass

from psyneulink.core.batched.ir import BatchedGraphIR, BatchedParamSpec


PARAMETER_SET_AXIS = "parameter_set"
SUBJECT_AXIS = "subject"
TRIAL_AXIS = "trial"
ESTIMATE_AXIS = "estimate"

EXECUTION_AXES = (
    PARAMETER_SET_AXIS,
    SUBJECT_AXIS,
    TRIAL_AXIS,
    ESTIMATE_AXIS,
)
_AXIS_ORDER = {axis: index for index, axis in enumerate(EXECUTION_AXES)}


@dataclass(frozen=True, order=True)
class AxisDependencyEdge:
    """A directed semantic dependency between graph components."""

    producer_component_id: int
    consumer_component_id: int
    kind: str


@dataclass(frozen=True)
class NodeAxisDependency:
    """Direct and transitive launch-axis dependencies for one component."""

    component_id: int
    node: str
    direct_axes: tuple[str, ...]
    axes: tuple[str, ...]

    @property
    def estimate_dependent(self) -> bool:
        return ESTIMATE_AXIS in self.axes


@dataclass(frozen=True)
class AxisDependencyAnalysis:
    """Stable, object-free result of graph execution-axis analysis."""

    nodes: tuple[NodeAxisDependency, ...]
    edges: tuple[AxisDependencyEdge, ...]
    estimate_frontier_edges: tuple[AxisDependencyEdge, ...]
    stochastic_root_component_ids: tuple[int, ...]

    @property
    def estimate_invariant_component_ids(self) -> tuple[int, ...]:
        return tuple(
            node.component_id for node in self.nodes if not node.estimate_dependent
        )

    @property
    def estimate_dependent_component_ids(self) -> tuple[int, ...]:
        return tuple(
            node.component_id for node in self.nodes if node.estimate_dependent
        )

    @property
    def has_estimate_frontier(self) -> bool:
        return bool(self.estimate_frontier_edges)

    def node(self, component_id: int) -> NodeAxisDependency:
        for node in self.nodes:
            if node.component_id == component_id:
                return node
        raise KeyError(component_id)

    def as_metadata(self) -> dict:
        """Return a primitive, deterministic form suitable for IR diagnostics."""

        return {
            "axes": EXECUTION_AXES,
            "nodes": tuple(
                (
                    node.component_id,
                    node.node,
                    node.direct_axes,
                    node.axes,
                )
                for node in self.nodes
            ),
            "edges": tuple(
                (
                    edge.producer_component_id,
                    edge.consumer_component_id,
                    edge.kind,
                )
                for edge in self.edges
            ),
            "estimate_invariant_component_ids": (
                self.estimate_invariant_component_ids
            ),
            "estimate_dependent_component_ids": (
                self.estimate_dependent_component_ids
            ),
            "stochastic_root_component_ids": self.stochastic_root_component_ids,
            "estimate_frontier_edges": tuple(
                (
                    edge.producer_component_id,
                    edge.consumer_component_id,
                    edge.kind,
                )
                for edge in self.estimate_frontier_edges
            ),
        }


def analyze_axis_dependencies(
    graph: BatchedGraphIR,
    params: tuple[BatchedParamSpec, ...] = (),
) -> AxisDependencyAnalysis:
    """Conservatively propagate launch-axis dependencies through ``graph``.

    Random streams are the only direct source of estimate dependence: inputs,
    parameter buffers, and retained state are shared by all stochastic
    estimates at the start of an execution.  Estimate dependence then flows
    through processing projections, scheduler predicates, and modulation
    pathways.

    Parameter buffers are conservatively treated as varying over parameter
    sets, subjects, and trials.  The runtime permits both scalar parameter-set
    values and explicit ``BatchedTrialParameter`` values, so compile-time
    analysis cannot assume the narrower runtime form.
    """

    nodes_by_id = {node.component_id: node for node in graph.nodes}
    if (
        len(nodes_by_id) != len(graph.nodes)
        or any(type(component_id) is not int or component_id < 0 for component_id in nodes_by_id)
    ):
        raise ValueError(
            "Axis dependency analysis requires unique non-negative component IDs."
        )

    direct_axes = {component_id: set() for component_id in nodes_by_id}
    edges: set[AxisDependencyEdge] = set()

    def require_component(component_id: int, label: str) -> None:
        if component_id not in nodes_by_id:
            raise ValueError(
                f"Axis dependency {label} references undeclared component "
                f"ID {component_id}."
            )

    def add_edge(producer: int, consumer: int, kind: str) -> None:
        require_component(producer, f"{kind} producer")
        require_component(consumer, f"{kind} consumer")
        edges.add(AxisDependencyEdge(producer, consumer, kind))

    for input_spec in graph.inputs:
        require_component(input_spec.component_id, "input owner")
        direct_axes[input_spec.component_id].update((SUBJECT_AXIS, TRIAL_AXIS))

    parameter_axes = (PARAMETER_SET_AXIS, SUBJECT_AXIS, TRIAL_AXIS)
    parameter_ids = {parameter.parameter_id for parameter in params}
    if len(parameter_ids) != len(params):
        raise ValueError("Axis dependency analysis requires unique parameter IDs.")
    for parameter in params:
        if not parameter.runtime_mutable or parameter.owner_component_id < 0:
            continue
        require_component(parameter.owner_component_id, "parameter owner")
        direct_axes[parameter.owner_component_id].update(parameter_axes)

    # Some public parameters are bound through function or controller records
    # whose declared owner can be broader than an ordinary graph node.  Mark
    # their exact consuming component as well.
    params_by_name = {parameter.name: parameter for parameter in params}
    for node in graph.nodes:
        if any(
            params_by_name.get(parameter_name, None) is not None
            and params_by_name[parameter_name].runtime_mutable
            for parameter_name in node.params.values()
        ):
            direct_axes[node.component_id].update(parameter_axes)

    for state in graph.states:
        require_component(state.component_id, "state owner")
        direct_axes[state.component_id].add(TRIAL_AXIS)
        if state.function_initializer is not None and any(
            params_by_name.get(parameter_name, None) is not None
            and params_by_name[parameter_name].runtime_mutable
            for parameter_name in state.function_initializer.params.values()
        ):
            direct_axes[state.component_id].update(parameter_axes)

    stochastic_roots = set()
    for stream in graph.rng_streams:
        require_component(stream.component_id, "RNG owner")
        direct_axes[stream.component_id].add(ESTIMATE_AXIS)
        stochastic_roots.add(stream.component_id)

    for projection in graph.projections:
        add_edge(
            projection.sender_component_id,
            projection.receiver_component_id,
            "projection",
        )

    for projection in graph.absorbed_projections:
        add_edge(
            projection.sender_component_id,
            projection.receiver_component_id,
            "absorbed_projection",
        )

    for condition in graph.scheduler:
        require_component(condition.component_id, "scheduler owner")
        for dependency_component_id in condition.dependency_component_ids:
            add_edge(
                dependency_component_id,
                condition.component_id,
                "scheduler",
            )

    # A lane-local dynamic scheduler's exit is itself a control dependency.
    # A repeatedly scheduled component can execute a different number of times
    # when any AllHaveRun operand is delayed (for example, a stochastic
    # terminator delaying its downstream gate).  Dataflow edges alone do not
    # express that feedback: an Always LCA can therefore become
    # estimate-dependent even though no random value flows into its inputs.
    # Keep fixed-pass/first-pass members outside this conservative closure;
    # they execute at a statically selected pass and cannot observe loop length.
    if graph.fusion_kind == "coevolving_graph":
        dynamic_condition_types = {
            "Always",
            "EveryNCalls",
            "AllEveryNCalls",
            "WhenFinished",
        }
        repeated_component_ids = tuple(
            condition.component_id
            for condition in graph.scheduler
            if condition.condition_type in dynamic_condition_types
        )
        trial_termination_ids = tuple(
            component_id
            for termination in graph.termination
            if termination.time_scale == "ENVIRONMENT_STATE_UPDATE"
            and termination.condition_type == "AllHaveRun"
            for component_id in termination.dependency_component_ids
        )
        for producer_component_id in trial_termination_ids:
            for consumer_component_id in repeated_component_ids:
                if producer_component_id != consumer_component_id:
                    add_edge(
                        producer_component_id,
                        consumer_component_id,
                        "schedule_termination_control",
                    )

    for modulation in graph.modulations:
        add_edge(
            modulation.source_component_id,
            modulation.controller_component_id,
            "modulation_monitor",
        )
        add_edge(
            modulation.controller_component_id,
            modulation.target_component_id,
            "modulation",
        )
        if any(
            binding.parameter_id in parameter_ids
            for binding in modulation.controller_param_bindings
        ):
            direct_axes[modulation.controller_component_id].update(parameter_axes)

    for control in graph.folded_affine_controls:
        add_edge(
            control.controller_component_id,
            control.target_component_id,
            "folded_modulation",
        )
        if (
            control.base_parameter_id in parameter_ids
            or control.delta_parameter_id in parameter_ids
        ):
            direct_axes[control.controller_component_id].update(parameter_axes)

    dependencies = {
        component_id: set(axes) for component_id, axes in direct_axes.items()
    }
    changed = True
    while changed:
        changed = False
        for edge in edges:
            before = len(dependencies[edge.consumer_component_id])
            dependencies[edge.consumer_component_id].update(
                dependencies[edge.producer_component_id]
            )
            changed |= len(dependencies[edge.consumer_component_id]) != before

    def ordered_axes(axes) -> tuple[str, ...]:
        return tuple(sorted(axes, key=_AXIS_ORDER.__getitem__))

    node_results = tuple(
        NodeAxisDependency(
            component_id=node.component_id,
            node=node.name,
            direct_axes=ordered_axes(direct_axes[node.component_id]),
            axes=ordered_axes(dependencies[node.component_id]),
        )
        for node in sorted(graph.nodes, key=lambda item: item.component_id)
    )
    estimate_dependent = {
        node.component_id for node in node_results if node.estimate_dependent
    }
    ordered_edges = tuple(sorted(edges))
    frontier = tuple(
        edge
        for edge in ordered_edges
        if edge.producer_component_id not in estimate_dependent
        and edge.consumer_component_id in estimate_dependent
    )
    return AxisDependencyAnalysis(
        nodes=node_results,
        edges=ordered_edges,
        estimate_frontier_edges=frontier,
        stochastic_root_component_ids=tuple(sorted(stochastic_roots)),
    )
