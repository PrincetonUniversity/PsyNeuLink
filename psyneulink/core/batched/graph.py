from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping
import inspect
import re
from typing import Any

import numpy as np

from psyneulink.core.batched import specs
from psyneulink.core.batched.bindings import BatchedComponentBindings, projection_binding_key
from psyneulink.core.batched.condition_validation import is_canonical_condition
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.ir import (
    FP32_EXACT_INTEGER_LIMIT,
    BatchedAbsorbedProjectionSpec,
    BatchedConsiderationSetSpec,
    BatchedEffectiveParameterSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedModulationSpec,
    BatchedNodeSpec,
    BatchedOp,
    BatchedOutputSpec,
    BatchedParameterBindingSpec,
    BatchedParamSpec,
    BatchedPortSpec,
    BatchedProjectionSpec,
    BatchedResetSpec,
    BatchedRngStreamSpec,
    BatchedScheduleRegionSpec,
    BatchedSchedulerSpec,
    BatchedStateFunctionInitializer,
    BatchedStateSpec,
    BatchedTerminationSpec,
)
from psyneulink.core.batched.schedule import (
    PRECOMPUTED_TRACE_COMPONENT_BUDGET,
    BatchedScheduleTraceError,
    plan_precomputed_schedule_trace,
)
from psyneulink.core.components.functions.nonstateful.transferfunctions import (
    Identity,
    Linear,
    TransferWithCosts,
)
from psyneulink.core.components.functions.nonstateful.transformfunctions import (
    LinearCombination,
    MatrixTransform,
)
from psyneulink.core.components.projections.modulatory.controlprojection import (
    ControlProjection,
)
from psyneulink.core.components.projections.pathway.mappingprojection import (
    MappingProjection,
)
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.components.mechanisms.modulatory.control.controlmechanism import (
    ControlMechanism,
)
from psyneulink.core.scheduling.condition import (
    All,
    AllHaveRun,
    Always,
    AtPass,
    AtTrialStart,
    EveryNCalls,
    Never,
    WhenFinished,
)
from psyneulink.core.scheduling.time import TimeScale


GRAPH_MODEL = "graph"
DDM_MODEL = "ddm"
STATELESS_GRAPH_FUSION = "stateless_graph"
DDM_GRAPH_FUSION = "ddm_graph"
STATEFUL_GRAPH_FUSION = "stateful_graph"
COEVOLVING_GRAPH_FUSION = "coevolving_graph"
STATIC_GRAPH_SCHEDULE = "static_graph"
PRECOMPUTED_TRACE_SCHEDULE = "precomputed_trace"
DYNAMIC_LANE_LOCAL_SCHEDULE = "dynamic_lane_local"
UNSUPPORTED_SCHEDULE = "unsupported"

_PRECOMPUTED_TRACE_CONDITIONS = {"EveryNCalls"}
_DYNAMIC_LANE_LOCAL_CONDITIONS = {"Threshold"}
_COEVOLVING_SCHEDULE_DIAGNOSTIC_DETAIL = (
    "coevolving Always/WhenFinished execution requires explicit finished "
    "predicates and conditional pass regions in KernelIR"
)


@dataclass(frozen=True)
class LoweringResult:
    graph: BatchedGraphIR | None
    params: tuple[BatchedParamSpec, ...]
    bindings: BatchedComponentBindings
    model_kind: str | None
    schedule_kind: str
    supported_nodes: tuple[str, ...]
    rejected_nodes: tuple[BatchedDiagnostic, ...]
    supported_conditions: tuple[str, ...]
    rejected_conditions: tuple[BatchedDiagnostic, ...]


def lower_composition(
    composition,
    outputs=None,
    *,
    ignored_control_nodes=(),
) -> LoweringResult:
    specs.ensure_builtin_specs()
    ignored_control_nodes = tuple(ignored_control_nodes)
    if len({id(node) for node in ignored_control_nodes}) != len(
        ignored_control_nodes
    ):
        raise ValueError(
            "Batched lowering ignored_control_nodes must contain unique "
            "component identities."
        )

    # PsyNeuLink completes deferred control projections, CIM routing, node
    # roles, and the scheduler dependency graph during normal Composition
    # analysis.  Snapshotting before that lifecycle step can describe a graph
    # that Python execution will never use (for example, a controller and its
    # controlled node may incorrectly share a consideration set until the
    # ControlProjection is activated).  Normalize the live structure before
    # assigning stable IDs so lowering is invariant to whether the Composition
    # has already executed.
    composition._analyze_graph()

    nodes = _composition_nodes(composition)
    ignored_control_ids = {id(node) for node in ignored_control_nodes}
    scheduler_dependency_dict = None
    scheduler_consideration_queue = None
    if ignored_control_ids:
        known_ids = {id(node) for node in nodes}
        if not ignored_control_ids <= known_ids or any(
            type(node) is not ControlMechanism
            for node in ignored_control_nodes
        ):
            raise ValueError(
                "Batched lowering may ignore only exact ControlMechanisms "
                "owned by the Composition."
            )
        # PEC injects one generic ControlMechanism for every fitted parameter.
        # The batched objective supplies those same values through parameter
        # rows, so retaining the injected controls would apply each fit twice
        # and would also change the model's scheduler topology.  The caller
        # passes the PEC-owned controls by identity; ordinary compilation never
        # takes this path.
        nodes = [node for node in nodes if id(node) not in ignored_control_ids]
        (
            scheduler_dependency_dict,
            scheduler_consideration_queue,
        ) = _scheduler_view_without_nodes(
            composition,
            ignored_control_ids,
        )
    # Nodes absorbed into another op's kernel (e.g. a collapsing-threshold
    # integrator folded into the DDM boundary) are not lowered as graph nodes.
    absorbed = _absorbed_nodes(composition, nodes)
    if absorbed:
        nodes = [node for node in nodes if _node_name(node) not in absorbed]
    topological_nodes, cyclic_nodes = _dependency_topological_order(composition, nodes)
    component_ids = {
        id(node): component_id
        for component_id, node in enumerate(topological_nodes)
    }
    port_ids, ports_by_id = _port_identity_maps(topological_nodes)
    port_specs = _port_specs(topological_nodes, component_ids, port_ids)
    params = _ParamBuilder()
    rejected_nodes: list[BatchedDiagnostic] = [
        BatchedDiagnostic(
            getattr(composition, "name", "Composition"),
            "cyclic processing dependencies are unsupported for batched v2",
            ", ".join(_node_name(node) for node in cyclic_nodes),
        )
    ] if cyclic_nodes else []
    rejected_nodes.extend(_duplicate_node_name_diagnostics(composition, topological_nodes))
    graph_blockers = list(rejected_nodes)
    supported_nodes: list[str] = []

    model_kind = _classify_model(nodes)
    executable_nodes = [
        node
        for node in topological_nodes
        if type(node) is not ControlMechanism
    ]
    coevolving = _is_coevolving(composition, executable_nodes)
    (
        scheduler_specs,
        schedule_regions,
        consideration_sets,
        finished_values,
        scheduler_declarations_complete,
    ) = _scheduler_ir_specs(
        composition,
        topological_nodes,
        component_ids,
        dependency_dict=scheduler_dependency_dict,
        consideration_queue=scheduler_consideration_queue,
    )
    finished_values_by_component_id = {
        value.component_id: value
        for value in finished_values
    }
    consideration_set_ids = {
        component_id: consideration_set.consideration_set_id
        for consideration_set in consideration_sets
        for component_id in consideration_set.component_ids
    }
    dynamic_controlled_finished_component_ids = (
        _dynamic_controlled_finished_component_ids(
            composition,
            topological_nodes,
            component_ids,
            consideration_set_ids,
        )
    )
    termination_specs, termination_rejections = _termination_ir_specs(
        composition,
        component_ids,
    )
    schedule_kind, supported_conditions, rejected_conditions = _classify_schedule(
        composition,
        topological_nodes,
        component_ids,
        consideration_set_ids,
        finished_values_by_component_id,
        coevolving,
    )
    rejected_conditions.extend(termination_rejections)
    node_bindings = {
        _node_name(node): node
        for node in topological_nodes
    }
    function_bindings = {
        _node_name(node): getattr(node, "function", None)
        for node in topological_nodes
    }
    node_specs = []
    state_specs = []

    for node in topological_nodes:
        component_type = type(node).__name__
        node_name = _node_name(node)
        if type(node) is ControlMechanism:
            diagnostic = _control_support_diagnostic(node, composition)
            if diagnostic is not None:
                rejected_nodes.append(diagnostic)
            else:
                supported_nodes.append(node_name)
            # Control execution is still fail-closed, but retain the component's
            # stable identity and scheduler predicate in declaration-only IR.
            # ControlMechanisms are not part of ``execution_order`` until their
            # dataflow/modulation semantics have executable KernelIR ops.
            node_specs.append(
                _node_spec(
                    node,
                    params,
                    model_kind,
                    composition,
                    component_id=component_ids[id(node)],
                    port_ids=port_ids,
                )
            )
            continue

        diagnostic = _node_support_diagnostic(
            node,
            composition,
            component_id=component_ids[id(node)],
            finished_values_by_component_id=finished_values_by_component_id,
            dynamic_controlled_finished=(
                component_ids[id(node)]
                in dynamic_controlled_finished_component_ids
            ),
        )
        if diagnostic is not None:
            rejected_nodes.append(diagnostic)
            graph_blockers.append(diagnostic)
            continue

        supported_nodes.append(node_name)
        node_spec = _node_spec(
            node,
            params,
            model_kind,
            composition,
            component_id=component_ids[id(node)],
            port_ids=port_ids,
        )
        node_specs.append(node_spec)
        mechanism_spec = specs.mechanism_spec_for(node)
        if mechanism_spec is not None:
            for state_decl in mechanism_spec.states:
                width = state_decl.width if state_decl.width is not None else node_spec.output_width
                function_initializer = None
                if state_decl.initialize_with_function:
                    function_spec = specs.function_spec_for(getattr(node, "function", None))
                    missing_params = (
                        ()
                        if function_spec is None
                        else tuple(
                            binding.arg
                            for binding in function_spec.params
                            if binding.arg not in node_spec.params
                        )
                    )
                    if function_spec is None or missing_params:
                        detail = (
                            type(getattr(node, "function", None)).__name__
                            if function_spec is None
                            else f"missing parameters={missing_params!r}"
                        )
                        diagnostic = BatchedDiagnostic(
                            node_name,
                            "unsupported state function initializer for batched v2",
                            detail,
                        )
                        rejected_nodes.append(diagnostic)
                        graph_blockers.append(diagnostic)
                        continue
                    function_initializer = BatchedStateFunctionInitializer(
                        spec_key=function_spec.key,
                        input_value=tuple([state_decl.initial] * width),
                        params={
                            binding.arg: node_spec.params[binding.arg]
                            for binding in function_spec.params
                        },
                    )
                state_specs.append(
                    BatchedStateSpec(
                        name=f"{node_name}.{state_decl.name}",
                        node=node_name,
                        width=width,
                        initial_value=tuple([state_decl.initial] * width),
                        component_id=node_spec.component_id,
                        state_id=len(state_specs),
                        function_initializer=function_initializer,
                    )
                )

    if ignored_control_nodes:
        unsupported_ignored_controls = tuple(
            control
            for control in ignored_control_nodes
            if not _ignored_parameter_control_is_lowered(
                control,
                composition,
                params,
            )
        )
        if unsupported_ignored_controls:
            raise ValueError(
                "Batched lowering may ignore only external parameter controls "
                "whose exact target Parameter is represented by a runtime lane: "
                + ", ".join(
                    _node_name(control)
                    for control in unsupported_ignored_controls
                )
            )

    if coevolving:
        # The coupled stepper/terminator region is the single unsupported
        # semantic unit.  Replace derivative per-node lane-local diagnostics
        # with one stable composition-level explanation.
        rejected_conditions = [
            diagnostic
            for diagnostic in rejected_conditions
            if not (
                diagnostic.reason
                == "batched schedule kind is not executable yet"
                and diagnostic.detail.endswith(
                    f"requires {DYNAMIC_LANE_LOCAL_SCHEDULE}"
                )
            )
        ]
        if not any(
            diagnostic.reason == "batched schedule kind is not executable yet"
            for diagnostic in rejected_nodes
        ):
            stepper = _coevolving_stepper(composition, executable_nodes)
            rejected_conditions.append(
                BatchedDiagnostic(
                    _node_name(stepper) if stepper is not None else getattr(
                        composition,
                        "name",
                        "Composition",
                    ),
                    "batched schedule kind is not executable yet",
                    _COEVOLVING_SCHEDULE_DIAGNOSTIC_DETAIL,
                )
            )

    _freeze_absorbed_control_parameters(
        node_specs,
        params,
        dynamic_coevolving_target_ids=(
            dynamic_controlled_finished_component_ids
            if coevolving
            else frozenset()
        ),
    )
    (
        absorbed_projection_specs,
        effective_parameter_specs,
        modulation_specs,
        absorbed_projection_bindings_by_id,
        modulation_bindings_by_id,
    ) = _modulation_ir_specs(
        composition,
        topological_nodes,
        node_specs,
        params,
        component_ids,
        port_ids,
        consideration_set_ids,
    )
    finished_values = _controlled_finished_value_specs(
        finished_values,
        modulation_specs,
    )

    (
        projections,
        projection_rejections,
        projection_bindings,
        projection_bindings_by_id,
    ) = _projection_specs(
        composition,
        topological_nodes,
        component_ids,
        port_ids,
    )
    rejected_nodes.extend(projection_rejections)
    graph_blockers.extend(projection_rejections)
    output_rejections = _output_support_diagnostics(outputs, nodes)
    rejected_nodes.extend(output_rejections)
    graph_blockers.extend(output_rejections)
    inputs = _input_specs(
        topological_nodes,
        projections,
        component_ids,
        port_ids,
    )
    input_rejections = _external_input_support_diagnostics(
        inputs,
        ports_by_id,
    )
    rejected_nodes.extend(input_rejections)
    graph_blockers.extend(input_rejections)
    reset_specs, reset_declarations_complete = _reset_ir_specs(
        topological_nodes,
        state_specs,
        component_ids,
    )

    graph = None
    # Recognized scheduler semantics are useful semantic IR even before a
    # backend can execute them.  Keep such graphs inspectable while capability
    # analysis continues to fail closed on ``rejected_conditions``.  Conditions
    # without a typed declaration still prevent graph construction.
    if (
        not graph_blockers
        and scheduler_declarations_complete
        and reset_declarations_complete
    ):
        outputs = _output_specs(
            composition,
            outputs,
            topological_nodes,
            component_ids,
            port_ids,
        )
        execution_order = tuple(
            _node_name(node)
            for node in topological_nodes
            if type(node) is not ControlMechanism
        )
        ops = tuple(
            BatchedOp(kind=_op_kind(node), target=_node_name(node))
            for node in topological_nodes
            if type(node) is not ControlMechanism
        ) + tuple(
            BatchedOp(
                kind="store_output",
                target=output.name,
                inputs=(f"{output.node}.{output.port}",),
            )
            for output in outputs
        )
        graph = BatchedGraphIR(
            nodes=tuple(node_specs),
            inputs=tuple(inputs),
            projections=tuple(projections),
            outputs=tuple(outputs),
            states=tuple(state_specs),
            scheduler=scheduler_specs,
            ops=ops,
            execution_order=execution_order,
            fusion_kind=_fusion_kind(model_kind, nodes, composition),
            executable=(
                not rejected_nodes
                and not rejected_conditions
                and not modulation_specs
            ),
            metadata={
                "composition_name": getattr(composition, "name", None),
                "schedule_kind": schedule_kind,
                "scheduler_executable": (
                    not rejected_nodes
                    and not rejected_conditions
                    and not modulation_specs
                ),
                "scheduler_requires_pass_region": (
                    bool(rejected_nodes)
                    or coevolving
                    or schedule_kind != STATIC_GRAPH_SCHEDULE
                ),
                "schedule_trace_component_budget": (
                    PRECOMPUTED_TRACE_COMPONENT_BUDGET
                ),
                # Warm-up steps before the co-evolving terminator begins (the ITI:
                # the LCA decays / integrates onset inputs first). = max node
                # onset; 0 when there is none.
                "coevolve_warmup": max(
                    (spec.attrs.get("onset_step", 0) for spec in node_specs), default=0
                ),
            },
            rng_streams=_rng_stream_specs(node_specs),
            ports=port_specs,
            absorbed_projections=absorbed_projection_specs,
            schedule_regions=schedule_regions,
            consideration_sets=consideration_sets,
            finished_values=finished_values,
            effective_parameters=effective_parameter_specs,
            modulations=modulation_specs,
            resets=reset_specs,
            termination=termination_specs,
        )

        if (
            schedule_kind == PRECOMPUTED_TRACE_SCHEDULE
            and not rejected_nodes
            and not rejected_conditions
        ):
            schedule_diagnostic = _precomputed_schedule_support_diagnostic(graph)
            if schedule_diagnostic is not None:
                rejected_conditions.append(schedule_diagnostic)
                graph = replace(
                    graph,
                    executable=False,
                    metadata={
                        **graph.metadata,
                        "scheduler_executable": False,
                    },
                )

        if _dynamic_controlled_finished_graph_eligible(
            graph,
            tuple(params.specs),
        ):
            follower_names = {
                condition.node
                for condition in graph.scheduler
                if condition.condition_type == "WhenFinished"
            }
            rejected_conditions = [
                diagnostic
                for diagnostic in rejected_conditions
                if not (
                    diagnostic.component in follower_names
                    and diagnostic.reason
                    == "batched schedule kind is not executable yet"
                    and diagnostic.detail
                    == "WhenFinished requires dynamic_lane_local"
                )
            ]
            graph = replace(
                graph,
                executable=not rejected_nodes and not rejected_conditions,
                metadata={
                    **graph.metadata,
                    "scheduler_executable": (
                        not rejected_nodes and not rejected_conditions
                    ),
                },
            )

        if _dynamic_controlled_coevolving_graph_eligible(
            graph,
            tuple(params.specs),
        ):
            _freeze_dynamic_coevolving_parameters(graph, params)
            rejected_conditions = [
                diagnostic
                for diagnostic in rejected_conditions
                if not (
                    diagnostic.reason
                    == "batched schedule kind is not executable yet"
                    and diagnostic.detail
                    == _COEVOLVING_SCHEDULE_DIAGNOSTIC_DETAIL
                )
            ]
            graph = replace(
                graph,
                executable=not rejected_nodes and not rejected_conditions,
                metadata={
                    **graph.metadata,
                    "scheduler_executable": (
                        not rejected_nodes and not rejected_conditions
                    ),
                },
            )

    return LoweringResult(
        graph=graph,
        params=tuple(params.specs),
        bindings=BatchedComponentBindings(
            nodes=node_bindings,
            functions=function_bindings,
            projections=projection_bindings,
            nodes_by_id={
                component_ids[id(node)]: node
                for node in topological_nodes
            },
            functions_by_id={
                component_ids[id(node)]: getattr(node, "function", None)
                for node in topological_nodes
            },
            parameters_by_id=params.bindings_by_id,
            ports_by_id=ports_by_id,
            projections_by_id=projection_bindings_by_id,
            absorbed_projections_by_id=absorbed_projection_bindings_by_id,
            modulations_by_id=modulation_bindings_by_id,
        ),
        model_kind=model_kind,
        schedule_kind=schedule_kind,
        supported_nodes=tuple(supported_nodes),
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=tuple(supported_conditions),
        rejected_conditions=tuple(rejected_conditions),
    )


def projection_inputs(
    graph: BatchedGraphIR,
    receiver: str,
    *,
    receiver_port_id: int | None = None,
) -> tuple[BatchedProjectionSpec, ...]:
    """Return projections into one node, optionally narrowed to one InputPort.

    ``receiver`` remains the public display-name lookup contract.  Kernel
    lowering supplies the stable numeric port id so two same-width InputPorts
    can never be merged merely because they share a receiver node.
    """

    return tuple(
        projection
        for projection in graph.projections
        if projection.receiver == receiver
        and (
            receiver_port_id is None
            or projection.receiver_port_id == receiver_port_id
        )
    )


class _ParamBuilder:
    def __init__(self):
        self.specs: list[BatchedParamSpec] = []
        self._names: set[str] = set()
        self.bindings_by_id: dict[int, object] = {}

    def add(
        self,
        name: str,
        default: float,
        aliases: Iterable[str] = (),
        *,
        parameter=None,
        minimum: float | None = None,
        minimum_inclusive: bool = True,
        maximum: float | None = None,
        maximum_inclusive: bool = True,
        owner_component_id: int = -1,
        owner_scope: str = "",
    ) -> str:
        if name in self._names:
            return name
        self._names.add(name)
        parameter_id = len(self.specs)
        self.specs.append(
            BatchedParamSpec(
                name,
                float(default),
                tuple(aliases),
                parameter_id=parameter_id,
                minimum=minimum,
                minimum_inclusive=minimum_inclusive,
                maximum=maximum,
                maximum_inclusive=maximum_inclusive,
                owner_component_id=owner_component_id,
                owner_scope=owner_scope,
            )
        )
        if parameter is not None:
            self.bindings_by_id[parameter_id] = parameter
        return name

    def freeze(self, name: str, reason: str) -> None:
        """Mark one lowered parameter as validated-default-only at runtime."""

        for index, spec in enumerate(self.specs):
            if spec.name == name:
                self.specs[index] = replace(
                    spec,
                    runtime_mutable=False,
                    runtime_constraint=reason,
                )
                return
        raise KeyError(f"Cannot freeze unknown batched parameter '{name}'.")

    def binding(self, argument: str, name: str) -> BatchedParameterBindingSpec:
        """Return the stable identity for a parameter previously added by name."""

        matches = tuple(spec for spec in self.specs if spec.name == name)
        if len(matches) != 1:
            raise KeyError(
                f"Cannot bind batched argument '{argument}' to parameter '{name}'."
            )
        return BatchedParameterBindingSpec(
            argument=argument,
            parameter=name,
            parameter_id=matches[0].parameter_id,
        )


def _freeze_absorbed_control_parameters(
    node_specs: list[BatchedNodeSpec],
    params: _ParamBuilder,
    *,
    dynamic_coevolving_target_ids: frozenset[int] = frozenset(),
) -> None:
    """Freeze values whose compile-time identity semantics are intentionally erased."""

    nodes_by_name = {node.name: node for node in node_specs}
    for node in node_specs:
        if node.component_type == "ControlMechanism":
            for parameter_name in node.params.values():
                params.freeze(
                    parameter_name,
                    "absorbed control parameters are frozen in KernelIR",
                )
        source_name = node.attrs.get("termination_input_node")
        if node.component_type != "LCAMechanism" or source_name is None:
            continue
        # A referenced source can have failed its own semantic lowering.  Its
        # structured rejection is already part of the capability report; do
        # not turn that unsupported model into an internal compiler error.
        source = nodes_by_name.get(source_name)
        if source is None:
            continue
        dynamic_coevolving_source = (
            node.component_id in dynamic_coevolving_target_ids
        )
        for argument, parameter_name in source.params.items():
            # A co-evolving CSI cue is not erased: its Linear computation is
            # emitted before the lane-local pass loop, so the fitted slope and
            # intercept must remain ordinary runtime lane parameters.  Scale
            # and offset stay fixed to keep this first affine boundary narrow.
            if dynamic_coevolving_source and argument in {
                "slope",
                "intercept",
            }:
                continue
            params.freeze(
                parameter_name,
                (
                    f"absorbed {'affine' if dynamic_coevolving_source else 'identity'} "
                    f"termination-threshold source for {node.name}"
                ),
            )


def _freeze_dynamic_coevolving_parameters(
    graph: BatchedGraphIR,
    params: _ParamBuilder,
) -> None:
    """Freeze DDM values whose coevolving form assumes their defaults.

    The folded threshold control is represented by runtime ``threshold`` and
    ``threshold_collapse`` lanes (with aliases for the absorbed source), so
    those two values deliberately remain mutable.  Noise is fixed when the
    graph is compiled, while region-local state is initialized from the frozen
    starting value and offset.
    """

    finished = next(
        value
        for value in graph.finished_values
        if value.predicate_kind == "dynamic"
    )
    terminator = graph.node(finished.node)
    reason = "first coevolving DDM boundary is frozen in KernelIR"
    for argument in (
        "noise",
        "starting_value",
        "offset",
    ):
        params.freeze(terminator.params[argument], reason)


def _typed_dynamic_control_chain_supported(
    composition,
    control,
    chain: _ResolvedControlChain,
    active_node_ids: set[int],
    component_ids,
    consideration_set_ids,
) -> bool:
    """Whether one exact control chain can be declared as dynamic finished IR."""

    source = chain.source
    target = chain.target
    function = getattr(control, "function", None)
    conditions = _scheduler_conditions(composition)
    source_set = consideration_set_ids.get(component_ids.get(id(source), -1), -1)
    controller_set = consideration_set_ids.get(
        component_ids.get(id(control), -1),
        -1,
    )
    target_set = consideration_set_ids.get(component_ids.get(id(target), -1), -1)
    return bool(
        id(source) in active_node_ids
        and id(control) in active_node_ids
        and id(target) in active_node_ids
        and type(target).__name__ == "LCAMechanism"
        and chain.target_port == "termination_threshold"
        # PNL freezes values at consideration-set entry.  The first declared
        # controlled-finished subset therefore requires the cue, controller,
        # and controlled target to occupy strictly ordered sets.  Same-set
        # control would make the target observe the prior held value.
        and 0 <= source_set < controller_set < target_set
        and (
            _supported_lca_termination_source(composition, source)
            or (
                _is_unmodeled_coevolving_lca_termination(
                    composition,
                    target,
                )
                and _supported_dynamic_lca_termination_source(
                    composition,
                    source,
                )
            )
        )
        and all(
            _port_width(port) == 1
            for port in (
                chain.source_port,
                chain.controller_input_port,
                chain.signal,
                chain.target_parameter_port,
            )
        )
        and _finite_fp32_scalar_value(
            _parameter_value(target, chain.target_port, None)
        )
        is not None
        and _finite_fp32_scalar_value(
            _parameter_default_value(chain.control_projection, "value", None)
        )
        is not None
        and (
            type(function) is Identity
            or (
                type(function) is Linear
                and specs.function_spec_for(function) is not None
            )
        )
        and _scheduler_condition_is_effective_always(
            composition,
            target,
            conditions,
        )
        and any(
            _when_finished_depends_on(condition, target)
            for candidate, condition in conditions.items()
            if candidate is not target
        )
    )


def _dynamic_controlled_finished_graph_eligible(
    graph: BatchedGraphIR,
    parameters: tuple[BatchedParamSpec, ...],
    *,
    op_specs: specs.BatchedOpSpecSnapshot | None = None,
) -> bool:
    """Recognize a disjoint union of executable controlled-finished chains."""

    lookup_spec = specs.lookup_spec if op_specs is None else op_specs.lookup_spec
    dynamic_finished = tuple(
        value
        for value in graph.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    node_component_ids = tuple(node.component_id for node in graph.nodes)
    chain_count = len(graph.modulations)
    if not (
        graph.metadata.get("schedule_kind") == DYNAMIC_LANE_LOCAL_SCHEDULE
        and graph.fusion_kind == STATEFUL_GRAPH_FUSION
        and not graph.rng_streams
        and chain_count > 0
        and len(graph.effective_parameters) == chain_count
        and len(dynamic_finished) == chain_count
        and len(graph.finished_values) == chain_count
        and len(graph.nodes) == 5 * chain_count
        and tuple(item.modulation_id for item in graph.modulations)
        == tuple(range(chain_count))
        and tuple(
            item.effective_parameter_id for item in graph.effective_parameters
        )
        == tuple(range(chain_count))
        and tuple(item.value_id for item in dynamic_finished)
        == tuple(range(chain_count))
        and tuple(
            item.effective_parameter_id for item in graph.modulations
        )
        == tuple(
            item.effective_parameter_id for item in graph.effective_parameters
        )
        and all(type(component_id) is int for component_id in node_component_ids)
        and node_component_ids == tuple(range(len(graph.nodes)))
    ):
        return False

    parameters_by_name = {parameter.name: parameter for parameter in parameters}
    if len(parameters_by_name) != len(parameters):
        return False

    scheduler_by_id = {
        condition.component_id: condition for condition in graph.scheduler
    }
    if (
        len(graph.scheduler) != len(graph.nodes)
        or set(scheduler_by_id) != set(node_component_ids)
    ):
        return False

    def exact_attrs(actual, expected) -> bool:
        return bool(
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                type(actual[key]) is type(value) and actual[key] == value
                for key, value in expected.items()
            )
        )

    def condition_matches(condition, condition_type, attrs) -> bool:
        return bool(
            condition.condition_type == condition_type
            and condition.region == "pass"
            and not condition.dependencies
            and not condition.dependency_component_ids
            and not condition.finished_value_ids
            and exact_attrs(condition.attrs, attrs)
        )

    consideration_sets = tuple(
        sorted(
            graph.consideration_sets,
            key=lambda item: item.consideration_set_id,
        )
    )
    if (
        not consideration_sets
        or tuple(item.consideration_set_id for item in consideration_sets)
        != tuple(range(len(consideration_sets)))
        or any(
            item.region != "pass"
            or item.inputs_frozen is not True
            or item.component_ids
            != tuple(graph.node(name).component_id for name in item.nodes)
            for item in consideration_sets
        )
    ):
        return False
    component_set_ids = {}
    for item in consideration_sets:
        for component_id in item.component_ids:
            if component_id in component_set_ids:
                return False
            component_set_ids[component_id] = item.consideration_set_id
    if set(component_set_ids) != set(node_component_ids) or any(
        scheduler_by_id[component_id].consideration_set_id != set_id
        for component_id, set_id in component_set_ids.items()
    ):
        return False

    if (
        tuple(
            (region.name, region.kind, region.time_scale, region.parent)
            for region in graph.schedule_regions
        )
        != (
            ("trial", "trial", "ENVIRONMENT_STATE_UPDATE", ""),
            ("pass", "pass", "PASS", "trial"),
        )
        or len(graph.termination) != 2
    ):
        return False
    termination_by_scale = {
        termination.time_scale: termination
        for termination in graph.termination
    }
    trial_termination = termination_by_scale.get("ENVIRONMENT_STATE_UPDATE")
    run_termination = termination_by_scale.get("ENVIRONMENT_SEQUENCE")
    all_component_ids = tuple(node.component_id for node in graph.nodes)
    if not (
        len(termination_by_scale) == len(graph.termination)
        and trial_termination is not None
        and trial_termination.condition_type == "AllHaveRun"
        and type(trial_termination.dependency_component_ids) is tuple
        and len(trial_termination.dependency_component_ids)
        == len(all_component_ids)
        and all(
            type(actual) is type(expected) and actual == expected
            for actual, expected in zip(
                trial_termination.dependency_component_ids,
                all_component_ids,
            )
        )
        and exact_attrs(trial_termination.attrs, {})
        and run_termination is not None
        and run_termination.condition_type == "Never"
        and type(run_termination.dependency_component_ids) is tuple
        and run_termination.dependency_component_ids == ()
        and exact_attrs(run_termination.attrs, {})
    ):
        return False

    projection_pairs = [
        (projection.sender_component_id, projection.receiver_component_id)
        for projection in graph.projections
    ]
    effective_by_id = {
        item.effective_parameter_id: item
        for item in graph.effective_parameters
    }
    resets_by_id = {item.component_id: item for item in graph.resets}
    if len(resets_by_id) != len(graph.resets):
        return False

    at_pass_zero = {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    used_component_ids = set()
    expected_projection_pairs = []
    expected_input_ids = set()
    expected_output_ids = set()
    target_ids = set()
    follower_ids = set()
    controller_ids = set()

    for modulation in graph.modulations:
        try:
            source = graph.node(modulation.source)
            controller = graph.node(modulation.controller)
            target = graph.node(modulation.target)
            target_spec = lookup_spec(target.attrs["spec_key"])
            parameter = effective_by_id[modulation.effective_parameter_id]
            source_condition = scheduler_by_id[modulation.source_component_id]
            controller_condition = scheduler_by_id[
                modulation.controller_component_id
            ]
            target_condition = scheduler_by_id[modulation.target_component_id]
        except (KeyError, specs.BatchedOpSpecError):
            return False

        controller_spec_key = controller.attrs.get("spec_key", "")
        try:
            controller_spec = (
                lookup_spec(controller_spec_key)
                if controller_spec_key
                else None
            )
        except specs.BatchedOpSpecError:
            return False
        controller_bindings = (
            tuple(binding.arg for binding in controller_spec.params)
            if isinstance(controller_spec, specs.ElementwiseFunctionSpec)
            else ()
        )
        controller_parameters = tuple(
            parameters_by_name.get(name) for name in controller.params.values()
        )
        identity_controller = bool(
            controller.function_type == "Identity"
            and controller_spec is None
            and not controller.params
            and controller.attrs.get("spec_kind") == "control"
            and controller.attrs.get("control_function") == "identity"
        )
        linear_count_controller = False
        if (
            isinstance(controller_spec, specs.ElementwiseFunctionSpec)
            and controller_spec.function_class is Linear
            and controller.function_type == "Linear"
            and controller.attrs.get("spec_kind") == "control"
            and controller.attrs.get("control_function") == "registered"
            and tuple(controller.params) == controller_bindings
            and controller_bindings
            == ("slope", "intercept", "scale", "offset")
            and all(
                parameter is not None for parameter in controller_parameters
            )
            and all(
                parameter.owner_component_id == controller.component_id
                and parameter.owner_scope == binding.scope
                and not parameter.runtime_mutable
                for parameter, binding in zip(
                    controller_parameters,
                    controller_spec.params,
                )
            )
        ):
            defaults = {
                argument: parameters_by_name[name].default
                for argument, name in controller.params.items()
            }
            intercept = defaults["intercept"]
            linear_count_controller = bool(
                all(type(value) is float for value in defaults.values())
                and defaults["slope"] == 1.0
                and defaults["scale"] == 1.0
                and defaults["offset"] == 0.0
                and np.isfinite(intercept)
                and 0.0 <= intercept <= FP32_EXACT_INTEGER_LIMIT
                and float(intercept).is_integer()
            )

        finished_matches = tuple(
            value
            for value in dynamic_finished
            if value.component_id == modulation.target_component_id
        )
        followers = tuple(
            condition
            for condition in graph.scheduler
            if condition.condition_type == "WhenFinished"
            and condition.dependency_component_ids
            == (modulation.target_component_id,)
            and len(finished_matches) == 1
            and condition.finished_value_ids == (finished_matches[0].value_id,)
        )
        inbound = tuple(
            projection
            for projection in graph.projections
            if projection.receiver_component_id
            == modulation.target_component_id
        )
        if len(finished_matches) != 1 or len(followers) != 1 or len(inbound) != 1:
            return False
        finished = finished_matches[0]
        follower_condition = followers[0]
        try:
            follower = graph.node(follower_condition.node)
            prelude = next(
                node
                for node in graph.nodes
                if node.component_id == inbound[0].sender_component_id
            )
            prelude_condition = scheduler_by_id[prelude.component_id]
        except (KeyError, StopIteration):
            return False

        role_ids = {
            source.component_id,
            controller.component_id,
            target.component_id,
            follower.component_id,
            prelude.component_id,
        }
        if len(role_ids) != 5 or used_component_ids.intersection(role_ids):
            return False
        used_component_ids.update(role_ids)
        target_ids.add(target.component_id)
        follower_ids.add(follower.component_id)
        controller_ids.add(controller.component_id)

        expected_finished_attrs = {
            "effective_parameter_id": modulation.effective_parameter_id,
            "target_parameter_port_id": modulation.target_parameter_port_id,
            "rounding": "ceil",
            "minimum": 1,
            "maximum": 2 ** 24,
        }
        if not (
            isinstance(target_spec, specs.MechanismOpSpec)
            and target_spec.can_step
            and (identity_controller or linear_count_controller)
            and target.attrs.get("termination_input_node") == source.name
            and not target.attrs.get("diagnostics")
            and source.component_id == modulation.source_component_id
            and controller.component_id == modulation.controller_component_id
            and target.component_id == modulation.target_component_id
            and modulation.controller not in graph.execution_order
            and modulation.source in graph.execution_order
            and modulation.target in graph.execution_order
            and type(controller.input_width) is int
            and controller.input_width == 1
            and type(controller.output_width) is int
            and controller.output_width == 1
            and controller.input_port_ids
            == (modulation.controller_input_port_id,)
            and controller.output_port_ids
            == (modulation.control_signal_port_id,)
            and len(controller.parameter_port_ids) == len(controller.params)
            and {name for name, _ in controller.parameter_port_ids}
            == set(controller.params)
            and parameter.target == target.name
            and parameter.target_component_id == target.component_id
            and parameter.target_parameter == modulation.target_parameter
            and parameter.target_parameter_port_id
            == modulation.target_parameter_port_id
            and finished.node == target.name
            and finished.producer_consideration_set_id
            == target_condition.consideration_set_id
            and exact_attrs(finished.attrs, expected_finished_attrs)
            and condition_matches(source_condition, "AtPass", at_pass_zero)
            and condition_matches(controller_condition, "AtPass", at_pass_zero)
            and condition_matches(prelude_condition, "AtPass", at_pass_zero)
            and condition_matches(target_condition, "Always", {})
            and follower_condition.region == "pass"
            and follower_condition.dependencies == (target.name,)
            and follower_condition.dependency_component_ids
            == (target.component_id,)
            and follower_condition.finished_value_ids == (finished.value_id,)
            and exact_attrs(
                follower_condition.attrs,
                {"predicate": "is_finished"},
            )
            and component_set_ids[source.component_id]
            == component_set_ids[prelude.component_id]
            and component_set_ids[source.component_id]
            < component_set_ids[controller.component_id]
            < component_set_ids[target.component_id]
            < component_set_ids[follower.component_id]
            and prelude.component_type == "TransferMechanism"
            and follower.component_type == "TransferMechanism"
        ):
            return False

        target_state_ids = tuple(
            state.state_id
            for state in graph.states
            if state.component_id == target.component_id
        )
        reset = resets_by_id.get(target.component_id)
        if (
            not target_state_ids
            or reset is None
            or reset.node != target.name
            or reset.state_ids != target_state_ids
            or reset.condition_type not in {"AtTrialStart", "Never"}
            or reset.region != "trial"
            or not exact_attrs(reset.attrs, {})
        ):
            return False

        expected_projection_pairs.extend((
            (prelude.component_id, target.component_id),
            (target.component_id, follower.component_id),
        ))
        expected_input_ids.update((source.component_id, prelude.component_id))
        expected_output_ids.add(follower.component_id)

    execution_order = tuple(
        node.name
        for node in graph.nodes
        if node.component_id not in controller_ids
    )
    input_ids = tuple(item.component_id for item in graph.inputs)
    output_ids = tuple(item.component_id for item in graph.outputs)
    return bool(
        used_component_ids == set(node_component_ids)
        and all(state.component_id in target_ids for state in graph.states)
        and set(resets_by_id) == target_ids
        and graph.execution_order == execution_order
        and sorted(projection_pairs) == sorted(expected_projection_pairs)
        and len(input_ids) == 2 * chain_count
        and len(set(input_ids)) == len(input_ids)
        and set(input_ids) == expected_input_ids
        and len(output_ids) == chain_count
        and len(set(output_ids)) == len(output_ids)
        and set(output_ids) == expected_output_ids == follower_ids
    )


def _dynamic_controlled_coevolving_graph_eligible(
    graph: BatchedGraphIR,
    parameters: tuple[BatchedParamSpec, ...],
    *,
    op_specs: specs.BatchedOpSpecSnapshot | None = None,
) -> bool:
    """Recognize the first executable CSI co-evolving graph exactly.

    Co-evolution changes scheduler timing, retained state, random-stream
    indexing, and which values may be observed after an early lane-local exit.
    Admission is therefore intentionally based on the complete lowered graph,
    not just on the presence of an ``Always`` LCA and a ``WhenFinished`` DDM.
    Every role below is recovered from typed identities and dataflow; public
    component names are used only to cross-check those identities.
    """

    lookup_spec = specs.lookup_spec if op_specs is None else op_specs.lookup_spec
    node_component_ids = tuple(node.component_id for node in graph.nodes)
    if not (
        graph.metadata.get("schedule_kind") == DYNAMIC_LANE_LOCAL_SCHEDULE
        and graph.fusion_kind == COEVOLVING_GRAPH_FUSION
        and len(graph.nodes) == 11
        and all(type(component_id) is int for component_id in node_component_ids)
        and node_component_ids == tuple(range(len(graph.nodes)))
        and len(graph.scheduler) == len(graph.nodes)
        and len(graph.consideration_sets) == 6
        and len(graph.modulations) == 1
        and len(graph.effective_parameters) == 1
        and len(graph.finished_values) == 2
        and len(graph.absorbed_projections) == 2
        and len(graph.rng_streams) == 1
        and len(graph.inputs) == 4
        and len(graph.projections) == 8
        and len(graph.outputs) == 2
    ):
        return False

    def exact_attrs(actual, expected) -> bool:
        return bool(
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(
                type(actual[key]) is type(value) and actual[key] == value
                for key, value in expected.items()
            )
        )

    parameters_by_name = {parameter.name: parameter for parameter in parameters}
    if len(parameters_by_name) != len(parameters):
        return False
    nodes_by_component_id = {node.component_id: node for node in graph.nodes}

    def parameter_for(node, argument):
        parameter_name = node.params.get(argument)
        if type(parameter_name) is not str:
            return None
        parameter = parameters_by_name.get(parameter_name)
        if (
            parameter is None
            or parameter.owner_component_id != node.component_id
        ):
            return None
        return parameter

    def registered_identity_linear(node, *, frozen: bool) -> bool:
        try:
            function_spec = lookup_spec(node.attrs["spec_key"])
        except (KeyError, specs.BatchedOpSpecError):
            return False
        if not (
            isinstance(function_spec, specs.ElementwiseFunctionSpec)
            and function_spec.function_class is Linear
            and node.function_type == "Linear"
            and tuple(node.params)
            == ("slope", "intercept", "scale", "offset")
        ):
            return False
        bound = tuple(
            (binding, parameter_for(node, binding.arg))
            for binding in function_spec.params
        )
        if (
            tuple(binding.arg for binding, _ in bound)
            != ("slope", "intercept", "scale", "offset")
            or any(parameter is None for _, parameter in bound)
        ):
            return False
        defaults = tuple(parameter.default for _, parameter in bound)
        return bool(
            all(type(value) is float for value in defaults)
            and defaults == (1.0, 0.0, 1.0, 0.0)
            and all(
                parameter.owner_scope == binding.scope
                and parameter.runtime_mutable is (not frozen)
                for binding, parameter in bound
            )
        )

    def registered_affine_count_source(node) -> bool:
        """Authenticate the emitted CSI cue transform and its mutability."""

        try:
            function_spec = lookup_spec(node.attrs["spec_key"])
        except (KeyError, specs.BatchedOpSpecError):
            return False
        if not (
            isinstance(function_spec, specs.ElementwiseFunctionSpec)
            and function_spec.function_class is Linear
            and node.function_type == "Linear"
            and tuple(node.params)
            == ("slope", "intercept", "scale", "offset")
        ):
            return False
        bound = {
            binding.arg: (binding, parameter_for(node, binding.arg))
            for binding in function_spec.params
        }
        if tuple(bound) != ("slope", "intercept", "scale", "offset"):
            return False
        if any(parameter is None for _, parameter in bound.values()):
            return False
        defaults = {
            argument: parameter.default
            for argument, (_, parameter) in bound.items()
        }
        return bool(
            all(
                type(value) is float
                and np.isfinite(value)
                for value in defaults.values()
            )
            and all(
                0.0 <= defaults[argument] <= FP32_EXACT_INTEGER_LIMIT
                and defaults[argument].is_integer()
                for argument in ("slope", "intercept")
            )
            and defaults["scale"] == 1.0
            and defaults["offset"] == 0.0
            and all(
                parameter.owner_scope == binding.scope
                and parameter.runtime_mutable
                is (argument in {"slope", "intercept"})
                for argument, (binding, parameter) in bound.items()
            )
        )

    def registered_count_controller_intercept(node) -> int | None:
        """Return a frozen integral CSI-controller intercept, if exact."""

        try:
            function_spec = lookup_spec(node.attrs["spec_key"])
        except (KeyError, specs.BatchedOpSpecError):
            return None
        if not (
            isinstance(function_spec, specs.ElementwiseFunctionSpec)
            and function_spec.function_class is Linear
            and node.function_type == "Linear"
            and tuple(node.params)
            == ("slope", "intercept", "scale", "offset")
        ):
            return None
        bound = tuple(
            (binding, parameter_for(node, binding.arg))
            for binding in function_spec.params
        )
        if (
            tuple(binding.arg for binding, _ in bound)
            != ("slope", "intercept", "scale", "offset")
            or any(parameter is None for _, parameter in bound)
            or any(
                parameter.owner_scope != binding.scope
                or parameter.runtime_mutable
                for binding, parameter in bound
            )
        ):
            return None
        defaults = tuple(parameter.default for _, parameter in bound)
        if not (
            all(type(value) is float and np.isfinite(value) for value in defaults)
            and defaults[0] == 1.0
            and defaults[2:] == (1.0, 0.0)
            and 0.0 <= defaults[1] <= FP32_EXACT_INTEGER_LIMIT
            and defaults[1].is_integer()
        ):
            return None
        return int(defaults[1])

    modulation = graph.modulations[0]
    effective = graph.effective_parameters[0]
    dynamic_finished = tuple(
        value
        for value in graph.finished_values
        if value.predicate_kind == "dynamic"
    )
    counted_finished = tuple(
        value
        for value in graph.finished_values
        if value.predicate_kind
        == "execution_count_at_least_effective_parameter"
    )
    if len(dynamic_finished) != 1 or len(counted_finished) != 1:
        return False
    stepper_finished = counted_finished[0]
    terminator_finished = dynamic_finished[0]
    try:
        source = graph.node(modulation.source)
        controller = graph.node(modulation.controller)
        stepper = graph.node(modulation.target)
        terminator = graph.node(terminator_finished.node)
        stepper_spec = lookup_spec(stepper.attrs["spec_key"])
        terminator_spec = lookup_spec(terminator.attrs["spec_key"])
    except (KeyError, specs.BatchedOpSpecError):
        return False

    expected_count_attrs = {
        "effective_parameter_id": effective.effective_parameter_id,
        "target_parameter_port_id": effective.target_parameter_port_id,
        "rounding": "ceil",
        "minimum": 1,
        "maximum": FP32_EXACT_INTEGER_LIMIT,
    }
    if not (
        modulation.modulation_id == 0
        and modulation.mode == "OVERRIDE"
        and modulation.width == 1
        and modulation.dtype == "float32"
        and modulation.absorbed_identity_chain is True
        and modulation.source == source.name
        and modulation.source_component_id == source.component_id
        and modulation.controller == controller.name
        and modulation.controller_component_id == controller.component_id
        and modulation.target == stepper.name
        and modulation.target_component_id == stepper.component_id
        and modulation.target_parameter == "termination_threshold"
        and modulation.effective_parameter_id
        == effective.effective_parameter_id
        and modulation.target_parameter_port_id
        == effective.target_parameter_port_id
        and effective.effective_parameter_id == 0
        and effective.target == stepper.name
        and effective.target_component_id == stepper.component_id
        and effective.target_parameter == "termination_threshold"
        and effective.base_value == (1.0,)
        and effective.initial_modulation_value == (1.0,)
        and effective.width == 1
        and effective.dtype == "float32"
        and effective.storage == "lane_persistent"
        and effective.reset == "Never"
        and effective.update_event == "after_controller_execution"
        and effective.sample_event == "at_target_parameter_update"
        and stepper_finished.value_id == 0
        and stepper_finished.node == stepper.name
        and stepper_finished.component_id == stepper.component_id
        and stepper_finished.width == 1
        and stepper_finished.dtype == "bool"
        and stepper_finished.storage == "combinational"
        and stepper_finished.producer_consideration_set_id == 2
        and exact_attrs(stepper_finished.attrs, expected_count_attrs)
        and terminator_finished.value_id == 1
        and terminator_finished.node == terminator.name
        and terminator_finished.component_id == terminator.component_id
        and terminator_finished.width == 1
        and terminator_finished.dtype == "bool"
        and terminator_finished.storage == "combinational"
        and terminator_finished.producer_consideration_set_id == 4
        and exact_attrs(terminator_finished.attrs, {})
    ):
        return False

    try:
        controller_spec = lookup_spec(
            modulation.controller_function_spec_key
        )
    except specs.BatchedOpSpecError:
        return False
    controller_arguments = ("slope", "intercept", "scale", "offset")
    controller_intercept = registered_count_controller_intercept(controller)
    if not (
        source.component_type == "ProcessingMechanism"
        and source.input_width == 1
        and source.output_width == 1
        and source.attrs.get("spec_kind") == "elementwise"
        and registered_affine_count_source(source)
        and controller.component_type == "ControlMechanism"
        and controller.input_width == 1
        and controller.output_width == 1
        and controller.attrs.get("spec_kind") == "control"
        and controller.attrs.get("control_function") == "registered"
        and controller_intercept is not None
        and isinstance(controller_spec, specs.ElementwiseFunctionSpec)
        and controller_spec.function_class is Linear
        and tuple(
            binding.argument for binding in modulation.controller_param_bindings
        )
        == controller_arguments
        and tuple(
            binding.parameter for binding in modulation.controller_param_bindings
        )
        == tuple(controller.params[argument] for argument in controller_arguments)
        and tuple(
            binding.parameter_id
            for binding in modulation.controller_param_bindings
        )
        == tuple(
            parameters_by_name[controller.params[argument]].parameter_id
            for argument in controller_arguments
        )
        and controller.input_port_ids
        == (modulation.controller_input_port_id,)
        and controller.output_port_ids
        == (modulation.control_signal_port_id,)
        and source.output_port_ids == (modulation.source_port_id,)
        and dict(stepper.parameter_port_ids).get("termination_threshold")
        == modulation.target_parameter_port_id
        and exact_attrs(
            controller.attrs.get("absorbed_control"),
            {
                "source": source.name,
                "target": stepper.name,
                "parameter": "termination_threshold",
                "modulation": "OVERRIDE",
            },
        )
    ):
        return False

    absorbed_by_id = {
        projection.projection_id: projection
        for projection in graph.absorbed_projections
    }
    if set(absorbed_by_id) != {0, 1}:
        return False
    monitor_projection = absorbed_by_id.get(modulation.monitor_projection_id)
    control_projection = absorbed_by_id.get(modulation.control_projection_id)
    if not (
        monitor_projection is not None
        and monitor_projection.kind == "MappingProjection"
        and monitor_projection.sender == source.name
        and monitor_projection.sender_component_id == source.component_id
        and monitor_projection.sender_port_id == modulation.source_port_id
        and monitor_projection.receiver == controller.name
        and monitor_projection.receiver_component_id == controller.component_id
        and monitor_projection.receiver_port_id
        == modulation.controller_input_port_id
        and monitor_projection.width == 1
        and monitor_projection.reason == "typed_scalar_override"
        and monitor_projection.initial_value == ()
        and control_projection is not None
        and control_projection.kind == "ControlProjection"
        and control_projection.sender == controller.name
        and control_projection.sender_component_id == controller.component_id
        and control_projection.sender_port_id
        == modulation.control_signal_port_id
        and control_projection.receiver == stepper.name
        and control_projection.receiver_component_id == stepper.component_id
        and control_projection.receiver_port_id
        == modulation.target_parameter_port_id
        and control_projection.width == 1
        and control_projection.reason == "typed_scalar_override"
        and control_projection.initial_value == (1.0,)
    ):
        return False

    control_nodes = tuple(
        node for node in graph.nodes if node.component_type == "ControlMechanism"
    )
    if len(control_nodes) != 2 or controller not in control_nodes:
        return False
    threshold_controller = next(
        (node for node in control_nodes if node.component_id != controller.component_id),
        None,
    )
    threshold_control = (
        threshold_controller.attrs.get("absorbed_control")
        if threshold_controller is not None
        else None
    )
    if (
        threshold_controller is None
        or not isinstance(threshold_control, Mapping)
        or not (
            threshold_controller.function_type == "Identity"
            and threshold_controller.input_width == 1
            and threshold_controller.output_width == 1
            and not threshold_controller.params
            and not threshold_controller.parameter_port_ids
            and set(threshold_controller.attrs)
            == {
                "input_ports",
                "output_ports",
                "absorbed_control",
                "absorbed_control_initial_value",
            }
            and type(
                threshold_controller.attrs["absorbed_control_initial_value"]
            ) is float
            and threshold_controller.attrs["absorbed_control_initial_value"]
            == 1.0
            and exact_attrs(
                threshold_control,
                {
                    "source": threshold_control.get("source"),
                    "target": terminator.name,
                    "parameter": "threshold",
                    "modulation": "OVERRIDE",
                },
            )
            and type(threshold_control["source"]) is str
            and bool(threshold_control["source"])
        )
    ):
        return False

    if not (
        isinstance(stepper_spec, specs.MechanismOpSpec)
        and stepper_spec.can_step
        and not stepper_spec.is_terminator
        and stepper.component_type == "LCAMechanism"
        and stepper.function_type == "Logistic"
        and stepper.input_width == 2
        and stepper.output_width == 2
        and stepper.attrs.get("spec_kind") == "mechanism"
        and stepper.attrs.get("termination_input_node") == source.name
        and stepper.attrs.get("initialize_noise_sender") is True
        and stepper.attrs.get("diagnostics") == ()
        and stepper.attrs.get("rng_streams") == ()
        and isinstance(terminator_spec, specs.MechanismOpSpec)
        and terminator_spec.can_step
        and terminator_spec.is_terminator
        and terminator.component_type == "DDM"
        and terminator.function_type == "DriftDiffusionIntegrator"
        and terminator.input_width == 1
        and terminator.output_width == 1
        and terminator.attrs.get("spec_kind") == "mechanism"
        and terminator.attrs.get("diagnostics") == ("truncated",)
        and terminator.attrs.get("rng_streams")
        == (("rng", "MAX_STEPS", 1),)
        and tuple(state.name for state in terminator_spec.trial_states)
        == ("value", "steps", "finished")
        and terminator_spec.finished_output == "finished"
    ):
        return False

    rng = graph.rng_streams[0]
    rng_decl = terminator_spec.rng[0] if len(terminator_spec.rng) == 1 else None
    if not (
        rng_decl is not None
        and rng.stream_id == 0
        and rng.name == f"{terminator.name}.{rng_decl.name}"
        and rng.node == terminator.name
        and rng.component_id == terminator.component_id
        and rng.width == 1
        and rng.step_extent == "MAX_STEPS"
        and rng_decl.name == "rng"
        and rng_decl.width == 1
        and rng_decl.step_extent == "MAX_STEPS"
    ):
        return False

    ddm_parameters = {
        argument: parameter_for(terminator, argument)
        for argument in terminator.params
    }
    stepper_noise = parameter_for(stepper, "noise")
    threshold = ddm_parameters.get("threshold")
    noise = ddm_parameters.get("noise")
    threshold_collapse = ddm_parameters.get("threshold_collapse")
    starting_value = ddm_parameters.get("starting_value")
    offset = ddm_parameters.get("offset")
    time_step_size = ddm_parameters.get("time_step_size")
    frozen_ddm_parameters = (noise, starting_value, offset)
    frozen_reason = "first coevolving DDM boundary is frozen in KernelIR"
    ddm_parameters_are_pre_freeze = all(
        parameter is not None
        and parameter.runtime_mutable is True
        and parameter.runtime_constraint == ""
        for parameter in frozen_ddm_parameters
    )
    ddm_parameters_are_post_freeze = all(
        parameter is not None
        and parameter.runtime_mutable is False
        and parameter.runtime_constraint == frozen_reason
        for parameter in frozen_ddm_parameters
    )
    threshold_source_name = threshold_control["source"]
    if not (
        stepper_noise is not None
        and stepper_noise.name == f"{stepper.name}.noise"
        and stepper_noise.aliases
        == _node_param_aliases(stepper.name, "noise")
        and type(stepper_noise.default) is float
        and np.isfinite(stepper_noise.default)
        and np.isfinite(np.float32(stepper_noise.default))
        and stepper_noise.minimum is None
        and stepper_noise.minimum_inclusive is True
        and stepper_noise.maximum is None
        and stepper_noise.maximum_inclusive is True
        and stepper_noise.owner_component_id == stepper.component_id
        and stepper_noise.owner_scope == "mechanism"
        and stepper_noise.runtime_mutable is True
        and stepper_noise.runtime_constraint == ""
        and all(parameter is not None for parameter in ddm_parameters.values())
        and threshold is not None
        and threshold.name == f"{terminator.name}.threshold"
        and threshold.aliases
        == (
            "ddm.threshold",
            "DDM.threshold",
            *_node_param_aliases(terminator.name, "threshold"),
            *_node_param_aliases(threshold_source_name, "intercept"),
        )
        and type(threshold.default) is float
        and np.isfinite(threshold.default)
        and threshold.default >= 0.0
        and threshold.minimum == 0.0
        and threshold.minimum_inclusive is True
        and threshold.maximum is None
        and threshold.maximum_inclusive is True
        and threshold.owner_component_id == terminator.component_id
        and threshold.owner_scope == "function"
        and threshold.runtime_mutable is True
        and threshold.runtime_constraint == ""
        and noise is not None
        and type(noise.default) is float
        and np.isfinite(noise.default)
        and noise.default >= 0.0
        and threshold_collapse is not None
        and threshold_collapse.name
        == f"{terminator.name}.threshold_collapse"
        and threshold_collapse.aliases
        == (
            "ddm.threshold_collapse",
            "DDM.threshold_collapse",
            *_node_param_aliases(terminator.name, "threshold_collapse"),
            *_node_param_aliases(
                threshold_source_name,
                "offset-integrator_function",
            ),
        )
        and type(threshold_collapse.default) is float
        and np.isfinite(threshold_collapse.default)
        and threshold_collapse.default <= 0.0
        and threshold_collapse.minimum is None
        and threshold_collapse.minimum_inclusive is True
        and threshold_collapse.maximum == 0.0
        and threshold_collapse.maximum_inclusive is True
        and threshold_collapse.owner_component_id
        == terminator.component_id
        and threshold_collapse.owner_scope == "function"
        and threshold_collapse.runtime_mutable is True
        and threshold_collapse.runtime_constraint == ""
        and starting_value is not None
        and type(starting_value.default) is float
        and starting_value.default == 0.0
        and offset is not None
        and type(offset.default) is float
        and offset.default == 0.0
        and time_step_size is not None
        and type(time_step_size.default) is float
        and np.isfinite(time_step_size.default)
        and time_step_size.default > 0.0
        and (
            ddm_parameters_are_pre_freeze
            or ddm_parameters_are_post_freeze
        )
    ):
        return False

    scheduler_by_id = {
        condition.component_id: condition for condition in graph.scheduler
    }
    if set(scheduler_by_id) != set(node_component_ids):
        return False

    def condition_matches(
        condition,
        node,
        condition_type,
        consideration_set_id,
        *,
        dependencies=(),
        dependency_component_ids=(),
        finished_value_ids=(),
        attrs=None,
    ) -> bool:
        return bool(
            condition is not None
            and condition.node == node.name
            and condition.component_id == node.component_id
            and condition.condition_type == condition_type
            and condition.region == "pass"
            and condition.consideration_set_id == consideration_set_id
            and condition.dependencies == dependencies
            and condition.dependency_component_ids == dependency_component_ids
            and condition.finished_value_ids == finished_value_ids
            and exact_attrs(condition.attrs, {} if attrs is None else attrs)
        )

    when_stepper_finished = tuple(
        condition
        for condition in graph.scheduler
        if condition.condition_type == "WhenFinished"
        and condition.dependencies == (stepper.name,)
        and condition.dependency_component_ids == (stepper.component_id,)
        and condition.finished_value_ids == (stepper_finished.value_id,)
    )
    if len(when_stepper_finished) != 3:
        return False
    drift_conditions = tuple(
        condition
        for condition in when_stepper_finished
        if condition.component_id
        not in {threshold_controller.component_id, terminator.component_id}
    )
    if len(drift_conditions) != 1:
        return False
    drift_condition = drift_conditions[0]
    try:
        drift = graph.node(drift_condition.node)
        drift_spec = lookup_spec(drift.attrs["spec_key"])
    except (KeyError, specs.BatchedOpSpecError):
        return False

    gate_conditions = tuple(
        condition
        for condition in graph.scheduler
        if condition.condition_type == "WhenFinished"
        and condition.dependencies == (terminator.name,)
        and condition.dependency_component_ids == (terminator.component_id,)
        and condition.finished_value_ids == (terminator_finished.value_id,)
    )
    if len(gate_conditions) != 2:
        return False
    try:
        gates = tuple(graph.node(condition.node) for condition in gate_conditions)
    except KeyError:
        return False

    at_pass_zero = {
        "pass_index": 0,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    origin_conditions = tuple(
        condition
        for condition in graph.scheduler
        if condition.condition_type == "AtPass"
        and condition.consideration_set_id == 0
    )
    if len(origin_conditions) != 4:
        return False
    try:
        origins = tuple(graph.node(condition.node) for condition in origin_conditions)
    except KeyError:
        return False

    stepper_inputs = tuple(
        projection
        for projection in graph.projections
        if projection.receiver_component_id == stepper.component_id
    )
    if len(stepper_inputs) != 1:
        return False
    task_projection = stepper_inputs[0]
    try:
        task = graph.node(task_projection.sender)
    except KeyError:
        return False
    task_condition = scheduler_by_id.get(task.component_id)
    task_at_pass = {
        "pass_index": controller_intercept,
        "time_scale": "ENVIRONMENT_STATE_UPDATE",
    }
    warmup = graph.metadata.get("coevolve_warmup")
    if (
        task not in origins
        or task_condition is None
        or controller_intercept is None
        or type(warmup) is not int
        or warmup != controller_intercept
    ):
        return False

    role_ids = {
        *(node.component_id for node in origins),
        threshold_controller.component_id,
        controller.component_id,
        stepper.component_id,
        drift.component_id,
        terminator.component_id,
        *(node.component_id for node in gates),
    }
    if len(role_ids) != len(graph.nodes) or source not in origins:
        return False

    when_finished_attrs = {"predicate": "is_finished"}
    if not (
        all(
            condition_matches(
                scheduler_by_id[node.component_id],
                node,
                "AtPass",
                0,
                attrs=at_pass_zero,
            )
            for node in origins
            if node is not task
        )
        and condition_matches(
            task_condition,
            task,
            "AtPass",
            0,
            attrs=task_at_pass,
        )
        and condition_matches(
            scheduler_by_id[controller.component_id],
            controller,
            "AtPass",
            1,
            attrs=at_pass_zero,
        )
        and condition_matches(
            scheduler_by_id[threshold_controller.component_id],
            threshold_controller,
            "WhenFinished",
            1,
            dependencies=(stepper.name,),
            dependency_component_ids=(stepper.component_id,),
            finished_value_ids=(stepper_finished.value_id,),
            attrs=when_finished_attrs,
        )
        and condition_matches(
            scheduler_by_id[stepper.component_id],
            stepper,
            "Always",
            2,
        )
        and condition_matches(
            scheduler_by_id[drift.component_id],
            drift,
            "WhenFinished",
            3,
            dependencies=(stepper.name,),
            dependency_component_ids=(stepper.component_id,),
            finished_value_ids=(stepper_finished.value_id,),
            attrs=when_finished_attrs,
        )
        and condition_matches(
            scheduler_by_id[terminator.component_id],
            terminator,
            "WhenFinished",
            4,
            dependencies=(stepper.name,),
            dependency_component_ids=(stepper.component_id,),
            finished_value_ids=(stepper_finished.value_id,),
            attrs=when_finished_attrs,
        )
        and all(
            condition_matches(
                scheduler_by_id[gate.component_id],
                gate,
                "WhenFinished",
                5,
                dependencies=(terminator.name,),
                dependency_component_ids=(terminator.component_id,),
                finished_value_ids=(terminator_finished.value_id,),
                attrs=when_finished_attrs,
            )
            for gate in gates
        )
    ):
        return False

    ordered_role_ids = (
        tuple(
            node.component_id
            for node in graph.nodes
            if node.component_id in {origin.component_id for origin in origins}
        ),
        tuple(
            node.component_id
            for node in graph.nodes
            if node.component_id
            in {threshold_controller.component_id, controller.component_id}
        ),
        (stepper.component_id,),
        (drift.component_id,),
        (terminator.component_id,),
        tuple(
            node.component_id
            for node in graph.nodes
            if node.component_id in {gate.component_id for gate in gates}
        ),
    )
    consideration_sets = tuple(
        sorted(
            graph.consideration_sets,
            key=lambda item: item.consideration_set_id,
        )
    )
    if any(
        component_id not in nodes_by_component_id
        for item in consideration_sets
        for component_id in item.component_ids
    ):
        return False
    if not (
        tuple(item.consideration_set_id for item in consideration_sets)
        == tuple(range(6))
        and all(
            item.region == "pass"
            and item.inputs_frozen is True
            and item.component_ids == component_ids
            and item.nodes
            == tuple(
                nodes_by_component_id[component_id].name
                for component_id in component_ids
            )
            for item, component_ids in zip(
                consideration_sets,
                ordered_role_ids,
            )
        )
    ):
        return False

    if (
        tuple(
            (region.name, region.kind, region.time_scale, region.parent)
            for region in graph.schedule_regions
        )
        != (
            ("trial", "trial", "ENVIRONMENT_STATE_UPDATE", ""),
            ("pass", "pass", "PASS", "trial"),
        )
        or len(graph.termination) != 2
    ):
        return False
    termination_by_scale = {
        termination.time_scale: termination for termination in graph.termination
    }
    trial_termination = termination_by_scale.get("ENVIRONMENT_STATE_UPDATE")
    sequence_termination = termination_by_scale.get("ENVIRONMENT_SEQUENCE")
    if not (
        len(termination_by_scale) == 2
        and trial_termination is not None
        and trial_termination.condition_type == "AllHaveRun"
        and trial_termination.dependency_component_ids == node_component_ids
        and exact_attrs(trial_termination.attrs, {})
        and sequence_termination is not None
        and sequence_termination.condition_type == "Never"
        and sequence_termination.dependency_component_ids == ()
        and exact_attrs(sequence_termination.attrs, {})
    ):
        return False

    stepper_states = tuple(
        state for state in graph.states if state.component_id == stepper.component_id
    )
    if not (
        len(graph.states) == 3
        and len(stepper_states) == 3
        and tuple(state.state_id for state in stepper_states) == (0, 1, 2)
        and tuple(
            (
                state.name.rsplit(".", 1)[-1],
                state.node,
                state.width,
                state.initial_value,
            )
            for state in stepper_states
        )
        == (
            ("pre", stepper.name, 2, (0.0, 0.0)),
            ("act", stepper.name, 2, (0.0, 0.0)),
            ("initialized", stepper.name, 1, (0.0,)),
        )
        and len(graph.resets) == 1
        and graph.resets[0].node == stepper.name
        and graph.resets[0].component_id == stepper.component_id
        and graph.resets[0].state_ids == (0, 1, 2)
        and graph.resets[0].condition_type == "Never"
        and graph.resets[0].region == "trial"
        and exact_attrs(graph.resets[0].attrs, {})
    ):
        return False

    projections = graph.projections
    if tuple(projection.projection_id for projection in projections) != tuple(
        range(len(projections))
    ):
        return False
    try:
        if not all(
            isinstance(lookup_spec(projection.spec_key), specs.DenseProjectionSpec)
            for projection in projections
        ):
            return False
    except specs.BatchedOpSpecError:
        return False

    def matching_projections(sender, receiver):
        return tuple(
            projection
            for projection in projections
            if projection.sender_component_id == sender.component_id
            and projection.receiver_component_id == receiver.component_id
            and projection.sender == sender.name
            and projection.receiver == receiver.name
        )

    remaining_origins = tuple(
        origin
        for origin in origins
        if origin.component_id not in {source.component_id, task.component_id}
    )
    if len(remaining_origins) != 2:
        return False
    correct = next((node for node in remaining_origins if node.output_width == 1), None)
    stimulus = next((node for node in remaining_origins if node.output_width == 4), None)
    if correct is None or stimulus is None:
        return False

    lca_to_drift = matching_projections(stepper, drift)
    correct_to_drift = matching_projections(correct, drift)
    stimulus_to_drift = matching_projections(stimulus, drift)
    drift_to_ddm = matching_projections(drift, terminator)
    if not all(
        len(matches) == 1
        for matches in (
            lca_to_drift,
            correct_to_drift,
            stimulus_to_drift,
            drift_to_ddm,
        )
    ):
        return False

    decision_links = tuple(
        projection
        for projection in projections
        if projection.sender_component_id == terminator.component_id
        and projection.sender_port == "DECISION_OUTCOME"
        and projection.receiver_component_id in {gate.component_id for gate in gates}
    )
    response_links = tuple(
        projection
        for projection in projections
        if projection.sender_component_id == terminator.component_id
        and projection.sender_port == "RESPONSE_TIME"
        and projection.receiver_component_id in {gate.component_id for gate in gates}
    )
    if len(decision_links) != 1 or len(response_links) != 1:
        return False
    try:
        decision_gate = graph.node(decision_links[0].receiver)
        response_gate = graph.node(response_links[0].receiver)
    except KeyError:
        return False
    cue_to_response = matching_projections(source, response_gate)
    if (
        decision_gate is response_gate
        or len(cue_to_response) != 1
        or any(
            projection.receiver_component_id == decision_gate.component_id
            and projection is not decision_links[0]
            for projection in projections
        )
        or {
            projection.sender_component_id
            for projection in projections
            if projection.receiver_component_id == response_gate.component_id
        }
        != {source.component_id, terminator.component_id}
    ):
        return False

    def matrix_matches(projection, expected) -> bool:
        actual = np.asarray(projection.matrix)
        expected_array = np.asarray(expected, dtype=np.float32)
        return bool(
            actual.dtype == np.dtype(np.float32)
            and actual.shape == expected_array.shape
            and np.array_equal(actual, expected_array)
        )

    expected_correct = np.zeros((1, 7), dtype=np.float32)
    expected_correct[0, 6] = 1.0
    expected_stimulus = np.zeros((4, 7), dtype=np.float32)
    expected_stimulus[:, :4] = np.eye(4, dtype=np.float32)
    expected_lca = np.zeros((2, 7), dtype=np.float32)
    expected_lca[:, 4:6] = np.eye(2, dtype=np.float32)
    if not (
        matrix_matches(task_projection, np.eye(2, dtype=np.float32))
        and matrix_matches(correct_to_drift[0], expected_correct)
        and matrix_matches(stimulus_to_drift[0], expected_stimulus)
        and matrix_matches(lca_to_drift[0], expected_lca)
        and matrix_matches(drift_to_ddm[0], ((1.0,),))
        and matrix_matches(decision_links[0], ((1.0,),))
        and matrix_matches(response_links[0], ((1.0,),))
        and matrix_matches(cue_to_response[0], ((time_step_size.default,),))
    ):
        return False

    if not (
        isinstance(drift_spec, specs.MechanismOpSpec)
        and not drift_spec.can_step
        and not drift_spec.is_terminator
        and not drift_spec.states
        and not drift_spec.trial_states
        and not drift_spec.rng
        and drift.component_type == "ProcessingMechanism"
        and drift.function_type == "UserDefinedFunction"
        and drift.input_width == 7
        and drift.output_width == 1
        and not drift.params
        and drift.attrs.get("spec_kind") == "mechanism"
        and drift.attrs.get("rng_streams") == ()
        and drift.attrs.get("diagnostics") == ()
        and task.component_type == "TransferMechanism"
        and task.input_width == 2
        and task.output_width == 2
        and task.attrs.get("integrator_pre") == (1.0, 0.0)
        and registered_identity_linear(task, frozen=False)
        and correct.component_type == "ProcessingMechanism"
        and correct.input_width == 1
        and registered_identity_linear(correct, frozen=False)
        and stimulus.component_type == "ProcessingMechanism"
        and stimulus.input_width == 4
        and stimulus.output_width == 4
        and registered_identity_linear(stimulus, frozen=False)
        and all(
            gate.component_type == "ProcessingMechanism"
            and gate.input_width == 1
            and gate.output_width == 1
            and registered_identity_linear(gate, frozen=False)
            for gate in gates
        )
    ):
        return False

    origin_ids_in_graph_order = tuple(
        node.component_id for node in graph.nodes if node in origins
    )
    if any(
        component_id not in nodes_by_component_id
        for component_id in (
            *(input_spec.component_id for input_spec in graph.inputs),
            *(output.component_id for output in graph.outputs),
        )
    ):
        return False
    if not (
        tuple(input_spec.component_id for input_spec in graph.inputs)
        == origin_ids_in_graph_order
        and all(
            input_spec.node == nodes_by_component_id[input_spec.component_id].name
            and input_spec.width
            == nodes_by_component_id[input_spec.component_id].input_width
            and input_spec.port_id
            == nodes_by_component_id[input_spec.component_id].input_port_ids[0]
            for input_spec in graph.inputs
        )
        and tuple(output.component_id for output in graph.outputs)
        == (decision_gate.component_id, response_gate.component_id)
        and tuple(output.width for output in graph.outputs) == (1, 1)
        and tuple((output.flat_start, output.flat_stop) for output in graph.outputs)
        == ((0, 1), (1, 2))
        and all(
            output.node == nodes_by_component_id[output.component_id].name
            and output.port_id
            == nodes_by_component_id[output.component_id].output_port_ids[0]
            for output in graph.outputs
        )
    ):
        return False

    controller_ids = {
        controller.component_id,
        threshold_controller.component_id,
    }
    return bool(
        graph.execution_order
        == tuple(
            node.name
            for node in graph.nodes
            if node.component_id not in controller_ids
        )
        and controller.name not in graph.execution_order
        and threshold_controller.name not in graph.execution_order
        and all(
            "onset_step" not in node.attrs
            for node in graph.nodes
            if node is not task
        )
        and (
            task.attrs.get("onset_step", 0) == controller_intercept
            and (
                controller_intercept > 0
                or "onset_step" not in task.attrs
            )
        )
        and not any(
            projection.sender_component_id == threshold_controller.component_id
            or projection.receiver_component_id == threshold_controller.component_id
            for projection in graph.projections
        )
    )


def _dynamic_controlled_finished_component_ids(
    composition,
    nodes,
    component_ids,
    consideration_set_ids,
) -> frozenset[int]:
    """Return targets whose lane-varying finished edge has complete typed data."""

    active_node_ids = {id(node) for node in nodes}
    targets = set()
    for control in nodes:
        if type(control) is not ControlMechanism:
            continue
        chain, diagnostic = _resolve_control_chain(control, composition)
        if (
            diagnostic is None
            and chain is not None
            and _typed_dynamic_control_chain_supported(
                composition,
                control,
                chain,
                active_node_ids,
                component_ids,
                consideration_set_ids,
            )
        ):
            targets.add(component_ids[id(chain.target)])
    return frozenset(targets)


def _modulation_ir_specs(
    composition,
    nodes,
    node_specs: list[BatchedNodeSpec],
    params: _ParamBuilder,
    component_ids,
    port_ids,
    consideration_set_ids,
) -> tuple[
    tuple[BatchedAbsorbedProjectionSpec, ...],
    tuple[BatchedEffectiveParameterSpec, ...],
    tuple[BatchedModulationSpec, ...],
    dict[int, object],
    dict[int, object],
]:
    """Snapshot exact scalar LCA ``OVERRIDE`` chains as typed data.

    Execution remains fail-closed.  This declaration replaces the informal
    name dictionary as the semantic description of the absorbed monitor,
    controller, ControlSignal, and ControlProjection chain.  Numeric endpoint
    identities are resolved from the same lowering-local maps as processing
    projections; live objects remain only in ``BatchedComponentBindings``.
    """

    active_node_ids = {id(node) for node in nodes}
    node_specs_by_id = {
        node.component_id: (index, node)
        for index, node in enumerate(node_specs)
    }
    effective_parameters = []
    modulations = []
    absorbed_projections = []
    absorbed_projection_bindings_by_id: dict[int, object] = {}
    bindings_by_id: dict[int, object] = {}
    for control in nodes:
        if type(control) is not ControlMechanism:
            continue
        chain, chain_diagnostic = _resolve_control_chain(control, composition)
        if chain_diagnostic is not None or chain is None:
            continue
        signal = chain.signal
        control_projection = chain.control_projection
        source_port = chain.source_port
        controller_input_port = chain.controller_input_port
        target_parameter_port = chain.target_parameter_port
        source = chain.source
        target = chain.target
        if not _typed_dynamic_control_chain_supported(
            composition,
            control,
            chain,
            active_node_ids,
            component_ids,
            consideration_set_ids,
        ):
            continue

        base_value = _finite_fp32_scalar_value(
            _parameter_value(target, chain.target_port, None)
        )
        initial_modulation_value = _finite_fp32_scalar_value(
            _parameter_default_value(control_projection, "value", None)
        )
        if base_value is None or initial_modulation_value is None:
            continue

        function = getattr(control, "function", None)
        function_spec = specs.function_spec_for(function)
        if type(function) is Identity:
            function_spec_key = ""
            controller_param_bindings = ()
            control_function_attrs = {
                "spec_kind": "control",
                "control_function": "identity",
            }
        elif (
            type(function) is Linear
            and function_spec is not None
            and _function_parameter_support_diagnostic(
                _node_name(control),
                function,
            ) is None
        ):
            function_spec_key = function_spec.key
            controller_param_bindings = []
            for binding in function_spec.params:
                public_name = f"{_node_name(control)}.{binding.arg}"
                parameter_name = params.add(
                    public_name,
                    binding.resolve(function),
                    aliases=_node_param_aliases(
                        _node_name(control),
                        binding.arg,
                    ),
                    parameter=_bound_parameter(binding, function),
                    minimum=binding.minimum,
                    minimum_inclusive=binding.minimum_inclusive,
                    maximum=binding.maximum,
                    maximum_inclusive=binding.maximum_inclusive,
                    owner_component_id=component_ids[id(control)],
                    owner_scope=binding.scope,
                )
                params.freeze(
                    parameter_name,
                    "absorbed control parameters are frozen in KernelIR",
                )
                controller_param_bindings.append(
                    params.binding(binding.arg, parameter_name)
                )
            controller_param_bindings = tuple(controller_param_bindings)
            control_function_attrs = {
                "spec_kind": "control",
                "spec_key": function_spec_key,
                "control_function": "registered",
            }
        else:
            continue

        controller_component_id = component_ids[id(control)]
        source_component_id = component_ids[id(source)]
        try:
            node_index, node_spec = node_specs_by_id[controller_component_id]
            _, source_node_spec = node_specs_by_id[source_component_id]
            endpoint_port_ids = tuple(
                port_ids[id(port)]
                for port in (
                    source_port,
                    controller_input_port,
                    signal,
                    target_parameter_port,
                )
            )
        except KeyError:
            continue
        dynamic_coevolving_source = (
            _is_unmodeled_coevolving_lca_termination(
                composition,
                target,
            )
        )
        for argument, parameter_name in source_node_spec.params.items():
            if dynamic_coevolving_source and argument in {
                "slope",
                "intercept",
            }:
                continue
            params.freeze(
                parameter_name,
                (
                    "absorbed affine source for typed OVERRIDE modulation"
                    if dynamic_coevolving_source
                    else "absorbed identity source for typed OVERRIDE modulation"
                ),
            )
        node_specs[node_index] = replace(
            node_spec,
            params={
                binding.argument: binding.parameter
                for binding in controller_param_bindings
            },
            attrs={**node_spec.attrs, **control_function_attrs},
        )
        source_port_id, controller_input_port_id, signal_port_id, target_port_id = (
            endpoint_port_ids
        )
        modulation_id = len(modulations)
        monitor_projection_id = len(absorbed_projections)
        absorbed_projections.append(
            BatchedAbsorbedProjectionSpec(
                projection_id=monitor_projection_id,
                name=getattr(
                    chain.monitor_projection,
                    "name",
                    "MappingProjection",
                ),
                kind="MappingProjection",
                sender=_node_name(source),
                sender_component_id=source_component_id,
                sender_port=getattr(source_port, "name", ""),
                sender_port_id=source_port_id,
                receiver=_node_name(control),
                receiver_component_id=controller_component_id,
                receiver_port=getattr(controller_input_port, "name", ""),
                receiver_port_id=controller_input_port_id,
            )
        )
        absorbed_projection_bindings_by_id[monitor_projection_id] = (
            chain.monitor_projection
        )
        control_projection_id = len(absorbed_projections)
        absorbed_projections.append(
            BatchedAbsorbedProjectionSpec(
                projection_id=control_projection_id,
                name=getattr(control_projection, "name", "ControlProjection"),
                kind="ControlProjection",
                sender=_node_name(control),
                sender_component_id=controller_component_id,
                sender_port=getattr(signal, "name", ""),
                sender_port_id=signal_port_id,
                receiver=_node_name(target),
                receiver_component_id=component_ids[id(target)],
                receiver_port=chain.target_port,
                receiver_port_id=target_port_id,
                initial_value=(initial_modulation_value,),
            )
        )
        absorbed_projection_bindings_by_id[control_projection_id] = (
            control_projection
        )
        effective_parameters.append(
            BatchedEffectiveParameterSpec(
                effective_parameter_id=modulation_id,
                target=_node_name(target),
                target_component_id=component_ids[id(target)],
                target_parameter=chain.target_port,
                target_parameter_port_id=target_port_id,
                base_value=(base_value,),
                initial_modulation_value=(initial_modulation_value,),
            )
        )
        modulations.append(
            BatchedModulationSpec(
                modulation_id=modulation_id,
                controller=_node_name(control),
                controller_component_id=controller_component_id,
                controller_input_port=getattr(controller_input_port, "name", ""),
                controller_input_port_id=controller_input_port_id,
                control_signal_port=getattr(signal, "name", ""),
                control_signal_port_id=signal_port_id,
                source=_node_name(source),
                source_component_id=source_component_id,
                source_port=getattr(source_port, "name", ""),
                source_port_id=source_port_id,
                target=_node_name(target),
                target_component_id=component_ids[id(target)],
                target_parameter=chain.target_port,
                target_parameter_port_id=target_port_id,
                effective_parameter_id=modulation_id,
                monitor_projection_id=monitor_projection_id,
                control_projection_id=control_projection_id,
                controller_function_spec_key=function_spec_key,
                controller_param_bindings=controller_param_bindings,
            )
        )
        bindings_by_id[modulation_id] = control_projection
    return (
        tuple(absorbed_projections),
        tuple(effective_parameters),
        tuple(modulations),
        absorbed_projection_bindings_by_id,
        bindings_by_id,
    )


def _controlled_finished_value_specs(
    finished_values: tuple[BatchedFinishedValueSpec, ...],
    modulations: tuple[BatchedModulationSpec, ...],
) -> tuple[BatchedFinishedValueSpec, ...]:
    """Bind scheduler-visible finished values to controlled effective values."""

    modulation_by_target = {
        modulation.target_component_id: modulation
        for modulation in modulations
        if modulation.target_parameter == "termination_threshold"
    }
    return tuple(
        replace(
            value,
            predicate_kind="execution_count_at_least_effective_parameter",
            attrs={
                "effective_parameter_id": modulation_by_target[
                    value.component_id
                ].effective_parameter_id,
                "target_parameter_port_id": modulation_by_target[
                    value.component_id
                ].target_parameter_port_id,
                "rounding": "ceil",
                "minimum": 1,
                "maximum": FP32_EXACT_INTEGER_LIMIT,
            },
        )
        if value.component_id in modulation_by_target
        else value
        for value in finished_values
    )


def _bound_parameter(binding, component):
    """Return the live PNL Parameter behind a signature binding when unique.

    A ``get=`` binding may still name its canonical live Parameter (for example
    LCA ``leak`` with an integrator fallback).  Truly derived values such as a
    collapsed DDM threshold have no matching Parameter and intentionally remain
    absent from the live-object sidecar while retaining an IR ``parameter_id``.
    """

    owners = []
    if binding.scope == "function":
        function = getattr(component, "function", None)
        if function is not None and hasattr(function, "parameters"):
            owners.append(function)
    owners.extend(
        (
            component,
            getattr(component, "integrator_function", None),
            getattr(component, "function", None),
        )
    )
    seen = set()
    for owner in owners:
        if owner is None or id(owner) in seen:
            continue
        seen.add(id(owner))
        parameters = getattr(owner, "parameters", None)
        if parameters is None:
            continue
        for name in (binding.pnl_name or binding.arg,) + binding.fallbacks:
            parameter = getattr(parameters, name, None)
            if parameter is not None:
                return parameter
    return None


def _port_identity_maps(nodes):
    """Assign deterministic lowering-local IDs to every port on each node."""

    port_ids: dict[int, int] = {}
    ports_by_id: dict[int, object] = {}
    for node in nodes:
        for collection_name in ("input_ports", "output_ports", "parameter_ports"):
            for port in tuple(getattr(node, collection_name, ())):
                object_id = id(port)
                if object_id in port_ids:
                    continue
                port_id = len(port_ids)
                port_ids[object_id] = port_id
                ports_by_id[port_id] = port
    return port_ids, ports_by_id


def _port_specs(nodes, component_ids, port_ids) -> tuple[BatchedPortSpec, ...]:
    """Declare every port ID, kind, owner, and width used by semantic IR."""

    result: list[BatchedPortSpec] = []
    for node in nodes:
        for collection_name in ("input_ports", "output_ports", "parameter_ports"):
            for port in tuple(getattr(node, collection_name, ())):
                result.append(
                    BatchedPortSpec(
                        port_id=port_ids[id(port)],
                        name=getattr(port, "name", type(port).__name__),
                        owner=_node_name(node),
                        owner_component_id=component_ids[id(node)],
                        kind=type(port).__name__,
                        width=_port_width(port),
                    )
                )
    return tuple(sorted(result, key=lambda spec: spec.port_id))


def _rng_stream_specs(nodes: list[BatchedNodeSpec]) -> tuple[BatchedRngStreamSpec, ...]:
    streams = []
    for node in nodes:
        for name, step_extent, width in node.attrs.get("rng_streams", ()):
            streams.append(
                BatchedRngStreamSpec(
                    name=f"{node.name}.{name}",
                    node=node.name,
                    width=int(width),
                    step_extent=step_extent,
                    component_id=node.component_id,
                    stream_id=len(streams),
                )
            )
    return tuple(streams)


def _node_spec(
    node,
    params: _ParamBuilder,
    model_kind: str | None,
    composition,
    *,
    component_id: int,
    port_ids,
) -> BatchedNodeSpec:
    component_type = type(node).__name__
    function = getattr(node, "function", None)
    function_type = type(function).__name__
    node_name = _node_name(node)
    input_port_specs = _input_port_attrs(node, port_ids)
    combine = input_port_specs[0][2] if input_port_specs else "sum"
    param_map: dict[str, str] = {}
    attrs: dict[str, Any] = {
        "input_ports": input_port_specs,
        "output_ports": tuple(port.name for port in getattr(node, "output_ports", [])),
    }
    if component_type == "ControlMechanism":
        attrs["absorbed_control"] = _absorbed_control_attrs(node)
        chain, diagnostic = _resolve_control_chain(node, composition)
        if (
            diagnostic is None
            and chain is not None
            and chain.target_port == "threshold"
        ):
            initial_value = _finite_fp32_scalar_value(
                _parameter_default_value(
                    chain.control_projection,
                    "value",
                    None,
                )
            )
            if initial_value is not None:
                attrs["absorbed_control_initial_value"] = initial_value
    # A delayed within-trial onset (AtPass(n>0)): the co-evolution loop withholds
    # this node's output until step n. Only meaningful/executable in a co-evolving
    # graph (AtPass(n>0) is rejected at the schedule level otherwise).
    onset = _onset_step(node, composition)
    if onset > 0:
        attrs["onset_step"] = onset

    mechanism_spec = specs.mechanism_spec_for(node)
    function_spec = specs.function_spec_for(function)
    output_width = _node_output_width(node, mechanism_spec)
    threshold_source = None
    threshold_source_parameter = None
    threshold_collapse_parameter = None
    if component_type == "DDM":
        # The threshold source is absent from GraphIR because its integrating
        # transfer and OVERRIDE controller are folded into the coevolving DDM
        # step.  Preserve the source's public fitting names as aliases for the
        # two kernel arguments that carry those exact semantics.
        threshold_binding = _exact_ddm_threshold_runtime_binding(
            composition,
            node,
        )
        if threshold_binding is not None:
            (
                threshold_source,
                threshold_source_parameter,
                threshold_collapse_parameter,
            ) = threshold_binding

    if mechanism_spec is not None:
        attrs["spec_kind"] = "mechanism"
        attrs["spec_key"] = mechanism_spec.key
        # Single-node model families (for example a lone DDM) keep unqualified
        # public parameter names; graph models use node-qualified names.
        single_node_model = model_kind is not None and model_kind != GRAPH_MODEL
        for binding in mechanism_spec.params:
            public_name = binding.arg if single_node_model else f"{node_name}.{binding.arg}"
            aliases = tuple(
                f"{prefix}.{binding.arg}" for prefix in mechanism_spec.param_alias_prefixes
            ) + _node_param_aliases(node_name, binding.arg)
            minimum = binding.minimum
            minimum_inclusive = binding.minimum_inclusive
            maximum = binding.maximum
            maximum_inclusive = binding.maximum_inclusive
            live_parameter = _bound_parameter(binding, node)
            if threshold_source is not None and binding.arg == "threshold":
                aliases += _node_param_aliases(
                    _node_name(threshold_source),
                    "intercept",
                )
                minimum = 0.0
                minimum_inclusive = True
                live_parameter = threshold_source_parameter
            elif (
                threshold_source is not None
                and binding.arg == "threshold_collapse"
            ):
                aliases += _node_param_aliases(
                    _node_name(threshold_source),
                    "offset-integrator_function",
                )
                maximum = 0.0
                maximum_inclusive = True
                live_parameter = threshold_collapse_parameter
            param_map[binding.arg] = params.add(
                public_name,
                binding.resolve(node),
                aliases=aliases,
                parameter=live_parameter,
                minimum=minimum,
                minimum_inclusive=minimum_inclusive,
                maximum=maximum,
                maximum_inclusive=maximum_inclusive,
                owner_component_id=component_id,
                owner_scope=binding.scope,
            )
        if mechanism_spec.extract_attrs is not None:
            attrs.update(mechanism_spec.extract_attrs(node, composition))
        if mechanism_spec.outputs is not None:
            attrs["op_outputs"] = tuple((decl.port, decl.width) for decl in mechanism_spec.outputs)
        else:
            primary_port = attrs["output_ports"][0] if attrs["output_ports"] else "RESULT"
            attrs["op_outputs"] = ((primary_port, output_width),)
        attrs["rng_streams"] = tuple(
            (decl.name, decl.step_extent, decl.width if decl.width is not None else output_width)
            for decl in mechanism_spec.rng
        )
        attrs["diagnostics"] = tuple(mechanism_spec.diagnostics)
    elif specs.passthrough_spec_for(node) is not None and function_spec is not None:
        attrs["spec_kind"] = "elementwise"
        attrs["spec_key"] = function_spec.key
        attrs["output_port_slices"] = _elementwise_output_port_slices(
            node,
            port_ids,
        )
        if component_type == "TransferMechanism":
            noise, _ = _transfer_noise_constant(node)
            clip, _ = _transfer_clip_bounds(node)
            if noise != 0.0:
                attrs["noise"] = noise
            if clip is not None:
                attrs["clip"] = clip
        # A fires-once, reset-each-trial integrator_mode transfer advances its
        # integrator a single step from its initializer, which is affine in the
        # input (integ = a*input + b).  Fold that affine step in front of the
        # node's function so the stateless elementwise path computes
        # function(a*input + b).
        affine = _integrating_transfer_affine(node, composition)
        if affine is not None:
            attrs["integrator_pre"] = affine
        for binding in function_spec.params:
            param_map[binding.arg] = params.add(
                f"{node_name}.{binding.arg}",
                binding.resolve(function),
                aliases=_node_param_aliases(node_name, binding.arg),
                parameter=_bound_parameter(binding, function),
                minimum=binding.minimum,
                minimum_inclusive=binding.minimum_inclusive,
                maximum=binding.maximum,
                maximum_inclusive=binding.maximum_inclusive,
                owner_component_id=component_id,
                owner_scope=binding.scope,
            )

    return BatchedNodeSpec(
        name=node_name,
        component_type=component_type,
        function_type=function_type,
        input_width=_input_width(node),
        output_width=output_width,
        combine=combine,
        params=param_map,
        attrs=attrs,
        component_id=component_id,
        input_port_ids=tuple(
            port_ids[id(port)]
            for port in tuple(getattr(node, "input_ports", ()))
        ),
        output_port_ids=tuple(
            port_ids[id(port)]
            for port in tuple(getattr(node, "output_ports", ()))
        ),
        parameter_port_ids=tuple(
            (getattr(port, "name", type(port).__name__), port_ids[id(port)])
            for port in tuple(getattr(node, "parameter_ports", ()))
        ),
    )


def _node_support_diagnostic(
    node,
    composition,
    *,
    component_id: int,
    finished_values_by_component_id: Mapping[int, BatchedFinishedValueSpec],
    dynamic_controlled_finished: bool = False,
) -> BatchedDiagnostic | None:
    function = getattr(node, "function", None)
    function_type = type(function).__name__
    node_name = _node_name(node)

    diagnostic = _single_input_support_diagnostic(node)
    if diagnostic is not None:
        return diagnostic
    diagnostic = _function_parameter_support_diagnostic(node_name, function)
    if diagnostic is not None:
        return diagnostic

    if type(node).__name__ == "TransferMechanism":
        _, diagnostic = _transfer_noise_constant(node)
        if diagnostic is not None:
            return diagnostic
        integrator_noise = _parameter_value(getattr(node, "integrator_function", None), "noise", 0.0)
        if _integrator_mode_enabled(node) and not _is_zero(integrator_noise):
            return BatchedDiagnostic(
                node_name,
                "unsupported TransferMechanism noise for batched v2",
                "integrator noise must be numeric zero",
            )
        _, diagnostic = _transfer_clip_bounds(node)
        if diagnostic is not None:
            return diagnostic

    mechanism_spec = specs.mechanism_spec_for(node)
    if mechanism_spec is not None:
        if mechanism_spec.function_class is not None and type(function) is not mechanism_spec.function_class:
            return BatchedDiagnostic(
                node_name,
                f"unsupported {mechanism_spec.label} function for batched v2",
                function_type,
            )
        if mechanism_spec.supports is not None:
            diagnostic = mechanism_spec.supports(node)
            if diagnostic is not None:
                return diagnostic
        if (
            type(node).__name__ == "DDM"
            and (
                type(getattr(node, "reset_stateful_function_when", None))
                is not AtTrialStart
                or not is_canonical_condition(
                    getattr(node, "reset_stateful_function_when", None)
                )
            )
        ):
            return BatchedDiagnostic(
                node_name,
                "unsupported DDM reset policy for batched v2",
                type(getattr(node, "reset_stateful_function_when", None)).__name__,
            )
        if type(node).__name__ == "DDM":
            diagnostic = _ddm_execution_support_diagnostic(node, composition)
            if diagnostic is not None:
                return diagnostic
        if type(node).__name__ == "LCAMechanism":
            diagnostic = _lca_execution_support_diagnostic(
                node,
                composition,
                component_id=component_id,
                finished_values_by_component_id=finished_values_by_component_id,
                dynamic_controlled_finished=dynamic_controlled_finished,
            )
            if diagnostic is not None:
                return diagnostic
        return None

    if specs.passthrough_spec_for(node) is not None:
        if _integrator_mode_enabled(node) and _integrating_transfer_affine(node, composition) is None:
            # A TransferMechanism with integrator_mode=True is a stateful leaky/
            # simple integrator.  We can lower it statelessly only when it fires
            # exactly once per trial from a per-trial reset (so its integrator
            # advances a single step from its initializer); otherwise it
            # accumulates within the trial and must be rejected rather than
            # silently mis-run as a stateless function.
            return BatchedDiagnostic(
                node_name,
                "unsupported stateful transfer (integrator_mode) for batched v2",
                "integrator_mode=True",
            )
        if specs.function_spec_for(function) is None:
            return BatchedDiagnostic(node_name, "unsupported function for batched v2", function_type)
        combine = _combine_name(node)
        if combine not in {"sum", "product"}:
            return BatchedDiagnostic(node_name, "unsupported input combine for batched v2", combine)
        return None

    return BatchedDiagnostic(node_name, "unsupported node for batched v2", type(node).__name__)


def _single_input_support_diagnostic(node) -> BatchedDiagnostic | None:
    input_ports = tuple(getattr(node, "input_ports", ()))
    if not input_ports:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported input-port routing for batched v2",
            "input_ports=0",
        )

    for input_port in input_ports:
        diagnostic = _input_port_function_support_diagnostic(node, input_port)
        if diagnostic is not None:
            return diagnostic

    output_ports = tuple(getattr(node, "output_ports", ()))
    duplicate_output_names = _duplicate_component_names(output_ports)
    if duplicate_output_names:
        return BatchedDiagnostic(
            _node_name(node),
            "duplicate OutputPort names are unsupported for batched v2",
            f"duplicates={duplicate_output_names!r}",
        )
    mechanism_spec = specs.mechanism_spec_for(node)
    if mechanism_spec is not None:
        if mechanism_spec.outputs is not None:
            diagnostic = _mechanism_output_port_support_diagnostic(
                node,
                output_ports,
                mechanism_spec.outputs,
            )
        else:
            diagnostic = _single_mechanism_output_port_support_diagnostic(
                node,
                output_ports,
            )
        if diagnostic is not None:
            return diagnostic
    elif specs.passthrough_spec_for(node) is not None:
        diagnostic = _elementwise_output_port_support_diagnostic(node)
        if diagnostic is not None:
            return diagnostic
    elif len(output_ports) != 1:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported multi-port output routing for batched v2",
            f"output_ports={len(output_ports)}",
        )
    return None


def _single_mechanism_output_port_support_diagnostic(
    node,
    output_ports,
) -> BatchedDiagnostic | None:
    """Validate the one direct result emitted by a custom mechanism op."""

    if len(output_ports) != 1:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported mechanism output-port configuration for batched v2",
            f"output_ports={len(output_ports)} (requires one direct result)",
        )
    output_port = output_ports[0]
    if (
        _owner_value_selector_index(output_port) != 0
        or not _is_identity_linear(getattr(output_port, "function", None))
    ):
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported mechanism output-port semantics for batched v2",
            f"{output_port.name}: requires identity OWNER_VALUE[0]",
        )
    return None


def _input_port_function_support_diagnostic(node, input_port) -> BatchedDiagnostic | None:
    """Validate the exact LinearCombination subset represented by Combine ops."""

    function = getattr(input_port, "function", None)
    port_name = getattr(input_port, "name", "InputPort")
    if type(input_port) is not InputPort:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported InputPort type for batched v2",
            f"{port_name}: {type(input_port).__name__}",
        )
    default_input = _parameter_value(input_port, "default_input", None)
    internal_only = bool(_parameter_value(input_port, "internal_only", False))
    if default_input is not None:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported InputPort default/internal binding for batched v2",
            (
                f"{port_name}: default_input={default_input!r}, "
                f"internal_only={internal_only!r}"
            ),
        )
    if type(function) is not LinearCombination:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported InputPort function for batched v2",
            f"{port_name}: {type(function).__name__}",
        )

    operation = _input_port_combine_name(input_port)
    if operation not in {"sum", "product"}:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported InputPort function for batched v2",
            f"{port_name}: LinearCombination(operation={operation!r})",
        )

    semantic_defaults = {
        "weights": None,
        "exponents": None,
        "scale": 1.0,
        "offset": 0.0,
    }
    for name, expected in semantic_defaults.items():
        value = _parameter_value(function, name, expected)
        supported = (
            value is None
            if expected is None
            else _numeric_scalar_exact(value, expected)
        )
        if not supported:
            return BatchedDiagnostic(
                _node_name(node),
                "unsupported InputPort function for batched v2",
                f"{port_name}: LinearCombination {name}={value!r}",
            )
    return None


def _mechanism_output_port_support_diagnostic(
    node,
    output_ports,
    output_decls,
) -> BatchedDiagnostic | None:
    """Validate the standard OutputPort semantics implemented by a mechanism op.

    A mechanism op's named outputs are semantic values, not aliases for any
    live port with the same name.  Compare every declared port with the
    owning mechanism's standard selector and function so a customized
    same-name port cannot silently acquire the built-in kernel behavior.
    """

    node_name = _node_name(node)
    standard_ports = getattr(node, "standard_output_ports", None)
    for output_decl in output_decls:
        matches = [
            port
            for port in output_ports
            if getattr(port, "name", "") == output_decl.port
        ]
        if len(matches) != 1:
            return BatchedDiagnostic(
                node_name,
                "unsupported mechanism output-port configuration for batched v2",
                f"requires exactly one {output_decl.port!r} port",
            )

        output_port = matches[0]
        actual_width = _port_width(output_port)
        if actual_width != output_decl.width:
            return BatchedDiagnostic(
                node_name,
                "unsupported mechanism output-port width for batched v2",
                f"{output_decl.port}: width={actual_width}, requires {output_decl.width}",
            )

        try:
            standard = standard_ports.get_port_dict(output_decl.port)
        except Exception:
            standard = None
        if not standard:
            return BatchedDiagnostic(
                node_name,
                "unsupported mechanism output-port semantics for batched v2",
                f"{output_decl.port}: no standard port specification",
            )

        expected_selector = standard.get("variable")
        expected_index = _owner_value_index(expected_selector)
        actual_index = _owner_value_selector_index(output_port)
        if expected_index is None or actual_index != expected_index:
            return BatchedDiagnostic(
                node_name,
                "unsupported mechanism output-port selector for batched v2",
                (
                    f"{output_decl.port}: selector="
                    f"{getattr(output_port, '_variable_spec', None)!r}, "
                    f"requires {expected_selector!r}"
                ),
            )

        expected_function = standard.get("function")
        actual_function = getattr(output_port, "function", None)
        if expected_function is None:
            function_matches = _is_identity_linear(actual_function)
        elif type(expected_function).__name__ == "UserDefinedFunction":
            function_matches = (
                type(actual_function) is type(expected_function)
                and _user_defined_callable(actual_function)
                is _user_defined_callable(expected_function)
            )
        else:
            # No generic Function-to-kernel semantic equivalence exists yet.
            # Future mechanism outputs must add a typed representation before
            # validation can safely accept their standard function.
            function_matches = False
        if not function_matches:
            return BatchedDiagnostic(
                node_name,
                "unsupported mechanism output-port function for batched v2",
                (
                    f"{output_decl.port}: {type(actual_function).__name__} "
                    "does not match the standard output semantics"
                ),
            )
    return None


def _elementwise_output_port_support_diagnostic(node) -> BatchedDiagnostic | None:
    """Accept OutputPorts that are identity slices of the mechanism value."""

    input_ports = tuple(getattr(node, "input_ports", ()))
    input_widths = tuple(_port_width(port) for port in input_ports)
    for output_port in tuple(getattr(node, "output_ports", ())):
        port_name = getattr(output_port, "name", "OutputPort")
        owner_value_index = _owner_value_selector_index(output_port)
        if (
            owner_value_index is None
            or owner_value_index < 0
            or owner_value_index >= len(input_widths)
            or _port_width(output_port) != input_widths[int(owner_value_index)]
        ):
            return BatchedDiagnostic(
                _node_name(node),
                "unsupported OutputPort function for batched v2",
                f"{port_name}: requires an identity OWNER_VALUE slice",
            )
        function = getattr(output_port, "function", None)
        if not _is_identity_linear(function):
            return BatchedDiagnostic(
                _node_name(node),
                "unsupported OutputPort function for batched v2",
                f"{port_name}: requires an identity Linear function",
            )
    return None


def _owner_value_index(selector) -> int | None:
    """Return an exact ``(OWNER_VALUE, integer index)`` selector's index."""

    if not isinstance(selector, tuple) or len(selector) != 2:
        return None
    source, index = selector
    if source != "OWNER_VALUE" or isinstance(index, (bool, np.bool_)):
        return None
    if not isinstance(index, (int, np.integer)):
        return None
    return int(index)


def _owner_value_selector_index(output_port) -> int | None:
    return _owner_value_index(getattr(output_port, "_variable_spec", None))


def _is_identity_linear(function) -> bool:
    if type(function) is not Linear:
        return False
    return all(
        _numeric_scalar_exact(
            _parameter_value(function, parameter, expected),
            expected,
        )
        for parameter, expected in (
            ("slope", 1.0),
            ("intercept", 0.0),
            ("scale", 1.0),
            ("offset", 0.0),
        )
    )


def _user_defined_callable(function):
    custom_function = getattr(function, "custom_function", None)
    if callable(custom_function):
        return custom_function
    custom_function = _parameter_value(function, "custom_function", None)
    return custom_function if callable(custom_function) else None


def _function_parameter_support_diagnostic(node_name, function) -> BatchedDiagnostic | None:
    function_type = type(function).__name__
    semantic_defaults = {
        "Linear": {"slope": 1.0, "intercept": 0.0, "scale": 1.0, "offset": 0.0},
        "Logistic": {
            "gain": 1.0,
            "bias": 0.0,
            "x_0": 0.0,
            "scale": 1.0,
            "offset": 0.0,
        },
    }.get(function_type, {})
    function_spec = specs.function_spec_for(function)
    modeled_parameters = (
        set()
        if function_spec is None
        else {binding.pnl_name or binding.arg for binding in function_spec.params}
    )
    for parameter_name in semantic_defaults:
        if parameter_name not in modeled_parameters:
            continue
        value = _parameter_value(
            function,
            parameter_name,
            semantic_defaults[parameter_name],
        )
        try:
            value_array = np.asarray(value)
            finite = (
                value_array.size > 0
                and value_array.dtype.kind in "biufc"
                and bool(np.all(np.isfinite(value_array)))
            )
        except Exception:
            value_array = np.asarray(())
            finite = False
        if value_array.size > 0 and value_array.dtype.kind == "c":
            return BatchedDiagnostic(
                node_name,
                f"unsupported complex {function_type} parameter for batched v2",
                f"{parameter_name} dtype={value_array.dtype}",
            )
        if not finite and value_array.size > 0 and value_array.dtype.kind in "biuf":
            return BatchedDiagnostic(
                node_name,
                f"unsupported non-finite {function_type} parameter for batched v2",
                f"{parameter_name} contains non-finite values",
            )
        if finite and bool(np.any(np.abs(value_array) > np.finfo(np.float32).max)):
            return BatchedDiagnostic(
                node_name,
                f"unsupported out-of-range {function_type} parameter for batched v2",
                f"{parameter_name} is not representable as float32",
            )
        if not _is_scalar_or_broadcast_scalar(value):
            return BatchedDiagnostic(
                node_name,
                f"unsupported non-scalar {function_type} parameter for batched v2",
                f"{parameter_name} shape={value_array.shape}",
            )
    unsupported_defaults = {
        name: default
        for name, default in semantic_defaults.items()
        if name not in modeled_parameters
    }
    for parameter_name, expected in unsupported_defaults.items():
        value = _parameter_value(function, parameter_name, expected)
        if not _numeric_equal(value, expected):
            return BatchedDiagnostic(
                node_name,
                f"unsupported {function_type} parameter for batched v2",
                f"{parameter_name}={value!r} (requires {expected!r})",
            )
    return None


def _lca_execution_support_diagnostic(
    node,
    composition,
    *,
    component_id: int,
    finished_values_by_component_id: Mapping[int, BatchedFinishedValueSpec],
    dynamic_controlled_finished: bool = False,
) -> BatchedDiagnostic | None:
    recurrent_projection = getattr(node, "recurrent_projection", None)
    projection_diagnostic = _mapping_projection_support_diagnostic(
        recurrent_projection
    )
    if projection_diagnostic is not None:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported LCA recurrent projection for batched v2",
            projection_diagnostic.detail,
        )

    conditions = _scheduler_conditions(composition)
    stepwise_ddm_pair = _scheduler_condition_is_effective_always(
        composition,
        node,
        conditions,
    ) and any(
        type(candidate).__name__ == "DDM"
        and type(candidate_condition) is WhenFinished
        and is_canonical_condition(candidate_condition)
        and tuple(getattr(candidate_condition, "args", ())) == (node,)
        for candidate, candidate_condition in conditions.items()
    )
    counted_finished_pair = (
        _fixed_finished_execution_count(
            finished_values_by_component_id.get(component_id)
        )
        is not None
        and _scheduler_condition_is_effective_always(
            composition,
            node,
            conditions,
        )
        and sum(
            _is_stateless_transfer_finished_follower(
                candidate,
                candidate_condition,
                node,
            )
            for candidate, candidate_condition in conditions.items()
        )
        == 1
    )
    stepwise = (
        stepwise_ddm_pair
        or counted_finished_pair
        or dynamic_controlled_finished
    )
    reset_condition = getattr(node, "reset_stateful_function_when", None)
    if (
        type(reset_condition) is AtTrialStart
        and is_canonical_condition(reset_condition)
        and not (counted_finished_pair or dynamic_controlled_finished)
    ):
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported LCA reset policy for batched v2",
            "AtTrialStart requires the fixed-count Always/WhenFinished schedule",
        )
    execute_until_finished = bool(_parameter_value(node, "execute_until_finished", True))
    if execute_until_finished == (not stepwise):
        return None
    return BatchedDiagnostic(
        _node_name(node),
        "unsupported LCA execution mode for batched v2",
        "execute_until_finished must be False only for an Always/WhenFinished stepwise pair",
    )


def _fixed_finished_execution_count(
    finished_value: BatchedFinishedValueSpec | None,
) -> int | None:
    """Read a validated execution count from one frozen finished declaration."""

    if (
        finished_value is None
        or finished_value.predicate_kind != "execution_count_at_least"
        or not isinstance(finished_value.attrs, Mapping)
        or set(finished_value.attrs) != {"count"}
    ):
        return None
    count = finished_value.attrs.get("count")
    return count if type(count) is int and count > 0 else None


def _is_stateless_transfer_finished_follower(candidate, condition, producer) -> bool:
    """Whether ``candidate`` is the narrow first counted-finished follower."""

    return (
        type(candidate).__name__ == "TransferMechanism"
        and specs.passthrough_spec_for(candidate) is not None
        and specs.function_spec_for(getattr(candidate, "function", None)) is not None
        and not bool(_parameter_value(candidate, "integrator_mode", False))
        and type(condition) is WhenFinished
        and is_canonical_condition(condition)
        and tuple(getattr(condition, "args", ())) == (producer,)
    )


def _ddm_execution_support_diagnostic(node, composition) -> BatchedDiagnostic | None:
    """Require stepwise DDM execution to belong to a typed coevolution pair."""

    execute_until_finished = _parameter_value(
        node,
        "execute_until_finished",
        True,
    )
    if bool(execute_until_finished):
        return None

    condition = _scheduler_conditions(composition).get(node)
    args = tuple(getattr(condition, "args", ()))
    stepper = (
        args[0]
        if (
            type(condition) is WhenFinished
            and is_canonical_condition(condition)
            and len(args) == 1
        )
        else None
    )
    stepper_spec = specs.mechanism_spec_for(stepper) if stepper is not None else None
    if (
        stepper_spec is not None
        and stepper_spec.can_step
        and stepper_spec.persistent_state
        and _scheduler_condition_is_effective_always(
            composition,
            stepper,
        )
    ):
        return None
    return BatchedDiagnostic(
        _node_name(node),
        "unsupported DDM execution mode for batched v2",
        "execute_until_finished=False requires a typed Always/WhenFinished "
        "coevolving stepper/terminator pair",
    )


_MISSING = object()


def _parameter_value(component, name, default=_MISSING):
    """Read a live parameter without the scalar coercion used for kernel bindings."""

    if component is not None:
        parameters = getattr(component, "parameters", None)
        parameter = getattr(parameters, name, None) if parameters is not None else None
        if parameter is not None:
            for getter_name in ("get", "_get"):
                getter = getattr(parameter, getter_name, None)
                if getter is not None:
                    try:
                        return getter(None)
                    except Exception:
                        pass
        defaults = getattr(component, "defaults", None)
        if defaults is not None and hasattr(defaults, name):
            return getattr(defaults, name)
        if hasattr(component, name):
            return getattr(component, name)
    return None if default is _MISSING else default


def _parameter_default_value(component, name, default=_MISSING):
    """Read a declared Parameter default without observing execution history."""

    if component is not None:
        parameters = getattr(component, "parameters", None)
        parameter = getattr(parameters, name, None) if parameters is not None else None
        if parameter is not None and hasattr(parameter, "default_value"):
            return parameter.default_value
        defaults = getattr(component, "defaults", None)
        if defaults is not None and hasattr(defaults, name):
            return getattr(defaults, name)
    return None if default is _MISSING else default


def _numeric_equal(value, expected) -> bool:
    try:
        array = np.asarray(value)
        return array.dtype.kind in "biufc" and bool(np.allclose(array, expected))
    except Exception:
        return False


def _numeric_exact(value, expected) -> bool:
    """Exact numeric equality for semantics omitted from the kernel IR."""

    try:
        array = np.asarray(value)
        return array.dtype.kind in "biufc" and bool(np.all(array == expected))
    except Exception:
        return False


def _numeric_scalar_exact(value, expected) -> bool:
    """Exact finite-real scalar equality for an absorbed scalar operation."""

    try:
        array = np.asarray(value)
        return bool(
            array.size == 1
            and array.dtype.kind in "biuf"
            and np.isfinite(array.reshape(-1)[0])
            and array.reshape(-1)[0] == expected
        )
    except Exception:
        return False


def _finite_fp32_scalar_value(value) -> float | None:
    """Return a finite real scalar in the representable fp32 range."""

    try:
        array = np.asarray(value)
        if array.size != 1 or array.dtype.kind not in "biuf":
            return None
        scalar = array.reshape(-1)[0]
        packed = np.float32(scalar)
        if not np.isfinite(scalar) or not np.isfinite(packed):
            return None
        return float(scalar)
    except Exception:
        return None


def _is_scalar_or_broadcast_scalar(value) -> bool:
    """Whether scalar parameter-buffer lowering preserves ``value`` exactly."""

    try:
        array = np.asarray(value)
        if array.size == 0 or array.dtype.kind not in "biufc":
            return False
        return bool(
            np.all(np.isfinite(array))
            and np.all(array == array.reshape(-1)[0])
        )
    except Exception:
        return False


def _transfer_noise_constant(node):
    value = _parameter_value(node, "noise", 0.0)
    if callable(value):
        return None, BatchedDiagnostic(
            _node_name(node),
            "unsupported TransferMechanism noise for batched v2",
            "requires deterministic finite scalar or broadcast-scalar noise",
        )
    shape = None
    try:
        array = np.asarray(value)
        shape = array.shape
        first = array.reshape(-1)[0]
        supported = (
            array.size > 0
            and array.dtype.kind in "biuf"
            and bool(np.all(np.isfinite(array)))
            and bool(np.all(array == first))
        )
    except Exception:
        supported = False
    if not supported:
        return None, BatchedDiagnostic(
            _node_name(node),
            "unsupported TransferMechanism noise for batched v2",
            f"requires deterministic finite scalar or broadcast-scalar noise; shape={shape}",
        )
    return float(first), None


def _transfer_clip_bounds(node):
    value = _parameter_value(node, "clip", None)
    if value is None:
        return None, None
    try:
        bounds = np.asarray(value)
        supported = (
            bounds.shape == (2,)
            and bounds.dtype.kind in "biuf"
            and bool(np.all(np.isfinite(bounds)))
        )
    except Exception:
        supported = False
    if not supported:
        return None, BatchedDiagnostic(
            _node_name(node),
            "unsupported TransferMechanism clip for batched v2",
            "requires exactly two finite scalar bounds",
        )
    lower, upper = (float(bound) for bound in bounds)
    if lower > upper:
        return None, BatchedDiagnostic(
            _node_name(node),
            "unsupported TransferMechanism clip for batched v2",
            f"lower bound {lower} exceeds upper bound {upper}",
        )
    return (lower, upper), None


def _is_zero(value) -> bool:
    return value is None or _numeric_equal(value, 0.0)


def _primary_input_port_name(node) -> str:
    ports = tuple(getattr(node, "input_ports", ()))
    return getattr(ports[0], "name", "InputPort-0") if ports else "InputPort-0"


def _supported_output_ports(node) -> set[str]:
    ports = tuple(getattr(node, "output_ports", ()))
    mechanism_spec = specs.mechanism_spec_for(node)
    if mechanism_spec is not None:
        if mechanism_spec.outputs is not None:
            return {decl.port for decl in mechanism_spec.outputs}
        return {getattr(ports[0], "name", "RESULT")} if ports else {"RESULT"}
    if specs.passthrough_spec_for(node) is not None:
        return {
            getattr(port, "name", "RESULT")
            for port in ports
            if _identity_output_port_slice(node, port) is not None
        }
    return {getattr(ports[0], "name", "RESULT")} if ports else {"RESULT"}


def _output_support_diagnostics(outputs, nodes) -> list[BatchedDiagnostic]:
    if outputs is None:
        return []
    node_names = {_node_name(node) for node in nodes}
    rejected = []
    for output in outputs:
        if isinstance(output, str):
            continue
        owner = getattr(output, "owner", None)
        port_name = getattr(output, "name", str(output))
        if owner is None or _node_name(owner) not in node_names or port_name not in _supported_output_ports(owner):
            rejected.append(
                BatchedDiagnostic(
                    _node_name(owner) if owner is not None else port_name,
                    "unsupported output-port routing for batched v2",
                    port_name,
                )
            )
    return rejected


def _control_edges(control):
    signals = tuple(getattr(control, "control_signals", ()))
    efferents = tuple(
        projection
        for signal in signals
        for projection in getattr(signal, "efferents", ())
    )
    monitors = tuple(
        projection
        for port in getattr(control, "input_ports", ())
        for projection in getattr(port, "path_afferents", ())
    )
    return signals, efferents, monitors


def _projection_is_active_in_composition(projection, composition) -> bool:
    """Require both structural membership and PNL's live activation flag."""

    if id(projection) not in {
        id(candidate)
        for candidate in getattr(composition, "projections", ())
    }:
        return False
    try:
        return bool(projection.is_active_in_composition(composition))
    except (AttributeError, TypeError, ValueError):
        return False


def _absorbed_control_attrs(control) -> dict[str, str]:
    _, efferents, monitors = _control_edges(control)
    target = getattr(getattr(efferents[0], "receiver", None), "owner", None) if efferents else None
    source = getattr(getattr(monitors[0], "sender", None), "owner", None) if monitors else None
    receiver = getattr(efferents[0], "receiver", None) if efferents else None
    return {
        "source": _node_name(source) if source is not None else "",
        "target": _node_name(target) if target is not None else "",
        "parameter": getattr(receiver, "name", ""),
        "modulation": "OVERRIDE",
    }


@dataclass(frozen=True)
class _ResolvedControlChain:
    signal: object
    control_projection: object
    monitor_projection: object
    source_port: object
    controller_input_port: object
    target_parameter_port: object
    source: object
    target: object
    target_port: str
    monitor_is_parameter_input: bool


def _parameter_port_has_canonical_source(target, parameter_name, port) -> bool:
    """Whether ``port`` is backed by the named live Parameter on its owner."""

    source = getattr(port, "source", None)
    names = [parameter_name]
    integrator_suffix = "-integrator_function"
    if parameter_name.endswith(integrator_suffix):
        names.append(parameter_name.removesuffix(integrator_suffix))
    owners = (
        target,
        getattr(target, "function", None),
        getattr(target, "integrator_function", None),
    )
    return any(
        source
        is getattr(
            getattr(owner, "parameters", None),
            candidate_name,
            None,
        )
        for owner in owners
        if owner is not None
        for candidate_name in names
    )


def _ignored_parameter_control_is_lowered(
    control,
    composition,
    params: _ParamBuilder,
) -> bool:
    """Authenticate one PEC control erased in favor of a parameter lane.

    Ignoring an arbitrary ControlMechanism would silently remove model
    semantics.  The PEC path is safe only because its generated controller
    monitors an external scalar parameter input and writes the exact live
    Parameter already bound to one mutable IR parameter.
    """

    chain, diagnostic = _resolve_control_chain(control, composition)
    if (
        diagnostic is not None
        or chain is None
        or not chain.monitor_is_parameter_input
        or type(getattr(control, "function", None)) is not Identity
    ):
        return False
    public_name = f"{_node_name(chain.target)}.{chain.target_port}"
    matches = tuple(
        parameter
        for parameter in params.specs
        if public_name in (parameter.name, *parameter.aliases)
    )
    return bool(
        len(matches) == 1
        and matches[0].runtime_mutable is True
        and params.bindings_by_id.get(matches[0].parameter_id)
        is getattr(chain.target_parameter_port, "source", None)
    )


def _resolve_control_chain(
    control,
    composition,
) -> tuple[_ResolvedControlChain | None, BatchedDiagnostic | None]:
    """Validate and resolve the exact scalar control chain once.

    Both capability diagnostics and typed modulation lowering consume this
    resolver, so a semantically rejected InputPort or projection can never be
    described as an absorbed identity edge in GraphIR.
    """

    name = _node_name(control)
    signals, efferents, monitors = _control_edges(control)
    if len(signals) != 1 or len(efferents) != 1 or len(monitors) != 1:
        return None, BatchedDiagnostic(
            name,
            "unsupported generic ControlMechanism for batched v2",
            "requires exactly one monitor, ControlSignal, and ControlProjection",
        )
    signal = signals[0]
    control_projection = efferents[0]
    monitor_projection = monitors[0]
    if (
        type(control_projection) is not ControlProjection
        or type(monitor_projection) is not MappingProjection
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported generic control projection for batched v2",
        )
    if not _projection_is_active_in_composition(
        monitor_projection,
        composition,
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported control monitor routing for batched v2",
            "monitor MappingProjection is not active in this Composition",
        )
    if not _is_override(
        getattr(signal, "modulation", getattr(control, "modulation", None))
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported control modulation for batched v2",
            str(getattr(signal, "modulation", None)),
        )
    input_ports = tuple(getattr(control, "input_ports", ()))
    if len(input_ports) != 1:
        return None, BatchedDiagnostic(
            name,
            "unsupported control input routing for batched v2",
            f"input_ports={len(input_ports)}",
        )
    controller_input_port = input_ports[0]
    input_diagnostic = _input_port_function_support_diagnostic(
        control,
        controller_input_port,
    )
    if input_diagnostic is not None:
        return None, BatchedDiagnostic(
            name,
            "unsupported control input semantics for batched v2",
            input_diagnostic.detail,
        )
    if (
        not _is_identity_linear(getattr(control_projection, "function", None))
        or _parameter_value(control_projection, "weight", None) is not None
        or _parameter_value(control_projection, "exponent", None) is not None
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported ControlProjection semantics for batched v2",
            "requires identity Linear with no weight or exponent",
        )
    if not _control_signal_is_identity(signal):
        return None, BatchedDiagnostic(
            name,
            "unsupported ControlSignal semantics for batched v2",
            "requires an identity TransferWithCosts transfer function",
        )

    source_port = getattr(monitor_projection, "sender", None)
    monitor_receiver = getattr(monitor_projection, "receiver", None)
    target_parameter_port = getattr(control_projection, "receiver", None)
    source = getattr(source_port, "owner", None)
    target = getattr(target_parameter_port, "owner", None)
    target_port = getattr(target_parameter_port, "name", "")
    monitor_is_identity = _is_identity_scalar_projection(monitor_projection)
    monitor_is_parameter_input = _is_external_parameter_projection(
        composition,
        monitor_projection,
    )
    if not _parameter_port_has_canonical_source(
        target,
        target_port,
        target_parameter_port,
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported control target parameter identity for batched v2",
            (
                f"{_node_name(target)}.{target_port} does not resolve to its "
                "canonical PsyNeuLink Parameter"
            ),
        )
    if (
        source is None
        or target is None
        or monitor_receiver is not controller_input_port
        or getattr(controller_input_port, "owner", None) is not control
        or getattr(signal, "owner", None) is not control
        or getattr(control_projection, "sender", None) is not signal
        or not (monitor_is_identity or monitor_is_parameter_input)
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported control monitor routing for batched v2",
            "monitor must be the controller's scalar identity InputPort projection",
        )
    try:
        target_afferents = tuple(target_parameter_port.mod_afferents)
    except Exception:
        target_afferents = ()
    # ControlProjections are not ordinary Composition.projections.  Exact
    # signal ownership plus the target ParameterPort's sole mod_afferent is the
    # authoritative active-edge relation in PNL.
    if target_afferents != (control_projection,):
        return None, BatchedDiagnostic(
            name,
            "ambiguous control projection routing for batched v2",
            f"{_node_name(target)}.{target_port}",
        )
    if not _projection_is_active_in_composition(
        control_projection,
        composition,
    ):
        return None, BatchedDiagnostic(
            name,
            "unsupported control projection routing for batched v2",
            "ControlProjection is not active in this Composition",
        )

    function_diagnostic = _function_parameter_support_diagnostic(
        name,
        getattr(control, "function", None),
    )
    if function_diagnostic is not None:
        return None, function_diagnostic
    return (
        _ResolvedControlChain(
            signal=signal,
            control_projection=control_projection,
            monitor_projection=monitor_projection,
            source_port=source_port,
            controller_input_port=controller_input_port,
            target_parameter_port=target_parameter_port,
            source=source,
            target=target,
            target_port=target_port,
            monitor_is_parameter_input=monitor_is_parameter_input,
        ),
        None,
    )


def _control_support_diagnostic(control, composition) -> BatchedDiagnostic | None:
    """Accept only control edges whose semantics are explicitly folded by an op."""

    name = _node_name(control)
    chain, diagnostic = _resolve_control_chain(control, composition)
    if diagnostic is not None:
        return diagnostic
    assert chain is not None
    source = chain.source
    target = chain.target
    target_port = chain.target_port
    monitor_is_parameter_input = chain.monitor_is_parameter_input
    function = getattr(control, "function", None)

    if (
        monitor_is_parameter_input
        and type(function) is Identity
        and _is_declared_batched_parameter(target, target_port)
    ):
        return None

    if type(target).__name__ == "LCAMechanism" and target_port == "termination_threshold":
        from psyneulink.core.batched.components.lca import _control_monitor_source_for

        schedule_diagnostic = _absorbed_lca_schedule_support_diagnostic(
            composition,
            source,
            control,
            target,
        )
        if schedule_diagnostic is not None:
            return schedule_diagnostic
        identity = type(function) is Identity or _is_identity_linear(function)
        registered_dynamic_transform = (
            type(function) is Linear
            and specs.function_spec_for(function) is not None
            and not bool(
                _parameter_value(target, "execute_until_finished", True)
            )
            and any(
                _when_finished_depends_on(condition, target)
                for candidate, condition in _scheduler_conditions(
                    composition
                ).items()
                if candidate is not target
            )
        )
        source_supported = _supported_lca_termination_source(
            composition,
            source,
        ) or (
            registered_dynamic_transform
            and _is_unmodeled_coevolving_lca_termination(
                composition,
                target,
            )
            and _supported_dynamic_lca_termination_source(
                composition,
                source,
            )
        )
        if (
            (identity or registered_dynamic_transform)
            and _control_monitor_source_for(composition, target) is source
            and source_supported
        ):
            return None
        if _is_unmodeled_coevolving_lca_termination(composition, target):
            return BatchedDiagnostic(
                name,
                "batched schedule kind is not executable yet",
                "coevolving Always/WhenFinished execution falls outside the "
                "typed controlled-finished subset and requires executable "
                "conditional pass regions",
            )
    elif type(target).__name__ == "DDM" and target_port == "threshold":
        if _supported_ddm_threshold_override(composition, control, source, target):
            return None

    return BatchedDiagnostic(
        name,
        "unsupported generic ControlMechanism for batched v2",
        f"{_node_name(source)}->{_node_name(target)}.{target_port}",
    )


def _absorbed_lca_schedule_support_diagnostic(
    composition,
    source,
    control,
    target,
) -> BatchedDiagnostic | None:
    """Validate timing that remains equivalent after an LCA control is absorbed."""

    conditions = _scheduler_conditions(composition)
    source_set = _live_consideration_set_id(composition, source)
    controller_set = _live_consideration_set_id(composition, control)
    target_set = _live_consideration_set_id(composition, target)
    if not (0 <= source_set < controller_set < target_set):
        return BatchedDiagnostic(
            _node_name(control),
            "unsupported absorbed control scheduler ordering for batched v2",
            "requires source consideration set < controller set < target set; "
            f"got {source_set}, {controller_set}, {target_set}",
        )

    for role, component in (
        ("source", source),
        ("controller", control),
        ("target", target),
    ):
        condition = conditions.get(component)
        if condition is None:
            if role == "target" and not bool(
                _parameter_value(target, "execute_until_finished", True)
            ):
                return BatchedDiagnostic(
                    _node_name(control),
                    "unsupported absorbed control scheduler condition for batched v2",
                    "stepwise controlled target requires explicit Always",
                )
            continue
        if role in {"source", "controller"} and _at_pass_spec(condition) == (
            0,
            "ENVIRONMENT_STATE_UPDATE",
        ):
            continue
        if (
            role == "target"
            and type(condition) is Always
            and is_canonical_condition(condition)
        ):
            continue
        return BatchedDiagnostic(
            _node_name(control),
            "unsupported absorbed control scheduler condition for batched v2",
            f"{role} {_node_name(component)} uses {_condition_label(condition)}",
        )
    return None


def _live_consideration_set_id(composition, component) -> int:
    """Return one component's normalized PNL scheduler-set ordinal."""

    queue = getattr(getattr(composition, "scheduler", None), "consideration_queue", ())
    matches = tuple(
        index
        for index, consideration_set in enumerate(queue)
        if component in consideration_set
    )
    return matches[0] if len(matches) == 1 else -1


def _condition_time_scale_name(condition) -> str | None:
    """Return a scheduler condition's captured time-scale name, if explicit."""

    time_scale = _condition_time_scale(condition)
    if time_scale is None:
        return None
    return getattr(time_scale, "name", str(time_scale))


def _condition_time_scale(condition):
    """Return the exact time-scale object captured by a Condition function."""

    try:
        return inspect.getclosurevars(condition.func).nonlocals["time_scale"]
    except (AttributeError, KeyError, TypeError):
        return None


def _condition_label(condition) -> str:
    condition_name = type(condition).__name__
    if condition_name != "AtPass":
        return condition_name
    args = tuple(getattr(condition, "args", ()))
    pass_count = args[0] if args else "?"
    return (
        f"AtPass({pass_count}, "
        f"time_scale={_condition_time_scale_name(condition) or 'unknown'})"
    )


def _is_unmodeled_coevolving_lca_termination(composition, lca) -> bool:
    """Whether ``lca`` controls the start of a coupled lane-local terminator.

    The current fused emitter infers a warm-up count, but KernelIR contains
    neither the LCA's per-lane ``finished`` value nor its effective controlled
    threshold.  Recognizing the surrounding composition as co-evolving is not
    sufficient to preserve this scheduling edge, even when sampled control
    values happen to be constant.
    """

    conditions = _scheduler_conditions(composition)
    if not _scheduler_condition_is_effective_always(
        composition,
        lca,
        conditions,
    ):
        return False
    for node, condition in conditions.items():
        mechanism_spec = specs.mechanism_spec_for(node)
        if (
            mechanism_spec is not None
            and mechanism_spec.is_terminator
            and mechanism_spec.can_step
            and _when_finished_depends_on(condition, lca)
        ):
            return True
    return False


def _is_override(value) -> bool:
    return type(value) is str and value == "OVERRIDE"


def _control_signal_is_identity(signal) -> bool:
    function = getattr(signal, "function", None)
    transfer_function = _parameter_value(function, "transfer_fct", None)
    return (
        type(function) is TransferWithCosts
        and _is_identity_linear(transfer_function)
        and _numeric_scalar_exact(
            _parameter_value(function, "transfer_fct_mult_param", 1.0),
            1.0,
        )
        and _numeric_scalar_exact(
            _parameter_value(function, "transfer_fct_add_param", 0.0),
            0.0,
        )
    )


def _is_identity_scalar_projection(projection) -> bool:
    matrix = np.asarray(_get_matrix(projection))
    sender = getattr(projection, "sender", None)
    owner = getattr(sender, "owner", None)
    ports = tuple(getattr(owner, "output_ports", ()))
    primary_name = getattr(ports[0], "name", "RESULT") if ports else "RESULT"
    return (
        _mapping_projection_support_diagnostic(projection) is None
        and matrix.shape == (1, 1)
        and _numeric_exact(matrix, 1.0)
        and getattr(sender, "name", primary_name) == primary_name
    )


def _is_external_parameter_projection(composition, projection) -> bool:
    sender = getattr(projection, "sender", None)
    source = getattr(sender, "owner", None)
    return (
        source is getattr(composition, "input_CIM", None)
        and _mapping_projection_support_diagnostic(projection) is None
        and np.asarray(_get_matrix(projection)).shape == (1, 1)
        and _numeric_exact(_get_matrix(projection), 1.0)
    )


def _is_declared_batched_parameter(target, parameter_name) -> bool:
    mechanism_spec = specs.mechanism_spec_for(target)
    if mechanism_spec is None:
        return False
    return any(
        parameter_name in {binding.arg, binding.pnl_name, *binding.fallbacks}
        for binding in mechanism_spec.params
    )


def _has_active_node_afferents(composition, node) -> bool:
    """Whether ``node`` receives a processing edge from another active node."""

    active_node_ids = {id(candidate) for candidate in _composition_nodes(composition)}
    active_projection_ids = {
        id(projection)
        for projection in getattr(composition, "projections", ())
    }
    return any(
        id(projection) in active_projection_ids
        and id(
            getattr(getattr(projection, "sender", None), "owner", None)
        )
        in active_node_ids
        for port in getattr(node, "input_ports", ())
        for projection in getattr(port, "path_afferents", ())
    )


def _supported_lca_termination_source(composition, source) -> bool:
    if specs.passthrough_spec_for(source) is None or len(tuple(getattr(source, "input_ports", ()))) != 1:
        return False
    if _has_active_node_afferents(composition, source):
        return False
    function = getattr(source, "function", None)
    return (
        _is_identity_linear(function)
        and not _integrator_mode_enabled(source)
        and _numeric_exact(_parameter_value(source, "noise", 0.0), 0.0)
        and _parameter_value(source, "clip", None) is None
    )


def _supported_dynamic_lca_termination_source(composition, source) -> bool:
    """Whether a co-evolving count source is an emitted scalar Linear op.

    Atomic controlled-finished graphs deliberately continue to use
    :func:`_supported_lca_termination_source` and therefore remain
    identity-only.  CSI keeps the source in the fused program, so its slope
    (switch CSI) and intercept (repeat CSI) may be affine lane parameters.
    """

    function = getattr(source, "function", None)
    values = tuple(
        _finite_fp32_scalar_value(_parameter_value(function, argument, None))
        for argument in ("slope", "intercept", "scale", "offset")
    )
    return bool(
        specs.passthrough_spec_for(source) is not None
        and len(tuple(getattr(source, "input_ports", ()))) == 1
        and not _has_active_node_afferents(composition, source)
        and type(function) is Linear
        and isinstance(
            specs.function_spec_for(function),
            specs.ElementwiseFunctionSpec,
        )
        and _function_parameter_support_diagnostic(
            _node_name(source),
            function,
        )
        is None
        and all(value is not None for value in values)
        and not _integrator_mode_enabled(source)
        and _numeric_exact(_parameter_value(source, "noise", 0.0), 0.0)
        and _parameter_value(source, "clip", None) is None
    )


def _supported_ddm_threshold_override(composition, control, source, target) -> bool:
    from psyneulink.core.batched.components.ddm import threshold_override_collapse

    try:
        chain = threshold_override_collapse(target)
    except Exception:
        return False
    if chain is None or chain[0] != _node_name(source):
        return False
    if type(source).__name__ != "TransferMechanism":
        return False
    if (
        not _integrator_mode_enabled(source)
        or _parameter_value(source, "execute_until_finished", False) is not False
        or _parameter_value(target, "execute_until_finished", False) is not False
    ):
        return False
    function = getattr(source, "function", None)
    integrator = getattr(source, "integrator_function", None)
    if type(function).__name__ != "Linear" or type(integrator).__name__ != "SimpleIntegrator":
        return False
    if _function_parameter_support_diagnostic(_node_name(source), function) is not None:
        return False
    if (
        not _numeric_equal(_parameter_value(function, "slope", 1.0), 1.0)
        or not _numeric_equal(_parameter_value(function, "scale", 1.0), 1.0)
        or not _numeric_equal(_parameter_value(function, "offset", 0.0), 0.0)
    ):
        return False
    if not _is_zero(_parameter_value(source, "noise", 0.0)) or _parameter_value(source, "clip", None) is not None:
        return False
    if not _is_zero(_parameter_value(integrator, "noise", 0.0)) or not _is_zero(_parameter_value(integrator, "initializer", 0.0)):
        return False
    source_reset = getattr(source, "reset_stateful_function_when", None)
    if type(source_reset) is not AtTrialStart or not is_canonical_condition(
        source_reset
    ):
        return False
    if _has_active_node_afferents(composition, source):
        return False
    if not _is_zero(getattr(getattr(source, "defaults", None), "variable", 0.0)):
        return False
    source_condition = _scheduler_conditions(composition).get(source)
    control_condition = _scheduler_conditions(composition).get(control)
    target_condition = _scheduler_conditions(composition).get(target)
    if source_condition is None:
        return False
    return _same_condition(source_condition, control_condition) and _same_condition(
        source_condition, target_condition
    )


def _exact_ddm_threshold_runtime_binding(composition, target):
    """Resolve direct runtime lanes for one absorbed CSI threshold chain.

    The public source parameters can alias the folded DDM arguments only when
    the omitted source/controller transform is an exact identity around the
    source Linear intercept and SimpleIntegrator offset.  Broader affine
    chains may still be described by declaration-only IR, but exposing their
    raw source names as direct kernel parameters would change their semantics.
    """

    matches = []
    for control in _composition_nodes(composition):
        if type(control) is not ControlMechanism:
            continue
        chain, diagnostic = _resolve_control_chain(control, composition)
        if (
            diagnostic is not None
            or chain is None
            or chain.target is not target
            or chain.target_port != "threshold"
            or type(getattr(control, "function", None)) is not Identity
            or not _supported_ddm_threshold_override(
                composition,
                control,
                chain.source,
                target,
            )
        ):
            continue
        source = chain.source
        function = getattr(source, "function", None)
        integrator = getattr(source, "integrator_function", None)
        source_intercept = _finite_fp32_scalar_value(
            _parameter_value(function, "intercept", None)
        )
        collapse = _finite_fp32_scalar_value(
            _parameter_value(integrator, "offset", None)
        )
        target_threshold = _finite_fp32_scalar_value(
            _parameter_value(
                getattr(target, "function", None),
                "threshold",
                None,
            )
        )
        threshold_parameter = getattr(
            getattr(function, "parameters", None),
            "intercept",
            None,
        )
        collapse_parameter = getattr(
            getattr(integrator, "parameters", None),
            "offset",
            None,
        )
        if (
            threshold_parameter is None
            or collapse_parameter is None
            or not _numeric_scalar_exact(
                _parameter_value(function, "slope", None),
                1.0,
            )
            or not _numeric_scalar_exact(
                _parameter_value(function, "scale", None),
                1.0,
            )
            or not _numeric_scalar_exact(
                _parameter_value(function, "offset", None),
                0.0,
            )
            or source_intercept is None
            or collapse is None
            or target_threshold is None
            or source_intercept != target_threshold
        ):
            continue
        matches.append(
            (source, threshold_parameter, collapse_parameter)
        )
    return matches[0] if len(matches) == 1 else None


def _same_condition(left, right) -> bool:
    if (
        type(left) is not type(right)
        or not is_canonical_condition(left)
        or not is_canonical_condition(right)
    ):
        return False
    left_args = tuple(_node_name(arg) if hasattr(arg, "name") else arg for arg in getattr(left, "args", ()))
    right_args = tuple(_node_name(arg) if hasattr(arg, "name") else arg for arg in getattr(right, "args", ()))
    return left_args == right_args


def _projection_specs(
    composition,
    nodes,
    component_ids,
    port_ids,
) -> tuple[
    list[BatchedProjectionSpec],
    list[BatchedDiagnostic],
    dict[str, object],
    dict[int, object],
]:
    projections: list[BatchedProjectionSpec] = []
    rejected: list[BatchedDiagnostic] = []
    bindings: dict[str, object] = {}
    bindings_by_id: dict[int, object] = {}
    node_ids = {id(node) for node in nodes}
    active_projection_ids = {
        id(projection)
        for projection in getattr(composition, "projections", ())
    }
    feedback_projection_ids = {
        id(projection)
        for projection in getattr(composition, "feedback_projections", ())
    }
    for node in nodes:
        for input_port in getattr(node, "input_ports", []):
            afferents = sorted(
                getattr(input_port, "path_afferents", []),
                key=lambda projection: (
                    component_ids.get(
                        id(getattr(getattr(projection, "sender", None), "owner", None)),
                        len(component_ids),
                    ),
                    port_ids.get(id(getattr(projection, "sender", None)), len(port_ids)),
                    type(projection).__module__,
                    type(projection).__qualname__,
                    getattr(projection, "name", ""),
                ),
            )
            for projection in afferents:
                # Ports retain projections owned by every Composition in which
                # they have participated.  Only projections active in this
                # Composition belong to the graph being lowered.
                if id(projection) not in active_projection_ids:
                    continue
                projection_type = type(projection).__name__
                sender = getattr(getattr(projection, "sender", None), "owner", None)
                receiver = getattr(getattr(projection, "receiver", None), "owner", None)
                if sender is None or receiver is None:
                    continue
                sender_name = _node_name(sender)
                receiver_name = _node_name(receiver)
                if id(sender) not in node_ids or id(receiver) not in node_ids:
                    continue
                if projection_type == "AutoAssociativeProjection":
                    continue
                if id(projection) in feedback_projection_ids:
                    rejected.append(
                        BatchedDiagnostic(
                            getattr(projection, "name", projection_type),
                            "unsupported feedback projection for batched v2",
                            f"{sender_name}->{receiver_name}",
                        )
                    )
                    continue
                # Monitor projections into an exactly-supported absorbed controller
                # are represented by that controller's semantic metadata, not as a
                # graph op.  No other path through a ControlMechanism is executable.
                if type(receiver) is ControlMechanism:
                    continue
                if type(sender) is ControlMechanism or type(projection) is ControlProjection:
                    rejected.append(
                        BatchedDiagnostic(
                            getattr(projection, "name", projection_type),
                            "unsupported control projection for batched v2",
                            f"{sender_name}->{receiver_name}",
                        )
                    )
                    continue
                projection_spec = specs.projection_spec_for(projection)
                if projection_spec is None:
                    rejected.append(
                        BatchedDiagnostic(
                            getattr(projection, "name", projection_type),
                            "unsupported projection for batched v2",
                            projection_type,
                        )
                    )
                    continue
                projection_diagnostic = _mapping_projection_support_diagnostic(
                    projection
                )
                if projection_diagnostic is not None:
                    rejected.append(projection_diagnostic)
                    continue
                sender_port = getattr(getattr(projection, "sender", None), "name", "RESULT")
                receiver_port = getattr(getattr(projection, "receiver", None), "name", "InputPort-0")
                supported_sender_ports = _supported_output_ports(sender)
                if sender_port not in supported_sender_ports:
                    rejected.append(
                        BatchedDiagnostic(
                            getattr(projection, "name", projection_type),
                            "unsupported output-port projection routing for batched v2",
                            f"{sender_name}.{sender_port}->{receiver_name}.{receiver_port}",
                        )
                    )
                    continue
                matrix = np.asarray(_get_matrix(projection), dtype=np.float32)
                projections.append(
                    BatchedProjectionSpec(
                        sender=sender_name,
                        sender_port=sender_port,
                        receiver=receiver_name,
                        receiver_port=receiver_port,
                        matrix=matrix,
                        spec_key=projection_spec.key,
                        projection_id=len(projections),
                        sender_component_id=component_ids[id(sender)],
                        sender_port_id=port_ids[id(projection.sender)],
                        receiver_component_id=component_ids[id(receiver)],
                        receiver_port_id=port_ids[id(projection.receiver)],
                    )
                )
                projection_id = projections[-1].projection_id
                bindings[
                    projection_binding_key(sender_name, sender_port, receiver_name, receiver_port)
                ] = projection
                bindings_by_id[projection_id] = projection
    return projections, rejected, bindings, bindings_by_id


def _mapping_projection_support_diagnostic(projection) -> BatchedDiagnostic | None:
    """Validate the exact dense real dot-product semantics lowered to KernelIR."""

    projection_name = getattr(projection, "name", type(projection).__name__)
    function = getattr(projection, "function", None)
    operation = _parameter_value(function, "operation", None)
    normalize = _parameter_value(function, "normalize", False)
    weight = _parameter_value(projection, "weight", None)
    exponent = _parameter_value(projection, "exponent", None)
    if (
        type(function) is not MatrixTransform
        or operation != "dot_product"
        or not _numeric_exact(normalize, False)
        or weight is not None
        or exponent is not None
    ):
        return BatchedDiagnostic(
            projection_name,
            "unsupported MappingProjection function for batched v2",
            (
                "requires MatrixTransform(operation=DOT_PRODUCT, normalize=False); "
                f"got {type(function).__name__}(operation={operation!r}, "
                f"normalize={normalize!r}, weight={weight!r}, exponent={exponent!r})"
            ),
        )

    try:
        matrix = np.asarray(_get_matrix(projection))
        real_numeric = matrix.dtype.kind in "biuf"
        finite = real_numeric and bool(np.all(np.isfinite(matrix)))
        representable = finite and bool(
            np.all(np.abs(matrix) <= np.finfo(np.float32).max)
        )
    except Exception:
        matrix = np.asarray(())
        representable = False
    if not representable:
        return BatchedDiagnostic(
            projection_name,
            "unsupported MappingProjection matrix for batched v2",
            (
                "requires finite real values representable as float32; "
                f"dtype={matrix.dtype}, shape={matrix.shape}"
            ),
        )
    return None


def _input_specs(
    nodes,
    projections: list[BatchedProjectionSpec],
    component_ids,
    port_ids,
) -> list[BatchedInputSpec]:
    receiver_port_ids = {projection.receiver_port_id for projection in projections}
    specs_out = []
    for node in nodes:
        component_type = type(node).__name__
        node_name = _node_name(node)
        if component_type == "ControlMechanism":
            continue
        input_ports = tuple(getattr(node, "input_ports", ()))
        for input_port in input_ports:
            port_id = port_ids[id(input_port)]
            if port_id in receiver_port_ids:
                continue
            specs_out.append(
                BatchedInputSpec(
                    name=(
                        node_name
                        if len(input_ports) == 1
                        else f"{node_name}.{input_port.name}"
                    ),
                    node=node_name,
                    width=_port_width(input_port),
                    component_id=component_ids[id(node)],
                    port_id=port_id,
                    port=input_port.name,
                )
            )
    return specs_out


def _external_input_support_diagnostics(
    input_specs: list[BatchedInputSpec],
    ports_by_id: Mapping[int, object],
) -> list[BatchedDiagnostic]:
    """Reject node-keyed input bindings that would alias distinct InputPorts.

    Count only the external ports that remain after successful in-composition
    projections have been lowered.  Live ``path_afferents`` may include stale
    projections owned by another Composition and are not evidence that a port
    is internally fed in this graph.
    """

    diagnostics = []
    by_component: dict[int, list[BatchedInputSpec]] = {}
    for input_spec in input_specs:
        input_port = ports_by_id[input_spec.port_id]
        if bool(_parameter_value(input_port, "internal_only", False)):
            diagnostics.append(
                BatchedDiagnostic(
                    input_spec.node,
                    "unsupported InputPort default/internal binding for batched v2",
                    (
                        f"{input_spec.port}: default_input=None, "
                        "internal_only=True"
                    ),
                )
            )
        by_component.setdefault(input_spec.component_id, []).append(input_spec)

    for external_ports in by_component.values():
        if len(external_ports) <= 1:
            continue
        diagnostics.append(
            BatchedDiagnostic(
                external_ports[0].node,
                "unsupported external multi-port input binding for batched v2",
                f"lowered external ports={[spec.port for spec in external_ports]!r}",
            )
        )
    return diagnostics


def _output_specs(
    composition,
    outputs,
    nodes,
    component_ids,
    port_ids,
) -> list[BatchedOutputSpec]:
    if outputs is not None:
        selected = [
            output
            if not isinstance(output, str)
            else _output_port_from_name(output, nodes)
            for output in outputs
        ]
        return _assign_output_specs(selected, component_ids, port_ids)

    terminal_names = _terminal_node_names(composition)
    if not terminal_names:
        terminal_names = [_node_name(nodes[-1])] if nodes else []
    selected_ports = []
    for node in nodes:
        if _node_name(node) not in terminal_names or type(node) is ControlMechanism:
            continue
        output_ports = tuple(getattr(node, "output_ports", []))
        mechanism_spec = specs.mechanism_spec_for(node)
        if mechanism_spec is not None:
            if mechanism_spec.outputs is not None:
                wanted = {decl.port for decl in mechanism_spec.outputs}
                selected = [port for port in output_ports if port.name in wanted]
            else:
                selected = [output_ports[0]] if output_ports else []
        elif specs.passthrough_spec_for(node) is not None:
            supported = _supported_output_ports(node)
            selected = [port for port in output_ports if port.name in supported]
        else:
            selected = [output_ports[0]] if output_ports else []
        selected_ports.extend(selected)
    return _assign_output_specs(selected_ports, component_ids, port_ids)


def _assign_output_specs(ports, component_ids, port_ids) -> list[BatchedOutputSpec]:
    specs_out = []
    flat_start = 0
    for port in ports:
        output = _output_spec_from_port(
            port,
            component_ids,
            port_ids,
            flat_start=flat_start,
        )
        specs_out.append(output)
        flat_start = output.flat_stop
    return specs_out


def _output_spec_from_port(
    port,
    component_ids,
    port_ids,
    *,
    flat_start: int,
) -> BatchedOutputSpec:
    owner = getattr(port, "owner", None)
    node_name = _node_name(owner)
    width = int(np.asarray(getattr(port, "value", [0.0])).reshape(-1).size)
    return BatchedOutputSpec(
        name=f"{node_name}.{port.name}",
        node=node_name,
        port=port.name,
        width=width,
        component_id=component_ids[id(owner)],
        port_id=port_ids[id(port)],
        flat_start=flat_start,
        flat_stop=flat_start + width,
    )


def _output_port_from_name(name: str, nodes):
    for node in nodes:
        if _node_name(node) == name:
            return getattr(node, "output_ports", [])[0]
    raise KeyError(f"Could not resolve batched output '{name}'.")


def _classify_model(nodes) -> str | None:
    executable_nodes = [node for node in nodes if type(node) is not ControlMechanism]
    if len(executable_nodes) == 1:
        mechanism_spec = specs.mechanism_spec_for(executable_nodes[0])
        if mechanism_spec is not None and mechanism_spec.single_node_model_kind:
            return mechanism_spec.single_node_model_kind
    if executable_nodes:
        return GRAPH_MODEL
    return None


def _fusion_kind(model_kind: str | None, nodes, composition=None) -> str | None:
    executable_nodes = [node for node in nodes if type(node) is not ControlMechanism]
    if not executable_nodes:
        return None

    mechanism_specs = []
    for node in executable_nodes:
        mechanism_spec = specs.mechanism_spec_for(node)
        if mechanism_spec is not None:
            mechanism_specs.append(mechanism_spec)
            continue
        if (
            specs.passthrough_spec_for(node) is not None
            and specs.function_spec_for(getattr(node, "function", None)) is not None
        ):
            continue
        return None

    if not mechanism_specs:
        return STATELESS_GRAPH_FUSION
    if composition is not None and _is_coevolving(composition, executable_nodes):
        return COEVOLVING_GRAPH_FUSION
    if any(spec.persistent_state for spec in mechanism_specs) or len(mechanism_specs) > 1:
        return STATEFUL_GRAPH_FUSION
    return DDM_GRAPH_FUSION


def _is_coevolving(composition, executable_nodes) -> bool:
    """A stateful terminator op (e.g. DDM, with a ``finished_output``) co-evolves
    with an upstream persistent stateful op that runs the whole trial (scheduled
    ``Always()``, e.g. an LCA), so they must step together in a fused loop rather
    than run sequentially.  Cue-terminated upstream ops (the toy stab-flex LCA,
    which settles before the DDM) are NOT ``Always`` and stay sequential.
    """

    return _coevolving_stepper(composition, executable_nodes) is not None


def _coevolving_stepper(composition, executable_nodes):
    """Return the persistent node in an Always/WhenFinished coupled loop."""

    conditions = _scheduler_conditions(composition)
    terminators = [
        node for node in executable_nodes
        if (spec := specs.mechanism_spec_for(node)) is not None and spec.is_terminator and spec.can_step
    ]
    if not terminators:
        return None
    for node in executable_nodes:
        spec = specs.mechanism_spec_for(node)
        if spec is None or not (spec.can_step and spec.persistent_state):
            continue
        if not _scheduler_condition_is_effective_always(
            composition,
            node,
            conditions,
        ):
            continue
        # An Always-scheduled persistent stepper must be a semantic ancestor of
        # the terminator.  Merely appearing before an unrelated terminator in
        # Composition.nodes is not a co-evolution relationship.
        if any(
            _processing_depends_on(composition, terminator, node)
            or _when_finished_depends_on(conditions.get(terminator), node)
            for terminator in terminators
        ):
            return node
    return None


def _processing_depends_on(composition, node, dependency) -> bool:
    dependency_dict = getattr(
        getattr(composition, "scheduler", None),
        "dependency_dict",
        {},
    )
    pending = list(dependency_dict.get(node, ()))
    visited = set()
    while pending:
        candidate = pending.pop()
        if candidate is dependency:
            return True
        if candidate in visited:
            continue
        visited.add(candidate)
        pending.extend(dependency_dict.get(candidate, ()))
    return False


def _when_finished_depends_on(condition, dependency) -> bool:
    return (
        type(condition) is WhenFinished
        and is_canonical_condition(condition)
        and tuple(getattr(condition, "args", ())) == (dependency,)
    )


def _op_kind(node) -> str:
    mechanism_spec = specs.mechanism_spec_for(node)
    if mechanism_spec is not None:
        return f"{mechanism_spec.label}IntegrateUntilFinished"
    return type(getattr(node, "function", None)).__name__


def _terminal_node_names(composition) -> list[str]:
    nodes = _composition_nodes(composition)
    dependency_dict = getattr(composition.graph_processing, "dependency_dict", {})
    parents = {parent for parents in dependency_dict.values() for parent in parents}
    dependents = set(dependency_dict.keys())
    terminal = []
    for node in nodes:
        if node not in parents and node in dependents:
            terminal.append(_node_name(node))
    return terminal


def _scheduler_view_without_nodes(composition, ignored_node_ids):
    """Build a read-only scheduler view with selected nodes and edges removed.

    PEC adds one origin ControlMechanism per fitted parameter.  Removing those
    mechanisms only from the lowered node list is insufficient: their outgoing
    dependency edges have already moved controlled mechanisms into later live
    consideration sets.  Rebuild topological levels from the scheduler's
    analyzed dependency graph so the resulting view has the schedule the model
    would have had without those host-side fit controls.

    The level graph deliberately retains every non-ignored Composition node,
    including nodes later absorbed into a batched op.  Such a node may order a
    surviving controller even though it has no component ID of its own.  The
    normal consideration-set snapshot projects these full levels onto lowered
    component IDs and compresses empty levels.  No live scheduler or
    Composition state is mutated.
    """

    scheduler = getattr(composition, "scheduler", None)
    live_dependency_dict = getattr(scheduler, "dependency_dict", None)
    if not isinstance(live_dependency_dict, Mapping):
        # An empty queue makes the subsequent declaration incomplete and keeps
        # compilation fail-closed if an analyzed scheduler cannot be inspected.
        return {}, ()

    active_nodes = tuple(
        node
        for node in _composition_nodes(composition)
        if id(node) not in ignored_node_ids
    )
    active_node_ids = {id(node) for node in active_nodes}
    dependency_dict = {
        node: frozenset(
            dependency
            for dependency in live_dependency_dict.get(node, ())
            if (
                id(dependency) in active_node_ids
                and dependency is not node
            )
        )
        for node in active_nodes
    }

    remaining = {
        node: set(dependencies)
        for node, dependencies in dependency_dict.items()
    }
    consideration_queue = []
    while remaining:
        ready = frozenset(
            node
            for node, dependencies in remaining.items()
            if not dependencies
        )
        if not ready:
            # Preserve the acyclic prefix.  Projection will observe the missing
            # components and mark the scheduler declaration incomplete.
            break
        consideration_queue.append(ready)
        for node in ready:
            del remaining[node]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)

    return dependency_dict, tuple(consideration_queue)


def _scheduler_ir_specs(
    composition,
    nodes,
    component_ids,
    *,
    dependency_dict=None,
    consideration_queue=None,
):
    """Lower typed explicit predicates and the scheduler's implicit defaults.

    This is semantic declaration only: whether a backend can execute the
    resulting pass region remains a separate capability decision.  In
    particular, no live Condition objects or component objects are retained.
    """

    regions = (
        BatchedScheduleRegionSpec(
            name="trial",
            kind="trial",
            time_scale="ENVIRONMENT_STATE_UPDATE",
        ),
        BatchedScheduleRegionSpec(
            name="pass",
            kind="pass",
            time_scale="PASS",
            parent="trial",
        ),
    )
    conditions = _scheduler_conditions(composition)
    if dependency_dict is None:
        dependency_dict = getattr(
            composition.graph_processing,
            "dependency_dict",
            {},
        )
    consideration_sets, consideration_set_ids, queue_complete = (
        _scheduler_consideration_set_specs(
            composition,
            nodes,
            component_ids,
            consideration_queue=consideration_queue,
        )
    )
    declared = []
    finished_dependencies = {}
    complete = queue_complete and not _scheduler_structural_conditions(composition)

    # Use dependency order rather than condition insertion order so component,
    # predicate, and finished-value IDs are deterministic across equivalent
    # Composition construction sequences.
    for node in nodes:
        condition = conditions.get(node)
        implicit = condition is None
        if implicit:
            # graph-scheduler supplies Always() for an origin with no explicit
            # condition, EveryNCalls(parent, 1) for one dependency, and an All
            # of those predicates for multiple dependencies.  Snapshot that
            # effective default; a future pass executor must not interpret an
            # absent predicate as unconditional execution.
            implicit_dependencies = tuple(sorted(
                (
                    dependency
                    for dependency in dependency_dict.get(node, ())
                    if id(dependency) in component_ids and dependency is not node
                ),
                key=lambda dependency: component_ids[id(dependency)],
            ))
            if not implicit_dependencies:
                condition_type = "Always"
            elif len(implicit_dependencies) == 1:
                condition_type = "EveryNCalls"
            else:
                condition_type = "AllEveryNCalls"
        else:
            condition_type = _supported_scheduler_condition_name(condition)
            if condition_type is None:
                complete = False
                continue
        component_id = component_ids[id(node)]
        consideration_set_id = consideration_set_ids.get(component_id, -1)
        if consideration_set_id < 0:
            complete = False
            continue
        dependencies = (
            tuple(_node_name(dependency) for dependency in implicit_dependencies)
            if implicit
            else ()
        )
        dependency_component_ids = (
            tuple(component_ids[id(dependency)] for dependency in implicit_dependencies)
            if implicit
            else ()
        )
        attrs = {"implicit": True} if implicit else {}
        region = "pass"

        if condition_type == "Always":
            pass
        elif condition_type in {"EveryNCalls", "AllEveryNCalls"}:
            attrs = {
                "implicit": True,
                "calls": 1,
                "time_scale": "ENVIRONMENT_STATE_UPDATE",
            }
        elif condition_type == "AtTrialStart":
            # AtTrialStart is evaluated in the first pass.  Trial-scoped state
            # reset is represented independently by BatchedResetSpec.
            attrs = {
                "pass_index": 0,
                "time_scale": "ENVIRONMENT_STATE_UPDATE",
            }
        elif condition_type == "AtPass":
            at_pass = _at_pass_spec(condition)
            if at_pass is None:
                complete = False
                continue
            pass_index, time_scale = at_pass
            if time_scale != "ENVIRONMENT_STATE_UPDATE":
                complete = False
                continue
            attrs = {
                "pass_index": pass_index,
                "time_scale": time_scale,
            }
        elif condition_type == "WhenFinished":
            args = tuple(getattr(condition, "args", ()))
            if len(args) != 1 or id(args[0]) not in component_ids:
                complete = False
                continue
            dependency = args[0]
            dependency_component_id = component_ids[id(dependency)]
            dependencies = (_node_name(dependency),)
            dependency_component_ids = (dependency_component_id,)
            finished_dependencies[dependency_component_id] = dependency
            attrs = {"predicate": "is_finished"}
        else:
            complete = False
            continue

        declared.append(
            BatchedSchedulerSpec(
                node=_node_name(node),
                condition_type=condition_type,
                dependencies=dependencies,
                attrs=attrs,
                component_id=component_id,
                dependency_component_ids=dependency_component_ids,
                region=region,
                consideration_set_id=consideration_set_id,
            )
        )

    finished_values = tuple(
        _finished_value_spec(
            dependency,
            composition,
            component_id=component_id,
            value_id=value_id,
            producer_consideration_set_id=consideration_set_ids[component_id],
        )
        for value_id, (component_id, dependency) in enumerate(
            sorted(finished_dependencies.items())
        )
    )
    finished_value_ids = {
        value.component_id: value.value_id
        for value in finished_values
    }
    declared = tuple(
        replace(
            condition,
            finished_value_ids=tuple(
                finished_value_ids[component_id]
                for component_id in condition.dependency_component_ids
            ) if condition.condition_type == "WhenFinished" else (),
        )
        for condition in declared
    )
    return declared, regions, consideration_sets, finished_values, complete


def _finished_value_spec(
    node,
    composition,
    *,
    component_id: int,
    value_id: int,
    producer_consideration_set_id: int,
) -> BatchedFinishedValueSpec:
    """Snapshot a registered, object-free scheduler-visible finished value."""

    predicate_kind = "dynamic"
    attrs = {}
    mechanism_spec = specs.mechanism_spec_for(node)
    resolver = (
        None
        if mechanism_spec is None
        else mechanism_spec.finished_after_execution_count
    )
    if resolver is not None:
        try:
            count = resolver(node, composition)
        except Exception:
            count = None
        if type(count) is int and count > 0:
            predicate_kind = "execution_count_at_least"
            attrs = {"count": count}

    return BatchedFinishedValueSpec(
        name=f"{_node_name(node)}.is_finished",
        node=_node_name(node),
        component_id=component_id,
        value_id=value_id,
        producer_consideration_set_id=producer_consideration_set_id,
        predicate_kind=predicate_kind,
        attrs=attrs,
    )


def _termination_ir_specs(composition, component_ids):
    """Snapshot the one termination contract precomputed scheduling can execute.

    PsyNeuLink evaluates termination between consideration sets, independently
    from node predicates.  The first executable scheduler tier therefore
    accepts only the scheduler defaults: trial-local ``AllHaveRun()`` over all
    lowered scheduler components and environment-sequence ``Never()``.  Custom
    termination remains inspectable as a structured capability rejection; no
    live Condition or Component object is retained in the graph IR.
    """

    composition_name = getattr(composition, "name", type(composition).__name__)
    scheduler = getattr(composition, "scheduler", None)
    conditions = getattr(scheduler, "termination_conds", None)
    if not isinstance(conditions, Mapping):
        return (), (
            BatchedDiagnostic(
                composition_name,
                "unsupported scheduler termination for batched v2",
                "scheduler does not expose typed termination predicates",
            ),
        )

    by_scale = {}
    for time_scale, condition in conditions.items():
        if time_scale is TimeScale.ENVIRONMENT_STATE_UPDATE:
            scale_name = "ENVIRONMENT_STATE_UPDATE"
        elif time_scale is TimeScale.ENVIRONMENT_SEQUENCE:
            scale_name = "ENVIRONMENT_SEQUENCE"
        else:
            return (), (
                BatchedDiagnostic(
                    composition_name,
                    "unsupported scheduler termination for batched v2",
                    "termination keys must be the canonical PsyNeuLink "
                    "ENVIRONMENT_STATE_UPDATE and ENVIRONMENT_SEQUENCE "
                    f"TimeScale members; got {time_scale!r}",
                ),
            )
        by_scale[scale_name] = condition

    expected_scales = {
        "ENVIRONMENT_STATE_UPDATE",
        "ENVIRONMENT_SEQUENCE",
    }
    if set(by_scale) != expected_scales:
        return (), (
            BatchedDiagnostic(
                composition_name,
                "unsupported scheduler termination for batched v2",
                "termination must contain exactly ENVIRONMENT_STATE_UPDATE "
                "AllHaveRun and ENVIRONMENT_SEQUENCE Never",
            ),
        )

    trial = by_scale["ENVIRONMENT_STATE_UPDATE"]
    raw_trial_args = getattr(trial, "args", None)
    trial_args = raw_trial_args if type(raw_trial_args) is tuple else None
    trial_time_scale = _condition_time_scale(trial)
    if (
        type(trial) is not AllHaveRun
        or trial_args != ()
        or trial_time_scale is not TimeScale.ENVIRONMENT_STATE_UPDATE
    ):
        return (), (
            BatchedDiagnostic(
                composition_name,
                "unsupported scheduler termination for batched v2",
                "ENVIRONMENT_STATE_UPDATE termination requires AllHaveRun over "
                "every scheduler component ID at "
                "time_scale=ENVIRONMENT_STATE_UPDATE; got "
                f"{type(trial).__name__}(args={raw_trial_args!r}, "
                "time_scale="
                f"{getattr(trial_time_scale, 'name', trial_time_scale) or 'unknown'})",
            ),
        )

    sequence = by_scale["ENVIRONMENT_SEQUENCE"]
    raw_sequence_args = getattr(sequence, "args", None)
    sequence_args = (
        raw_sequence_args if type(raw_sequence_args) is tuple else None
    )
    if type(sequence) is not Never or sequence_args != ():
        return (), (
            BatchedDiagnostic(
                composition_name,
                "unsupported scheduler termination for batched v2",
                "ENVIRONMENT_SEQUENCE termination requires Never with no "
                f"component operands; got {type(sequence).__name__}"
                f"(args={raw_sequence_args!r})",
            ),
        )

    scheduler_component_ids = tuple(sorted(component_ids.values()))
    return (
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_STATE_UPDATE",
            condition_type="AllHaveRun",
            dependency_component_ids=scheduler_component_ids,
        ),
        BatchedTerminationSpec(
            time_scale="ENVIRONMENT_SEQUENCE",
            condition_type="Never",
        ),
    ), ()


def _scheduler_consideration_set_specs(
    composition,
    nodes,
    component_ids,
    *,
    consideration_queue=None,
):
    """Snapshot the scheduler's ordered consideration queue without live objects."""

    if consideration_queue is None:
        scheduler = getattr(composition, "scheduler", None)
        queue = getattr(scheduler, "consideration_queue", None)
    else:
        queue = consideration_queue
    if queue is None:
        return (), {}, False

    lowered_component_ids = set(component_ids.values())
    seen = set()
    declarations = []
    consideration_set_ids = {}
    for live_set in queue:
        members = sorted(
            (
                (component_ids[id(node)], _node_name(node))
                for node in live_set
                if id(node) in component_ids
            ),
            key=lambda item: item[0],
        )
        if not members:
            continue
        consideration_set_id = len(declarations)
        component_id_tuple = tuple(component_id for component_id, _ in members)
        declarations.append(
            BatchedConsiderationSetSpec(
                consideration_set_id=consideration_set_id,
                nodes=tuple(name for _, name in members),
                component_ids=component_id_tuple,
            )
        )
        for component_id in component_id_tuple:
            # A scheduler node belongs to exactly one consideration set.  Treat
            # malformed queues as an incomplete declaration rather than
            # choosing one occurrence.
            if component_id in seen:
                return tuple(declarations), consideration_set_ids, False
            seen.add(component_id)
            consideration_set_ids[component_id] = consideration_set_id

    return (
        tuple(declarations),
        consideration_set_ids,
        seen == lowered_component_ids,
    )


def _reset_ir_specs(nodes, states, component_ids):
    """Declare reset policies for state retained in ``BatchedGraphIR``.

    Optimized-away per-trial state (currently the affine one-step integrating
    TransferMechanism path) has no state identity and therefore needs no reset
    event.  Trial-local mechanism-op storage is not yet part of GraphIR; moving
    that storage out of emitter-private declarations is the next reset slice.
    """

    states_by_component = {}
    for state in states:
        states_by_component.setdefault(state.component_id, []).append(state)

    declarations = []
    complete = True
    for node in nodes:
        component_id = component_ids[id(node)]
        component_states = states_by_component.get(component_id, ())
        if not component_states:
            continue
        reset_condition = getattr(node, "reset_stateful_function_when", None)
        if type(reset_condition) is AtTrialStart and is_canonical_condition(
            reset_condition
        ):
            condition_type = "AtTrialStart"
        elif type(reset_condition) is Never and is_canonical_condition(
            reset_condition
        ):
            condition_type = "Never"
        else:
            complete = False
            continue
        declarations.append(
            BatchedResetSpec(
                node=_node_name(node),
                condition_type=condition_type,
                state_ids=tuple(state.state_id for state in component_states),
                component_id=component_id,
            )
        )
    return tuple(declarations), complete


def _classify_schedule(
    composition,
    nodes,
    component_ids,
    consideration_set_ids,
    finished_values_by_component_id,
    coevolving=False,
) -> tuple[str, list[str], list[BatchedDiagnostic]]:
    conditions = _scheduler_conditions(composition)
    structural_conditions = _scheduler_structural_conditions(composition)
    structural_rejections = [
        BatchedDiagnostic(
            component=_node_name(node),
            reason="unsupported structural scheduler condition for batched v2",
            detail=type(condition).__name__,
        )
        for node, node_conditions in structural_conditions.items()
        for condition in node_conditions
    ]
    if not conditions:
        if structural_rejections:
            return UNSUPPORTED_SCHEDULE, [], structural_rejections
        return STATIC_GRAPH_SCHEDULE, [], []

    node_consideration_set = {
        id(node): consideration_set_ids.get(component_ids[id(node)], -1)
        for node in nodes
    }
    supported: list[str] = []
    rejected: list[BatchedDiagnostic] = structural_rejections
    required_schedule_kind = (
        UNSUPPORTED_SCHEDULE if structural_rejections else STATIC_GRAPH_SCHEDULE
    )

    for node, condition in conditions.items():
        node_name = _node_name(node)
        # Skip conditions on nodes that are not lowered as graph ops (absorbed
        # into another op's kernel); their timing is handled by that op.
        if id(node) not in component_ids or type(node) is ControlMechanism:
            continue
        condition_name = type(condition).__name__
        condition_schedule_kind = _condition_schedule_kind(
            condition,
            node,
            node_consideration_set,
            component_ids,
            finished_values_by_component_id,
            coevolving,
        )
        if condition_schedule_kind == UNSUPPORTED_SCHEDULE:
            rejected.append(
                BatchedDiagnostic(
                    component=node_name,
                    reason="unsupported scheduler condition for static batched graph",
                    detail=_unsupported_scheduler_condition_detail(condition),
                )
            )
            required_schedule_kind = UNSUPPORTED_SCHEDULE
            continue

        supported.append(f"{node_name}: {condition_name}")
        if condition_schedule_kind == STATIC_GRAPH_SCHEDULE:
            continue

        if required_schedule_kind != UNSUPPORTED_SCHEDULE:
            required_schedule_kind = condition_schedule_kind
        if not (
            condition_schedule_kind == PRECOMPUTED_TRACE_SCHEDULE
            and condition_name in {"AtPass", "WhenFinished"}
        ):
            rejected.append(
                BatchedDiagnostic(
                    component=node_name,
                    reason="batched schedule kind is not executable yet",
                    detail=f"{condition_name} requires {condition_schedule_kind}",
                )
            )

    if rejected:
        return required_schedule_kind, supported, rejected
    return required_schedule_kind, supported, []


def _precomputed_schedule_support_diagnostic(
    graph: BatchedGraphIR,
) -> BatchedDiagnostic | None:
    """Validate the exact first executable precomputed-schedule boundary."""

    node_ids = tuple(sorted(node.component_id for node in graph.nodes))
    execution_ids = tuple(sorted(
        graph.node(node_name).component_id
        for node_name in graph.execution_order
    ))
    scheduler_ids = tuple(sorted(
        condition.component_id for condition in graph.scheduler
    ))
    consideration_ids = tuple(sorted(
        component_id
        for consideration_set in graph.consideration_sets
        for component_id in consideration_set.component_ids
    ))
    stateless_trace = (
        graph.fusion_kind == STATELESS_GRAPH_FUSION
        and not graph.states
        and not graph.rng_streams
        and not graph.resets
        and not graph.finished_values
    )
    counted_finished_trace = _is_counted_finished_precomputed_graph(graph)
    boundary_reasons = []
    if not stateless_trace and not counted_finished_trace:
        if graph.fusion_kind != STATELESS_GRAPH_FUSION:
            boundary_reasons.append(f"fusion_kind={graph.fusion_kind!r}")
        if graph.states:
            boundary_reasons.append("retained state")
        if graph.rng_streams:
            boundary_reasons.append("RNG streams")
        if graph.resets:
            boundary_reasons.append("reset policies")
        if graph.finished_values:
            boundary_reasons.append("finished-value dependencies")
        boundary_reasons.append("not an exact counted-finished stepper/follower graph")
    if not (
        node_ids == execution_ids == scheduler_ids == consideration_ids
    ):
        boundary_reasons.append("scheduler components without executable bodies")
    if any(node.attrs.get("diagnostics") for node in graph.nodes):
        boundary_reasons.append("mechanism diagnostics")
    if boundary_reasons:
        return BatchedDiagnostic(
            component=graph.metadata.get("composition_name") or "Composition",
            reason="batched schedule is not executable",
            detail=(
                "precomputed trace requires a stateless trial-lane graph or "
                "the exact counted-finished stateful boundary; "
            )
            + ", ".join(boundary_reasons),
        )

    try:
        plan_precomputed_schedule_trace(
            scheduler=graph.scheduler,
            consideration_sets=graph.consideration_sets,
            termination=graph.termination,
            expansion_budget=PRECOMPUTED_TRACE_COMPONENT_BUDGET,
            projections=graph.projections,
            finished_values=graph.finished_values,
        )
    except BatchedScheduleTraceError as error:
        return BatchedDiagnostic(
            component=graph.metadata.get("composition_name") or "Composition",
            reason="batched schedule is not executable",
            detail=f"{error.code}: {error.detail}",
        )
    return None


def _is_counted_finished_precomputed_graph(graph: BatchedGraphIR) -> bool:
    """Recognize the first typed counted-finished execution boundary.

    The implementation is intentionally expressed in typed component,
    scheduler, state, and port semantics.  Display names and Composition node
    insertion order are irrelevant.
    """

    if (
        graph.fusion_kind != STATEFUL_GRAPH_FUSION
        or len(graph.nodes) != 2
        or len(graph.finished_values) != 1
        or graph.rng_streams
        or len(graph.projections) != 1
    ):
        return False

    finished = graph.finished_values[0]
    if (
        finished.predicate_kind != "execution_count_at_least"
        or finished.storage != "combinational"
        or finished.width != 1
        or finished.dtype != "bool"
        or not isinstance(finished.attrs, Mapping)
        or set(finished.attrs) != {"count"}
        or type(finished.attrs.get("count")) is not int
        or finished.attrs["count"] <= 0
    ):
        return False

    nodes_by_id = {node.component_id: node for node in graph.nodes}
    producer = nodes_by_id.get(finished.component_id)
    follower_ids = set(nodes_by_id) - {finished.component_id}
    if producer is None or len(follower_ids) != 1:
        return False
    follower = nodes_by_id[follower_ids.pop()]
    if (
        producer.component_type != "LCAMechanism"
        or producer.attrs.get("spec_kind") != "mechanism"
        or producer.attrs.get("diagnostics")
        or follower.component_type != "TransferMechanism"
        or follower.attrs.get("spec_kind") != "elementwise"
        or follower.attrs.get("diagnostics")
    ):
        return False

    conditions = {
        condition.component_id: condition
        for condition in graph.scheduler
    }
    if set(conditions) != set(nodes_by_id):
        return False
    producer_condition = conditions[producer.component_id]
    follower_condition = conditions[follower.component_id]
    if (
        producer_condition.condition_type != "Always"
        or producer_condition.dependencies
        or producer_condition.dependency_component_ids
        or producer_condition.finished_value_ids
        or follower_condition.condition_type != "WhenFinished"
        or follower_condition.dependencies != (producer.name,)
        or follower_condition.dependency_component_ids != (producer.component_id,)
        or follower_condition.finished_value_ids != (finished.value_id,)
        or producer_condition.consideration_set_id
        >= follower_condition.consideration_set_id
        or finished.producer_consideration_set_id
        != producer_condition.consideration_set_id
    ):
        return False

    producer_state_ids = tuple(
        state.state_id
        for state in graph.states
        if state.component_id == producer.component_id
    )
    if not producer_state_ids or len(producer_state_ids) != len(graph.states):
        return False
    if len(graph.resets) != 1:
        return False
    reset = graph.resets[0]
    if (
        reset.node != producer.name
        or reset.component_id != producer.component_id
        or reset.condition_type not in {"Never", "AtTrialStart"}
        or reset.state_ids != producer_state_ids
        or reset.attrs
        or reset.region != "trial"
    ):
        return False

    projection = graph.projections[0]
    return (
        projection.sender_component_id == producer.component_id
        and projection.receiver_component_id == follower.component_id
        and graph.execution_order == (producer.name, follower.name)
        and bool(graph.inputs)
        and all(
            input_spec.component_id == producer.component_id
            for input_spec in graph.inputs
        )
    )


def _condition_schedule_kind(
    condition,
    node,
    node_consideration_set: dict[int, int],
    component_ids,
    finished_values_by_component_id,
    coevolving=False,
) -> str:
    condition_name = _supported_scheduler_condition_name(condition)
    if condition_name is None:
        deferred_name = type(condition).__name__
        if deferred_name in _PRECOMPUTED_TRACE_CONDITIONS:
            return PRECOMPUTED_TRACE_SCHEDULE
        if deferred_name in _DYNAMIC_LANE_LOCAL_CONDITIONS:
            return DYNAMIC_LANE_LOCAL_SCHEDULE
        return UNSUPPORTED_SCHEDULE
    if condition_name in {"Always", "AtTrialStart"}:
        return STATIC_GRAPH_SCHEDULE
    if condition_name == "WhenFinished":
        args = getattr(condition, "args", ())
        if len(args) != 1:
            return DYNAMIC_LANE_LOCAL_SCHEDULE
        target = args[0]
        target_set = node_consideration_set.get(id(target), -1)
        receiver_set = node_consideration_set.get(id(node), -1)
        if target_set >= 0 and target_set < receiver_set:
            target_component_id = component_ids.get(id(target))
            if not bool(
                _parameter_value(
                    target,
                    "execute_until_finished",
                    True,
                )
            ):
                if (
                    not coevolving
                    and _fixed_finished_execution_count(
                        finished_values_by_component_id.get(target_component_id)
                    )
                    is not None
                ):
                    return PRECOMPUTED_TRACE_SCHEDULE
                # A one-step mechanism with a lane-varying or otherwise
                # non-constant finished predicate requires conditional
                # lane-local pass execution.  Treating it as a static graph
                # would execute the target only once.
                return DYNAMIC_LANE_LOCAL_SCHEDULE
            return STATIC_GRAPH_SCHEDULE
        return DYNAMIC_LANE_LOCAL_SCHEDULE
    if condition_name == "AtPass":
        # AtPass(0) means "fire only on pass 0 of the trial".  In the batched
        # static/stateful graph every node already computes once per trial
        # (origins load their input once and hold it), so "fire once at trial
        # start" is exactly the batched origin semantics -> static graph.
        # AtPass(n>0) is a delayed within-trial onset (e.g. the ITI before
        # taskInput becomes active).  The fused co-evolution loop can gate that
        # per step (the input is withheld until step n, and the terminator is
        # frozen), so it is executable there; without a per-step loop it is only
        # recognized, not executable (`precomputed_trace`).
        at_pass = _at_pass_spec(condition)
        if at_pass is None:
            return UNSUPPORTED_SCHEDULE
        n, time_scale = at_pass
        if time_scale != "ENVIRONMENT_STATE_UPDATE":
            return UNSUPPORTED_SCHEDULE
        if n == 0:
            return STATIC_GRAPH_SCHEDULE
        return STATIC_GRAPH_SCHEDULE if coevolving else PRECOMPUTED_TRACE_SCHEDULE
    return UNSUPPORTED_SCHEDULE


def _onset_step(node, composition) -> int:
    """The `AtPass(n)` onset step for `node` (0 if none / not AtPass)."""
    condition = _scheduler_conditions(composition).get(node)
    at_pass = _at_pass_spec(condition)
    if at_pass is None or at_pass[1] != "ENVIRONMENT_STATE_UPDATE":
        return 0
    return at_pass[0]


_SUPPORTED_SCHEDULER_CONDITION_TYPES = {
    Always: "Always",
    AtPass: "AtPass",
    AtTrialStart: "AtTrialStart",
    WhenFinished: "WhenFinished",
}


def _supported_scheduler_condition_name(condition) -> str | None:
    """Name an exact supported PNL condition class; subclasses fail closed."""

    if not is_canonical_condition(condition):
        return None
    return _SUPPORTED_SCHEDULER_CONDITION_TYPES.get(type(condition))


def _at_pass_spec(condition) -> tuple[int, str] | None:
    """Validate and snapshot an exact PNL ``AtPass`` predicate."""

    if type(condition) is not AtPass or not is_canonical_condition(condition):
        return None
    args = tuple(getattr(condition, "args", ()))
    if len(args) != 1 or isinstance(args[0], (bool, np.bool_)):
        return None
    try:
        pass_index = int(args[0])
        exact_integer = bool(args[0] == pass_index)
    except (OverflowError, TypeError, ValueError):
        return None
    time_scale = _condition_time_scale_name(condition)
    if not exact_integer or pass_index < 0 or time_scale is None:
        return None
    return pass_index, time_scale


def _unsupported_scheduler_condition_detail(condition) -> str:
    if type(condition) is not AtPass:
        return type(condition).__name__
    return (
        "AtPass requires one non-negative non-bool integer index at "
        "time_scale=ENVIRONMENT_STATE_UPDATE; "
        f"args={tuple(getattr(condition, 'args', ()))!r}, "
        f"time_scale={_condition_time_scale_name(condition) or 'unknown'}"
    )


def _scheduler_conditions(composition):
    scheduler = getattr(composition, "scheduler", None)
    if scheduler is None:
        return {}
    # ``Scheduler.run`` materializes graph-scheduler's implicit defaults into
    # the live condition set, while ``Scheduler.remove_condition`` can leave
    # its user-provenance sidecar stale.  The live set is the semantic truth:
    # filter only predicates exactly equivalent to the scheduler default, then
    # lower those defaults independently from the dependency graph above.
    condition_set = getattr(scheduler, "conditions", None)
    conditions_basic = getattr(condition_set, "conditions_basic", {})
    if not hasattr(conditions_basic, "items"):
        return {}
    dependency_dict = getattr(scheduler, "dependency_dict", {})
    return {
        node: condition
        for node, condition in conditions_basic.items()
        if not _is_implicit_scheduler_default(
            condition,
            tuple(dependency_dict.get(node, ())),
        )
    }


def _scheduler_condition_is_effective_always(
    composition,
    node,
    conditions=None,
) -> bool:
    """Whether a node's explicit or dependency-derived predicate is Always."""

    if node is None:
        return False
    if conditions is None:
        conditions = _scheduler_conditions(composition)
    condition = conditions.get(node)
    if condition is not None:
        return type(condition) is Always and is_canonical_condition(condition)
    dependency_dict = getattr(
        getattr(composition, "scheduler", None),
        "dependency_dict",
        {},
    )
    return not tuple(dependency_dict.get(node, ()))


def _scheduler_structural_conditions(composition):
    scheduler = getattr(composition, "scheduler", None)
    if scheduler is None:
        return {}
    condition_set = getattr(scheduler, "conditions", None)
    conditions_structural = getattr(condition_set, "conditions_structural", {})
    if not hasattr(conditions_structural, "items"):
        return {}
    return conditions_structural


def _is_implicit_scheduler_default(condition, dependencies) -> bool:
    """Whether ``condition`` is exactly graph-scheduler's generated default."""

    dependencies = tuple(dependencies)
    if not dependencies:
        return type(condition) is Always and is_canonical_condition(condition)
    if len(dependencies) == 1:
        return _is_default_every_n_calls(condition, dependencies[0])
    if type(condition) is not All or not is_canonical_condition(condition):
        return False
    operands = tuple(getattr(condition, "args", ()))
    if len(operands) != len(dependencies):
        return False
    operand_dependencies = []
    for operand in operands:
        args = tuple(getattr(operand, "args", ()))
        if (
            type(operand) is not EveryNCalls
            or not is_canonical_condition(operand)
            or len(args) != 2
            or args[1] != 1
        ):
            return False
        operand_dependencies.append(args[0])
    return {id(item) for item in operand_dependencies} == {
        id(item) for item in dependencies
    }


def _is_default_every_n_calls(condition, dependency) -> bool:
    args = tuple(getattr(condition, "args", ()))
    return (
        type(condition) is EveryNCalls
        and is_canonical_condition(condition)
        and len(args) == 2
        and args[0] is dependency
        and args[1] == 1
    )


def _dependency_topological_order(composition, nodes):
    """Return a deterministic dependency order and any nodes in dependency cycles.

    PsyNeuLink's processing dependency graph is authoritative for execution
    precedence.  Self dependencies represent recurrent state (for example an
    LCA's AutoAssociativeProjection) and do not constrain within-pass emission.
    Nodes at the same topological level are ordered by their stable PNL names so
    disconnected components do not inherit Composition insertion order.
    """

    node_set = set(nodes)
    dependency_dict = getattr(composition.graph_processing, "dependency_dict", {})
    dependencies = {
        node: {
            dependency
            for dependency in dependency_dict.get(node, ())
            if dependency in node_set and dependency is not node
        }
        for node in nodes
    }
    dependents = {node: set() for node in nodes}
    for node, node_dependencies in dependencies.items():
        for dependency in node_dependencies:
            dependents[dependency].add(node)

    key = lambda node: (_node_name(node), type(node).__module__, type(node).__qualname__)
    ready = sorted((node for node in nodes if not dependencies[node]), key=key)
    ordered = []
    while ready:
        node = ready.pop(0)
        ordered.append(node)
        for dependent in sorted(dependents[node], key=key):
            dependencies[dependent].discard(node)
            if not dependencies[dependent] and dependent not in ordered and dependent not in ready:
                ready.append(dependent)
        ready.sort(key=key)

    cyclic = tuple(sorted((node for node in nodes if node not in ordered), key=key))
    return ordered + list(cyclic), cyclic


def _composition_nodes(composition) -> list[Any]:
    return list(getattr(composition, "nodes", []))


def _node_name(node) -> str:
    return getattr(node, "name", str(node))


def _duplicate_component_names(components) -> list[str]:
    names = [getattr(component, "name", str(component)) for component in components]
    return sorted({name for name in names if names.count(name) > 1})


def _duplicate_node_name_diagnostics(composition, nodes) -> list[BatchedDiagnostic]:
    """Temporary fail-closed boundary until all graph lookup is ID-native."""

    return [
        BatchedDiagnostic(
            getattr(composition, "name", "Composition"),
            "duplicate live node names are unsupported for batched v2",
            f"name={name!r}",
        )
        for name in _duplicate_component_names(nodes)
    ]


def _node_param_aliases(node_name: str, param_name: str) -> tuple[str, ...]:
    qualified = f"{node_name}.{param_name}"
    base_name = _unsuffixed_node_name(node_name)
    if base_name == node_name:
        return (qualified,)
    return (qualified, f"{base_name}.{param_name}")


def _unsuffixed_node_name(node_name: str) -> str:
    return re.sub(r"-\d+$", "", node_name)


def _integrator_mode_enabled(node) -> bool:
    parameters = getattr(node, "parameters", None)
    param = getattr(parameters, "integrator_mode", None) if parameters is not None else None
    if param is not None:
        try:
            return bool(param.get(None))
        except Exception:
            pass
    return bool(getattr(getattr(node, "defaults", None), "integrator_mode", False))


def _integrating_transfer_affine(node, composition) -> tuple[float, float] | None:
    """Affine single-step integrator coefficients ``(a, b)`` for a stateless
    integrating transfer (``integ = a*input + b``), or ``None`` if the node is
    not a supported fires-once, reset-each-trial integrator_mode transfer.

    Sound only when the node advances its integrator exactly one step per trial
    from its initializer: it must reset every trial (``AtTrialStart``) and fire
    once per trial (an ``AtPass`` schedule).  Supports the AdaptiveIntegrator
    (``value = (1-rate)*init + rate*input``) and SimpleIntegrator
    (``value = init + rate*input + offset``); noise is assumed 0.
    """

    if not _integrator_mode_enabled(node):
        return None
    reset_condition = getattr(node, "reset_stateful_function_when", None)
    if type(reset_condition) is not AtTrialStart or not is_canonical_condition(
        reset_condition
    ):
        return None
    at_pass = _at_pass_spec(_scheduler_conditions(composition).get(node))
    if at_pass is None or at_pass[1] != "ENVIRONMENT_STATE_UPDATE":
        return None
    integrator = getattr(node, "integrator_function", None)
    integrator_type = type(integrator).__name__
    init = specs.resolve_component_param(integrator, "initializer", 0.0)
    rate = specs.resolve_component_param(integrator, "rate", 1.0)
    if integrator_type == "AdaptiveIntegrator":
        return (rate, (1.0 - rate) * init)
    if integrator_type == "SimpleIntegrator":
        offset = specs.resolve_component_param(integrator, "offset", 0.0)
        return (rate, init + offset)
    return None


def _absorbed_nodes(composition, nodes) -> set[str]:
    """Names of nodes folded into another op's kernel rather than lowered.

    Currently: the stateful integrating transfer that drives a DDM's collapsing
    threshold (its per-step offset is read directly into the DDM boundary, so the
    node itself is not executed as a graph op).
    """

    from psyneulink.core.batched.components.ddm import threshold_override_collapse

    absorbed: set[str] = set()
    for node in nodes:
        if type(node).__name__ != "DDM":
            continue
        try:
            chain = threshold_override_collapse(node)
        except Exception:
            chain = None
        if chain is None:
            continue
        for control in nodes:
            if type(control) is not ControlMechanism:
                continue
            attrs = _absorbed_control_attrs(control)
            if (
                attrs["source"] == chain[0]
                and attrs["target"] == _node_name(node)
                and attrs["parameter"] == "threshold"
                and _control_support_diagnostic(control, composition) is None
            ):
                absorbed.add(chain[0])
                break
    return absorbed


def _combine_name(node) -> str:
    input_ports = getattr(node, "input_ports", [])
    if not input_ports:
        return "sum"
    return _input_port_combine_name(input_ports[0])


def _input_port_combine_name(input_port) -> str:
    """Canonical operation for the port's actual LinearCombination function."""

    function = getattr(input_port, "function", None)
    operation = _parameter_value(function, "operation", None)
    if operation is None:
        operation = getattr(input_port, "combine", None)
    return str(operation or "sum").lower()


def _input_port_attrs(node, port_ids) -> tuple[tuple[str, int, str, int, int, int], ...]:
    """Semantic per-InputPort layout in mechanism-variable order.

    Each entry is ``(name, width, combine, port_id, flat_start, flat_stop)``.
    The flattened bounds describe only the handoff from the independent port
    values to the node function; projection accumulation never crosses them.
    """

    result = []
    flat_start = 0
    for input_port in tuple(getattr(node, "input_ports", ())):
        width = _port_width(input_port)
        result.append(
            (
                input_port.name,
                width,
                _input_port_combine_name(input_port),
                port_ids[id(input_port)],
                flat_start,
                flat_start + width,
            )
        )
        flat_start += width
    return tuple(result)


def _elementwise_output_port_slices(
    node,
    port_ids,
) -> tuple[tuple[str, int, int, int, int], ...]:
    """Identity OutputPort slices as ``(name,width,id,start,stop)``."""

    result = []
    for output_port in tuple(getattr(node, "output_ports", ())):
        bounds = _identity_output_port_slice(node, output_port)
        if bounds is None:
            # Validation reports this before node lowering.  Retain a hard
            # internal guard for direct/private callers so no port is omitted.
            raise ValueError(
                f"Batched elementwise node '{_node_name(node)}' has an unmodeled "
                f"OutputPort '{getattr(output_port, 'name', output_port)}'."
            )
        flat_start, flat_stop = bounds
        result.append(
            (
                output_port.name,
                flat_stop - flat_start,
                port_ids[id(output_port)],
                flat_start,
                flat_stop,
            )
        )
    return tuple(result)


def _identity_output_port_slice(node, output_port) -> tuple[int, int] | None:
    """Flattened mechanism-value slice selected by an identity OutputPort."""

    function = getattr(output_port, "function", None)
    owner_value_index = _owner_value_selector_index(output_port)
    input_ports = tuple(getattr(node, "input_ports", ()))
    if (
        owner_value_index is None
        or owner_value_index < 0
        or owner_value_index >= len(input_ports)
        or not _is_identity_linear(function)
    ):
        return None
    widths = tuple(_port_width(port) for port in input_ports)
    index = int(owner_value_index)
    if _port_width(output_port) != widths[index]:
        return None
    start = sum(widths[:index])
    return start, start + widths[index]


def _input_width(node) -> int:
    input_ports = getattr(node, "input_ports", [])
    if not input_ports:
        return _primary_output_width(node)
    return sum(_port_width(port) for port in input_ports)


def _port_width(port) -> int:
    try:
        return int(np.asarray(port.value).reshape(-1).size)
    except Exception:
        return 1


def _node_output_width(node, mechanism_spec) -> int:
    if mechanism_spec is not None:
        if mechanism_spec.outputs:
            return mechanism_spec.outputs[0].width
        return _primary_output_width(node)
    if specs.passthrough_spec_for(node) is not None:
        # Built-in passthrough functions are elementwise over the complete
        # mechanism value.  Multi-port values are flattened in InputPort order
        # for the backend and split back into modeled OutputPort slices.
        return _input_width(node)
    return _primary_output_width(node)


def _primary_output_width(node) -> int:
    output_ports = getattr(node, "output_ports", [])
    if not output_ports:
        return 1
    try:
        return int(np.asarray(output_ports[0].value).reshape(-1).size)
    except Exception:
        return 1


def _get_matrix(projection) -> np.ndarray:
    parameters = getattr(projection, "parameters", None)
    if parameters is not None and hasattr(parameters, "matrix"):
        try:
            return np.asarray(parameters.matrix.get(None))
        except Exception:
            pass
    defaults = getattr(projection, "defaults", None)
    if defaults is not None and hasattr(defaults, "matrix"):
        return np.asarray(defaults.matrix)
    return np.eye(1, dtype=np.float32)
