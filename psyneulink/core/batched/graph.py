from __future__ import annotations

from dataclasses import dataclass, replace
from collections.abc import Iterable, Mapping
import inspect
import re
from typing import Any

import numpy as np

from psyneulink.core.batched import specs
from psyneulink.core.batched.bindings import BatchedComponentBindings, projection_binding_key
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.ir import (
    BatchedConsiderationSetSpec,
    BatchedFinishedValueSpec,
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedNodeSpec,
    BatchedOp,
    BatchedOutputSpec,
    BatchedParamSpec,
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


def lower_composition(composition, outputs=None) -> LoweringResult:
    specs.ensure_builtin_specs()

    nodes = _composition_nodes(composition)
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
        if type(node).__name__ != "ControlMechanism"
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
    )
    finished_values_by_component_id = {
        value.component_id: value
        for value in finished_values
    }
    termination_specs, termination_rejections = _termination_ir_specs(
        composition,
        component_ids,
    )
    consideration_set_ids = {
        component_id: consideration_set.consideration_set_id
        for consideration_set in consideration_sets
        for component_id in consideration_set.component_ids
    }
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
        if component_type == "ControlMechanism":
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

    if coevolving and not any(
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
                "coevolving Always/WhenFinished execution requires explicit "
                "finished predicates and conditional pass regions in KernelIR",
            )
        )

    _freeze_absorbed_control_parameters(node_specs, params)

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
    input_rejections = _external_input_support_diagnostics(inputs)
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
            if type(node).__name__ != "ControlMechanism"
        )
        ops = tuple(
            BatchedOp(kind=_op_kind(node), target=_node_name(node))
            for node in topological_nodes
            if type(node).__name__ != "ControlMechanism"
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
            executable=not rejected_nodes and not rejected_conditions,
            metadata={
                "composition_name": getattr(composition, "name", None),
                "schedule_kind": schedule_kind,
                "scheduler_executable": not rejected_nodes and not rejected_conditions,
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
            schedule_regions=schedule_regions,
            consideration_sets=consideration_sets,
            finished_values=finished_values,
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


def _freeze_absorbed_control_parameters(
    node_specs: list[BatchedNodeSpec],
    params: _ParamBuilder,
) -> None:
    """Freeze values whose compile-time identity semantics are intentionally erased."""

    nodes_by_name = {node.name: node for node in node_specs}
    for node in node_specs:
        source_name = node.attrs.get("termination_input_node")
        if node.component_type != "LCAMechanism" or source_name is None:
            continue
        # A referenced source can have failed its own semantic lowering.  Its
        # structured rejection is already part of the capability report; do
        # not turn that unsupported model into an internal compiler error.
        source = nodes_by_name.get(source_name)
        if source is None:
            continue
        for parameter_name in source.params.values():
            params.freeze(
                parameter_name,
                (
                    f"absorbed identity termination-threshold source for "
                    f"{node.name}"
                ),
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
    # A delayed within-trial onset (AtPass(n>0)): the co-evolution loop withholds
    # this node's output until step n. Only meaningful/executable in a co-evolving
    # graph (AtPass(n>0) is rejected at the schedule level otherwise).
    onset = _onset_step(node, composition)
    if onset > 0:
        attrs["onset_step"] = onset

    mechanism_spec = specs.mechanism_spec_for(node)
    function_spec = specs.function_spec_for(function)
    output_width = _node_output_width(node, mechanism_spec)

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
            param_map[binding.arg] = params.add(
                public_name,
                binding.resolve(node),
                aliases=aliases,
                parameter=_bound_parameter(binding, node),
                minimum=binding.minimum,
                minimum_inclusive=binding.minimum_inclusive,
                maximum=binding.maximum,
                maximum_inclusive=binding.maximum_inclusive,
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
    )


def _node_support_diagnostic(
    node,
    composition,
    *,
    component_id: int,
    finished_values_by_component_id: Mapping[int, BatchedFinishedValueSpec],
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
            and type(getattr(node, "reset_stateful_function_when", None))
            is not AtTrialStart
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
    default_input = _parameter_value(input_port, "default_input", None)
    internal_only = bool(_parameter_value(input_port, "internal_only", False))
    if default_input is not None or internal_only:
        return BatchedDiagnostic(
            _node_name(node),
            "unsupported InputPort default/internal binding for batched v2",
            (
                f"{port_name}: default_input={default_input!r}, "
                f"internal_only={internal_only!r}"
            ),
        )
    if type(function).__name__ != "LinearCombination":
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
    if type(function).__name__ != "Linear":
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
    stepwise = stepwise_ddm_pair or counted_finished_pair
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
    stepper = args[0] if type(condition) is WhenFinished and len(args) == 1 else None
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


def _control_support_diagnostic(control, composition) -> BatchedDiagnostic | None:
    """Accept only control edges whose semantics are explicitly folded by an op."""

    name = _node_name(control)
    signals, efferents, monitors = _control_edges(control)
    if len(signals) != 1 or len(efferents) != 1 or len(monitors) != 1:
        return BatchedDiagnostic(
            name,
            "unsupported generic ControlMechanism for batched v2",
            "requires exactly one monitor, ControlSignal, and ControlProjection",
        )
    control_projection = efferents[0]
    monitor_projection = monitors[0]
    if type(control_projection).__name__ != "ControlProjection" or type(monitor_projection).__name__ != "MappingProjection":
        return BatchedDiagnostic(name, "unsupported generic control projection for batched v2")
    signal = signals[0]
    if not _is_override(getattr(signal, "modulation", getattr(control, "modulation", None))):
        return BatchedDiagnostic(
            name,
            "unsupported control modulation for batched v2",
            str(getattr(signal, "modulation", None)),
        )
    input_ports = tuple(getattr(control, "input_ports", ()))
    if len(input_ports) != 1:
        return BatchedDiagnostic(
            name,
            "unsupported control input routing for batched v2",
            f"input_ports={len(input_ports)}",
        )
    input_diagnostic = _input_port_function_support_diagnostic(control, input_ports[0])
    if input_diagnostic is not None:
        return BatchedDiagnostic(
            name,
            "unsupported control input semantics for batched v2",
            input_diagnostic.detail,
        )
    if (
        not _is_identity_linear(getattr(control_projection, "function", None))
        or _parameter_value(control_projection, "weight", None) is not None
        or _parameter_value(control_projection, "exponent", None) is not None
    ):
        return BatchedDiagnostic(
            name,
            "unsupported ControlProjection semantics for batched v2",
            "requires identity Linear with no weight or exponent",
        )
    if not _control_signal_is_identity(signal):
        return BatchedDiagnostic(
            name,
            "unsupported ControlSignal semantics for batched v2",
            "requires an identity TransferWithCosts transfer function",
        )

    receiver = getattr(control_projection, "receiver", None)
    target = getattr(receiver, "owner", None)
    source = getattr(getattr(monitor_projection, "sender", None), "owner", None)
    target_port = getattr(receiver, "name", "")
    monitor_is_identity = _is_identity_scalar_projection(monitor_projection)
    monitor_is_parameter_input = _is_external_parameter_projection(
        composition, monitor_projection
    )
    if target is None or source is None or not (monitor_is_identity or monitor_is_parameter_input):
        return BatchedDiagnostic(
            name,
            "unsupported control monitor routing for batched v2",
            "monitor must be a scalar identity projection",
        )
    try:
        target_afferents = tuple(receiver.mod_afferents)
    except Exception:
        target_afferents = ()
    if target_afferents != (control_projection,):
        return BatchedDiagnostic(
            name,
            "ambiguous control projection routing for batched v2",
            f"{_node_name(target)}.{target_port}",
        )

    function = getattr(control, "function", None)
    function_diagnostic = _function_parameter_support_diagnostic(name, function)
    if function_diagnostic is not None:
        return function_diagnostic

    if (
        monitor_is_parameter_input
        and type(function).__name__ == "Identity"
        and _is_declared_batched_parameter(target, target_port)
    ):
        return None

    if type(target).__name__ == "LCAMechanism" and target_port == "termination_threshold":
        from psyneulink.core.batched.components.lca import _control_monitor_source_for

        if _is_unmodeled_coevolving_lca_termination(composition, target):
            return BatchedDiagnostic(
                name,
                "batched schedule kind is not executable yet",
                "coevolving Always/WhenFinished execution requires an LCA "
                "finished predicate and termination-control value that KernelIR "
                "does not model",
            )
        schedule_diagnostic = _absorbed_lca_schedule_support_diagnostic(
            composition,
            source,
            control,
            target,
        )
        if schedule_diagnostic is not None:
            return schedule_diagnostic
        identity = type(function).__name__ == "Identity" or _is_identity_linear(function)
        if (
            identity
            and _control_monitor_source_for(composition, target) is source
            and _supported_lca_termination_source(source)
        ):
            return None
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
    for role, component in (
        ("source", source),
        ("controller", control),
        ("target", target),
    ):
        condition = conditions.get(component)
        if condition is None:
            continue
        if (
            role == "source"
            and type(condition).__name__ == "AtPass"
            and tuple(getattr(condition, "args", ())) == (0,)
            and _condition_time_scale_name(condition)
            == "ENVIRONMENT_STATE_UPDATE"
        ):
            continue
        return BatchedDiagnostic(
            _node_name(control),
            "unsupported absorbed control scheduler condition for batched v2",
            f"{role} {_node_name(component)} uses {_condition_label(condition)}",
        )
    return None


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
    return str(value).upper().endswith("OVERRIDE")


def _control_signal_is_identity(signal) -> bool:
    function = getattr(signal, "function", None)
    transfer_function = _parameter_value(function, "transfer_fct", None)
    return (
        type(function).__name__ == "TransferWithCosts"
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


def _supported_lca_termination_source(source) -> bool:
    if specs.passthrough_spec_for(source) is None or len(tuple(getattr(source, "input_ports", ()))) != 1:
        return False
    if any(getattr(port, "path_afferents", ()) for port in getattr(source, "input_ports", ())):
        return False
    function = getattr(source, "function", None)
    return (
        _is_identity_linear(function)
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
    function = getattr(source, "function", None)
    integrator = getattr(source, "integrator_function", None)
    if type(function).__name__ != "Linear" or type(integrator).__name__ != "SimpleIntegrator":
        return False
    if _function_parameter_support_diagnostic(_node_name(source), function) is not None:
        return False
    if not _numeric_equal(_parameter_value(function, "slope", 1.0), 1.0):
        return False
    if not _is_zero(_parameter_value(source, "noise", 0.0)) or _parameter_value(source, "clip", None) is not None:
        return False
    if not _is_zero(_parameter_value(integrator, "noise", 0.0)) or not _is_zero(_parameter_value(integrator, "initializer", 0.0)):
        return False
    if type(getattr(source, "reset_stateful_function_when", None)) is not AtTrialStart:
        return False
    if any(getattr(port, "path_afferents", ()) for port in getattr(source, "input_ports", ())):
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


def _same_condition(left, right) -> bool:
    if type(left) is not type(right):
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
                if type(receiver).__name__ == "ControlMechanism":
                    continue
                if type(sender).__name__ == "ControlMechanism" or projection_type == "ControlProjection":
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
        type(function).__name__ != "MatrixTransform"
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
) -> list[BatchedDiagnostic]:
    """Reject node-keyed input bindings that would alias distinct InputPorts.

    Count only the external ports that remain after successful in-composition
    projections have been lowered.  Live ``path_afferents`` may include stale
    projections owned by another Composition and are not evidence that a port
    is internally fed in this graph.
    """

    by_component: dict[int, list[BatchedInputSpec]] = {}
    for input_spec in input_specs:
        by_component.setdefault(input_spec.component_id, []).append(input_spec)

    diagnostics = []
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
        if _node_name(node) not in terminal_names or type(node).__name__ == "ControlMechanism":
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
    executable_nodes = [node for node in nodes if type(node).__name__ != "ControlMechanism"]
    if len(executable_nodes) == 1:
        mechanism_spec = specs.mechanism_spec_for(executable_nodes[0])
        if mechanism_spec is not None and mechanism_spec.single_node_model_kind:
            return mechanism_spec.single_node_model_kind
    if executable_nodes:
        return GRAPH_MODEL
    return None


def _fusion_kind(model_kind: str | None, nodes, composition=None) -> str | None:
    executable_nodes = [node for node in nodes if type(node).__name__ != "ControlMechanism"]
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


def _scheduler_ir_specs(composition, nodes, component_ids):
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
    dependency_dict = getattr(composition.graph_processing, "dependency_dict", {})
    consideration_sets, consideration_set_ids, queue_complete = (
        _scheduler_consideration_set_specs(composition, nodes, component_ids)
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


def _scheduler_consideration_set_specs(composition, nodes, component_ids):
    """Snapshot the scheduler's ordered consideration queue without live objects."""

    scheduler = getattr(composition, "scheduler", None)
    queue = getattr(scheduler, "consideration_queue", None)
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
        if type(reset_condition) is AtTrialStart:
            condition_type = "AtTrialStart"
        elif type(reset_condition) is Never:
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
        if id(node) not in component_ids or type(node).__name__ == "ControlMechanism":
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
    if (
        len(graph.resets) != 1
        or graph.resets[0].component_id != producer.component_id
        or graph.resets[0].condition_type != "Never"
        or graph.resets[0].state_ids != producer_state_ids
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
            if (
                not coevolving
                and not bool(
                    _parameter_value(
                        target,
                        "execute_until_finished",
                        True,
                    )
                )
                and _fixed_finished_execution_count(
                    finished_values_by_component_id.get(target_component_id)
                )
                is not None
            ):
                return PRECOMPUTED_TRACE_SCHEDULE
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

    return _SUPPORTED_SCHEDULER_CONDITION_TYPES.get(type(condition))


def _at_pass_spec(condition) -> tuple[int, str] | None:
    """Validate and snapshot an exact PNL ``AtPass`` predicate."""

    if type(condition) is not AtPass:
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
        return type(condition) is Always
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
        return type(condition) is Always and not tuple(condition.args)
    if len(dependencies) == 1:
        return _is_default_every_n_calls(condition, dependencies[0])
    if type(condition) is not All:
        return False
    operands = tuple(getattr(condition, "args", ()))
    if len(operands) != len(dependencies):
        return False
    operand_dependencies = []
    for operand in operands:
        args = tuple(getattr(operand, "args", ()))
        if type(operand) is not EveryNCalls or len(args) != 2 or args[1] != 1:
            return False
        operand_dependencies.append(args[0])
    return {id(item) for item in operand_dependencies} == {
        id(item) for item in dependencies
    }


def _is_default_every_n_calls(condition, dependency) -> bool:
    args = tuple(getattr(condition, "args", ()))
    return (
        type(condition) is EveryNCalls
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
    if type(getattr(node, "reset_stateful_function_when", None)) is not AtTrialStart:
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
            if type(control).__name__ != "ControlMechanism":
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
