from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import re
from typing import Any

import numpy as np

from psyneulink.core.batched import specs
from psyneulink.core.batched.bindings import BatchedComponentBindings, projection_binding_key
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.ir import (
    BatchedGraphIR,
    BatchedInputSpec,
    BatchedNodeSpec,
    BatchedOp,
    BatchedOutputSpec,
    BatchedParamSpec,
    BatchedProjectionSpec,
    BatchedSchedulerSpec,
    BatchedStateSpec,
)


GRAPH_MODEL = "graph"
DDM_MODEL = "ddm"
STATELESS_GRAPH_FUSION = "stateless_graph"
DDM_GRAPH_FUSION = "ddm_graph"
STATEFUL_GRAPH_FUSION = "stateful_graph"
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
    node_names = {_node_name(node) for node in nodes}
    params = _ParamBuilder()
    rejected_nodes: list[BatchedDiagnostic] = []
    supported_nodes: list[str] = []

    model_kind = _classify_model(nodes)
    schedule_kind, supported_conditions, rejected_conditions = _classify_schedule(composition, nodes)
    node_bindings = {
        _node_name(node): node
        for node in nodes
        if type(node).__name__ != "ControlMechanism"
    }
    function_bindings = {
        _node_name(node): getattr(node, "function", None)
        for node in nodes
        if type(node).__name__ != "ControlMechanism"
    }
    node_specs = []
    state_specs = []

    for node in nodes:
        component_type = type(node).__name__
        node_name = _node_name(node)
        if component_type == "ControlMechanism":
            supported_nodes.append(node_name)
            node_specs.append(_node_spec(node, params, model_kind, composition))
            continue

        diagnostic = _node_support_diagnostic(node)
        if diagnostic is not None:
            rejected_nodes.append(diagnostic)
            continue

        supported_nodes.append(node_name)
        node_spec = _node_spec(node, params, model_kind, composition)
        node_specs.append(node_spec)
        mechanism_spec = specs.mechanism_spec_for(node)
        if mechanism_spec is not None:
            for state_decl in mechanism_spec.states:
                width = state_decl.width if state_decl.width is not None else node_spec.output_width
                state_specs.append(
                    BatchedStateSpec(
                        f"{node_name}.{state_decl.name}",
                        node_name,
                        width,
                        tuple([state_decl.initial] * width),
                    )
                )

    projections, projection_rejections, projection_bindings = _projection_specs(composition, node_names)
    rejected_nodes.extend(projection_rejections)

    graph = None
    if not rejected_nodes and not rejected_conditions:
        inputs = _input_specs(nodes, projections)
        outputs = _output_specs(composition, outputs, nodes)
        execution_order = tuple(
            _node_name(node)
            for node in nodes
            if type(node).__name__ != "ControlMechanism"
        )
        ops = tuple(
            BatchedOp(kind=_op_kind(node), target=_node_name(node))
            for node in nodes
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
            scheduler=tuple(
                BatchedSchedulerSpec(_node_name(node), type(condition).__name__)
                for node, condition in _scheduler_conditions(composition).items()
            ),
            ops=ops,
            execution_order=execution_order,
            fusion_kind=_fusion_kind(model_kind, nodes),
            metadata={
                "composition_name": getattr(composition, "name", None),
                "schedule_kind": schedule_kind,
            },
        )

    return LoweringResult(
        graph=graph,
        params=tuple(params.specs),
        bindings=BatchedComponentBindings(
            nodes=node_bindings,
            functions=function_bindings,
            projections=projection_bindings,
        ),
        model_kind=model_kind,
        schedule_kind=schedule_kind,
        supported_nodes=tuple(supported_nodes),
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=tuple(supported_conditions),
        rejected_conditions=tuple(rejected_conditions),
    )


def projection_inputs(graph: BatchedGraphIR, receiver: str) -> tuple[BatchedProjectionSpec, ...]:
    return tuple(projection for projection in graph.projections if projection.receiver == receiver)


class _ParamBuilder:
    def __init__(self):
        self.specs: list[BatchedParamSpec] = []
        self._names: set[str] = set()

    def add(self, name: str, default: float, aliases: Iterable[str] = ()) -> str:
        if name in self._names:
            return name
        self._names.add(name)
        self.specs.append(BatchedParamSpec(name, float(default), tuple(aliases)))
        return name


def _node_spec(node, params: _ParamBuilder, model_kind: str | None, composition) -> BatchedNodeSpec:
    component_type = type(node).__name__
    function = getattr(node, "function", None)
    function_type = type(function).__name__
    node_name = _node_name(node)
    combine = _combine_name(node)
    param_map: dict[str, str] = {}
    attrs: dict[str, Any] = {
        "output_ports": tuple(port.name for port in getattr(node, "output_ports", [])),
    }

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
            param_map[binding.arg] = params.add(public_name, binding.resolve(node), aliases=aliases)
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
    elif specs.passthrough_spec_for(node) is not None and function_spec is not None:
        attrs["spec_kind"] = "elementwise"
        attrs["spec_key"] = function_spec.key
        for binding in function_spec.params:
            param_map[binding.arg] = params.add(
                f"{node_name}.{binding.arg}",
                binding.resolve(function),
                aliases=_node_param_aliases(node_name, binding.arg),
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
    )


def _node_support_diagnostic(node) -> BatchedDiagnostic | None:
    function = getattr(node, "function", None)
    function_type = type(function).__name__
    node_name = _node_name(node)

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
        return None

    if specs.passthrough_spec_for(node) is not None:
        if _integrator_mode_enabled(node):
            # A TransferMechanism with integrator_mode=True is a stateful leaky/
            # simple integrator, not the stateless transfer the passthrough spec
            # assumes. Reject it rather than silently lower it as a stateless
            # function (a stateful integrating-transfer op is a future milestone).
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


def _projection_specs(
    composition,
    node_names: set[str],
) -> tuple[list[BatchedProjectionSpec], list[BatchedDiagnostic], dict[str, object]]:
    projections: list[BatchedProjectionSpec] = []
    rejected: list[BatchedDiagnostic] = []
    bindings: dict[str, object] = {}
    for node in _composition_nodes(composition):
        for input_port in getattr(node, "input_ports", []):
            for projection in getattr(input_port, "path_afferents", []):
                projection_type = type(projection).__name__
                sender = getattr(getattr(projection, "sender", None), "owner", None)
                receiver = getattr(getattr(projection, "receiver", None), "owner", None)
                if sender is None or receiver is None:
                    continue
                sender_name = _node_name(sender)
                receiver_name = _node_name(receiver)
                if sender_name not in node_names or receiver_name not in node_names:
                    continue
                if projection_type in {"AutoAssociativeProjection", "ControlProjection"}:
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
                sender_port = getattr(getattr(projection, "sender", None), "name", "RESULT")
                receiver_port = getattr(getattr(projection, "receiver", None), "name", "InputPort-0")
                projections.append(
                    BatchedProjectionSpec(
                        sender=sender_name,
                        sender_port=sender_port,
                        receiver=receiver_name,
                        receiver_port=receiver_port,
                        matrix=np.asarray(_get_matrix(projection), dtype=np.float32),
                        spec_key=projection_spec.key,
                    )
                )
                bindings[
                    projection_binding_key(sender_name, sender_port, receiver_name, receiver_port)
                ] = projection
    return projections, rejected, bindings


def _input_specs(nodes, projections: list[BatchedProjectionSpec]) -> list[BatchedInputSpec]:
    receiver_names = {projection.receiver for projection in projections}
    specs_out = []
    for node in nodes:
        component_type = type(node).__name__
        node_name = _node_name(node)
        if component_type == "ControlMechanism":
            continue
        if node_name in receiver_names:
            continue
        specs_out.append(BatchedInputSpec(name=node_name, node=node_name, width=_input_width(node)))
    return specs_out


def _output_specs(composition, outputs, nodes) -> list[BatchedOutputSpec]:
    if outputs is not None:
        return [_output_spec_from_port(output) if not isinstance(output, str) else _output_spec_from_name(output, nodes) for output in outputs]

    terminal_names = _terminal_node_names(composition)
    if not terminal_names:
        terminal_names = [_node_name(nodes[-1])] if nodes else []
    specs_out = []
    for node in nodes:
        if _node_name(node) not in terminal_names or type(node).__name__ == "ControlMechanism":
            continue
        output_ports = tuple(getattr(node, "output_ports", []))
        mechanism_spec = specs.mechanism_spec_for(node)
        if mechanism_spec is not None and mechanism_spec.outputs is not None:
            wanted = {decl.port for decl in mechanism_spec.outputs}
            selected = [port for port in output_ports if port.name in wanted]
        else:
            selected = [output_ports[0]] if output_ports else []
        for port in selected:
            specs_out.append(_output_spec_from_port(port))
    return specs_out


def _output_spec_from_port(port) -> BatchedOutputSpec:
    owner = getattr(port, "owner", None)
    node_name = _node_name(owner)
    width = int(np.asarray(getattr(port, "value", [0.0])).reshape(-1).size)
    return BatchedOutputSpec(name=f"{node_name}.{port.name}", node=node_name, port=port.name, width=width)


def _output_spec_from_name(name: str, nodes) -> BatchedOutputSpec:
    for node in nodes:
        if _node_name(node) == name:
            port = getattr(node, "output_ports", [])[0]
            return _output_spec_from_port(port)
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


def _fusion_kind(model_kind: str | None, nodes) -> str | None:
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
    if any(spec.persistent_state for spec in mechanism_specs) or len(mechanism_specs) > 1:
        return STATEFUL_GRAPH_FUSION
    return DDM_GRAPH_FUSION


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


def _classify_schedule(composition, nodes) -> tuple[str, list[str], list[BatchedDiagnostic]]:
    conditions = _scheduler_conditions(composition)
    if not conditions:
        return STATIC_GRAPH_SCHEDULE, [], []

    node_index = {_node_name(node): idx for idx, node in enumerate(nodes)}
    supported: list[str] = []
    rejected: list[BatchedDiagnostic] = []
    required_schedule_kind = STATIC_GRAPH_SCHEDULE

    for node, condition in conditions.items():
        node_name = _node_name(node)
        condition_name = type(condition).__name__
        condition_schedule_kind = _condition_schedule_kind(condition, node, node_index)
        supported.append(f"{node_name}: {condition_name}")

        if condition_schedule_kind == STATIC_GRAPH_SCHEDULE:
            continue
        if condition_schedule_kind == UNSUPPORTED_SCHEDULE:
            rejected.append(
                BatchedDiagnostic(
                    component=node_name,
                    reason="unsupported scheduler condition for static batched graph",
                    detail=condition_name,
                )
            )
            required_schedule_kind = UNSUPPORTED_SCHEDULE
            continue

        if required_schedule_kind != UNSUPPORTED_SCHEDULE:
            required_schedule_kind = condition_schedule_kind
        rejected.append(
            BatchedDiagnostic(
                component=node_name,
                reason="batched schedule kind is not executable yet",
                detail=f"{condition_name} requires {condition_schedule_kind}",
            )
        )

    if rejected:
        return required_schedule_kind, supported, rejected
    return STATIC_GRAPH_SCHEDULE, supported, []


def _condition_schedule_kind(condition, node, node_index: dict[str, int]) -> str:
    condition_name = type(condition).__name__
    if condition_name in {"Always", "AtTrialStart"}:
        return STATIC_GRAPH_SCHEDULE
    if condition_name == "WhenFinished":
        args = getattr(condition, "args", ())
        if len(args) != 1:
            return DYNAMIC_LANE_LOCAL_SCHEDULE
        target = args[0]
        target_name = _node_name(target)
        if node_index.get(target_name, -1) < node_index.get(_node_name(node), -1):
            return STATIC_GRAPH_SCHEDULE
        return DYNAMIC_LANE_LOCAL_SCHEDULE
    if condition_name in _PRECOMPUTED_TRACE_CONDITIONS:
        return PRECOMPUTED_TRACE_SCHEDULE
    if condition_name in _DYNAMIC_LANE_LOCAL_CONDITIONS:
        return DYNAMIC_LANE_LOCAL_SCHEDULE
    return UNSUPPORTED_SCHEDULE


def _scheduler_conditions(composition):
    scheduler = getattr(composition, "scheduler", None)
    if scheduler is None:
        return {}
    condition_set = getattr(scheduler, "conditions", None)
    conditions_basic = getattr(condition_set, "conditions_basic", {})
    if not hasattr(conditions_basic, "items"):
        return {}
    return conditions_basic


def _composition_nodes(composition) -> list[Any]:
    return list(getattr(composition, "nodes", []))


def _node_name(node) -> str:
    return getattr(node, "name", str(node))


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


def _combine_name(node) -> str:
    input_ports = getattr(node, "input_ports", [])
    if not input_ports:
        return "sum"
    combine = getattr(input_ports[0], "combine", None)
    if combine is None:
        return "sum"
    return str(combine).lower()


def _input_width(node) -> int:
    input_ports = getattr(node, "input_ports", [])
    if not input_ports:
        return _primary_output_width(node)
    try:
        return int(np.asarray(input_ports[0].value).reshape(-1).size)
    except Exception:
        return _primary_output_width(node)


def _node_output_width(node, mechanism_spec) -> int:
    if mechanism_spec is not None and mechanism_spec.outputs:
        return mechanism_spec.outputs[0].width
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
            return np.asarray(parameters.matrix.get(None), dtype=np.float32)
        except Exception:
            pass
    defaults = getattr(projection, "defaults", None)
    if defaults is not None and hasattr(defaults, "matrix"):
        return np.asarray(defaults.matrix, dtype=np.float32)
    return np.eye(1, dtype=np.float32)
