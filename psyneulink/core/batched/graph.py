from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

import numpy as np

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
STABILITY_FLEXIBILITY_MODEL = "stability_flexibility"
STATELESS_GRAPH_FUSION = "stateless_graph"
DDM_GRAPH_FUSION = "ddm_graph"
STATEFUL_GRAPH_FUSION = "stateful_graph"

_SUPPORTED_SCHEDULER_CONDITIONS = {
    "Always",
    "AtTrialStart",
    "EveryNCalls",
    "WhenFinished",
}

_STATELESS_MECHANISMS = {"TransferMechanism", "ProcessingMechanism"}
_STATELESS_FUNCTIONS = {"Linear", "Logistic"}


@dataclass(frozen=True)
class LoweringResult:
    graph: BatchedGraphIR | None
    params: tuple[BatchedParamSpec, ...]
    bindings: BatchedComponentBindings
    model_kind: str | None
    supported_nodes: tuple[str, ...]
    rejected_nodes: tuple[BatchedDiagnostic, ...]
    supported_conditions: tuple[str, ...]
    rejected_conditions: tuple[BatchedDiagnostic, ...]


def lower_composition(composition, outputs=None) -> LoweringResult:
    nodes = _composition_nodes(composition)
    node_names = {_node_name(node) for node in nodes}
    params = _ParamBuilder()
    rejected_nodes: list[BatchedDiagnostic] = []
    supported_nodes: list[str] = []

    roles = _infer_stability_flexibility_roles(composition, nodes)
    model_kind = _classify_model(nodes, roles)
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
            node_specs.append(_node_spec(node, params, model_kind, roles))
            continue

        diagnostic = _node_support_diagnostic(node)
        if diagnostic is not None:
            rejected_nodes.append(diagnostic)
            continue

        supported_nodes.append(node_name)
        node_spec = _node_spec(node, params, model_kind, roles)
        node_specs.append(node_spec)
        if component_type == "LCAMechanism":
            state_specs.extend(
                (
                    BatchedStateSpec(f"{node_name}.pre", node_name, node_spec.output_width, tuple([0.0] * node_spec.output_width)),
                    BatchedStateSpec(f"{node_name}.act", node_name, node_spec.output_width, tuple([0.0] * node_spec.output_width)),
                )
            )
        elif component_type == "DDM":
            state_specs.append(BatchedStateSpec(f"{node_name}.value", node_name, 1, (0.0,)))

    projections, projection_rejections, projection_bindings = _projection_specs(composition, node_names)
    rejected_nodes.extend(projection_rejections)
    supported_conditions, rejected_conditions = _analyze_scheduler_conditions(composition)

    graph = None
    if not rejected_nodes and not rejected_conditions:
        inputs = _input_specs(nodes, projections, roles)
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
                "stability_flexibility_roles": roles,
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
        supported_nodes=tuple(supported_nodes),
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=tuple(supported_conditions),
        rejected_conditions=tuple(rejected_conditions),
    )


def is_stateless_graph(graph: BatchedGraphIR) -> bool:
    return all(graph.node(node_name).component_type in _STATELESS_MECHANISMS for node_name in graph.execution_order)


def node_by_name(graph: BatchedGraphIR, name: str) -> BatchedNodeSpec:
    return graph.node(name)


def projection_inputs(graph: BatchedGraphIR, receiver: str) -> tuple[BatchedProjectionSpec, ...]:
    return tuple(projection for projection in graph.projections if projection.receiver == receiver)


def scheduler_condition_names(composition) -> tuple[list[BatchedDiagnostic], list[str]]:
    rejected: list[BatchedDiagnostic] = []
    supported: list[str] = []
    for node, condition in _scheduler_conditions(composition).items():
        condition_name = type(condition).__name__
        node_name = _node_name(node)
        if condition_name in _SUPPORTED_SCHEDULER_CONDITIONS:
            supported.append(f"{node_name}: {condition_name}")
        else:
            rejected.append(
                BatchedDiagnostic(
                    component=node_name,
                    reason="unsupported scheduler condition",
                    detail=condition_name,
                )
            )
    return rejected, supported


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


def _node_spec(node, params: _ParamBuilder, model_kind: str | None, roles: dict[str, str]) -> BatchedNodeSpec:
    component_type = type(node).__name__
    function = getattr(node, "function", None)
    function_type = type(function).__name__
    node_name = _node_name(node)
    input_width = _input_width(node)
    output_width = _primary_output_width(node)
    combine = _combine_name(node)
    param_map: dict[str, str] = {}
    attrs: dict[str, Any] = {}

    if component_type in _STATELESS_MECHANISMS:
        attrs["output_ports"] = tuple(port.name for port in getattr(node, "output_ports", []))
        if function_type == "Linear":
            slope_name = _stateless_param_name(node_name, "slope", roles)
            intercept_name = _stateless_param_name(node_name, "intercept", roles)
            param_map["slope"] = params.add(
                slope_name,
                _get_param(function, "slope", 1.0),
                aliases=(f"{node_name}.slope",),
            )
            param_map["intercept"] = params.add(
                intercept_name,
                _get_param(function, "intercept", 0.0),
                aliases=(f"{node_name}.intercept",),
            )
        elif function_type == "Logistic":
            param_map["gain"] = params.add(
                f"{node_name}.gain",
                _get_param(function, "gain", 1.0),
                aliases=(f"{node_name}.gain",),
            )
    elif component_type == "DDM":
        function = node.function
        prefix = "ddm_" if model_kind == STABILITY_FLEXIBILITY_MODEL else ""
        aliases_prefix = "DDM"
        param_map["rate"] = params.add(
            "rate" if model_kind == DDM_MODEL else f"{node_name}.rate",
            _get_param(function, "rate", 1.0),
            aliases=("ddm.rate", f"{aliases_prefix}.rate"),
        )
        param_map["noise"] = params.add(
            f"{prefix}noise" if model_kind == STABILITY_FLEXIBILITY_MODEL else "noise",
            _get_param(function, "noise", 0.0),
            aliases=("ddm.noise", f"{aliases_prefix}.noise"),
        )
        param_map["threshold"] = params.add(
            "threshold",
            _get_param(function, "threshold", 1.0),
            aliases=("ddm.threshold", f"{aliases_prefix}.threshold"),
        )
        param_map["non_decision_time"] = params.add(
            "non_decision_time",
            _get_param(function, "non_decision_time", 0.0),
            aliases=("ddm.non_decision_time", f"{aliases_prefix}.non_decision_time"),
        )
        param_map["time_step_size"] = params.add(
            f"{prefix}time_step_size" if model_kind == STABILITY_FLEXIBILITY_MODEL else "time_step_size",
            _get_param(function, "time_step_size", 1.0),
            aliases=("ddm.time_step_size", f"{aliases_prefix}.time_step_size"),
        )
        param_map["starting_value"] = params.add(
            "starting_value",
            _get_param(function, "initializer", _get_param(function, "starting_value", 0.0)),
            aliases=("ddm.starting_value", f"{aliases_prefix}.starting_value"),
        )
        param_map["offset"] = params.add(
            "ddm_offset" if model_kind == STABILITY_FLEXIBILITY_MODEL else "offset",
            _get_param(function, "offset", 0.0),
            aliases=("ddm.offset", f"{aliases_prefix}.offset"),
        )
        attrs["output_ports"] = tuple(port.name for port in getattr(node, "output_ports", []))
    elif component_type == "LCAMechanism":
        function = getattr(node, "function", None)
        param_map["gain"] = params.add("gain", _get_param(function, "gain", 1.0), aliases=(f"{node_name}.gain",))
        param_map["leak"] = params.add(
            "leak",
            _get_param(node, "leak", _get_param(getattr(node, "integrator_function", None), "rate", 1.0)),
            aliases=(f"{node_name}.leak",),
        )
        param_map["competition"] = params.add(
            "competition",
            _get_param(node, "competition", 1.0),
            aliases=(f"{node_name}.competition",),
        )
        param_map["self_excitation"] = params.add(
            "self_excitation",
            _get_param(node, "self_excitation", _get_param(node, "auto", 0.0)),
            aliases=(f"{node_name}.self_excitation",),
        )
        param_map["noise"] = params.add("lca_noise", _get_param(node, "noise", 0.0), aliases=(f"{node_name}.noise",))
        param_map["time_step_size"] = params.add(
            "lca_time_step_size",
            _get_param(node, "time_step_size", 0.01),
            aliases=(f"{node_name}.time_step_size",),
        )
        attrs["termination_input_node"] = roles.get("cue_node")
        attrs["termination_threshold"] = _get_param(node, "termination_threshold", 1200)

    return BatchedNodeSpec(
        name=node_name,
        component_type=component_type,
        function_type=function_type,
        input_width=input_width,
        output_width=output_width,
        combine=combine,
        params=param_map,
        attrs=attrs,
    )


def _node_support_diagnostic(node) -> BatchedDiagnostic | None:
    component_type = type(node).__name__
    function_type = type(getattr(node, "function", None)).__name__
    node_name = _node_name(node)
    if component_type in _STATELESS_MECHANISMS:
        if function_type not in _STATELESS_FUNCTIONS:
            return BatchedDiagnostic(node_name, "unsupported function for batched v2", function_type)
        combine = _combine_name(node)
        if combine not in {"sum", "product"}:
            return BatchedDiagnostic(node_name, "unsupported input combine for batched v2", combine)
        return None
    if component_type == "DDM":
        if function_type != "DriftDiffusionIntegrator":
            return BatchedDiagnostic(node_name, "unsupported DDM function for batched v2", function_type)
        return None
    if component_type == "LCAMechanism":
        if function_type != "Logistic":
            return BatchedDiagnostic(node_name, "unsupported LCA function for batched v2", function_type)
        width = _primary_output_width(node)
        if width != 2:
            return BatchedDiagnostic(node_name, "unsupported LCA width for batched v2", f"width={width}")
        return None
    return BatchedDiagnostic(node_name, "unsupported node for batched v2", component_type)


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
                if projection_type != "MappingProjection":
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
                    )
                )
                bindings[
                    projection_binding_key(sender_name, sender_port, receiver_name, receiver_port)
                ] = projection
    return projections, rejected, bindings


def _input_specs(nodes, projections: list[BatchedProjectionSpec], roles: dict[str, str]) -> list[BatchedInputSpec]:
    receiver_names = {projection.receiver for projection in projections}
    specs = []
    for node in nodes:
        component_type = type(node).__name__
        node_name = _node_name(node)
        if component_type == "ControlMechanism":
            continue
        if node_name in receiver_names:
            continue
        specs.append(BatchedInputSpec(name=node_name, node=node_name, width=_input_width(node)))

    for role in ("cue_node", "correct_node"):
        node_name = roles.get(role)
        if node_name and node_name not in {spec.node for spec in specs}:
            node = _find_node_by_name(nodes, node_name)
            if node is not None:
                specs.append(BatchedInputSpec(name=node_name, node=node_name, width=_input_width(node)))
    return specs


def _output_specs(composition, outputs, nodes) -> list[BatchedOutputSpec]:
    if outputs is not None:
        return [_output_spec_from_port(output) if not isinstance(output, str) else _output_spec_from_name(output, nodes) for output in outputs]

    terminal_names = _terminal_node_names(composition)
    if not terminal_names:
        terminal_names = [_node_name(nodes[-1])] if nodes else []
    specs = []
    for node in nodes:
        if _node_name(node) not in terminal_names or type(node).__name__ == "ControlMechanism":
            continue
        output_ports = tuple(getattr(node, "output_ports", []))
        if type(node).__name__ == "DDM":
            selected = [port for port in output_ports if port.name in {"DECISION_OUTCOME", "RESPONSE_TIME"}]
        else:
            selected = [output_ports[0]] if output_ports else []
        for port in selected:
            specs.append(_output_spec_from_port(port))
    return specs


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


def _classify_model(nodes, roles: dict[str, str]) -> str | None:
    executable_nodes = [node for node in nodes if type(node).__name__ != "ControlMechanism"]
    ddm_nodes = [node for node in executable_nodes if type(node).__name__ == "DDM"]
    lca_nodes = [node for node in executable_nodes if type(node).__name__ == "LCAMechanism"]
    if len(executable_nodes) == 1 and len(ddm_nodes) == 1:
        return DDM_MODEL
    if len(ddm_nodes) == 1 and len(lca_nodes) == 1 and roles:
        return STABILITY_FLEXIBILITY_MODEL
    if executable_nodes:
        return GRAPH_MODEL
    return None


def _fusion_kind(model_kind: str | None, nodes) -> str | None:
    if model_kind == DDM_MODEL:
        return DDM_MODEL
    executable_nodes = [node for node in nodes if type(node).__name__ != "ControlMechanism"]
    if executable_nodes and all(type(node).__name__ in _STATELESS_MECHANISMS for node in executable_nodes):
        return STATELESS_GRAPH_FUSION
    if _is_ddm_graph_fusible(executable_nodes):
        return DDM_GRAPH_FUSION
    if _is_stateful_graph_fusible(executable_nodes):
        return STATEFUL_GRAPH_FUSION
    return None


def _is_ddm_graph_fusible(nodes) -> bool:
    ddm_count = sum(type(node).__name__ == "DDM" for node in nodes)
    return ddm_count == 1 and all(
        type(node).__name__ in _STATELESS_MECHANISMS or type(node).__name__ == "DDM"
        for node in nodes
    )


def _is_stateful_graph_fusible(nodes) -> bool:
    has_stateful_node = False
    for node in nodes:
        component_type = type(node).__name__
        if component_type in _STATELESS_MECHANISMS:
            continue
        if component_type == "LCAMechanism":
            has_stateful_node = True
            continue
        if component_type == "DDM":
            has_stateful_node = True
            continue
        return False
    return has_stateful_node


def _infer_stability_flexibility_roles(composition, nodes) -> dict[str, str]:
    ddm_nodes = [node for node in nodes if type(node).__name__ == "DDM"]
    lca_nodes = [node for node in nodes if type(node).__name__ == "LCAMechanism"]
    if len(ddm_nodes) != 1 or len(lca_nodes) != 1:
        return {}

    ddm = ddm_nodes[0]
    lca = lca_nodes[0]
    roles: dict[str, str] = {"ddm_node": _node_name(ddm), "lca_node": _node_name(lca)}
    lca_sources = _mapping_sources(lca)
    task_sources = [source for source in lca_sources if source is not lca]
    if task_sources:
        roles["task_node"] = _node_name(task_sources[0])
    cue = _control_monitor_source_for(composition, lca)
    if cue is not None:
        roles["cue_node"] = _node_name(cue)

    ddm_sources = _mapping_sources(ddm)
    if ddm_sources:
        scale = ddm_sources[0]
        roles["scale_node"] = _node_name(scale)
        recode_sources = _mapping_sources(scale)
        if recode_sources:
            recode = recode_sources[0]
            roles["recode_node"] = _node_name(recode)
            recode_input_sources = _mapping_sources(recode)
            product_sources = _source_names(recode_input_sources)
            upstream = _first_source_with_combine(recode_input_sources, "sum")
            if upstream is not None:
                roles["combination_node"] = _node_name(upstream)
                for source in recode_input_sources:
                    if source is not upstream:
                        roles["correct_node"] = _node_name(source)
                combo_sources = _mapping_sources(upstream)
                product_node = _first_source_with_combine(combo_sources, "product")
                if product_node is not None:
                    roles["nonautomatic_node"] = _node_name(product_node)
                    for source in combo_sources:
                        if source is not product_node:
                            roles["automaticity_node"] = _node_name(source)
                    for source in _mapping_sources(product_node):
                        if source is not lca:
                            roles["stimulus_node"] = _node_name(source)
            elif len(product_sources) == 2:
                roles["correct_node"] = product_sources[1]

    required = {"task_node", "stimulus_node", "cue_node", "correct_node", "automaticity_node", "scale_node"}
    return roles if required.issubset(roles) else {}


def _control_monitor_source_for(composition, controlled_node):
    deps = getattr(composition.graph_processing, "dependency_dict", {}).get(controlled_node, [])
    for dependency in deps:
        if type(dependency).__name__ != "ControlMechanism":
            continue
        for input_port in getattr(dependency, "input_ports", []):
            for projection in getattr(input_port, "path_afferents", []):
                sender = getattr(getattr(projection, "sender", None), "owner", None)
                if sender is not None:
                    return sender
    return None


def _mapping_sources(node):
    sources = []
    for input_port in getattr(node, "input_ports", []):
        for projection in getattr(input_port, "path_afferents", []):
            if type(projection).__name__ != "MappingProjection":
                continue
            sender = getattr(getattr(projection, "sender", None), "owner", None)
            if sender is not None:
                sources.append(sender)
    return sources


def _first_source_with_combine(sources, combine: str):
    for source in sources:
        if _combine_name(source) == combine:
            return source
    return None


def _source_names(sources) -> list[str]:
    return [_node_name(source) for source in sources]


def _stateless_param_name(node_name: str, param: str, roles: dict[str, str]) -> str:
    if roles.get("automaticity_node") == node_name and param == "slope":
        return "automaticity"
    if roles.get("scale_node") == node_name and param == "slope":
        return "scale"
    return f"{node_name}.{param}"


def _op_kind(node) -> str:
    component_type = type(node).__name__
    if component_type == "DDM":
        return "DDMIntegrateUntilFinished"
    if component_type == "LCAMechanism":
        return "LCAIntegrateUntilFinished"
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


def _scheduler_conditions(composition):
    scheduler = getattr(composition, "scheduler", None)
    if scheduler is None:
        return {}
    condition_set = getattr(scheduler, "conditions", None)
    conditions_basic = getattr(condition_set, "conditions_basic", {})
    if not hasattr(conditions_basic, "items"):
        return {}
    return conditions_basic


def _analyze_scheduler_conditions(composition) -> tuple[list[str], list[BatchedDiagnostic]]:
    rejected, supported = scheduler_condition_names(composition)
    return supported, rejected


def _composition_nodes(composition) -> list[Any]:
    return list(getattr(composition, "nodes", []))


def _find_node_by_name(nodes, name: str):
    for node in nodes:
        if _node_name(node) == name:
            return node
    return None


def _node_name(node) -> str:
    return getattr(node, "name", str(node))


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


def _primary_output_width(node) -> int:
    output_ports = getattr(node, "output_ports", [])
    if not output_ports:
        return 1
    if type(node).__name__ == "DDM":
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


def _get_param(component, name: str, default: float) -> float:
    if component is None:
        return float(default)

    parameters = getattr(component, "parameters", None)
    if parameters is not None and hasattr(parameters, name):
        parameter = getattr(parameters, name)
        for getter in ("get", "_get"):
            if hasattr(parameter, getter):
                try:
                    return _as_float(getattr(parameter, getter)(None))
                except Exception:
                    pass

    defaults = getattr(component, "defaults", None)
    if defaults is not None and hasattr(defaults, name):
        try:
            return _as_float(getattr(defaults, name))
        except Exception:
            pass

    if hasattr(component, name):
        try:
            return _as_float(getattr(component, name))
        except Exception:
            pass

    return float(default)


def _as_float(value) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) == 0:
        return 0.0
    return float(array[0])
