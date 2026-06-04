from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from psyneulink.core.batched.diagnostics import BatchedCapabilityReport, BatchedDiagnostic
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedParamSpec


DDM_MODEL = "ddm"
STABILITY_FLEXIBILITY_MODEL = "stability_flexibility"

_SUPPORTED_SCHEDULER_CONDITIONS = {
    "Always",
    "AtTrialStart",
    "EveryNCalls",
    "WhenFinished",
}

_STABILITY_FLEXIBILITY_REQUIRED_NAMES = (
    "Task Input [I1, I2]",
    "Stimulus Input [S1, S2]",
    "Cue-Stimulus Interval",
    "Correct Response Info",
    "Task Activations [Act1, Act2]",
    "Automaticity-weighted Stimulus Input [w*S1, w*S2]",
    "DDM",
)


def analyze_composition(composition, backend: str = "reference", outputs=None, max_steps: int | None = None):
    nodes = _composition_nodes(composition)
    rejected_nodes: list[BatchedDiagnostic] = []
    supported_nodes: list[str] = []

    ddm_nodes = [node for node in nodes if _is_ddm_node(node)]
    lca_nodes = [node for node in nodes if _is_lca_node(node)]

    model_kind = None
    if len(ddm_nodes) == 1 and _is_drift_diffusion_integrator(getattr(ddm_nodes[0], "function", None)):
        if _looks_like_stability_flexibility(nodes):
            model_kind = STABILITY_FLEXIBILITY_MODEL
        elif len(lca_nodes) == 0:
            model_kind = DDM_MODEL

    for node in nodes:
        node_name = _node_name(node)
        if model_kind == STABILITY_FLEXIBILITY_MODEL and _is_stability_flexibility_supported_node(node):
            supported_nodes.append(node_name)
        elif model_kind == DDM_MODEL and node in ddm_nodes:
            supported_nodes.append(node_name)
        elif _is_control_mechanism(node):
            supported_nodes.append(node_name)
        else:
            rejected_nodes.append(
                BatchedDiagnostic(
                    component=node_name,
                    reason="unsupported node for batched v1",
                    detail=type(node).__name__,
                )
            )

    rejected_conditions, supported_conditions = _analyze_scheduler_conditions(composition)
    backend_available, backend_messages = _backend_availability(backend)

    if model_kind is None and not rejected_nodes:
        rejected_nodes.append(
            BatchedDiagnostic(
                component=getattr(composition, "name", type(composition).__name__),
                reason="unsupported composition topology",
                detail="expected one integrator-mode DDM or the stability-flexibility topology",
            )
        )

    report = BatchedCapabilityReport(
        backend=backend,
        model_kind=model_kind,
        supported_nodes=tuple(supported_nodes),
        rejected_nodes=tuple(rejected_nodes),
        supported_conditions=tuple(supported_conditions),
        rejected_conditions=tuple(rejected_conditions),
        messages=tuple(backend_messages),
        backend_available=backend_available,
        metadata={"num_nodes": len(nodes)},
    )

    ir = None
    if report.is_supported:
        ir = _build_ir(composition, model_kind, outputs, max_steps=max_steps)

    return report, ir


def _build_ir(composition, model_kind: str, outputs=None, max_steps: int | None = None) -> BatchedCompositionIR:
    nodes = _composition_nodes(composition)
    ddm = next(node for node in nodes if _is_ddm_node(node))
    ddm_params = _extract_ddm_params(ddm)
    if model_kind == DDM_MODEL:
        params = (
            BatchedParamSpec("rate", ddm_params["rate"], aliases=("ddm.rate", "DDM.rate")),
            BatchedParamSpec("noise", ddm_params["noise"], aliases=("ddm.noise", "DDM.noise")),
            BatchedParamSpec("threshold", ddm_params["threshold"], aliases=("ddm.threshold", "DDM.threshold")),
            BatchedParamSpec(
                "non_decision_time",
                ddm_params["non_decision_time"],
                aliases=("ddm.non_decision_time", "DDM.non_decision_time"),
            ),
            BatchedParamSpec(
                "time_step_size",
                ddm_params["time_step_size"],
                aliases=("ddm.time_step_size", "DDM.time_step_size"),
            ),
            BatchedParamSpec(
                "starting_value",
                ddm_params["starting_value"],
                aliases=("ddm.starting_value", "DDM.starting_value"),
            ),
            BatchedParamSpec("offset", ddm_params["offset"], aliases=("ddm.offset", "DDM.offset")),
        )
        metadata = {"composition_name": getattr(composition, "name", None)}
    else:
        sf_params = _extract_stability_flexibility_params(nodes, ddm_params)
        params = tuple(
            BatchedParamSpec(name, value, aliases=(f"stability_flexibility.{name}",))
            for name, value in sf_params.items()
        )
        lca = _find_node_by_prefix(nodes, "Task Activations [Act1, Act2]")
        metadata = {
            "composition_name": getattr(composition, "name", None),
            "lca_max_steps": int(np.ceil(_get_param(lca, "termination_threshold", 1200))),
        }

    output_names = tuple(outputs) if outputs is not None else ("decision", "response_time")
    return BatchedCompositionIR(
        model_kind=model_kind,
        node_names=tuple(_node_name(node) for node in nodes),
        params=params,
        output_names=output_names,
        max_steps=256 if max_steps is None else int(max_steps),
        metadata=metadata,
    )


def _extract_ddm_params(ddm) -> dict[str, float]:
    function = ddm.function
    return {
        "rate": _get_param(function, "rate", 1.0),
        "noise": _get_param(function, "noise", 0.0),
        "threshold": _get_param(function, "threshold", 1.0),
        "non_decision_time": _get_param(function, "non_decision_time", 0.0),
        "time_step_size": _get_param(function, "time_step_size", 1.0),
        "starting_value": _get_param(function, "initializer", _get_param(function, "starting_value", 0.0)),
        "offset": _get_param(function, "offset", 0.0),
    }


def _extract_stability_flexibility_params(nodes, ddm_params: dict[str, float]) -> dict[str, float]:
    lca = _find_node_by_prefix(nodes, "Task Activations [Act1, Act2]")
    automaticity = _find_node_by_prefix(nodes, "Automaticity-weighted Stimulus Input [w*S1, w*S2]")
    scale = _find_node_by_prefix(nodes, "Scaled DDM Input")
    return {
        "gain": _get_param(getattr(lca, "function", None), "gain", 1.0),
        "leak": _get_param(lca, "leak", _get_param(getattr(lca, "integrator_function", None), "rate", 1.0)),
        "competition": _get_param(lca, "competition", 1.0),
        "self_excitation": _get_param(lca, "self_excitation", _get_param(lca, "auto", 0.0)),
        "lca_noise": _get_param(lca, "noise", 0.0),
        "lca_time_step_size": _get_param(lca, "time_step_size", 0.01),
        "automaticity": _get_param(getattr(automaticity, "function", None), "slope", 0.0),
        "scale": _get_param(getattr(scale, "function", None), "slope", 1.0),
        "starting_value": ddm_params["starting_value"],
        "threshold": ddm_params["threshold"],
        "ddm_noise": ddm_params["noise"],
        "ddm_time_step_size": ddm_params["time_step_size"],
        "non_decision_time": ddm_params["non_decision_time"],
        "ddm_offset": ddm_params["offset"],
    }


def _backend_availability(backend: str) -> tuple[bool, list[str]]:
    if backend == "reference":
        return True, []
    if backend != "triton":
        return False, [f"Unknown batched backend '{backend}'."]

    if importlib.util.find_spec("triton") is None:
        return False, ["Triton is not installed; install psyneulink[triton] to execute this backend."]
    if importlib.util.find_spec("torch") is None:
        return False, ["Torch is not installed; Triton execution uses torch tensors for launch buffers."]

    return True, []


def _analyze_scheduler_conditions(composition) -> tuple[list[BatchedDiagnostic], list[str]]:
    scheduler = getattr(composition, "scheduler", None)
    if scheduler is None:
        return [], []

    condition_set = getattr(scheduler, "conditions", None)
    conditions_basic = getattr(condition_set, "conditions_basic", {})
    if not hasattr(conditions_basic, "items"):
        return [], []

    rejected: list[BatchedDiagnostic] = []
    supported: list[str] = []
    for node, condition in conditions_basic.items():
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


def _looks_like_stability_flexibility(nodes: list[Any]) -> bool:
    node_names = tuple(_node_name(node) for node in nodes)
    return all(any(name.startswith(required) for name in node_names) for required in _STABILITY_FLEXIBILITY_REQUIRED_NAMES)


def _is_stability_flexibility_supported_node(node) -> bool:
    name = _node_name(node)
    if any(name.startswith(required) for required in _STABILITY_FLEXIBILITY_REQUIRED_NAMES):
        return True
    if name.startswith("Non-Automatic Component"):
        return True
    if name.startswith("Drift ="):
        return True
    if name.startswith("Recoded Drift"):
        return True
    if name.startswith("Scaled DDM Input"):
        return True
    if name.startswith("DECISION_GATE") or name.startswith("RESPONSE_GATE"):
        return True
    return _is_control_mechanism(node)


def _composition_nodes(composition) -> list[Any]:
    return list(getattr(composition, "nodes", []))


def _find_node_by_prefix(nodes: list[Any], prefix: str):
    for node in nodes:
        if _node_name(node).startswith(prefix):
            return node
    return None


def _node_name(node) -> str:
    return getattr(node, "name", str(node))


def _is_ddm_node(node) -> bool:
    return type(node).__name__ == "DDM"


def _is_lca_node(node) -> bool:
    return type(node).__name__ == "LCAMechanism"


def _is_control_mechanism(node) -> bool:
    return type(node).__name__ == "ControlMechanism"


def _is_drift_diffusion_integrator(function) -> bool:
    return type(function).__name__ == "DriftDiffusionIntegrator"


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
