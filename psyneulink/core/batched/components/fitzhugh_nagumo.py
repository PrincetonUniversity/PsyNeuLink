"""Scalar Euler FitzHugh–Nagumo integration inside a TransferMechanism."""

from functools import lru_cache

from psyneulink.core.batched.backend.triton.api import TritonOpCall, pnl_triton_op
from psyneulink.core.batched.components.lca import _raw_parameter, _finite_broadcast_scalar_parameter
from psyneulink.core.batched.condition_validation import is_canonical_condition
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.specs import (
    LikelihoodEffectContract, MechanismOpSpec, OutputDecl, ParamBinding, StateDecl,
    register_batched_specializer, spec_key,
)
from psyneulink.core.components.functions.stateful.integratorfunctions import FitzHughNagumoIntegrator
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.scheduling.condition import AtTrialStart, Never
from psyneulink.core.scheduling.time import TimeScale


_PARAMETERS = ("a_v", "b_v", "c_v", "d_v", "e_v", "f_v", "threshold", "time_constant_v",
               "a_w", "b_w", "c_w", "mode", "uncorrelated_activity", "time_constant_w", "time_step_size")


@pnl_triton_op(constexpr=("steps",))
def _euler(x, v, w, t, active, a_v, b_v, c_v, d_v, e_v, f_v, threshold, time_constant_v,
           a_w, b_w, c_w, mode, uncorrelated_activity, time_constant_w, time_step_size, steps):
    for _ in range(steps):
        dv = (a_v * v * v * v + (1.0 + threshold) * b_v * v * v
              - threshold * c_v * v + d_v + e_v * w + f_v * x) / time_constant_v
        dw = (mode * a_w * v + b_w * w + c_w + (1.0 - mode) * uncorrelated_activity) / time_constant_w
        v = tl.where(active, v + time_step_size * dv, v)
        w = tl.where(active, w + time_step_size * dw, w)
        t = tl.where(active, t + time_step_size, t)
    return v, w, t


def _supports(node):
    def reject(detail):
        return BatchedDiagnostic(node.name, "unsupported FitzHughNagumo transfer semantics", detail)

    if type(node.function) is not Linear:
        return reject("requires a Linear transfer function")
    if not _raw_parameter(node, "integrator_mode", False) or node.input_port.value.size != 1 or len(node.input_ports) != 1:
        return reject("requires scalar integrating transfer")
    integrator = node.integrator_function
    if _raw_parameter(integrator, "integration_method") != "EULER":
        return reject("requires EULER integration")
    if _raw_parameter(node, "clip") is not None or _finite_broadcast_scalar_parameter(node, "noise") != 0:
        return reject("requires no clipping and zero transfer noise")
    reset = node.reset_stateful_function_when
    if type(reset) not in {Never, AtTrialStart} or not is_canonical_condition(reset):
        return reject("requires Never or AtTrialStart reset")
    for name in _PARAMETERS:
        value = _finite_broadcast_scalar_parameter(integrator, name)
        if value is None or (name in {"time_step_size", "time_constant_v", "time_constant_w"} and value <= 0):
            return reject(f"invalid scalar {name}")
    for name in ("initial_v", "initial_w", "t_0"):
        if _finite_broadcast_scalar_parameter(integrator, name) != 0:
            return reject("requires zero state initializers")
    if _raw_parameter(node, "termination_comparison_op") != ">=":
        return reject("requires >= termination")
    maximum = _finite_broadcast_scalar_parameter(node, "max_executions_before_finished")
    if maximum is None or maximum < 1 or maximum != int(maximum):
        return reject("requires a positive integer execution limit")
    if _raw_parameter(node, "execute_until_finished", True):
        count = _finite_broadcast_scalar_parameter(node, "termination_threshold")
        if _raw_parameter(node, "termination_measure") != TimeScale.PASS or count is None or not 1 <= count <= 1024 or count != int(count):
            return reject("atomic integration requires a positive integer PASS count <= 1024")
    return None


def _outputs_supported(node):
    from psyneulink.core.batched.graph import _is_identity_linear, _owner_value_selector_index
    if any(_owner_value_selector_index(port) not in {0, 1, 2}
           or port.value.size != 1 or not _is_identity_linear(port.function) for port in node.output_ports):
        return BatchedDiagnostic(node.name, "unsupported FitzHughNagumo output", "requires identity v, w or time slices")


def _attrs(node, composition):
    from psyneulink.core.batched.graph import _owner_value_selector_index
    steps = int(_raw_parameter(node, "termination_threshold")) if _raw_parameter(node, "execute_until_finished") else 1
    steps = min(steps, int(_raw_parameter(node, "max_executions_before_finished")))
    return {"integration_steps": steps, "output_indices": tuple(_owner_value_selector_index(p) for p in node.output_ports)}


def _step(ctx, node, inputs, outputs, step_var, finished_var):
    states = tuple(ctx.state(f"{node.name}.{name}", 0) for name in ("v", "w", "time"))
    ctx.emit_call(TritonOpCall(
        _euler, states, (inputs[0], *states, f"({finished_var} == 0.0)",
                        *(ctx.param(node, name) for name in _PARAMETERS), str(node.attrs["integration_steps"])),
    ))
    slope, intercept, scale, offset = (ctx.param(node, name) for name in ("slope", "intercept", "scale", "offset"))
    return tuple(f"({scale} * ({slope} * {states[index]} + {intercept}) + {offset})"
                 for index in node.attrs["output_indices"])


def _atomic(ctx, node, inputs, outputs):
    return _step(ctx, node, inputs, outputs, "0", "0.0")


@lru_cache(None)
def _specialized(ports):
    return MechanismOpSpec(
        TransferMechanism, Linear, display_name="FitzHughNagumo transfer",
        key=f"{spec_key(TransferMechanism)}:fhn_euler:{ports!r}",
        params=tuple(ParamBinding(name, scope="mechanism", get=lambda n, name=name:
                                _finite_broadcast_scalar_parameter(n.integrator_function, name),
                                minimum=0.0 if name in {"time_step_size", "time_constant_v", "time_constant_w"} else None,
                                minimum_inclusive=False)
                     for name in _PARAMETERS)
        + tuple(ParamBinding(name, default=1.0 if name in {"slope", "scale"} else 0.0)
                for name in ("slope", "intercept", "scale", "offset")),
        states=tuple(StateDecl(name, width=1) for name in ("v", "w", "time")),
        outputs=tuple(OutputDecl(name) for name in ports),
        supports=_supports, extract_attrs=_attrs, validate_outputs=_outputs_supported,
        triton_emit=_atomic, step_emit=_step, likelihood_contract=LikelihoodEffectContract(),
    )


def _resolve(node):
    if type(getattr(node, "integrator_function", None)) is FitzHughNagumoIntegrator:
        return _specialized(tuple(port.name for port in node.output_ports))
    return None


register_batched_specializer(TransferMechanism, _resolve)
