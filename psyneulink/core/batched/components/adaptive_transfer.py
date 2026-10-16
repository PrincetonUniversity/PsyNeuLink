"""Retained AdaptiveIntegrator state with a Logistic transfer readout."""

from functools import lru_cache

from psyneulink.core.batched.components.lca import _raw_parameter, _finite_broadcast_scalar_parameter
from psyneulink.core.batched.condition_validation import is_canonical_condition
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.specs import (
    LikelihoodEffectContract, MechanismOpSpec, OutputDecl, ParamBinding, StateDecl,
    spec_key,
)
from psyneulink.core.components.functions.stateful.integratorfunctions import AdaptiveIntegrator
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.scheduling.condition import AtTrialStart, Never


def _supports(node):
    def reject(detail):
        return BatchedDiagnostic(node.name, "unsupported adaptive transfer semantics", detail)

    if len(node.input_ports) != 1 or node.input_port.value.size != 1:
        return reject("requires one scalar input")
    if _raw_parameter(node, "clip") is not None or _finite_broadcast_scalar_parameter(node, "noise") != 0:
        return reject("requires zero noise and no clipping")
    integrator = node.integrator_function
    if _finite_broadcast_scalar_parameter(integrator, "noise") != 0:
        return reject("requires zero integrator noise")
    for name in ("rate", "offset", "initializer"):
        value = _finite_broadcast_scalar_parameter(integrator, name)
        if value is None or (name == "rate" and not 0 <= value <= 1):
            return reject(f"invalid scalar {name}")
    reset = node.reset_stateful_function_when
    if type(reset) not in {Never, AtTrialStart} or not is_canonical_condition(reset):
        return reject("requires Never or AtTrialStart reset")
    if _raw_parameter(node, "execute_until_finished") and _raw_parameter(node, "termination_threshold") is not None:
        return reject("requires one integration update per execution")


def _outputs_supported(node):
    from psyneulink.core.batched.graph import _is_identity_linear, _owner_value_selector_index
    if any(_owner_value_selector_index(port) != 0 or port.value.size != 1
           or not _is_identity_linear(port.function) for port in node.output_ports):
        return BatchedDiagnostic(node.name, "unsupported adaptive transfer output", "requires identity scalar readouts")


def _attrs(node, composition):
    return {"integrator_initial": (_finite_broadcast_scalar_parameter(node.integrator_function, "initializer"),)}


def _step(ctx, node, inputs, outputs, step_var, finished_var):
    state = ctx.state(f"{node.name}.previous_value", 0)
    rate, offset = (ctx.param(node, name) for name in ("integration_rate", "integrator_offset"))
    ctx.line(f"{state} = tl.where({finished_var} == 0.0, "
             f"(1.0 - {rate}) * {state} + {rate} * {inputs[0]} + {offset}, {state})")
    gain, bias, x_0, scale, offset = (ctx.param(node, p) for p in ("gain", "bias", "x_0", "scale", "offset"))
    value = f"({scale} / (1.0 + tl.exp(-{gain} * ({state} + {bias} - {x_0}))) + {offset})"
    return (value,) * len(outputs)


def _atomic(ctx, node, inputs, outputs):
    return _step(ctx, node, inputs, outputs, "0", "0.0")


@lru_cache(None)
def _specialized(ports):
    return MechanismOpSpec(
        TransferMechanism, Logistic, display_name="Adaptive Logistic transfer",
        key=f"{spec_key(TransferMechanism)}:adaptive_logistic:{ports!r}",
        params=(
            ParamBinding("integration_rate", scope="mechanism", minimum=0.0, maximum=1.0,
                         get=lambda n: _finite_broadcast_scalar_parameter(n.integrator_function, "rate")),
            ParamBinding("integrator_offset", scope="mechanism", pnl_name="offset",
                         get=lambda n: _finite_broadcast_scalar_parameter(n.integrator_function, "offset")),
        ) + tuple(ParamBinding(name, default=1.0 if name in {"gain", "scale"} else 0.0)
                  for name in ("gain", "bias", "x_0", "scale", "offset")),
        states=(StateDecl("previous_value", width=1, initial_attribute="integrator_initial"),),
        outputs=tuple(OutputDecl(name) for name in ports),
        supports=_supports, validate_outputs=_outputs_supported, extract_attrs=_attrs,
        triton_emit=_atomic, step_emit=_step, likelihood_contract=LikelihoodEffectContract(),
    )


def resolve(node):
    if (type(getattr(node, "integrator_function", None)) is AdaptiveIntegrator
            and type(node.function) is Logistic and _raw_parameter(node, "integrator_mode", False)
            and len(node.input_ports) == 1 and node.input_port.value.size == 1):
        return _specialized(tuple(port.name for port in node.output_ports))
    return None
