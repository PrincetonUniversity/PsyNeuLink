"""Scheduled Logistic LCAs with dense recurrence, Gaussian noise and readouts."""

from dataclasses import replace
from functools import lru_cache
import inspect

import numpy as np

from psyneulink.core.batched.components import lca
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.specs import (
    LikelihoodEffectContract, OutputDecl, ParamBinding, RngDecl, StateDecl,
    lookup_spec, register_batched_specializer, spec_key,
)
from psyneulink.core.components.functions.nonstateful.distributionfunctions import NormalDist
from psyneulink.core.components.functions.nonstateful.objectivefunctions import Energy
from psyneulink.core.globals.keywords import RESULT, ENERGY, OWNER_VALUE
from psyneulink.core.scheduling.condition import AtTrialStart, Never
from psyneulink.core.scheduling.time import TimeScale
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import (
    LCAMechanism, DECISION_INDEX, DECISION_TIME, DECISION_STEPS,
)


def _noise(node):
    value = lca._raw_parameter(node, "noise")
    return getattr(value, "__self__", value)


def _constructed_activity(node):
    # PNL initializes the RESULT port separately from the mechanism value.
    # Recurrent projections (and PEC's simulation copies) use that port value.
    # Read its construction default, never the mutable, last-executed activity.
    return np.asarray(node.output_port.defaults.value).reshape(-1)


def _supports(node):
    diagnostic = lca._lca_supports(node, extended=True)
    if diagnostic is not None:
        return diagnostic

    def reject(detail):
        return BatchedDiagnostic(node.name, "unsupported scheduled LCA semantics", detail)

    width = lca._primary_output_width(node)
    if not 1 <= width <= 32 or len(node.input_ports) != 1 or node.input_port.value.size != width:
        return reject("requires one input and RESULT of equal width, from 1 through 32")
    if bool(lca._raw_parameter(node, "execute_until_finished", True)):
        return reject("requires execute_until_finished=False")
    matrix = np.asarray(lca._raw_parameter(node, "matrix"))
    if matrix.shape != (width, width) or matrix.dtype.kind not in "biuf" or not np.all(np.isfinite(matrix)):
        return reject("requires a finite square recurrent matrix")
    if np.any(np.abs(matrix) > np.finfo(np.float32).max):
        return reject("recurrent weights must be representable as float32")
    measure = lca._raw_parameter(node, "termination_measure")
    if measure is not max and measure != TimeScale.TRIAL:
        return reject("requires max activity or TimeScale.TRIAL termination")
    noise = _noise(node)
    if type(noise) is NormalDist:
        for name in ("mean", "standard_deviation"):
            value = lca._finite_broadcast_scalar_parameter(noise, name)
            if value is None or (name == "standard_deviation" and value < 0):
                return reject("requires finite scalar NormalDist mean and nonnegative standard deviation")
        if type(node.reset_stateful_function_when) is Never:
            initial = _constructed_activity(node)
            if (initial.shape != (width,) or initial.dtype.kind not in "biuf"
                    or not np.all(np.isfinite(initial))
                    or np.any(np.abs(initial) > np.finfo(np.float32).max)):
                return reject("requires finite float32 construction-time RESULT activity")
        integrator_noise = _noise(node.integrator_function)
        if type(integrator_noise) is not NormalDist or any(
            lca._finite_broadcast_scalar_parameter(integrator_noise, name)
            != lca._finite_broadcast_scalar_parameter(noise, name)
            for name in ("mean", "standard_deviation")
        ):
            return reject("mechanism and integrator noise must agree")
    elif lca._finite_broadcast_scalar_parameter(node, "noise") is None:
        return reject("requires numeric broadcast noise or NormalDist")
    elif lca._finite_broadcast_scalar_parameter(node, "noise") != lca._finite_broadcast_scalar_parameter(node.integrator_function, "noise"):
        return reject("mechanism and integrator noise must agree")
    return None


def _outputs_supported(node):
    from psyneulink.core.batched.graph import _is_identity_linear, _user_defined_callable

    ports = tuple(node.output_ports)
    if not ports or ports[0].name != RESULT:
        return BatchedDiagnostic(node.name, "unsupported scheduled LCA outputs", "RESULT must be first")
    for port in ports:
        valid = port.name in {RESULT, ENERGY, DECISION_INDEX, DECISION_TIME, DECISION_STEPS}
        standard = node.standard_output_ports.get_port_dict(port.name) if valid else {}
        if valid:
            selector = standard.get("variable", (OWNER_VALUE, 0))
            valid = repr(port._variable_spec) == repr(selector)
        if valid and port.name in {RESULT, DECISION_STEPS}:
            valid = _is_identity_linear(port.function)
        elif valid and port.name == ENERGY:
            valid = (
                type(port.function) is Energy
                and np.array_equal(lca._raw_parameter(port.function, "matrix"), lca._raw_parameter(node, "matrix"))
                and not lca._raw_parameter(port.function, "normalize", False)
                and lca._raw_parameter(port.function, "transfer_fct", None) is None
            )
        elif valid:
            expected = standard.get("function")
            valid = _user_defined_callable(port.function) is (expected if inspect.isfunction(expected)
                                                            else _user_defined_callable(expected))
        expected_width = lca._primary_output_width(node) if port.name == RESULT else 1
        if not valid or port.value.size != expected_width:
            return BatchedDiagnostic(node.name, "unsupported scheduled LCA outputs", f"noncanonical {port.name}")
    return None


def _attrs(node, composition):
    width = lca._primary_output_width(node)
    matrix = np.asarray(lca._raw_parameter(node, "matrix"), dtype=float)
    competition = lca._finite_broadcast_scalar_parameter(node, "competition")
    excitation = lca._finite_broadcast_scalar_parameter(node, "self_excitation")
    canonical = np.full((width, width), -competition)
    np.fill_diagonal(canonical, excitation)
    attrs = {
        "scheduled_lca": True,
        "recurrent_matrix": tuple(tuple(float(v) for v in row) for row in matrix),
        "canonical_recurrence": np.array_equal(matrix, canonical),
        "activity_termination": lca._raw_parameter(node, "termination_measure") is max,
        "gaussian_noise": type(_noise(node)) is NormalDist,
        "initialize_noise_sender": type(node.reset_stateful_function_when) is not AtTrialStart,
        "max_executions_before_finished": int(lca._raw_parameter(node, "max_executions_before_finished")),
    }
    if attrs["gaussian_noise"] and attrs["initialize_noise_sender"]:
        attrs["constructed_activity"] = tuple(float(value) for value in _constructed_activity(node))
    return attrs


def _parameter(ctx, node, name):
    return ctx.sampled_effective_parameter(node, name) or ctx.param(node, name)


def _readouts(ctx, node):
    width = node.output_width
    act = [ctx.state(f"{node.name}.act", i) for i in range(width)]
    count = ctx.state(f"{node.name}.count", 0)
    result = []
    for port, _ in node.attrs["op_outputs"]:
        if port == RESULT:
            result.extend(act)
        elif port == DECISION_STEPS:
            result.append(count)
        elif port == DECISION_TIME:
            result.append(f"({count} * {_parameter(ctx, node, 'time_step_size')})")
        elif port == DECISION_INDEX:
            # NumPy argmax chooses the first index on ties.
            winner = "0.0"
            best = act[0]
            for i in range(1, width):
                winner = f"tl.where({act[i]} > {best}, {float(i)}, {winner})"
                best = f"tl.maximum({best}, {act[i]})"
            result.append(winner)
        elif port == ENERGY:
            terms = [f"({act[i]} * {float(w)} * {act[j]})"
                     for i, row in enumerate(node.attrs["recurrent_matrix"])
                     for j, w in enumerate(row) if w and i != j]
            result.append(f"(-0.5 * ({' + '.join(terms) or '0.0'}))")
    return tuple(result)


def _step(ctx, node, inputs, outputs, step_var, finished_var):
    width = node.output_width
    stem = ctx.component_symbol(node)
    pre = [ctx.state(f"{node.name}.pre", i) for i in range(width)]
    act = [ctx.state(f"{node.name}.act", i) for i in range(width)]
    initialized = ctx.state(f"{node.name}.initialized", 0)
    count = ctx.state(f"{node.name}.count", 0)
    finished = ctx.state(f"{node.name}.finished", 0)
    params = {name: _parameter(ctx, node, name) for name in node.params}
    dt, leak = params["time_step_size"], params["leak"]

    def logistic(x):
        return (f"({params['scale']} / (1.0 + tl.exp(-{params['gain']} * "
                f"({x} + {params['bias']} - {params['x_0']}))) + {params['offset']})")

    # Persistent Gaussian LCAs start from the frozen construction-time RESULT
    # port, already loaded by InitializeState. Do not redraw it per estimate
    # or trial, or transform it again when runtime parameters change.
    if node.attrs["initialize_noise_sender"] and not node.attrs["gaussian_noise"]:
        initial = logistic(f"{params['noise']} * tl.sqrt({dt})")
        for value in act:
            ctx.line(f"{value} = tl.where({initialized} == 0.0, {initial}, {value})")
    # Compute every recurrent input before overwriting any sender activity.
    for j in range(width):
        if node.attrs["canonical_recurrence"]:
            terms = [f"{act[i]} * {params['self_excitation']}" if i == j
                     else f"(-{act[i]} * {params['competition']})" for i in range(width)]
        else:
            terms = [f"{act[i]} * {float(node.attrs['recurrent_matrix'][i][j])}" for i in range(width)]
        ctx.line(f"{stem}_rec_{j} = {' + '.join(terms)}")
    for j in range(width):
        if node.attrs["gaussian_noise"]:
            offset = ctx.rng_stream_offset(node.name, j) - ctx.rng_stream_offset(node.name, 0)
            draw = f"tl.randn({ctx.seed}, {ctx.rng_base(node.name)} + {offset} + {step_var})"
            noise = f"({params['noise_mean']} + {params['noise_standard_deviation']} * {draw})"
        else:
            noise = params["noise"]
        ctx.line(f"{pre[j]} = tl.where({finished_var} == 0.0, {pre[j]} + "
                 f"({inputs[j]} + {stem}_rec_{j} - {leak} * {pre[j]}) * {dt} + "
                 f"{noise} * tl.sqrt({dt}), {pre[j]})")
        ctx.line(f"{act[j]} = tl.where({finished_var} == 0.0, {logistic(pre[j])}, {act[j]})")
    ctx.line(f"{initialized} = tl.where({finished_var} == 0.0, 1.0, {initialized})")
    ctx.line(f"{count} = tl.where({finished_var} == 0.0, tl.where({finished} != 0.0, 1.0, {count} + 1.0), {count})")
    if node.attrs["activity_termination"]:
        measure = act[0]
        for value in act[1:]:
            measure = f"tl.maximum({measure}, {value})"
    else:
        measure = f"({step_var} + 1.0)"
    ctx.line(f"{finished} = tl.where({finished_var} == 0.0, "
             f"(({measure} >= {params['termination_threshold']}) | "
             f"({count} >= {node.attrs['max_executions_before_finished']})), {finished})")
    return _readouts(ctx, node)


def _readout(ctx, node, outputs):
    for output, value in zip(outputs, _readouts(ctx, node)):
        ctx.line(f"{output} = {value}")


def _atomic(ctx, node, inputs, outputs):
    # The supported execution mode performs exactly one integration per call.
    return _step(ctx, node, inputs, outputs, "0", "0.0")


@lru_cache(None)
def _specialized(base, width, ports, gaussian, activity, persistent_gaussian):
    params = base.params
    if gaussian:
        params = tuple(p for p in params if p.arg != "noise") + tuple(
            ParamBinding(f"noise_{name}", get=lambda node, name=name: lca._finite_broadcast_scalar_parameter(_noise(node), name),
                         minimum=0.0 if name == "standard_deviation" else None)
            for name in ("mean", "standard_deviation")
        )
    params += (ParamBinding("termination_threshold", scope="mechanism", minimum=0.0),)
    states = tuple(
        replace(state, initialize_with_function=False, initial_attribute="constructed_activity")
        if persistent_gaussian and state.name == "act" else state
        for state in base.states
    )
    return replace(
        base, key=f"{base.key}:scheduled:{width}:{ports!r}:{gaussian}:{activity}:{persistent_gaussian}",
        params=params,
        states=states,
        outputs=tuple(OutputDecl(port, width if port == RESULT else 1) for port in ports),
        trial_states=(StateDecl("count", width=1), StateDecl("finished", width=1)),
        rng=(RngDecl("rng", width=width),) if gaussian else (),
        supports=_supports, extract_attrs=_attrs, validate_outputs=_outputs_supported,
        triton_emit=_atomic, step_emit=_step, readout_emit=_readout,
        finished_output="finished" if activity else "",
        continue_after_finished=True,
        finished_after_execution_count=None if activity else lca._lca_finished_after_execution_count,
        likelihood_contract=LikelihoodEffectContract(randomness="declared_streams" if gaussian else "none"),
    )


def _resolve(node):
    if bool(lca._raw_parameter(node, "execute_until_finished", True)):
        return None
    width = lca._primary_output_width(node)
    ports = tuple(port.name for port in node.output_ports)
    gaussian = type(_noise(node)) is NormalDist
    activity = lca._raw_parameter(node, "termination_measure") is max
    # Keep the established CSI implementation and its authenticated contract.
    if width == 2 and ports == (RESULT,) and not gaussian and not activity:
        return None
    persistent_gaussian = gaussian and type(node.reset_stateful_function_when) is Never
    return _specialized(lookup_spec(spec_key(LCAMechanism)), width, ports, gaussian, activity, persistent_gaussian)


register_batched_specializer(LCAMechanism, _resolve)
