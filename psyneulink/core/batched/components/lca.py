"""Batched op for width-2 `LCAMechanism`.

This is a narrow, performance-oriented lowering for width-2 LCA-like
stateful graphs, not a full implementation of PsyNeuLink ``LCAMechanism``
semantics (see BATCH_COMPILE_WIP.md, "LCA Caveats").  Because its state, RNG,
and termination handling are still custom, it uses the ``triton_emit`` escape
hatch instead of the declarative body form.  The single kernel body runs
compiled on the GPU and interpreted on the CPU; there is no separate numpy
implementation.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.api import TritonOpCall, pnl_triton_op
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.specs import (
    MechanismOpSpec,
    ParamBinding,
    RngDecl,
    StateDecl,
    register_batched_op,
    resolve_component_param,
)
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import (
    LCAMechanism,
)
from psyneulink.core.scheduling.time import TimeScale


@pnl_triton_op
def _pnl_triton_lca_width2_recurrence(
    input0,
    input1,
    pre0,
    pre1,
    act0,
    act1,
    active,
    gain,
    leak,
    competition,
    self_excitation,
    noise,
    dt,
    n0,
    n1,
):
    # The single shared leaky-competing recurrence step (width 2).  Both the
    # run-to-completion integrate loop and the co-evolution step call this, so
    # the recurrence math lives in exactly one place.  `active` masks the update
    # (lanes that are past their step budget / whose terminator finished freeze);
    # `n0`/`n1` are the caller-drawn noise samples.
    sqrt_dt = tl.sqrt(dt)
    rec0 = self_excitation * act0 - competition * act1
    rec1 = -competition * act0 + self_excitation * act1
    pre0 = tl.where(active, pre0 + (input0 + rec0 - leak * pre0) * dt + noise * sqrt_dt * n0, pre0)
    pre1 = tl.where(active, pre1 + (input1 + rec1 - leak * pre1) * dt + noise * sqrt_dt * n1, pre1)
    act0 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre0)), act0)
    act1 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre1)), act1)
    return pre0, pre1, act0, act1


@pnl_triton_op(
    constexpr=("stream0", "stream1", "lca_max_steps"),
    helpers=(_pnl_triton_lca_width2_recurrence,),
)
def _pnl_triton_lca_width2_integrate(
    input0,
    input1,
    pre0,
    pre1,
    act0,
    act1,
    gain,
    leak,
    competition,
    self_excitation,
    noise,
    dt,
    lca_steps,
    seed,
    random_base,
    stream0,
    stream1,
    lca_max_steps,
    lane_mask,
):
    # `stream0`/`stream1` are absolute Philox offsets from the lane's base, one
    # full counter space each, so the step index just adds into the low bits and
    # the draws do not depend on `lca_max_steps` (which is only the loop cap).
    #
    # `lca_max_steps` is the cap over *every* trial (the largest cue anywhere in
    # the data), while `lca_steps` is what this trial actually demands -- so
    # masking through to the cap makes every trial pay the worst one.
    #
    # Unlike a DDM, this settling length is known up front (it is cue-driven, not
    # data-dependent), so the stopping point is computed once rather than tested
    # each iteration: a per-step block reduction here costs more than it saves
    # whenever the cues happen to be uniform.  Out-of-range lanes are excluded --
    # when the cue is a constant rather than an input they look active and would
    # hold the bound at the cap.
    block_steps = tl.minimum(tl.max(tl.where(lane_mask, lca_steps, 0.0)), lca_max_steps)
    step = 0
    while step < block_steps:
        active = step < lca_steps
        n0 = tl.randn(seed, random_base + stream0 + step)
        n1 = tl.randn(seed, random_base + stream1 + step)
        pre0, pre1, act0, act1 = _pnl_triton_lca_width2_recurrence(
            input0, input1, pre0, pre1, act0, act1, active,
            gain, leak, competition, self_excitation, noise, dt, n0, n1,
        )
        step += 1
    return pre0, pre1, act0, act1


@pnl_triton_op(
    constexpr=("stream0", "stream1"),
    helpers=(_pnl_triton_lca_width2_recurrence,),
)
def _pnl_triton_lca_width2_step(
    input0,
    input1,
    pre0,
    pre1,
    act0,
    act1,
    finished,
    gain,
    leak,
    competition,
    self_excitation,
    noise,
    dt,
    seed,
    random_base,
    step,
    stream0,
    stream1,
):
    # One integration step for the fused co-evolution loop where the LCA steps
    # alongside a terminator.  Lanes whose terminator has finished freeze, so the
    # persisted state carried to the next trial matches when the trial ended.
    #
    # A co-evolving LCA steps up to MAX_STEPS times, not LCA_MAX_STEPS.  Its
    # stream owns a full counter space, so that is fine -- when the stride was
    # LCA_MAX_STEPS this path could run off the end of its own stream and into
    # the next one.
    active = finished == 0.0
    n0 = tl.randn(seed, random_base + stream0 + step)
    n1 = tl.randn(seed, random_base + stream1 + step)
    pre0, pre1, act0, act1 = _pnl_triton_lca_width2_recurrence(
        input0, input1, pre0, pre1, act0, act1, active,
        gain, leak, competition, self_excitation, noise, dt, n0, n1,
    )
    return pre0, pre1, act0, act1


def _lca_step_emit(ctx, node_spec, inputs, outputs, step_var, finished_var):
    if node_spec.output_width != 2:
        raise ValueError(
            "Triton batched LCA step supports width 2, "
            f"got {node_spec.output_width} for '{node_spec.name}'."
        )
    pre_state = f"{node_spec.name}.pre"
    act_state = f"{node_spec.name}.act"
    pre0 = ctx.state(pre_state, 0)
    pre1 = ctx.state(pre_state, 1)
    act0 = ctx.state(act_state, 0)
    act1 = ctx.state(act_state, 1)
    stream0 = ctx.rng_stream_offset(node_spec.name, 0)
    stream1 = ctx.rng_stream_offset(node_spec.name, 1)
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_lca_width2_step,
            outputs=(pre0, pre1, act0, act1),
            args=(
                inputs[0],
                inputs[1],
                pre0,
                pre1,
                act0,
                act1,
                finished_var,
                ctx.param(node_spec, "gain"),
                ctx.param(node_spec, "leak"),
                ctx.param(node_spec, "competition"),
                ctx.param(node_spec, "self_excitation"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "time_step_size"),
                ctx.seed,
                "random_base",
                step_var,
                str(stream0),
                str(stream1),
            ),
        )
    )
    return (act0, act1)


def _lca_supports(node) -> BatchedDiagnostic | None:
    name = getattr(node, "name", str(node))
    width = _primary_output_width(node)
    if width != 2:
        return BatchedDiagnostic(
            name,
            "unsupported LCA width for batched v2",
            f"width={width}",
        )
    if type(getattr(node, "integrator_function", None)).__name__ != "LeakyCompetingIntegrator":
        return BatchedDiagnostic(name, "unsupported LCA integrator for batched v2")
    if not bool(_raw_parameter(node, "integrator_mode", True)):
        return BatchedDiagnostic(name, "unsupported LCA integrator_mode for batched v2", "False")
    if type(getattr(node, "reset_stateful_function_when", None)).__name__ != "Never":
        return BatchedDiagnostic(
            name,
            "unsupported LCA reset policy for batched v2",
            type(getattr(node, "reset_stateful_function_when", None)).__name__,
        )
    if _raw_parameter(node, "clip", None) is not None:
        return BatchedDiagnostic(name, "unsupported LCA clip for batched v2")
    if not _numeric_array_matches(_raw_parameter(node, "initial_value", 0.0), 0.0):
        return BatchedDiagnostic(name, "unsupported LCA initial_value for batched v2", "requires zero")

    # The current LCA op has its own Logistic implementation and binds only
    # ``gain``.  Full Logistic support for passthrough TransferMechanisms must
    # not make these other behavior-affecting parameters look supported here.
    function = getattr(node, "function", None)
    logistic_defaults = {
        "gain": 1.0,
        "bias": 0.0,
        "x_0": 0.0,
        "scale": 1.0,
        "offset": 0.0,
    }
    for parameter_name, expected in logistic_defaults.items():
        value = np.asarray(_raw_parameter(function, parameter_name, expected))
        if (
            value.size == 0
            or value.dtype.kind not in "biufc"
            or not np.allclose(value, value.reshape(-1)[0])
        ):
            return BatchedDiagnostic(
                name,
                "unsupported non-scalar LCA Logistic parameter for batched v2",
                parameter_name,
            )
        if parameter_name != "gain" and not np.allclose(value, expected):
            return BatchedDiagnostic(
                name,
                "unsupported LCA Logistic parameter for batched v2",
                f"{parameter_name}={value.reshape(-1)[0]!r} (requires {expected!r})",
            )

    scalar_names = ("leak", "competition", "self_excitation", "noise", "time_step_size")
    values = {}
    for parameter_name in scalar_names:
        value = np.asarray(_raw_parameter(node, parameter_name, 0.0))
        if (
            value.size == 0
            or value.dtype.kind not in "biufc"
            or not np.allclose(value, value.reshape(-1)[0])
        ):
            return BatchedDiagnostic(
                name,
                "unsupported non-scalar LCA parameter for batched v2",
                parameter_name,
            )
        values[parameter_name] = float(value.reshape(-1)[0])
    if values["time_step_size"] <= 0:
        return BatchedDiagnostic(name, "unsupported LCA time_step_size for batched v2", "must be > 0")

    expected_matrix = np.array(
        [
            [values["self_excitation"], -values["competition"]],
            [-values["competition"], values["self_excitation"]],
        ]
    )
    matrix = np.asarray(_raw_parameter(node, "matrix", expected_matrix))
    if matrix.shape != (2, 2) or not np.allclose(matrix, expected_matrix):
        return BatchedDiagnostic(
            name,
            "unsupported LCA recurrent matrix for batched v2",
            "requires canonical self-excitation/competition matrix",
        )

    if _raw_parameter(node, "termination_measure", None) != TimeScale.TRIAL:
        return BatchedDiagnostic(
            name,
            "unsupported LCA termination measure for batched v2",
            "requires TimeScale.TRIAL step-count semantics",
        )
    if _raw_parameter(node, "termination_comparison_op", ">=") != ">=":
        return BatchedDiagnostic(
            name,
            "unsupported LCA termination comparison for batched v2",
            "requires >=",
        )
    threshold = _raw_parameter(node, "termination_threshold", None)
    try:
        termination_port = node.parameter_ports["termination_threshold"]
    except Exception:
        termination_port = None
    controlled = bool(getattr(termination_port, "mod_afferents", ()))
    if not controlled:
        threshold_value = np.asarray(threshold)
        if (
            threshold is None
            or threshold_value.size != 1
            or threshold_value.dtype.kind not in "biufc"
            or not np.isfinite(threshold_value.reshape(-1)[0])
            or threshold_value.reshape(-1)[0] < 0
        ):
            return BatchedDiagnostic(
                name,
                "unsupported LCA termination_threshold for batched v2",
                "requires a finite nonnegative scalar or supported OVERRIDE control",
            )
    return None


def _raw_parameter(component, name, default=None):
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
    return getattr(defaults, name, default) if defaults is not None else default


def _numeric_array_matches(value, expected) -> bool:
    try:
        array = np.asarray(value)
        return array.dtype.kind in "biufc" and bool(np.allclose(array, expected))
    except Exception:
        return False


def _lca_extract_attrs(node, composition) -> dict:
    termination_input_node = _control_monitor_source_for(composition, node)
    return {
        "termination_input_node": (
            None
            if termination_input_node is None
            else getattr(termination_input_node, "name", str(termination_input_node))
        ),
        "termination_threshold": resolve_component_param(node, "termination_threshold", 1200),
    }


def _lca_triton_emit(ctx, node_spec, inputs, outputs):
    if node_spec.output_width != 2:
        raise ValueError(
            "Triton batched LCA op supports width 2, "
            f"got {node_spec.output_width} for '{node_spec.name}'."
        )

    pre_state = f"{node_spec.name}.pre"
    act_state = f"{node_spec.name}.act"
    pre0 = ctx.state(pre_state, 0)
    pre1 = ctx.state(pre_state, 1)
    act0 = ctx.state(act_state, 0)
    act1 = ctx.state(act_state, 1)
    termination_node = node_spec.attrs.get("termination_input_node")
    if termination_node:
        cue_value = ctx.raw_input_value(termination_node)
    else:
        cue_value = ctx.float_literal(node_spec.attrs.get("termination_threshold", 1.0))
    steps_var = f"{ctx.component_symbol(node_spec)}_lca_steps"
    stream0 = ctx.rng_stream_offset(node_spec.name, 0)
    stream1 = ctx.rng_stream_offset(node_spec.name, 1)

    ctx.line(
        f"{steps_var} = tl.minimum(tl.maximum(tl.ceil({cue_value}), 0.0), "
        "LCA_MAX_STEPS)"
    )
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_lca_width2_integrate,
            outputs=(pre0, pre1, act0, act1),
            args=(
                inputs[0],
                inputs[1],
                pre0,
                pre1,
                act0,
                act1,
                ctx.param(node_spec, "gain"),
                ctx.param(node_spec, "leak"),
                ctx.param(node_spec, "competition"),
                ctx.param(node_spec, "self_excitation"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "time_step_size"),
                steps_var,
                ctx.seed,
                "random_base",
                str(stream0),
                str(stream1),
                ctx.lca_max_steps,
                "mask",
            ),
        )
    )
    return (act0, act1)


def _control_monitor_source_for(composition, controlled_node):
    active_node_ids = {id(node) for node in getattr(composition, "nodes", ())}
    active_projection_ids = {
        id(projection) for projection in getattr(composition, "projections", ())
    }
    for parameter_port in getattr(controlled_node, "parameter_ports", ()):
        for control_projection in getattr(parameter_port, "mod_afferents", ()):
            if id(control_projection) not in active_projection_ids:
                continue
            control = getattr(getattr(control_projection, "sender", None), "owner", None)
            if type(control).__name__ != "ControlMechanism" or id(control) not in active_node_ids:
                continue
            for input_port in getattr(control, "input_ports", ()):
                for monitor_projection in getattr(input_port, "path_afferents", ()):
                    if id(monitor_projection) not in active_projection_ids:
                        continue
                    source = getattr(
                        getattr(monitor_projection, "sender", None),
                        "owner",
                        None,
                    )
                    if source is not None and id(source) in active_node_ids:
                        return source
    return None


def _primary_output_width(node) -> int:
    output_ports = getattr(node, "output_ports", [])
    if not output_ports:
        return 1
    try:
        return int(np.asarray(output_ports[0].value).reshape(-1).size)
    except Exception:
        return 1


register_batched_op(
    MechanismOpSpec(
        mechanism_class=LCAMechanism,
        function_class=Logistic,
        display_name="LCA",
        params=(
            ParamBinding(arg="gain", pnl_name="gain", default=1.0, scope="function"),
            ParamBinding(
                arg="leak",
                get=lambda node: resolve_component_param(
                    node,
                    "leak",
                    resolve_component_param(
                        getattr(node, "integrator_function", None), "rate", 1.0
                    ),
                ),
            ),
            ParamBinding(arg="competition", pnl_name="competition", default=1.0, scope="mechanism"),
            ParamBinding(
                arg="self_excitation",
                get=lambda node: resolve_component_param(
                    node, "self_excitation", resolve_component_param(node, "auto", 0.0)
                ),
            ),
            ParamBinding(arg="noise", pnl_name="noise", default=0.0, scope="mechanism"),
            ParamBinding(
                arg="time_step_size", pnl_name="time_step_size", default=0.01, scope="mechanism"
            ),
        ),
        states=(StateDecl("pre", width=None, initial=0.0), StateDecl("act", width=None, initial=0.0)),
        rng=(RngDecl(name="lca", step_extent="LCA_MAX_STEPS", width=None),),
        outputs=None,
        supports=_lca_supports,
        extract_attrs=_lca_extract_attrs,
        triton_emit=_lca_triton_emit,
        step_emit=_lca_step_emit,
    )
)
