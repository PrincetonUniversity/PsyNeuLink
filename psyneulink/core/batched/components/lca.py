"""Batched op for the exact, deliberately narrow width-2 LCA subset.

The op implements canonical recurrent ``LCAMechanism`` semantics for the
validated deterministic configuration. Broader widths, nonzero/custom state
initialization, function-valued or non-broadcast noise, reset policies beyond
exact ``Never``/``AtTrialStart``, clipping, and scheduler behavior remain
fail-closed until represented.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.api import TritonOpCall, pnl_triton_op
from psyneulink.core.batched.condition_validation import is_canonical_condition
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.ir import FP32_EXACT_INTEGER_LIMIT
from psyneulink.core.batched.specs import (
    MechanismOpSpec,
    ParamBinding,
    StateDecl,
    register_batched_op,
    resolve_component_param,
)
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic
from psyneulink.core.scheduling.condition import AtTrialStart, Never
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
    initialized,
    initialize_noise_sender,
    active,
    gain,
    leak,
    competition,
    self_excitation,
    dt,
    noise,
    bias,
    x_0,
    scale,
    offset,
):
    # The single shared leaky-competing recurrence step (width 2).  Both the
    # run-to-completion integrate loop and scheduled step call this, so
    # the recurrence math lives in exactly one place.  `active` masks the update
    # (lanes that are past their step budget / whose terminator finished freeze);
    # Numeric LCI noise is deterministic: PNL adds the same
    # ``noise * sqrt(dt)`` term to each accumulator on every integration step.
    # Distribution/function-valued noise remains outside the accepted subset.
    # PNL also evaluates the integrator once while initializing the mechanism:
    # that call exposes Logistic(noise * sqrt(dt)) as the initial recurrent
    # sender value without advancing ``previous_value``.  Reconstruct that
    # parameter-lane-specific activity on the first real integration step.
    noise_step = noise * tl.sqrt(dt)
    initial_act = (
        scale / (1.0 + tl.exp(-gain * (noise_step + bias - x_0))) + offset
    )
    initialize_sender = (initialized == 0.0) & (initialize_noise_sender != 0.0)
    act0 = tl.where(initialize_sender, initial_act, act0)
    act1 = tl.where(initialize_sender, initial_act, act1)
    rec0 = self_excitation * act0 - competition * act1
    rec1 = -competition * act0 + self_excitation * act1
    pre0 = tl.where(
        active,
        pre0 + (input0 + rec0 - leak * pre0) * dt + noise_step,
        pre0,
    )
    pre1 = tl.where(
        active,
        pre1 + (input1 + rec1 - leak * pre1) * dt + noise_step,
        pre1,
    )
    act0 = tl.where(
        active,
        scale / (1.0 + tl.exp(-gain * (pre0 + bias - x_0))) + offset,
        act0,
    )
    act1 = tl.where(
        active,
        scale / (1.0 + tl.exp(-gain * (pre1 + bias - x_0))) + offset,
        act1,
    )
    initialized = tl.where(active, 1.0, initialized)
    return pre0, pre1, act0, act1, initialized


@pnl_triton_op(helpers=(_pnl_triton_lca_width2_recurrence,))
def _pnl_triton_lca_width2_integrate(
    input0,
    input1,
    pre0,
    pre1,
    act0,
    act1,
    initialized,
    initialize_noise_sender,
    gain,
    leak,
    competition,
    self_excitation,
    dt,
    noise,
    bias,
    x_0,
    scale,
    offset,
    lca_steps,
    lca_max_steps,
    lane_mask,
):
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
        pre0, pre1, act0, act1, initialized = _pnl_triton_lca_width2_recurrence(
            input0, input1, pre0, pre1, act0, act1, initialized,
            initialize_noise_sender, active,
            gain, leak, competition, self_excitation, dt,
            noise, bias, x_0, scale, offset,
        )
        step += 1
    return pre0, pre1, act0, act1, initialized


@pnl_triton_op(
    helpers=(_pnl_triton_lca_width2_recurrence,),
)
def _pnl_triton_lca_width2_step(
    input0,
    input1,
    pre0,
    pre1,
    act0,
    act1,
    initialized,
    initialize_noise_sender,
    finished,
    gain,
    leak,
    competition,
    self_excitation,
    dt,
    noise,
    bias,
    x_0,
    scale,
    offset,
):
    # One integration step for a dynamic schedule where the LCA advances
    # alongside a terminator.  Lanes whose terminator has finished freeze, so
    # persisted state carried to the next trial matches when the trial ended.
    #
    active = finished == 0.0
    pre0, pre1, act0, act1, initialized = _pnl_triton_lca_width2_recurrence(
        input0, input1, pre0, pre1, act0, act1, initialized,
        initialize_noise_sender, active,
        gain, leak, competition, self_excitation, dt,
        noise, bias, x_0, scale, offset,
    )
    return pre0, pre1, act0, act1, initialized


def _lca_step_emit(ctx, node_spec, inputs, outputs, step_var, finished_var):
    del step_var
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
    initialized = ctx.state(f"{node_spec.name}.initialized", 0)
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_lca_width2_step,
            outputs=(pre0, pre1, act0, act1, initialized),
            args=(
                inputs[0],
                inputs[1],
                pre0,
                pre1,
                act0,
                act1,
                initialized,
                "1.0" if node_spec.attrs["initialize_noise_sender"] else "0.0",
                finished_var,
                ctx.param(node_spec, "gain"),
                ctx.param(node_spec, "leak"),
                ctx.param(node_spec, "competition"),
                ctx.param(node_spec, "self_excitation"),
                ctx.param(node_spec, "time_step_size"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "bias"),
                ctx.param(node_spec, "x_0"),
                ctx.param(node_spec, "scale"),
                ctx.param(node_spec, "offset"),
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
    reset_condition = getattr(node, "reset_stateful_function_when", None)
    if (
        type(reset_condition) not in {Never, AtTrialStart}
        or not is_canonical_condition(reset_condition)
    ):
        return BatchedDiagnostic(
            name,
            "unsupported LCA reset policy for batched v2",
            type(reset_condition).__name__,
        )
    if _raw_parameter(node, "clip", None) is not None:
        return BatchedDiagnostic(name, "unsupported LCA clip for batched v2")
    integrator = getattr(node, "integrator_function", None)
    initial_value_parameter = getattr(
        getattr(node, "parameters", None),
        "initial_value",
        None,
    )
    node_initial_value_matches = _parameter_matches(node, "initial_value", 0.0)
    if (
        not node_initial_value_matches
        and bool(getattr(initial_value_parameter, "_user_specified", False))
    ):
        return BatchedDiagnostic(name, "unsupported LCA initial_value for batched v2", "requires zero")
    if not _parameter_matches(integrator, "initializer", 0.0):
        return BatchedDiagnostic(
            name,
            "unsupported LCA integrator initializer for batched v2",
            "requires zero",
        )
    if not node_initial_value_matches:
        return BatchedDiagnostic(name, "unsupported LCA initial_value for batched v2", "requires zero")
    if not _parameter_matches(integrator, "offset", 0.0):
        return BatchedDiagnostic(
            name,
            "unsupported LCA integrator offset for batched v2",
            "requires zero",
        )
    noise_values = []
    for noise_owner in (node, integrator):
        noise_value = _finite_broadcast_scalar_parameter(noise_owner, "noise")
        if noise_value is None:
            return BatchedDiagnostic(
                name,
                "unsupported LCA noise for batched v2",
                "requires a finite float32 scalar or broadcast-scalar numeric value",
            )
        noise_values.append(noise_value)
    if noise_values[0] != noise_values[1]:
        return BatchedDiagnostic(
            name,
            "unsupported LCA noise for batched v2",
            "mechanism and integrator noise values must agree",
        )

    function = getattr(node, "function", None)
    logistic_defaults = {
        "gain": 1.0,
        "bias": 0.0,
        "x_0": 0.0,
        "scale": 1.0,
        "offset": 0.0,
    }
    for parameter_name, default in logistic_defaults.items():
        value = np.asarray(_raw_parameter(function, parameter_name, default))
        if value.size == 0 or value.dtype.kind not in "biuf":
            return BatchedDiagnostic(
                name,
                "unsupported non-scalar LCA Logistic parameter for batched v2",
                parameter_name,
            )
        scalar = value.reshape(-1)[0]
        if not np.isfinite(scalar) or not np.all(value == scalar):
            return BatchedDiagnostic(
                name,
                "unsupported non-scalar LCA Logistic parameter for batched v2",
                parameter_name,
            )
        if abs(scalar) > np.finfo(np.float32).max:
            return BatchedDiagnostic(
                name,
                "unsupported out-of-range LCA Logistic parameter for batched v2",
                f"{parameter_name} is not representable as float32",
            )

    scalar_names = ("leak", "competition", "self_excitation", "time_step_size")
    values = {}
    for parameter_name in scalar_names:
        value = np.asarray(_raw_parameter(node, parameter_name, 0.0))
        if (
            value.size == 0
            or value.dtype.kind not in "biuf"
            or not np.isfinite(value.reshape(-1)[0])
            or not np.all(value == value.reshape(-1)[0])
        ):
            return BatchedDiagnostic(
                name,
                "unsupported non-scalar LCA parameter for batched v2",
                parameter_name,
            )
        if abs(value.reshape(-1)[0]) > np.finfo(np.float32).max:
            return BatchedDiagnostic(
                name,
                "unsupported out-of-range LCA parameter for batched v2",
                f"{parameter_name} is not representable as float32",
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
    if (
        matrix.shape != (2, 2)
        or matrix.dtype.kind not in "biuf"
        or not np.all(np.isfinite(matrix))
        or not np.array_equal(matrix, expected_matrix)
    ):
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
    max_executions = np.asarray(
        _raw_parameter(node, "max_executions_before_finished", np.iinfo(np.int64).max)
    )
    if (
        max_executions.size != 1
        or max_executions.dtype.kind not in "iuf"
        or not np.isfinite(max_executions.reshape(-1)[0])
        or max_executions.reshape(-1)[0] < 1
        or max_executions.reshape(-1)[0] != int(max_executions.reshape(-1)[0])
    ):
        return BatchedDiagnostic(
            name,
            "unsupported LCA maximum execution count for batched v2",
            "requires a positive integer",
        )
    threshold = _raw_parameter(node, "termination_threshold", None)
    threshold_value = np.asarray(threshold)
    if (
        threshold is None
        or threshold_value.size != 1
        or threshold_value.dtype.kind not in "biuf"
        or not np.isfinite(threshold_value.reshape(-1)[0])
        or threshold_value.reshape(-1)[0] < 0
    ):
        return BatchedDiagnostic(
            name,
            "unsupported LCA termination_threshold for batched v2",
            "requires a finite nonnegative scalar",
        )
    static_steps = min(
        max(1, int(np.ceil(threshold_value.reshape(-1)[0]))),
        int(max_executions.reshape(-1)[0]),
    )
    if static_steps > FP32_EXACT_INTEGER_LIMIT:
        return BatchedDiagnostic(
            name,
            "unsupported LCA termination step count for batched v2",
            f"requires no more than {FP32_EXACT_INTEGER_LIMIT} executions",
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
        return bool(
            array.dtype.kind in "biuf"
            and np.all(np.isfinite(array))
            and np.all(array == expected)
        )
    except Exception:
        return False


def _parameter_matches(component, name, expected) -> bool:
    if not _numeric_array_matches(_raw_parameter(component, name, expected), expected):
        return False
    defaults = getattr(component, "defaults", None)
    default_value = getattr(defaults, name, expected) if defaults is not None else expected
    return _numeric_array_matches(default_value, expected)


def _finite_broadcast_scalar_parameter(component, name) -> float | None:
    """Resolve an exactly broadcast numeric parameter as one fp32-safe scalar."""

    try:
        value = np.asarray(_raw_parameter(component, name, None))
        if value.size == 0 or value.dtype.kind not in "biuf":
            return None
        scalar = value.reshape(-1)[0]
        if (
            not np.isfinite(scalar)
            or not np.all(value == scalar)
            or abs(scalar) > np.finfo(np.float32).max
        ):
            return None
        return float(scalar)
    except Exception:
        return None


def _lca_extract_attrs(node, composition) -> dict:
    termination_input_node = _control_monitor_source_for(composition, node)
    threshold = _raw_parameter(node, "termination_threshold", 1200)
    max_executions = int(
        np.asarray(
            _raw_parameter(
                node,
                "max_executions_before_finished",
                np.iinfo(np.int64).max,
            )
        ).reshape(-1)[0]
    )
    return {
        "termination_input_node": (
            None
            if termination_input_node is None
            else getattr(termination_input_node, "name", str(termination_input_node))
        ),
        "termination_threshold": threshold,
        "termination_steps": (
            None
            if termination_input_node is not None
            else min(max(1, int(np.ceil(float(threshold)))), max_executions)
        ),
        "max_executions_before_finished": max_executions,
        # Construction initializes a Never-reset LCA's recurrent sender from
        # Logistic(noise * sqrt(dt)) without advancing previous_value.  An
        # AtTrialStart reset replaces that sender with Logistic(initializer),
        # including before trial zero, so it must not replay construction.
        "initialize_noise_sender": type(
            getattr(node, "reset_stateful_function_when", None)
        ) is Never,
    }


def _lca_finished_after_execution_count(node, composition) -> int | None:
    """Return the exact scheduler-call count for a fixed LCA finished value.

    A stepwise LCA becomes finished after ``ceil(termination_threshold)``
    calls.  PsyNeuLink resets the per-call execution cap on every one-step
    ``execute``, so ``max_executions_before_finished`` does not reduce this
    scheduler count.  A run-to-completion LCA keeps its ordinary atomic
    ``CallMechanism`` lowering and needs no counted scheduler predicate.

    A controlled threshold is lane/runtime dependent and intentionally remains
    dynamic until conditional scheduler execution is represented in KernelIR.
    """

    if bool(_raw_parameter(node, "execute_until_finished", True)):
        return None
    if _control_monitor_source_for(composition, node) is not None:
        return None
    threshold = np.asarray(_raw_parameter(node, "termination_threshold", None))
    if threshold.size != 1:
        return None
    return max(1, int(np.ceil(float(threshold.reshape(-1)[0]))))


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
    initialized = ctx.state(f"{node_spec.name}.initialized", 0)
    termination_node = node_spec.attrs.get("termination_input_node")
    if termination_node:
        cue_value = ctx.raw_input_value(termination_node)
    else:
        # Resolve the discrete static step demand on the host.  Sending a value
        # such as 1.00000001 through an fp32 literal before ``ceil`` would
        # silently change PsyNeuLink's two executions into one.
        # Keep the literal floating-point because compiled Triton (unlike its
        # interpreter) rejects an integer operand to ``tl.ceil``.
        cue_value = f"{int(node_spec.attrs.get('termination_steps', 1))}.0"
    steps_var = f"{ctx.component_symbol(node_spec)}_lca_steps"
    node_step_cap = min(
        int(
            node_spec.attrs.get(
                "max_executions_before_finished",
                FP32_EXACT_INTEGER_LIMIT,
            )
        ),
        FP32_EXACT_INTEGER_LIMIT,
    )
    ctx.line(
        f"{steps_var} = tl.minimum(tl.maximum(tl.ceil({cue_value}), 1.0), "
        f"tl.minimum({node_step_cap}, LCA_MAX_STEPS))"
    )
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_lca_width2_integrate,
            outputs=(pre0, pre1, act0, act1, initialized),
            args=(
                inputs[0],
                inputs[1],
                pre0,
                pre1,
                act0,
                act1,
                initialized,
                "1.0" if node_spec.attrs["initialize_noise_sender"] else "0.0",
                ctx.param(node_spec, "gain"),
                ctx.param(node_spec, "leak"),
                ctx.param(node_spec, "competition"),
                ctx.param(node_spec, "self_excitation"),
                ctx.param(node_spec, "time_step_size"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "bias"),
                ctx.param(node_spec, "x_0"),
                ctx.param(node_spec, "scale"),
                ctx.param(node_spec, "offset"),
                steps_var,
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
            control = getattr(getattr(control_projection, "sender", None), "owner", None)
            signal = getattr(control_projection, "sender", None)
            if (
                type(control).__name__ != "ControlMechanism"
                or id(control) not in active_node_ids
                or getattr(signal, "owner", None) is not control
                or control_projection not in tuple(getattr(signal, "efferents", ()))
            ):
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
            ParamBinding(arg="bias", pnl_name="bias", default=0.0, scope="function"),
            ParamBinding(arg="x_0", pnl_name="x_0", default=0.0, scope="function"),
            ParamBinding(arg="scale", pnl_name="scale", default=1.0, scope="function"),
            ParamBinding(arg="offset", pnl_name="offset", default=0.0, scope="function"),
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
            ParamBinding(
                arg="time_step_size",
                pnl_name="time_step_size",
                default=0.01,
                scope="mechanism",
                minimum=0.0,
                minimum_inclusive=False,
            ),
            ParamBinding(
                arg="noise",
                pnl_name="noise",
                default=0.0,
                scope="mechanism",
                get=lambda node: _finite_broadcast_scalar_parameter(node, "noise"),
            ),
        ),
        states=(
            StateDecl("pre", width=None, initial=0.0),
            StateDecl(
                "act",
                width=None,
                initial=0.0,
                initialize_with_function=True,
            ),
            StateDecl("initialized", width=1, initial=0.0),
        ),
        rng=(),
        outputs=None,
        supports=_lca_supports,
        extract_attrs=_lca_extract_attrs,
        triton_emit=_lca_triton_emit,
        step_emit=_lca_step_emit,
        finished_after_execution_count=_lca_finished_after_execution_count,
    )
)
