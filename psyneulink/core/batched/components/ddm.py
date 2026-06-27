"""Batched op for `DDM` with a `DriftDiffusionIntegrator` function.

The reference declarative mechanism op: the decorated function is the single
kernel body (run compiled on GPU, interpreted on CPU).  Its arguments are
auto-bound from the body signature against the DriftDiffusionIntegrator
Parameters; the RNG stream is implied by the ``seed``/``rng_base`` arguments.

Collapsing boundary: a control mechanism may OVERRIDE the DDM ``threshold`` with
the output of a stateful integrating transfer (a SimpleIntegrator with a nonzero
``offset``), giving a per-step boundary ``threshold(step) = base + collapse*step``.
``threshold_override_collapse`` recognizes that chain from the DDM node;
``ddm_threshold_collapse`` binds the per-step ``collapse`` into the kernel (0 for
an ordinary fixed-threshold DDM).
"""

import psyneulink as pnl

from psyneulink.core.batched.backend.triton.api import TritonOpCall, pnl_triton_op
from psyneulink.core.batched.specs import (
    StateDecl,
    batched_op,
    param,
    resolve_component_param,
)
from psyneulink.core.components.functions.stateful.integratorfunctions import (
    DriftDiffusionIntegrator,
)
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM


def _monitored_source(control_mechanism):
    for input_port in getattr(control_mechanism, "input_ports", []):
        for projection in getattr(input_port, "path_afferents", []):
            sender = getattr(getattr(projection, "sender", None), "owner", None)
            if sender is not None:
                return sender
    return None


def _control_function_affine(function):
    """(slope, intercept) for a control mechanism's function, or None.

    Identity passes its input through (1, 0); Linear contributes its own
    slope/intercept.  Anything else is unsupported here.
    """

    name = type(function).__name__
    if name == "Identity":
        return (1.0, 0.0)
    if name == "Linear":
        return (
            resolve_component_param(function, "slope", 1.0),
            resolve_component_param(function, "intercept", 0.0),
        )
    return None


def threshold_override_collapse(ddm_node):
    """Return ``(threshold_source_name, collapse_per_step)`` when ``ddm_node``'s
    ``threshold`` is OVERRIDDEN by a control mechanism monitoring a
    SimpleIntegrator transfer (a collapsing boundary), else ``None``.

    The effective boundary is ``base + collapse*step`` where, for a control
    function ``slope*z + intercept`` over the source ``base_src + offset*step``,
    ``base = slope*base_src + intercept`` and ``collapse = slope*offset``.  Only
    supported when that ``base`` matches the DDM's own ``threshold`` (the kernel
    uses the DDM threshold as the boundary's starting value).
    """

    try:
        port = ddm_node.parameter_ports["threshold"]
    except Exception:
        return None
    for projection in getattr(port, "mod_afferents", []):
        controller = getattr(getattr(projection, "sender", None), "owner", None)
        if controller is None or type(controller).__name__ != "ControlMechanism":
            continue
        if getattr(controller, "modulation", None) != pnl.OVERRIDE:
            continue
        source = _monitored_source(controller)
        integrator = getattr(source, "integrator_function", None)
        if source is None or type(integrator).__name__ != "SimpleIntegrator":
            continue
        affine = _control_function_affine(getattr(controller, "function", None))
        if affine is None:
            continue
        slope, intercept = affine
        base_src = resolve_component_param(getattr(source, "function", None), "intercept", 0.0)
        offset = resolve_component_param(integrator, "offset", 0.0)
        base = slope * base_src + intercept
        ddm_threshold = resolve_component_param(getattr(ddm_node, "function", None), "threshold", 0.0)
        if abs(base - ddm_threshold) > 1.0e-9:
            continue
        return (getattr(source, "name", str(source)), slope * offset)
    return None


def ddm_threshold_collapse(ddm_node) -> float:
    """Per-step threshold collapse bound into the kernel (0.0 if no chain)."""

    chain = threshold_override_collapse(ddm_node)
    return 0.0 if chain is None else chain[1]


@pnl_triton_op
def _pnl_triton_ddm_step(
    value,
    steps,
    finished,
    drift,
    rate,
    noise,
    threshold,
    threshold_collapse,
    time_step_size,
    offset,
    seed,
    rng_base,
    step,
):
    # One DDM integration step for the fused co-evolution loop: accumulate the
    # (possibly time-varying) drift toward a (possibly collapsing) boundary, and
    # raise `finished` for lanes that have crossed it.  Finished lanes freeze.
    sqrt_dt = tl.sqrt(time_step_size)
    thr = threshold + threshold_collapse * step
    boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)
    active = (finished == 0.0) & (tl.abs(value) + boundary_tolerance < thr)
    draw = tl.randn(seed, rng_base + step)
    updated = value + rate * drift * time_step_size + noise * sqrt_dt * draw
    updated = tl.minimum(tl.maximum(updated + offset, -thr), thr)
    value = tl.where(active, updated, value)
    steps = tl.where(active, steps + 1.0, steps)
    finished = tl.where(tl.abs(value) + boundary_tolerance >= thr, 1.0, finished)
    return value, steps, finished


def _ddm_step_emit(ctx, node_spec, inputs, outputs, step_var, finished_var):
    name = node_spec.name
    value = ctx.state(f"{name}.value", 0)
    steps = ctx.state(f"{name}.steps", 0)
    finished = ctx.state(f"{name}.finished", 0)
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_ddm_step,
            outputs=(value, steps, finished),
            args=(
                value,
                steps,
                finished,
                inputs[0],
                ctx.param(node_spec, "rate"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "threshold"),
                ctx.param(node_spec, "threshold_collapse"),
                ctx.param(node_spec, "time_step_size"),
                ctx.param(node_spec, "offset"),
                ctx.seed,
                ctx.rng_base(name),
                step_var,
            ),
        )
    )


def _ddm_readout_emit(ctx, node_spec, output_vars):
    # After the step loop, turn the accumulated state into the modeled outputs:
    # DECISION_OUTCOME = sign(value); RESPONSE_TIME = non_decision_time + steps*dt.
    name = node_spec.name
    value = ctx.state(f"{name}.value", 0)
    steps = ctx.state(f"{name}.steps", 0)
    decision, response_time = output_vars
    ctx.line(f"{decision} = tl.where({value} > 0.0, 1.0, 0.0)")
    ctx.line(
        f"{response_time} = {ctx.param(node_spec, 'non_decision_time')} "
        f"+ {steps} * {ctx.param(node_spec, 'time_step_size')}"
    )


@batched_op(
    DDM,
    function=DriftDiffusionIntegrator,
    outputs=(("DECISION_OUTCOME", 1), ("RESPONSE_TIME", 1)),
    bind={
        "starting_value": param("initializer", fallback="starting_value", default=0.0),
        "threshold_collapse": param(get=ddm_threshold_collapse),
    },
    constexpr=("max_steps",),
    single_node_model_kind="ddm",
    param_alias_prefixes=("ddm", "DDM"),
    diagnostics=("truncated",),
    step_emit=_ddm_step_emit,
    readout_emit=_ddm_readout_emit,
    trial_states=(
        StateDecl("value", width=1, initial=0.0),
        StateDecl("steps", width=1, initial=0.0),
        StateDecl("finished", width=1, initial=0.0),
    ),
    finished_output="finished",
)
def ddm_integrate(
    x,
    rate,
    noise,
    threshold,
    threshold_collapse,
    non_decision_time,
    time_step_size,
    starting_value,
    offset,
    seed,
    rng_base,
    max_steps,
):
    value = starting_value
    steps = tl.zeros_like(x)
    sqrt_dt = tl.sqrt(time_step_size)
    # Boundary may collapse per step: threshold(step) = threshold + collapse*step
    # (collapse is 0 for an ordinary fixed-threshold DDM).
    boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)
    thr = threshold
    for step in tl.range(0, max_steps, 1, loop_unroll_factor=1):
        thr = threshold + threshold_collapse * step
        active = tl.abs(value) + boundary_tolerance < thr
        draw = tl.randn(seed, rng_base + step)
        updated = value + rate * x * time_step_size + noise * sqrt_dt * draw
        updated = tl.minimum(tl.maximum(updated + offset, -thr), thr)
        value = tl.where(active, updated, value)
        steps += tl.where(active, 1.0, 0.0)
    truncated = tl.where(tl.abs(value) + boundary_tolerance < thr, 1.0, 0.0)
    return tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * time_step_size, truncated
