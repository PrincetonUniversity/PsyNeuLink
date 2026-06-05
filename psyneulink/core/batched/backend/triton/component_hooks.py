from __future__ import annotations

from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.backend.triton.api import (
    TritonOpCall,
    pnl_triton_op,
)


@pnl_triton_op
def _pnl_triton_linear(x, slope, intercept):
    return slope * x + intercept


@pnl_triton_op
def _pnl_triton_logistic(x, gain):
    return 1.0 / (1.0 + tl.exp(-gain * x))


@pnl_triton_op
def _pnl_triton_projection_term(x, coefficient):
    return x * coefficient


@pnl_triton_op(constexpr=("max_steps",))
def _pnl_triton_ddm_integrate(
    drift,
    rate,
    noise,
    threshold,
    non_decision_time,
    dt,
    starting_value,
    offset,
    seed,
    random_base,
    max_steps,
):
    value = starting_value
    steps = tl.zeros_like(drift)
    sqrt_dt = tl.sqrt(dt)
    boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)
    for step in tl.range(0, max_steps, 1, loop_unroll_factor=1):
        active = tl.abs(value) + boundary_tolerance < threshold
        draw = tl.randn(seed, random_base + step)
        updated = value + rate * drift * dt + noise * sqrt_dt * draw
        updated = tl.minimum(tl.maximum(updated + offset, -threshold), threshold)
        value = tl.where(active, updated, value)
        steps += tl.where(active, 1.0, 0.0)
    return tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * dt


@pnl_triton_op(constexpr=("stream0", "stream1", "lca_max_steps"))
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
):
    sqrt_dt = tl.sqrt(dt)
    for step in tl.range(0, lca_max_steps, 1, loop_unroll_factor=1):
        active = step < lca_steps
        rec0 = self_excitation * act0 - competition * act1
        rec1 = -competition * act0 + self_excitation * act1
        n0 = tl.randn(seed, random_base + stream0 * lca_max_steps + step)
        n1 = tl.randn(seed, random_base + stream1 * lca_max_steps + step)
        upd0 = (input0 + rec0 - leak * pre0) * dt + noise * sqrt_dt * n0
        upd1 = (input1 + rec1 - leak * pre1) * dt + noise * sqrt_dt * n1
        pre0 = tl.where(active, pre0 + upd0, pre0)
        pre1 = tl.where(active, pre1 + upd1, pre1)
        act0 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre0)), act0)
        act1 = tl.where(active, 1.0 / (1.0 + tl.exp(-gain * pre1)), act1)
    return pre0, pre1, act0, act1


_HOOKS_INSTALLED = False


def ensure_triton_hooks_installed() -> None:
    """Install private Triton batched hooks on supported component classes."""

    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return

    from psyneulink.core.components.functions.nonstateful.transferfunctions import (
        Linear,
        Logistic,
    )
    from psyneulink.core.components.projections.pathway.mappingprojection import (
        MappingProjection,
    )
    from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM
    from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import (
        LCAMechanism,
    )

    _install_hook(Linear, "_gen_triton_function", _linear_gen_triton_function)
    _install_hook(Logistic, "_gen_triton_function", _logistic_gen_triton_function)
    _install_hook(MappingProjection, "_gen_triton_projection", _mapping_gen_triton_projection)
    _install_hook(DDM, "_gen_triton_mechanism", _ddm_gen_triton_mechanism)
    _install_hook(LCAMechanism, "_gen_triton_mechanism", _lca_gen_triton_mechanism)
    _HOOKS_INSTALLED = True


def triton_hook_diagnostics(graph, bindings) -> tuple[BatchedDiagnostic, ...]:
    ensure_triton_hooks_installed()
    diagnostics: list[BatchedDiagnostic] = []

    for projection in graph.projections:
        projection_object = bindings.projection(
            projection.sender,
            projection.sender_port,
            projection.receiver,
            projection.receiver_port,
        )
        if not hasattr(projection_object, "_gen_triton_projection"):
            diagnostics.append(
                BatchedDiagnostic(
                    component=(
                        f"{projection.sender}.{projection.sender_port}->"
                        f"{projection.receiver}.{projection.receiver_port}"
                    ),
                    reason="missing Triton projection hook for batched v2",
                    detail=type(projection_object).__name__,
                )
            )

    for node in graph.nodes:
        if node.component_type in {"TransferMechanism", "ProcessingMechanism"}:
            function = bindings.function(node.name)
            if not hasattr(function, "_gen_triton_function"):
                diagnostics.append(
                    BatchedDiagnostic(
                        component=node.name,
                        reason="missing Triton function hook for batched v2",
                        detail=node.function_type,
                    )
                )
        elif node.component_type in {"DDM", "LCAMechanism"}:
            mechanism = bindings.node(node.name)
            if not hasattr(mechanism, "_gen_triton_mechanism"):
                diagnostics.append(
                    BatchedDiagnostic(
                        component=node.name,
                        reason="missing Triton mechanism hook for batched v2",
                        detail=node.component_type,
                    )
                )

    return tuple(diagnostics)


def _install_hook(cls, method_name: str, method) -> None:
    existing = getattr(cls, method_name, None)
    if existing is not None and not getattr(existing, "_pnl_batched_triton_hook", False):
        return
    method._pnl_batched_triton_hook = True
    setattr(cls, method_name, method)


def _linear_gen_triton_function(self, ctx, node_spec, inputs, outputs):
    slope = ctx.param(node_spec, "slope")
    intercept = ctx.param(node_spec, "intercept")
    for input_value, output_value in zip(inputs, outputs):
        ctx.emit_call(
            TritonOpCall(
                template=_pnl_triton_linear,
                outputs=(output_value,),
                args=(input_value, slope, intercept),
            )
        )
    return outputs


def _logistic_gen_triton_function(self, ctx, node_spec, inputs, outputs):
    gain = ctx.param(node_spec, "gain")
    for input_value, output_value in zip(inputs, outputs):
        ctx.emit_call(
            TritonOpCall(
                template=_pnl_triton_logistic,
                outputs=(output_value,),
                args=(input_value, gain),
            )
        )
    return outputs


def _mapping_gen_triton_projection(self, ctx, projection_spec, sender_values, output_vars):
    helper_name = ctx.helper_name(_pnl_triton_projection_term)
    for col_idx, output_var in enumerate(output_vars):
        terms = []
        for row_idx, sender_value in enumerate(sender_values):
            coefficient = float(projection_spec.matrix[row_idx, col_idx])
            if coefficient:
                terms.append(
                    f"{helper_name}({sender_value}, {ctx.float_literal(coefficient)})"
                )
        ctx.line(f"{output_var} = {' + '.join(terms) if terms else ctx.zero_vector()}")
    return output_vars


def _ddm_gen_triton_mechanism(self, ctx, node_spec, inputs, outputs):
    ctx.emit_trial_random_base_if_needed()
    ctx.emit_call(
        TritonOpCall(
            template=_pnl_triton_ddm_integrate,
            outputs=tuple(outputs),
            args=(
                inputs[0],
                ctx.param(node_spec, "rate"),
                ctx.param(node_spec, "noise"),
                ctx.param(node_spec, "threshold"),
                ctx.param(node_spec, "non_decision_time"),
                ctx.param(node_spec, "time_step_size"),
                ctx.param(node_spec, "starting_value"),
                ctx.param(node_spec, "offset"),
                ctx.seed,
                ctx.ddm_random_base(node_spec.name),
                ctx.max_steps,
            ),
        )
    )
    return outputs


def _lca_gen_triton_mechanism(self, ctx, node_spec, inputs, outputs):
    if node_spec.output_width != 2:
        raise ValueError(
            "Triton batched LCA hook supports width 2, "
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
    steps_var = f"n_{_safe_ident(node_spec.name)}_lca_steps"
    stream0 = ctx.lca_stream_index(node_spec.name)
    stream1 = stream0 + 1

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
            ),
        )
    )
    return (act0, act1)


def _safe_ident(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
