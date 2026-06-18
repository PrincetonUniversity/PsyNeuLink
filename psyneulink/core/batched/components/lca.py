"""Batched op for width-2 `LCAMechanism`.

This is a narrow, performance-oriented lowering for width-2 LCA-like
stateful graphs, not a full implementation of PsyNeuLink ``LCAMechanism``
semantics (see BATCH_COMPILE_WIP.md, "LCA Caveats").  Because its state, RNG,
and termination handling are still custom, it uses the ``cpu_execute`` /
``triton_emit`` escape hatches instead of the declarative body form.
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


def _lca_supports(node) -> BatchedDiagnostic | None:
    width = _primary_output_width(node)
    if width != 2:
        return BatchedDiagnostic(
            getattr(node, "name", str(node)),
            "unsupported LCA width for batched v2",
            f"width={width}",
        )
    return None


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


def _lca_cpu_execute(
    node,
    input_value,
    raw_inputs,
    state,
    params,
    subject_idx,
    trial_idx,
    rng,
    max_steps,
):
    termination_input_node = node.attrs.get("termination_input_node")
    if termination_input_node is not None and termination_input_node in raw_inputs:
        cue = float(
            np.asarray(raw_inputs[termination_input_node][subject_idx, trial_idx]).reshape(-1)[0]
        )
    else:
        cue = float(node.attrs.get("termination_threshold", 1.0))
    pre_key = f"{node.name}.pre"
    act_key = f"{node.name}.act"
    pre, act = _simulate_lca_cue_period(
        task=input_value,
        pre_activity=state[pre_key],
        activity=state[act_key],
        cue=cue,
        gain=params["gain"],
        leak=params["leak"],
        competition=params["competition"],
        self_excitation=params["self_excitation"],
        noise=params["noise"],
        time_step_size=params["time_step_size"],
        max_steps=max(int(np.ceil(cue)), 1),
        rng=rng,
    )
    state[pre_key] = pre
    state[act_key] = act
    return {_primary_output_port(node): act.astype(np.float32)}


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


def _simulate_lca_cue_period(
    *,
    task: np.ndarray,
    pre_activity: np.ndarray,
    activity: np.ndarray,
    cue: float,
    gain: float,
    leak: float,
    competition: float,
    self_excitation: float,
    noise: float,
    time_step_size: float,
    max_steps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    steps = min(max(int(np.ceil(cue)), 0), max_steps)
    sqrt_dt = np.sqrt(time_step_size)
    pre = np.array(pre_activity, dtype=float, copy=True)
    act = np.array(activity, dtype=float, copy=True)
    for _ in range(steps):
        recurrent = np.array(
            [
                self_excitation * act[0] - competition * act[1],
                -competition * act[0] + self_excitation * act[1],
            ]
        )
        update = (task + recurrent - leak * pre) * time_step_size
        if noise:
            update = update + noise * sqrt_dt * rng.normal(size=2)
        pre = pre + update
        act = 1.0 / (1.0 + np.exp(-gain * pre))
    return pre, act


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


def _primary_output_width(node) -> int:
    output_ports = getattr(node, "output_ports", [])
    if not output_ports:
        return 1
    try:
        return int(np.asarray(output_ports[0].value).reshape(-1).size)
    except Exception:
        return 1


def _primary_output_port(node_spec) -> str:
    output_ports = tuple(node_spec.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _safe_ident(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)


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
        cpu_execute=_lca_cpu_execute,
        triton_emit=_lca_triton_emit,
    )
)
