from __future__ import annotations

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
)
from psyneulink.core.batched.kernel_ir import KernelIR, KernelOp
from psyneulink.core.batched.backend.triton.source_builder import SourceBuilder, emit_triton_header


_KERNEL_NAMES = {
    STATELESS_GRAPH_FUSION: "pnl_batched_stateless_graph_kernel",
    DDM_GRAPH_FUSION: "pnl_batched_ddm_graph_kernel",
    STATEFUL_GRAPH_FUSION: "pnl_batched_stateful_graph_kernel",
}


def triton_graph_kernel_source(kernel: KernelIR) -> str:
    """Emit inspectable Triton source for a generated graph kernel."""

    return TritonGraphEmitter(kernel).emit()


class TritonGraphEmitter:
    """Lower backend-neutral KernelIR ops to Triton source.

    This class is intentionally Triton-specific.  KernelIR remains free of
    `tl.*` syntax and source fragments so another backend can lower the same
    execution ops independently.
    """

    def __init__(self, kernel: KernelIR):
        self.kernel = kernel
        self.graph = kernel.graph
        self.builder = SourceBuilder()
        self.input_index = {
            input_spec.node: idx
            for idx, input_spec in enumerate(kernel.inputs)
        }
        self.param_vars: dict[str, str] = {}
        self.state_vars: dict[tuple[str, int], str] = {}
        self.value_vars: dict[str, list[str]] = {}
        self.lca_stream_index: dict[str, int] = {}
        self.ddm_stream_index: dict[str, int] = {}
        self.lca_stream_count = 0
        self.ddm_stream_count = 0
        self.output_cursor = 0
        self.lane_out_emitted = False

    def emit(self) -> str:
        self._index_rng_streams()
        emit_triton_header(
            self.builder,
            self._kernel_name(),
            self._signature_args(),
        )
        with self.builder.indent():
            self._emit_lane_decode()
            self._emit_params()
            self._emit_top_level_ops()
        return self.builder.render()

    def _kernel_name(self) -> str:
        try:
            return _KERNEL_NAMES[self.kernel.fusion_kind]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Triton graph fusion kind '{self.kernel.fusion_kind}'."
            ) from error

    def _signature_args(self) -> tuple[str, ...]:
        args = [f"input_{idx}" for idx, _ in enumerate(self.kernel.inputs)]
        args.extend(f"param_{idx}" for idx, _ in enumerate(self.kernel.params))
        args.extend(
            [
                "out",
                "total_lanes: tl.constexpr",
                "num_subjects: tl.constexpr",
            ]
        )

        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            args.extend(
                [
                    "num_estimates: tl.constexpr",
                    "num_trials",
                    "LCA_MAX_STEPS: tl.constexpr",
                    "MAX_STEPS: tl.constexpr",
                    "COMMON_RANDOM: tl.constexpr",
                    "SEED: tl.constexpr",
                    "BLOCK: tl.constexpr",
                ]
            )
            return tuple(args)

        args.extend(
            [
                "num_trials: tl.constexpr",
                "num_estimates: tl.constexpr",
            ]
        )
        if self.kernel.fusion_kind == DDM_GRAPH_FUSION:
            args.extend(
                [
                    "MAX_STEPS: tl.constexpr",
                    "COMMON_RANDOM: tl.constexpr",
                    "SEED: tl.constexpr",
                ]
            )
        args.append("BLOCK: tl.constexpr")
        return tuple(args)

    def _emit_lane_decode(self) -> None:
        self.builder.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = offsets < total_lanes")
        self.builder.line("estimate_idx = offsets % num_estimates")
        self.builder.line("tmp = offsets // num_estimates")

        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            self.builder.line("subject_idx = tmp % num_subjects")
            self.builder.line("param_idx = tmp // num_subjects")
            self.builder.line()
            return

        self.builder.line("trial_idx = tmp % num_trials")
        self.builder.line("tmp = tmp // num_trials")
        self.builder.line("subject_idx = tmp % num_subjects")
        self.builder.line("param_idx = tmp // num_subjects")
        self.builder.line()

    def _emit_params(self) -> None:
        for idx, param_spec in enumerate(self.kernel.params):
            var = f"param_{idx}_value"
            self.param_vars[param_spec.name] = var
            default = _float_literal(param_spec.default)
            self.builder.line(
                f"{var} = tl.load(param_{idx} + param_idx, mask=mask, other={default})"
            )
        if self.kernel.params:
            self.builder.line()

    def _emit_top_level_ops(self) -> None:
        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            for op in self.kernel.ops:
                if op.kind == "InitializeState":
                    self._emit_initialize_state()
                elif op.kind == "ForTrials":
                    self._emit_trial_loop(tuple(op.attrs["body"]))
                else:
                    raise ValueError(f"Unsupported stateful top-level op '{op.kind}'.")
            return

        self._emit_ops(self.kernel.ops)

    def _emit_initialize_state(self) -> None:
        for state in self.kernel.states:
            safe_state = _safe_ident(state.name)
            for idx, value in enumerate(state.initial_value):
                var = f"{safe_state}_{idx}"
                self.state_vars[(state.name, idx)] = var
                self.builder.line(
                    f"{var} = tl.full((BLOCK,), {_float_literal(value)}, tl.float32)"
                )
        if self.kernel.states:
            self.builder.line()

    def _emit_trial_loop(self, body: tuple[KernelOp, ...]) -> None:
        self.builder.line("trial_idx = 0")
        with self.builder.block("while trial_idx < num_trials"):
            self._emit_stateful_random_base()
            self.output_cursor = 0
            self.lane_out_emitted = False
            self._emit_ops(body)
            self.builder.line("trial_idx += 1")
        self.builder.line()

    def _emit_stateful_random_base(self) -> None:
        random_stride = (
            f"({self.lca_stream_count}) * LCA_MAX_STEPS "
            f"+ ({self.ddm_stream_count}) * MAX_STEPS"
        )
        self.builder.line(f"random_stride = {random_stride}")
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                "* num_trials + trial_idx) * random_stride"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = (((param_idx * num_subjects + subject_idx) "
                "* num_estimates + estimate_idx) * num_trials + trial_idx) "
                "* random_stride"
            )
        self.builder.line()

    def _emit_ops(self, ops: tuple[KernelOp, ...]) -> None:
        self.output_cursor = 0
        self.lane_out_emitted = False
        for op in ops:
            self._emit_op(op)

    def _emit_op(self, op: KernelOp) -> None:
        if op.kind == "LoadInput":
            self._emit_load_input(op)
        elif op.kind == "DenseMatVec":
            self._emit_dense_matvec(op)
        elif op.kind in {"CombineSum", "CombineProduct"}:
            self._emit_combine(op)
        elif op.kind in {"ElementwiseLinear", "ElementwiseLogistic"}:
            self._emit_stateless_function(op)
        elif op.kind == "LCAIntegrateUntilFinished":
            self._emit_lca(op)
        elif op.kind == "DDMIntegrateUntilFinished":
            self._emit_ddm(op)
        elif op.kind == "StoreOutput":
            self._emit_store_output(op)
        else:
            raise ValueError(f"Unsupported Triton KernelIR op '{op.kind}'.")

    def _emit_load_input(self, op: KernelOp) -> None:
        node_name = op.attrs["node"]
        input_spec = self.kernel.inputs[self.input_index[node_name]]
        values = [
            self._raw_input_value(node_name, idx)
            for idx in range(input_spec.width)
        ]
        self._set_value(op.outputs[0].name, values)

    def _emit_dense_matvec(self, op: KernelOp) -> None:
        sender_values = self._get_value(op.inputs[0].name)
        matrix = op.attrs["matrix"]
        output_vars = self._component_vars(op.outputs[0].name, matrix.shape[1])
        for col_idx, var in enumerate(output_vars):
            terms = []
            for row_idx, sender_var in enumerate(sender_values):
                coeff = float(matrix[row_idx, col_idx])
                if coeff:
                    terms.append(f"({sender_var}) * {_float_literal(coeff)}")
            expr = " + ".join(terms) if terms else _zero_vector()
            self.builder.line(f"{var} = {expr}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_combine(self, op: KernelOp) -> None:
        width = op.outputs[0].width
        output_vars = self._component_vars(op.outputs[0].name, width)
        input_values = [self._get_value(value.name) for value in op.inputs]
        operator = " * " if op.kind == "CombineProduct" else " + "
        for idx, var in enumerate(output_vars):
            components = [values[idx] for values in input_values]
            expr = operator.join(f"({component})" for component in components)
            self.builder.line(f"{var} = {expr or _zero_vector()}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_stateless_function(self, op: KernelOp) -> None:
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        params = op.attrs["params"]
        if op.kind == "ElementwiseLinear":
            slope = self.param_vars[params["slope"]]
            intercept = self.param_vars[params["intercept"]]
            output_values = [
                f"({slope}) * ({input_value}) + ({intercept})"
                for input_value in input_values
            ]
        else:
            gain = self.param_vars[params["gain"]]
            output_values = [
                f"1.0 / (1.0 + tl.exp(-({gain}) * ({input_value})))"
                for input_value in input_values
            ]

        for var, expr in zip(output_vars, output_values):
            self.builder.line(f"{var} = {expr}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_lca(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        if node.output_width != 2:
            raise ValueError(
                f"Stateful Triton graph supports LCAMechanism width 2, got {node.output_width}."
            )

        input_values = self._get_value(op.inputs[0].name)
        params = op.attrs["params"]
        pre0 = self.state_vars[(op.attrs["pre_state"], 0)]
        pre1 = self.state_vars[(op.attrs["pre_state"], 1)]
        act0 = self.state_vars[(op.attrs["act_state"], 0)]
        act1 = self.state_vars[(op.attrs["act_state"], 1)]
        gain = self.param_vars[params["gain"]]
        leak = self.param_vars[params["leak"]]
        competition = self.param_vars[params["competition"]]
        self_excitation = self.param_vars[params["self_excitation"]]
        noise = self.param_vars[params["noise"]]
        dt = self.param_vars[params["time_step_size"]]
        safe_name = _safe_ident(node.name)
        termination_node = op.attrs.get("termination_input_node")
        if termination_node:
            cue_value = self._raw_input_value(termination_node)
        else:
            cue_value = _float_literal(node.attrs.get("termination_threshold", 1.0))
        stream0 = self.lca_stream_index[node.name]
        stream1 = stream0 + 1

        self.builder.line(
            f"{safe_name}_lca_steps = tl.minimum(tl.maximum(tl.ceil({cue_value}), 0.0), "
            "LCA_MAX_STEPS)"
        )
        self.builder.line(f"{safe_name}_sqrt_dt = tl.sqrt({dt})")
        with self.builder.block("for step in tl.range(0, LCA_MAX_STEPS, 1, loop_unroll_factor=1)"):
            self.builder.line(f"active_lca = step < {safe_name}_lca_steps")
            self.builder.line(f"rec0 = ({self_excitation}) * {act0} - ({competition}) * {act1}")
            self.builder.line(f"rec1 = -({competition}) * {act0} + ({self_excitation}) * {act1}")
            self.builder.line(f"n0 = tl.randn(SEED, random_base + ({stream0}) * LCA_MAX_STEPS + step)")
            self.builder.line(f"n1 = tl.randn(SEED, random_base + ({stream1}) * LCA_MAX_STEPS + step)")
            self.builder.line(
                f"upd0 = (({input_values[0]}) + rec0 - ({leak}) * {pre0}) * ({dt}) "
                f"+ ({noise}) * {safe_name}_sqrt_dt * n0"
            )
            self.builder.line(
                f"upd1 = (({input_values[1]}) + rec1 - ({leak}) * {pre1}) * ({dt}) "
                f"+ ({noise}) * {safe_name}_sqrt_dt * n1"
            )
            self.builder.line(f"{pre0} = tl.where(active_lca, {pre0} + upd0, {pre0})")
            self.builder.line(f"{pre1} = tl.where(active_lca, {pre1} + upd1, {pre1})")
            self.builder.line(
                f"{act0} = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-({gain}) * {pre0})), {act0})"
            )
            self.builder.line(
                f"{act1} = tl.where(active_lca, 1.0 / (1.0 + tl.exp(-({gain}) * {pre1})), {act1})"
            )
        self._set_value(op.outputs[0].name, [act0, act1])
        self.builder.line()

    def _emit_ddm(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        input_values = self._get_value(op.inputs[0].name)
        params = op.attrs["params"]
        rate = self.param_vars[params["rate"]]
        noise = self.param_vars[params["noise"]]
        threshold = self.param_vars[params["threshold"]]
        non_decision_time = self.param_vars[params["non_decision_time"]]
        dt = self.param_vars[params["time_step_size"]]
        starting_value = self.param_vars[params["starting_value"]]
        step_offset = self.param_vars[params["offset"]]
        safe_name = _safe_ident(node.name)
        decision_var = f"{safe_name}_decision"
        response_time_var = f"{safe_name}_response_time"
        random_base = self._ddm_random_base(node.name)

        self.builder.line(f"{safe_name}_value = {starting_value}")
        self.builder.line(f"{safe_name}_steps = tl.zeros((BLOCK,), dtype=tl.float32)")
        self.builder.line(f"{safe_name}_sqrt_dt = tl.sqrt({dt})")
        self.builder.line(
            f"{safe_name}_boundary_tolerance = tl.maximum(1.0e-7, {threshold} * 1.0e-6)"
        )
        self._emit_trial_random_base_if_needed()
        with self.builder.block("for step in tl.range(0, MAX_STEPS, 1, loop_unroll_factor=1)"):
            self.builder.line(
                f"{safe_name}_active = tl.abs({safe_name}_value) "
                f"+ {safe_name}_boundary_tolerance < {threshold}"
            )
            self.builder.line(f"random_draw = tl.randn(SEED, {random_base} + step)")
            self.builder.line(
                f"{safe_name}_updated = {safe_name}_value + ({rate}) "
                f"* ({input_values[0]}) * ({dt}) "
                f"+ ({noise}) * {safe_name}_sqrt_dt * random_draw"
            )
            self.builder.line(
                f"{safe_name}_updated = tl.minimum(tl.maximum({safe_name}_updated "
                f"+ ({step_offset}), -({threshold})), {threshold})"
            )
            self.builder.line(
                f"{safe_name}_value = tl.where({safe_name}_active, "
                f"{safe_name}_updated, {safe_name}_value)"
            )
            self.builder.line(
                f"{safe_name}_steps += tl.where({safe_name}_active, 1.0, 0.0)"
            )
        self.builder.line(f"{decision_var} = tl.where({safe_name}_value > 0.0, 1.0, 0.0)")
        self.builder.line(f"{response_time_var} = ({non_decision_time}) + {safe_name}_steps * ({dt})")
        self._set_value(op.outputs[0].name, [decision_var])
        self._set_value(op.outputs[1].name, [response_time_var])
        self._set_primary_alias(node.name, [decision_var])
        self.builder.line()

    def _emit_trial_random_base_if_needed(self) -> None:
        if self.kernel.fusion_kind != DDM_GRAPH_FUSION:
            return
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                "* num_trials + trial_idx) * MAX_STEPS"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = (((param_idx * num_subjects + subject_idx) "
                "* num_estimates + estimate_idx) * num_trials + trial_idx) "
                "* MAX_STEPS"
            )

    def _emit_store_output(self, op: KernelOp) -> None:
        if not self.lane_out_emitted:
            self._emit_lane_out()
        source_values = self._get_value(op.inputs[0].name)
        for idx in range(op.attrs["width"]):
            self.builder.line(
                f"tl.store(out + lane_out + {self.output_cursor + idx}, "
                f"{source_values[idx]}, mask=mask)"
            )
        self.output_cursor += op.attrs["width"]

    def _emit_lane_out(self) -> None:
        output_width = sum(output.width for output in self.kernel.outputs)
        self.builder.line(
            "lane_out = (((param_idx * num_subjects + subject_idx) "
            "* num_trials + trial_idx) * "
            f"num_estimates + estimate_idx) * {output_width}"
        )
        self.lane_out_emitted = True

    def _raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        if node_name in self.input_index:
            input_spec = self.kernel.inputs[self.input_index[node_name]]
            base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
            return (
                f"tl.load(input_{self.input_index[node_name]} + {base} + {component_idx}, "
                "mask=mask, other=0.0)"
            )
        node = self.graph.node(node_name)
        return self._get_value(f"{node_name}:{_primary_output_port_name(node)}")[component_idx]

    def _ddm_random_base(self, node_name: str) -> str:
        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            return (
                f"random_base + ({self.lca_stream_count}) * LCA_MAX_STEPS "
                f"+ ({self.ddm_stream_index[node_name]}) * MAX_STEPS"
            )
        return "random_base"

    def _index_rng_streams(self) -> None:
        for stream in self.kernel.rng_streams:
            if stream.step_extent == "LCA_MAX_STEPS":
                self.lca_stream_index[stream.node] = self.lca_stream_count
                self.lca_stream_count += stream.width
            elif stream.step_extent == "MAX_STEPS":
                self.ddm_stream_index[stream.node] = self.ddm_stream_count
                self.ddm_stream_count += stream.width

    def _component_vars(self, value_name: str, width: int) -> list[str]:
        base = _safe_ident(value_name)
        return [f"{base}_{idx}" for idx in range(width)]

    def _set_value(self, name: str, values: list[str]) -> None:
        self.value_vars[name] = values

    def _set_primary_alias(self, node_name: str, values: list[str]) -> None:
        node = self.graph.node(node_name)
        self.value_vars[f"{node_name}:{_primary_output_port_name(node)}"] = values

    def _get_value(self, name: str) -> list[str]:
        try:
            return self.value_vars[name]
        except KeyError as error:
            raise ValueError(f"Triton graph emitter has no value for '{name}'.") from error


def _primary_output_port_name(node) -> str:
    output_ports = tuple(node.attrs.get("output_ports", ()))
    if output_ports:
        return output_ports[0]
    return "RESULT"


def _safe_ident(name: str) -> str:
    return "n_" + "".join(ch if ch.isalnum() else "_" for ch in name)


def _float_literal(value: float) -> str:
    return repr(float(value))


def _zero_vector() -> str:
    return "tl.zeros((BLOCK,), dtype=tl.float32)"
