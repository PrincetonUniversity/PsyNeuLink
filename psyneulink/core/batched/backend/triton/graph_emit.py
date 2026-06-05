from __future__ import annotations

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
)
from psyneulink.core.batched.bindings import (
    EMPTY_COMPONENT_BINDINGS,
    BatchedComponentBindings,
)
from psyneulink.core.batched.kernel_ir import KernelIR, KernelOp
from psyneulink.core.batched.backend.triton.api import (
    TritonEmitContext,
    TritonOpTemplate,
)
from psyneulink.core.batched.backend.triton.component_hooks import (
    ensure_triton_hooks_installed,
)
from psyneulink.core.batched.backend.triton.source_builder import (
    SourceBuilder,
    emit_triton_function_header,
    emit_triton_imports,
)


_KERNEL_NAMES = {
    STATELESS_GRAPH_FUSION: "pnl_batched_stateless_graph_kernel",
    DDM_GRAPH_FUSION: "pnl_batched_ddm_graph_kernel",
    STATEFUL_GRAPH_FUSION: "pnl_batched_stateful_graph_kernel",
}


def triton_graph_kernel_source(
    kernel: KernelIR,
    component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS,
) -> str:
    """Emit inspectable Triton source for a generated graph kernel."""

    return TritonGraphEmitter(kernel, component_bindings).emit()


class TritonGraphEmitter:
    """Lower backend-neutral KernelIR ops to Triton source.

    This class is intentionally Triton-specific.  KernelIR remains free of
    `tl.*` syntax and source fragments so another backend can lower the same
    execution ops independently.
    """

    def __init__(
        self,
        kernel: KernelIR,
        component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS,
    ):
        self.kernel = kernel
        self.graph = kernel.graph
        self.component_bindings = component_bindings
        self.builder = SourceBuilder()
        self.templates: dict[str, TritonOpTemplate] = {}
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
        ensure_triton_hooks_installed()
        self._index_rng_streams()
        with self.builder.indent():
            self._emit_lane_decode()
            self._emit_params()
            self._emit_top_level_ops()
        body_source = self.builder.render()

        module_builder = SourceBuilder()
        emit_triton_imports(module_builder)
        for template in self.templates.values():
            module_builder.lines(template.source.splitlines())
            module_builder.line()
            module_builder.line()
        emit_triton_function_header(
            module_builder,
            self._kernel_name(),
            self._signature_args(),
        )
        module_builder.lines(body_source.splitlines())
        return module_builder.render()

    def register_template(self, template: TritonOpTemplate) -> str:
        existing = self.templates.get(template.name)
        if existing is not None and existing.source != template.source:
            raise ValueError(f"Conflicting Triton helper template '{template.name}'.")
        self.templates[template.name] = template
        return template.name

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
        elif op.kind == "CallProjection":
            self._emit_projection_call(op)
        elif op.kind in {"CombineSum", "CombineProduct"}:
            self._emit_combine(op)
        elif op.kind == "CallFunction":
            self._emit_function_call(op)
        elif op.kind == "CallMechanism":
            self._emit_mechanism_call(op)
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

    def _emit_projection_call(self, op: KernelOp) -> None:
        projection_spec = self._projection_spec_for_op(op)
        projection = self.component_bindings.projection(
            projection_spec.sender,
            projection_spec.sender_port,
            projection_spec.receiver,
            projection_spec.receiver_port,
        )
        hook = getattr(projection, "_gen_triton_projection", None)
        if hook is None:
            raise ValueError(
                "Triton graph emitter has no projection hook for "
                f"{type(projection).__name__} "
                f"'{projection_spec.sender}->{projection_spec.receiver}'."
            )
        sender_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        values = hook(
            TritonEmitContext(self),
            projection_spec,
            sender_values,
            output_vars,
        )
        self._set_value(op.outputs[0].name, list(values))
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

    def _emit_function_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        function = self.component_bindings.function(node.name)
        hook = getattr(function, "_gen_triton_function", None)
        if hook is None:
            raise ValueError(
                "Triton graph emitter has no function hook for "
                f"{node.function_type} on '{node.name}'."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        values = hook(
            TritonEmitContext(self),
            node,
            input_values,
            output_vars,
        )
        self._set_value(op.outputs[0].name, list(values))
        self.builder.line()

    def _emit_mechanism_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        mechanism = self.component_bindings.node(node.name)
        hook = getattr(mechanism, "_gen_triton_mechanism", None)
        if hook is None:
            raise ValueError(
                "Triton graph emitter has no mechanism hook for "
                f"{node.component_type} '{node.name}'."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = []
        for output in op.outputs:
            output_vars.extend(self._component_vars(output.name, output.width))
        values = list(
            hook(
                TritonEmitContext(self),
                node,
                input_values,
                output_vars,
            )
        )
        expected_width = sum(output.width for output in op.outputs)
        if len(values) != expected_width:
            raise ValueError(
                f"Triton hook for '{node.name}' returned {len(values)} values, "
                f"expected {expected_width}."
            )
        cursor = 0
        for output in op.outputs:
            self._set_value(output.name, values[cursor:cursor + output.width])
            cursor += output.width
        if node.component_type == "DDM":
            self._set_primary_alias(node.name, self._get_value(op.outputs[0].name))
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

    def emit_trial_random_base_if_needed(self) -> None:
        self._emit_trial_random_base_if_needed()

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

    def raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        return self._raw_input_value(node_name, component_idx)

    def _ddm_random_base(self, node_name: str) -> str:
        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            return (
                f"random_base + ({self.lca_stream_count}) * LCA_MAX_STEPS "
                f"+ ({self.ddm_stream_index[node_name]}) * MAX_STEPS"
            )
        return "random_base"

    def ddm_random_base(self, node_name: str) -> str:
        return self._ddm_random_base(node_name)

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

    def _projection_spec_for_op(self, op: KernelOp):
        for projection in self.graph.projections:
            if (
                projection.sender == op.attrs["sender"]
                and projection.sender_port == op.attrs["sender_port"]
                and projection.receiver == op.attrs["receiver"]
                and projection.receiver_port == op.attrs["receiver_port"]
            ):
                return projection
        raise ValueError(
            "Triton graph emitter could not resolve projection "
            f"{op.attrs['sender']}.{op.attrs['sender_port']}->"
            f"{op.attrs['receiver']}.{op.attrs['receiver_port']}."
        )


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
