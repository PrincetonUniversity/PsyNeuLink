"""KernelOp emission for the Triton emitter.

`OpEmitMixin` holds the op-dispatch and per-op emitters of `TritonGraphEmitter`,
plus the value table.  This is the module future milestones extend when adding
new `KernelOp` kinds (e.g. truncation `StoreFlag`, scheduling variants,
time-varying threshold).  It shares the emitter's mutable state and is mixed
into the concrete emitter in `emitter.py`.
"""

from __future__ import annotations

from psyneulink.core.batched import specs
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION
from psyneulink.core.batched.kernel_ir import KernelOp
from psyneulink.core.batched.backend.triton.api import TritonEmitContext, TritonOpCall
from psyneulink.core.batched.backend.triton.emit._helpers import (
    primary_output_port_name,
    safe_ident,
    zero_vector,
)


class OpEmitMixin:
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

    def _emit_trial_loop(self, body: tuple[KernelOp, ...]) -> None:
        self.builder.line("trial_idx = 0")
        with self.builder.block("while trial_idx < num_trials"):
            self._emit_stateful_random_base()
            self.output_cursor = 0
            self.lane_out_emitted = False
            self._emit_ops(body)
            self.builder.line("trial_idx += 1")
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
        spec = specs.lookup_spec(projection_spec.spec_key)
        if spec.triton_emit is None:
            raise ValueError(
                "Batched op spec for projection "
                f"'{projection_spec.sender}->{projection_spec.receiver}' has no "
                "Triton implementation."
            )
        sender_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        values = spec.triton_emit(
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
            self.builder.line(f"{var} = {expr or zero_vector()}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_function_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        spec = specs.lookup_spec(op.attrs["spec_key"])
        if spec.triton_template is None:
            raise ValueError(
                "Batched op spec for function "
                f"{node.function_type} on '{node.name}' has no Triton implementation."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        param_args = tuple(
            self.param_vars[node.params[binding.arg]] for binding in spec.params
        )
        ctx = TritonEmitContext(self)
        for input_value, output_var in zip(input_values, output_vars):
            ctx.emit_call(
                TritonOpCall(
                    template=spec.triton_template,
                    outputs=(output_var,),
                    args=(input_value,) + param_args,
                )
            )
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_mechanism_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        spec = specs.lookup_spec(op.attrs["spec_key"])
        if not spec.has_triton:
            raise ValueError(
                "Batched op spec for mechanism "
                f"{node.component_type} '{node.name}' has no Triton implementation."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = []
        for output in op.outputs:
            output_vars.extend(self._component_vars(output.name, output.width))

        if spec.triton_emit is not None:
            values = list(
                spec.triton_emit(
                    TritonEmitContext(self),
                    node,
                    input_values,
                    output_vars,
                )
            )
        else:
            values = self._emit_declarative_mechanism(spec, node, input_values, output_vars)

        expected_width = sum(output.width for output in op.outputs)
        if len(values) != expected_width:
            raise ValueError(
                f"Batched Triton op for '{node.name}' returned {len(values)} values, "
                f"expected {expected_width}."
            )
        cursor = 0
        for output in op.outputs:
            self._set_value(output.name, values[cursor:cursor + output.width])
            cursor += output.width
        primary_value_name = f"{node.name}:{primary_output_port_name(node)}"
        if primary_value_name not in self.value_vars:
            self._set_value(primary_value_name, self._get_value(op.outputs[0].name))
        self.builder.line()

    def _emit_declarative_mechanism(self, spec, node, input_values, output_vars) -> list[str]:
        if spec.states:
            raise ValueError(
                f"Batched op for '{node.name}' declares lane state; declarative "
                "stateful mechanisms are not supported yet - provide triton_emit."
            )
        if node.input_width != 1:
            raise ValueError(
                f"Declarative batched mechanism op for '{node.name}' requires "
                f"input width 1, got {node.input_width}."
            )
        if spec.rng:
            self._emit_trial_random_base_if_needed()

        args = []
        for binding in spec.triton_bindings:
            if binding.role == "input":
                args.append(input_values[0])
            elif binding.role == "param":
                args.append(self.param_vars[node.params[binding.name]])
            elif binding.role == "seed":
                args.append("SEED")
            elif binding.role == "rng_base":
                args.append(self._rng_base(node.name))
            elif binding.role == "max_steps":
                args.append("MAX_STEPS")
            else:
                raise ValueError(
                    f"Batched op for '{node.name}' has an unsupported Triton arg "
                    f"role '{binding.role}'."
                )
        TritonEmitContext(self).emit_call(
            TritonOpCall(
                template=spec.triton_template,
                outputs=tuple(output_vars),
                args=tuple(args),
            )
        )
        return list(output_vars)

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

    def _component_vars(self, value_name: str, width: int) -> list[str]:
        base = safe_ident(value_name)
        return [f"{base}_{idx}" for idx in range(width)]

    def _set_value(self, name: str, values: list[str]) -> None:
        self.value_vars[name] = values

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
