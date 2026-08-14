"""KernelOp emission for the Triton emitter.

`OpEmitMixin` holds the op-dispatch and per-op emitters of `TritonGraphEmitter`,
plus the value table.  This is the module future milestones extend when adding
new `KernelOp` kinds (e.g. truncation `StoreFlag`, scheduling variants,
time-varying threshold).  It shares the emitter's mutable state and is mixed
into the concrete emitter in `emitter.py`.
"""

from __future__ import annotations

from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION, STATEFUL_GRAPH_FUSION
from psyneulink.core.batched.kernel_ir import (
    KernelOp,
    component_symbol,
    node_output_value_name,
)
from psyneulink.core.batched.backend.triton.api import TritonEmitContext, TritonOpCall
from psyneulink.core.batched.backend.triton.emit._helpers import (
    float_literal,
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

        if self.kernel.fusion_kind == COEVOLVING_GRAPH_FUSION:
            for op in self.kernel.ops:
                if op.kind == "InitializeState":
                    self._emit_initialize_state()
                elif op.kind == "ForTrials":
                    self._emit_coevolving_trial_loop(tuple(op.attrs["body"]))
                else:
                    raise ValueError(f"Unsupported co-evolving top-level op '{op.kind}'.")
            return

        self._emit_ops(self.kernel.ops)

    # ---- co-evolution (fused per-step loop over coupled stateful ops) ----

    def _emit_coevolving_trial_loop(self, body: tuple[KernelOp, ...]) -> None:
        terminator_index = self._coevolving_terminator_index(body)
        terminator_op = body[terminator_index]
        # In-loop ops co-evolve each step (up to and including the terminator's
        # step); ops after the terminator depend on its final readout (gates,
        # store-output, truncation StoreFlag) and run once after the loop.
        in_loop_ops = body[: terminator_index + 1]
        post_loop_ops = body[terminator_index + 1 :]

        self.builder.line("trial_idx = 0")
        with self.builder.block("while trial_idx < num_trials"):
            self._emit_stateful_random_base()
            self.output_cursor = 0
            self.lane_out_emitted = False
            self.diag_lane_emitted = False
            self._emit_init_trial_states(terminator_op)
            terminator_spec = self._spec_for_op(terminator_op)
            terminator_node = self.graph.node(terminator_op.target)
            finished_var = self.state_vars[
                (f"{terminator_node.name}.{terminator_spec.finished_output}", 0)
            ]
            # Hoist loop-invariant ops (constant inputs and projections that do
            # not depend on a stepper's evolving state) out of the step loop:
            # they are computed once, and Triton loop-body variables do not
            # escape the loop, so post-loop ops must read them from outer scope.
            hoisted_ops, stepping_ops = self._partition_coevolving_ops(in_loop_ops)
            for op in hoisted_ops:
                self._emit_op(op)
            # Early exit: every stepper freezes once the terminator reports
            # `finished` (that is the `step_emit` contract), so iterations after
            # the last active lane finishes are no-ops.  Stop the block as soon
            # as none are left rather than always running MAX_STEPS, which
            # otherwise makes runtime scale with the cap instead of with the
            # decision times.  Lanes past `total_lanes` are excluded: they carry
            # default parameters, never finish, and would pin the loop open.
            self.builder.line("step = 0")
            with self.builder.block(
                f"while (step < MAX_STEPS) & ({self._any_running_expr(finished_var)})"
            ):
                for op in stepping_ops:
                    self._emit_coevolving_op(op, "step", finished_var)
                self.builder.line("step += 1")
            self.builder.line()
            self._emit_terminator_readout(terminator_op)
            for op in post_loop_ops:
                self._emit_op(op)
            self.builder.line("trial_idx += 1")
        self.builder.line()

    @staticmethod
    def _any_running_expr(finished_var: str) -> str:
        """Block-wide scalar test: does any in-range lane still have work to do?

        Reduces to a scalar so it can drive a Triton `while` condition; the
        exit is per lane *block*, so a block runs until its slowest lane
        finishes (not until its own lane does).
        """

        return f"tl.max(tl.where(mask & ({finished_var} == 0.0), 1, 0)) > 0"

    def _partition_coevolving_ops(self, in_loop_ops):
        """Split into (hoisted, stepping): an op steps each iteration if it is a
        stepper or transitively consumes a stepper's evolving output; everything
        else is loop-invariant and is emitted once before the loop.
        """

        variant: set[str] = set()
        hoisted = []
        stepping = []
        for op in in_loop_ops:
            is_stepper = (
                op.kind == "CallMechanism"
                and self._spec_for_op(op).can_step
            )
            # A delayed-onset node's output depends on `step` (withheld until its
            # onset), so it and everything downstream must run inside the loop.
            has_onset = op.attrs.get("onset_step") is not None
            depends_on_variant = any(inp.name in variant for inp in op.inputs)
            if is_stepper or has_onset or depends_on_variant:
                stepping.append(op)
                for output in op.outputs:
                    variant.add(output.name)
            else:
                hoisted.append(op)
        return hoisted, stepping

    def _coevolving_terminator_index(self, body: tuple[KernelOp, ...]) -> int:
        for index, op in enumerate(body):
            if op.kind == "CallMechanism":
                spec = self._spec_for_op(op)
                if spec.is_terminator:
                    return index
        raise ValueError("co-evolving graph has no terminator op")

    def _emit_coevolving_op(self, op: KernelOp, step_var: str, finished_var: str) -> None:
        if op.kind == "CallMechanism":
            spec = self._spec_for_op(op)
            if spec.can_step:
                self._emit_step_mechanism(op, spec, step_var, finished_var)
                return
        self._emit_op(op)

    def _emit_step_mechanism(self, op: KernelOp, spec, step_var: str, finished_var: str) -> None:
        node = self.graph.node(op.target)
        input_values = self._get_value(op.inputs[0].name)
        output_vars = []
        for output in op.outputs:
            output_vars.extend(self._component_vars(output.name, output.width))
        result = spec.step_emit(
            TritonEmitContext(self), node, input_values, output_vars, step_var, finished_var
        )
        # A non-terminator stepper (e.g. LCA) returns its current outputs; the
        # terminator (e.g. DDM) only advances state and returns None — its
        # outputs are produced by the readout after the loop.
        if result is not None:
            values = list(result)
            cursor = 0
            for output in op.outputs:
                self._set_value(output.name, values[cursor : cursor + output.width])
                cursor += output.width

    def _emit_init_trial_states(self, terminator_op: KernelOp) -> None:
        spec = self._spec_for_op(terminator_op)
        node = self.graph.node(terminator_op.target)
        node_symbol = component_symbol(self.graph, node)
        for state_slot, decl in enumerate(spec.trial_states):
            state_name = f"{node.name}.{decl.name}"
            for idx in range(decl.width or 1):
                var = f"{node_symbol}_trial_state_{state_slot}_{idx}"
                self.state_vars[(state_name, idx)] = var
                self.builder.line(
                    f"{var} = tl.full((BLOCK,), {float_literal(decl.initial)}, tl.float32)"
                )
        self.builder.line()

    def _emit_terminator_readout(self, terminator_op: KernelOp) -> None:
        spec = self._spec_for_op(terminator_op)
        node = self.graph.node(terminator_op.target)
        output_vars = []
        for output in terminator_op.outputs:
            output_vars.extend(self._component_vars(output.name, output.width))
        spec.readout_emit(TritonEmitContext(self), node, output_vars)
        cursor = 0
        for output in terminator_op.outputs:
            self._set_value(output.name, output_vars[cursor : cursor + output.width])
            cursor += output.width
        # A terminator's diagnostic (e.g. DDM "truncated") is exactly "never
        # finished within MAX_STEPS"; route it through the existing diag channel.
        finished_var = self.state_vars[(f"{node.name}.{spec.finished_output}", 0)]
        diagnostic_names = tuple(terminator_op.attrs.get("diagnostics", ()))
        diagnostic_values = tuple(terminator_op.attrs.get("diagnostic_values", ()))
        for name, value_name in zip(diagnostic_names, diagnostic_values):
            diag_var = self._component_vars(value_name, 1)[0]
            self.builder.line(f"{diag_var} = tl.where({finished_var} == 0.0, 1.0, 0.0)")
            self._set_value(value_name, [diag_var])
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

    def _emit_ops(self, ops: tuple[KernelOp, ...]) -> None:
        self.output_cursor = 0
        self.lane_out_emitted = False
        self.diag_lane_emitted = False
        for op in ops:
            self._emit_op(op)

    def _emit_op(self, op: KernelOp) -> None:
        if op.kind == "LoadInput":
            self._emit_load_input(op)
        elif op.kind == "CallProjection":
            self._emit_projection_call(op)
        elif op.kind in {"CombineSum", "CombineProduct"}:
            self._emit_combine(op)
        elif op.kind == "Concatenate":
            self._emit_concatenate(op)
        elif op.kind == "ExtractSlice":
            self._emit_extract_slice(op)
        elif op.kind == "AddConstant":
            self._emit_add_constant(op)
        elif op.kind == "Clamp":
            self._emit_clamp(op)
        elif op.kind == "CallFunction":
            self._emit_function_call(op)
        elif op.kind == "CallMechanism":
            self._emit_mechanism_call(op)
        elif op.kind == "StoreOutput":
            self._emit_store_output(op)
        elif op.kind == "StoreFlag":
            self._emit_store_flag(op)
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
        spec = self._spec_for_op(op)
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

    def _emit_concatenate(self, op: KernelOp) -> None:
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        components = [
            component
            for input_value in op.inputs
            for component in self._get_value(input_value.name)
        ]
        for output_var, component in zip(output_vars, components):
            self.builder.line(f"{output_var} = {component}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_extract_slice(self, op: KernelOp) -> None:
        input_values = self._get_value(op.inputs[0].name)
        start = int(op.attrs["start"])
        stop = int(op.attrs["stop"])
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        for output_var, component in zip(output_vars, input_values[start:stop]):
            self.builder.line(f"{output_var} = {component}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_add_constant(self, op: KernelOp) -> None:
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        constants = op.attrs["value"]
        for index, (input_value, output_var) in enumerate(
            zip(input_values, output_vars)
        ):
            constant = constants[0] if len(constants) == 1 else constants[index]
            self.builder.line(
                f"{output_var} = {input_value} + ({float_literal(constant)})"
            )
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_clamp(self, op: KernelOp) -> None:
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        lower = op.attrs["lower"]
        upper = op.attrs["upper"]
        for index, (input_value, output_var) in enumerate(
            zip(input_values, output_vars)
        ):
            component_lower = lower[0] if len(lower) == 1 else lower[index]
            component_upper = upper[0] if len(upper) == 1 else upper[index]
            self.builder.line(
                f"{output_var} = tl.minimum(tl.maximum({input_value}, "
                f"{float_literal(component_lower)}), "
                f"{float_literal(component_upper)})"
            )
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_function_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        spec = self._spec_for_op(op)
        if spec.triton_template is None:
            raise ValueError(
                "Batched op spec for function "
                f"{node.function_type} on '{node.name}' has no Triton implementation."
            )
        input_values = self._get_value(op.inputs[0].name)
        integrator_pre = op.attrs.get("integrator_pre")
        if integrator_pre is not None:
            # Fold a fires-once integrator's single affine step (a*input + b) in
            # front of the function: function(a*input + b).
            a, b = integrator_pre
            input_values = [f"({a!r} * {value} + {b!r})" for value in input_values]
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
        # Delayed within-trial onset (ITI): withhold this node's output (0) until
        # its onset step. `step` is the fused co-evolution loop index; onset_step
        # is only set in that context.
        onset = op.attrs.get("onset_step")
        if onset is not None:
            for output_var in output_vars:
                self.builder.line(f"{output_var} = tl.where(step >= {onset}, {output_var}, 0.0)")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_mechanism_call(self, op: KernelOp) -> None:
        node = self.graph.node(op.target)
        spec = self._spec_for_op(op)
        if not spec.has_triton:
            raise ValueError(
                "Batched op spec for mechanism "
                f"{node.component_type} '{node.name}' has no Triton implementation."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = []
        for output in op.outputs:
            output_vars.extend(self._component_vars(output.name, output.width))

        diag_names = tuple(op.attrs.get("diagnostics", ()))
        diag_value_names = tuple(op.attrs.get("diagnostic_values", ()))
        diag_vars = [self._component_vars(name, 1)[0] for name in diag_value_names]

        if spec.triton_emit is not None:
            if diag_names:
                raise ValueError(
                    f"Batched op for '{node.name}' declares diagnostics, which the "
                    "triton_emit escape hatch does not support yet."
                )
            values = list(
                spec.triton_emit(
                    TritonEmitContext(self),
                    node,
                    input_values,
                    output_vars,
                )
            )
        else:
            values = self._emit_declarative_mechanism(
                spec, node, input_values, output_vars, diag_vars
            )

        for value_name, var in zip(diag_value_names, diag_vars):
            self._set_value(value_name, [var])

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
        primary_value_name = node_output_value_name(
            self.graph,
            node,
            primary_output_port_name(node),
        )
        if primary_value_name not in self.value_vars:
            self._set_value(primary_value_name, self._get_value(op.outputs[0].name))
        self.builder.line()

    def _emit_declarative_mechanism(
        self, spec, node, input_values, output_vars, diag_vars=()
    ) -> list[str]:
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
            elif binding.role == "lane_mask":
                args.append("mask")
            else:
                raise ValueError(
                    f"Batched op for '{node.name}' has an unsupported Triton arg "
                    f"role '{binding.role}'."
                )
        TritonEmitContext(self).emit_call(
            TritonOpCall(
                template=spec.triton_template,
                outputs=tuple(output_vars) + tuple(diag_vars),
                args=tuple(args),
            )
        )
        return list(output_vars)

    def _emit_store_output(self, op: KernelOp) -> None:
        if not self.lane_out_emitted:
            self._emit_lane_out()
        source_values = self._get_value(op.inputs[0].name)
        flat_start = int(op.attrs.get("flat_start", -1))
        if flat_start < 0:
            flat_start = self.output_cursor
        for idx in range(op.attrs["width"]):
            self.builder.line(
                f"tl.store(out + lane_out + {flat_start + idx}, "
                f"{source_values[idx]}, mask=mask)"
            )
        self.output_cursor = max(self.output_cursor, flat_start + op.attrs["width"])

    def _emit_store_flag(self, op: KernelOp) -> None:
        if not self.diag_lane_emitted:
            self._emit_diag_lane()
        value = self._get_value(op.inputs[0].name)[0]
        self.builder.line(
            f"tl.store(diag + diag_lane + {op.attrs['slot']}, {value}, mask=mask)"
        )

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

    def _spec_for_op(self, op: KernelOp):
        return self.kernel.op_specs.lookup_spec(op.attrs["spec_key"])

    def _projection_spec_for_op(self, op: KernelOp):
        projection_id = int(op.attrs.get("projection_id", -1))
        if projection_id >= 0:
            for projection in self.graph.projections:
                if projection.projection_id == projection_id:
                    return projection
            raise ValueError(
                "Triton graph emitter could not resolve projection id "
                f"{projection_id}."
            )
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
