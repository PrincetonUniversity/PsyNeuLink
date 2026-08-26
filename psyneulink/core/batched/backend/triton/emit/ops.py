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
    KernelDynamicScheduleProgram,
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
        if self.kernel.fusion_kind in {
            STATEFUL_GRAPH_FUSION,
            COEVOLVING_GRAPH_FUSION,
        }:
            for op in self.kernel.ops:
                if op.kind == "InitializeState":
                    self._emit_initialize_state()
                elif op.kind == "InitializeEffectiveParameter":
                    self._emit_initialize_effective_parameter(op)
                elif op.kind == "ForTrials":
                    self._emit_trial_loop(tuple(op.attrs["body"]))
                else:
                    raise ValueError(
                        f"Unsupported stateful top-level op '{op.kind}'."
                    )
            return

        self._emit_ops(self.kernel.ops)

    def _emit_step_mechanism(
        self,
        op: KernelOp,
        spec,
        step_var: str,
        finished_var: str,
        *,
        require_outputs: bool = False,
        model_outputs=None,
        readout_after_step: bool = False,
    ) -> None:
        node = self.graph.node(op.target)
        input_values = self._get_value(op.inputs[0].name)
        if model_outputs is None:
            model_outputs = op.outputs
        output_vars = []
        for output in model_outputs:
            output_vars.extend(self._component_vars(output.name, output.width))
        result = spec.step_emit(
            TritonEmitContext(self), node, input_values, output_vars, step_var, finished_var
        )
        # A non-terminator stepper (e.g. LCA) returns its current outputs; the
        # terminator (e.g. DDM) only advances state and returns None — its
        # outputs are produced by the readout after the loop.
        if readout_after_step:
            if spec.readout_emit is None:
                raise ValueError(
                    f"Scheduled terminator '{op.target}' has no readout emitter."
                )
            spec.readout_emit(TritonEmitContext(self), node, output_vars)
            values = output_vars
        elif result is not None:
            try:
                values = list(result)
            except TypeError as error:
                raise ValueError(
                    f"One-step mechanism '{op.target}' returned a non-iterable "
                    "result."
                ) from error
        elif require_outputs:
            raise ValueError(
                f"Scheduled one-step mechanism '{op.target}' did not return its "
                "declared outputs."
            )
        else:
            return

        expected_values = sum(output.width for output in model_outputs)
        if len(values) != expected_values:
            raise ValueError(
                f"One-step mechanism '{op.target}' returned {len(values)} "
                f"value(s), expected {expected_values}."
            )
        cursor = 0
        for output in model_outputs:
            self._set_value(output.name, values[cursor : cursor + output.width])
            cursor += output.width

    def _emit_trial_loop(self, body: tuple[KernelOp, ...]) -> None:
        self.builder.line("trial_idx = 0")
        with self.builder.block("while trial_idx < num_trials"):
            # Scalar parameters remain lane-persistent.  The strides are
            # constexpr launch metadata, so these branches and all scalar
            # reloads disappear from the compiled kernel.
            self._emit_params(trial_varying_only=True)
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
        elif op.kind == "AffineSchedulerValue":
            self._emit_affine_scheduler_value(op)
        elif op.kind == "CallMechanism":
            self._emit_mechanism_call(op)
        elif op.kind == "ApplyModulation":
            self._emit_apply_modulation(op)
        elif op.kind == "StepMechanism":
            self._emit_scheduled_step_mechanism(op)
        elif op.kind == "ResetState":
            self._emit_reset_state(op)
        elif op.kind == "StoreOutput":
            self._emit_store_output(op)
        elif op.kind == "StoreFlag":
            self._emit_store_flag(op)
        elif op.kind == "ForPasses":
            self._emit_for_passes(op)
        elif op.kind == "ExecuteConsiderationSet":
            self._emit_precomputed_consideration_set(op)
        else:
            raise ValueError(f"Unsupported Triton KernelIR op '{op.kind}'.")

    def _emit_scheduled_step_mechanism(self, op: KernelOp) -> None:
        spec = self._spec_for_op(op)
        if not spec.can_step:
            raise ValueError(
                "Batched op spec for scheduled mechanism "
                f"'{op.target}' has no one-step Triton implementation."
            )
        if op.attrs.get("active_lanes") == "parent_member_predicate":
            self._emit_dynamic_step_mechanism(op, spec)
            self.builder.line()
            return
        # A precomputed trace is lane-invariant: every in-range lane executes
        # this occurrence.  The existing one-step adapter accepts a 0/1
        # finished mask (0 means active), so pass an all-zero tensor.  Dynamic
        # schedules take the member-masked path above.
        self._emit_step_mechanism(
            op,
            spec,
            str(op.attrs["execution_index"]),
            "tl.zeros((BLOCK,), tl.float32)",
            require_outputs=True,
        )
        self.builder.line()

    def _emit_dynamic_step_mechanism(self, op: KernelOp, spec) -> None:
        """Emit one member-masked step with explicit state candidates.

        Persistent and per-trial state candidates are rebound into the
        component-owned step adapter.  Nothing is committed here: the enclosing
        consideration set publishes candidates after every member has evaluated
        the same frozen snapshot.
        """

        state_ids = tuple(op.attrs["state_ids"])
        trial_state_ids = tuple(op.attrs["trial_state_ids"])
        finished_trial_state_id = op.attrs["finished_trial_state_id"]
        rng_stream_ids = tuple(op.attrs["rng_stream_ids"])
        sampled_effective_parameter_ids = tuple(
            op.attrs.get("sampled_effective_parameter_ids", ())
        )
        state_count = len(state_ids) + len(trial_state_ids)
        effective_count = len(sampled_effective_parameter_ids)
        sampled_effective_inputs = tuple(op.inputs[1 : 1 + effective_count])
        state_inputs = tuple(op.inputs[1 + effective_count :])
        if (
            len(sampled_effective_inputs) != effective_count
            or len(state_inputs) != state_count
            or len(op.outputs) <= state_count
        ):
            raise ValueError(
                "Triton dynamic StepMechanism requires exact sampled-effective, "
                "persistent-state, and per-trial-state input suffixes."
            )

        state_output_start = len(op.outputs) - state_count
        leading_outputs = tuple(op.outputs[:state_output_start])
        state_outputs = tuple(op.outputs[state_output_start:])
        if finished_trial_state_id is None:
            model_outputs = leading_outputs
            finished_output = None
        else:
            if len(leading_outputs) < 2:
                raise ValueError(
                    "Triton dynamic terminator requires model outputs followed "
                    "by one explicit finished candidate."
                )
            model_outputs = leading_outputs[:-1]
            finished_output = leading_outputs[-1]

        persistent_inputs = state_inputs[:len(state_ids)]
        trial_inputs = state_inputs[len(state_ids):]
        persistent_outputs = state_outputs[:len(state_ids)]
        trial_outputs = state_outputs[len(state_ids):]
        states_by_id = {state.state_id: state for state in self.kernel.states}
        if self.dynamic_program is None or self.dynamic_slot_vars is None:
            raise ValueError(
                "Triton dynamic StepMechanism has no active typed program."
            )
        trial_carries = {
            (carry.owner_component_id, carry.value_id): carry
            for carry in self.dynamic_program.loop_carries
            if carry.kind == "trial_state"
        }
        effective_carries = {
            carry.value_id: carry
            for carry in self.dynamic_program.loop_carries
            if carry.kind == "effective_parameter"
        }

        sampled_effective_values = {}
        effective_parameters = {
            parameter.effective_parameter_id: parameter
            for parameter in self.kernel.effective_parameters
        }
        for effective_id, effective_input in zip(
            sampled_effective_parameter_ids,
            sampled_effective_inputs,
        ):
            try:
                parameter = effective_parameters[effective_id]
                carry = effective_carries[effective_id]
                storage_var = self.effective_parameter_vars[effective_id]
            except KeyError as error:
                raise ValueError(
                    "Triton dynamic StepMechanism samples an undeclared "
                    f"effective parameter ID {effective_id}."
                ) from error
            input_values = self._get_value(effective_input.name)
            if (
                parameter.target_component_id != op.attrs["component_id"]
                or carry.owner_component_id != op.attrs["component_id"]
                or carry.value.name != effective_input.name
                or carry.value.width != 1
                or effective_input.width != 1
                or input_values != [storage_var]
            ):
                raise ValueError(
                    "Triton dynamic StepMechanism sampled effective parameter "
                    "does not match its typed target carry."
                )
            sampled_effective_values[effective_id] = storage_var

        saved_state_vars = {}
        new_state_keys = set()
        trial_candidates = {}

        def bind_candidate(state_name, width, state_input, state_output):
            current_values = self._get_value(state_input.name)
            candidate_values = self._component_vars(
                state_output.name,
                state_output.width,
            )
            if len(current_values) != width or state_output.width != width:
                raise ValueError(
                    "Triton dynamic StepMechanism state width does not match "
                    "its explicit candidate."
                )
            for index, (candidate, current) in enumerate(
                zip(candidate_values, current_values)
            ):
                self.builder.line(f"{candidate} = {current}")
                key = (state_name, index)
                if key in self.state_vars:
                    saved_state_vars[key] = self.state_vars[key]
                else:
                    new_state_keys.add(key)
                self.state_vars[key] = candidate
            self._set_value(state_output.name, candidate_values)
            return candidate_values

        saved_sampled_effective = self.dynamic_sampled_effective_parameters
        saved_consumed_effective = self.dynamic_consumed_effective_parameter_ids
        try:
            self.dynamic_sampled_effective_parameters = sampled_effective_values
            self.dynamic_consumed_effective_parameter_ids = set()
            for state_id, state_input, state_output in zip(
                state_ids,
                persistent_inputs,
                persistent_outputs,
            ):
                try:
                    state = states_by_id[state_id]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic StepMechanism references undeclared "
                        f"persistent state ID {state_id}."
                    ) from error
                if state.component_id != op.attrs["component_id"]:
                    raise ValueError(
                        "Triton dynamic persistent state is owned by another "
                        "component."
                    )
                bind_candidate(
                    state.name,
                    state.width,
                    state_input,
                    state_output,
                )

            node = self.graph.node(op.target)
            for trial_state_id, state_input, state_output in zip(
                trial_state_ids,
                trial_inputs,
                trial_outputs,
            ):
                try:
                    carry = trial_carries[
                        (op.attrs["component_id"], trial_state_id)
                    ]
                    declaration = spec.trial_states[trial_state_id]
                except (KeyError, IndexError) as error:
                    raise ValueError(
                        "Triton dynamic StepMechanism references undeclared "
                        f"per-trial state ID {trial_state_id}."
                    ) from error
                width = declaration.width or node.output_width
                if carry.value.name != state_input.name or carry.value.width != width:
                    raise ValueError(
                        "Triton dynamic per-trial state does not match its typed "
                        "carry declaration."
                    )
                trial_candidates[trial_state_id] = bind_candidate(
                    f"{node.name}.{declaration.name}",
                    width,
                    state_input,
                    state_output,
                )

            if self.dynamic_execution_index is None:
                raise ValueError(
                    "Triton dynamic StepMechanism has no parent component clock."
                )
            step_index = self.dynamic_execution_index
            if rng_stream_ids:
                rng_clocks = tuple(
                    self._dynamic_slot_var(
                        self.dynamic_slot_vars,
                        "rng_clock",
                        owner=op.attrs["component_id"],
                        rng_stream=stream_id,
                    )
                    for stream_id in rng_stream_ids
                )
                # All streams owned by one scheduled execution advance together;
                # the component step consumes their exact shared pre-increment
                # clock while stream bases keep the random draws disjoint.
                step_index = rng_clocks[0]

            finished_var = (
                f"{component_symbol(self.graph, op.target)}_dynamic_member_finished"
            )
            self.builder.line(
                f"{finished_var} = tl.where({self.dynamic_active_mask}, 0.0, 1.0)"
            )
            self._emit_step_mechanism(
                op,
                spec,
                step_index,
                finished_var,
                require_outputs=True,
                model_outputs=model_outputs,
                readout_after_step=finished_output is not None,
            )

            if finished_output is not None:
                try:
                    finished_state = trial_candidates[finished_trial_state_id]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic finished candidate has no matching "
                        "per-trial state."
                    ) from error
                if len(finished_state) != 1:
                    raise ValueError(
                        "Triton dynamic finished state must be scalar."
                    )
                finished_values = self._component_vars(
                    finished_output.name,
                    finished_output.width,
                )
                self.builder.line(
                    f"{finished_values[0]} = {finished_state[0]} != 0.0"
                )
                self._set_value(finished_output.name, finished_values)
            if self.dynamic_consumed_effective_parameter_ids != set(
                sampled_effective_parameter_ids
            ):
                raise ValueError(
                    "Triton dynamic StepMechanism did not consume every declared "
                    "sampled effective parameter."
                )
        finally:
            self.dynamic_sampled_effective_parameters = saved_sampled_effective
            self.dynamic_consumed_effective_parameter_ids = saved_consumed_effective
            self.state_vars.update(saved_state_vars)
            for key in new_state_keys:
                self.state_vars.pop(key, None)

    def _emit_for_passes(self, op: KernelOp) -> None:
        if op.attrs.get("trace_kind") == "lane_local_dynamic":
            self._emit_lane_local_dynamic_passes(op)
            return
        self._emit_precomputed_passes(op)

    @staticmethod
    def _dynamic_slot_key(slot):
        return (
            slot.kind,
            slot.owner_component_id,
            slot.producer_component_id,
            slot.consumer_component_id,
            slot.finished_value_id,
            slot.rng_stream_id,
        )

    @staticmethod
    def _dynamic_carry_key(carry):
        return (carry.kind, carry.owner_component_id, carry.value_id)

    def _emit_lane_local_dynamic_passes(self, op: KernelOp) -> None:
        """Emit one typed, lane-local consideration-set scheduler."""

        program = op.attrs.get("program")
        if type(program) is not KernelDynamicScheduleProgram:
            raise ValueError(
                "Triton lane-local dynamic ForPasses requires a typed program."
            )

        outer_values = dict(self.value_vars)
        carry_vars = self._emit_dynamic_carry_initializers(program)
        slot_vars = self._emit_dynamic_scheduler_initializers(program)
        self.dynamic_program = program
        self.dynamic_slot_vars = slot_vars
        done_var = "dynamic_done"
        round_var = "dynamic_round"
        self.builder.line(
            f"{done_var} = tl.zeros((BLOCK,), dtype=tl.int32)"
        )
        self.builder.line(f"{round_var} = 0")
        with self.builder.block(
            f"while ({round_var} < {program.schedule_fuel}) & "
            f"(tl.max(tl.where(mask & ({done_var} == 0), 1, 0)) > 0)"
        ):
            for consideration_set in program.consideration_sets:
                self._emit_dynamic_termination(program, slot_vars, done_var)
                self._emit_dynamic_consideration_set(
                    program,
                    consideration_set,
                    slot_vars,
                    carry_vars,
                    done_var,
                )
            self._emit_dynamic_termination(program, slot_vars, done_var)
            pass_var = self._dynamic_slot_var(
                slot_vars,
                "pass_index",
            )
            self.builder.line(
                f"{pass_var} = tl.where(mask & ({done_var} == 0), "
                f"{pass_var} + 1, {pass_var})"
            )
            self.builder.line(f"{round_var} += 1")

        # Exhaustion is a region-level safety result, not a fabricated
        # scheduler `finished` value.  Exactly-at-cap lanes have already
        # visited every set in the final round and therefore reach `done` when
        # their later WhenFinished member executes.
        for carry in program.loop_carries:
            if carry.kind != "diagnostic":
                continue
            values = carry_vars[self._dynamic_carry_key(carry)]
            if carry.value.width != 1 or len(values) != 1:
                raise ValueError(
                    "Triton dynamic exhaustion diagnostics must be scalar."
                )
            finished_var = self._dynamic_slot_var(
                slot_vars,
                "finished",
                owner=carry.owner_component_id,
                finished=carry.value_id,
            )
            self.builder.line(
                f"{values[0]} = tl.where(mask & ({finished_var} == 0), "
                "1.0, 0.0)"
            )

        self.value_vars.clear()
        self.value_vars.update(outer_values)
        for carry in program.loop_carries:
            values = carry_vars[self._dynamic_carry_key(carry)]
            self._set_value(carry.value.name, values)
        for output in op.outputs:
            if output.name not in self.value_vars:
                raise ValueError(
                    "Triton dynamic ForPasses output is not a declared loop carry: "
                    f"'{output.name}'."
                )
        self.dynamic_active_mask = "mask"
        self.dynamic_execution_index = None
        self.dynamic_program = None
        self.dynamic_slot_vars = None
        self.builder.line()

    def _emit_dynamic_carry_initializers(self, program):
        carry_vars = {}
        states_by_id = {state.state_id: state for state in self.kernel.states}
        for carry in program.loop_carries:
            key = self._dynamic_carry_key(carry)
            if carry.kind == "state":
                try:
                    state = states_by_id[carry.value_id]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic carry references undeclared state ID "
                        f"{carry.value_id}."
                    ) from error
                values = [
                    self.state_vars[(state.name, index)]
                    for index in range(state.width)
                ]
            elif carry.kind == "trial_state":
                base = f"{safe_ident(carry.value.name)}_dynamic_current"
                values = [
                    f"{base}_{index}" for index in range(carry.value.width)
                ]
                if carry.initial_parameter_id is not None:
                    try:
                        parameter = self.kernel.params[carry.initial_parameter_id]
                        initial_var = self.param_vars[parameter.name]
                    except (IndexError, KeyError) as error:
                        raise ValueError(
                            "Triton dynamic per-trial state references an "
                            "unbound initializer parameter."
                        ) from error
                    if (
                        parameter.parameter_id != carry.initial_parameter_id
                        or parameter.owner_component_id
                        != carry.owner_component_id
                    ):
                        raise ValueError(
                            "Triton dynamic per-trial state initializer has the "
                            "wrong typed parameter owner."
                        )
                    for value in values:
                        self.builder.line(f"{value} = {initial_var}")
                else:
                    for value, initial in zip(values, carry.initial_value):
                        self.builder.line(
                            f"{value} = tl.full((BLOCK,), "
                            f"{float_literal(initial)}, tl.float32)"
                        )
            elif carry.kind == "effective_parameter":
                try:
                    values = [self.effective_parameter_vars[carry.value_id]]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic carry references uninitialized effective "
                        f"parameter ID {carry.value_id}."
                    ) from error
            else:
                base = f"{safe_ident(carry.value.name)}_dynamic_current"
                values = [f"{base}_{index}" for index in range(carry.value.width)]
                for value in values:
                    self.builder.line(
                        f"{value} = tl.zeros((BLOCK,), dtype=tl.float32)"
                    )
            carry_vars[key] = values
            self._set_value(carry.value.name, values)
        self.builder.line()
        return carry_vars

    def _emit_dynamic_scheduler_initializers(self, program):
        slot_vars = {}
        finished_by_id = {
            value.value_id: value for value in self.kernel.finished_values
        }
        for slot in program.scheduler_state_slots:
            key = self._dynamic_slot_key(slot)
            value = self._component_vars(slot.value.name, 1)[0]
            if slot.initialization == "zero":
                self.builder.line(
                    f"{value} = tl.zeros((BLOCK,), dtype=tl.int32)"
                )
            elif slot.initialization == "count_zero_vs_effective_parameter":
                try:
                    finished = finished_by_id[slot.finished_value_id]
                    effective_parameter_id = (
                        slot.initial_effective_parameter_id
                    )
                    effective = self.effective_parameter_vars[
                        effective_parameter_id
                    ]
                except (KeyError, TypeError) as error:
                    raise ValueError(
                        "Triton dynamic finished initializer references an "
                        "undeclared effective parameter."
                    ) from error
                if (
                    finished.component_id != slot.owner_component_id
                    or finished.predicate_kind
                    != "execution_count_at_least_effective_parameter"
                    or finished.attrs.get("effective_parameter_id")
                    != effective_parameter_id
                ):
                    raise ValueError(
                        "Triton dynamic finished initializer does not match "
                        "its typed count-effective predicate."
                    )
                # PNL evaluates WhenFinished before the first owner execution
                # against count zero and the lane-persistent effective value.
                # Its minimum-one rule applies only after the owner executes.
                self.builder.line(
                    f"{value} = tl.where(mask & (0.0 >= {effective}), "
                    "1, 0)"
                )
            else:
                raise ValueError(
                    "Triton dynamic scheduler has an unsupported slot "
                    f"initializer '{slot.initialization}'."
                )
            slot_vars[key] = value
            self._set_value(slot.value.name, [value])
        self.builder.line()
        return slot_vars

    @staticmethod
    def _dynamic_slot_var(
        slot_vars,
        kind,
        *,
        owner=None,
        producer=None,
        consumer=None,
        finished=None,
        rng_stream=None,
    ):
        key = (kind, owner, producer, consumer, finished, rng_stream)
        try:
            return slot_vars[key]
        except KeyError as error:
            raise ValueError(
                f"Triton dynamic scheduler has no declared slot {key}."
            ) from error

    def _emit_dynamic_termination(self, program, slot_vars, done_var: str) -> None:
        has_run = [
            self._dynamic_slot_var(slot_vars, "has_run", owner=component_id)
            for component_id in program.trial_termination.dependency_component_ids
        ]
        expression = " & ".join(f"({value} != 0)" for value in has_run)
        if not expression:
            raise ValueError(
                "Triton dynamic AllHaveRun termination requires dependencies."
            )
        self.builder.line(
            f"{done_var} = tl.where(mask & ({expression}), 1, {done_var})"
        )

    def _emit_dynamic_consideration_set(
        self,
        program,
        consideration_set,
        slot_vars,
        carry_vars,
        done_var: str,
    ) -> None:
        set_id = consideration_set.consideration_set_id
        component_ids = tuple(member.component_id for member in consideration_set.members)
        self.builder.line(
            f"# dynamic scheduler consideration set {set_id}, components {component_ids}"
        )
        frozen_values = dict(self.value_vars)
        budgets_by_component = {
            budget.component_id: budget
            for budget in program.execution_budgets
        }
        member_masks = {}
        for member in consideration_set.members:
            predicate = self._dynamic_predicate_expression(
                member.predicate,
                slot_vars,
                member.component_id,
            )
            count_var = self._dynamic_slot_var(
                slot_vars,
                "execution_count",
                owner=member.component_id,
            )
            try:
                budget = budgets_by_component[member.component_id]
            except KeyError as error:
                raise ValueError(
                    "Triton dynamic member has no execution budget."
                ) from error
            budget_gate = f"({count_var} < {budget.maximum})"
            if budget.post_finish != "unrestricted":
                finished_var = self._dynamic_slot_var(
                    slot_vars,
                    "finished",
                    owner=member.component_id,
                    finished=budget.finished_value_id,
                )
                unfinished_gate = f"({count_var} < {budget.unfinished_maximum})"
                if budget.post_finish == "continue":
                    budget_gate += (
                        f" & (({finished_var} != 0) | {unfinished_gate})"
                    )
                elif budget.post_finish == "stop":
                    budget_gate += (
                        f" & ({finished_var} == 0) & {unfinished_gate}"
                    )
                else:
                    raise ValueError(
                        "Triton dynamic execution budget has an unsupported "
                        f"post-finish policy '{budget.post_finish}'."
                    )
            mask_var = f"dynamic_s{set_id}_n{member.component_id}_active"
            self.builder.line(
                f"{mask_var} = mask & ({done_var} == 0) & ({predicate}) & "
                f"({budget_gate})"
            )
            member_masks[member.component_id] = mask_var

        member_values = {}
        for member in consideration_set.members:
            self.value_vars.clear()
            self.value_vars.update(frozen_values)
            self.dynamic_active_mask = member_masks[member.component_id]
            self.dynamic_execution_index = self._dynamic_slot_var(
                slot_vars,
                "execution_count",
                owner=member.component_id,
            )
            for child in member.body:
                self._emit_op(child)
            member_values[member.component_id] = dict(self.value_vars)

        self.value_vars.clear()
        self.value_vars.update(frozen_values)
        carries_by_key = {
            self._dynamic_carry_key(carry): carry
            for carry in program.loop_carries
        }
        for member in consideration_set.members:
            active = member_masks[member.component_id]
            values = member_values[member.component_id]
            for publication in member.publications:
                try:
                    sources = values[publication.source.name]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic publication does not resolve to its "
                        "declared member-local candidate."
                    ) from error
                if publication.kind == "finished":
                    if len(sources) != 1:
                        raise ValueError(
                            "Triton dynamic finished publication must be scalar."
                        )
                    destination = self._dynamic_slot_var(
                        slot_vars,
                        "finished",
                        owner=publication.owner_component_id,
                        finished=publication.value_id,
                    )
                    self.builder.line(
                        f"{destination} = tl.where({active}, "
                        f"tl.where({sources[0]}, 1, 0), {destination})"
                    )
                    continue
                key = (
                    publication.kind,
                    publication.owner_component_id,
                    publication.value_id,
                )
                try:
                    carry = carries_by_key[key]
                    destinations = carry_vars[key]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic publication does not resolve to its "
                        "declared candidate and carry."
                    ) from error
                if len(destinations) != len(sources):
                    raise ValueError(
                        "Triton dynamic publication candidate/carry widths differ."
                    )
                for destination, source in zip(destinations, sources):
                    self.builder.line(
                        f"{destination} = tl.where({active}, {source}, {destination})"
                    )
                self._set_value(carry.value.name, destinations)

        # Effects read the same frozen carry bank plus their owner's local
        # candidates, and become visible only after every member body has been
        # emitted.
        for member in consideration_set.members:
            if not member.effects:
                continue
            effect_values = dict(frozen_values)
            effect_values.update(member_values[member.component_id])
            self.value_vars.clear()
            self.value_vars.update(effect_values)
            self.dynamic_active_mask = member_masks[member.component_id]
            for effect in member.effects:
                self._emit_apply_modulation(effect)

        self.value_vars.clear()
        self.value_vars.update(frozen_values)
        for carry in program.loop_carries:
            self._set_value(
                carry.value.name,
                carry_vars[self._dynamic_carry_key(carry)],
            )
        self._emit_dynamic_scheduler_updates(
            program,
            consideration_set,
            slot_vars,
            member_masks,
        )
        self.dynamic_active_mask = "mask"
        self.dynamic_execution_index = None
        self.builder.line()

    def _dynamic_predicate_expression(
        self,
        predicate,
        slot_vars,
        consumer_component_id: int,
    ) -> str:
        if predicate.kind == "Always":
            return "True"
        if predicate.kind in {"AtPass", "AtTrialStart"}:
            pass_var = self._dynamic_slot_var(slot_vars, "pass_index")
            return f"{pass_var} == {predicate.pass_index}"
        if predicate.kind in {"EveryNCalls", "AllEveryNCalls"}:
            credits = [
                self._dynamic_slot_var(
                    slot_vars,
                    "usable_call",
                    producer=dependency_id,
                    consumer=consumer_component_id,
                )
                for dependency_id in predicate.dependency_component_ids
            ]
            return " & ".join(
                f"({credit} >= {predicate.call_count})" for credit in credits
            )
        if predicate.kind == "WhenFinished":
            return self._dynamic_slot_var(
                slot_vars,
                "finished",
                owner=predicate.dependency_component_ids[0],
                finished=predicate.finished_value_ids[0],
            ) + " != 0"
        raise ValueError(
            f"Unsupported Triton dynamic predicate '{predicate.kind}'."
        )

    def _emit_dynamic_scheduler_updates(
        self,
        program,
        consideration_set,
        slot_vars,
        member_masks,
    ) -> None:
        for member in consideration_set.members:
            active = member_masks[member.component_id]
            count_var = self._dynamic_slot_var(
                slot_vars,
                "execution_count",
                owner=member.component_id,
            )
            has_run_var = self._dynamic_slot_var(
                slot_vars,
                "has_run",
                owner=member.component_id,
            )
            self.builder.line(
                f"{count_var} = tl.where({active}, {count_var} + 1, {count_var})"
            )
            self.builder.line(
                f"{has_run_var} = tl.where({active}, 1, {has_run_var})"
            )
            for slot in program.scheduler_state_slots:
                if (
                    slot.kind == "rng_clock"
                    and slot.owner_component_id == member.component_id
                ):
                    rng_var = slot_vars[self._dynamic_slot_key(slot)]
                    self.builder.line(
                        f"{rng_var} = tl.where({active}, {rng_var} + 1, {rng_var})"
                    )

        for slot in program.scheduler_state_slots:
            if slot.kind != "usable_call":
                continue
            credit_var = slot_vars[self._dynamic_slot_key(slot)]
            produced = member_masks.get(slot.producer_component_id)
            consumed = member_masks.get(slot.consumer_component_id)
            produced_delta = f"tl.where({produced}, 1, 0)" if produced else "0"
            consumed_delta = f"tl.where({consumed}, 1, 0)" if consumed else "0"
            self.builder.line(
                f"{credit_var} = {credit_var} + {produced_delta} - {consumed_delta}"
            )

        finished_by_id = {
            value.value_id: value for value in self.kernel.finished_values
        }
        published_finished = {
            (publication.owner_component_id, publication.value_id)
            for item in program.consideration_sets
            for member in item.members
            for publication in member.publications
            if publication.kind == "finished"
        }
        for slot in program.scheduler_state_slots:
            if slot.kind != "finished":
                continue
            if (slot.owner_component_id, slot.finished_value_id) in published_finished:
                # Stateful/dynamic finished values are committed by their owner
                # at the same deferred publication boundary as its outputs and
                # trial state.  Re-deriving them from execution count here would
                # overwrite the modeled termination result.
                continue
            owner_active = member_masks.get(slot.owner_component_id)
            if owner_active is None:
                # Count-derived is_finished changes only when its owner
                # executes and samples its effective parameter.  In
                # particular, preserve the typed count-zero value through
                # earlier consideration sets at trial start.
                continue
            try:
                finished = finished_by_id[slot.finished_value_id]
            except KeyError as error:
                raise ValueError(
                    "Triton dynamic finished slot references an undeclared value."
                ) from error
            count_var = self._dynamic_slot_var(
                slot_vars,
                "execution_count",
                owner=slot.owner_component_id,
            )
            if finished.predicate_kind == "execution_count_at_least":
                required = str(finished.attrs["count"])
            elif (
                finished.predicate_kind
                == "execution_count_at_least_effective_parameter"
            ):
                effective_parameter_id = finished.attrs["effective_parameter_id"]
                try:
                    effective = self.effective_parameter_vars[effective_parameter_id]
                except KeyError as error:
                    raise ValueError(
                        "Triton dynamic finished value references an uninitialized "
                        "effective parameter."
                    ) from error
                required = (
                    "tl.minimum(tl.maximum(tl.ceil("
                    f"{effective}), {float_literal(finished.attrs['minimum'])}), "
                    f"{float_literal(finished.attrs['maximum'])})"
                )
            else:
                raise ValueError(
                    "Triton dynamic scheduler cannot evaluate finished predicate "
                    f"'{finished.predicate_kind}'."
                )
            finished_var = slot_vars[self._dynamic_slot_key(slot)]
            self.builder.line(
                f"{finished_var} = tl.where({owner_active}, "
                f"tl.where({count_var} >= {required}, 1, 0), {finished_var})"
            )

    def _emit_precomputed_passes(self, op: KernelOp) -> None:
        if op.attrs.get("declaration_only") is not False:
            raise ValueError(
                "Cannot emit a declaration-only KernelIR ForPasses region."
            )
        if op.attrs.get("trace_kind") != "precomputed":
            raise ValueError(
                "Triton only emits typed precomputed KernelIR ForPasses regions."
            )
        for child in op.attrs["body"]:
            if child.kind != "ExecuteConsiderationSet":
                raise ValueError(
                    "Executable KernelIR ForPasses contains a non-consideration-set "
                    f"op '{child.kind}'."
                )
            self._emit_op(child)

    def _emit_precomputed_consideration_set(self, op: KernelOp) -> None:
        pass_index = op.attrs["pass_index"]
        consideration_set_id = op.attrs["consideration_set_id"]
        component_ids = op.attrs["component_ids"]
        self.builder.line(
            "# precomputed scheduler pass "
            f"{pass_index}, consideration set {consideration_set_id}, "
            f"components {component_ids}"
        )
        for child in op.attrs["body"]:
            if child.kind in {"ForPasses", "ExecuteConsiderationSet"}:
                raise ValueError(
                    "Nested precomputed scheduler regions are unsupported in "
                    "ExecuteConsiderationSet."
                )
            self._emit_op(child)

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
        if op.attrs.get("spec_key") == "":
            self._emit_identity_function_call(op, node)
            return
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
            if self.dynamic_execution_index is not None:
                # In a dynamic region the scheduler owns delayed onset.  Keep
                # the absorbed fires-once integrator lane-local to the selected
                # member; publication retains the previous output elsewhere.
                input_values = [
                    f"tl.where({self.dynamic_active_mask}, {value}, 0.0)"
                    for value in input_values
                ]
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

    def _emit_affine_scheduler_value(self, op: KernelOp) -> None:
        """Evaluate a folded affine controller execution ordinal."""

        if len(op.inputs) != 1 or len(op.outputs) != 1:
            raise ValueError(
                "Triton AffineSchedulerValue requires one scheduler input and "
                "one output."
            )
        count_values = self._get_value(op.inputs[0].name)
        if len(count_values) != 1 or op.inputs[0].width != 1 or op.outputs[0].width != 1:
            raise ValueError(
                "Triton AffineSchedulerValue requires scalar typed values."
            )
        parameters = {
            parameter.parameter_id: parameter for parameter in self.kernel.params
        }
        try:
            base_parameter = parameters[op.attrs["base_parameter_id"]]
            delta_parameter = parameters[op.attrs["delta_parameter_id"]]
            base_var = self.param_vars[base_parameter.name]
            delta_var = self.param_vars[delta_parameter.name]
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Triton AffineSchedulerValue references an unbound parameter."
            ) from error
        output_var = self._component_vars(
            op.outputs[0].name,
            op.outputs[0].width,
        )[0]
        # The folded source is a reset-each-trial SimpleIntegrator.  Its first
        # execution returns ``base + delta``; the scheduler slot is the
        # controller's pre-execution count, so add one to form that ordinal.
        self.builder.line(
            f"{output_var} = {base_var} + {delta_var} * "
            f"({count_values[0]} + 1)"
        )
        self._set_value(op.outputs[0].name, [output_var])
        self.builder.line()

    def _emit_identity_function_call(self, op: KernelOp, node) -> None:
        """Emit the sole keyless function form: an absorbed Identity control."""

        if (
            node.component_type != "ControlMechanism"
            or node.function_type != "Identity"
            or node.attrs.get("control_function") != "identity"
            or op.attrs.get("component_type") != "ControlMechanism"
            or op.attrs.get("function_type") != "Identity"
            or op.attrs.get("params") != {}
            or node.params
            or len(op.inputs) != 1
            or len(op.outputs) != 1
            or op.inputs[0].width != op.outputs[0].width
        ):
            raise ValueError(
                "Triton only accepts an authenticated absorbed Identity "
                "ControlMechanism without a registry key."
            )
        input_values = self._get_value(op.inputs[0].name)
        output_vars = self._component_vars(op.outputs[0].name, op.outputs[0].width)
        for input_value, output_var in zip(input_values, output_vars):
            self.builder.line(f"{output_var} = {input_value}")
        self._set_value(op.outputs[0].name, output_vars)
        self.builder.line()

    def _emit_apply_modulation(self, op: KernelOp) -> None:
        """Assign an OVERRIDE into real, outer-scope effective storage."""

        effective_parameter_id = op.attrs["effective_parameter_id"]
        try:
            storage_var = self.effective_parameter_vars[effective_parameter_id]
        except KeyError as error:
            raise ValueError(
                "Triton ApplyModulation has no dominating effective-parameter "
                f"initializer for id {effective_parameter_id}."
            ) from error
        held_values = self._get_value(op.inputs[0].name)
        controller_values = self._get_value(op.inputs[1].name)
        if (
            op.attrs.get("mode") != "OVERRIDE"
            or held_values != [storage_var]
            or len(controller_values) != 1
        ):
            raise ValueError(
                "Triton ApplyModulation requires exact scalar OVERRIDE storage."
            )
        self.builder.line(
            f"{storage_var} = tl.where("
            f"{self.dynamic_active_mask}, {controller_values[0]}, {storage_var})"
        )
        self._set_value(op.outputs[0].name, [storage_var])
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
                args.append(self.dynamic_active_mask)
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
