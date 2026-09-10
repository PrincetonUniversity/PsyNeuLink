"""Generated observed-event replay and instrumentation of the source scheduler.

Only the checked stochastic StepMechanism is replaced. All deterministic
component bodies, resets, publication order, and held-control updates use the
ordinary emitter. Placeholder stochastic outputs are never returned as data;
the planner rejects their influence on carried state or controls.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.emit.emitter import TritonGraphEmitter
from psyneulink.core.batched.history import HistoryReplayError, HistoryTrace, replay_program
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs


class HistoryTraceEmitter(TritonGraphEmitter):
    def __init__(self, kernel, witness, *, replay):
        super().__init__(kernel)
        self.witness = witness
        self.replay = replay
        self.program = replay_program(kernel)
        self.history_width = self.state_width + len(kernel.effective_parameters)

    def _signature_args(self):
        args = list(super()._signature_args())
        index = args.index("LCA_MAX_STEPS")
        args[index:index] = ["observed_counts", "history_values", "history_calls", "history_status"]
        return tuple(args)

    def _emit_history_state(self, position):
        values = [self.state_vars[(state.name, index)]
                  for state in self.kernel.states for index in range(state.width)]
        values.extend(self.effective_parameter_vars[item.effective_parameter_id]
                      for item in self.kernel.effective_parameters)
        for index, value in enumerate(values):
            self.builder.line(
                f"tl.store(history_values + (history_trial * 2 + {position}) * "
                f"{self.history_width} + {index}, {value}, mask=mask)"
            )

    def _emit_trial_start_inspection(self):
        self.builder.line(f"history_trial = {self._history_trial_index()}")
        if self.replay:
            self.builder.line("history_target = tl.load(observed_counts + history_trial, mask=mask, other=1)")
        for component in self.witness.component_ids:
            self.builder.line(f"history_calls_{component} = tl.zeros((BLOCK,), tl.int32)")
        self._emit_history_state(0)

    def _history_trial_index(self):
        return "offsets * num_trials + trial_idx"

    def _dynamic_member_replay_gate(self, member, slot_vars):
        if self.replay and member.component_id == self.witness.zero_step_gate_component:
            finished = self._dynamic_slot_var(slot_vars, "finished", owner=member.component_id,
                                              finished=self.witness.zero_step_gate_finished)
            declaration = next(item for item in self.kernel.finished_values
                               if item.value_id == self.witness.zero_step_gate_finished)
            effective = self.effective_parameter_vars[declaration.attrs["effective_parameter_id"]]
            count = self._dynamic_slot_var(slot_vars, "execution_count", owner=member.component_id)
            # The source's finished slot may still reflect the previous trial's
            # held value. Projected zero histories use the current prelude
            # control, including a zero published by an AtPass(0) controller.
            required = f"tl.where({count} == 0, {effective}, tl.maximum(tl.ceil({effective}), {declaration.attrs['minimum']}))"
            self.builder.line(f"{finished} = tl.where(history_target == 0, ({count} >= {required}).to(tl.int32), {finished})")
            # Complete positive settling, but do not perform a post-settling
            # persistent update for a zero-count event. In particular, an
            # initially finished (zero-settling) gate executes no state step.
            return f" & ~((history_target == 0) & ({finished} != 0))"
        return ""

    def _emit_trial_end_inspection(self):
        self._emit_history_state(1)
        for index, component in enumerate(self.witness.component_ids):
            self.builder.line(
                f"tl.store(history_calls + history_trial * {len(self.witness.component_ids)} + "
                f"{index}, history_calls_{component}, mask=mask)"
            )

    def _emit_dynamic_scheduler_updates(self, program, consideration_set, slot_vars, member_masks, has_run_bits):
        super()._emit_dynamic_scheduler_updates(program, consideration_set, slot_vars, member_masks, has_run_bits)
        for member in consideration_set.members:
            component = member.component_id
            if self.replay and component == self.witness.zero_step_gate_component:
                finished = self._dynamic_slot_var(slot_vars, "finished", owner=component,
                                                  finished=self.witness.zero_step_gate_finished)
                word, bit = has_run_bits[component]
                # A zero-length prelude is virtually completed, not executed.
                # Satisfy AllHaveRun without incrementing calls or state. This
                # is part of the declared projected-history policy only.
                self.builder.line(f"{word} = tl.where(mask & (history_target == 0) & ({finished} != 0), {word} | {bit}, {word})")
            self.builder.line(
                f"history_calls_{component} += tl.where({member_masks[component]}, 1, 0)"
            )

    def _emit_lane_local_dynamic_passes(self, op):
        super()._emit_lane_local_dynamic_passes(op)
        spec = self.kernel.op_specs.lookup_spec(self.witness.endpoint.clock_spec_key)
        counter_id = next(index for index, state in enumerate(spec.trial_states)
                          if state.name == self.witness.endpoint.counter_state)
        carry = next(carry for carry in self.program.loop_carries
                     if carry.kind == "trial_state" and carry.owner_component_id == self.witness.endpoint.clock_component_id
                     and carry.value_id == counter_id)
        count = self._get_value(carry.value.name)[0]
        for index, value in enumerate(("dynamic_round", "dynamic_done", count)):
            self.builder.line(f"tl.store(history_status + history_trial * 3 + {index}, {value}, mask=mask)")

    def _emit_dynamic_step_mechanism(self, op, spec):
        if not self.replay or op.attrs["component_id"] != self.witness.endpoint.clock_component_id:
            return super()._emit_dynamic_step_mechanism(op, spec)
        # Contract: trial-local zero counter, one active step per scheduled
        # invocation, and absorbing finished state. State/publication masks and
        # scheduling remain owned by the ordinary dynamic scheduler emitter.
        state_ids = op.attrs["trial_state_ids"]
        state_count = len(state_ids)
        state_inputs = op.inputs[-state_count:]
        state_outputs = op.outputs[-state_count:]
        model_outputs = op.outputs[:-state_count - 1]
        finished_output = op.outputs[-state_count - 1]
        count_index = next(index for index, state in enumerate(spec.trial_states)
                           if state.name == self.witness.endpoint.counter_state)
        previous = self._get_value(state_inputs[count_index].name)[0]
        next_count = "history_event_next_count"
        self.builder.line(f"{next_count} = tl.minimum({previous} + 1.0, history_target.to(tl.float32))")
        for declaration, source, output in zip(spec.trial_states, state_inputs, state_outputs):
            if declaration.name == self.witness.endpoint.counter_state:
                values = [next_count]
            elif declaration.name == spec.finished_output:
                values = [f"({next_count} >= history_target)"]
            else:
                values = self._get_value(source.name)
            self._set_value(output.name, values)
        self._set_value(finished_output.name, [f"({next_count} >= history_target)"])
        node = self.graph.node(op.target)
        readout = spec.likelihood_contract.event_readout
        for declaration, output in zip(spec.outputs, model_outputs):
            values = self._component_vars(output.name, output.width)
            if declaration.port == readout.output_port:
                offset = self.param_vars[node.params[readout.offset_parameter]]
                step = self.param_vars[node.params[readout.step_parameter]]
                self.builder.line(f"{values[0]} = {offset} + {next_count} * {step}")
            else:
                for value in values:
                    self.builder.line(f"{value} = tl.zeros((BLOCK,), tl.float32)")
            self._set_value(output.name, values)


def run_history_trace(plan, inputs, parameter_sets, *, counts=None, seed=0):
    """Checked scheduler execution on the plan's interpreter or GPU backend."""
    return _run_history_trace(plan, inputs, parameter_sets, counts=counts, seed=seed)


def _run_history_trace(
    plan, inputs, parameter_sets, *, counts=None, seed=0,
    source_override=None, extra_kernel_args=(), parallel_trial_lanes=False,
):
    """Shared private launcher for validated history/path inspection programs."""
    from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
    from psyneulink.core.batched.backend.triton.runtime import (
        _import_torch_triton, _normalize_launch_options, _report_truncation, _run_stateful_graph_kernel,
    )

    simulation = plan.simulation_plan
    if simulation.backend not in ("triton_cpu", "triton"):
        raise HistoryReplayError("history.backend_unsupported", "History execution requires a Triton simulation plan.")
    interpret = simulation.backend == "triton_cpu"
    device = "cpu" if interpret else "cuda"
    ir = simulation.ir
    rows = normalize_parameter_sets(parameter_sets, ir)
    if not rows:
        raise HistoryReplayError("history.empty_candidates", "At least one parameter candidate is required.")
    prepared = prepare_inputs(ir, inputs, parameter_sets=rows, component_bindings=simulation.component_bindings)
    first = next(iter(prepared.values()))
    subjects, trials = first.shape[:2]
    if subjects != 1 or trials == 0:
        raise HistoryReplayError("history.subject_layout", "Reference replay requires one nonempty, contiguous subject.")
    if any(not np.all(np.isfinite(value)) for value in prepared.values()):
        raise HistoryReplayError("history.input_nonfinite", "History inputs must be finite.")
    torch, triton = _import_torch_triton(interpret)
    width = sum(state.width for state in ir.graph.states)
    history_width = width + len(simulation.kernel_ir.effective_parameters)
    shape = (len(rows), trials)
    history = torch.empty((*shape, 2, history_width), dtype=torch.float32, device=device)
    calls = torch.empty((*shape, len(plan.witness.component_ids)), dtype=torch.int32, device=device)
    status = torch.empty((*shape, 3), dtype=torch.int32, device=device)
    observed = torch.ones(shape, dtype=torch.int32, device=device) if counts is None else torch.tensor(np.array(counts), dtype=torch.int32, device=device)
    if tuple(observed.shape) != shape:
        raise HistoryReplayError("history.count_shape", "Observed counts must match candidate and trial axes.")
    source = plan.source(replay=counts is not None) if source_override is None else source_override
    with interpret_scope(interpret):
        module = load_triton_kernel_module(source, "history_replay", ir.model_kind, interpret=interpret)
        values, diagnostics, _ = _run_stateful_graph_kernel(
            torch, triton, module, ir, prepared, rows, 1, seed, True, device, diag_slots(simulation.kernel_ir),
            kernel_name="pnl_batched_coevolving_graph_kernel",
            launch=_normalize_launch_options(None, interpret=interpret),
            extra_kernel_args=(observed, history, calls, status, *extra_kernel_args),
            parallel_trial_lanes=parallel_trial_lanes,
        )
    _report_truncation(diagnostics, ir.max_steps, True)
    states = history.cpu().numpy()
    status = status.cpu().numpy()
    if not np.all(status[..., 1] == 1):
        raise HistoryReplayError("history.schedule_incomplete", "Scheduler fuel exhausted before trial termination.")
    if counts is not None and not np.array_equal(status[..., 2], counts):
        raise HistoryReplayError("history.event_mismatch", "Replay did not reach the requested active count.")
    if not np.all(np.isfinite(states)):
        raise HistoryReplayError("history.state_nonfinite", "Reconstructed state or held control is nonfinite.")
    arrays = [states[:, :, 0, :width], states[:, :, 1, :width],
              states[:, :, 0, width:], states[:, :, 1, width:],
              calls.cpu().numpy(), status[..., 0], status[..., 2]]
    arrays.append(values.cpu().numpy()[:, 0, :, 0, :] if counts is None else None)
    for value in arrays:
        if value is not None:
            value.flags.writeable = False
    return HistoryTrace(*arrays)
