"""Generated boundary sampling and trial-parallel deterministic path lanes."""

import numpy as np

from psyneulink.core.batched.backend.triton.history import HistoryTraceEmitter, _run_history_trace
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs
from psyneulink.core.batched.trajectories import BoundaryTrajectories, BoundaryTrajectoryError


class BoundaryTrajectoryEmitter(HistoryTraceEmitter):
    def __init__(self, kernel, witness, *, parallel_trials):
        super().__init__(kernel, witness.history, replay=parallel_trials)
        self.boundary_witness = witness
        self.parallel_trials = parallel_trials
        self.loading_path_starts = False
        self.boundary_width = sum(field.width for field in witness.fields)

    def _signature_args(self):
        args = list(super()._signature_args())
        index = args.index("LCA_MAX_STEPS")
        args[index:index] = ["boundary_values", "boundary_valid", "boundary_passes", "path_starts", "PATH_STEPS: tl.constexpr"]
        return tuple(args)

    def _emit_lane_decode(self):
        if not self.parallel_trials:
            return super()._emit_lane_decode()
        self.builder.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = offsets < total_lanes")
        self.builder.line("path_trial = offsets % num_trials")
        self.builder.line("param_idx = offsets // num_trials")
        self.builder.line("subject_idx = tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("estimate_idx = tl.zeros((BLOCK,), tl.int32)")

    def _emit_params(self, *, trial_varying_only=False):
        if self.parallel_trials:
            # The source emitter sets trial zero for ordinary sequential
            # initialization. Path lanes initialize at their own trial instead.
            self.builder.line("trial_idx = path_trial")
        super()._emit_params(trial_varying_only=trial_varying_only)

    def _emit_initialize_state(self):
        self.loading_path_starts = self.parallel_trials
        try:
            super()._emit_initialize_state()
        finally:
            self.loading_path_starts = False

    def _emit_state_initializer_value(self, state, index, value, output):
        if not self.loading_path_starts:
            # ResetState must use the actual source initializer, not the saved
            # pre-reset value. Only lane entry restores canonical history.
            return super()._emit_state_initializer_value(state, index, value, output)
        column = index
        for declaration in self.kernel.states:
            if declaration.state_id == state.state_id:
                break
            column += declaration.width
        self.builder.line(f"{output} = tl.load(path_starts + offsets * {self.history_width} + {column}, mask=mask, other=0.0)")

    def _emit_initialize_effective_parameter(self, op):
        super()._emit_initialize_effective_parameter(op)
        if self.parallel_trials:
            identity = op.attrs["effective_parameter_id"]
            index = self.witness.effective_parameter_ids.index(identity) + self.state_width
            value = self.effective_parameter_vars[identity]
            self.builder.line(f"{value} = tl.load(path_starts + offsets * {self.history_width} + {index}, mask=mask, other=0.0)")

    def _emit_trial_loop(self, body):
        if not self.parallel_trials:
            return super()._emit_trial_loop(body)
        # Exactly one trial per lane. Its reset operations are still executed,
        # but a hypothetical tail never feeds any other trial's start state.
        self._emit_stateful_random_base()
        self.output_cursor = 0
        self.lane_out_emitted = False
        self._emit_trial_start_inspection()
        self._emit_ops(body)
        self._emit_trial_end_inspection()

    def _history_trial_index(self):
        return "offsets" if self.parallel_trials else super()._history_trial_index()

    def _emit_dynamic_step_mechanism(self, op, spec):
        if op.attrs["component_id"] == self.boundary_witness.consumer_component_id:
            trial_count = len(op.attrs["trial_state_ids"])
            trial_inputs = op.inputs[-trial_count:]
            counter_index = next(index for index, state in enumerate(spec.trial_states)
                                 if state.name == self.witness.endpoint.counter_state)
            count = self._get_value(trial_inputs[counter_index].name)[0]
            self.builder.line(f"boundary_step = {count}.to(tl.int32)")
            self.builder.line(f"boundary_mask = {self.dynamic_active_mask} & (boundary_step < PATH_STEPS)")
            self.builder.line("boundary_row = history_trial * PATH_STEPS + boundary_step")
            for field in self.boundary_witness.fields:
                for index, value in enumerate(self._get_value(field.value_name)):
                    self.builder.line(
                        f"tl.store(boundary_values + boundary_row * {self.boundary_width} + "
                        f"{field.column_start + index}, {value}, mask=boundary_mask)"
                    )
            self.builder.line("tl.store(boundary_valid + boundary_row, True, mask=boundary_mask)")
            self.builder.line("tl.store(boundary_passes + boundary_row, dynamic_round, mask=boundary_mask)")
        return super()._emit_dynamic_step_mechanism(op, spec)


def run_boundary_trajectories(plan, inputs, data, parameter_sets, horizon, max_buffer_bytes, *, reference=False, seed=0):
    from psyneulink.core.batched.backend.triton.runtime import _import_torch_triton

    simulation = plan.history_plan.simulation_plan
    if simulation.backend != "triton_cpu":
        raise BoundaryTrajectoryError("boundary.backend_unsupported", "Boundary inspection currently requires a triton_cpu plan.")
    if horizon is None:
        horizon = simulation.ir.max_steps
    if type(horizon) is not int or not 1 <= horizon <= simulation.ir.max_steps:
        raise BoundaryTrajectoryError("boundary.horizon", "Path horizon must be a positive integer within the source step cap.")
    if type(max_buffer_bytes) is not int or max_buffer_bytes <= 0:
        raise BoundaryTrajectoryError("boundary.memory_budget", "The inspection buffer budget must be a positive integer.")
    rows = normalize_parameter_sets(parameter_sets, simulation.ir)
    prepared = prepare_inputs(simulation.ir, inputs, parameter_sets=rows, component_bindings=simulation.component_bindings)
    subjects, trials = next(iter(prepared.values())).shape[:2]
    if not rows or subjects != 1 or trials == 0:
        raise BoundaryTrajectoryError("boundary.layout", "Boundary inspection requires nonempty candidates and one contiguous subject.")
    shape = (len(rows), trials, horizon)
    width = sum(field.width for field in plan.witness.fields)
    state_width = sum(state.width for state in simulation.kernel_ir.states)
    history_width = state_width + len(simulation.kernel_ir.effective_parameters)
    # Conservatively include live path buffers, copied starts, both history
    # traces, status/call buffers and output scratch. This excludes JIT/compiler
    # and framework workspace, so it is an explicit buffer limit, not an RSS cap.
    trial_slots = len(rows) * trials
    output_width = sum(output.width for output in simulation.kernel_ir.outputs)
    required = trial_slots * (horizon * (4 * width + 1 + 4)
                             + 8 * history_width * 4 + 3 * (len(plan.witness.history.component_ids) + 3 + output_width) * 4)
    if required > max_buffer_bytes:
        raise BoundaryTrajectoryError("boundary.memory_budget", f"Inspection buffers require approximately {required} bytes; budget is {max_buffer_bytes}.")
    torch, _ = _import_torch_triton(True)
    values = torch.full((*shape, width), float("nan"), dtype=torch.float32)
    valid = torch.zeros(shape, dtype=torch.bool)
    passes = torch.full(shape, -1, dtype=torch.int32)
    if reference:
        canonical = None
        starts = torch.empty((1,), dtype=torch.float32)
        counts = None
    else:
        canonical = plan.history_plan.reconstruct(inputs, data, rows)
        starts = torch.tensor(np.concatenate((canonical.start_states, canonical.start_effective_parameters), axis=-1))
        counts = np.full(shape[:2], horizon, dtype=np.int32)
    trace = _run_history_trace(
        plan.history_plan, inputs, rows, counts=counts, seed=seed,
        source_override=plan.source(parallel_trials=not reference),
        extra_kernel_args=(values, valid, passes, starts, horizon), parallel_trial_lanes=not reference,
    )
    values, valid, passes = values.numpy(), valid.numpy(), passes.numpy()
    if not np.all(np.isfinite(values[valid])):
        raise BoundaryTrajectoryError("boundary.nonfinite", "A consumed boundary value is nonfinite.")
    expected_counts = np.minimum(trace.event_counts, horizon)
    expected_valid = np.arange(horizon)[None, None, :] < expected_counts[..., None]
    if not np.array_equal(valid, expected_valid):
        raise BoundaryTrajectoryError("boundary.prefix_missing", "The generated boundary is not a complete active-step prefix.")
    if canonical is not None:
        if (not np.array_equal(trace.start_states, canonical.start_states)
                or not np.array_equal(trace.start_effective_parameters, canonical.start_effective_parameters)):
            raise BoundaryTrajectoryError("boundary.start_mismatch", "Trial-parallel lanes did not restore the canonical states and held controls.")
    for array in (values, valid, passes):
        array.flags.writeable = False
    return BoundaryTrajectories(
        trace if reference else canonical, values, valid, passes, plan.witness.fields,
        "coupled_reference" if reference else "observed_history_paths",
    )
