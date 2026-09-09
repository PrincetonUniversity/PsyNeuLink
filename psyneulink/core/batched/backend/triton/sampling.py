"""Stochastic-region sampling using registered source step/readout emitters."""

import numpy as np

from psyneulink.core.batched.backend.triton.emit.emitter import TritonGraphEmitter
from psyneulink.core.batched.backend.triton.trajectories import BoundaryTrajectoryEmitter
from psyneulink.core.batched.history import replay_program
from psyneulink.core.batched.kernel_ir import diag_slots, node_output_value_name
from psyneulink.core.batched.prep import lca_max_steps, normalize_parameter_sets, prepare_inputs
from psyneulink.core.batched.sampling import PrimitiveSamples, StochasticSamplingError, stochastic_step


def _emit_sample_lanes(emitter):
    emitter.builder.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
    emitter.builder.line("mask = offsets < total_lanes")
    emitter.builder.line("estimate_idx = offsets % num_estimates")
    emitter.builder.line("sample_trial = offsets // num_estimates")
    emitter.builder.line("path_trial = sample_trial % num_trials")
    emitter.builder.line("param_idx = sample_trial // num_trials")
    emitter.builder.line("subject_idx = tl.zeros((BLOCK,), tl.int32)")


class StochasticRegionEmitter(TritonGraphEmitter):
    def __init__(self, kernel, witness):
        super().__init__(kernel)
        self.witness = witness
        self.program = replay_program(kernel)
        self.step = stochastic_step(kernel, witness.boundary.consumer_component_id)

    def _signature_args(self):
        args = list(super()._signature_args())
        index = args.index("LCA_MAX_STEPS")
        args[index:index] = ["path_values", "sample_status", "PATH_STEPS: tl.constexpr"]
        return tuple(args)

    def _emit_lane_decode(self):
        _emit_sample_lanes(self)

    def _emit_params(self, *, trial_varying_only=False):
        self.builder.line("trial_idx = path_trial")
        super()._emit_params(trial_varying_only=trial_varying_only)

    def _emit_top_level_ops(self):
        spec = self.kernel.op_specs.lookup_spec(self.witness.spec_key)
        clock = self.witness.boundary.consumer_component_id
        effective_ids = self.step.attrs["sampled_effective_parameter_ids"]
        for op in self.kernel.ops:
            if op.kind == "InitializeEffectiveParameter" and op.attrs["effective_parameter_id"] in effective_ids:
                self._emit_initialize_effective_parameter(op)
        carries_by_id = {carry.value_id: carry for carry in self.program.loop_carries
                         if carry.kind == "trial_state" and carry.owner_component_id == clock}
        carries = tuple(carries_by_id[identity] for identity in self.step.attrs["trial_state_ids"])
        carry_vars = self._emit_dynamic_carry_initializers(self.program, carries=carries)
        self.dynamic_program = self.program
        self.dynamic_slot_vars = {
            self._dynamic_slot_key(slot): "sample_step" for slot in self.program.scheduler_state_slots
            if slot.kind == "rng_clock" and slot.owner_component_id == clock
        }
        self._emit_dynamic_normal_cache_initializers(self.program)
        self._emit_stateful_random_base()
        state_inputs = self.step.inputs[-len(carries):]
        counter_index = next(i for i, decl in enumerate(spec.trial_states)
                             if decl.name == self.witness.boundary.history.endpoint.counter_state)
        finished_index = self.step.attrs["finished_trial_state_id"]
        count = self._get_value(state_inputs[counter_index].name)[0]
        finished = self._get_value(state_inputs[finished_index].name)[0]
        raw_outputs = self.step.outputs[:len(self.witness.outputs)]
        raw_vars = []
        for field in self.witness.outputs:
            names = [f"sample_output_{field.column_start + index}" for index in range(field.width)]
            for name in names:
                self.builder.line(f"{name} = tl.zeros((BLOCK,), tl.float32)")
            raw_vars.append(names)
        self.builder.line("sample_step = 0")
        with self.builder.block(f"while (sample_step < PATH_STEPS) & (tl.max(tl.where(mask & ({finished} == 0), 1, 0)) > 0)"):
            self.builder.line(f"sample_active = mask & ({finished} == 0)")
            self.dynamic_active_mask = "sample_active"
            self.dynamic_execution_index = "sample_step"
            width = sum(field.width for field in self.witness.boundary.fields)
            self.builder.line("path_row = (param_idx * num_trials + trial_idx) * PATH_STEPS + sample_step")
            for field in self.witness.boundary.fields:
                if field.effective_parameter_id is None:
                    names = [f"sample_input_{index}" for index in range(field.width)]
                else:
                    names = [self.effective_parameter_vars[field.effective_parameter_id]]
                for index, name in enumerate(names):
                    self.builder.line(f"{name} = tl.load(path_values + path_row * {width} + {field.column_start + index}, mask=sample_active, other=0.0)")
                self._set_value(field.value_name, names)
            # Reuse the same checked binding, state candidates, RNG clocks,
            # primitive step and readout as ordinary scheduled execution.
            self._emit_dynamic_step_mechanism(self.step, spec)
            for carry, output in zip(carries, self.step.outputs[-len(carries):]):
                old = carry_vars[self._dynamic_carry_key(carry)]
                for target, source in zip(old, self._get_value(output.name)):
                    self.builder.line(f"{target} = tl.where(sample_active, {source}, {target})")
                self._set_value(carry.value.name, old)
            for targets, output in zip(raw_vars, raw_outputs):
                for target, source in zip(targets, self._get_value(output.name)):
                    self.builder.line(f"{target} = tl.where(sample_active, {source}, {target})")
            self.builder.line("sample_step += 1")
        width = sum(field.width for field in self.witness.outputs)
        for field, names in zip(self.witness.outputs, raw_vars):
            for index, name in enumerate(names):
                self.builder.line(f"tl.store(out + offsets * {width} + {field.column_start + index}, {name}, mask=mask)")
        for index, value in enumerate(("sample_step", finished, count)):
            self.builder.line(f"tl.store(sample_status + offsets * 3 + {index}, {value}, mask=mask)")


class CanonicalTrialReferenceEmitter(BoundaryTrajectoryEmitter):
    """Unmodified coupled trials, broadcast from complete canonical starts."""

    def __init__(self, kernel, witness):
        super().__init__(kernel, witness.boundary, parallel_trials=True)
        self.sampler_witness = witness
        self.replay = False
        self.raw_outputs_stored = False

    def _emit_lane_decode(self):
        _emit_sample_lanes(self)

    def _path_start_index(self):
        return "param_idx * num_trials + trial_idx"

    def _emit_trial_start_inspection(self):
        self.builder.line("history_trial = offsets")

    def _emit_trial_end_inspection(self):
        pass

    def _emit_dynamic_scheduler_updates(self, *args):
        TritonGraphEmitter._emit_dynamic_scheduler_updates(self, *args)

    def _emit_dynamic_step_mechanism(self, op, spec):
        TritonGraphEmitter._emit_dynamic_step_mechanism(self, op, spec)

    def _emit_store_output(self, op):
        if self.raw_outputs_stored:
            return
        self.raw_outputs_stored = True
        node = self.graph.node(stochastic_step(self.kernel, self.sampler_witness.boundary.consumer_component_id).target)
        width = sum(field.width for field in self.sampler_witness.outputs)
        for field in self.sampler_witness.outputs:
            values = self._get_value(node_output_value_name(self.graph, node, field.port_name))
            for index, value in enumerate(values):
                self.builder.line(f"tl.store(out + offsets * {width} + {field.column_start + index}, {value}, mask=mask)")


def run_stochastic_samples(plan, inputs, data, parameter_sets, estimates, seed, common_random, horizon, strict, budget, *, reference):
    from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
    from psyneulink.core.batched.backend.triton.runtime import (
        _check_step_caps, _compiler_launch_options, _import_torch_triton,
        _input_tensors, _normalize_launch_options, _param_tensors,
    )

    simulation = plan.path_plan.history_plan.simulation_plan
    if simulation.backend != "triton_cpu":
        raise StochasticSamplingError("sampling.backend", "The current sampler is a triton_cpu reference implementation.")
    if type(estimates) is not int or estimates < 1:
        raise StochasticSamplingError("sampling.estimates", "num_estimates must be a positive integer.")
    if type(common_random) is not bool or type(strict) is not bool:
        raise StochasticSamplingError("sampling.flags", "Random-number and truncation flags must be booleans.")
    if type(seed) is not int:
        raise StochasticSamplingError("sampling.seed", "seed must be an integer.")
    horizon = simulation.ir.max_steps if horizon is None else horizon
    if type(horizon) is not int or not 1 <= horizon <= simulation.ir.max_steps:
        raise StochasticSamplingError("sampling.horizon", "horizon must be a positive integer within the source step cap.")
    if type(budget) is not int or budget <= 0:
        raise StochasticSamplingError("sampling.memory_budget", "The inspection buffer budget must be a positive integer.")
    rows = normalize_parameter_sets(parameter_sets, simulation.ir)
    prepared = prepare_inputs(simulation.ir, inputs, parameter_sets=rows, component_bindings=simulation.component_bindings)
    subjects, trials = next(iter(prepared.values())).shape[:2]
    if subjects != 1 or trials == 0 or not rows:
        raise StochasticSamplingError("sampling.layout", "Sampling requires nonempty candidates and one contiguous subject.")
    width = sum(field.width for field in plan.witness.outputs)
    slots = diag_slots(simulation.kernel_ir)
    lanes = len(rows) * trials * estimates
    path_width = sum(field.width for field in plan.witness.boundary.fields)
    # Reserve the output/status/diagnostic arrays, validation temporaries, and
    # the extra path/start copy held by this launcher. The path generator then
    # budgets its own history/path workspace from the remaining allowance.
    sample_bytes = lanes * (4 * (width + 3 + len(slots)) + width + 3)
    if reference:
        history_width = sum(state.width for state in simulation.kernel_ir.states) + len(simulation.kernel_ir.effective_parameters)
        sample_bytes += len(rows) * trials * history_width * 4
    else:
        sample_bytes += len(rows) * trials * horizon * path_width * 4
    if (sample_bytes >= budget or lanes * max(width, 3, len(slots)) >= 2**31
            or len(rows) * trials * horizon * path_width >= 2**31):
        raise StochasticSamplingError("sampling.memory_budget", "Sample buffers exceed the budget or supported index domain.")
    # Generate internally rather than accepting externally cached paths whose
    # parameter/history provenance could differ from this sampling request.
    paths = plan.path_plan.generate(inputs, data, rows, horizon=horizon, max_buffer_bytes=budget - sample_bytes)
    torch, triton = _import_torch_triton(True)
    shape = (len(rows), trials, estimates)
    values = torch.empty((*shape, width), dtype=torch.float32)
    status = torch.empty((*shape, 3), dtype=torch.int32)
    diag = torch.zeros((*shape, len(slots)), dtype=torch.float32) if slots else None
    dummy = torch.empty((1,), dtype=torch.float32)
    if reference:
        starts = torch.from_numpy(np.concatenate((paths.history.start_states, paths.history.start_effective_parameters), axis=-1))
        extra = (dummy, dummy, dummy, status, dummy, dummy, dummy, starts, horizon)
    else:
        path_values = torch.from_numpy(np.array(paths.values, copy=True))
        extra = (path_values, status, horizon)
    input_tensors = _input_tensors(torch, simulation.ir.graph, prepared, "cpu")
    param_tensors, strides = _param_tensors(torch, simulation.ir, rows, "cpu", num_subjects=1, num_trials=trials, subject_slices=None)
    lca_steps = lca_max_steps(simulation.ir, prepared, rows)
    _check_step_caps(max_steps=horizon, lca_max_steps=lca_steps)
    launch = _normalize_launch_options(None, interpret=True)
    with interpret_scope(True):
        module = load_triton_kernel_module(plan.source(reference=reference), "stochastic_region", simulation.ir.model_kind, interpret=True)
        module.pnl_batched_coevolving_graph_kernel[(triton.cdiv(lanes, launch["block_size"]),)](
            *input_tensors, *param_tensors, *strides, values, *(() if diag is None else (diag,)),
            dummy, dummy, False, False, lanes, 1, estimates, trials, *extra,
            LCA_MAX_STEPS=lca_steps, MAX_STEPS=horizon, COMMON_RANDOM=common_random,
            SEED=seed, TRIAL_OFFSET=0, RNG_NUM_TRIALS=trials, BLOCK=launch["block_size"],
            **_compiler_launch_options(launch),
        )
    values, status = values.numpy(), status.numpy()
    truncated = status[..., 1] == 0
    if diag is not None:
        truncated |= np.any(diag.numpy() != 0, axis=-1)
    if not np.all(np.isfinite(values)):
        raise StochasticSamplingError("sampling.nonfinite", "A primitive output is nonfinite.")
    if strict and np.any(truncated):
        raise StochasticSamplingError("sampling.truncated", f"{np.count_nonzero(truncated)} sample lane(s) did not finish within the configured horizon.")
    counts = status[..., 2]
    for array in (values, counts, truncated):
        array.flags.writeable = False
    return PrimitiveSamples(paths.history, values, counts, truncated, plan.witness.outputs,
                            "canonical_coupled_reference" if reference else "conditional_primitive_samples")
