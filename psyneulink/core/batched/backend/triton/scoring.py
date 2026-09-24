"""Compact reductions attached to the checked observation sampling emitter."""

import numpy as np

from psyneulink.core.batched.backend.triton.sampling import ObservationRegionEmitter
from psyneulink.core.batched.backend.triton.emit.lanes import DEFAULT_NORMAL_RNG
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.observed_sampling import validate_observation_sampling_witness
from psyneulink.core.batched.prep import lca_max_steps, normalize_parameter_sets, prepare_inputs
from psyneulink.core.batched.sampling import StochasticSamplingError


class ReducedObservationEmitter(ObservationRegionEmitter):
    """One program per candidate/trial/estimate tile, integer reductions only.

    Primitive steps, clocks, RNG and observation expressions are inherited.
    Only lane packing and output consumption differ from sample inspection.
    """

    def __init__(self, kernel, witness, histogram=None, *, execution="strict", normal_rng=DEFAULT_NORMAL_RNG):
        super().__init__(kernel, witness, normal_rng=normal_rng)
        self.histogram = histogram
        self.execution = execution

    def _signature_args(self):
        args = list(super()._signature_args())
        index = args.index("LCA_MAX_STEPS")
        args[index:index] = ["observed", "observed_counts", "histogram_edges", "observed_bins", "observed_valid",
                             "CANDIDATE_OFFSET", "ESTIMATE_START", "ESTIMATE_STOP", "ESTIMATE_BLOCKS: tl.constexpr",
                             "sampled_trial_ids", "SAMPLED_TRIALS: tl.constexpr", "count_limits"]
        return tuple(args)

    def _do_not_specialize_args(self):
        return (*super()._do_not_specialize_args(), "CANDIDATE_OFFSET", "ESTIMATE_START", "ESTIMATE_STOP")

    def _emit_lane_decode(self):
        self.builder.line("packed_trial = tl.program_id(0) // ESTIMATE_BLOCKS")
        self.builder.line("original_trial = tl.load(sampled_trial_ids + packed_trial % SAMPLED_TRIALS)")
        self.builder.line("sample_trial = (packed_trial // SAMPLED_TRIALS) * num_trials + original_trial")
        self.builder.line("estimate_idx = (tl.program_id(0) % ESTIMATE_BLOCKS) * BLOCK + tl.arange(0, BLOCK) + ESTIMATE_START")
        self.builder.line("mask = (estimate_idx < ESTIMATE_STOP) & (estimate_idx < num_estimates)")
        self.builder.line("path_trial = sample_trial % num_trials + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("param_idx = sample_trial // num_trials + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("subject_idx = tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("offsets = sample_trial.to(tl.int64) * num_estimates + estimate_idx")
        self.builder.line("reduced_bad = tl.zeros((BLOCK,), tl.int32) != 0")

    def _emit_sample_limit(self):
        if self.execution == "window":
            self.builder.line("sample_limit = tl.load(count_limits + sample_trial)")
            return "sample_limit"
        return super()._emit_sample_limit()

    def _emit_stateful_random_base(self):
        # Address parameters locally but identify random streams globally.
        self.builder.line("local_param_idx = param_idx")
        self.builder.line("param_idx += CANDIDATE_OFFSET")
        super()._emit_stateful_random_base()
        self.builder.line("param_idx = local_param_idx")

    def _emit_store_flag(self, op):
        value = self._get_value(op.inputs[0].name)[0]
        self.builder.line(f"reduced_bad = reduced_bad | ({value} != 0)")

    def _emit_sample_outputs(self, raw_vars):
        self.observation_values = self._emit_observation_values(raw_vars)

    def _emit_sample_status(self, finished, count):
        width = len(self.observation_values)
        finite = " & ".join(f"(tl.abs({value}) <= 3.4028234663852886e38)" for _, value in self.observation_values)
        self.builder.line(f"observation_finite = {finite}")
        self.builder.line(f"window_stopped = ({finished} == 0) & (sample_limit < PATH_STEPS)" if self.execution == "window"
                          else "window_stopped = tl.zeros((BLOCK,), tl.int32) != 0")
        # A deliberately stopped lane has no endpoint observation. Never
        # score its intermediate readout or mistake it for a failed finish.
        self.builder.line("tl.atomic_add(sample_status + sample_trial * 3, tl.sum((mask & (sample_step > 0) & ~observation_finite).to(tl.int32), 0))")
        self.builder.line(f"tl.atomic_add(sample_status + sample_trial * 3 + 1, tl.sum((mask & ((({finished} == 0) & ~window_stopped) | reduced_bad)).to(tl.int32), 0))")
        self.builder.line("tl.atomic_add(sample_status + sample_trial * 3 + 2, tl.sum((mask & window_stopped).to(tl.int32), 0))")
        self.builder.line("matched = mask & ~window_stopped & observation_finite")
        for field, value in self.observation_values:
            if self.histogram is None:
                if not field.score:
                    continue
                if field.role == "event_time":
                    self.builder.line(f"matched = matched & ({count} == tl.load(observed_counts + sample_trial))")
                else:
                    self.builder.line(f"matched = matched & ({value} == tl.load(observed + path_trial * {width} + {field.column_start}))")
            elif field.column_start in self.histogram.categorical_dims:
                self.builder.line(f"matched = matched & (tl.abs({value} - tl.load(observed + path_trial * {width} + {field.column_start})) <= 1.0e-6)")
        if self.histogram is None:
            self.builder.line("tl.atomic_add(out + sample_trial, tl.sum(matched.to(tl.int32), 0))")
            return
        plan = self.histogram
        radius, bins = plan.radius, plan.bins
        self.builder.line(f"histogram_value = observation_{plan.continuous_dim}")
        self.builder.line("observed_bin = tl.load(observed_bins + path_trial)")
        self.builder.line("matched = matched & (tl.load(observed_valid + path_trial) != 0)")
        with self.builder.block(f"for bin_offset in tl.static_range(-{radius}, {radius + 1})"):
            self.builder.line("target_bin = observed_bin + bin_offset")
            self.builder.line(f"valid_bin = (target_bin >= 0) & (target_bin < {bins})")
            self.builder.line(f"safe_bin = tl.minimum(tl.maximum(target_bin, 0), {bins - 1})")
            self.builder.line("lower = tl.load(histogram_edges + safe_bin)")
            self.builder.line("upper = tl.load(histogram_edges + safe_bin + 1)")
            # Torch bucketize(right=False): interior edge belongs to lower bin.
            self.builder.line("in_lower = tl.where(safe_bin == 0, histogram_value >= lower, histogram_value > lower)")
            self.builder.line("hit = matched & valid_bin & in_lower & (histogram_value <= upper)")
            self.builder.line(f"tl.atomic_add(out + sample_trial * {2 * radius + 1} + bin_offset + {radius}, tl.sum(hit.to(tl.int32), 0))")


def run_reduced_observations(observation_plan, inputs, data, parameter_sets, estimates, seed,
                             common_random, horizon, budget, *, histogram=None, include_mask=None,
                             candidate_batch_size=None, estimate_batch_size=None, triton_launch_options=None,
                             execution="strict"):
    from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
    from psyneulink.core.batched.backend.triton.runtime import (
        _check_step_caps, _compiler_launch_options, _import_torch_triton, _input_tensors,
        _normalize_launch_options, _param_tensors,
    )

    validate_observation_sampling_witness(observation_plan.sampler, observation_plan.witness)
    if execution not in ("strict", "score_only", "window") or (histogram is None and execution != "strict"):
        raise ValueError("Reduced execution shortcuts currently require a histogram plan.")
    simulation = observation_plan.sampler.path_plan.history_plan.simulation_plan
    if simulation.backend not in ("triton_cpu", "triton"):
        raise StochasticSamplingError("sampling.backend", "Reduced scoring requires a Triton plan.")
    if type(estimates) is not int or not 1 <= estimates < 2**31:
        raise StochasticSamplingError("sampling.estimates", "num_estimates must be a positive int32 count.")
    if type(seed) is not int or type(common_random) is not bool:
        raise StochasticSamplingError("sampling.flags", "Seed must be integer and common_random_numbers boolean.")
    if type(budget) is not int or budget <= 0:
        raise StochasticSamplingError("sampling.memory_budget", "Buffer budget must be a positive integer.")
    for value in (candidate_batch_size, estimate_batch_size):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError("Batch sizes must be positive integers or None.")
    horizon = simulation.ir.max_steps if horizon is None else horizon
    if type(horizon) is not int or not 1 <= horizon <= simulation.ir.max_steps:
        raise StochasticSamplingError("sampling.horizon", "Horizon must be within the source step cap.")
    rows = normalize_parameter_sets(parameter_sets, simulation.ir)
    prepared = prepare_inputs(simulation.ir, inputs, parameter_sets=rows, component_bindings=simulation.component_bindings)
    subjects, trials = next(iter(prepared.values())).shape[:2]
    if subjects != 1 or trials == 0 or not rows:
        raise StochasticSamplingError("sampling.layout", "Scoring requires candidates and one contiguous subject.")
    if len(rows) * trials * estimates >= 2**31:
        raise StochasticSamplingError("sampling.index_domain", "Population exceeds the registered RNG lane-index domain.")
    width = len(observation_plan.witness.readouts)
    data = np.asarray(data, dtype=np.float64)
    if data.shape != (trials, width) or not np.all(np.isfinite(data)):
        raise StochasticSamplingError("sampling.data", "Data must be finite and match trial/observation axes.")
    if include_mask is not None and np.asarray(include_mask).size != trials:
        raise ValueError("include_mask must have one entry per trial.")
    sampled_trials = (np.ones(trials, dtype=bool) if execution == "strict" or include_mask is None
                      else np.asarray(include_mask, dtype=bool).reshape(-1))
    trial_ids = np.flatnonzero(sampled_trials).astype(np.int32)
    interpret = simulation.backend == "triton_cpu"
    device = "cpu" if interpret else "cuda"
    torch, triton = _import_torch_triton(interpret)
    path_plan = observation_plan.sampler.path_plan
    path_width = sum(field.width for field in path_plan.witness.fields)
    history_width = sum(state.width for state in simulation.kernel_ir.states) + len(simulation.kernel_ir.effective_parameters)
    output_width = sum(output.width for output in simulation.kernel_ir.outputs)
    count_width = 1 if histogram is None else 2 * histogram.radius + 1
    # Device path generation retains no host copy; the returned tensor is the
    # sampling input itself, not a second allocation. Include validation scratch
    # and the small host history just as the path launcher does.
    path_bytes = trials * (horizon * (4 * path_width + 5)
                           + 8 * history_width * 4
                           + 3 * (len(path_plan.witness.history.component_ids) + 3 + output_width) * 4)
    extra_per_candidate = trials * ((count_width + 3) * 32 + 8
                                    + len(simulation.ir.params) * 12)
    fixed = (trials * (width * 12 + count_width * 16 + 64 + sum(item.width for item in simulation.ir.graph.inputs) * 8)
             + len(rows) * trials * count_width * 32
             + (0 if histogram is None else (histogram.bins + 1) * 8))
    capacity = (budget - fixed) // (path_bytes + extra_per_candidate)
    if capacity < 1 or trials * horizon * path_width >= 2**31:
        raise StochasticSamplingError("sampling.memory_budget", "Even one candidate's paths/reductions exceed the budget or index domain.")
    batch_size = min(len(rows), capacity, candidate_batch_size or len(rows), (2**31 - 1) // (trials * horizon * max(1, path_width)))
    estimate_batch_size = min(estimates, estimate_batch_size or estimates)
    observed = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
    if not torch.isfinite(observed).all().item():
        raise StochasticSamplingError("sampling.data", "Observed values exceed finite FP32.")
    dummy = torch.empty(1, device=device)
    edges = observed_bin = valid = dummy
    if histogram is not None:
        from psyneulink.core.batched.histogram_score import prepare_histogram, histogram_result

        observed, edges, observed_bin, valid, weights, joint_bins = prepare_histogram(histogram, data, device)
    launch = _normalize_launch_options(triton_launch_options, interpret=interpret)
    if launch["trial_schedule"] != "synchronized":
        raise ValueError("Independent trial scheduling applies to complete dynamic sequences, not observed-history sampling.")
    source = ReducedObservationEmitter(simulation.kernel_ir, observation_plan.witness, histogram,
                                       execution=execution, normal_rng=launch["normal_rng"]).emit()
    input_tensors = _input_tensors(torch, simulation.ir.graph, prepared, device)
    device_trial_ids = torch.tensor(trial_ids, device=device)
    results, stopped = [], []
    with interpret_scope(interpret):
        module = load_triton_kernel_module(source, "reduced_observations", simulation.ir.model_kind, interpret=interpret)
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            remaining = budget - fixed - len(batch) * extra_per_candidate
            paths = path_plan.generate_device(inputs, data, batch, horizon=horizon, max_buffer_bytes=remaining)
            path_values = paths.values
            target = torch.tensor(np.array(paths.history.event_counts), dtype=torch.int32, device=device)
            limits = dummy
            if execution == "window":
                from psyneulink.core.batched.scoring_windows import histogram_count_limits

                limits = torch.tensor(histogram_count_limits(
                    histogram, simulation, prepared, batch, edges.cpu().numpy(), observed_bin.cpu().numpy(), horizon,
                ), device=device)
            counts = torch.zeros((len(batch), trials, count_width), dtype=torch.int32, device=device)
            status = torch.zeros((len(batch), trials, 3), dtype=torch.int32, device=device)
            parameters, strides = _param_tensors(torch, simulation.ir, batch, device, num_subjects=1, num_trials=trials, subject_slices=None)
            lca_steps = lca_max_steps(simulation.ir, prepared, batch)
            _check_step_caps(max_steps=horizon, lca_max_steps=lca_steps)
            for start in range(0, estimates, estimate_batch_size):
                if not len(trial_ids):
                    break  # History still reconstructed; no score factors requested.
                stop = min(start + estimate_batch_size, estimates)
                blocks = triton.cdiv(stop - start, launch["block_size"])
                module.pnl_batched_coevolving_graph_kernel[(len(batch) * len(trial_ids) * blocks,)](
                    *input_tensors, *parameters, *strides, counts,
                    *(() if not diag_slots(simulation.kernel_ir) else (dummy,)),
                    dummy, dummy, False, False, len(batch) * trials * estimates, 1, estimates, trials,
                    path_values, status, horizon, observed, target, edges, observed_bin, valid,
                    offset, start, stop, blocks, device_trial_ids, len(trial_ids), limits,
                    LCA_MAX_STEPS=lca_steps, MAX_STEPS=horizon, COMMON_RANDOM=common_random,
                    SEED=seed, TRIAL_OFFSET=0, RNG_NUM_TRIALS=trials, BLOCK=launch["block_size"],
                    **_compiler_launch_options(launch),
                )
            diagnostics = status.sum((0, 1)).cpu().numpy()
            if diagnostics[0]:
                raise StochasticSamplingError("sampling.nonfinite", "A sampled observation is nonfinite.")
            if diagnostics[1]:
                raise StochasticSamplingError("sampling.truncated", f"{int(diagnostics[1])} sample lane(s) did not finish within the configured horizon.")
            results.append(counts.cpu().numpy())
            stopped.append(status[..., 2].cpu().numpy())
            del paths, path_values, target, counts, status, parameters, strides, limits
    counts = np.concatenate(results, axis=0)
    if histogram is None:
        return counts[..., 0].astype(np.int64)
    return histogram_result(histogram, torch.tensor(counts, device=device), weights, edges, joint_bins,
                            estimates, include_mask, simulation.backend, execution=execution,
                            sampled_trials=sampled_trials, window_stopped=np.concatenate(stopped))
