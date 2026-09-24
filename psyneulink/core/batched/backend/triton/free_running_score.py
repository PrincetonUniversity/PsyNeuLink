"""Histogram reductions on the ordinary stateful simulator's trial outputs.

Every estimate executes the full trial sequence with its own retained state.
Only output consumption changes: reduce matches and diagnostics in the kernel
instead of retaining an outcome and diagnostic array for every estimate.
For one continuous outcome, Gaussian smoothing retains neighboring bin counts
and weights them after simulation. Categories are never smoothed together.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.emit import TritonGraphEmitter
from psyneulink.core.batched.backend.triton.emit.lanes import DEFAULT_NORMAL_RNG
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.likelihood import (
    ZERO_PROB, _as_categorical_mask, _bin_edges, _categorical_cardinalities,
)
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs, lca_max_steps


class HistogramEmitter(TritonGraphEmitter):
    def __init__(self, kernel, indices, categorical, radius=0, *, normal_rng=DEFAULT_NORMAL_RNG,
                 trial_schedule="synchronized"):
        super().__init__(kernel, normal_rng=normal_rng, trial_schedule=trial_schedule)
        self.indices = tuple(indices)
        self.categorical = tuple(categorical)
        self.radius = radius
        self.histogram_outputs = {}

    def _emit_lane_decode(self):
        self.builder.line("hist_group = tl.program_id(0) // tl.cdiv(num_estimates, BLOCK)")
        self.builder.line("estimate_idx = (tl.program_id(0) % tl.cdiv(num_estimates, BLOCK)) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = estimate_idx < num_estimates")
        self.builder.line("subject_idx = hist_group % num_subjects + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("param_idx = hist_group // num_subjects + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("offsets = hist_group * num_estimates + estimate_idx")

    def _emit_trial_output_begin(self):
        self.histogram_outputs = {}
        self.builder.line("hist_row = hist_group * num_trials + trial_idx")
        self.builder.line("hist_nonfinite = tl.zeros((BLOCK,), tl.int32)")

    def _emit_store_output(self, op):
        values = self._get_value(op.inputs[0].name)
        start = int(op.attrs.get("flat_start", self.output_cursor))
        if start < 0:
            start = self.output_cursor
        for i in range(op.attrs["width"]):
            name = f"hist_value_{start + i}"
            self.builder.line(f"{name} = {values[i]}")
            self.builder.line(f"hist_nonfinite += (~(tl.abs({name}) <= 3.4028234663852886e38)).to(tl.int32)")
            self.histogram_outputs[start + i] = name
        self.output_cursor = max(self.output_cursor, start + op.attrs["width"])

    def _emit_store_flag(self, op):
        value = self._get_value(op.inputs[0].name)[0]
        self._emit_histogram_add(f"diag + hist_row * {self.diag_slot_count + 1} + {op.attrs['slot']}", value)

    def _emit_histogram_add(self, pointer, value):
        if self.trial_schedule == "independent":
            # Different completed lanes can publish to different trial rows.
            # Integer atomics preserve exact counts regardless of arrival order.
            self.builder.line(f"tl.atomic_add({pointer}, ({value}).to(tl.int32), mask=mask & (({value}) != 0), sem='relaxed')")
        else:
            self.builder.line(f"tl.atomic_add({pointer}, tl.sum(tl.where(mask, {value}, 0).to(tl.int32), 0))")

    def _histogram_load(self, pointer):
        if self.trial_schedule == "independent":
            return f"tl.load({pointer}, mask=mask, other=0)"
        return f"tl.load({pointer})"

    def _emit_trial_end_inspection(self):
        width = len(self.indices)
        count_width = 2 * self.radius + 1
        self.builder.line(f"hist_match = mask & ({self._histogram_load('observed_valid + trial_idx')} != 0)")
        for column, index in enumerate(self.indices):
            value = self.histogram_outputs[index]
            address = f"trial_idx * {width} + {column}"
            if self.categorical[column]:
                self.builder.line(f"hist_match = hist_match & (tl.abs({value} - {self._histogram_load(f'observed + {address}')}) <= 1.0e-6)")
            elif not self.radius:
                self.builder.line(f"hist_lower = {self._histogram_load(f'lower + {address}')}")
                self.builder.line(f"hist_upper = {self._histogram_load(f'upper + {address}')}")
                self.builder.line(f"hist_inclusive = {self._histogram_load(f'lower_inclusive + {address}')}")
                self.builder.line(f"hist_match = hist_match & tl.where(hist_inclusive != 0, {value} >= hist_lower, {value} > hist_lower) & ({value} <= hist_upper)")
        if self.radius:
            column = self.categorical.index(False)
            value = self.histogram_outputs[self.indices[column]]
            # A loop keeps code/register growth bounded as sigma increases.
            # Every offset is reduced as integers, independently of block order.
            with self.builder.block(f"for hist_slot in range({count_width})"):
                address = f"(trial_idx * {width} + {column}) * {count_width} + hist_slot"
                self.builder.line(f"hist_lower = {self._histogram_load(f'lower + {address}')}")
                self.builder.line(f"hist_upper = {self._histogram_load(f'upper + {address}')}")
                self.builder.line(f"hist_inclusive = {self._histogram_load(f'lower_inclusive + {address}')}")
                self.builder.line(f"hist_hit = hist_match & (hist_inclusive >= 0) & tl.where(hist_inclusive == 1, {value} >= hist_lower, {value} > hist_lower) & ({value} <= hist_upper)")
                self._emit_histogram_add(f"out + hist_row * {count_width} + hist_slot", "hist_hit")
        else:
            self._emit_histogram_add("out + hist_row", "hist_match")
        self._emit_histogram_add(f"diag + hist_row * {self.diag_slot_count + 1} + {self.diag_slot_count}", "hist_nonfinite")

    def _signature_args(self):
        args = list(super()._signature_args())
        # Nonfinite checks need a status pointer even when there are no flags.
        if not self.diag_slot_count:
            args.insert(args.index("out") + 1, "diag")
        return (*args, "observed", "lower", "upper", "lower_inclusive", "observed_valid")


def supports_fused_histogram(plan, smoothing_sigma, categorical_dims=None, outcome_indices=None):
    if (plan.backend not in ("triton", "triton_cpu") or plan.ir.graph is None
            or plan.ir.graph.fusion_kind not in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION)):
        return False
    if smoothing_sigma == 0:
        return True
    width = (sum(output.width for output in plan.kernel_ir.outputs)
             if outcome_indices is None else len(outcome_indices))
    # Joint neighborhoods grow exponentially with the number of numeric axes.
    # Keep the materialized oracle for multidimensional smoothing.
    return np.count_nonzero(~_as_categorical_mask(categorical_dims, width)) <= 1


def fused_histogram_log_likelihood(plan, inputs, parameter_sets, num_estimates, data, categorical_dims,
                                   *, outcome_indices, bins, bin_range, smoothing_sigma, pseudocount, categorical_cardinalities,
                                   include_mask, subject_slices, seed, common_random_numbers,
                                   strict_truncation, triton_launch_options):
    from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
    from psyneulink.core.batched.backend.triton.runtime import (
        _check_step_caps, _compiler_launch_options, _import_torch_triton,
        _input_tensors, _normalize_launch_options, _param_tensors, _report_truncation,
    )
    from psyneulink.core.batched.errors import BatchedNumericalError
    from psyneulink.core.batched.likelihood import _sum_histogram_log_likelihood

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}.")
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError("pseudocount must be finite and nonnegative.")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError("smoothing_sigma must be finite and nonnegative.")
    if not isinstance(num_estimates, (int, np.integer)) or not 0 < num_estimates < 2**31:
        raise ValueError("num_estimates must be a positive int32 count.")
    interpret = plan.backend == "triton_cpu"
    device = "cpu" if interpret else "cuda"
    launch = _normalize_launch_options(triton_launch_options, interpret=interpret)
    torch, triton = _import_torch_triton(interpret)
    ir, kernel = plan.ir, plan.kernel_ir
    rows = normalize_parameter_sets(parameter_sets, ir)
    subject_slices = None if subject_slices is None else tuple(subject_slices)
    prepared = prepare_inputs(ir, inputs, subject_slices, parameter_sets=rows,
                              component_bindings=plan.component_bindings)
    input_tensors = _input_tensors(torch, ir.graph, prepared, device)
    subjects, trials = input_tensors[0].shape[:2]
    width = sum(output.width for output in kernel.outputs)
    indices = tuple(range(width)) if outcome_indices is None else tuple(outcome_indices)
    if not indices or any(not isinstance(i, (int, np.integer)) or not 0 <= i < width for i in indices):
        raise ValueError("outcome_indices must select valid output columns.")
    data = np.asarray(data, dtype=float)
    if data.shape != (trials, len(indices)):
        raise ValueError("Data must match the simulation's trial and selected outcome axes.")
    categorical = _as_categorical_mask(categorical_dims, len(indices))
    numeric = np.flatnonzero(~categorical).tolist()
    # Clip before multiplying by three, including for very large finite sigma.
    radius = (min(bins - 1, max(1, int(np.ceil(3 * min(float(smoothing_sigma), bins)))))
              if smoothing_sigma and numeric else 0)
    count_width = 2 * radius + 1
    observed = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
    lower = torch.zeros((*observed.shape, count_width), device=device)
    upper = torch.zeros_like(lower)
    inclusive = torch.zeros_like(lower, dtype=torch.int32)
    valid = torch.ones(trials, dtype=torch.bool, device=device)
    numeric_values = observed[:, numeric]
    edges = _bin_edges(numeric_values, numeric_values, bins, bin_range, torch)
    volume = torch.tensor(1., device=device)
    weights = None
    for j, column in enumerate(numeric):
        edge = edges[j]
        index = torch.bucketize(observed[:, column].contiguous(), edge[1:-1])
        offsets = torch.arange(-radius, radius + 1, device=device)
        neighbors = index[:, None] + offsets
        valid_neighbors = (neighbors >= 0) & (neighbors < bins)
        safe = neighbors.clamp(0, bins - 1)
        lower[:, column], upper[:, column] = edge[safe], edge[safe + 1]
        # -1 marks an offset beyond the finite range; 0/1 are lower-edge rules.
        inclusive[:, column] = torch.where(valid_neighbors, (safe == 0).to(torch.int32), -1)
        if radius:
            kernel_weights = torch.exp(-.5 * (offsets.float() / float(smoothing_sigma)) ** 2)
            weights = kernel_weights[None, :] * valid_neighbors
            weights = weights / weights.sum(-1, keepdim=True)
        valid &= (observed[:, column] >= edge[0]) & (observed[:, column] <= edge[-1])
        volume *= edge[1] - edge[0]
    slots = diag_slots(kernel)
    counts = torch.zeros((len(rows), subjects, trials, count_width), dtype=torch.int32, device=device)
    diagnostics = torch.zeros((*counts.shape[:-1], len(slots) + 1), dtype=torch.int64, device=device)
    params, strides = _param_tensors(torch, ir, rows, device, num_subjects=subjects,
                                   num_trials=trials, subject_slices=subject_slices)
    lca_steps = lca_max_steps(ir, prepared, rows)
    _check_step_caps(max_steps=ir.max_steps, lca_max_steps=lca_steps)
    emitter = HistogramEmitter(kernel, indices, categorical, radius, normal_rng=launch["normal_rng"],
                               trial_schedule=launch["trial_schedule"])
    source = emitter.emit()
    dummy = torch.empty(1, device=device)
    with interpret_scope(interpret):
        module = load_triton_kernel_module(source, "free_running_histogram", ir.model_kind, interpret=interpret)
        grid = (len(rows) * subjects * triton.cdiv(num_estimates, launch["block_size"]),)
        getattr(module, emitter._kernel_name())[grid](
            *input_tensors, *params, *strides, counts, diagnostics, dummy, dummy, False, False,
            len(rows) * subjects * num_estimates, subjects, num_estimates, trials,
            LCA_MAX_STEPS=lca_steps, MAX_STEPS=ir.max_steps, COMMON_RANDOM=bool(common_random_numbers),
            SEED=0 if seed is None else int(seed), TRIAL_OFFSET=0, RNG_NUM_TRIALS=trials,
            BLOCK=launch["block_size"], observed=observed, lower=lower, upper=upper,
            lower_inclusive=inclusive, observed_valid=valid, **_compiler_launch_options(launch),
        )
    diagnostics = diagnostics.cpu().numpy()
    nonfinite = int(diagnostics[..., -1].sum())
    if nonfinite:
        raise BatchedNumericalError(f"Batched simulation produced {nonfinite} NaN or infinite outcome value(s) on backend '{plan.backend}'.")
    truncated = {}
    for i, (node, _) in enumerate(slots):
        fraction = float(diagnostics[..., i].sum()) / (len(rows) * subjects * trials * num_estimates)
        truncated[node] = truncated.get(node, 0.) + fraction
    _report_truncation(truncated, ir.max_steps, strict_truncation)
    cardinalities = _categorical_cardinalities(data, categorical, categorical_cardinalities) if pseudocount else ()
    joint_bins = float(bins ** len(numeric) * np.prod(cardinalities))
    weighted = counts[..., 0].to(torch.float32) if weights is None else (counts.to(torch.float32) * weights).sum(-1)
    # The edge-normalized weights sum to one, so smoothing the symmetric prior
    # adds exactly one pseudocount here. Its denominator covers every joint bin.
    density = (weighted + pseudocount) / ((num_estimates + pseudocount * joint_bins) * volume)
    density = torch.clamp(density, min=ZERO_PROB).reshape(len(rows) * subjects, trials).cpu().numpy()
    totals = np.asarray(_sum_histogram_log_likelihood(density, include_mask)).reshape(len(rows), subjects).sum(1)
    return float(totals[0]) if len(rows) == 1 else totals
