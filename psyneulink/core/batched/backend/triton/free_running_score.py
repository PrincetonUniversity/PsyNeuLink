"""Histogram reductions on the ordinary stateful simulator's trial outputs.

Every estimate executes the full trial sequence with its own retained state.
Only output consumption changes: reduce matches and diagnostics in the kernel
instead of retaining an outcome and diagnostic array for every estimate.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.emit import TritonGraphEmitter
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.likelihood import (
    ZERO_PROB, _as_categorical_mask, _bin_edges, _categorical_cardinalities,
)
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs, lca_max_steps


class HistogramEmitter(TritonGraphEmitter):
    def __init__(self, kernel, indices, categorical):
        super().__init__(kernel)
        self.indices = tuple(indices)
        self.categorical = tuple(categorical)
        self.histogram_outputs = {}

    def _emit_lane_decode(self):
        self.builder.line("hist_group = tl.program_id(0) // tl.cdiv(num_estimates, BLOCK)")
        self.builder.line("estimate_idx = (tl.program_id(0) % tl.cdiv(num_estimates, BLOCK)) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = estimate_idx < num_estimates")
        self.builder.line("subject_idx = hist_group % num_subjects + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("param_idx = hist_group // num_subjects + tl.zeros((BLOCK,), tl.int32)")
        self.builder.line("offsets = hist_group * num_estimates + estimate_idx")

    def _emit_trial_start_inspection(self):
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
        self.builder.line(
            f"tl.atomic_add(diag + hist_row * {self.diag_slot_count + 1} + {op.attrs['slot']}, "
            f"tl.sum(tl.where(mask, {value}, 0).to(tl.int32), 0))"
        )

    def _emit_trial_end_inspection(self):
        width = len(self.indices)
        self.builder.line("hist_match = mask & (tl.load(observed_valid + trial_idx) != 0)")
        for column, index in enumerate(self.indices):
            value = self.histogram_outputs[index]
            address = f"trial_idx * {width} + {column}"
            if self.categorical[column]:
                self.builder.line(f"hist_match = hist_match & (tl.abs({value} - tl.load(observed + {address})) <= 1.0e-6)")
            else:
                self.builder.line(f"hist_lower = tl.load(lower + {address})")
                self.builder.line(f"hist_upper = tl.load(upper + {address})")
                self.builder.line(f"hist_inclusive = tl.load(lower_inclusive + {address})")
                self.builder.line(f"hist_match = hist_match & tl.where(hist_inclusive != 0, {value} >= hist_lower, {value} > hist_lower) & ({value} <= hist_upper)")
        self.builder.line("tl.atomic_add(out + hist_row, tl.sum(hist_match.to(tl.int32), 0))")
        self.builder.line(
            f"tl.atomic_add(diag + hist_row * {self.diag_slot_count + 1} + {self.diag_slot_count}, "
            "tl.sum(tl.where(mask, hist_nonfinite, 0), 0))"
        )

    def _signature_args(self):
        args = list(super()._signature_args())
        # Nonfinite checks need a status pointer even when there are no flags.
        if not self.diag_slot_count:
            args.insert(args.index("out") + 1, "diag")
        return (*args, "observed", "lower", "upper", "lower_inclusive", "observed_valid")


def supports_fused_histogram(plan, smoothing_sigma):
    return (plan.backend in ("triton", "triton_cpu") and smoothing_sigma == 0
            and plan.ir.graph is not None
            and plan.ir.graph.fusion_kind in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION))


def fused_histogram_log_likelihood(plan, inputs, parameter_sets, num_estimates, data, categorical_dims,
                                   *, outcome_indices, bins, bin_range, pseudocount, categorical_cardinalities,
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
    observed = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
    lower, upper = torch.zeros_like(observed), torch.zeros_like(observed)
    inclusive = torch.zeros_like(observed, dtype=torch.int32)
    valid = torch.ones(trials, dtype=torch.bool, device=device)
    numeric = np.flatnonzero(~categorical).tolist()
    numeric_values = observed[:, numeric]
    edges = _bin_edges(numeric_values, numeric_values, bins, bin_range, torch)
    volume = torch.tensor(1., device=device)
    for j, column in enumerate(numeric):
        edge = edges[j]
        index = torch.bucketize(observed[:, column].contiguous(), edge[1:-1])
        lower[:, column], upper[:, column] = edge[index], edge[index + 1]
        inclusive[:, column] = (index == 0).to(torch.int32)
        valid &= (observed[:, column] >= edge[0]) & (observed[:, column] <= edge[-1])
        volume *= edge[1] - edge[0]
    slots = diag_slots(kernel)
    counts = torch.zeros((len(rows), subjects, trials), dtype=torch.int32, device=device)
    diagnostics = torch.zeros((*counts.shape, len(slots) + 1), dtype=torch.int64, device=device)
    params, strides = _param_tensors(torch, ir, rows, device, num_subjects=subjects,
                                   num_trials=trials, subject_slices=subject_slices)
    lca_steps = lca_max_steps(ir, prepared, rows)
    _check_step_caps(max_steps=ir.max_steps, lca_max_steps=lca_steps)
    emitter = HistogramEmitter(kernel, indices, categorical)
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
    density = (counts.to(torch.float32) + pseudocount) / ((num_estimates + pseudocount * joint_bins) * volume)
    density = torch.clamp(density, min=ZERO_PROB).reshape(len(rows) * subjects, trials).cpu().numpy()
    totals = np.asarray(_sum_histogram_log_likelihood(density, include_mask)).reshape(len(rows), subjects).sum(1)
    return float(totals[0]) if len(rows) == 1 else totals
