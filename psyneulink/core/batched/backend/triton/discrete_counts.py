"""Exact finite-support reductions of stateful simulator outputs."""

from dataclasses import dataclass
import math

import numpy as np

from psyneulink.core.batched.backend.triton.free_running_score import HistogramEmitter, supports_fused_histogram
from psyneulink.core.batched.errors import BatchedNumericalError
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.likelihood import _as_categorical_mask
from psyneulink.core.batched.prep import normalize_parameter_sets, prepare_inputs, lca_max_steps


@dataclass
class DiscreteOutputCounts:
    counts: object
    support: object
    num_estimates: int
    valid_candidates: np.ndarray


class DiscreteCountEmitter(HistogramEmitter):
    def __init__(self, *args, support_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_size = support_size

    @property
    def histogram_diag_width(self):
        return self.diag_slot_count + 2

    def _signature_args(self):
        return (*super()._signature_args(), "support", "block_seeds", "block_rows")

    def _do_not_specialize_args(self):
        return (*super()._do_not_specialize_args(), "block_rows")

    def _emit_lane_decode(self):
        super()._emit_lane_decode()
        # A separate launch dimension adds independent blocks without changing
        # candidate/subject/estimate addresses used by the RNG or parameters.
        self.builder.line("SEED = tl.load(block_seeds + tl.program_id(1))")
        self.builder.line("out += tl.program_id(1).to(tl.int64) * block_rows * " + str(self.support_size))
        self.builder.line("diag += tl.program_id(1).to(tl.int64) * block_rows * " + str(self.histogram_diag_width))

    def _emit_trial_end_inspection(self):
        width = len(self.indices)
        self.builder.line("hist_match = mask")
        for column, index in enumerate(self.indices):
            if self.categorical[column]:
                observed = self._histogram_load(f"observed + trial_idx * {width} + {column}")
                self.builder.line(f"hist_match = hist_match & (tl.abs({self.histogram_outputs[index]} - {observed}) <= 1.0e-6)")
        value = self.histogram_outputs[self.indices[self.categorical.index(False)]]
        self.builder.line("support_lo = tl.zeros((BLOCK,), tl.int32)")
        self.builder.line(f"support_hi = tl.full((BLOCK,), {self.support_size - 1}, tl.int32)")
        with self.builder.block(f"for support_iteration in range({math.ceil(math.log2(self.support_size))})"):
            self.builder.line("support_mid = (support_lo + support_hi) // 2")
            self.builder.line("support_key = tl.load(support + support_mid)")
            self.builder.line(f"support_right = support_key < ({value})")
            self.builder.line("support_lo = tl.where((support_lo < support_hi) & support_right, support_mid + 1, support_lo)")
            self.builder.line("support_hi = tl.where(~support_right, support_mid, support_hi)")
        self.builder.line(f"support_hit = ({value}) == tl.load(support + support_lo)")
        # Different estimates hit different addresses even with synchronized trials.
        self.builder.line(f"tl.atomic_add(out + hist_row * {self.support_size} + support_lo, 1, mask=hist_match & support_hit, sem='relaxed')")
        self._emit_histogram_add(f"diag + hist_row * {self.histogram_diag_width} + {self.diag_slot_count}", "hist_nonfinite")
        self._emit_histogram_add(f"diag + hist_row * {self.histogram_diag_width} + {self.diag_slot_count + 1}", "(~support_hit).to(tl.int32)")
        if self.stop_truncated:
            self.builder.line("trial_idx = tl.where(mask & hist_truncated, num_trials - 1, trial_idx)")


def discrete_output_counts(plan, inputs, parameter_sets, num_estimates, data, categorical_dims, *, support,
                           outcome_indices=None, subject_slices=None, seed=None, common_random_numbers=True,
                           invalid_candidates="raise", triton_launch_options=None):
    return discrete_output_count_blocks(
        plan, inputs, parameter_sets, num_estimates, data, categorical_dims, support=support,
        seeds=[0 if seed is None else int(seed)], outcome_indices=outcome_indices, subject_slices=subject_slices,
        common_random_numbers=common_random_numbers, invalid_candidates=invalid_candidates,
        triton_launch_options=triton_launch_options,
    )[0]


def discrete_output_count_blocks(plan, inputs, parameter_sets, num_estimates, data, categorical_dims, *, support,
                                seeds, outcome_indices=None, subject_slices=None, common_random_numbers=True,
                                invalid_candidates="raise", triton_launch_options=None):
    from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
    from psyneulink.core.batched.backend.triton.runtime import (
        _check_step_caps, _compiler_launch_options, _import_torch_triton, _input_tensors,
        _normalize_launch_options, _param_tensors, _report_truncation,
    )

    if not supports_fused_histogram(plan, 0):
        raise ValueError("Discrete output counts require a supported stateful Triton graph")
    if invalid_candidates not in ("raise", "nan"):
        raise ValueError("invalid_candidates must be 'raise' or 'nan'")
    if isinstance(num_estimates, bool) or not isinstance(num_estimates, (int, np.integer)) or not 0 < num_estimates < 2**31:
        raise ValueError("num_estimates must be a positive int32 count")
    seeds = tuple(seeds)
    if not seeds or any(isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer))
                        or not 0 <= s < 2**64 for s in seeds):
        raise ValueError("seeds must be a nonempty sequence of unsigned 64-bit integers")
    values = np.asarray(support, dtype=np.float32)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("support must contain finite, strictly increasing, distinct FP32 values")
    interpret = plan.backend == "triton_cpu"
    device = "cpu" if interpret else "cuda"
    torch, triton = _import_torch_triton(interpret)
    launch = _normalize_launch_options(triton_launch_options, interpret=interpret)
    ir, kernel = plan.ir, plan.kernel_ir
    rows = normalize_parameter_sets(parameter_sets, ir)
    subject_slices = None if subject_slices is None else tuple(subject_slices)
    prepared = prepare_inputs(ir, inputs, subject_slices, parameter_sets=rows, component_bindings=plan.component_bindings)
    input_tensors = _input_tensors(torch, ir.graph, prepared, device)
    subjects, trials = input_tensors[0].shape[:2]
    width = sum(output.width for output in kernel.outputs)
    indices = tuple(range(width)) if outcome_indices is None else tuple(outcome_indices)
    if not indices or any(not isinstance(i, (int, np.integer)) or not 0 <= i < width for i in indices):
        raise ValueError("outcome_indices must select valid output columns")
    categorical = _as_categorical_mask(categorical_dims, len(indices))
    if np.count_nonzero(~categorical) != 1:
        raise ValueError("Discrete counts require exactly one numeric output")
    data = np.asarray(data, dtype=float)
    if data.shape != (trials, len(indices)) or not np.isfinite(data).all():
        raise ValueError("Data must be finite and match trial and selected outcome axes")
    slots = diag_slots(kernel)
    counts = torch.zeros((len(seeds), len(rows), subjects, trials, len(values)), dtype=torch.int32, device=device)
    diagnostics = torch.zeros((*counts.shape[:-1], len(slots) + 2), dtype=torch.int64, device=device)
    support_tensor = torch.tensor(values, device=device)
    # Preserve both Philox seed words, including seeds greater than 2**32.
    block_seeds = torch.tensor(np.asarray(seeds, dtype=np.uint64).view(np.int64), device=device)
    observed = torch.tensor(data, dtype=torch.float32, device=device).contiguous()
    params, strides = _param_tensors(torch, ir, rows, device, num_subjects=subjects,
                                   num_trials=trials, subject_slices=subject_slices)
    lca_steps = lca_max_steps(ir, prepared, rows)
    _check_step_caps(max_steps=ir.max_steps, lca_max_steps=lca_steps)
    emitter = DiscreteCountEmitter(kernel, indices, categorical, support_size=len(values),
                                  normal_rng=launch["normal_rng"], trial_schedule=launch["trial_schedule"],
                                  stop_truncated=invalid_candidates == "nan")
    dummy = torch.empty(1, device=device)
    with interpret_scope(interpret):
        module = load_triton_kernel_module(emitter.cached_source(len(values)), "discrete_output_counts", ir.model_kind, interpret=interpret)
        grid = (len(rows) * subjects * triton.cdiv(num_estimates, launch["block_size"]), len(seeds))
        getattr(module, emitter._kernel_name())[grid](
            *input_tensors, *params, *strides, counts, diagnostics, dummy, dummy, False, False,
            len(rows) * subjects * num_estimates, subjects, num_estimates, trials,
            LCA_MAX_STEPS=lca_steps, MAX_STEPS=ir.max_steps, COMMON_RANDOM=bool(common_random_numbers),
            SEED=0, TRIAL_OFFSET=0, RNG_NUM_TRIALS=trials,
            BLOCK=launch["block_size"], observed=observed, lower=dummy, upper=dummy,
            lower_inclusive=dummy, observed_valid=dummy, support=support_tensor,
            block_seeds=block_seeds, block_rows=len(rows) * subjects * trials,
            **_compiler_launch_options(launch),
        )
    diagnostic = diagnostics.cpu().numpy()
    if np.any(diagnostic[..., -2]):
        raise BatchedNumericalError("Discrete output reduction encountered nonfinite simulation outputs")
    if np.any(diagnostic[..., -1]):
        raise BatchedNumericalError("A simulated numeric output is outside the declared exact support")
    valid = ~np.any(diagnostic[..., :-2] != 0, axis=(2, 3, 4))
    if invalid_candidates == "raise":
        truncated = {}
        for i, (node, _) in enumerate(slots):
            fraction = float(diagnostic[..., i].sum()) / (len(seeds) * len(rows) * subjects * trials * num_estimates)
            truncated[node] = truncated.get(node, 0.) + fraction
        _report_truncation(truncated, ir.max_steps, True)
    return tuple(DiscreteOutputCounts(counts[b], support_tensor, num_estimates, valid[b]) for b in range(len(seeds)))
