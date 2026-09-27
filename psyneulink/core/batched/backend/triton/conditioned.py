"""Per-sequence launch preparation for observation-conditioned GPU sampling.

The ordinary generated simulator remains the execution kernel. Preparation is
local to one likelihood evaluation, so fitted parameters and mutable IR cannot
become stale through a process-global launch cache.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.cache import interpret_scope
from psyneulink.core.batched.backend.triton.runtime import (
    _check_step_caps, _compiler_launch_options, _import_torch_triton,
    _load_kernel_module, _normalize_launch_options,
)
from psyneulink.core.batched.backend.triton.state import retained_control_layout
from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION
from psyneulink.core.batched.ir import BatchedSimulationResult, BatchedTrialParameter
from psyneulink.core.batched.kernel_ir import diag_slots
from psyneulink.core.batched.prep import lca_max_steps, prepare_inputs, prepare_parameter_values


def _packed_device_views(torch, arrays):
    """Upload heterogeneous flat arrays once, preserving their individual views."""
    if not arrays:
        return []
    sizes = [array.size for array in arrays]
    storage = torch.as_tensor(np.concatenate([array.reshape(-1) for array in arrays]),
                              dtype=torch.float32, device="cuda")
    offset = 0
    views = []
    for array, size in zip(arrays, sizes):
        views.append(storage[offset:offset + size].reshape(array.shape))
        offset += size
    return views


def prepare_conditioned_runner(plan, inputs, parameter_rows, num_estimates, *, num_trials,
                               seed=None, common_random_numbers=True, triton_launch_options=None):
    """Prepare a CUDA callable ``run(trial_index, initial_states)`` once.

    ``parameter_rows`` must already be normalized by the likelihood entry point.
    One contiguous subject is supported, matching that entry point's contract.
    Returned device outcome/state buffers are reused: consumers must enqueue
    their reductions and state gathering before calling this runner again.
    """
    return _ConditionedRunner(plan, inputs, parameter_rows, num_estimates, num_trials,
                              seed, common_random_numbers, triton_launch_options)


class _ConditionedRunner:
    def __init__(self, plan, inputs, rows, estimates, trials, seed, common_random, launch_options):
        if plan.backend != "triton" or plan.kernel_ir.fusion_kind != COEVOLVING_GRAPH_FUSION:
            raise ValueError("Prepared conditioned execution requires a co-evolving CUDA plan.")
        if isinstance(estimates, bool) or not isinstance(estimates, (int, np.integer)) or estimates < 1:
            raise ValueError("num_estimates must be a positive integer.")
        if not rows:
            raise ValueError("Conditioned execution requires at least one parameter set.")
        for value in inputs.values():
            array = np.asarray(value)
            if array.ndim == 0 or array.shape[0] != trials:
                raise ValueError("Each conditioned-likelihood input must have the data trial axis "
                                 f"first (expected {trials}, got shape {array.shape}).")
        torch, triton = _import_torch_triton(False)
        if not torch.cuda.is_available():
            raise RuntimeError("The Triton batched backend requires an available CUDA device.")
        self.torch = torch
        self.plan = plan
        self.trials = trials
        self.estimates = estimates
        self.seed = 0 if seed is None else int(seed)
        self.common_random = bool(common_random)
        self.launch = _normalize_launch_options(launch_options, interpret=False)
        prepared = prepare_inputs(plan.ir, inputs, parameter_sets=rows,
                                  component_bindings=plan.component_bindings)
        graph = plan.ir.graph
        input_arrays = []
        for spec in graph.inputs:
            array = np.asarray(prepared[spec.node], dtype=np.float32)
            if array.shape[:2] != (1, trials):
                raise ValueError("Prepared conditioned execution requires one contiguous subject sequence.")
            input_arrays.append(array.reshape(trials, spec.width))
        parameter_arrays, strides = prepare_parameter_values(
            plan.ir, rows, num_subjects=1, num_trials=trials,
        )
        # A time-major layout makes each launched trial a contiguous candidate
        # vector. With one subject/trial per launch its ABI strides are (1, 0),
        # exactly like ordinary scalar parameter rows.
        packed_parameters = [array.reshape(len(rows), trials).T.copy() if strides[2 * index + 1] else array
                             for index, array in enumerate(parameter_arrays)]
        device_inputs = _packed_device_views(torch, input_arrays)
        device_parameters = _packed_device_views(torch, packed_parameters)
        self.trial_inputs = [tuple(array[index] for array in device_inputs) for index in range(trials)]
        self.trial_parameters = [tuple(array[index] if strides[2 * column + 1] else array
                                      for column, array in enumerate(device_parameters))
                                 for index in range(trials)]
        self.parameter_strides = (1, 0) * len(parameter_arrays)
        # Input-driven count limits can vary across trials (e.g. CSI's cue
        # interval). Preserve the reference launch's exact cap as well as its
        # RNG clock. Other graphs compute this constant only once.
        input_dependent_cap = any(node.attrs.get("termination_input_node") is not None for node in graph.nodes)
        input_dependent_cap |= any(item.target_parameter == "termination_threshold" for item in graph.modulations)
        if input_dependent_cap:
            self.caps = []
            for trial in range(trials):
                trial_rows = [dict((name, float(np.asarray(value.values).reshape(1, trials)[0, trial])
                                   if isinstance(value, BatchedTrialParameter) else value)
                                  for name, value in row.items()) for row in rows]
                self.caps.append(lca_max_steps(plan.ir, {name: array[:, trial:trial + 1]
                                                       for name, array in prepared.items()}, trial_rows))
        else:
            self.caps = [lca_max_steps(plan.ir, prepared, rows)] * trials
        _check_step_caps(max_steps=plan.ir.max_steps, lca_max_steps=max(self.caps))
        with interpret_scope(False):
            module = _load_kernel_module(plan.ir, kernel_ir=plan.kernel_ir, interpret=False,
                                         normal_rng=self.launch["normal_rng"], trial_schedule=self.launch["trial_schedule"])
        self.kernel = module.pnl_batched_coevolving_graph_kernel
        self.slots = diag_slots(plan.kernel_ir)
        _, state_width = retained_control_layout(graph)
        self.state_shape = (len(rows), 1, estimates, state_width)
        self.final_state = torch.empty(self.state_shape, dtype=torch.float32, device="cuda")
        self.no_initial_state = torch.empty((1,), dtype=torch.float32, device="cuda")
        self.out = torch.empty((len(rows), 1, 1, estimates, sum(item.width for item in graph.outputs)),
                               dtype=torch.float32, device="cuda")
        self.diag = (torch.empty((len(rows), 1, 1, estimates, len(self.slots)), dtype=torch.float32, device="cuda")
                     if self.slots else None)
        self.total_lanes = len(rows) * estimates
        self.grid = (triton.cdiv(self.total_lanes, self.launch["block_size"]),)
        self.compiler_options = _compiler_launch_options(self.launch)

    def __call__(self, trial_index, initial_states):
        if not 0 <= trial_index < self.trials:
            raise ValueError("Trial index is outside the prepared sequence.")
        torch = self.torch
        if initial_states is None:
            initial = self.no_initial_state
        else:
            initial = torch.as_tensor(initial_states, dtype=torch.float32, device="cuda")
            if tuple(initial.shape) != self.state_shape:
                raise ValueError(f"initial_states must have shape {self.state_shape}, got {tuple(initial.shape)}.")
            initial = initial.contiguous()
        if self.diag is not None:
            self.diag.zero_()
        with interpret_scope(False):
            self.kernel[self.grid](
                *self.trial_inputs[trial_index], *self.trial_parameters[trial_index], *self.parameter_strides,
                self.out, *(() if self.diag is None else (self.diag,)),
                initial, self.final_state, initial_states is not None, True,
                self.total_lanes, 1, self.estimates, 1,
                LCA_MAX_STEPS=self.caps[trial_index], MAX_STEPS=self.plan.ir.max_steps,
                COMMON_RANDOM=self.common_random, SEED=self.seed, TRIAL_OFFSET=trial_index,
                RNG_NUM_TRIALS=self.trials, BLOCK=self.launch["block_size"], **self.compiler_options,
            )
        checks = {
            "nonfinite_count": (~torch.isfinite(self.out)).sum(),
            "diagnostic_sums": None if self.diag is None else self.diag.reshape(-1, len(self.slots)).sum(dim=0),
            "diagnostic_count": 0 if self.diag is None else self.total_lanes,
            "slots": self.slots,
        }
        return BatchedSimulationResult(
            values=self.out, output_names=self.plan.ir.output_names, backend="triton",
            metadata={"model_kind": self.plan.ir.model_kind, "device": "cuda", "truncation": {},
                      "triton_launch_options": self.launch, "final_states": self.final_state,
                      "_deferred_device_checks": checks},
        )
