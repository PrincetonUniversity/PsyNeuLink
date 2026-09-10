"""Conservative event-count cutoffs for a bounded histogram objective.

Use the authenticated endpoint expression, not a model name or a caller's
inverse. A cutoff certifies that every later count up to the source step cap
has a readout outside the required bins. Unsupported/unsafe arithmetic retains
full execution. This is a numerical enclosure check, not a formal certificate
or a claim about events outside the source's supported count domain.
"""

import numpy as np

from psyneulink.core.batched.endpoints import (
    EndpointReconstructionError, _evaluate_count_interval, validate_endpoint_witness,
)
from psyneulink.core.batched.prep import prepare_parameter_values


def _window_count_limits(expression, parameters, inputs, lower, upper, maximum):
    """Return a safe last count to execute for each closed observation window.

    Binary search for a suffix whose entire interval enclosure is disjoint.
    Only a successfully certified suffix can shorten execution; a loose bound
    can cost performance but cannot discard a possible contributing count.
    The closed lower edge is conservative for right-closed histogram bins.
    """
    lower, upper = np.broadcast_arrays(lower, upper)
    shape = lower.shape
    lower, upper = lower.ravel(), upper.ravel()
    left = np.ones(lower.size, dtype=np.int64)
    right = np.full(lower.size, maximum + 1, dtype=np.int64)  # Empty suffix.
    while np.any(left < right):
        indices = np.flatnonzero(left < right)
        middle = (left[indices] + right[indices]) // 2
        try:
            lo, hi, safe = _evaluate_count_interval(
                expression, {key: value[indices] for key, value in parameters.items()},
                {key: value[indices] for key, value in inputs.items()},
                middle.astype(float), np.full(len(indices), maximum, dtype=float),
            )
        except EndpointReconstructionError as error:
            if error.code != "endpoint.arithmetic_domain":
                raise
            # A batch arithmetic failure cannot authorize any new cutoff.
            break
        excluded = safe & ((lo > upper[indices]) | (hi < lower[indices]))
        right[indices[excluded]] = middle[excluded]
        left[indices[~excluded]] = middle[~excluded] + 1
    return (right - 1).reshape(shape)


def histogram_count_limits(plan, simulation, prepared_inputs, rows, edges, observed_bins, horizon):
    """Compile per-candidate/trial cutoffs; unsupported numeric fields fall back.

    The present tier only recognizes the already checked conditioning event.
    It does not invent time dependence for arbitrary numeric observations.
    """
    trials = len(observed_bins)
    shape = (len(rows), trials)
    endpoint = plan.observation_plan.witness.sampler.boundary.history.endpoint
    if endpoint.observation.column_start != plan.continuous_dim:
        return np.full(shape, horizon, dtype=np.int32)
    validate_endpoint_witness(simulation.kernel_ir, endpoint)
    buffers, _ = prepare_parameter_values(simulation.ir, rows, num_subjects=1, num_trials=trials)
    parameters = {
        spec.parameter_id: np.broadcast_to(buffer[:, None] if buffer.ndim == 1 else buffer[:, 0, :], shape).astype(float).ravel()
        for spec, buffer in zip(simulation.ir.params, buffers)
    }
    inputs = {
        item.component_id: np.broadcast_to(np.asarray(prepared_inputs[item.node][0]).reshape(-1), shape).astype(float).ravel()
        for item in simulation.ir.graph.inputs if item.width == 1
    }
    lower = np.broadcast_to(edges[np.maximum(observed_bins - plan.radius, 0)], shape)
    upper = np.broadcast_to(edges[np.minimum(observed_bins + plan.radius + 1, plan.bins)], shape)
    limits = _window_count_limits(endpoint.expression, parameters, inputs, lower, upper, simulation.ir.max_steps)
    return np.minimum(limits, horizon).astype(np.int32)
