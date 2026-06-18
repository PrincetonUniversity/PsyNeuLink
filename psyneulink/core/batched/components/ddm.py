"""Batched op for `DDM` with a `DriftDiffusionIntegrator` function.

This is the reference declarative mechanism op: parameters are auto-bound
from the body signature against the DriftDiffusionIntegrator Parameters, the
RNG stream is implied by the ``rng``/``seed``/``rng_base`` arguments, and the
two bodies share the declared argument order.
"""

import numpy as np

from psyneulink.core.batched.backend.triton.api import pnl_triton_op
from psyneulink.core.batched.specs import batched_op, param
from psyneulink.core.components.functions.stateful.integratorfunctions import (
    DriftDiffusionIntegrator,
)
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM


@pnl_triton_op(constexpr=("max_steps",))
def _pnl_triton_ddm_integrate(
    x,
    rate,
    noise,
    threshold,
    non_decision_time,
    time_step_size,
    starting_value,
    offset,
    seed,
    rng_base,
    max_steps,
):
    value = starting_value
    steps = tl.zeros_like(x)
    sqrt_dt = tl.sqrt(time_step_size)
    boundary_tolerance = tl.maximum(1.0e-7, threshold * 1.0e-6)
    for step in tl.range(0, max_steps, 1, loop_unroll_factor=1):
        active = tl.abs(value) + boundary_tolerance < threshold
        draw = tl.randn(seed, rng_base + step)
        updated = value + rate * x * time_step_size + noise * sqrt_dt * draw
        updated = tl.minimum(tl.maximum(updated + offset, -threshold), threshold)
        value = tl.where(active, updated, value)
        steps += tl.where(active, 1.0, 0.0)
    return tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * time_step_size


@batched_op(
    DDM,
    function=DriftDiffusionIntegrator,
    triton=_pnl_triton_ddm_integrate,
    outputs=(("DECISION_OUTCOME", 1), ("RESPONSE_TIME", 1)),
    bind={"starting_value": param("initializer", fallback="starting_value", default=0.0)},
    single_node_model_kind="ddm",
    param_alias_prefixes=("ddm", "DDM"),
)
def ddm_trial(
    x,
    rate,
    noise,
    threshold,
    non_decision_time,
    time_step_size,
    starting_value,
    offset,
    rng,
    max_steps,
):
    value = float(starting_value)
    steps = 0
    sqrt_dt = np.sqrt(time_step_size)
    boundary_tolerance = max(1e-7, abs(threshold) * 1e-6)
    for _ in range(int(max_steps)):
        if abs(value) + boundary_tolerance >= threshold:
            break
        random_draw = rng.normal()
        value = value + rate * x * time_step_size + noise * sqrt_dt * random_draw
        value = float(np.clip(value + offset, -threshold, threshold))
        steps += 1
    return (1.0 if value > 0 else 0.0), float(non_decision_time + steps * time_step_size)
