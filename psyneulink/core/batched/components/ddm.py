"""Batched op for `DDM` with a `DriftDiffusionIntegrator` function.

The reference declarative mechanism op: the decorated function is the single
kernel body (run compiled on GPU, interpreted on CPU).  Its arguments are
auto-bound from the body signature against the DriftDiffusionIntegrator
Parameters; the RNG stream is implied by the ``seed``/``rng_base`` arguments.
"""

from psyneulink.core.batched.specs import batched_op, param
from psyneulink.core.components.functions.stateful.integratorfunctions import (
    DriftDiffusionIntegrator,
)
from psyneulink.library.components.mechanisms.processing.integrator.ddm import DDM


@batched_op(
    DDM,
    function=DriftDiffusionIntegrator,
    outputs=(("DECISION_OUTCOME", 1), ("RESPONSE_TIME", 1)),
    bind={"starting_value": param("initializer", fallback="starting_value", default=0.0)},
    constexpr=("max_steps",),
    single_node_model_kind="ddm",
    param_alias_prefixes=("ddm", "DDM"),
    diagnostics=("truncated",),
)
def ddm_integrate(
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
    truncated = tl.where(tl.abs(value) + boundary_tolerance < threshold, 1.0, 0.0)
    return tl.where(value > 0.0, 1.0, 0.0), non_decision_time + steps * time_step_size, truncated
