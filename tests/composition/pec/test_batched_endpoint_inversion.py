"""Fast count inversion agrees with exhaustive floating-point enclosures."""

import numpy as np
import pytest

from psyneulink.core.batched import EndpointExpression as Expr, EndpointReconstructionError
from psyneulink.core.batched import endpoints
from test_batched_endpoints import _ddm_plan


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _add(a, b):
    return Expr("add", (a, b))


def _mul(a, b):
    return Expr("multiply", (a, b))


def _outcome(expression, parameters, inputs, observed, maximum):
    try:
        return endpoints._exhaustive_count(expression, parameters, inputs, observed, 1, maximum, "test")
    except EndpointReconstructionError as error:
        return error.code


@pytest.mark.parametrize("maximum", [1, 257, 12000])
@pytest.mark.parametrize("sign", [1., -1.])
def test_batched_interval_inverse_matches_randomized_exhaustive_readouts(maximum, sign):
    rng = np.random.default_rng(8901)
    size = 48
    counts = rng.integers(1, maximum + 1, size=size)
    counts[:2] = [1, maximum]
    parameters = {
        0: rng.uniform(.001, .01, size).astype(np.float32).astype(float),
        1: rng.uniform(.15, .3, size).astype(np.float32).astype(float),
        2: (sign * rng.uniform(.5, 3., size)).astype(np.float32).astype(float),
    }
    inputs = {7: (sign * rng.uniform(.01, .1, size)).astype(np.float32).astype(float)}
    expression = _add(_mul(_add(Expr("parameter", identity=1), _mul(
        Expr("count"), Expr("parameter", identity=0))), Expr("parameter", identity=2)), Expr("input", identity=7))
    # Independent, explicitly separated FP32 evaluation of the declared gate.
    observed = np.float32(np.float32(np.float32(parameters[1]) + np.float32(
        counts.astype(np.float32) * np.float32(parameters[0]))) * np.float32(parameters[2]))
    observed = np.float32(observed + np.float32(inputs[7])).astype(float)
    observed[::7] += sign * .00031  # Include incompatible off-lattice observations.
    fast = endpoints._invert_count_intervals(expression, parameters, inputs, observed, 1, maximum)
    assert np.all(fast != 0), "Ordinary signed affine readouts should not enumerate counts"
    for i in range(size):
        expected = _outcome(expression, {k: v[i] for k, v in parameters.items()},
                            {k: v[i] for k, v in inputs.items()}, observed[i], maximum)
        assert fast[i] == ({"endpoint.count_incompatible": -1, "endpoint.count_ambiguous": -2}.get(expected, expected))


def test_enclosure_boundaries_and_adjacent_floats_match_exhaustive():
    expression = _add(Expr("parameter", identity=0), _mul(Expr("count"), Expr("parameter", identity=1)))
    scalar = {0: float(np.float32(.237)), 1: float(np.float32(.003))}
    lo, hi = endpoints._evaluate_enclosure(expression, scalar, {}, np.array([1., 33., 128.]))
    observed = np.concatenate((lo, hi, np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)))
    parameters = {key: np.full(len(observed), value) for key, value in scalar.items()}
    fast = endpoints._invert_count_intervals(expression, parameters, {}, observed, 1, 128)
    for i, value in enumerate(observed):
        expected = _outcome(expression, scalar, {}, value, 128)
        assert fast[i] == (-1 if expected == "endpoint.count_incompatible" else expected)


@pytest.mark.parametrize("expression, parameters, observed, code", [
    (_add(Expr("parameter", identity=0), _mul(Expr("count"), Expr("parameter", identity=1))),
     {0: 1e38, 1: 1e38}, 2e38, "endpoint.arithmetic_domain"),
    (_mul(_add(Expr("count"), Expr("constant", value=-4)), Expr("parameter", identity=0)),
     {0: 1e-38}, 6e-38, "endpoint.arithmetic_domain"),
    (_add(Expr("count"), _mul(Expr("count"), Expr("constant", value=-1))),
     {}, 0., "endpoint.count_ambiguous"),
])
def test_unsafe_domains_and_cancellation_retain_reference_fallback(expression, parameters, observed, code):
    # In the first two cases arithmetic at a *different* count from the
    # observation is invalid. Pruning must not hide that reference rejection.
    fast = endpoints._invert_count_intervals(expression, {k: np.array([v]) for k, v in parameters.items()},
                                             {}, np.array([observed]), 1, 300)
    np.testing.assert_array_equal(fast, [0])
    assert _outcome(expression, parameters, {}, observed, 300) == code


def test_interval_dependency_work_is_bounded():
    expression = _mul(Expr("count"), Expr("constant", value=0))
    # Every count is compatible. It is cheaper to fall back and discover the
    # ambiguity in the first reference block than expand the entire tree.
    fast = endpoints._invert_count_intervals(expression, {}, {}, np.array([0.]), 1, 2**24)
    np.testing.assert_array_equal(fast, [0])
    assert _outcome(expression, {}, {}, 0., 256) == "endpoint.count_ambiguous"


def test_unique_small_count_at_maximum_supported_cap_does_not_fall_back():
    expression = _add(Expr("parameter", identity=0), _mul(Expr("count"), Expr("parameter", identity=1)))
    parameters = {0: np.array([float(np.float32(.2))]), 1: np.array([float(np.float32(.01))])}
    fast = endpoints._invert_count_intervals(expression, parameters, {}, np.array([.21]), 1, 2**24)
    np.testing.assert_array_equal(fast, [1])


def test_mixed_safe_and_uncertified_lanes_are_partitioned():
    expression = _add(Expr("count"), Expr("parameter", identity=0))
    # Lane 1 crosses zero over the count domain and must use the reference.
    parameters = {0: np.array([10., -5.])}
    fast = endpoints._invert_count_intervals(expression, parameters, {}, np.array([12., 2.]), 1, 10)
    np.testing.assert_array_equal(fast, [2, 0])
    assert _outcome(expression, {0: -5.}, {}, 2., 10) == 7


def test_public_auto_path_is_batched_and_does_not_call_exhaustive(monkeypatch):
    plan, decision = _ddm_plan(cap=12000)
    inputs = {decision: np.ones((32, 1))}
    data = np.column_stack((np.ones(32), .2 + np.arange(1, 33) * .01))
    expected = plan.reconstruct(inputs, data, method="exhaustive")

    def forbidden(*args, **kwargs):
        pytest.fail("Well-conditioned count inversion unexpectedly enumerated the domain")

    monkeypatch.setattr(endpoints, "_exhaustive_count", forbidden)
    np.testing.assert_array_equal(plan.reconstruct(inputs, data), expected)


@pytest.mark.parametrize("observed, parameters", [
    (.255, {}), (.2, {}), (.60, {}),
    (float(2**24), {"non_decision_time": float(2**24)}),
    (.25, {"time_step_size": 0.}),
])
def test_public_modes_agree_on_rejection_codes(observed, parameters):
    plan, decision = _ddm_plan()
    codes = []
    for method in ("auto", "exhaustive"):
        with pytest.raises(EndpointReconstructionError) as error:
            plan.reconstruct({decision: [[1.]]}, [[1., observed]], parameters, method=method)
        codes.append(error.value.code)
    assert codes[0] == codes[1]


def test_unknown_inversion_method_is_rejected():
    plan, decision = _ddm_plan()
    with pytest.raises(ValueError, match="method"):
        plan.reconstruct({decision: [[1.]]}, [[1., .25]], method="round")
