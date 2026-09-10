"""Window cutoffs conservatively retain every potentially contributing count."""

import numpy as np
import pytest

from psyneulink.core.batched import EndpointExpression as Expr
from psyneulink.core.batched.endpoints import _evaluate_enclosure
from psyneulink.core.batched.scoring_windows import _window_count_limits


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("sign", [1., -1.])
@pytest.mark.parametrize("maximum", [1, 257, 12000])
def test_cutoffs_retain_exhaustive_enclosure_hits(sign, maximum):
    rng = np.random.default_rng(2703)
    size = 32
    expr = Expr("multiply", (Expr("add", (Expr("parameter", identity=0),
                Expr("multiply", (Expr("count"), Expr("parameter", identity=1))))),
                Expr("parameter", identity=2)))
    params = {0: rng.uniform(.1, .3, size).astype(np.float32).astype(float),
              1: rng.uniform(.001, .02, size).astype(np.float32).astype(float),
              2: (sign * rng.uniform(.5, 3., size)).astype(np.float32).astype(float)}
    centers = rng.integers(1, maximum + 1, size)
    lo, hi = _evaluate_enclosure(expr, params, {}, centers)
    lower, upper = lo - .01, hi + .02
    limits = _window_count_limits(expr, params, {}, lower, upper, maximum)
    for i, limit in enumerate(limits):
        counts = np.arange(1, maximum + 1, dtype=float)
        point_lo, point_hi = _evaluate_enclosure(expr, {k: v[i] for k, v in params.items()}, {}, counts)
        possible = (point_lo <= upper[i]) & (point_hi >= lower[i])
        assert np.all(counts[possible] <= limit)
        # For ordinary affine readouts, the enclosure should also be tight.
        assert limit <= counts[possible].max(initial=0) + 1


def test_closed_bin_edges_never_discard_boundary_count():
    expr = Expr("multiply", (Expr("count"), Expr("constant", value=.01)))
    lo, hi = _evaluate_enclosure(expr, {}, {}, np.array([10.]))
    for edge in (lo, hi, np.nextafter(hi, np.inf)):
        limit = _window_count_limits(expr, {}, {}, lo, edge, 100)
        assert limit[0] >= 10


def test_empty_window_support_and_unsafe_arithmetic():
    expr = Expr("count")
    np.testing.assert_array_equal(_window_count_limits(expr, {}, {}, np.array([-2.]), np.array([-1.]), 100), [0])
    bad = Expr("multiply", (expr, Expr("constant", value=1e38)))
    np.testing.assert_array_equal(_window_count_limits(bad, {}, {}, np.array([1.]), np.array([2.]), 100), [100])
    # Dependency/cancellation cannot cause an unsound monotonicity assumption.
    cancel = Expr("add", (expr, Expr("multiply", (expr, Expr("constant", value=-1.)))))
    np.testing.assert_array_equal(_window_count_limits(cancel, {}, {}, np.array([-.1]), np.array([.1]), 100), [100])


def test_source_cap_not_requested_runtime_horizon_defines_suffix():
    # A decreasing readout can return to the window after a short runtime cap.
    expr = Expr("add", (Expr("constant", value=100.),
                        Expr("multiply", (Expr("count"), Expr("constant", value=-1.)))))
    limit = _window_count_limits(expr, {}, {}, np.array([10.]), np.array([20.]), 100)
    assert limit[0] >= 90
