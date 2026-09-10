"""Explicit approximate history timing must not weaken exact endpoint scoring."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import (
    EndpointExpression as Expr, EndpointReconstructionError,
    ObservationField, ObservationSpec, StochasticSamplingError,
)
from psyneulink.core.batched.endpoints import _ceil_history_counts
from psyneulink.core.batched.backend.triton.csi_deterministic import _observed_endpoint_steps
from test_batched_endpoints import _ddm_plan
from test_batched_reduced_scoring import scoring_case


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.mark.parametrize("dt", [.01, .001])
def test_affine_ceiling_matches_legacy_snap_and_cell_crossings(dt):
    rng = np.random.default_rng(2718)
    size = 1000
    ndt = rng.uniform(.1, .3, size).astype(np.float32).astype(float)
    cue = rng.integers(0, 200, size)
    step = float(np.float32(dt))
    expr = Expr("add", (Expr("parameter", identity=0), Expr("add", (
        Expr("multiply", (Expr("count"), Expr("constant", value=dt))),
        Expr("multiply", (Expr("input", identity=1), Expr("constant", value=dt))),
    ))))
    observed = ndt + cue * step + rng.uniform(2., 300., size) * step
    observed[:100] = np.float32(ndt[:100] + (cue[:100] + 30) * step)
    observed[100:200] = np.nextafter(np.float32(ndt[100:200] + (cue[100:200] + 30) * step), np.float32(np.inf))
    expected = _observed_endpoint_steps(observed, ndt, cue, step, step)
    actual = _ceil_history_counts(expr, {0: ndt}, {1: cue.astype(float)}, observed, 1, 12000)
    np.testing.assert_array_equal(actual, expected)


def test_public_policy_accepts_off_lattice_data_and_labels_witness():
    exact, ddm = _ddm_plan()
    spec = ObservationSpec((ObservationField(ddm.output_ports[0], "counting"),
                            ObservationField(ddm.output_ports[1], "lebesgue", role="event_time", history_timing="ceil_fp32_8ulp")))
    projected = exact.simulation_plan.compile_observed_endpoints(spec)
    inputs, data = {ddm: [[1.]]}, [[1., .2573]]
    with pytest.raises(EndpointReconstructionError):
        exact.reconstruct(inputs, data)
    np.testing.assert_array_equal(projected.reconstruct(inputs, data, {"non_decision_time": .2037}), [[[6]]])
    assert projected.witnesses[0].guarantee == "declared_affine_ceiling_history_projection"
    with pytest.raises(ValueError, match="Exhaustive"):
        projected.reconstruct(inputs, data, method="exhaustive")


@pytest.mark.parametrize("observed,offset,slope,code", [
    (.2, .2, .01, "endpoint.projected_count_below_minimum"),
    (.1, .2, .01, "endpoint.projected_count_below_minimum"),
    (10., .2, .01, "endpoint.projected_count_above_cap"),
    (.3, .2, 0., "endpoint.ceiling_direction"),
    (.3, .2, -.01, "endpoint.ceiling_direction"),
])
def test_ceiling_does_not_silently_clamp_unsupported_history(observed, offset, slope, code):
    expr = Expr("add", (Expr("parameter", identity=0), Expr("multiply", (Expr("count"), Expr("parameter", identity=1)))))
    with pytest.raises(EndpointReconstructionError) as error:
        _ceil_history_counts(expr, {0: np.array([offset]), 1: np.array([slope])}, {}, np.array([observed]), 1, 32)
    assert error.value.code == code


def test_projected_histogram_matches_materialized_but_cannot_be_exact_mass(scoring_case):
    original, inputs, data, rows = scoring_case
    history = original.sampler.path_plan.history_plan
    spec = replace(history.observations, fields=tuple(
        replace(field, history_timing="ceil_fp32_8ulp") if field.role == "event_time" else field
        for field in history.observations.fields
    ))
    sampler = history.simulation_plan.compile_history_replay(spec).compile_boundary_trajectories().compile_stochastic_sampler().compile_observation_sampler()
    with pytest.raises(StochasticSamplingError) as error:
        sampler.compile_empirical_mass()
    assert error.value.code == "mass.history_policy"
    plan = sampler.compile_histogram_score(categorical_dims=[0], bins=13, smoothing_sigma=.5, pseudocount=.1)
    observed = data.astype(float)
    observed[:, 1] += .0037
    result = plan.score(inputs, observed, rows, num_estimates=37, execution="window", common_random_numbers=False)
    reference = plan.score(inputs, observed, rows, num_estimates=37, reference=True, common_random_numbers=False)
    np.testing.assert_array_equal(result.bin_counts, reference.bin_counts)
    np.testing.assert_array_equal(result.log_likelihood, reference.log_likelihood)
    assert result.history_timing == "ceil_fp32_8ulp"
    report = history.simulation_plan.diagnose_likelihood(spec)
    assert any(item.code == "observation.projected_history" for item in report.obligations)


def test_history_policy_declaration_is_explicit():
    for kwargs in (dict(history_timing="ceil"), dict(history_timing="ceil_fp32_8ulp"),
                   dict(role="event_time", recording="rounded", precision=.01, history_timing="ceil_fp32_8ulp")):
        with pytest.raises(ValueError):
            ObservationField(object(), "counting", **kwargs)


def test_ceiling_is_derived_from_affine_expression_not_a_ddm_formula():
    # A rescaled and shifted observation still defines its own count lattice.
    base = Expr("add", (Expr("constant", value=.2), Expr("multiply", (
        Expr("count"), Expr("constant", value=.01)))))
    transformed = Expr("add", (Expr("constant", value=1.), Expr("multiply", (
        Expr("constant", value=2.), base))))
    np.testing.assert_array_equal(_ceil_history_counts(transformed, {}, {}, np.array([1.511]), 1, 100), [6])
    nonlinear = Expr("multiply", (Expr("count"), Expr("count")))
    with pytest.raises(EndpointReconstructionError) as error:
        _ceil_history_counts(nonlinear, {}, {}, np.array([9.]), 1, 100)
    assert error.value.code == "endpoint.ceiling_nonaffine"


def test_ceiling_eight_ulp_snap_boundary():
    expr = Expr("add", (Expr("constant", value=.5), Expr("multiply", (
        Expr("count"), Expr("constant", value=.125)))))
    endpoint = np.float32(1.)
    ulp = float(np.spacing(endpoint))
    observed = endpoint + np.array([0., 8., 9.]) * ulp
    np.testing.assert_array_equal(_ceil_history_counts(expr, {}, {}, observed, 1, 100), [4, 4, 5])
