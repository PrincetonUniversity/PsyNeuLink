"""Explicit parameter constants preserve dynamic fitting and control semantics."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source
from psyneulink.core.batched.prep import normalize_parameter_sets
from test_batched_vector_normal import _persistent_model
from test_batched_fused_histogram import _model as _histogram_model
from test_batched_parallel_dynamic_schedules import _build_parallel_controlled_chains

pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _linear(backend="triton_cpu"):
    m = pnl.TransferMechanism(name="fixed target", function=pnl.Linear(slope=.1))
    return BatchedCompositionCompiler.compile(pnl.Composition(pathways=m), backend=backend), m


def test_specialization_snapshot_and_parameter_resolution():
    plan, _ = _linear()
    fixed = {"fixed target.slope": .2}
    new = plan.specialize_parameters(fixed)
    fixed["fixed target.slope"] = 99.
    assert plan.fixed_parameters == {}
    assert new.fixed_parameters == {"fixed target.slope": .2}
    assert new.ir.params[0].default == plan.ir.params[0].default
    assert new.ir.params == new.kernel_ir.params
    assert normalize_parameter_sets(None, new.ir)[0]["fixed target.slope"] == .2
    # Source-derived cache keys distinguish both the specialization and value.
    sources = [triton_graph_kernel_source(p.kernel_ir) for p in
               (plan, new, plan.specialize_parameters({"fixed target.slope": .3}))]
    assert len(set(sources)) == 3
    with pytest.raises(ValueError, match="Multiple fixed"):
        plan.specialize_parameters({"fixed target.slope": .2, "prefix.fixed target.slope": .2})
    with pytest.raises(ValueError, match="Unknown batched"):
        plan.specialize_parameters({"missing": 1.})
    with pytest.raises(TypeError, match="mapping"):
        plan.specialize_parameters(["fixed target.slope"])


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e40, 1j, [1., 2.], BatchedTrialParameter([1., 1.])])
def test_specialization_rejects_invalid_constants(value):
    plan, _ = _linear()
    with pytest.raises(ValueError):
        plan.specialize_parameters({"fixed target.slope": value})


def test_specialization_rejects_ambiguous_names():
    a = pnl.TransferMechanism(name="a")
    b = pnl.TransferMechanism(name="b")
    p = BatchedCompositionCompiler.compile(pnl.Composition(pathways=[a, b]))
    params = tuple(replace(s, aliases=(*s.aliases, "shared.slope")) if s.name.endswith(".slope") else s
                   for s in p.ir.params)
    p = replace(p, ir=replace(p.ir, params=params), kernel_ir=replace(p.kernel_ir, params=params))
    with pytest.raises(ValueError, match="Ambiguous"):
        p.specialize_parameters({"shared.slope": 2.})


def test_constant_overrides_check_fp32_bits_in_every_lane_and_trial():
    p, _ = _linear()
    p = p.specialize_parameters({"fixed target.slope": .1, "fixed target.intercept": -0.})
    normalize_parameter_sets({"fixed target.slope": float(np.float32(.1))}, p.ir)
    normalize_parameter_sets({"fixed target.slope": BatchedTrialParameter([.1, .1])}, p.ir)
    for values in ({"fixed target.slope": [.1, .2]}, [{"fixed target.slope": .2}],
                   {"fixed target.slope": BatchedTrialParameter([.1, .2])}, {"fixed target.intercept": 0.},
                   {"fixed target.slope": .2, "prefix.fixed target.slope": .1}):
        with pytest.raises(ValueError, match="specialized"):
            normalize_parameter_sets(values, p.ir)
    with pytest.raises(ValueError, match="specialized"):
        p.specialize_parameters({"fixed target.slope": .2})


def test_specialization_respects_source_constraints_and_validates_ir():
    p, _ = _histogram_model("triton_cpu")
    name = next(s.name for s in p.ir.params if s.name.endswith(".time_step_size"))
    with pytest.raises(ValueError, match="must be"):
        p.specialize_parameters({name: -1.})
    params = (replace(p.ir.params[0], constant_value=float('nan')), *p.ir.params[1:])
    with pytest.raises(ValueError, match="finite"):
        replace(p.kernel_ir, params=params)


def test_specialized_nondefault_values_and_dynamic_trials(batched_backend):
    p, m = _linear(batched_backend)
    inputs = {m: [[2.], [3.], [4.]]}
    rows = [{"fixed target.slope": .2, "fixed target.intercept": BatchedTrialParameter([0., 1., 2.])}]
    expected = p.run(inputs, rows, 7).values
    a = p.specialize_parameters({"fixed target.slope": .2})
    b = p.specialize_parameters({"fixed target.slope": .3})
    np.testing.assert_allclose(a.run(inputs, rows, 7).values, expected, rtol=1e-6)
    assert not np.array_equal(a.run(inputs, None, 7).values, b.run(inputs, None, 7).values)
    # Exercise runtime validation rather than just the normalization helper.
    with pytest.raises(ValueError, match="specialized"):
        a.run(inputs, [{"fixed target.slope": .3}], 7)


@pytest.mark.parametrize("schedule", ["synchronized", "independent"])
def test_specialization_keeps_control_modulation_and_resets(batched_backend, schedule):
    model = _build_parallel_controlled_chains()
    p = BatchedCompositionCompiler.compile(model.composition, backend=batched_backend,
                                          outputs=model.outputs, max_steps=32)
    fixed = p.specialize_parameters(p.ir.param_defaults)
    options = dict(seed=29, strict_truncation=True, return_final_states=True,
                   triton_launch_options={"trial_schedule": schedule})
    expected = p.run(model.inputs, [{}], 7, **options)
    actual = fixed.run(model.inputs, [{}], 7, **options)
    np.testing.assert_allclose(actual.values, expected.values, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual.metadata['final_states'], expected.metadata['final_states'], rtol=2e-6, atol=1e-7)


def test_specialization_keeps_noisy_history_and_trial_parameters(batched_backend):
    p, lca = _persistent_model(batched_backend)
    gain = f"{lca.name}.gain"
    leak = f"{lca.name}.leak"
    fixed = p.specialize_parameters({**{s.name: s.default for s in p.ir.params if s.name != gain}, leak: .4})
    inputs = {lca: [[3., 1., 1., 1., 1.], [1., 3., 1., 1., 1.], [1., 1., 3., 1., 1.]]}
    rows = [{gain: BatchedTrialParameter([1.1, 1.2, .9]), leak: .4}, {gain: 1.2, leak: .4}]
    opts = dict(seed=29, strict_truncation=True, return_final_states=True,
                triton_launch_options={"trial_schedule": "independent"})
    expected = p.run(inputs, rows, 37, **opts)
    actual = fixed.run(inputs, rows, 37, **opts)
    np.testing.assert_allclose(actual.values, expected.values, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(actual.metadata['final_states'], expected.metadata['final_states'], rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("mode", ["philox4x_v1", "philox4x_fast_v1"])
def test_specialized_fused_scoring_matches_materialized(batched_backend, mode):
    if mode == "philox4x_fast_v1" and batched_backend == "triton_cpu":
        pytest.skip("CUDA bounded-angle intrinsics require the compiled GPU")
    p, lca = _histogram_model(batched_backend)
    p = p.specialize_parameters(p.ir.param_defaults)
    inputs = {lca: [[3., 1.], [1., 3.], [2., 1.]]}
    opts = dict(data=[[.2, 0], [.3, 1], [.4, 0]], categorical_dims=[1], outcome_indices=[3, 2],
                bins=7, bin_range=[(0., 2.)], smoothing_sigma=.5, pseudocount=1.,
                seed=29, strict_truncation=True,
                triton_launch_options={"trial_schedule": "independent", "normal_rng": mode})
    fused = p.log_likelihood(inputs, [{}], 37, **opts)
    materialized = p.log_likelihood(inputs, [{}], 37, fused=False, **opts)
    np.testing.assert_allclose(fused, materialized, rtol=2e-6, atol=1e-6)
    with pytest.raises(ValueError, match="specialized"):
        p.log_likelihood(inputs, [{f"{lca.name}.gain": 2.}], 37, **opts)
