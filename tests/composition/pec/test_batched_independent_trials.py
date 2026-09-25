"""Independent trial progression must preserve the synchronized simulator."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched import likelihood
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source
from psyneulink.core.batched.errors import BatchedNumericalError
from test_batched_vector_normal import _persistent_model
from test_batched_fused_histogram import _model as _histogram_model
from test_batched_dynamic_terminator_acceptance import _build_scheduled_terminator
from test_batched_csi_coevolving_acceptance import (
    _model as _csi_model,
    _compile_csi,
    registered_csi_drift_rate,  # noqa: F401 -- imported pytest fixture
)
from test_batched_parallel_dynamic_schedules import _build_parallel_controlled_chains


pytestmark = [pytest.mark.batched, pytest.mark.composition, pytest.mark.triton]


def _options(backend, schedule, **kwargs):
    return {**({"block_size": 32, "num_warps": 1} if backend == "triton" else {}),
            "trial_schedule": schedule, **kwargs}


@pytest.mark.parametrize("normal_rng", ["legacy", "philox4x_v1"])
@pytest.mark.parametrize("common_random", [True, False])
def test_independent_lca_exact_outputs_state_subjects_and_rng(batched_backend, normal_rng, common_random):
    plan, lca = _persistent_model(batched_backend)
    rows = np.array([[3., 1., 1., 1., 1.], [1., 3., 1., 1., 1.], [1., 1., 3., 1., 1.]] * 4)
    candidates = [{}, {f"{lca.name}.gain": BatchedTrialParameter(np.linspace(.9, 1.2, 12))}]
    kwargs = dict(inputs={lca: rows}, parameter_sets=candidates, num_estimates=37,
                  subject_slices=[slice(0, 6), slice(6, 12)], seed=29,
                  common_random_numbers=common_random, return_final_states=True, strict_truncation=True)
    reference = plan.run(**kwargs, triton_launch_options=_options(batched_backend, "synchronized", normal_rng=normal_rng))
    actual = plan.run(**kwargs, triton_launch_options=_options(batched_backend, "independent", normal_rng=normal_rng))
    np.testing.assert_array_equal(actual.values, reference.values)
    np.testing.assert_array_equal(actual.metadata["final_states"], reference.metadata["final_states"])
    assert actual.metadata["truncation"] == reference.metadata["truncation"]


def test_independent_sequence_resume_and_launch_geometry(batched_backend):
    plan, lca = _persistent_model(batched_backend)
    rows = np.array([[3., 1., 1., 1., 1.], [1., 3., 1., 1., 1.], [1., 1., 3., 1., 1.]])
    options = dict(seed=29, strict_truncation=True, return_final_states=True,
                   triton_launch_options=_options(batched_backend, "independent"))
    full = plan.run({lca: rows}, [{}, {}], 37, **options)
    first = plan.run({lca: rows[:1]}, [{}, {}], 37, rng_sequence_trials=3, **options)
    last = plan.run({lca: rows[1:]}, [{}, {}], 37, rng_sequence_trials=3, rng_trial_offset=1,
                    initial_states=first.metadata["final_states"], **options)
    np.testing.assert_array_equal(np.concatenate((first.values, last.values), axis=2), full.values)
    np.testing.assert_array_equal(last.metadata["final_states"], full.metadata["final_states"])
    if batched_backend == "triton":
        options["triton_launch_options"].update(block_size=128, num_warps=4)
        wide = plan.run({lca: rows}, [{}, {}], 37, **options)
        np.testing.assert_array_equal(wide.values, full.values)


@pytest.mark.parametrize("sigma", [0., .5, 1.])
@pytest.mark.parametrize("common_random", [True, False])
def test_independent_fused_counts_smoothing_and_prior(batched_backend, sigma, common_random, monkeypatch):
    plan, lca = _histogram_model(batched_backend)
    inputs = {lca: [[3., 1.], [1., 3.], [2., 1.], [1., 3.], [3., 1.], [1., 2.]]}
    candidates = [{}, {f"{lca.name}.gain": BatchedTrialParameter(np.linspace(.9, 1.2, 6))}]
    densities = []
    original = likelihood._sum_histogram_log_likelihood

    def capture(values, include_mask):
        densities.append(values.copy())
        return original(values, include_mask)

    monkeypatch.setattr(likelihood, "_sum_histogram_log_likelihood", capture)
    options = dict(data=np.array([[.2, 0], [.3, 1], [2.1, 0]]), categorical_dims=[1],
                   outcome_indices=[3, 2], bins=7, bin_range=[(0., 2.)], pseudocount=.3,
                   categorical_cardinalities=[2], include_mask=[True, False, True],
                   subject_slices=[slice(0, 3), slice(3, 6)], seed=29, smoothing_sigma=sigma,
                   common_random_numbers=common_random, strict_truncation=True)
    expected = plan.log_likelihood(inputs, candidates, 37, **options)
    options["triton_launch_options"] = _options(batched_backend, "independent")
    actual = plan.log_likelihood(inputs, candidates, 37, **options)
    materialized = plan.log_likelihood(inputs, candidates, 37, fused=False, **options)
    np.testing.assert_array_equal(densities[0], densities[1])
    np.testing.assert_array_equal(expected, actual)
    np.testing.assert_allclose(densities[1], densities[2], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(actual, materialized, rtol=2e-6, atol=1e-6)


def test_independent_diagnostics_and_nonfinite_unscored_outputs(batched_backend):
    plan, lca = _histogram_model(batched_backend, max_steps=1)
    inputs = {lca: [[0., 0.], [3., 1.], [0., 0.]]}
    results = []
    for schedule in ("synchronized", "independent"):
        with pytest.warns(UserWarning, match="truncated"):
            results.append(plan.run(inputs, [{}], 37, seed=29,
                                    triton_launch_options=_options(batched_backend, schedule)))
    np.testing.assert_array_equal(results[0].values, results[1].values)
    assert results[0].metadata["truncation"] == results[1].metadata["truncation"]
    options = dict(data=[[0.]], outcome_indices=[2], include_mask=[False], smoothing_sigma=.5,
                   strict_truncation=True, triton_launch_options=_options(batched_backend, "independent"))
    with pytest.raises(BatchedTruncationError):
        plan.log_likelihood({lca: [[0., 0.]]}, [{}], 37, **options)
    options.update(data=[[0.], [0.]], include_mask=[False, False])
    with pytest.raises(BatchedNumericalError):
        plan.log_likelihood({lca: [[3., 1.], [3., 1.]]}, [{f"{lca.name}.time_step_size": 3e38}], 37, **options)


@pytest.mark.parametrize("model_kind", ["ddm", "csi", "parallel"])
def test_independent_scalar_rng_delays_and_controlled_resets(
    batched_backend, model_kind, registered_csi_drift_rate
):
    if model_kind == "ddm":
        model = _build_scheduled_terminator(noise=.25)
        plan = BatchedCompositionCompiler.compile(model.composition, backend=batched_backend,
                                                 outputs=model.outputs, max_steps=64)
        inputs = model.inputs
    elif model_kind == "csi":
        comp, inputs, outputs = _csi_model(ddm_noise=.15, iti=2, cue_values=[[0.], [3.], [1.], [0.]])
        plan = _compile_csi(comp, backend=batched_backend, outputs=outputs, max_steps=128)
    else:
        model = _build_parallel_controlled_chains(cue_values=([[0.], [3.]], [[3.], [0.]]))
        plan = BatchedCompositionCompiler.compile(model.composition, backend=batched_backend,
                                                 outputs=model.outputs, max_steps=32)
        inputs = model.inputs
    options = dict(seed=29, strict_truncation=True, return_final_states=True)
    reference = plan.run(inputs, [{}, {}], 37, **options)
    actual = plan.run(inputs, [{}, {}], 37, **options,
                      triton_launch_options=_options(batched_backend, "independent"))
    np.testing.assert_array_equal(reference.values, actual.values)
    np.testing.assert_array_equal(reference.metadata["final_states"], actual.metadata["final_states"])


@pytest.mark.triton_gpu
@pytest.mark.parametrize("mode", ["philox4x_v1", "philox4x_fast_v1"])
@pytest.mark.parametrize("seed", [17, 29])
def test_independent_full_dawa_network(mode, seed):
    path = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py"
    spec = importlib.util.spec_from_file_location("dawa_independent_trials_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    comp, inputs, outputs = module.build_model(trials=12, c_noise=.1, s_noise=.1, d_noise=.1, r_noise=.1)
    plan = BatchedCompositionCompiler.compile(comp, backend="triton", outputs=outputs, max_steps=500)
    candidates = [{}, {"LC.mode": BatchedTrialParameter(np.resize([.3, .7], 12)),
                       "RT_GATE.intercept": BatchedTrialParameter(np.linspace(.1, .3, 12))}]
    options = dict(seed=seed, strict_truncation=True, return_final_states=True)
    expected = plan.run(inputs, candidates, 37, **options, triton_launch_options=_options("triton", "synchronized", normal_rng=mode))
    actual = plan.run(inputs, candidates, 37, **options, triton_launch_options=_options("triton", "independent", normal_rng=mode))
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_array_equal(actual.metadata["final_states"], expected.metadata["final_states"])
    fixed = plan.specialize_parameters({p.name: p.default for p in plan.ir.params
                                       if p.name not in {"LC.mode", "RT_GATE.intercept"}})
    specialized = fixed.run(inputs, candidates, 37, **options,
                            triton_launch_options=_options("triton", "independent", normal_rng=mode))
    np.testing.assert_allclose(specialized.values, actual.values, rtol=2e-6, atol=1e-7)
    np.testing.assert_allclose(specialized.metadata["final_states"], actual.metadata["final_states"], rtol=2e-6, atol=1e-7)


def test_independent_trials_reject_unsupported_programs():
    simple = pnl.TransferMechanism()
    plan = BatchedCompositionCompiler.compile(pnl.Composition(pathways=simple))
    with pytest.raises(ValueError, match="stateful dynamic"):
        triton_graph_kernel_source(plan.kernel_ir, trial_schedule="independent")
    lca = pnl.LCAMechanism(input_shapes=2, noise=0., termination_threshold=2,
                          termination_measure=pnl.TimeScale.TRIAL)
    plan = BatchedCompositionCompiler.compile(pnl.Composition(pathways=lca))
    with pytest.raises(ValueError, match="dynamic"):
        triton_graph_kernel_source(plan.kernel_ir, trial_schedule="independent")
