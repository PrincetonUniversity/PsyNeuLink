"""Observed-history filtering must carry Dawa's noisy states and controls."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.backend.triton.state import retained_control_layout


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
    pytest.mark.triton,
    pytest.mark.triton_gpu,
]


@pytest.fixture(scope="module")
def dawa_plan():
    path = (
        Path(__file__).resolve().parents[3]
        / "Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py"
    )
    spec = importlib.util.spec_from_file_location("dawa_conditioned_test_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model, inputs, outputs = module.build_model(
        trials=4, c_noise=0.1, s_noise=0.1, d_noise=0.1, r_noise=0.1
    )
    inputs[module.node(model, "Task Input")] = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    )
    plan = BatchedCompositionCompiler.compile(
        model, backend="triton", outputs=outputs, max_steps=1000
    )
    varying = {
        ("LC", "mode"): [0.2, 0.8, 0.3, 0.7],
        ("Bias Mechanism", "intercept"): [-0.45, -0.4, -0.48, -0.43],
        ("w1 Mechanism", "intercept"): [1.0, 1.3, 0.8, 1.1],
        ("w2 Mechanism", "intercept"): [1.2, 0.9, 1.1, 0.8],
        ("RT_GATE", "intercept"): [0.2, 0.1, 0.3, 0.15],
    }
    parameters = {
        f"{module.node(model, name).name}.{parameter}": BatchedTrialParameter(values)
        for (name, parameter), values in varying.items()
    }
    return plan, inputs, parameters


def _slice_parameters(parameters, trial):
    return {
        name: float(value.values[trial])
        if isinstance(value, BatchedTrialParameter)
        else value
        for name, value in parameters.items()
    }


def _launch(schedule):
    return dict(
        block_size=32,
        num_warps=1,
        trial_schedule=schedule,
        normal_rng="philox4x_fast_v1",
    )


@pytest.mark.parametrize("specialized", [False, True])
@pytest.mark.parametrize("schedule", ["synchronized", "independent"])
def test_dawa_split_resume_preserves_noisy_states_and_sampled_controls(
    dawa_plan, schedule, specialized
):
    plan, inputs, parameters = dawa_plan
    if specialized:
        plan = plan.specialize_parameters(
            {p.name: p.default for p in plan.ir.params if p.name not in parameters}
        )
    options = dict(
        seed=29,
        strict_truncation=True,
        return_final_states=True,
        triton_launch_options=_launch(schedule),
    )
    candidates = [parameters, parameters]
    full = plan.run(inputs, candidates, 37, **options)
    state = None
    split_values = []
    for trial in range(4):
        result = plan.run(
            {
                node: np.asarray(value)[trial : trial + 1]
                for node, value in inputs.items()
            },
            [_slice_parameters(parameters, trial)] * 2,
            37,
            initial_states=state,
            rng_sequence_trials=4,
            rng_trial_offset=trial,
            **options,
        )
        split_values.append(result.values)
        state = result.metadata["final_states"]
    np.testing.assert_array_equal(np.concatenate(split_values, axis=2), full.values)
    np.testing.assert_array_equal(state, full.metadata["final_states"])
    controls, width = retained_control_layout(plan.ir.graph)
    mechanism_width = sum(item.width for item in plan.ir.graph.states)
    assert width == mechanism_width + 2 * len(plan.kernel_ir.effective_parameters)
    assert state.shape[-1] == width
    assert np.isfinite(state[..., list(controls.values())]).all()
    # A mechanism-only state buffer used to silently omit held modulation and
    # the sampled values used by next-trial reset functions.
    with pytest.raises(ValueError, match="initial_states must have shape"):
        plan.run(
            {node: np.asarray(value)[:1] for node, value in inputs.items()},
            [_slice_parameters(parameters, 0)] * 2,
            37,
            initial_states=state[..., :mechanism_width],
            **options,
        )


def test_dawa_resetting_controls_after_trial_changes_next_prediction(dawa_plan):
    plan, inputs, parameters = dawa_plan
    options = dict(
        seed=29,
        strict_truncation=True,
        return_final_states=True,
        rng_sequence_trials=4,
        triton_launch_options=_launch("independent"),
    )
    first = plan.run(
        {node: np.asarray(value)[:1] for node, value in inputs.items()},
        [_slice_parameters(parameters, 0)],
        37,
        **options,
    )
    state = first.metadata["final_states"]
    # Reproduce the omitted-control-state bug while retaining every mechanism
    # state: a fresh launch used to initialize these values from defaults.
    fresh_controls = state.copy()
    offsets, _ = retained_control_layout(plan.ir.graph)
    row = _slice_parameters(parameters, 1)
    for op in plan.kernel_ir.ops:
        if op.kind != "InitializeEffectiveParameter":
            continue
        fresh_controls[..., offsets[op.outputs[0].name]] = op.attrs[
            "initial_modulation_value"
        ][0]
        if "sampled_base_parameter_id" in op.attrs:
            parameter = plan.kernel_ir.params[op.attrs["sampled_base_parameter_id"]]
            fresh_controls[..., offsets[op.outputs[1].name]] = row.get(
                parameter.name, parameter.default
            )
    second_inputs = {node: np.asarray(value)[1:2] for node, value in inputs.items()}
    proper = plan.run(
        second_inputs, [row], 37, initial_states=state, rng_trial_offset=1, **options
    )
    omitted = plan.run(
        second_inputs,
        [row],
        37,
        initial_states=fresh_controls,
        rng_trial_offset=1,
        **options,
    )
    assert not np.array_equal(proper.values, omitted.values)


def test_dawa_earlier_observation_changes_later_filtered_prediction(
    dawa_plan, monkeypatch
):
    plan, inputs, parameters = dawa_plan
    plan = plan.specialize_parameters(
        {p.name: p.default for p in plan.ir.params if p.name not in parameters}
    )
    options = dict(
        seed=29, strict_truncation=True, triton_launch_options=_launch("independent")
    )
    reference = plan.run(inputs, [parameters], 1024, **options).values[0, 0]
    observed = reference[:, 0, :].copy()
    first = reference[0]
    same_choice = first[first[:, 0] == observed[0, 0], 1]
    observed[0, 1] = np.quantile(same_choice, 0.2)
    changed = observed.copy()
    changed[0, 1] = np.quantile(same_choice, 0.8)
    assert changed[0, 1] > observed[0, 1]
    likelihood_options = dict(
        categorical_dims=[0],
        bins=100,
        bin_range=[(0.0, 3.0)],
        smoothing_sigma=0.5,
        pseudocount=0.01,
        categorical_cardinalities=[2],
        include_mask=[False, True, False, False],
        **options,
    )
    # The trial-marginal objective is insensitive to an earlier unscored
    # observation; the conditional objective must still assimilate that row.
    marginal = [
        plan.histogram_likelihood(
            inputs, [parameters], 1024, data=data, **likelihood_options
        )
        for data in (observed, changed)
    ]
    np.testing.assert_array_equal(marginal[0][..., 1:], marginal[1][..., 1:])
    runs = []
    original_run = type(plan).run

    def capture(self, *args, **kwargs):
        result = original_run(self, *args, **kwargs)
        runs.append(
            (
                result.values.detach().cpu().numpy().copy(),
                None
                if kwargs.get("initial_states") is None
                else kwargs["initial_states"].detach().cpu().numpy().copy(),
            )
        )
        return result

    monkeypatch.setattr(type(plan), "run", capture)
    scores = [
        plan.conditioned_log_likelihood(
            inputs,
            [parameters],
            1024,
            data=data,
            execution="reference",
            **likelihood_options,
        )
        for data in (observed, changed)
    ]
    assert np.isfinite(scores).all()
    np.testing.assert_array_equal(runs[0][0], runs[4][0])
    assert not np.array_equal(runs[1][1], runs[5][1])
    assert not np.array_equal(runs[1][0], runs[5][0])
    assert not np.isclose(scores[0], scores[1], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("schedule", ["synchronized", "independent"])
@pytest.mark.parametrize("common_random", [False, True])
def test_dawa_prepared_conditioned_execution_matches_reference(
    dawa_plan, schedule, common_random
):
    plan, inputs, parameters = dawa_plan
    plan = plan.specialize_parameters(
        {p.name: p.default for p in plan.ir.params if p.name not in parameters}
    )
    mode = next(name for name in parameters if name.endswith(".mode"))
    # Exercise both [trial] and [subject, trial] candidate parameters together.
    candidates = [
        parameters,
        {**parameters, mode: BatchedTrialParameter([[0.7, 0.3, 0.8, 0.2]])},
    ]
    data = plan.run(
        inputs,
        [parameters],
        1,
        seed=31,
        strict_truncation=True,
        triton_launch_options=_launch(schedule),
    ).values[0, 0, :, 0]
    options = dict(
        categorical_dims=[0],
        bins=100,
        bin_range=[(0.0, 3.0)],
        smoothing_sigma=0.5,
        pseudocount=0.01,
        categorical_cardinalities=[2],
        include_mask=[False, True, True, True],
        seed=29,
        strict_truncation=True,
        triton_launch_options=_launch(schedule),
        common_random_numbers=common_random,
        return_diagnostics=True,
    )
    reference, ref_diagnostics = plan.conditioned_log_likelihood(
        inputs, candidates, 128, data=data, execution="reference", **options
    )
    prepared, diagnostics = plan.conditioned_log_likelihood(
        inputs, candidates, 128, data=data, execution="prepared", **options
    )
    np.testing.assert_array_equal(prepared, reference)
    for name in (
        "per_trial_densities",
        "effective_sample_size",
        "prior_mixture_fraction",
        "zero_support",
    ):
        np.testing.assert_array_equal(
            diagnostics[name], ref_diagnostics[name], err_msg=name
        )
