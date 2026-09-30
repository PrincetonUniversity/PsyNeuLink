"""Staged fitting must rerun whole filters and separate search and validation."""

import numpy as np
import optuna
import pytest

from psyneulink.core.components.functions.nonstateful.adaptivefit import (
    StagedConfig,
    score_candidates,
    fit_staged,
    PENALTY,
)


def run_policy(sample, *, reserved=(), **options):
    bounds = {"y": (0.0, 1.0, 0.01), "x": (0.0, 1.0, 0.01)}
    initial = {"y": 0.2, "x": 0.2}
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            x0=initial, sigma0=0.2, lr_adapt=True, popsize=4, seed=10
        ),
    )
    study.enqueue_trial(initial)
    records = []

    def log(rows, scores, elapsed, metadata):
        records.extend(
            (row, score, metadata) for row, score in zip(rows, scores, strict=True)
        )

    config = StagedConfig(
        search_estimates=4,
        reference_estimates=32,
        refine_evaluations=9,
        check_every=4,
        min_evaluations=9,
        patience=1,
        progress_tolerance=100.0,
        **options,
    )
    result = fit_staged(
        study,
        bounds,
        initial,
        sample,
        config,
        evaluations=30,
        population=4,
        simulation_seed=29,
        optimizer_seed=10,
        reserved_seeds=reserved,
        log_batch=log,
    )
    return *result, records


def test_reference_checks_ignore_low_count_bias_and_reuse_covariance():
    calls = []

    def sample(rows, count, seed):
        calls.append((np.asarray(rows), count, seed))
        return -np.square(np.asarray(rows) - 0.6).sum(-1) + (
            100.0 if count == 4 else 0.0
        )

    # Deliberately reserve the policy RNG's first draw, as well as validation seeds.
    reserved = {
        8101,
        8102,
        int(
            np.random.default_rng(np.random.SeedSequence([29, 69471])).integers(
                0, 2**31 - 1
            )
        ),
    }
    fit, detail, refinement, records = run_policy(sample, reserved=reserved)
    fitted = np.array(list(fit["fitted_params"].values()))
    assert fit["optimal_value"] == pytest.approx(-np.square(fitted - 0.6).sum())
    assert detail["search_evaluations"] == 9
    assert detail["refinement_evaluations"] == len(refinement.trials) == 9
    assert len(records) == 18
    assert detail["refinement_covariance"]["reused"]
    assert detail["refinement_covariance"]["parameter_order"] == ["x", "y"]
    assert max(score for _, score, _ in records) > 99.0
    assert {count for _, count, _ in calls} == {4, 32}
    assert all(seed not in reserved for _, _, seed in calls)
    assert all(count == 32 for _, count, seed in calls if seed != 29)
    assert {
        metadata["estimates"]
        for _, _, metadata in records
        if metadata["phase"] == "staged_search"
    } == {4}
    assert {
        metadata["estimates"]
        for _, _, metadata in records
        if metadata["phase"] == "refinement"
    } == {32}
    assert all(check["reference_score"] <= 0 for check in detail["checkpoints"])
    assert "convergence not asserted" in detail["search_stop_reason"]
    selection = detail["final_selection"]
    assert len(set(selection["seeds"])) == 3
    assert set(selection["seeds"]).isdisjoint(reserved | {29})
    np.testing.assert_allclose(
        selection["mean_log_scores"], np.mean(selection["replicate_log_scores"], axis=0)
    )
    assert fit["optimal_value"] == selection["reference_scores"][selection["winner"]]


def test_selection_averages_whole_log_scores_and_excludes_any_truncation():
    selection_order = []

    def sample(rows, count, seed):
        if seed == 29:
            return -np.square(np.asarray(rows) - 0.6).sum(-1)
        if seed not in selection_order:
            selection_order.append(seed)
        # Candidate 0 wins a log-mean-exp but loses the mean log score;
        # candidate 2 must be rejected despite large scores on other seeds.
        return np.array(
            [[10.0, 2.0, PENALTY], [-10.0, 2.0, 100.0], [-10.0, 2.0, 100.0]]
        )[selection_order.index(seed)]

    fit, detail, _, _ = run_policy(sample, selection_candidates=3)
    selection = detail["final_selection"]
    assert selection["winner"] == 1
    assert selection["valid"] == [True, True, False]
    assert selection["mean_log_scores"] == pytest.approx([-10 / 3, 2.0, PENALTY])
    assert list(fit["fitted_params"].values()) == selection["candidates"][1]


@pytest.mark.parametrize("bad", [np.ones((1, 2)), np.array([np.nan])])
def test_policy_rejects_trial_factors_and_nonfinite_scores(bad):
    with pytest.raises(FloatingPointError, match="one finite complete-filter"):
        run_policy(lambda *args: bad)


@pytest.mark.parametrize(
    "setting",
    [
        dict(search_estimates=0),
        dict(search_estimates=33),
        dict(selection_repeats=0),
        dict(progress_tolerance=np.nan),
        dict(refine_evaluations=29),
    ],
)
def test_invalid_policy_configuration(setting):
    options = dict(search_estimates=4, reference_estimates=32, refine_evaluations=9)
    options.update(setting)
    with pytest.raises(ValueError):
        StagedConfig(**options).validate(30, 4)


def test_callback_retries_truncating_batches_without_accepting_partial_paths():
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    seen = []

    def score(rows):
        seen.append(rows)
        if len(rows) > 1 or rows[0][0] == 0.9:
            raise BatchedTruncationError("execution cap")
        return [2.0]

    invalid, work = [], {}
    scores = score_candidates(
        score,
        [[0.1], [0.9]],
        estimates=25,
        seed=77,
        invalid=invalid,
        work=work,
        truncation="penalize",
    )
    np.testing.assert_array_equal(scores, [2.0, PENALTY])
    assert seen == [[[0.1], [0.9]], [[0.1]], [[0.9]]]
    assert invalid == [
        {"parameters": [0.9], "estimates": 25, "seed": 77, "reason": "execution cap"}
    ]
    assert work == {
        "filter_batch_calls": 3,
        "candidate_filter_runs": 4,
        "candidate_particles": 100,
    }


def test_all_truncated_finalists_are_an_error():
    def sample(rows, count, seed):
        return (
            -np.square(np.asarray(rows) - 0.6).sum(-1)
            if seed == 29
            else np.full(len(rows), PENALTY)
        )

    with pytest.raises(RuntimeError, match="All final candidates truncated"):
        run_policy(sample)


def _small_pec(
    *, backend="triton_cpu", strategy="adaptive", callback=None, **overrides
):
    import pandas as pd
    import psyneulink as pnl

    memory = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.5),
        leak=0.3,
        competition=0.1,
        noise=pnl.NormalDist(standard_deviation=0.05),
        time_step_size=0.05,
        termination_measure=pnl.TimeScale.TRIAL,
        termination_threshold=3,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.Never(),
        name="adaptive memory",
    )
    response = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=1.1, bias=-0.1),
        leak=0.3,
        competition=0.2,
        self_excitation=0.0,
        noise=pnl.NormalDist(standard_deviation=0.05),
        time_step_size=0.1,
        termination_threshold=0.65,
        execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.RESULT, pnl.DECISION_TIME],
        name="adaptive response",
    )
    readout = pnl.ProcessingMechanism(input_shapes=1, name="adaptive readout")
    stimulus = pnl.ProcessingMechanism(input_shapes=2, name="adaptive stimulus")
    model = pnl.Composition(pathways=[stimulus, memory, response])
    model.add_node(readout)
    model.add_projection(
        sender=response.output_ports[pnl.DECISION_TIME], receiver=readout
    )
    model.scheduler.add_condition(memory, pnl.Always())
    model.scheduler.add_condition(response, pnl.Always())
    model.scheduler.add_condition(readout, pnl.WhenFinished(response))
    inputs = {stimulus: np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])}
    options = dict(
        method=optuna.samplers.CmaEsSampler,
        max_iterations=22,
        batched_backend=backend,
        batched_max_steps=100,
        batched_seed=29,
        batched_bins=20,
        batched_bin_range=[(0.0, 2.0)],
        batched_smoothing_sigma=0.5,
        batched_pseudocount=0.1,
        batched_strict_truncation=True,
        batched_parameter_batch_size=4,
        conditioned_likelihood=True,
        fit_strategy=strategy,
        fit_callback=callback,
        adaptive_options=dict(
            search_estimates=8,
            refine_evaluations=9,
            check_every=2,
            min_evaluations=5,
            patience=1,
            progress_tolerance=1e6,
            checkpoint_candidates=2,
            selection_candidates=2,
            selection_repeats=2,
            optimizer_seed=10,
            reserved_seeds=[8101],
        ),
    )
    if strategy == "fixed":
        options.pop("adaptive_options")
    options.update(overrides)
    pec = pnl.ParameterEstimationComposition(
        model=model,
        parameters={
            ("gain", memory): np.linspace(1.0, 2.0, 1001),
            ("intercept", readout): np.linspace(0.1, 0.2, 101),
        },
        outcome_variables=[readout.output_port],
        data=pd.DataFrame({"response_time": [0.5, 0.6, 0.7]}),
        likelihood_include_mask=np.array([False, True, True]),
        num_estimates=32,
        initial_seed=29,
        same_seed_for_all_parameter_combinations=True,
        optimization_function=pnl.PECOptimizationFunction(**options),
    )
    function = pec.controller.function
    initial = dict(zip(function.fit_param_names, [1.5, 0.15], strict=True))
    function.method = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(
            x0=initial, popsize=4, seed=10, lr_adapt=True
        ),
    )
    return pec, inputs


def _fake_plan(pec, monkeypatch, *, failure=False):
    from types import SimpleNamespace

    function = pec.controller.function
    calls = []

    def score(inputs, rows, **kwargs):
        calls.append((inputs, rows, kwargs))
        if failure:
            raise RuntimeError("unexpected scoring failure")
        return [
            -sum((float(value) - 0.4) ** 2 for value in row.values()) for row in rows
        ]

    plan = SimpleNamespace(
        conditioned_log_likelihood=score,
        ir=SimpleNamespace(
            graph=SimpleNamespace(
                inputs=[SimpleNamespace(node=next(iter(pec.model.nodes)).name)]
            )
        ),
    )
    monkeypatch.setattr(function, "_compile_batched_plan", lambda: plan)
    monkeypatch.setattr(function, "_batched_outcome_indices", lambda _: [0])
    return calls


@pytest.mark.parametrize("failure", [False, True])
def test_public_batch_budget_overrides_preserve_settings_even_on_error(
    monkeypatch, failure
):
    pec, inputs = _small_pec()
    function = pec.controller.function
    calls = _fake_plan(pec, monkeypatch, failure=failure)
    before = (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    )
    if failure:
        with pytest.raises(RuntimeError, match="unexpected scoring"):
            pec.log_likelihood_batch(
                [[1.5, 0.3]], inputs=inputs, num_estimates=8, seed=8101
            )
    else:
        actual = pec.log_likelihood_batch(
            [[1.5, 0.3]], inputs=inputs, num_estimates=8, seed=8101
        )
        assert actual.shape == (1,)
    assert before == (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    )
    _, _, options = calls[0]
    assert (options["num_estimates"], options["seed"], options["pseudocount"]) == (
        8,
        8101,
        0.025,
    )
    np.testing.assert_array_equal(options["include_mask"], [False, True, True])
    assert options["data"].shape == (3, 1)


@pytest.mark.parametrize("strategy", ["fixed", "adaptive"])
@pytest.mark.parametrize("failure", [False, True])
def test_public_adaptive_likelihood_is_independent_of_fit_policy(
    monkeypatch, strategy, failure
):
    import psyneulink as pnl

    pec, inputs = _small_pec(strategy=strategy)
    function = pec.controller.function
    calls = _fake_plan(pec, monkeypatch, failure=failure)
    before = (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    )
    options = dict(min_estimates=8, repeats=2, reference_index=0)
    if failure:
        with pytest.raises(RuntimeError, match="unexpected scoring"):
            pec.log_likelihood_batch(
                [[1.5, 0.3], [1.2, 0.4]],
                inputs=inputs,
                adaptive=True,
                adaptive_options=options,
            )
    else:
        result = pec.log_likelihood_batch(
            [[1.5, 0.3], [1.2, 0.4]],
            inputs=inputs,
            adaptive=True,
            adaptive_options=options,
        )
        assert isinstance(result, pnl.AdaptiveLikelihoodResult)
        assert result.converged and result.num_estimates == 8
        assert result.log_likelihood.shape == (2,)
        assert len(calls) == 4  # Two pilots and two independent final replicates.
        assert len({call[2]["seed"] for call in calls}) == 4
        assert all(call[2]["seed"] != 29 for call in calls)
    assert before == (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    )
    assert function.fit_diagnostics is None
    assert function.method.trials == []
    for _, _, kwargs in calls:
        assert kwargs["num_estimates"] == 8
        assert kwargs["pseudocount"] == 0.025
        np.testing.assert_array_equal(kwargs["include_mask"], [False, True, True])
        assert kwargs["data"].shape == (3, 1)


@pytest.mark.parametrize(
    "case,match",
    [
        ("options_without_adaptive", "adaptive_options requires"),
        ("not_boolean", "adaptive must"),
        ("wrong_columns", "one column"),
        ("generated_observations", "batched_observations"),
        ("window_execution", "strict_truncation"),
        ("invalid_options", "target_se"),
    ],
)
def test_invalid_public_adaptive_scoring_fails_before_compilation(
    monkeypatch, case, match
):
    pec, inputs = _small_pec(strategy="fixed")
    function = pec.controller.function
    monkeypatch.setattr(
        function, "_compile_batched_plan", lambda: pytest.fail("Unexpected compilation")
    )
    kwargs = dict(adaptive=True)
    rows = [[1.5, 0.3]]
    if case == "options_without_adaptive":
        kwargs = dict(adaptive_options={})
    elif case == "not_boolean":
        kwargs["adaptive"] = "adaptive"
    elif case == "wrong_columns":
        rows = [[1.5]]
    elif case == "generated_observations":
        function.batched_observations = object()
    elif case == "window_execution":
        function.batched_strict_truncation = False
    else:
        kwargs["adaptive_options"] = dict(target_se=0)
    from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
        OptimizationFunctionError,
    )

    with pytest.raises((ValueError, OptimizationFunctionError), match=match):
        pec.log_likelihood_batch(rows, inputs=inputs, **kwargs)


@pytest.mark.triton_gpu
@pytest.mark.batched
def test_adaptive_likelihood_replays_complete_masked_noisy_stateful_filters():
    pec, inputs = _small_pec(backend="triton", strategy="fixed")
    rows = [[1.5, 0.3], [1.2, 0.4]]
    options = dict(min_estimates=8, repeats=3, target_se=1e6, reference_index=0)
    result = pec.log_likelihood_batch(
        rows, inputs=inputs, adaptive=True, adaptive_options=options, seed=8101
    )
    replay = np.asarray(
        [
            pec.log_likelihood_batch(
                rows, inputs=inputs, num_estimates=result.num_estimates, seed=seed
            )
            for seed in result.seeds
        ]
    )
    np.testing.assert_array_equal(result.replicate_log_likelihoods, replay)
    np.testing.assert_array_equal(result.log_likelihood, replay.mean(axis=0))
    np.testing.assert_array_equal(
        result.standard_error, replay.std(axis=0, ddof=1) / np.sqrt(3)
    )
    differences = replay - replay[:, 0, None]
    np.testing.assert_array_equal(
        result.log_likelihood_difference, differences.mean(axis=0)
    )
    np.testing.assert_array_equal(
        result.difference_standard_error, differences.std(axis=0, ddof=1) / np.sqrt(3)
    )
    repeated = pec.log_likelihood_batch(
        rows, inputs=inputs, adaptive=True, adaptive_options=options, seed=8101
    )
    assert result.seeds == repeated.seeds
    np.testing.assert_array_equal(result.log_likelihood, repeated.log_likelihood)


@pytest.mark.parametrize("strategy", ["fixed", "adaptive"])
def test_public_run_uses_generic_policy_and_exposes_results(monkeypatch, strategy):
    records = []
    pec, inputs = _small_pec(
        strategy=strategy, callback=lambda *args: records.append(args)
    )
    function = pec.controller.function
    calls = _fake_plan(pec, monkeypatch)
    pec.run(inputs=inputs)
    assert set(pec.optimized_parameter_values) == set(function.fit_param_names)
    assert np.isfinite(pec.optimal_value)
    assert function.fit_study is function.method
    assert sum(len(rows) for rows, *_ in records) == function.num_evals
    assert all(
        call[2]["include_mask"].tolist() == [False, True, True] for call in calls
    )
    if strategy == "adaptive":
        detail = function.fit_diagnostics
        assert detail["policy"] == "staged"
        assert {call[2]["num_estimates"] for call in calls} == {8, 32}
        assert all(
            call[2]["pseudocount"] / call[2]["num_estimates"] == 0.1 / 32
            for call in calls
        )
        selection = detail["final_selection"]
        assert pec.optimal_value == selection["selected_reference_score"]
        assert (
            list(pec.optimized_parameter_values.values())
            == selection["candidates"][selection["winner"]]
        )
        assert set(selection["seeds"]).isdisjoint({29, 8101})
        assert len(function.refinement_study.trials) == 9
        assert {metadata["phase"] for *_, metadata in records} == {
            "staged_search",
            "refinement",
        }
    else:
        assert function.num_evals == 22 and function.refinement_study is None


@pytest.mark.parametrize(
    "options",
    [
        {"conditioned_likelihood": False},
        {"batched_backend": None},
        {"batched_parameter_batch_size": None},
        {"direction": "minimize"},
        {"distributed": True},
    ],
)
def test_adaptive_rejects_unsupported_likelihood_or_execution(options):
    import psyneulink as pnl

    kwargs = dict(
        method=optuna.samplers.CmaEsSampler,
        fit_strategy="adaptive",
        conditioned_likelihood=True,
        batched_backend="triton_cpu",
        batched_parameter_batch_size=2,
    )
    kwargs.update(options)
    with pytest.raises(ValueError, match="Adaptive fitting currently requires"):
        pnl.PECOptimizationFunction(**kwargs)


@pytest.mark.triton_gpu
@pytest.mark.batched
@pytest.mark.parametrize("precompile", [False, True])
def test_public_adaptive_fits_a_small_noisy_retained_state_model(precompile):
    pec, inputs = _small_pec(backend="triton")
    rows = [[1.5, 0.3], [1.2, 0.4]]
    before = (
        pec.log_likelihood_batch(rows, inputs=inputs, seed=8101) if precompile else None
    )
    pec.run(inputs=inputs)
    after = pec.log_likelihood_batch(rows, inputs=inputs, seed=8101)
    if before is None:
        before = pec.log_likelihood_batch(rows, inputs=inputs, seed=8101)
    np.testing.assert_array_equal(after, before)
    assert pec.controller.function.fit_diagnostics["policy"] == "staged"
    np.testing.assert_allclose(
        pec.log_likelihood_batch(
            [list(pec.optimized_parameter_values.values())], inputs=inputs
        ),
        [pec.optimal_value],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("count", "integers"),
        ("unknown", "unexpected keyword"),
        ("reserved", "must not be reserved"),
        ("used_study", "fresh study"),
        ("minimize", "maximizing study"),
        ("sampler", "CmaEsSampler"),
    ],
)
def test_invalid_adaptive_settings_fail_before_sampling(monkeypatch, case, match):
    from psyneulink.core.components.functions.nonstateful.optimizationfunctions import (
        OptimizationFunctionError,
    )

    pec, inputs = _small_pec()
    function = pec.controller.function
    calls = _fake_plan(pec, monkeypatch)
    if case == "count":
        function.adaptive_options["search_estimates"] = 2.5
    elif case == "unknown":
        function.adaptive_options["min_particles"] = 8
    elif case == "reserved":
        function.adaptive_options["reserved_seeds"] = [29]
    elif case == "used_study":
        function.method.ask()
    elif case == "minimize":
        function.method = optuna.create_study(
            direction="minimize", sampler=function.method.sampler
        )
    elif case == "sampler":
        function.method = optuna.samplers.RandomSampler(seed=3)
    with pytest.raises((ValueError, TypeError, OptimizationFunctionError), match=match):
        pec.run(inputs=inputs)
    assert calls == []


def test_score_failure_marks_pending_search_trials_failed(monkeypatch):
    pec, inputs = _small_pec()
    function = pec.controller.function
    _fake_plan(pec, monkeypatch)
    plan = function._compile_batched_plan()
    original = plan.conditioned_log_likelihood
    calls = 0

    def fail_during_search(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("search interrupted")
        return original(*args, **kwargs)

    plan.conditioned_log_likelihood = fail_during_search
    with pytest.raises(RuntimeError, match="search interrupted"):
        pec.run(inputs=inputs)
    assert [t.state for t in function.fit_study.trials] == [
        optuna.trial.TrialState.FAIL
    ]
    assert (
        pec.controller.num_estimates,
        function.batched_seed,
        function.batched_pseudocount,
    ) == (32, 29, 0.1)


@pytest.mark.parametrize("strategy", ["fixed", "adaptive"])
def test_callback_errors_propagate_without_stranded_trials(monkeypatch, strategy):
    def callback(*args):
        raise RuntimeError("reporting failed")

    pec, inputs = _small_pec(strategy=strategy, callback=callback)
    _fake_plan(pec, monkeypatch)
    with pytest.raises(RuntimeError, match="reporting failed"):
        pec.run(inputs=inputs)
    assert all(
        t.state != optuna.trial.TrialState.RUNNING
        for t in pec.controller.function.fit_study.trials
    )


def test_strict_truncation_is_not_silently_penalized():
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    def truncate(rows):
        raise BatchedTruncationError("cap reached")

    work, invalid = {}, []
    with pytest.raises(BatchedTruncationError, match="cap reached"):
        score_candidates(
            truncate,
            [[0.1], [0.2]],
            estimates=8,
            seed=3,
            invalid=invalid,
            work=work,
            truncation="raise",
        )
    assert invalid == [] and work["filter_batch_calls"] == 1


def test_fixed_penalized_fit_does_not_publish_an_all_truncated_winner(monkeypatch):
    from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError

    pec, inputs = _small_pec(strategy="fixed", fit_truncation="penalize")
    function = pec.controller.function
    _fake_plan(pec, monkeypatch)

    def truncate(*args, **kwargs):
        raise BatchedTruncationError("cap reached")

    function._compile_batched_plan().conditioned_log_likelihood = truncate
    with pytest.raises(RuntimeError, match="did not find a valid candidate"):
        pec.run(inputs=inputs)
    assert len(function.fit_diagnostics["invalid_proposals"]) == 22
