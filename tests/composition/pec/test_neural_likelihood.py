"""Tests for neural likelihood estimation."""
import dataclasses
import sys

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful import (
    neurallikelihoodfunctions as nlf,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.nle

RATE_BOUNDS = (-1.5, 1.5)
THRESHOLD_BOUNDS = (0.3, 1.5)
BOUNDS = {"rate": RATE_BOUNDS, "threshold": THRESHOLD_BOUNDS}
OUTCOMES = ("decision", "response_time")


def _toy_training_data(n):
    """A mixed (choice, response time) sample of ``n`` trials, each drawn at parameters of its own.

    Returns the outcomes encoded for training, the parameters, which outcomes are categorical and
    their categories, and the outcomes as drawn.
    """
    rng = np.random.default_rng(0)
    theta = np.column_stack(
        [rng.uniform(0.1, 0.9, n), rng.uniform(-0.5, 0.5, n)]
    )
    decision = rng.binomial(1, theta[:, 0]).astype(float)
    rt = np.exp(theta[:, 1] + 0.3 * rng.normal(size=n))
    raw = np.column_stack([decision, rt])
    categorical = nlf._infer_categorical(raw)
    categories = tuple(
        tuple(float(v) for v in np.unique(raw[:, j])) if c else ()
        for j, c in enumerate(categorical)
    )
    x = nlf._encode_outcomes(raw, categorical, categories, OUTCOMES)
    return x, torch.as_tensor(theta, dtype=torch.float32), categorical, categories, raw


@pytest.fixture(scope="module")
def toy_likelihood():
    """A small trained estimator, and the outcomes it was trained on.

    Too small for accuracy claims, enough for behaviour.  Its record is that of a model driven by an
    input of 1 on every trial, as the models here are.
    """
    x, cond, categorical, categories, raw = _toy_training_data(6000)
    estimator, val_nll = nlf._fit_estimator(
        x, cond, categorical, categories, True, n_params=2, epochs=3, batch_size=512,
        learning_rate=5e-4, validation_fraction=0.1, seed=0,
    )
    metadata = nlf.NeuralLikelihoodMetadata(
        fit_param_names=("rate", "threshold"),
        lower=(RATE_BOUNDS[0], THRESHOLD_BOUNDS[0]),
        upper=(RATE_BOUNDS[1], THRESHOLD_BOUNDS[1]),
        outcome_names=OUTCOMES, categorical=categorical, categories=categories,
        log_transform=True, trial_feature_columns=(), constant_inputs=(1.0,),
        val_nll=val_nll,
    )
    return nlf.NeuralLikelihood(
        estimator, metadata, (x[:256].clone(), cond[:256].clone())
    ), raw


def _with_metadata(likelihood, **changes):
    """``likelihood``'s trained network, with ``changes`` made to the record of what it was trained for."""
    return nlf.NeuralLikelihood(
        likelihood._estimator, dataclasses.replace(likelihood.metadata, **changes),
        likelihood._shape_probe,
    )


def test_the_estimator_returned_is_the_one_its_held_out_score_describes():
    # A setting whose held-out score is best a few epochs before the last one.
    x, cond, categorical, categories, _ = _toy_training_data(2000)
    seed = 2
    estimator, val_nll = nlf._fit_estimator(
        x, cond, categorical, categories, True, n_params=2, epochs=6, batch_size=128,
        learning_rate=0.02, validation_fraction=0.1, seed=seed,
    )
    # The same held-out rows the training chose.
    _, val_idx = nlf._held_out_draws(cond, 2, 0.1, torch.Generator().manual_seed(seed))
    with torch.no_grad():
        rescored = float(estimator.loss(x[val_idx], condition=cond[val_idx]).mean())
    assert rescored == pytest.approx(val_nll, rel=1e-6)


def test_held_out_rows_come_from_draws_not_trained_on():
    # 20 draws of 50 rows each; the second column varies within a draw, as a trial feature does.
    theta = np.repeat(np.arange(20.0), 50)
    cond = torch.as_tensor(np.column_stack([theta, np.arange(theta.size)]), dtype=torch.float32)
    train, held_out = nlf._held_out_draws(cond, 1, 0.1, torch.Generator().manual_seed(0))
    assert set(theta[held_out.numpy()]).isdisjoint(theta[train.numpy()])
    assert len(held_out) == 100



# ---------------------------------------------------------------- metadata


def test_unseen_category_is_rejected(toy_likelihood):
    likelihood, _ = toy_likelihood
    outcomes = np.column_stack([np.full(4, 7.0), np.ones(4)])
    with pytest.raises(nlf.NeuralLikelihoodError, match="never simulated during training"):
        likelihood.log_likelihood([0.5, 0.9], outcomes)


# ------------------------------------------------------------------ scoring


def test_outcomes_are_reordered_for_the_estimator():
    """sbi requires continuous columns first; PEC's order puts the categorical first."""
    categorical, categories = (True, False), ((0.0, 1.0), ())
    raw = np.array([[1.0, 0.5], [0.0, 0.8]])
    encoded = nlf._encode_outcomes(raw, categorical, categories, OUTCOMES)
    np.testing.assert_allclose(encoded.numpy(), [[0.5, 1.0], [0.8, 0.0]])


def test_log_likelihood_is_differentiable(toy_likelihood):
    likelihood, raw = toy_likelihood
    theta = torch.tensor([0.5, 0.9], dtype=torch.float32, requires_grad=True)
    likelihood.trial_log_prob(theta, raw[:64]).sum().backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert (theta.grad.abs() > 0).any()


def test_wrong_number_of_outcome_columns_is_rejected(toy_likelihood):
    likelihood, _ = toy_likelihood
    with pytest.raises(nlf.NeuralLikelihoodError, match="Expected outcomes with 2 columns"):
        likelihood.log_likelihood([0.5, 0.9], np.zeros((4, 3)))


def test_missing_trial_features_are_reported(toy_likelihood):
    likelihood, raw = toy_likelihood
    likelihood = _with_metadata(likelihood, trial_feature_columns=(0, 1))
    with pytest.raises(nlf.NeuralLikelihoodError, match="requires trial_features"):
        likelihood.log_likelihood([0.5, 0.9], raw[:8])


# -------------------------------------------------------------- persistence


def test_save_and_load_round_trip_scores_identically(tmp_path, toy_likelihood):
    likelihood, raw = toy_likelihood
    path = tmp_path / "toy.pt"
    likelihood.save(path)
    reloaded = nlf.NeuralLikelihood.load(path)
    assert reloaded.metadata == likelihood.metadata
    assert reloaded.log_likelihood([0.5, 0.9], raw[:128]) == likelihood.log_likelihood(
        [0.5, 0.9], raw[:128]
    )


def test_an_estimator_for_continuous_outcomes_alone_scores_after_reloading(tmp_path):
    """With no categorical outcome, the estimator is a plain flow rather than the mixed one."""
    # Three parameters and two outcomes, so that the two cannot be taken for one another.
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.5, 0.5, size=(2000, 3))
    raw = theta[:, :2] + theta[:, 2:] + 0.3 * rng.normal(size=(2000, 2))
    categorical, categories, names = (False, False), ((), ()), ("first", "second")
    x = nlf._encode_outcomes(raw, categorical, categories, names)
    cond = torch.as_tensor(theta, dtype=torch.float32)
    estimator, val_nll = nlf._fit_estimator(
        x, cond, categorical, categories, False, n_params=3, epochs=1, batch_size=512,
        learning_rate=5e-4, validation_fraction=0.1, seed=0,
    )
    metadata = nlf.NeuralLikelihoodMetadata(
        fit_param_names=("a", "b", "c"), lower=(-0.5,) * 3, upper=(0.5,) * 3,
        outcome_names=names, categorical=categorical, categories=categories, log_transform=False,
        trial_feature_columns=(), constant_inputs=(), val_nll=val_nll,
    )
    likelihood = nlf.NeuralLikelihood(estimator, metadata, (x[:256].clone(), cond[:256].clone()))
    likelihood.save(tmp_path / "continuous.pt")
    reloaded = nlf.NeuralLikelihood.load(tmp_path / "continuous.pt")

    score = likelihood.log_likelihood([0.1, -0.1, 0.2], raw[:64])
    assert np.isfinite(score)
    assert reloaded.log_likelihood([0.1, -0.1, 0.2], raw[:64]) == score


# ------------------------------------------------------------- trial features


def test_input_columns_are_the_same_however_the_inputs_are_listed():
    a = pnl.ProcessingMechanism(name="a")
    b = pnl.ProcessingMechanism(name="b", default_variable=[0, 0])
    model = pnl.Composition(nodes=[a, b])
    inputs = {a: np.arange(10.0), b: np.column_stack([np.ones(10), np.zeros(10)])}

    columns = nlf._input_columns(inputs, 10, model)
    assert columns.shape == (10, 3)
    np.testing.assert_allclose(columns[:, 0], np.arange(10.0))
    np.testing.assert_array_equal(nlf._input_columns({b: inputs[b], a: inputs[a]}, 10, model),
                                  columns)


# --------------------------------------------------------------- PEC wiring


def _ddm_pec(data, depends_on=None, **kwargs):
    """A drift-diffusion model fitted to ``data``, and inputs of 1 for each of its trials.

    ``depends_on`` maps the name of a parameter to the column of ``data`` it depends on; the other
    keyword arguments are passed to the ParameterEstimationComposition.
    """
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=0.3, noise=1.0, threshold=0.6,
            non_decision_time=0.15, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    pec = pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={(name, decision): np.linspace(*bounds, 100) for name, bounds in BOUNDS.items()},
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        depends_on={(name, decision): column for name, column in (depends_on or {}).items()},
        **kwargs,
    )
    return pec, {comp: np.ones((len(data), 1))}


@pytest.fixture
def ddm_data():
    frame = pd.DataFrame({"decision": [0.0, 1.0, 1.0, 0.0], "response_time": [0.4, 0.5, 0.6, 0.7]})
    frame["decision"] = frame["decision"].astype("category")
    return frame


@pytest.mark.composition
def test_neural_requires_an_artifact(ddm_data):
    with pytest.raises(pnl.ParameterEstimationCompositionError, match="requires likelihood_estimator_kwargs"):
        _ddm_pec(ddm_data, likelihood_estimator="neural")


@pytest.mark.composition
def test_unknown_estimator_kwarg_is_rejected(ddm_data):
    with pytest.raises(pnl.ParameterEstimationCompositionError, match="Unknown likelihood_estimator_kwargs"):
        _ddm_pec(ddm_data, likelihood_estimator="neural",
                 likelihood_estimator_kwargs={"artifact": "x.pt", "epochs": 3})


@pytest.mark.composition
def test_estimator_kwargs_rejected_for_kde(ddm_data):
    with pytest.raises(pnl.ParameterEstimationCompositionError, match="applies only to"):
        _ddm_pec(ddm_data, likelihood_estimator_kwargs={"artifact": "x.pt"})


@pytest.mark.composition
@pytest.mark.parametrize(
    "trained_for, expected",
    [
        ({"fit_param_names": ("threshold", "rate")}, "trained for parameters"),
        ({"fit_param_names": ("rate", "non_decision_time")}, "trained for parameters"),
        ({"lower": (-1.0, 0.3)}, "reaches outside"),
        ({"outcome_names": ("decision", "rt")}, "trained for outcome variables"),
        ({"categorical": (False, True)}, "trained with categorical outcomes"),
    ],
    ids=["reordered", "renamed", "wider-bounds", "outcome-names", "categorical-flags"],
)
def test_a_mismatched_artifact_is_rejected_before_fitting(
    ddm_data, toy_likelihood, trained_for, expected
):
    """An estimator trained for a model other than this one is refused when the PEC is built."""
    likelihood = _with_metadata(toy_likelihood[0], **trained_for)
    with pytest.raises(nlf.NeuralLikelihoodError, match=expected):
        _ddm_pec(ddm_data, likelihood_estimator="neural",
                 likelihood_estimator_kwargs={"artifact": likelihood})


def _ddm_training_pec(data):
    """A factory for training, at module scope so a Dask worker can unpickle it."""
    pec, inputs = _ddm_pec(
        data, num_estimates=5, initial_seed=0, same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, inputs


@pytest.fixture
def training_frame():
    frame = pd.DataFrame({"decision": [0.0, 1.0] * 5, "response_time": [0.5] * 10})
    frame["decision"] = frame["decision"].astype("category")
    return frame


@pytest.mark.composition
def test_training_data_is_generated_from_the_composition(tmp_path):
    columns = []

    def factory(data):
        columns.append(list(data.columns))
        return _ddm_training_pec(data)

    likelihood = nlf.train_neural_likelihood(
        BOUNDS, OUTCOMES, pec_factory=factory,
        n_parameter_samples=8, n_trials_per_sample=10, epochs=1,
    )
    # The factory is given the outcome columns.
    assert columns == [list(OUTCOMES)]
    metadata = likelihood.metadata
    assert metadata.fit_param_names == ("rate", "threshold")
    assert metadata.categorical == (True, False)
    # The model is driven by a constant input, so nothing distinguishes one trial from another.
    assert metadata.trial_feature_columns == ()
    assert metadata.constant_inputs == (1.0,)
    assert np.isfinite(metadata.val_nll)
    # What training records can be read back.
    likelihood.save(tmp_path / "trained.pt")
    assert nlf.NeuralLikelihood.load(tmp_path / "trained.pt").metadata == metadata


@pytest.mark.composition
@pytest.mark.dask
def test_training_data_generation_distributes():
    likelihood = nlf.train_neural_likelihood(
        BOUNDS, OUTCOMES, pec_factory=_ddm_training_pec,
        n_parameter_samples=8, n_trials_per_sample=10, epochs=1,
        distributed_options={"n_workers": 2},
    )
    assert np.isfinite(likelihood.metadata.val_nll)


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "exactly one of pec"),
        ({"pec": object(), "pec_factory": _ddm_training_pec}, "exactly one of pec"),
        ({"pec": object(), "inputs": {}, "distributed_options": {"n_workers": 2}},
         "requires pec_factory"),
        ({"pec": object()}, "pec requires inputs"),
        ({"pec": object(), "inputs": {}, "n_trials_per_sample": 10},
         "applies to pec_factory only"),
    ],
    ids=["neither", "both", "pec-distributed", "pec-without-inputs", "pec-trial-count"],
)
def test_model_source_is_validated(kwargs, expected):
    with pytest.raises(nlf.NeuralLikelihoodError, match=expected):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, **kwargs)


@pytest.mark.composition
def test_training_accepts_an_already_built_model(training_frame):
    """A model built here needs no factory: nothing has to cross a process boundary."""
    pec, inputs = _ddm_training_pec(training_frame)
    likelihood = nlf.train_neural_likelihood(
        BOUNDS, OUTCOMES, pec=pec, inputs=inputs, n_parameter_samples=8, epochs=1,
    )
    assert likelihood.metadata.fit_param_names == ("rate", "threshold")


@pytest.mark.composition
def test_training_leaves_the_model_it_was_given_intact(training_frame):
    """Simulating for training must not disturb a model the caller is still using."""
    pec, inputs = _ddm_training_pec(training_frame)

    before = pec.log_likelihood(0.3, 0.6, inputs=inputs)
    nlf._simulate(pec, inputs, np.array([[0.3, 0.6]]), ("rate", "threshold"))
    assert pec.log_likelihood(0.3, 0.6, inputs=inputs) == before


@pytest.mark.composition
def test_each_training_draw_gets_noise_of_its_own(training_frame):
    """Even from a model built to share noise across evaluations, as one for fitting may be."""
    pec, inputs = _ddm_training_pec(training_frame)
    shared_noise = pec.controller.parameters.same_seed_for_all_allocations
    shared_noise.set(True)
    pec.log_likelihood(0.3, 0.6, inputs=inputs)

    same_draw_twice = np.array([[0.3, 0.6], [0.3, 0.6]])
    _, x, _ = nlf._simulate(pec, inputs, same_draw_twice, ("rate", "threshold"))
    first, second = np.split(x, 2)
    assert not np.array_equal(first, second)
    assert all(shared_noise.values.values())


@pytest.mark.composition
def test_draws_are_simulated_alike_however_they_are_divided_among_workers():
    """Each worker builds its own model, whose seeds start where every other's do."""
    placeholder = pd.DataFrame(np.zeros((10, 2)), columns=list(OUTCOMES))
    thetas = np.array([[0.3, 0.6], [0.3, 0.6], [-0.5, 1.0]])
    names = ("rate", "threshold")

    _, whole, _ = nlf._simulate(*_ddm_training_pec(placeholder), thetas, names, seed=4)
    _, first, _ = nlf._simulate(*_ddm_training_pec(placeholder), thetas[:1], names, seed=4)
    _, rest, _ = nlf._simulate(*_ddm_training_pec(placeholder), thetas[1:], names, seed=4,
                               first_draw=1)
    np.testing.assert_array_equal(np.concatenate([first, rest]), whole)


def test_a_worker_builds_its_model_once_per_training(monkeypatch):
    from psyneulink.core.components.functions.nonstateful import fitfunctions
    import psyneulink.core.globals.threads as threads_module

    threads, builds = [], []
    monkeypatch.setattr(threads_module, "set_num_threads", threads.append)
    monkeypatch.setattr(fitfunctions, "_PEC_FALLBACK_CACHE", {})

    def factory(data):
        builds.append(len(builds))
        return _ddm_training_pec(data)

    data = pd.DataFrame(np.zeros((10, 2)), columns=list(OUTCOMES))
    for training in ("first", "first", "second"):
        nlf._simulate_chunk(factory, data, np.array([[0.3, 0.6]]), 0, ("rate", "threshold"), 0,
                            3, training)
    assert builds == [0, 1]
    assert threads == [3, 3]


def test_inputs_set_how_many_trials_each_draw_simulates(training_frame):
    """Trials come from the inputs, not from the data the model was built around."""
    pec, _ = _ddm_training_pec(training_frame)
    n_estimates = pec.controller.parameters.num_estimates.get()
    # More trials than the 10 in the data.
    _, x, _ = nlf._simulate(pec, {pec.nodes[0]: np.ones((30, 1))}, np.array([[0.3, 0.6]]),
                            ("rate", "threshold"))
    assert len(x) == 30 * n_estimates


def test_a_missing_sbi_is_reported_before_anything_is_simulated(monkeypatch):
    built = []

    def factory(data):
        built.append(True)
        return _ddm_training_pec(data)

    monkeypatch.setitem(sys.modules, "sbi", None)
    with pytest.raises(ImportError, match="requires the sbi package"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec_factory=factory,
                                    n_parameter_samples=8)
    assert not built


@pytest.mark.composition
def test_a_model_whose_outcomes_are_all_categorical_is_refused():
    with pytest.raises(nlf.NeuralLikelihoodError, match="at least one continuous"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec_factory=_ddm_training_pec,
                                    categorical=(True, True), n_parameter_samples=4,
                                    n_trials_per_sample=5, epochs=1)


@pytest.mark.composition
def test_an_estimator_that_did_not_train_is_refused(monkeypatch):
    # The held-out loss is not finite after any epoch, as when training diverges.
    monkeypatch.setattr(nlf, "_fit_estimator", lambda *args, **kwargs: (None, float("inf")))
    with pytest.raises(nlf.NeuralLikelihoodError, match="Training failed"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec_factory=_ddm_training_pec,
                                    n_parameter_samples=4, n_trials_per_sample=5)


@pytest.mark.composition
def test_training_rejects_a_model_that_orders_its_parameters_differently():
    """Draws are matched to parameters by position, so the two orders have to agree."""
    # The model declares rate first.
    with pytest.raises(nlf.NeuralLikelihoodError, match="matched by position"):
        nlf.train_neural_likelihood(
            {"threshold": THRESHOLD_BOUNDS, "rate": RATE_BOUNDS}, OUTCOMES,
            pec_factory=_ddm_training_pec, n_parameter_samples=4, n_trials_per_sample=5, epochs=1,
        )


@pytest.mark.composition
@pytest.mark.dask
def test_training_rejects_a_reordered_model_when_distributing():
    """The same check has to hold on a worker, which builds its own model."""
    with pytest.raises(Exception, match="matched by position"):
        nlf.train_neural_likelihood(
            {"threshold": THRESHOLD_BOUNDS, "rate": RATE_BOUNDS}, OUTCOMES,
            pec_factory=_ddm_training_pec, n_parameter_samples=4, n_trials_per_sample=5, epochs=1,
            distributed_options={"n_workers": 1},
        )


@pytest.mark.composition
def test_excluded_trials_do_not_reach_the_estimator(ddm_data, toy_likelihood):
    """A mask means the same for a trained estimator as it does for a simulated one."""
    likelihood, _ = toy_likelihood
    mask = np.array([True, False, True, False])
    pec, _ = _ddm_pec(ddm_data, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood},
                      likelihood_include_mask=mask)

    expected = likelihood.log_likelihood([0.3, 0.9], pec._data_numpy[mask])
    assert pec.log_likelihood(0.3, 0.9) == pytest.approx(expected, rel=1e-5)


@pytest.mark.composition
def test_a_fit_scores_with_the_estimator(ddm_data, toy_likelihood):
    likelihood, _ = toy_likelihood
    pec, inputs = _ddm_pec(ddm_data, likelihood_estimator="neural",
                           likelihood_estimator_kwargs={"artifact": likelihood},
                           optimization_function=pnl.PECOptimizationFunction(
                               method="differential_evolution", max_iterations=2))
    pec.run(inputs=inputs)

    rate, threshold = pec.optimized_parameter_values.values()
    assert RATE_BOUNDS[0] <= rate <= RATE_BOUNDS[1]
    assert THRESHOLD_BOUNDS[0] <= threshold <= THRESHOLD_BOUNDS[1]
    np.testing.assert_allclose(pec.optimal_value, pec.log_likelihood(rate, threshold))


@pytest.mark.composition
@pytest.mark.parametrize("on_optimization_function", [False, True],
                         ids=["on-composition", "on-optimization-function"])
def test_a_distributed_fit_is_refused_with_a_neural_likelihood(
    ddm_data, toy_likelihood, on_optimization_function
):
    """Workers score the models the factory builds, so the estimator would go unused."""
    likelihood, _ = toy_likelihood
    distributed = dict(distributed=True, distributed_options={"pec_factory": _ddm_training_pec})
    if on_optimization_function:
        settings = dict(optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", **distributed))
    else:
        settings = dict(optimization_function="differential_evolution", **distributed)
    with pytest.raises(pnl.ParameterEstimationCompositionError, match="cannot be combined"):
        _ddm_pec(ddm_data, likelihood_estimator="neural",
                 likelihood_estimator_kwargs={"artifact": likelihood}, **settings)


def _record_trial_features(likelihood, monkeypatch):
    """Make ``likelihood`` record the trial features it is asked to score with, and score zero."""
    seen = []

    def log_likelihood(theta, outcomes, trial_features=None):
        seen.append(trial_features)
        return 0.0

    monkeypatch.setattr(likelihood, "log_likelihood", log_likelihood)
    return seen


@pytest.mark.composition
def test_trial_features_follow_the_inputs_of_each_call(ddm_data, toy_likelihood, monkeypatch):
    """A later call with different inputs must not be scored against the first call's."""
    likelihood = _with_metadata(toy_likelihood[0], trial_feature_columns=(0,), constant_inputs=())
    seen = _record_trial_features(likelihood, monkeypatch)
    pec, _ = _ddm_pec(ddm_data, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood})
    node = pec.nodes[0]

    pec.log_likelihood(0.3, 0.9, inputs={node: np.arange(4.0).reshape(-1, 1)})
    pec.log_likelihood(0.3, 0.9, inputs={node: (10 + np.arange(4.0)).reshape(-1, 1)})

    np.testing.assert_allclose(seen[0].ravel(), [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(seen[1].ravel(), [10.0, 11.0, 12.0, 13.0])


@pytest.mark.composition
def test_trial_features_are_the_columns_training_used(ddm_data, toy_likelihood, monkeypatch):
    """Taken by position, even where the column training used does not vary in these data."""
    likelihood = _with_metadata(toy_likelihood[0], trial_feature_columns=(0,))
    seen = _record_trial_features(likelihood, monkeypatch)
    pec, _ = _ddm_pec(ddm_data, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood})

    one_condition = np.column_stack([np.full(4, 3.0), np.ones(4)])
    pec.log_likelihood(0.3, 0.9, inputs={pec.nodes[0]: one_condition})
    np.testing.assert_allclose(seen[0].ravel(), 3.0)


@pytest.mark.composition
def test_inputs_laid_out_differently_from_training_are_refused(ddm_data, toy_likelihood):
    likelihood = _with_metadata(toy_likelihood[0], trial_feature_columns=(0,))
    pec, _ = _ddm_pec(ddm_data, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood})
    with pytest.raises(nlf.NeuralLikelihoodError, match="laid out as they were"):
        pec.log_likelihood(0.3, 0.9, inputs={pec.nodes[0]: np.arange(4.0).reshape(-1, 1)})


@pytest.mark.composition
def test_an_input_held_constant_in_training_has_to_keep_its_value(ddm_data, toy_likelihood):
    """Training saw only the one value, so says nothing of trials with another."""
    likelihood, _ = toy_likelihood
    pec, _ = _ddm_pec(ddm_data, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood})
    with pytest.raises(nlf.NeuralLikelihoodError, match="held at"):
        pec.log_likelihood(0.3, 0.9, inputs={pec.nodes[0]: np.full((4, 1), 2.0)})


@pytest.mark.composition
@pytest.mark.parametrize("response_time", [0.0, np.nan], ids=["zero", "missing"])
def test_outcomes_the_estimator_cannot_score_are_refused(ddm_data, toy_likelihood, response_time):
    """The estimator models the logarithm of response times, so they have to be positive."""
    likelihood, _ = toy_likelihood
    ddm_data.loc[1, "response_time"] = response_time
    with pytest.raises(nlf.NeuralLikelihoodError, match="cannot score"):
        _ddm_pec(ddm_data, likelihood_estimator="neural",
                 likelihood_estimator_kwargs={"artifact": likelihood})


@pytest.mark.composition
def test_a_parameter_that_depends_on_a_condition_is_scored_at_each_trials_value(
    ddm_data, toy_likelihood
):
    likelihood, _ = toy_likelihood
    data = ddm_data.assign(condition=pd.Categorical(["easy", "hard", "easy", "hard"]))
    pec, _ = _ddm_pec(data, depends_on={"rate": "condition"}, likelihood_estimator="neural",
                      likelihood_estimator_kwargs={"artifact": likelihood})
    outcomes, easy = pec._data_numpy, np.array([True, False, True, False])

    expected = (likelihood.log_likelihood([0.2, 0.9], outcomes[easy])
                + likelihood.log_likelihood([-0.4, 0.9], outcomes[~easy]))
    assert pec.log_likelihood(0.2, -0.4, 0.9) == pytest.approx(expected, rel=1e-5)


@pytest.mark.composition
def test_training_refuses_a_model_scored_by_an_estimator(ddm_data, toy_likelihood):
    likelihood, _ = toy_likelihood
    pec, inputs = _ddm_pec(ddm_data, likelihood_estimator="neural",
                           likelihood_estimator_kwargs={"artifact": likelihood})
    with pytest.raises(nlf.NeuralLikelihoodError, match="scored by simulating it"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec=pec, inputs=inputs,
                                    n_parameter_samples=4, epochs=1)


@pytest.mark.composition
def test_training_refuses_a_model_whose_parameters_depend_on_a_condition(ddm_data):
    """Training covers each parameter's range; which condition a value is fitted for comes later."""
    data = ddm_data.assign(condition=pd.Categorical(["easy", "hard", "easy", "hard"]))
    pec, inputs = _ddm_pec(data, depends_on={"rate": "condition"})
    with pytest.raises(nlf.NeuralLikelihoodError, match="without depends_on"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec=pec, inputs=inputs,
                                    n_parameter_samples=4, epochs=1)


@pytest.fixture(scope="module")
def trained_artifact(toy_likelihood, tmp_path_factory):
    """A trained estimator on disk, for factories that have to load it on a worker."""
    path = tmp_path_factory.mktemp("nle") / "toy.pt"
    toy_likelihood[0].save(path)
    return str(path)


def _neural_participant_pec(artifact, data, subject_index=None):
    """A participant model scored by a trained estimator, with no common random numbers."""
    return _ddm_pec(data, likelihood_estimator="neural",
                    likelihood_estimator_kwargs={"artifact": artifact})


def _group_frame(n_participants=2, n_trials=6):
    rng = np.random.default_rng(0)
    frames = []
    for s in range(n_participants):
        frame = pd.DataFrame({
            "decision": rng.integers(0, 2, n_trials).astype(float),
            "response_time": rng.uniform(0.3, 1.2, n_trials),
            "subject": f"S{s}",
        })
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    data["decision"] = data["decision"].astype("category")
    return data


def _fit_group(artifact, distributed=False, **distributed_options):
    import functools

    options = {"pec_factory": functools.partial(_neural_participant_pec, artifact)}
    options.update(distributed_options)
    pec = pnl.ParameterEstimationComposition(
        data=_group_frame(),
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject", "max_iterations": 1,
            "estep_options": {"xatol": 1e-1, "fatol": 1e-1, "maxiter": 12},
        },
        distributed=distributed,
        distributed_options=options,
    )
    return pec.run()


@pytest.mark.composition
def test_a_hierarchical_fit_scores_participants_with_their_estimator(trained_artifact, ddm_data):
    """Participants scored by simulation would be refused here, having no common random numbers."""
    participant, _ = _neural_participant_pec(trained_artifact, ddm_data)
    assert participant.scores_by_simulation is False
    assert participant.controller.parameters.same_seed_for_all_allocations.get() in (None, False)

    results = _fit_group(trained_artifact)
    assert results.beta.shape == (1, 2)
    assert np.isfinite(results.objective)


@pytest.mark.composition
@pytest.mark.dask
def test_a_distributed_hierarchical_fit_scores_the_same_way(trained_artifact):
    """Each worker builds and loads its own, and has to reach the same answer."""
    here = _fit_group(trained_artifact)
    there = _fit_group(trained_artifact, distributed=True, n_workers=2)

    np.testing.assert_allclose(there.beta, here.beta, rtol=1e-10)
    np.testing.assert_allclose(there.sigma, here.sigma, rtol=1e-10)
