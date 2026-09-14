"""Tests for neural likelihood estimation."""
import sys

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful import (
    neurallikelihoodfunctions as nlf,
)

# Optional dependencies: skipped without them, rather than failing to collect.
torch = pytest.importorskip("torch")
pytest.importorskip("sbi")

RATE_BOUNDS = (-1.5, 1.5)
THRESHOLD_BOUNDS = (0.3, 1.5)
OUTCOMES = ("decision", "response_time")


def _toy_arrays(n=6000, seed=0):
    """A mixed (choice, RT) sample whose density is known in closed form."""
    rng = np.random.default_rng(seed)
    theta = np.column_stack(
        [rng.uniform(0.1, 0.9, n), rng.uniform(-0.5, 0.5, n)]
    )
    decision = rng.binomial(1, theta[:, 0]).astype(float)
    rt = np.exp(theta[:, 1] + 0.3 * rng.normal(size=n))
    return theta, np.column_stack([decision, rt])


def _toy_likelihood(epochs=3, seed=0):
    """A small trained estimator; too small for accuracy claims, enough for behaviour."""
    theta, raw = _toy_arrays(seed=seed)
    categorical = nlf._infer_categorical(raw)
    categories = tuple(
        tuple(float(v) for v in np.unique(raw[:, j])) if c else ()
        for j, c in enumerate(categorical)
    )
    x = nlf._encode_outcomes(raw, categorical, categories, OUTCOMES)
    cond = torch.as_tensor(theta, dtype=torch.float32)
    estimator, val_nll = nlf._fit_estimator(
        x, cond, categorical, categories, True, epochs=epochs, batch_size=512,
        learning_rate=5e-4, validation_fraction=0.1, seed=seed,
    )
    provenance = nlf.NeuralLikelihoodProvenance(
        fit_param_names=("rate", "threshold"),
        lower=(RATE_BOUNDS[0], THRESHOLD_BOUNDS[0]),
        upper=(RATE_BOUNDS[1], THRESHOLD_BOUNDS[1]),
        outcome_names=OUTCOMES, categorical=categorical, categories=categories,
        log_transform=True, n_input_columns=0, trial_feature_columns=(),
        n_parameter_samples=len(theta),
        n_trials_per_sample=1, epochs=epochs, val_nll=val_nll, seed=seed,
        psyneulink_version="test", sbi_version="test",
    )
    return nlf.NeuralLikelihood(
        estimator, provenance, (x[:256].clone(), cond[:256].clone())
    ), raw


def test_the_estimator_returned_is_the_one_its_held_out_score_describes():
    # A setting whose held-out score is best a few epochs before the last one.
    theta, raw = _toy_arrays(n=2000)
    categorical = nlf._infer_categorical(raw)
    categories = tuple(
        tuple(float(v) for v in np.unique(raw[:, j])) if c else ()
        for j, c in enumerate(categorical)
    )
    x = nlf._encode_outcomes(raw, categorical, categories, OUTCOMES)
    cond = torch.as_tensor(theta, dtype=torch.float32)
    seed = 2
    estimator, val_nll = nlf._fit_estimator(
        x, cond, categorical, categories, True, epochs=6, batch_size=128,
        learning_rate=0.02, validation_fraction=0.1, seed=seed,
    )
    # The same held-out rows the training chose.
    order = torch.randperm(x.shape[0], generator=torch.Generator().manual_seed(seed))
    val_idx = order[:max(1, int(0.1 * x.shape[0]))]
    with torch.no_grad():
        rescored = float(estimator.loss(x[val_idx], condition=cond[val_idx]).mean())
    assert rescored == pytest.approx(val_nll, rel=1e-6)


# ---------------------------------------------------------------- provenance


@pytest.mark.parametrize(
    "names, lower, upper, outcomes, categorical, expected",
    [
        (("threshold", "rate"), (-1.5, 0.3), (1.5, 1.5), OUTCOMES, (True, False),
         "trained for parameters"),
        (("rate", "non_decision_time"), (-1.5, 0.3), (1.5, 1.5), OUTCOMES, (True, False),
         "trained for parameters"),
        (("rate", "threshold"), (-2.0, 0.3), (1.5, 1.5), OUTCOMES, (True, False),
         "reaches outside"),
        (("rate", "threshold"), (-1.5, 0.3), (1.5, 1.5), ("decision", "rt"), (True, False),
         "trained for outcome variables"),
        (("rate", "threshold"), (-1.5, 0.3), (1.5, 1.5), OUTCOMES, (False, True),
         "trained with categorical outcomes"),
    ],
    ids=["reordered", "renamed", "wider-bounds", "outcome-names", "categorical-flags"],
)
def test_provenance_rejects_a_mismatched_model(
    names, lower, upper, outcomes, categorical, expected
):
    likelihood, _ = _toy_likelihood(epochs=1)
    with pytest.raises(nlf.NeuralLikelihoodError, match=expected):
        likelihood.provenance.check_matches(names, lower, upper, outcomes, categorical)


def test_provenance_accepts_the_model_it_was_trained_for():
    likelihood, _ = _toy_likelihood(epochs=1)
    likelihood.provenance.check_matches(
        ("rate", "threshold"), (-1.5, 0.3), (1.5, 1.5), OUTCOMES, (True, False)
    )


def test_unseen_category_is_rejected():
    likelihood, _ = _toy_likelihood(epochs=1)
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


def test_log_likelihood_is_differentiable():
    likelihood, raw = _toy_likelihood()
    theta = torch.tensor([0.5, 0.9], dtype=torch.float32, requires_grad=True)
    likelihood.trial_log_prob(theta, raw[:64]).sum().backward()
    assert theta.grad is not None
    assert torch.isfinite(theta.grad).all()
    assert (theta.grad.abs() > 0).any()


def test_wrong_number_of_outcome_columns_is_rejected():
    likelihood, _ = _toy_likelihood(epochs=1)
    with pytest.raises(nlf.NeuralLikelihoodError, match="Expected outcomes with 2 columns"):
        likelihood.log_likelihood([0.5, 0.9], np.zeros((4, 3)))


def test_missing_trial_features_are_reported(tmp_path):
    likelihood, raw = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "trial_feature_columns", (0, 1))
    with pytest.raises(nlf.NeuralLikelihoodError, match="requires trial_features"):
        likelihood.log_likelihood([0.5, 0.9], raw[:8])


# -------------------------------------------------------------- persistence


def test_save_and_load_round_trip_scores_identically(tmp_path):
    likelihood, raw = _toy_likelihood()
    path = tmp_path / "toy.pt"
    likelihood.save(path)
    reloaded = nlf.NeuralLikelihood.load(path)
    assert reloaded.provenance == likelihood.provenance
    assert reloaded.log_likelihood([0.5, 0.9], raw[:128]) == likelihood.log_likelihood(
        [0.5, 0.9], raw[:128]
    )


# ------------------------------------------------------------------- gates


def test_gates_reject_an_estimator_that_did_not_train():
    likelihood, raw = _toy_likelihood(epochs=1)
    theta = torch.zeros((8, 2))
    x = torch.zeros((8, 2))
    with pytest.raises(nlf.NeuralLikelihoodError, match="did not pass its validation gates"):
        nlf._check_gates(likelihood, x, theta, float("nan"), strict=True)


def test_gates_warn_rather_than_raise_when_not_strict():
    likelihood, _ = _toy_likelihood(epochs=1)
    with pytest.warns(nlf.NeuralLikelihoodWarning, match="did not pass its validation gates"):
        nlf._check_gates(likelihood, torch.zeros((8, 2)), torch.zeros((8, 2)),
                         float("nan"), strict=False)


# ------------------------------------------------------------- trial features


def test_input_columns_line_up_every_input_trial_by_trial():
    inputs = {"a": np.arange(10.0), "b": np.column_stack([np.ones(10), np.zeros(10)])}
    columns = nlf._input_columns(inputs, 10)
    assert columns.shape == (10, 3)
    np.testing.assert_allclose(columns[:, 0], np.arange(10.0))


# --------------------------------------------------------------- PEC wiring


def _ddm_pec(data, **kwargs):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=0.3, noise=1.0, threshold=0.6,
            non_decision_time=0.15, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    comp = pnl.Composition(pathways=decision)
    return pnl.ParameterEstimationComposition(
        nodes=[comp],
        parameters={
            ("rate", decision): np.linspace(*RATE_BOUNDS, 100),
            ("threshold", decision): np.linspace(*THRESHOLD_BOUNDS, 100),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        **kwargs,
    )


@pytest.fixture
def ddm_data():
    frame = pd.DataFrame({"decision": [0.0, 1.0, 1.0, 0.0], "response_time": [0.4, 0.5, 0.6, 0.7]})
    frame["decision"] = frame["decision"].astype("category")
    return frame


@pytest.mark.composition
def test_unknown_likelihood_estimator_is_rejected(ddm_data):
    with pytest.raises(Exception, match="kde"):
        _ddm_pec(ddm_data, likelihood_estimator="histogram")


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
def test_a_mismatched_artifact_is_rejected_before_fitting(ddm_data):
    """The check happens against the PEC, not only at training time."""
    likelihood, _ = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "fit_param_names", ("rate", "non_decision_time"))
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood})
    with pytest.raises(nlf.NeuralLikelihoodError, match="trained for parameters"):
        pec._setup_neural_likelihood()


def _ddm_training_pec(data):
    """A factory for training, at module scope so a Dask worker can unpickle it."""
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
        parameters={
            ("rate", decision): np.linspace(*RATE_BOUNDS, 100),
            ("threshold", decision): np.linspace(*THRESHOLD_BOUNDS, 100),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        num_estimates=5,
        initial_seed=0,
        same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: np.ones((len(data), 1))}


@pytest.mark.composition
def test_training_data_is_generated_from_the_composition():
    likelihood = nlf.train_neural_likelihood(
        {"rate": RATE_BOUNDS, "threshold": THRESHOLD_BOUNDS},
        OUTCOMES,
        pec_factory=_ddm_training_pec,
        n_parameter_samples=8, n_trials_per_sample=10, epochs=1,
    )
    provenance = likelihood.provenance
    assert provenance.fit_param_names == ("rate", "threshold")
    assert provenance.categorical == (True, False)
    # The model is driven by a constant input, so nothing distinguishes one trial from another.
    assert provenance.n_input_columns == 1
    assert provenance.trial_feature_columns == ()
    assert np.isfinite(provenance.val_nll)


@pytest.mark.composition
def test_training_data_generation_distributes():
    pytest.importorskip("dask.distributed")
    likelihood = nlf.train_neural_likelihood(
        {"rate": RATE_BOUNDS, "threshold": THRESHOLD_BOUNDS},
        OUTCOMES,
        pec_factory=_ddm_training_pec,
        n_parameter_samples=8, n_trials_per_sample=10, epochs=1,
        distributed_options={"n_workers": 2},
    )
    assert np.isfinite(likelihood.provenance.val_nll)


BOUNDS = {"rate": RATE_BOUNDS, "threshold": THRESHOLD_BOUNDS}


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "exactly one of pec"),
        ({"pec": object(), "pec_factory": _ddm_training_pec}, "exactly one of pec"),
        ({"pec": object(), "inputs": {}, "distributed_options": {"n_workers": 2}},
         "requires pec_factory"),
        ({"pec": object(), "inputs": {}, "n_trials_per_sample": 10},
         "applies to pec_factory only"),
    ],
    ids=["neither", "both", "pec-distributed", "pec-trial-count"],
)
def test_model_source_is_validated(kwargs, expected):
    with pytest.raises(nlf.NeuralLikelihoodError, match=expected):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, **kwargs)


@pytest.mark.composition
def test_training_accepts_an_already_built_model():
    """A model built here needs no factory: nothing has to cross a process boundary."""
    pec, inputs = _ddm_training_pec(
        pd.DataFrame({"decision": [0.0] * 10, "response_time": [0.5] * 10})
    )
    likelihood = nlf.train_neural_likelihood(
        BOUNDS, OUTCOMES, pec=pec, inputs=inputs, n_parameter_samples=8, epochs=1,
    )
    assert likelihood.provenance.fit_param_names == ("rate", "threshold")
    # the trial count comes from the model rather than from an argument
    assert likelihood.provenance.n_trials_per_sample == 10


@pytest.mark.composition
def test_training_leaves_the_model_it_was_given_intact():
    """Simulating for training must not disturb a model the caller is still using."""
    frame = pd.DataFrame({"decision": [0.0, 1.0] * 5, "response_time": [0.5] * 10})
    frame["decision"] = frame["decision"].astype("category")
    pec, inputs = _ddm_training_pec(frame)

    before = pec.log_likelihood(0.3, 0.6, inputs=inputs)
    nlf._simulate(pec, inputs, np.array([[0.3, 0.6]]), ("rate", "threshold"), 2)
    assert pec.log_likelihood(0.3, 0.6, inputs=inputs) == before


@pytest.mark.composition
def test_each_training_draw_gets_noise_of_its_own():
    """Even from a model built to share noise across evaluations, as one for fitting may be."""
    frame = pd.DataFrame({"decision": [0.0, 1.0] * 5, "response_time": [0.5] * 10})
    frame["decision"] = frame["decision"].astype("category")
    pec, inputs = _ddm_training_pec(frame)
    shared_noise = pec.controller.parameters.same_seed_for_all_allocations
    shared_noise.set(True)
    pec.log_likelihood(0.3, 0.6, inputs=inputs)

    same_draw_twice = np.array([[0.3, 0.6], [0.3, 0.6]])
    _, x, n_trials, _ = nlf._simulate(pec, inputs, same_draw_twice, ("rate", "threshold"), 2)
    first, second = np.split(x, 2)
    assert not np.array_equal(first, second)
    assert all(shared_noise.values.values())


def test_inputs_set_how_many_trials_each_draw_simulates():
    """Trials come from the inputs, not from the data the model was built around."""
    frame = pd.DataFrame({"decision": [0.0, 1.0] * 5, "response_time": [0.5] * 10})
    frame["decision"] = frame["decision"].astype("category")
    pec, _ = _ddm_training_pec(frame)
    node = pec.nodes[0]

    _, _, ten, _ = nlf._simulate(pec, {node: np.ones((10, 1))}, np.array([[0.3, 0.6]]),
                                 ("rate", "threshold"), 2)
    _, _, thirty, _ = nlf._simulate(pec, {node: np.ones((30, 1))}, np.array([[0.3, 0.6]]),
                                    ("rate", "threshold"), 2)
    assert (ten, thirty) == (10, 30)


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
def test_a_model_without_inputs_is_rejected(ddm_data):
    with pytest.raises(nlf.NeuralLikelihoodError, match="pec requires inputs"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec=object(), n_parameter_samples=8)


def _reversed_ddm_pec(data):
    """Declares the same parameters as _ddm_training_pec, in the opposite order."""
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
        parameters={
            ("threshold", decision): np.linspace(*THRESHOLD_BOUNDS, 100),
            ("rate", decision): np.linspace(*RATE_BOUNDS, 100),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        num_estimates=2, initial_seed=0, same_seed_for_all_parameter_combinations=True,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")
    return pec, {comp: np.ones((len(data), 1))}


@pytest.mark.composition
def test_training_rejects_a_model_that_orders_its_parameters_differently():
    """Draws are matched to parameters by position, so the two orders have to agree."""
    with pytest.raises(nlf.NeuralLikelihoodError, match="matched by position"):
        nlf.train_neural_likelihood(
            BOUNDS, OUTCOMES, pec_factory=_reversed_ddm_pec,
            n_parameter_samples=4, n_trials_per_sample=5, epochs=1,
        )


@pytest.mark.composition
def test_training_rejects_a_reordered_model_when_distributing():
    """The same check has to hold on a worker, which builds its own model."""
    pytest.importorskip("dask.distributed")
    with pytest.raises(Exception, match="matched by position"):
        nlf.train_neural_likelihood(
            BOUNDS, OUTCOMES, pec_factory=_reversed_ddm_pec,
            n_parameter_samples=4, n_trials_per_sample=5, epochs=1,
            distributed_options={"n_workers": 1},
        )


@pytest.mark.composition
def test_excluded_trials_do_not_reach_the_estimator(ddm_data):
    """A mask means the same for a trained estimator as it does for a simulated one."""
    likelihood, _ = _toy_likelihood(epochs=1)
    mask = np.array([True, False, True, False])
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood},
                   likelihood_include_mask=mask)
    pec._setup_neural_likelihood()
    scored = pec.controller.function._neural_outcomes

    assert len(scored) == 2
    np.testing.assert_allclose(scored, pec._data_numpy[mask])


@pytest.mark.composition
def test_a_fit_scores_with_the_estimator(ddm_data):
    likelihood, _ = _toy_likelihood(epochs=1)
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood},
                   optimization_function=pnl.PECOptimizationFunction(
                       method="differential_evolution", max_iterations=2))
    pec.run(inputs={pec.nodes[0]: np.ones((len(ddm_data), 1))})

    rate, threshold = pec.optimized_parameter_values.values()
    assert RATE_BOUNDS[0] <= rate <= RATE_BOUNDS[1]
    assert THRESHOLD_BOUNDS[0] <= threshold <= THRESHOLD_BOUNDS[1]
    np.testing.assert_allclose(pec.optimal_value, pec.log_likelihood(rate, threshold))


@pytest.mark.composition
def test_a_distributed_fit_is_refused_with_a_neural_likelihood(ddm_data):
    """Workers score the models the factory builds, so the estimator would go unused."""
    likelihood, _ = _toy_likelihood(epochs=1)
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood},
                   optimization_function="differential_evolution",
                   distributed=True, distributed_options={"pec_factory": _ddm_training_pec})
    with pytest.raises(Exception, match="cannot be combined"):
        pec.run(inputs={pec.nodes[0]: np.ones((len(ddm_data), 1))})


@pytest.mark.composition
def test_trial_features_follow_the_inputs_of_each_call(ddm_data):
    """A later call with different inputs must not be scored against the first call's."""
    likelihood, _ = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "n_input_columns", 1)
    object.__setattr__(likelihood.provenance, "trial_feature_columns", (0,))
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood})
    node = pec.nodes[0]

    pec._setup_neural_likelihood({node: np.arange(4.0).reshape(-1, 1)})
    first = pec.controller.function._neural_trial_features.copy()
    pec._setup_neural_likelihood({node: (10 + np.arange(4.0)).reshape(-1, 1)})
    second = pec.controller.function._neural_trial_features

    assert not np.allclose(first, second)
    np.testing.assert_allclose(second.ravel(), [10.0, 11.0, 12.0, 13.0])


@pytest.mark.composition
def test_trial_features_are_the_columns_training_used(ddm_data):
    """Taken by position, even where the column training used does not vary in these data."""
    likelihood, _ = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "n_input_columns", 2)
    object.__setattr__(likelihood.provenance, "trial_feature_columns", (0,))
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood})

    one_condition = np.column_stack([np.full(4, 3.0), np.arange(4.0)])
    pec._setup_neural_likelihood({pec.nodes[0]: one_condition})
    np.testing.assert_allclose(pec.controller.function._neural_trial_features.ravel(), 3.0)


@pytest.mark.composition
def test_inputs_laid_out_differently_from_training_are_refused(ddm_data):
    likelihood, _ = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "n_input_columns", 2)
    object.__setattr__(likelihood.provenance, "trial_feature_columns", (0,))
    pec = _ddm_pec(ddm_data, likelihood_estimator="neural",
                   likelihood_estimator_kwargs={"artifact": likelihood})
    with pytest.raises(pnl.ParameterEstimationCompositionError, match="laid out as they were"):
        pec._setup_neural_likelihood({pec.nodes[0]: np.arange(4.0).reshape(-1, 1)})


@pytest.fixture(scope="module")
def trained_artifact(tmp_path_factory):
    """A trained estimator on disk, for factories that have to load it on a worker."""
    likelihood, _ = _toy_likelihood(epochs=3)
    path = tmp_path_factory.mktemp("nle") / "toy.pt"
    likelihood.save(path)
    return str(path)


def _neural_participant_pec(artifact, data, subject_index=None):
    """A participant model scored by a trained estimator, with no common random numbers."""
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
        parameters={
            ("rate", decision): np.linspace(*RATE_BOUNDS, 100),
            ("threshold", decision): np.linspace(*THRESHOLD_BOUNDS, 100),
        },
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        likelihood_estimator="neural",
        likelihood_estimator_kwargs={"artifact": artifact},
    )
    return pec, {comp: np.ones((len(data), 1))}


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
def test_a_hierarchical_fit_scores_participants_with_their_estimator(trained_artifact):
    """Participants scored by simulation would be refused here, having no common random numbers."""
    results = _fit_group(trained_artifact)
    assert results.beta.shape == (1, 2)
    assert np.isfinite(results.objective)


@pytest.mark.composition
def test_a_participant_scored_by_an_estimator_needs_no_common_random_numbers(trained_artifact):
    """Common random numbers are required of participants scored by simulation only."""
    from psyneulink.core.compositions.hierarchical.subjectlikelihood import (
        check_scoring_is_deterministic,
    )

    # One participant's trials, without the column identifying them, as the factory receives them.
    trials = _group_frame(n_participants=1).drop(columns=["subject"])
    pec, _ = _neural_participant_pec(trained_artifact, trials)
    assert pec.scores_by_simulation is False
    assert pec.controller.parameters.same_seed_for_all_allocations.get() in (None, False)
    check_scoring_is_deterministic(pec, "a participant scored by an estimator")


@pytest.mark.composition
def test_a_distributed_hierarchical_fit_scores_the_same_way(trained_artifact):
    """Each worker builds and loads its own, and has to reach the same answer."""
    pytest.importorskip("dask.distributed")
    here = _fit_group(trained_artifact)
    there = _fit_group(trained_artifact, distributed=True, n_workers=2)

    np.testing.assert_allclose(there.beta, here.beta, rtol=1e-10)
    np.testing.assert_allclose(there.sigma, here.sigma, rtol=1e-10)


# ===========================================================================
# Sampling the hierarchical posterior
#
# What the estimator's gradient is for: fitting that reports the posterior
# rather than a Gaussian placed at its peak.  The budgets here are far too
# small for the draws to mean anything; what is checked is that the pieces
# reach each other and that the fit says so when it has not converged.
# ===========================================================================
def _sample_group(artifact, **hierarchical):
    import functools

    options = {
        "subject_id": "subject",
        "sampler": "nuts",
        # max_tree_depth is capped well below the default: a barely-warmed sampler on an
        # awkward posterior doubles all the way to the cap on every draw, and one draw would
        # otherwise cost a thousand network calls. It costs efficiency, which these do not
        # measure, and nothing else.
        "sampler_options": {
            "draws": 25, "warmup": 25, "chains": 2, "seed": 0, "max_tree_depth": 3,
        },
    }
    options.update(hierarchical)
    pec = pnl.ParameterEstimationComposition(
        data=_group_frame(),
        fit_method="hierarchical",
        hierarchical_options=options,
        distributed_options={
            "pec_factory": functools.partial(_neural_participant_pec, artifact)
        },
    )
    return pec, pec.run()


@pytest.mark.composition
@pytest.mark.usefixtures("single_threaded_torch")
def test_a_sampled_fit_reports_draws_for_the_group_and_every_participant(trained_artifact):
    pec, results = _sample_group(trained_artifact)
    n_subjects, n_params = 2, 2

    assert results.group_draws.shape == (2, 25, 1, n_params)
    assert results.covariance_draws.shape == (2, 25, n_params, n_params)
    assert results.subject_draws.shape == (2, 25, n_subjects, n_params)
    assert results.subject_parameters.shape == (n_subjects, n_params)
    assert list(results.group_parameters.columns) == [
        "mean_z", "sd_z", "value", "lower_95", "upper_95", "lower", "upper"
    ]
    # Estimates stay inside the range the participants' models search.
    assert np.all(results.subject_draws >= np.array(RATE_BOUNDS[0]).min())
    assert results.settings["sampler"] == "nuts"
    assert pec.fit_results is results


@pytest.mark.composition
@pytest.mark.usefixtures("single_threaded_torch")
def test_a_sampled_fit_reports_whether_it_converged(trained_artifact):
    # 25 draws cannot have converged, and the result has to say so rather than presenting the
    # estimates as though they described the posterior.
    _, results = _sample_group(trained_artifact)
    assert set(results.convergence.columns) == {"r_hat", "ess"}
    assert len(results.convergence) == 4          # two group means, two log scales
    assert np.all(np.isfinite(results.convergence["r_hat"]))
    assert "r_hat" in repr(results) or "converged" in repr(results)


@pytest.mark.composition
@pytest.mark.usefixtures("single_threaded_torch")
def test_a_sampled_fit_with_a_full_covariance_samples_the_off_diagonals(trained_artifact):
    _, results = _sample_group(trained_artifact, covariance="full")
    # One more sampled quantity than the diagonal fit: the single below-diagonal entry.
    assert len(results.convergence) == 5
    correlation = results.group_correlation.to_numpy()
    assert np.allclose(np.diag(correlation), 1.0)
    assert not np.allclose(correlation, np.eye(2))


@pytest.mark.composition
@pytest.mark.usefixtures("single_threaded_torch")
def test_a_sampled_fit_reports_no_single_best_value(trained_artifact):
    # The draws are the result; the highest density among them describes where the sampler went.
    pec, results = _sample_group(trained_artifact)
    assert pec.optimal_value is None
    assert set(pec.optimized_parameter_values) == set(results.fit_param_names)


@pytest.mark.composition
@pytest.mark.usefixtures("single_threaded_torch")
def test_sampling_refuses_a_simulated_likelihood(trained_artifact):
    # Sampling differentiates the score, which simulating the model cannot provide.
    def simulated_participant(data, subject_index=None):
        pec, inputs = _neural_participant_pec(trained_artifact, data, subject_index)
        pec._likelihood_estimator = "kde"
        # Otherwise it is refused first for a reason every simulated model shares -- scoring that
        # does not repeat itself -- and the refusal being checked here would never be reached.
        pec.controller.parameters.same_seed_for_all_allocations.set(True)
        return pec, inputs

    pec = pnl.ParameterEstimationComposition(
        data=_group_frame(),
        fit_method="hierarchical",
        hierarchical_options={
            "subject_id": "subject", "sampler": "nuts",
            "sampler_options": {"draws": 2, "warmup": 2, "chains": 1, "max_tree_depth": 2},
        },
        distributed_options={"pec_factory": simulated_participant},
    )
    with pytest.raises(Exception, match="gives no gradient"):
        pec.run()
