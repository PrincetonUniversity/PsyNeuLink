"""Tests for neural likelihood estimation."""
import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)
from psyneulink.core.components.functions.nonstateful import (
    neurallikelihoodfunctions as nlf,
)

# Both are optional, and PsyNeuLink is tested without them, so these have to be reached
# through importorskip rather than imported: importing torch at module scope makes the
# module fail to collect instead of skipping.
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
        log_transform=True, n_trial_features=0, n_parameter_samples=len(theta),
        n_trials_per_sample=1, epochs=epochs, val_nll=val_nll, seed=seed,
        simulator_hash="toy", psyneulink_version="test", sbi_version="test",
    )
    return nlf.NeuralLikelihood(
        estimator, provenance, (x[:256].clone(), cond[:256].clone())
    ), raw


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


def test_curvature_is_available_by_autograd():
    """The neural path takes an exact Hessian rather than a finite-difference step."""
    likelihood, raw = _toy_likelihood()
    hessian = torch.autograd.functional.hessian(
        lambda t: likelihood.trial_log_prob(t, raw[:64]).sum(),
        torch.tensor([0.5, 0.9]),
    )
    assert hessian.shape == (2, 2)
    assert torch.isfinite(hessian).all()


def test_wrong_number_of_outcome_columns_is_rejected():
    likelihood, _ = _toy_likelihood(epochs=1)
    with pytest.raises(nlf.NeuralLikelihoodError, match="Expected outcomes with 2 columns"):
        likelihood.log_likelihood([0.5, 0.9], np.zeros((4, 3)))


def test_missing_trial_features_are_reported(tmp_path):
    likelihood, raw = _toy_likelihood(epochs=1)
    object.__setattr__(likelihood.provenance, "n_trial_features", 2)
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


def test_constant_inputs_contribute_no_trial_features():
    """A model whose trials are identical is conditioned on parameters alone."""
    assert nlf._trial_features({"node": np.ones((10, 1))}, 10) is None


def test_varying_inputs_become_trial_features():
    inputs = {"node": np.column_stack([np.arange(10.0), np.ones(10)])}
    features = nlf._trial_features(inputs, 10)
    assert features.shape == (10, 1)
    np.testing.assert_allclose(features[:, 0], np.arange(10.0))


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
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
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
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
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
    assert provenance.n_trial_features == 0
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
    nlf._simulate(pec, inputs, np.array([[0.3, 0.6]]), 2)
    assert pec.log_likelihood(0.3, 0.6, inputs=inputs) == before


@pytest.mark.composition
def test_inputs_set_how_many_trials_each_draw_simulates():
    """Trials come from the inputs, not from the data the model was built around."""
    frame = pd.DataFrame({"decision": [0.0, 1.0] * 5, "response_time": [0.5] * 10})
    frame["decision"] = frame["decision"].astype("category")
    pec, _ = _ddm_training_pec(frame)
    node = pec.nodes[0]

    _, _, ten = nlf._simulate(pec, {node: np.ones((10, 1))}, np.array([[0.3, 0.6]]), 2)
    _, _, thirty = nlf._simulate(pec, {node: np.ones((30, 1))}, np.array([[0.3, 0.6]]), 2)
    assert (ten, thirty) == (10, 30)


@pytest.mark.composition
def test_a_model_without_inputs_is_rejected(ddm_data):
    with pytest.raises(nlf.NeuralLikelihoodError, match="pec requires inputs"):
        nlf.train_neural_likelihood(BOUNDS, OUTCOMES, pec=object(), n_parameter_samples=8)
