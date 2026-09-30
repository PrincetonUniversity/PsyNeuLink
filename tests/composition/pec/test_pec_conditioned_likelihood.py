import copy

import numpy as np
import optuna
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.particlefilter import ParticleSupportError

pytestmark = [pytest.mark.composition, pytest.mark.llvm, pytest.mark.usefixtures("set_threads_to_one")]


def make_pec(*, data=(.8, 1.4, 2.8), estimates=64, same_seed=True, mode="auto",
             depends=False, method=None, options=None, mask=None, width=1):
    source = pnl.ProcessingMechanism(input_shapes=width)
    node = pnl.IntegratorMechanism(
        function=pnl.SimpleIntegrator(rate=1., noise=pnl.NormalDist(standard_deviation=.4)),
        reset_stateful_function_when=pnl.Never(),
    )
    model = pnl.Composition(pathways=[source, node])
    frame = pd.DataFrame({"value": data})
    if depends:
        frame["condition"] = pd.Categorical(["a", "b", "a"])
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={("rate", node): [.5, 1.]},
        depends_on={("rate", node): "condition"} if depends else None,
        outcome_variables=[node.output_port], data=frame,
        likelihood_include_mask=None if mask is None else np.array(mask),
        optimization_function=pnl.PECOptimizationFunction(
            method=method, max_iterations=2, conditioned_likelihood=mode,
            likelihood_options={"bandwidth": .3} if options is None else options,
        ),
        num_estimates=estimates, initial_seed=42,
        same_seed_for_all_parameter_combinations=same_seed,
    )
    return pec, source


@pytest.mark.parametrize("form", ["node", "port", "model"])
def test_public_score_replay_input_forms_and_immutability(form):
    pec, source = make_pec()
    key = {"node": source, "port": source.input_port, "model": pec.model}[form]
    inputs = {key: np.ones((3, 1))}
    original = inputs[key].copy()
    first, simulated = pec.log_likelihood(1., inputs=inputs, return_sim_data=True)
    diagnostics = copy.deepcopy(pec.controller.function.likelihood_diagnostics)
    second = pec.log_likelihood(1., inputs=inputs)
    assert first == second
    assert np.isfinite(first)
    assert simulated.shape == (3, 64, 1)
    np.testing.assert_array_equal(original, inputs[key])
    assert first == pytest.approx(diagnostics["per_trial_log_densities"].sum())
    assert pec.controller.parameters.num_trials_per_estimate.get() == 3


def test_depends_on_uses_full_sequence_masks_before_slicing():
    pec, source = make_pec(depends=True, options={"bandwidth": 1e8})
    _, simulations = pec.log_likelihood(.5, 1., inputs={source: np.ones((3, 1))}, return_sim_data=True)
    # Broad weights leave the nearly uniform ancestry intact. The middle
    # condition changes the drift, not the ordering of the trial sequence.
    np.testing.assert_allclose(np.mean(simulations, axis=1)[:, 0], [.5, 1.5, 2.], atol=.2)
    assert len(pec.controller.function.fit_param_names) == 2


def test_fresh_seeds_apply_between_candidate_evaluations():
    pec, source = make_pec(same_seed=False)
    first, x = pec.log_likelihood(1., inputs={source: np.ones((3, 1))}, return_sim_data=True)
    second, y = pec.log_likelihood(1., inputs={source: np.ones((3, 1))}, return_sim_data=True)
    assert first != second
    assert not np.array_equal(x[0], y[0])


def test_masked_history_changes_later_prediction_without_changing_initial_simulations():
    pec, source = make_pec(mask=[False, True, True])
    first, x = pec.log_likelihood(1., inputs={source: np.ones((3, 1))}, return_sim_data=True)
    pec._data_numpy[0, 0] += 1.
    second, y = pec.log_likelihood(1., inputs={source: np.ones((3, 1))}, return_sim_data=True)
    np.testing.assert_array_equal(x[0], y[0])
    assert not np.array_equal(x[1], y[1])
    assert first != second


def test_support_error_restores_controller_and_following_call_replays():
    pec, source = make_pec(options={"kernel": "histogram", "bin_range": [(0., 100.)], "bins": 1000,
                                   "smoothing_sigma": 0.})
    pec._data_numpy[:] = 99.
    with pytest.raises(ParticleSupportError, match="trial 0"):
        pec.log_likelihood(1., inputs={source: np.ones((3, 1))})
    assert pec.controller.parameters.num_trials_per_estimate.get() == 3
    assert len(pec.controller._pec_input_values[pec.model]) == 3
    assert pec.controller.function.likelihood_diagnostics is None
    pec.controller.function.likelihood_options = {"bandwidth": 1.}
    assert np.isfinite(pec.log_likelihood(1., inputs={source: np.ones((3, 1))}))


def test_optimizer_uses_the_same_conditional_scorer():
    pec, source = make_pec(method=optuna.samplers.RandomSampler(seed=1))
    inputs = {source: np.ones((3, 1))}
    pec.run(inputs=inputs)
    params = list(pec.optimized_parameter_values.values())
    assert len(params) == 1
    assert pec.optimal_value == pytest.approx(pec.log_likelihood(*params, inputs=inputs))


def test_separate_pecs_do_not_share_particle_populations():
    pec, source = make_pec()
    inputs = {source: np.ones((3, 1))}
    # Public scoring uses initialized component contexts; independent PECs also
    # isolate subject histories and reproduce their declared initial seed.
    other, other_source = make_pec()
    assert pec.log_likelihood(1., inputs=inputs) == other.log_likelihood(1., inputs={other_source: np.ones((3, 1))})


def test_ragged_inputs_and_dictionary_order_agree_with_composition_inputs():
    scalar = pnl.ProcessingMechanism(input_shapes=1)
    vector = pnl.ProcessingMechanism(input_shapes=2)
    node = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist()))
    model = pnl.Composition(pathways=[[scalar, node], [vector, node]])
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={("rate", node): [.5, 1.]}, outcome_variables=[node.output_port],
        data=pd.DataFrame({"value": [1., 2., 3.]}), num_estimates=8, initial_seed=4,
        same_seed_for_all_parameter_combinations=True,
        optimization_function=pnl.PECOptimizationFunction(method=None, likelihood_options={"bandwidth": .5}),
    )
    scalar_values = np.array([[1.], [.5], [2.]])
    vector_values = np.array([[.1, .2], [.2, .3], [.3, .4]])
    reference = pec.log_likelihood(.5, inputs={scalar: scalar_values, vector: vector_values}, return_sim_data=True)
    for inputs in (
        {vector.input_port: vector_values, scalar.input_port: scalar_values},
        {model: [[a.copy(), b.copy()] for a, b in zip(scalar_values, vector_values)]},
    ):
        score, simulations = pec.log_likelihood(.5, inputs=inputs, return_sim_data=True)
        assert score == reference[0]
        np.testing.assert_array_equal(simulations, reference[1])


def test_forced_filter_on_trial_resetting_ddm_preserves_predictive_draws():
    node = pnl.DDM(function=pnl.DriftDiffusionIntegrator(rate=.5, noise=.3, threshold=.2,
                                                        time_step_size=.05))
    model = pnl.Composition(pathways=[node])
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={("rate", node): [.5, 1.]},
        outcome_variables=[node.output_ports[pnl.RESPONSE_TIME]],
        data=pd.DataFrame({"rt": [.4, .5, .6]}), num_estimates=8, initial_seed=3,
        same_seed_for_all_parameter_combinations=True,
        optimization_function=pnl.PECOptimizationFunction(method=None, conditioned_likelihood=False),
    )
    inputs = {node: np.ones((3, 1))}
    # Bypass the independent KDE here; only compare its unchanged simulator.
    pec.controller.function.set_pec_objective_function(lambda samples: 0.)
    _, full = pec.log_likelihood(.5, inputs=inputs, return_sim_data=True)
    assert not pec.likelihood_history.requires_conditioning
    pec.controller.function.conditioned_likelihood = True
    _, split = pec.log_likelihood(.5, inputs=inputs, return_sim_data=True)
    np.testing.assert_array_equal(split, full)


@pytest.mark.parametrize("control_mode,expected", [(pnl.BEFORE, [0., 2., 3.]), (pnl.AFTER, [1., 2., 3.])])
def test_model_controller_is_executed_inside_particle_simulations(control_mode, expected):
    target = pnl.TransferMechanism()
    allocation = pnl.ProcessingMechanism()
    controller = pnl.ControlMechanism(monitor_for_control=allocation,
                                       control_signals=[("slope", target)], modulation=pnl.OVERRIDE)
    model = pnl.Composition(nodes=[target, allocation], controller=controller,
                             enable_controller=True, controller_mode=control_mode)
    model.require_node_roles(target, pnl.NodeRole.OUTPUT)
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={("slope", allocation): [1., 2.]}, outcome_variables=[target.output_port],
        data=pd.DataFrame({"value": expected}), num_estimates=1,
        optimization_function=pnl.PECOptimizationFunction(method=None, likelihood_options={"bandwidth": 1.}),
    )
    assert pec.likelihood_history.requires_conditioning
    _, samples = pec.log_likelihood(1., inputs={target: [[1.]] * 3, allocation: [[2.], [3.], [4.]]}, return_sim_data=True)
    np.testing.assert_allclose(samples[:, 0, 0], expected)


def _conditioned_worker_factory(data, subject_index=None):
    pec, source = make_pec(data=data["value"], estimates=8)
    return pec, {source: np.ones((len(data), 1))}


def test_worker_and_subject_factory_scoring_replay_conditional_likelihood():
    from psyneulink.core.components.functions.nonstateful.fitfunctions import _dask_evaluate_loglik, _PEC_FALLBACK_CACHE
    from psyneulink.core.compositions.hierarchical.subjectlikelihood import PECFactorySubjectLikelihood
    frame = pd.DataFrame({"value": [.8, 1.4, 2.8]})
    pec, inputs = _conditioned_worker_factory(frame)
    reference = pec.log_likelihood(1., inputs=inputs)
    fit_id = object()
    provider = PECFactorySubjectLikelihood(_conditioned_worker_factory, [frame, frame.copy()])
    try:
        for _ in range(2):
            assert _dask_evaluate_loglik(_conditioned_worker_factory, [1.], frame, 1, fit_id) == reference
        for subject in [0, 1, 0]:
            assert provider.log_likelihood([1.], subject) == reference
    finally:
        provider.close()
        _PEC_FALLBACK_CACHE.clear()


def test_worker_zero_support_has_same_invalid_candidate_result_as_local_fit(monkeypatch):
    from psyneulink.core.components.functions.nonstateful.fitfunctions import _dask_evaluate_loglik, _PEC_FALLBACK_CACHE
    from psyneulink.core.compositions.hierarchical import distributedestep
    from psyneulink.core.compositions.hierarchical.subjectlikelihood import PECFactorySubjectLikelihood

    def factory(data, subject_index=None):
        pec, source = make_pec(data=(99., 99., 99.), estimates=4,
                              options={"kernel": "histogram", "bin_range": [(0., 100.)],
                                       "bins": 1000, "smoothing_sigma": 0.})
        return pec, {source: np.ones((3, 1))}

    with pytest.warns(pnl.BadLikelihoodWarning, match="Zero particle"):
        assert _dask_evaluate_loglik(factory, [1.], None, 1, object()) == -np.inf
    provider = PECFactorySubjectLikelihood(factory, [None])
    with pytest.warns(pnl.BadLikelihoodWarning, match="Zero particle"):
        assert provider.log_likelihood([1.], 0) == -np.inf

    def evaluate_candidate(neg_log_post, **kwargs):
        assert neg_log_post(np.zeros(1)) == np.inf
        return "invalid candidate"

    monkeypatch.setattr(distributedestep, "subject_map_estep", evaluate_candidate)
    fit_id = object()
    try:
        with pytest.warns(pnl.BadLikelihoodWarning, match="Zero particle"):
            _, result, _ = distributedestep._dask_subject_estep(
                factory, 0, None, np.zeros(1), np.ones(1), provider.schema,
                np.zeros(1), 1, fit_id, None,
            )
        assert result == "invalid candidate"
    finally:
        distributedestep._release_fit_models(fit_id)
        provider.close()
        _PEC_FALLBACK_CACHE.clear()


@pytest.mark.parametrize("mode,expected", [("auto", True), (False, False), (True, True)])
def test_pec_auto_and_explicit_selection(mode, expected):
    node = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator(noise=pnl.NormalDist()))
    model = pnl.Composition(pathways=[node])
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={("rate", node): [.5, 1.]}, outcome_variables=[node.output_port],
        data=pd.DataFrame({"value": [1., 2.]}), num_estimates=4,
        optimization_function=pnl.PECOptimizationFunction(method=None, conditioned_likelihood=mode),
    )
    assert pec.likelihood_history.requires_conditioning
    assert pec.controller.function._uses_conditioned_likelihood() is expected


def test_custom_objective_does_not_select_particle_likelihood():
    node = pnl.IntegratorMechanism(function=pnl.SimpleIntegrator)
    pec = pnl.ParameterEstimationComposition(
        model=pnl.Composition(pathways=[node]), parameters={("rate", node): [.5, 1.]},
        outcome_variables=[node.output_port], objective_function=lambda samples: np.mean(samples),
        optimization_function=pnl.PECOptimizationFunction(method=None),
    )
    assert not pec.controller.function._uses_conditioned_likelihood()


def test_llvm_conditional_likelihood_matches_kalman_sequence_oracle():
    from scipy.stats import norm

    observed = np.array([.6, 1.6, 3.1, 5.])
    pec, source = make_pec(data=observed, estimates=8192, options={"bandwidth": .7})
    score = pec.log_likelihood(1., inputs={source: np.ones((4, 1))})
    mean, variance, expected = 0., 0., []
    for value in observed:
        mean += 1.
        variance += .4 ** 2
        expected.append(norm.logpdf(value, mean, np.sqrt(variance + .7 ** 2)))
        gain = variance / (variance + .7 ** 2)
        mean += gain * (value - mean)
        variance *= 1 - gain
    diagnostics = pec.controller.function.likelihood_diagnostics
    np.testing.assert_allclose(diagnostics["per_trial_log_densities"], expected, atol=.04, rtol=0)
    assert score == pytest.approx(sum(expected), abs=.08)
