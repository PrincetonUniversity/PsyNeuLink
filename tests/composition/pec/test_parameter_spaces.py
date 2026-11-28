"""PEC search-space semantics at the optimizer and simulation boundaries."""

import numpy as np
import optuna
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.parameterspace import _parse_parameter_space
from psyneulink.core.compositions.hierarchical.subjectlikelihood import ParameterSchema

pytestmark = pytest.mark.usefixtures("set_threads_to_one")

CONTINUOUS = {"bounds": (0., 3e-6)}
GRID = {"bounds": (0., 3e-6), "step": 1e-6}
CHOICES = {"values": [0., 1e-6, 3e-6]}


def _make_pec(spec, conditional=False):
    node = pnl.ProcessingMechanism(name="node")
    model = pnl.Composition(pathways=node)
    data = pd.DataFrame({"value": [0., 1.]})
    options = {}
    if conditional:
        data["condition"] = ["a", "b"]
        options["depends_on"] = {("slope", node): "condition"}
    return pnl.ParameterEstimationComposition(
        model=model,
        parameters={("slope", node): spec, ("intercept", node): {"bounds": (0., .3)}},
        outcome_variables=node,
        data=data,
        optimization_function=pnl.PECOptimizationFunction(method=optuna.samplers.RandomSampler, max_iterations=8),
        **options,
    )


class _ImmediateClient:
    """Record distributed submissions, evaluating an analytic objective locally."""

    def __init__(self):
        self.submitted = []
        self.released = False

    def nthreads(self):
        return {"worker": 1}

    def scatter(self, data, **kwargs):
        return data

    def submit(self, fn, factory, values, *args, **kwargs):
        self.submitted.append(list(values))
        return -sum(v * v for v in values)

    def gather(self, futures):
        return futures

    def cancel(self, data):
        self.released = True


@pytest.mark.parametrize("spec", [CONTINUOUS, GRID, CHOICES], ids=["continuous", "grid", "choices"])
@pytest.mark.parametrize("conditional", [False, True])
def test_serial_and_distributed_search_use_the_same_domains(spec, conditional):
    pec = _make_pec(spec, conditional)
    function = pec.controller.function
    names = function.fit_param_names
    serial = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7), direction="maximize")
    distributed = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=7), direction="maximize")
    evaluated = []

    def objective(*values):
        evaluated.append(list(values))
        return -sum(v * v for v in values)

    serial_result = function._fit_optuna(objective, serial, display_iter=False)
    function._distributed_options = {"pec_factory": lambda data: None, "max_concurrent_evaluations": 1}
    client = _ImmediateClient()
    distributed_result = function._fit_optuna_distributed(distributed, client)

    assert serial_result == distributed_result
    assert evaluated == client.submitted
    assert client.released
    assert len(evaluated) == 8
    assert list(function.fit_param_bounds) == names
    if spec == CONTINUOUS:
        schema = ParameterSchema.from_pec(pec, "test model")
        assert schema.lower == tuple(bound[0] for bound in function.fit_param_bounds.values())
        assert schema.upper == tuple(bound[1] for bound in function.fit_param_bounds.values())
    for index, name in enumerate(names[:-1]):
        values = np.array([trial.params[name] for trial in serial.trials])
        np.testing.assert_array_equal(values, np.array(evaluated)[:, index])
        if "values" in spec:
            assert set(values) <= set(spec["values"])
        elif "step" in spec:
            np.testing.assert_allclose(values / spec["step"], np.round(values / spec["step"]))
        else:
            assert np.all((values > 0.) & (values < 3e-6))
            assert serial.trials[0].distributions[name].step is None


@pytest.mark.parametrize("method, spec", [
    pytest.param("differential_evolution", CONTINUOUS, id="de_continuous"),
    pytest.param("optuna", CONTINUOUS, id="optuna_continuous"),
    pytest.param("optuna", GRID, id="optuna_grid"),
    pytest.param("optuna", CHOICES, id="optuna_choices"),
])
def test_parameter_spaces_reach_model_simulations(method, spec):
    node = pnl.TransferMechanism(name="node", noise=pnl.NormalDist(standard_deviation=0.))
    model = pnl.Composition(pathways=node)
    evaluated = []

    def objective(sim_data):
        value = float(np.mean(sim_data))
        evaluated.append(value)
        return (value - 8e-7) ** 2

    pec = pnl.ParameterEstimationComposition(
        model=model,
        parameters={("slope", node): spec},
        outcome_variables=node,
        objective_function=objective,
        optimization_function=pnl.PECOptimizationFunction(
            method=optuna.samplers.RandomSampler(seed=7) if method == "optuna" else method,
            max_iterations=8 if method == "optuna" else 1,
            direction="minimize",
        ),
    )
    pec.run(inputs={node: [[1.]]})
    fitted = next(iter(pec.optimized_parameter_values.values()))
    assert pec.optimal_value == pytest.approx((fitted - 8e-7) ** 2, abs=1e-25)
    assert len(evaluated) >= 8
    assert all(0. <= v <= 3e-6 for v in evaluated)
    if spec == CHOICES:
        assert set(evaluated) <= set(CHOICES["values"])
    elif spec == GRID:
        np.testing.assert_allclose(np.array(evaluated) / 1e-6, np.round(np.array(evaluated) / 1e-6))
    else:
        assert any(0. < v < 3e-6 and v not in CHOICES["values"] for v in evaluated)


@pytest.mark.parametrize("spec", [GRID, CHOICES], ids=["grid", "choices"])
def test_continuous_solvers_reject_explicit_discrete_spaces(spec):
    pec = _make_pec(spec)
    function = pec.controller.function
    # Bounds remain available for input preparation and likelihood scoring.
    assert next(iter(function.fit_param_bounds.values())) == (0., 3e-6)
    with pytest.raises(ValueError, match=r"node\.slope: continuous solvers"):
        ParameterSchema.from_pec(pec, "test model")


@pytest.mark.parametrize("method", ["shorthand", "distributed"])
def test_de_rejects_a_discrete_parameter_before_modifying_model(method):
    node = pnl.ProcessingMechanism(name="node")
    model = pnl.Composition(pathways=node)
    optimizer = "differential_evolution"
    if method == "distributed":
        optimizer = pnl.PECOptimizationFunction(method=optimizer, distributed=True)
    with pytest.raises(ValueError, match=r"node\.intercept: continuous solvers"):
        pnl.ParameterEstimationComposition(
            model=model, outcome_variables=node, objective_function=np.sum, optimization_function=optimizer,
            parameters={("slope", node): CONTINUOUS, ("intercept", node): GRID},
        )
    assert list(model.nodes) == [node]


@pytest.mark.parametrize("distributed", [False, True])
def test_sampler_cannot_override_pec_grid(distributed):
    function = _make_pec({"bounds": (0., 1.), "step": .5}).controller.function
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler(seed=7))
    # A valid first trial and an off-grid second trial in the same distributed batch.
    for value in (.5, .25):
        study.enqueue_trial({"node.slope": value})
    evaluated = []
    client = _ImmediateClient()
    with pytest.raises(ValueError, match=r"0\.25 for node\.slope is outside"):
        if distributed:
            function._distributed_options = {"pec_factory": lambda data: None, "max_concurrent_evaluations": 2}
            function._fit_optuna_distributed(study, client)
        else:
            def evaluate(*values):
                evaluated.append(values[0])
                return 0.
            function._fit_optuna(evaluate, study, display_iter=False)
    assert evaluated == ([] if distributed else [.5])
    assert client.submitted == []
    assert all(t.state.is_finished() for t in study.trials)


def test_grid_sampler_cannot_evaluate_an_off_grid_point():
    function = _make_pec({"bounds": (0., 1.), "step": .5}).controller.function
    sampler = optuna.samplers.GridSampler({"node.slope": [.25], "node.intercept": [0.]})
    with pytest.raises(ValueError, match=r"0\.25 for node\.slope is outside"):
        function._fit_optuna(lambda *args: pytest.fail("Evaluated an off-grid point"), sampler, display_iter=False)


def test_small_step_at_nonzero_offset_accepts_sampler_grid_points():
    function = _make_pec({"bounds": (1., 1.000000000001), "step": 1e-14}).controller.function
    result = function._fit_optuna(lambda slope, intercept: slope, optuna.samplers.RandomSampler(seed=7),
                                  display_iter=False)
    assert 1. < result["fitted_params"]["node.slope"] <= 1.000000000001


def test_resumed_study_cannot_return_a_best_trial_outside_current_bounds():
    function = _make_pec(CONTINUOUS).controller.function
    study = optuna.create_study(direction="maximize")
    study.add_trial(optuna.trial.create_trial(
        params={"node.slope": 2., "node.intercept": 0.}, value=10.,
        distributions={name: optuna.distributions.FloatDistribution(0., 2.) for name in function.fit_param_names},
    ))
    with pytest.raises(ValueError, match=r"2\.0 for node\.slope is outside"):
        function._fit_optuna(lambda *args: 0., study, display_iter=False)


@pytest.mark.parametrize("spec", [
    {}, {"bounds": (0., 1.), "values": [0., 1.]}, {"bound": (0., 1.)},
    {"bounds": (1., 0.)}, {"bounds": (0., 1., 2.)}, {"bounds": (0., np.inf)},
    {"bounds": (0., 1.), "step": 0.}, {"bounds": (0., 1.), "step": -1.},
    {"bounds": (0., 1.), "step": np.nan}, {"values": []}, {"values": [np.nan]},
    {"values": [[0., 1.]]}, {"values": np.array([1 + 2j])},
    {"values": np.array([np.complex128(1 + 2j)], dtype=object)},
    {"bounds": (0., 1.), "step": None}, {"bounds": (0., 1.), "step": [0.1]},
    [], [0., np.inf],
])
def test_invalid_parameter_spaces_are_rejected(spec):
    with pytest.raises(ValueError):
        _parse_parameter_space(spec)


@pytest.mark.parametrize("spec", [[0., 1.], (0., 1.), np.linspace(0., 1., 5)],
                         ids=["list", "tuple", "array"])
@pytest.mark.parametrize("method", ["differential_evolution", optuna.samplers.RandomSampler, None],
                         ids=["de", "optuna", "scoring_only"])
def test_bare_samples_fail_before_modifying_the_model(spec, method):
    node = pnl.ProcessingMechanism(name="node")
    model = pnl.Composition(pathways=node)
    with pytest.raises(ValueError, match="node.slope:.*explicit dictionaries") as error:
        pnl.ParameterEstimationComposition(
            model=model,
            parameters={("slope", node): spec},
            outcome_variables=node,
            data=pd.DataFrame({"value": [0., 1.]}),
            optimization_function=pnl.PECOptimizationFunction(method=method),
        )
    for form in ("'bounds'", "'step'", "'values'"):
        assert form in str(error.value)
    assert list(model.nodes) == [node]
    assert model.controller is None


@pytest.mark.parametrize("conditional", [False, True])
@pytest.mark.parametrize("bounds, step", [
    ((0., .0003), 1e-6),
    ((.1, .1003), 1e-6),
    ((0., 3e-16), 1e-16),
])
def test_explicit_grid_step_is_passed_through_without_rounding(bounds, step, conditional):
    pec = _make_pec({"bounds": bounds, "step": step}, conditional)
    function = pec.controller.function
    names = ["node.slope[a]", "node.slope[b]"] if conditional else ["node.slope"]
    assert function.fit_param_names == [*names, "node.intercept"]
    for name in names:
        assert function.fit_param_bounds[name] == bounds
        distribution = function.fit_param_distributions[name]
        assert distribution.step == step
        assert (distribution.low, distribution.high) == bounds


@pytest.mark.parametrize("values", [[.25], np.array([.9, .1, .25], dtype=np.float32)],
                         ids=["singleton", "unordered_float32"])
def test_exact_choices_preserve_supplied_values(values):
    pec = _make_pec({"values": values})
    distribution = pec.controller.function.fit_param_distributions["node.slope"]
    assert distribution.choices == tuple(float(value) for value in values)


def test_explicit_step_uses_upper_bound_as_a_cap():
    with pytest.warns(UserWarning, match="not divisible"):
        distribution = _parse_parameter_space({"bounds": (0., 1.), "step": .3}).optuna_distribution()
    assert distribution.high == .9
    assert distribution.step == .3
