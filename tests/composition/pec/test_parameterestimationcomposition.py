import numpy as np
import optuna
import pandas as pd
import pytest
import scipy
import contextlib

from packaging import version as pversion

import psyneulink as pnl

from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)

def _run_ddm_with_params(
    starting_value,
    rate,
    noise,
    threshold,
    non_decision_time,
    time_step_size,
    trial_inputs,
):
    """Create a composition with DDM and run it with the given parameters."""

    # Create a simple one mechanism composition containing a DDM in integrator mode.
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=starting_value,
            rate=rate,
            noise=noise,
            threshold=threshold,
            non_decision_time=non_decision_time,
            time_step_size=time_step_size,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    comp = pnl.Composition(pathways=decision)

    # Run the composition to generate some data to fit
    comp.run(inputs={decision: trial_inputs})
    results = comp.results

    data_to_fit = pd.DataFrame(
        np.squeeze(np.array(results)), columns=["decision", "response_time"]
    )
    data_to_fit["decision"] = data_to_fit["decision"].astype("category")

    return comp, data_to_fit


input_node_1 = pnl.ProcessingMechanism(input_shapes=1)
input_node_2 = pnl.ProcessingMechanism(input_shapes=3)
input_node_3 = pnl.ProcessingMechanism(input_shapes=2)
output_node = pnl.ProcessingMechanism(input_shapes=2)
model = pnl.Composition(
    [{input_node_1, input_node_2, input_node_3}, output_node], name="model"
)
pec = pnl.ParameterEstimationComposition(
    name="pec",
    model=model,
    parameters={("slope", output_node): np.linspace(1.0, 3.0, 3)},
    outcome_variables=output_node,
    objective_function=lambda x: np.sum(x),
    optimization_function=PECOptimizationFunction(method="differential_evolution"),
)
run_input_test_args = [
    pytest.param(
        {
            model: [
                [np.array([1.0]), np.array([2.0, 3.0, 4.0]), np.array([5.0, 6.0])],
                [np.array([7.0]), np.array([8.0, 9.0, 10.0]), np.array([11.0, 12.0])],
                [
                    np.array([13.0]),
                    np.array([14.0, 15.0, 16.0]),
                    np.array([17.0, 18.0]),
                ],
                [
                    np.array([19.0]),
                    np.array([20.0, 21.0, 22.0]),
                    np.array([23.0, 24.0]),
                ],
            ]
        },
        None,
        id="pec_good",
    ),
    pytest.param(
        {
            model: [
                [np.array([1.0]), np.array([2.0, 3.0, 4.0])],
                [np.array([7.0]), np.array([8.0, 9.0, 10.0]), np.array([11.0, 12.0])],
                [
                    np.array([13.0]),
                    np.array([14.0, 15.0, 16.0]),
                    np.array([17.0, 18.0]),
                ],
                [
                    np.array([19.0]),
                    np.array([20.0, 21.0, 22.0]),
                    np.array([23.0, 24.0]),
                ],
            ]
        },
        f"The array in the dict specified for the 'inputs' arg of pec.run() is badly formatted: "
        f"the length of each item in the outer dimension (a trial's worth of inputs) "
        f"must be equal to the number of inputs to 'model' (3).",
        id="pec_bad",
    ),
    pytest.param(
        {
            input_node_1: [
                [np.array([1.0])],
                [np.array([7.0])],
                [np.array([13.0])],
                [np.array([19.0])],
            ],
            input_node_2: [
                [np.array([2.0, 3.0, 4])],
                [np.array([8.0, 9.0, 10.0])],
                [np.array([14.0, 15.0, 16.0])],
                [np.array([20.0, 21.0, 22.0])],
            ],
            input_node_3: [
                [np.array([5.0, 6.0])],
                [np.array([11.0, 12.0])],
                [np.array([17.0, 18.0])],
                [np.array([23.0, 24.0])],
            ],
        },
        None,
        id="model_good",
    ),
    pytest.param(
        {
            input_node_1: [
                [np.array([1.0])],
                [np.array([7.0])],
                [np.array([13.0])],
                [np.array([19.0])],
            ],
            input_node_2: [
                [np.array([2.0, 3.0, 4])],
                [np.array([8.0, 9.0, 10.0])],
                [np.array([14.0, 15.0, 16.0])],
                [np.array([20.0, 21.0, 22.0])],
            ],
        },
        f"The dict specified in the `input` arg of pec.run() is badly formatted: "
        f"the number of entries should equal the number of inputs to 'model' (3).",
        id="model_bad",
    ),
]


@pytest.mark.composition
@pytest.mark.parametrize("inputs_dict, error_msg", run_input_test_args)
def test_pec_run_input_formats(inputs_dict, error_msg):
    if error_msg:
        with pytest.raises(pnl.ParameterEstimationCompositionError) as error:
            pec.run(inputs=inputs_dict)
        assert error.value.args[0] == error_msg
    else:
        pec.run(inputs=inputs_dict)


# SciPy changed their implementation of differential evolution and the way it selects
# samples to evaluate in 1.12 [0,1], and then again in 1.14 [2,3], leading to slightly
# different results
#
# [0] https://docs.scipy.org/doc/scipy/release/1.12.0-notes.html#scipy-optimize-improvements
# [1] https://github.com/scipy/scipy/pull/18496
# [2] https://docs.scipy.org/doc/scipy/release/1.14.0-notes.html#scipy-optimize-improvements
# [3] https://github.com/scipy/scipy/pull/20677
if pversion.parse(scipy.version.version) >= pversion.parse('1.14.0'):
    expected_differential_evolution = [0.010113000942356953]
elif pversion.parse(scipy.version.version) >= pversion.parse('1.12.0'):
    expected_differential_evolution = [0.010074123395259815]
else:
    expected_differential_evolution = [0.010363518438648106]

@pytest.mark.composition
@pytest.mark.parametrize(
    "opt_method, optuna_kwargs, expected_result, execution_context",
    [
        ("differential_evolution", None, expected_differential_evolution, contextlib.nullcontext()),
        (optuna.samplers.RandomSampler(seed=0), None, [0.01], contextlib.nullcontext()),
        (optuna.samplers.QMCSampler(seed=0), None, [0.01], contextlib.nullcontext()),
        (optuna.samplers.RandomSampler, {'seed': 0}, [0.01],
         pytest.warns(UserWarning, match="Overriding seed passed to optuna sampler with seed passed to PEC.")),
        (optuna.samplers.RandomSampler(), None, None,
         pytest.warns(UserWarning, match="initial_seed on PEC is not None, but instantiated optuna sampler is being used."))
    ],
    ids=[
        "differential_evolution",
        "optuna_random_sampler",
        "optuna_qmc_sampler",
        "optuna_random_sampler_with_kwargs",
        "optuna_random_sampler_no_seed"
    ],
)
def test_parameter_optimization_ddm(func_mode, opt_method, optuna_kwargs, expected_result, execution_context):
    """Test parameter optimization of a DDM in integrator mode"""

    if func_mode == "Python":
        pytest.skip(
            "Test not yet implemented for Python. Parameter estimation is too slow."
        )

    # High-level parameters the impact performance of the test
    num_trials = 50
    time_step_size = 0.01
    num_estimates = 300

    ddm_params = dict(
        starting_value=0.0,
        rate=0.3,
        noise=1.0,
        threshold=0.6,
        non_decision_time=0.15,
        time_step_size=time_step_size,
    )

    # Create a simple one mechanism composition containing a DDM in integrator mode.
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(**ddm_params),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    comp = pnl.Composition(pathways=decision)

    def reward_rate(sim_data):
        """
        Objective function for PEC to optimize. This function takes in the simulation data,
        a 3d array of shape (num_trials, num_estimates, num_outcome_vars), and returns a
        scalar value that is the reward rate.
        """
        return np.mean(sim_data[:, :, 0][:] / sim_data[:, :, 1][:])

    fit_parameters = {
        ("threshold", decision): np.linspace(0.01, 0.5, 10),  # Threshold
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=comp,
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        objective_function=reward_rate,
        optimization_function=PECOptimizationFunction(
            method=opt_method, optuna_kwargs=optuna_kwargs, max_iterations=50, direction="maximize"
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )
    pec.controller.parameters.comp_execution_mode.set(func_mode)

    # Let's generate an "experimental" dataset to fit. This is a parameter recovery test
    # Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
    # have a negative stimulus drift rate.
    rng = np.random.default_rng(12345)
    trial_inputs = rng.choice(
        [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
    )

    # Make the first and last input positive for sure. This helps make sure inputs are really getting
    # passed to the composition correctly during parameter fitting, and we aren't just getting a single
    # trials worth of a cached input.
    trial_inputs[0] = np.abs(trial_inputs[0])
    trial_inputs[-1] = np.abs(trial_inputs[-1])

    with execution_context:
        pec.run(inputs={comp: trial_inputs})

    if expected_result is not None:
        tolerance_args = {"atol": 1e-2} if opt_method == "differential_evolution" else {}
        np.testing.assert_allclose(
            list(pec.optimized_parameter_values.values()), expected_result, **tolerance_args
        )


def test_parameter_estimation_ddm_cond(func_mode):

    if func_mode == "Python":
        pytest.skip(
            "Test not yet implemented for Python. Parameter estimate is too slow."
        )

    # High-level parameters the impact performance of the test
    num_trials = 50
    time_step_size = 0.01
    num_estimates = 20

    # Let's generate an "experimental" dataset to fit. This is a parameter recovery test
    # Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
    # have a negative stimulus drift rate.
    rng = np.random.default_rng(12345)
    trial_inputs = rng.choice(
        [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
    )

    # Make the first and last input positive for sure. This helps make sure inputs are really getting
    # passed to the composition correctly during parameter fitting, and we aren't just getting a single
    # trials worth of a cached input.
    trial_inputs[0] = np.abs(trial_inputs[0])
    trial_inputs[-1] = np.abs(trial_inputs[-1])

    ddm_params = dict(
        starting_value=0.0,
        rate=0.3,
        noise=1.0,
        non_decision_time=0.15,
        time_step_size=time_step_size,
    )

    # We will generate a dataset that comprises two different conditions. Each condition will have a different
    # threshold.
    params_cond1 = dict(
        threshold=0.7,
    )

    params_cond2 = dict(
        threshold=0.3,
    )

    comp, data_cond1 = _run_ddm_with_params(**ddm_params, **params_cond1, trial_inputs=trial_inputs)
    _, data_cond2 = _run_ddm_with_params(**ddm_params, **params_cond2, trial_inputs=trial_inputs)

    # Combine the data from the two conditions
    data_cond1['condition'] = 'cond_t=0.7'
    data_cond2['condition'] = 'cond_t=0.3'
    data_to_fit = pd.concat([data_cond1, data_cond2])

    # Add the inputs as columns to the data temporarily so we can shuffle the data and shuffle the inputs together
    data_to_fit['inputs'] = np.concatenate([trial_inputs, trial_inputs])

    # Shuffle the data, seed is set for reproducibility
    data_to_fit = data_to_fit.sample(frac=1, random_state=42)

    # Extract the shuffled inputs
    trial_inputs = data_to_fit['inputs'].to_numpy().reshape(-1, 1)
    data_to_fit = data_to_fit.drop(columns='inputs')

    fit_parameters = {
        ("rate", comp.nodes['DDM']): np.linspace(-0.5, 0.5, 1000),
        ("non_decision_time", comp.nodes['DDM']): np.linspace(0.0, 1.0, 1000),
        ("threshold", comp.nodes['DDM']): np.linspace(0.1, 1.0, 1000),
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters=fit_parameters,
        depends_on={("threshold", comp.nodes['DDM']): 'condition'},
        outcome_variables=[
            comp.nodes['DDM'].output_ports[pnl.DECISION_OUTCOME],
            comp.nodes['DDM'].output_ports[pnl.RESPONSE_TIME],
        ],
        data=data_to_fit,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution", max_iterations=1,
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )

    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.parameters.save_values.set(True)
    pec.run(inputs={comp: trial_inputs})

    np.testing.assert_allclose(
        list(pec.optimized_parameter_values.values()),
        [0.13574824786818707, 0.04513454296326741, 0.49615574384553446, 0.8985587363124521]
    )


@pytest.mark.parametrize('likelihood_include_mask', [
    pytest.param('include', id='likelihood_include_mask'),
    pytest.param(None, id='no_likelihood_include_mask'),]
)
# func_mode is a hacky wa to get properly marked; Python, LLVM, and CUDA
@pytest.mark.composition
def test_parameter_estimation_ddm_mle(func_mode, likelihood_include_mask):
    """Test parameter estimation of a DDM in integrator mode with MLE."""

    if func_mode == "Python":
        pytest.skip(
            "Test not yet implemented for Python. Parameter estimate is too slow."
        )

    # High-level parameters the impact performance of the test
    num_trials = 50
    time_step_size = 0.01
    num_estimates = 200

    ddm_params = dict(
        starting_value=0.0,
        rate=0.3,
        noise=1.0,
        threshold=0.6,
        non_decision_time=0.15,
        time_step_size=time_step_size,
    )

    # Let's generate an "experimental" dataset to fit. This is a parameter recovery test
    # Lets make 10% of the trials have a positive stimulus drift rate, and the other 90%
    # have a negative stimulus drift rate.
    # trial_inputs = np.ones((num_trials, 1))
    rng = np.random.default_rng(12345)
    trial_inputs = rng.choice(
        [5.0, -5.0], size=(num_trials, 1), p=[0.10, 0.9], replace=True
    )

    # Make the first and last input positive for sure. This helps make sure inputs are really getting
    # passed to the composition correctly during parameter fitting, and we aren't just getting a single
    # trials worth of a cached input.
    trial_inputs[0] = np.abs(trial_inputs[0])
    trial_inputs[-1] = np.abs(trial_inputs[-1])

    # Creat and run the composition to generate some data to fit
    comp, data_to_fit = _run_ddm_with_params(**ddm_params, trial_inputs=trial_inputs)

    if likelihood_include_mask == 'include':
        likelihood_include_mask = np.ones((len(data_to_fit),), dtype=bool)

    # Create a parameter estimation composition to fit the data we just generated and hopefully recover the
    # parameters of the DDM.

    fit_parameters = {
        ("rate", comp.nodes['DDM']): np.linspace(-0.5, 0.5, 1000),
        ("threshold", comp.nodes['DDM']): np.linspace(0.5, 1.0, 1000),
        ("non_decision_time", comp.nodes['DDM']): np.linspace(0.0, 1.0, 1000),
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            comp.nodes['DDM'].output_ports[pnl.DECISION_OUTCOME],
            comp.nodes['DDM'].output_ports[pnl.RESPONSE_TIME],
        ],
        data=data_to_fit,
        likelihood_include_mask=likelihood_include_mask,
        optimization_function=PECOptimizationFunction(
            method="differential_evolution", max_iterations=1
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )

    pec.controller.parameters.comp_execution_mode.set(func_mode)
    pec.controller.function.parameters.save_values.set(True)
    pec.run(inputs={comp: trial_inputs})

    # The PEC was setup with max_iterations=1, we are just testing.
    # We won't recover the parameters accurately but we can check
    # against hardcoded values to make sure we are reproducing
    # the same search trajectory from a known working example.
    np.testing.assert_allclose(
        list(pec.optimized_parameter_values.values()),
        [0.2227273962084888, 0.5976130662377002, 0.1227723651473831],
    )


@pytest.mark.composition
def test_pec_bad_outcome_var_spec():
    """
    Tests that exception is raised when outcome variables specifies and output port that doesn't exist on the
    composition being fit.
    """
    ddm_params = dict(
        starting_value=0.0,
        rate=0.3,
        noise=1.0,
        threshold=0.6,
        non_decision_time=0.15,
        time_step_size=0.01,
    )

    # Create a simple one mechanism composition containing a DDM in integrator mode.
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(**ddm_params),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )

    # Add another dummy mechanism so output ports ont he composition are longer the DDM output ports.
    transfer = pnl.TransferMechanism()

    comp = pnl.Composition(pathways=[decision, transfer])

    # Make up some random data to fit
    data_to_fit = pd.DataFrame(
        np.random.random((20, 2)), columns=["decision", "response_time"]
    )
    data_to_fit["decision"] = data_to_fit["decision"] > 0.5
    data_to_fit["decision"] = data_to_fit["decision"].astype("category")

    # Create a parameter estimation composition to fit the data we just generated and hopefully recover the
    # parameters of the DDM.

    fit_parameters = {
        ("rate", decision): np.linspace(0.0, 0.4, 1000),
        ("threshold", decision): np.linspace(0.5, 1.0, 1000),
    }

    with pytest.raises(KeyError) as ex:
        pnl.ParameterEstimationComposition(
            name="pec",
            nodes=[comp],
            parameters=fit_parameters,
            outcome_variables=[
                decision.output_ports[pnl.DECISION_OUTCOME],
                decision.output_ports[pnl.RESPONSE_TIME],
            ],
            data=data_to_fit,
            optimization_function="differential_evolution",
            num_estimates=20,
            num_trials_per_estimate=10,
        )
    assert "Could not find outcome variable" in str(ex)

    with pytest.raises(ValueError) as ex:
        pnl.ParameterEstimationComposition(
            name="pec",
            nodes=[comp],
            parameters=fit_parameters,
            outcome_variables=[transfer.output_ports[0]],
            data=data_to_fit,
            optimization_function="differential_evolution",
            num_estimates=20,
            num_trials_per_estimate=10,
        )
    assert "The number of columns in the data to fit must match" in str(ex)


@pytest.mark.composition
def test_pec_controller_specified():
    """Test that an exception is raised if a controller is specified for the PEC."""
    with pytest.raises(ValueError):
        pnl.ParameterEstimationComposition(
            parameters={},
            outcome_variables=[],
            optimization_function="differential_evolution",
            controller=pnl.ControlMechanism(),
        )
