import numpy as np
import pandas as pd
import pytest
import optuna

import psyneulink as pnl

from psyneulink.core.components.functions.nonstateful.fitfunctions import (
    PECOptimizationFunction,
)


input_node_1 = pnl.ProcessingMechanism(size=1)
input_node_2 = pnl.ProcessingMechanism(size=2)
input_node_3 = pnl.ProcessingMechanism(size=3)
output_node = pnl.ProcessingMechanism(size=2)
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


@pytest.mark.parametrize("inputs_dict, error_msg", run_input_test_args)
def test_pec_run_input_formats(inputs_dict, error_msg):
    if error_msg:
        with pytest.raises(pnl.ParameterEstimationCompositionError) as error:
            pec.run(inputs=inputs_dict)
        assert error.value.args[0] == error_msg
    else:
        pec.run(inputs=inputs_dict)


@pytest.mark.parametrize(
    "opt_method",
    [
        "differential_evolution",
        optuna.samplers.RandomSampler(),
        optuna.samplers.CmaEsSampler(),
    ],
    ids=["differential_evolultion", "optuna_random_sampler", "optuna_cmaes_sampler"],
)
def test_parameter_optimization_ddm(func_mode, opt_method):
    """Test parameter optimization of a DDM in integrator mode"""

    if func_mode == "Python":
        pytest.skip(
            "Test not yet implemented for Python. Parameter estimation is too slow."
        )

    # High-level parameters the impact performance of the test
    num_trials = 50
    time_step_size = 0.01
    num_estimates = 400

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
        a 3D array of shape (num_trials, num_estimates, num_outcome_vars), and returns a
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
            method=opt_method, max_iterations=50, direction="maximize"
        ),
        num_estimates=num_estimates,
        initial_seed=42,
    )
    pec.controller.parameters.comp_execution_mode.set(func_mode)

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

    inputs_dict = {decision: trial_inputs}

    ret = pec.run(inputs={comp: trial_inputs})

    np.testing.assert_allclose(
        pec.optimized_parameter_values, [0.010363518438648106], atol=1e-2
    )


# func_mode is a hacky wa to get properly marked; Python, LLVM, and CUDA
def test_parameter_estimation_ddm_mle(func_mode):
    """Test parameter estimation of a DDM in integrator mode with MLE."""

    if func_mode == "Python":
        pytest.skip(
            "Test not yet implemented for Python. Parameter estimate is too slow."
        )

    # High-level parameters the impact performance of the test
    num_trials = 50
    time_step_size = 0.01
    num_estimates = 40000

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

    inputs_dict = {decision: trial_inputs}

    # Store the results of this "experiment" as a numpy array. This should be a
    # 2D array of shape (len(input), 2). The first column being a discrete variable
    # specifying whether the upper or lower decision boundary is reached and the second column is the
    # reaction time. We will put the data into a pandas DataFrame, this makes it
    # easier to specify which columns in the data are categorical or not.

    # Run the composition to generate some data to fit
    comp.run(inputs=inputs_dict)
    results = comp.results

    data_to_fit = pd.DataFrame(
        np.squeeze(np.array(results)), columns=["decision", "response_time"]
    )
    data_to_fit["decision"] = data_to_fit["decision"].astype("category")

    # Create a parameter estimation composition to fit the data we just generated and hopefully recover the
    # parameters of the DDM.

    fit_parameters = {
        ("rate", decision): np.linspace(-0.5, 0.5, 1000),
        ("threshold", decision): np.linspace(0.5, 1.0, 1000),
        ("non_decision_time", decision): np.linspace(0.0, 1.0, 1000),
    }

    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters=fit_parameters,
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data_to_fit,
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
        pec.optimized_parameter_values,
        [0.2227273962084888, 0.5976130662377002, 0.1227723651473831],
    )


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

    with pytest.raises(ValueError) as ex:
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


def test_pec_controller_specified():
    """Test that an exception is raised if a controller is specified for the PEC."""
    with pytest.raises(ValueError):
        pnl.ParameterEstimationComposition(
            parameters={},
            outcome_variables=[],
            optimization_function="differential_evolution",
            controller=pnl.ControlMechanism(),
        )
