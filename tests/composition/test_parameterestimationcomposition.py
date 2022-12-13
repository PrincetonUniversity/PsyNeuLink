import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.components.functions.nonstateful.combinationfunctions import \
    LinearCombination, Concatenate
from psyneulink.core.components.functions.nonstateful.distributionfunctions import DriftDiffusionAnalytical
from psyneulink.core.components.functions.nonstateful.optimizationfunctions import GridSearch
from psyneulink.core.components.functions.fitfunctions import MaxLikelihoodEstimator
from psyneulink.core.components.projections.modulatory.controlprojection import ControlProjection
from psyneulink.library.components.mechanisms.processing.integrator.ddm import \
    DDM, DECISION_VARIABLE, RESPONSE_TIME, PROBABILITY_UPPER_THRESHOLD


# All tests are set to run. If you need to skip certain tests,
# see http://doc.pytest.org/en/latest/skipping.html

# Unit tests for ParameterEstimationComposition

# objective_function = {None: 2, Concatenate: 2, LinearCombination: 1}
# expected

pec_test_args = [
    # (None, 2, True, False),               # No ObjectiveMechanism, 2 inputs, model, no nodes or pathways arg
    (None, 2, False, True),               # No ObjectiveMechanism, 2 inputs, no model, nodes or pathways arg
    (Concatenate, 2, True, False),        # ObjectiveMechanism, 2 inputs, model, no nodes or pathways arg
    (LinearCombination, 1, True, False),  # ObjectiveMechanism, 1 input, model, no nodes or pathways arg
    # (None, 2, True, True), <- USE TO TEST ERROR
    # (None, 2, False, False), <- USE TO TEST ERROR
]

@pytest.mark.parametrize(
    'objective_function_arg, expected_outcome_input_len, model_spec, node_spec',
    pec_test_args,
    ids=[f"{x[0]}-{'model' if x[2] else None}-{'nodes' if x[3] else None})" for x in pec_test_args]
)
def test_parameter_estimation_composition(objective_function_arg, expected_outcome_input_len, model_spec, node_spec):
    """Test with and without ObjectiveMechanism specified, and use of model vs. nodes arg of PEC constructor"""
    samples = np.arange(0.1, 1.01, 0.3)
    Input = pnl.TransferMechanism(name='Input')
    reward = pnl.TransferMechanism(output_ports=[pnl.RESULT, pnl.MEAN, pnl.VARIANCE],
                                   name='reward',
                                   # integrator_mode=True,
                                   # noise=NormalDist  # <- FIX 11/3/31: TEST ALLOCATION OF SEED FOR THIS WHEN WORKING
                                   )
    Decision = DDM(function=DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                 ControlProjection(function=pnl.Linear,
                                                                                   control_signal_params={
                                                                                       pnl.ALLOCATION_SAMPLES: samples,
                                                                                   })),
                                                     threshold=(1.0,
                                                                ControlProjection(function=pnl.Linear,
                                                                                  control_signal_params={
                                                                                      pnl.ALLOCATION_SAMPLES: samples,
                                                                                  })),
                                                     noise=0.5,
                                                     starting_value=0,
                                                     non_decision_time=0.45),
                   output_ports=[DECISION_VARIABLE,
                                 RESPONSE_TIME,
                                 PROBABILITY_UPPER_THRESHOLD],
                   name='Decision1')
    Decision2 = DDM(function=DriftDiffusionAnalytical(drift_rate=1.0,
                                                      threshold=1.0,
                                                      noise=0.5,
                                                      starting_value=0,
                                                      non_decision_time=0.45),
                    output_ports=[DECISION_VARIABLE,
                                  RESPONSE_TIME,
                                  PROBABILITY_UPPER_THRESHOLD],
                    name='Decision2')


    comp = pnl.Composition(name="evc", retain_old_simulation_data=True)
    comp.add_node(reward, required_roles=[pnl.NodeRole.OUTPUT])
    comp.add_node(Decision, required_roles=[pnl.NodeRole.OUTPUT])
    comp.add_node(Decision2, required_roles=[pnl.NodeRole.OUTPUT])
    task_execution_pathway = [Input, pnl.IDENTITY_MATRIX, Decision, Decision2]
    comp.add_linear_processing_pathway(task_execution_pathway)

    pec = pnl.ParameterEstimationComposition(name='pec',
                                             model=comp if model_spec else None,
                                             nodes=comp if node_spec else None,
                                             # data = [1,2,3],    # For testing error
                                             parameters={('drift_rate',Decision):[.1, .2],
                                                         ('threshold',Decision):[.5, .6],},
                                             # parameters={('shrimp_boo',Decision):[1,2],   # For testing error
                                             #             ('scripblat',Decision2):[1,2],}, # For testing error
                                             outcome_variables=[Decision.output_ports[DECISION_VARIABLE],
                                                                Decision.output_ports[RESPONSE_TIME]],
                                             objective_function=objective_function_arg,
                                             optimization_function=GridSearch,
                                             num_estimates=3,
                                             # controller_mode=AFTER,   # For testing error
                                             # enable_controller=False  # For testing error
                                             )
    ctlr = pec.controller

    assert ctlr.num_outcome_input_ports == 1
    if objective_function_arg:
        # pec.show_graph(show_cim=True)
        # pec.show_graph(show_node_structure=pnl.ALL)
        assert ctlr.objective_mechanism                         # For objective_function specified
    else:
        # pec.show_graph(show_cim=True)
        # pec.show_graph(show_node_structure=pnl.ALL)
        assert not ctlr.objective_mechanism                         # For objective_function specified
    assert len(ctlr.input_ports[pnl.OUTCOME].variable) == expected_outcome_input_len
    assert len(ctlr.control_signals) == 3
    assert ctlr.function.num_estimates == 3
    assert pnl.RANDOMIZATION_CONTROL_SIGNAL in ctlr.control_signals.names
    assert ctlr.control_signals[pnl.RANDOMIZATION_CONTROL_SIGNAL].allocation_samples.num == 3
    pec.run()


# func_mode is a hacky wa to get properly marked; Python, LLVM, and CUDA
def test_parameter_estimation_ddm_mle(func_mode):
    """Test parameter estimation of a DDM in integrator mode with MLE."""

    if func_mode == 'Python':
        pytest.skip("Test not yet implemented for Python. Parameter estimate is too slow.")
        return

    # High-level parameters the impact performance of the test
    num_trials = 25
    time_step_size = 0.01
    num_estimates = 40000

    ddm_params = dict(starting_value=0.0, rate=0.3, noise=1.0,
                      threshold=0.6, non_decision_time=0.15, time_step_size=time_step_size)

    # Create a simple one mechanism composition containing a DDM in integrator mode.
    decision = pnl.DDM(function=pnl.DriftDiffusionIntegrator(**ddm_params),
                       output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
                       name='DDM')

    comp = pnl.Composition(pathways=decision)

    # Let's generate an "experimental" dataset to fit. This is a parameter recovery test
    # The input will be num_trials trials of the same constant stimulus drift rate of 1
    # input = np.concatenate((np.repeat(-30.0, 30), np.repeat(30.0, 30)))[:, None]
    trial_inputs = np.ones((num_trials, 1))
    inputs_dict = {decision: trial_inputs}

    # Store the results of this "experiment" as a numpy array. This should be a
    # 2D array of shape (len(input), 2). The first column being a discrete variable
    # specifying whether the upper or lower decision boundary is reached and the second column is the
    # reaction time. We will put the data into a pandas DataFrame, this makes it
    # easier to specify which columns in the data are categorical or not.

    # Run the composition to generate some data to fit
    comp.run(inputs=inputs_dict)
    results = comp.results

    data_to_fit = pd.DataFrame(np.squeeze(np.array(results)), columns=['decision', 'response_time'])
    data_to_fit['decision'] = data_to_fit['decision'].astype('category')

    # Create a parameter estimation composition to fit the data we just generated and hopefully recover the
    # parameters of the DDM.

    fit_parameters = {
        ('rate', decision): np.linspace(0.0, 0.4, 1000),
        ('threshold', decision): np.linspace(0.5, 1.0, 1000),
        # ('non_decision_time', decision): np.linspace(0.0, 1.0, 1000),
    }

    pec = pnl.ParameterEstimationComposition(name='pec',
                                             # nodes=[comp],
                                             model=comp,
                                             parameters=fit_parameters,
                                             outcome_variables=[decision.output_ports[pnl.DECISION_OUTCOME],
                                                                decision.output_ports[pnl.RESPONSE_TIME]],
                                             data=data_to_fit,
                                             optimization_function=MaxLikelihoodEstimator(),
                                             num_estimates=num_estimates,
                                             num_trials_per_estimate=len(trial_inputs),
                                             )

    pec.controller.parameters.comp_execution_mode.set("LLVM")
    pec.controller.function.parameters.save_values.set(True)
    ret = pec.run(inputs={comp: trial_inputs}, num_trials=len(trial_inputs))

    # Check that the parameters are recovered and that the log-likelihood is correct, set the tolerance pretty high,
    # things are noisy because of the low number of trials and estimates.
    assert np.allclose(pec.controller.optimal_parameters, [0.3, 0.6], atol=0.1)
