from psyneulink import *
import numpy as np

# Mechanisms
Input = TransferMechanism(name='Input')
reward = TransferMechanism(output_states=[RESULT, OUTPUT_MEAN, OUTPUT_VARIANCE],
                               name='reward')
Decision = DDM(function=DriftDiffusionAnalytical(drift_rate=(1.0,
                                                                     ControlProjection(function=Linear,
                                                                                           control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                         threshold=(1.0,
                                                                    ControlProjection(function=Linear,
                                                                                          control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})),
                                                         noise=0.5,
                                                         starting_point=0,
                                                         t0=0.45),
                   output_states=[DECISION_VARIABLE,
                                RESPONSE_TIME,
                                PROBABILITY_UPPER_THRESHOLD],
                   name='Decision')

comp = Composition(name="evc")
comp.add_node(reward, required_roles=[NodeRole.OUTPUT])
comp.add_node(Decision, required_roles=[NodeRole.OUTPUT])
task_execution_pathway = [Input, IDENTITY_MATRIX, Decision]
comp.add_linear_processing_pathway(task_execution_pathway)

comp.add_model_based_optimizer(optimizer=OptimizationControlMechanism(name='OCM',
                                                                      agent_rep=comp,
                                                                      features=[Input.input_state, reward.input_state],
                                                                      feature_function=AdaptiveIntegrator(rate=0.5),
                                                                      objective_mechanism=ObjectiveMechanism(
                                                                              name='OCM Objective Mechanism',
                                                                              function=LinearCombination(operation=PRODUCT),
                                                                              monitor=[reward,
                                                                                       Decision.output_states[PROBABILITY_UPPER_THRESHOLD],
                                                                                       (Decision.output_states[RESPONSE_TIME], -1, 1)]),
                                                                      function=GridSearch(),
                                                                      control_signals=[("drift_rate", Decision),
                                                                                       ("threshold", Decision)])
                               )

comp.enable_model_based_optimizer = True

stim_list_dict = {
    Input: [0.5, 0.123],
    reward: [20, 20]
}

comp.show_graph(show_model_based_optimizer=True, show_node_structure=ALL)
# comp.show_graph(show_model_based_optimizer=True)
# comp.run(inputs=stim_list_dict)
