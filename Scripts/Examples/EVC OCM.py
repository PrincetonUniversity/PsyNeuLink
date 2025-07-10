from psyneulink import *
import numpy as np

# Mechanisms
Input = TransferMechanism(name='Input')
reward = TransferMechanism(
    output_ports=[RESULT, MEAN, VARIANCE],
    name='reward'
)
Decision = DDM(
    function=DriftDiffusionAnalytical(
        drift_rate=(1.0,
                    ControlProjection(
                        function=Linear,
                        control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})
                    ),
        threshold=(1.0,
                   ControlProjection(
                       function=Linear,
                       control_signal_params={ALLOCATION_SAMPLES: np.arange(0.1, 1.01, 0.3)})
                   ),
        noise=0.5,
        starting_value=0,
        non_decision_time=0.45
    ),
    output_ports=[DECISION_VARIABLE,
                  RESPONSE_TIME,
                  PROBABILITY_UPPER_THRESHOLD],
    name='Decision'
)

comp = Composition(name="evc")
comp.add_node(reward, required_roles=[NodeRole.OUTPUT])
comp.add_node(Decision, required_roles=[NodeRole.OUTPUT])
task_execution_pathway = [Input, IDENTITY_MATRIX, Decision]
comp.add_linear_processing_pathway(task_execution_pathway)

comp.add_controller(OptimizationControlMechanism(
    name='OCM',
    agent_rep=comp,
    state_features=[Input.input_port, reward.input_port],
    state_feature_function=AdaptiveIntegrator(rate=0.5),
    objective_mechanism=ObjectiveMechanism(
        name='OCM Objective Mechanism',
        function=LinearCombination(operation=PRODUCT),
        monitor=[reward,
                 Decision.output_ports[PROBABILITY_UPPER_THRESHOLD],
                 (Decision.output_ports[RESPONSE_TIME], -1, 1)]),
    function=GridSearch(),
    control_signals=[
        ControlSignal(modulates=("drift_rate", Decision), allocation_samples=[0.1, 0.3, 0.5, 0.7, 0.9]),
        ControlSignal(modulates=("threshold", Decision), allocation_samples=[0.1, 0.3, 0.5, 0.7, 0.9]),
    ])
)

comp.enable_model_based_optimizer = True

stim_list_dict = {
    Input: [0.5, 0.123],
    reward: [20, 20]
}

comp.show_graph(show_controller=True)


