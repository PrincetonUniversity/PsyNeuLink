from psyneulink import *

# Mechanisms
input = ProcessingMechanism(name='Input')
reward = ProcessingMechanism(name='Reward')

decision = DDM(
    name='Decision',
    function=DriftDiffusionAnalytical(
        drift_rate=1.0,
        threshold=1.0,
        noise=0.5,
        starting_value=0,
        non_decision_time=0.45
    ),
    output_ports=[DECISION_VARIABLE,
                  RESPONSE_TIME,
                  PROBABILITY_UPPER_THRESHOLD],
)

# Composition
comp = Composition(name="evc")
comp.add_node(reward)
comp.add_node(decision)
task_execution_pathway = [input, IDENTITY_MATRIX, decision]
comp.add_linear_processing_pathway(task_execution_pathway)

# Controller
comp.add_controller(OptimizationControlMechanism(
    name='OCM',
    agent_rep=comp,
    state_features=[input.input_port, reward.input_port],
    state_feature_function=AdaptiveIntegrator(rate=0.5),
    objective_mechanism=ObjectiveMechanism(
        name='OCM Objective Mechanism',
        function=LinearCombination(operation=PRODUCT),
        monitor=[reward,
                 decision.output_ports[PROBABILITY_UPPER_THRESHOLD],
                 (decision.output_ports[RESPONSE_TIME], -1, 1)]),
    function=GridSearch(),
    control_signals=[
        ControlSignal(modulates=(DRIFT_RATE, decision), allocation_samples=[0.1, 0.3, 0.5, 0.7, 0.9]),
        ControlSignal(modulates=(THRESHOLD, decision), allocation_samples=[0.1, 0.3, 0.5, 0.7, 0.9]),
    ])
)

stim_list_dict = {
    input: [0.5, 0.123],
    reward: [20, 20]
}

comp.show_graph(show_controller=True)

comp.run(inputs=stim_list_dict)

print(comp.results)
