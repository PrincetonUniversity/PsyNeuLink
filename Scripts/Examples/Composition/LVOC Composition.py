from psyneulink import *

m1 = TransferMechanism(input_states=["InputState A", "InputState B"])
m2 = TransferMechanism()
c = Composition()
c.add_node(m1, required_roles=NodeRole.INPUT)
c.add_node(m2, required_roles=NodeRole.INPUT)
c._analyze_graph()
lvoc = OptimizationControlMechanism(agent_rep=RegressionCFA,
                                        features=[m1.input_states[0], m1.input_states[1], m2.input_state],
                                        objective_mechanism=ObjectiveMechanism(
                                            monitor=[m1, m2]),
                                        function=GridSearch(max_iterations=1),
                                        control_signals=[(SLOPE, m1), (SLOPE, m2)])
c.add_node(lvoc)
input_dict = {m1: [[1], [1]], m2: [1]}

c.show_graph(model_based_optimizer_color=True)

# c.run(inputs=input_dict)

