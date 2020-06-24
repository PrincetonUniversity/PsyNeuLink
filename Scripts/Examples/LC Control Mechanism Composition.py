from psyneulink import *

G = 1.0
k = 0.5
starting_value_LC = 2.0
user_specified_gain = 1.0

A = TransferMechanism(function=Logistic(gain=user_specified_gain), name='A')
B = TransferMechanism(function=Logistic(gain=user_specified_gain), name='B')
# B.output_ports[0].value *= 0.0  # Reset after init | Doesn't matter here b/c default var = zero, no intercept

LC = LCControlMechanism(
    modulated_mechanisms=[A, B],
    base_level_gain=G,
    scaling_factor_gain=k,
    objective_mechanism=ObjectiveMechanism(
        function=Linear,
        monitor=[B],
        name='LC ObjectiveMechanism'
    )
)
for output_port in LC.output_ports:
    output_port.value *= starting_value_LC

path = [A, B, LC]
S = Composition()
S.add_node(A, required_roles=NodeRole.INPUT)
S.add_linear_processing_pathway(pathway=path)
S.add_node(LC, required_roles=NodeRole.OUTPUT)
LC.reset_stateful_function_when = Never()

gain_created_by_LC_output_port_1 = []
mod_gain_assigned_to_A = []
base_gain_assigned_to_A = []
mod_gain_assigned_to_B = []
base_gain_assigned_to_B = []
A_value = []
B_value = []
LC_value = []

def report_trial(system):
    gain_created_by_LC_output_port_1.append(LC.output_port.parameters.value.get(system)[0])
    mod_gain_assigned_to_A.append(A.get_mod_gain(system))
    mod_gain_assigned_to_B.append(B.get_mod_gain(system))
    base_gain_assigned_to_A.append(A.function.parameters.gain.get())
    base_gain_assigned_to_B.append(B.function.parameters.gain.get())
    A_value.append(A.parameters.value.get(system))
    B_value.append(B.parameters.value.get(system))
    LC_value.append(LC.parameters.value.get(system))

S.show_graph(show_controller=True)
# S.show_graph(show_controller=True, show_node_structure=ALL)

# result = S.run(inputs={A: [[1.0], [1.0], [1.0], [1.0], [1.0]]},
#               call_after_trial=functools.partial(report_trial, S))

