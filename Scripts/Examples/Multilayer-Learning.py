import functools
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

Input_Layer = pnl.TransferMechanism(
    name='Input Layer',
    function=psyneulink.core.components.functions.transferfunctions.Logistic,
    params={pnl.INPUT_LABELS_DICT:{'red': [-1, 30]}},
    default_variable=np.zeros((2,)))

Hidden_Layer_1 = pnl.TransferMechanism(
    name='Hidden Layer_1',
    function=psyneulink.core.components.functions.transferfunctions.Logistic(),
    default_variable=np.zeros((5,)))

Hidden_Layer_2 = pnl.TransferMechanism(
    name='Hidden Layer_2',
    function=psyneulink.core.components.functions.transferfunctions.Logistic(),
    default_variable=[0, 0, 0, 0])

Output_Layer = pnl.TransferMechanism(
    name='Output Layer',
    function=psyneulink.core.components.functions.transferfunctions.Logistic,
    default_variable=[0, 0, 0])

Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

# This Projection will be used by the Process below by referencing it in the Process' pathway;
#    note: sender and receiver args don't need to be specified
Input_Weights = pnl.MappingProjection(
    name='Input Weights',
    matrix=Input_Weights_matrix
)

# This Projection will be used by the Process below by assigning its sender and receiver args
#    to mechanisms in the pathway
Middle_Weights = pnl.MappingProjection(
    name='Middle Weights',
    sender=Hidden_Layer_1,
    receiver=Hidden_Layer_2,
    matrix=Middle_Weights_matrix
)

# Treated same as Middle_Weights Projection
Output_Weights = pnl.MappingProjection(
    name='Output Weights',
    sender=Hidden_Layer_2,
    receiver=Output_Layer,
    matrix=Output_Weights_matrix
)

z = pnl.Pathway(
    pathway=[
        Input_Layer,
        # The following reference to Input_Weights is needed to use it in the pathway
        #    since it's sender and receiver args are not specified in its declaration above
        Input_Weights,
        Hidden_Layer_1,
        # Middle_Weights,
        # No Projection specification is needed here since the sender arg for Middle_Weights
        #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
        Hidden_Layer_2,
        # Output_Weights,
        # Output_Weights does not need to be listed for the same reason as Middle_Weights
        # If Middle_Weights and/or Output_Weights is not declared above, then the Process
        #    will assign a default for rhe missing Projection
        Output_Layer
    ]
)


def print_header(comp):
    print("\n\n**** Time: ", comp.scheduler.clock.simple_time)


def show_target(comp):
    i = comp.external_input_values
    t = comp.pathways[0].target.input_ports[0].parameters.value.get(comp)
    print('\nOLD WEIGHTS: \n')
    print('- Input Weights: \n', Input_Weights.parameters.matrix.get(comp))
    print('- Middle Weights: \n', Middle_Weights.parameters.matrix.get(comp))
    print('- Output Weights: \n', Output_Weights.parameters.matrix.get(comp))

    print('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
    print('ACTIVITY FROM OLD WEIGHTS: \n')
    print('- Middle 1: \n', Hidden_Layer_1.parameters.value.get(comp))
    print('- Middle 2: \n', Hidden_Layer_2.parameters.value.get(comp))
    print('- Output:\n', Output_Layer.parameters.value.get(comp))


comp = pnl.Composition(name='Multilayer-Learning',
                       pathways=[(z, pnl.BackPropagation)],
                       targets=[0, 0, 1],
                       learning_rate=2.0,
                       prefs={pnl.VERBOSE_PREF: False,
                              pnl.REPORT_OUTPUT_PREF: True}
)

# Log Middle_Weights of MappingProjection to Hidden_Layer_2
# Hidden_Layer_2.set_log_conditions('Middle Weights')
Middle_Weights.set_log_conditions('mod_matrix')

comp.reportOutputPref = True
# Shows graph will full information:
# comp.show_graph(show_dimensions=pnl.ALL)
comp.show_graph(show_learning=pnl.ALL, show_node_structure=True)
# comp.show_graph(show_learning=pnl.ALL, show_processes=True)
# comp.show_graph(show_learning=pnl.ALL, show_dimensions=pnl.ALL, show_mechanism_structure=True)
# Shows minimal graph:
# comp.show_graph()


stim_list = {Input_Layer: ['red']}
target_list = {Output_Layer: [[0, 0, 1]]}

comp.learn(
    num_trials=1,
    inputs=stim_list,
    targets=target_list,
    clamp_input=pnl.SOFT_CLAMP,
    call_before_trial=functools.partial(print_header, comp),
    call_after_trial=functools.partial(show_target, comp),
    termination_processing={pnl.TimeScale.TRIAL: pnl.AfterNCalls(Output_Layer, 1)},
    animate={'show_learning':pnl.ALL, 'unit':pnl.EXECUTION_SET, pnl.SAVE_IMAGES:True},
)

# Print out logged weights for Middle_Weights
# print('\nMiddle Weights (to Hidden_Layer_2): \n', Hidden_Layer_2.log.nparray(entries='Middle Weights', header=False))
print('\nMiddle Weights (to Hidden_Layer_2): \n', Middle_Weights.log.nparray(entries='mod_matrix', header=False))
