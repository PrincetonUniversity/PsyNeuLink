import functools
import numpy as np
import psyneulink as pnl

Input_Layer = pnl.TransferMechanism(
    name='Input Layer',
    function=pnl.Logistic,
    default_variable=np.zeros((2,)))

Hidden_Layer_1 = pnl.TransferMechanism(
    name='Hidden Layer_1',
    function=pnl.Logistic(),
    default_variable=np.zeros((5,)))

Hidden_Layer_2 = pnl.TransferMechanism(
    name='Hidden Layer_2',
    function=pnl.Logistic(),
    default_variable=[0, 0, 0, 0])

Output_Layer = pnl.TransferMechanism(
    name='Output Layer',
    function=pnl.Logistic,
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

z = pnl.Process(
    default_variable=[0, 0],
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
    ],
    clamp_input=pnl.SOFT_CLAMP,
    learning=pnl.LEARNING,
    target=[0, 0, 1],
    prefs={
        pnl.VERBOSE_PREF: False,
        pnl.REPORT_OUTPUT_PREF: True
    }
)


def print_header(system):
    print("\n\n**** TRIAL: ", system.scheduler_processing.times[pnl.TimeScale.RUN][pnl.TimeScale.TRIAL])


def show_target(system):
    i = system.input
    t = system.target_input_states[0].value
    print('\nOLD WEIGHTS: \n')
    print('- Input Weights: \n', Input_Weights.matrix)
    print('- Middle Weights: \n', Middle_Weights.matrix)
    print('- Output Weights: \n', Output_Weights.matrix)

    print('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
    print('ACTIVITY FROM OLD WEIGHTS: \n')
    print('- Middle 1: \n', Hidden_Layer_1.value)
    print('- Middle 2: \n', Hidden_Layer_2.value)
    print('- Output:\n', Output_Layer.value)


mySystem = pnl.System(
    processes=[z],
    targets=[0, 0, 1],
    learning_rate=2.0
)

mySystem.reportOutputPref = True
# Shows graph will full information:
# mySystem.show_graph(show_learning=pnl.ALL, show_dimensions=pnl.ALL)
# Shows minimal graph:
mySystem.show_graph()

stim_list = {Input_Layer: [[-1, 30]]}
target_list = {Output_Layer: [[0, 0, 1]]}

mySystem.run(
    num_trials=10,
    inputs=stim_list,
    targets=target_list,
    call_before_trial=functools.partial(print_header, mySystem),
    call_after_trial=functools.partial(show_target, mySystem),
    termination_processing={pnl.TimeScale.TRIAL: pnl.AfterNCalls(Output_Layer, 1)}
)

