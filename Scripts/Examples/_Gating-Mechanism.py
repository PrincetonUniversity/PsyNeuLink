import functools
import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

Input_Layer = pnl.TransferMechanism(
    name='Input Layer',
    default_variable=np.zeros((2,)),
    function=psyneulink.core.components.functions.transferfunctions.Logistic
)

Hidden_Layer_1 = pnl.TransferMechanism(
    name='Hidden Layer_1',
    default_variable=np.zeros((5,)),
    function=psyneulink.core.components.functions.transferfunctions.Logistic()
)

Hidden_Layer_2 = pnl.TransferMechanism(
    name='Hidden Layer_2',
    default_variable=[0, 0, 0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Logistic()
)

Output_Layer = pnl.TransferMechanism(
    name='Output Layer',
    default_variable=[0, 0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Logistic
)

Gating_Mechanism = pnl.GatingMechanism(
    # default_gating_allocation=0.0,
    size=[1],
    gating_signals=[
        Hidden_Layer_1,
        Hidden_Layer_2,
        Output_Layer,
    ]
)

Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

# TEST PROCESS.LEARNING WITH:
# CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
# INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
# NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and
# Output_Weights)

# This projection will be used by the process below by referencing it in the process' pathway;
#    note: sender and receiver args don't need to be specified
Input_Weights = pnl.MappingProjection(
    name='Input Weights',
    matrix=Input_Weights_matrix
)

# This projection will be used by the process below by assigning its sender and receiver args
#    to mechanismss in the pathway
Middle_Weights = pnl.MappingProjection(
    name='Middle Weights',
    sender=Hidden_Layer_1,
    receiver=Hidden_Layer_2,
    matrix={
        pnl.VALUE: Middle_Weights_matrix,
        pnl.FUNCTION: psyneulink.core.components.functions.statefulfunctions.integratorfunctions.AccumulatorIntegrator,
        pnl.FUNCTION_PARAMS: {
            pnl.INITIALIZER: Middle_Weights_matrix,
            pnl.RATE: Middle_Weights_matrix
        },
    }
)

Output_Weights = pnl.MappingProjection(
    name='Output Weights',
    sender=Hidden_Layer_2,
    receiver=Output_Layer,
    matrix=Output_Weights_matrix
)

z = pnl.Pathway(
    # default_variable=[0, 0],
    pathway=[
        Input_Layer,
        # The following reference to Input_Weights is needed to use it in the pathway
        # since it's sender and receiver args are not specified in its
        # declaration above
        Input_Weights,
        Hidden_Layer_1,
        # No projection specification is needed here since the sender arg for Middle_Weights
        #    is Hidden_Layer_1 and its receiver arg is Hidden_Layer_2
        # Middle_Weights,
        Hidden_Layer_2,
        # Output_Weights does not need to be listed for the same reason as Middle_Weights
        # If Middle_Weights and/or Output_Weights is not declared above, then the process
        #    will assign a default for missing projection
        # Output_Weights,
        Output_Layer
    ]
)

g = pnl.Pathway(
    pathway=[Gating_Mechanism]
)

stim_list = {
    Input_Layer: [[-1, 30]],
    Gating_Mechanism: [1.0]
}
target_list = {
    Output_Layer: [[0, 0, 1]]
}


def print_header(system):
    print("\n\n**** Time: ", system.scheduler.clock.simple_time)


def show_target():
    i = comp.input
    t = comp.target_input_ports[0].value
    print('\nOLD WEIGHTS: \n')
    print('- Input Weights: \n', Input_Weights.matrix)
    print('- Middle Weights: \n', Middle_Weights.matrix)
    print('- Output Weights: \n', Output_Weights.matrix)
    print('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
    print('ACTIVITY FROM OLD WEIGHTS: \n')
    print('- Middle 1: \n', Hidden_Layer_1.value)
    print('- Middle 2: \n', Hidden_Layer_2.value)
    print('- Output:\n', Output_Layer.value)


comp = pnl.Composition(
    pathways=[z, g],
    targets=[0, 0, 1],
    learning_rate=1.0
)

comp.reportOutputPref = True
# mySystem.show_graph(show_learning=True)

results = comp.learn(
    num_trials=10,
    inputs=stim_list,
    targets=target_list,
    clamp_input=pnl.SOFT_CLAMP,
    learning_rate=1.0,
    prefs={
        pnl.VERBOSE_PREF: False,
        pnl.REPORT_OUTPUT_PREF: True
    },
    call_before_trial=functools.partial(print_header, comp),
    call_after_trial=show_target,
)
