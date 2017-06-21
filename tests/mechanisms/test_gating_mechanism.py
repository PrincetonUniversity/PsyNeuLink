import numpy as np
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection

from PsyNeuLink.Components.Functions.Function import ConstantIntegrator, Logistic
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanisms.GatingMechanism import GatingMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import FUNCTION, FUNCTION_PARAMS, INITIALIZER, LEARNING, RATE, SOFT_CLAMP, VALUE
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Globals.TimeScale import CentralClock


def test_gating():
    Input_Layer = TransferMechanism(
        name='Input Layer',
        function=Logistic,
        default_input_value=np.zeros((2,))
    )

    Hidden_Layer_1 = TransferMechanism(
        name='Hidden Layer_1',
        function=Logistic(),
        default_input_value=np.zeros((5,))
    )

    Hidden_Layer_2 = TransferMechanism(
        name='Hidden Layer_2',
        function=Logistic(),
        default_input_value=[0, 0, 0, 0]
    )

    Output_Layer = TransferMechanism(
        name='Output Layer',
        function=Logistic,
        default_input_value=[0, 0, 0]
    )

    Gating_Mechanism = GatingMechanism(
        default_gating_policy=0.0,
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
    Input_Weights = MappingProjection(
        name='Input Weights',
        matrix=Input_Weights_matrix
    )

    # This projection will be used by the process below by assigning its sender and receiver args
    #    to mechanismss in the pathway
    Middle_Weights = MappingProjection(
        name='Middle Weights',
        sender=Hidden_Layer_1,
        receiver=Hidden_Layer_2,
        matrix={
            VALUE: Middle_Weights_matrix,
            FUNCTION: ConstantIntegrator,
            FUNCTION_PARAMS: {
                INITIALIZER: Middle_Weights_matrix,
                RATE: Middle_Weights_matrix
            },
        }
    )

    Output_Weights = MappingProjection(
        name='Output Weights',
        sender=Hidden_Layer_2,
        receiver=Output_Layer,
        matrix=Output_Weights_matrix
    )

    z = process(
        default_input_value=[0, 0],
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
        ],
        clamp_input=SOFT_CLAMP,
        learning=LEARNING,
        learning_rate=1.0,
        target=[0, 0, 1],
        prefs={
            VERBOSE_PREF: False,
            REPORT_OUTPUT_PREF: True
        }
    )

    g = process(
        default_input_value=[1.0],
        pathway=[Gating_Mechanism]
    )

    stim_list = {
        Input_Layer: [[-1, 30]],
        Gating_Mechanism: [1.0]
    }
    target_list = {
        Output_Layer: [[0, 0, 1]]
    }

    def print_header():
        print("\n\n**** TRIAL: ", CentralClock.trial)

    def show_target():
        i = s.input
        t = s.targetInputStates[0].value
        print('\nOLD WEIGHTS: \n')
        print('- Input Weights: \n', Input_Weights.matrix)
        print('- Middle Weights: \n', Middle_Weights.matrix)
        print('- Output Weights: \n', Output_Weights.matrix)
        print('\nSTIMULI:\n\n- Input: {}\n- Target: {}\n'.format(i, t))
        print('ACTIVITY FROM OLD WEIGHTS: \n')
        print('- Middle 1: \n', Hidden_Layer_1.value)
        print('- Middle 2: \n', Hidden_Layer_2.value)
        print('- Output:\n', Output_Layer.value)

    s = system(
        processes=[z, g],
        targets=[0, 0, 1],
        learning_rate=1.0
    )

    s.reportOutputPref = True
    # s.show_graph(show_learning=True)

    results = s.run(
        num_executions=10,
        inputs=stim_list,
        targets=target_list,
        call_before_trial=print_header,
        call_after_trial=show_target,
    )
