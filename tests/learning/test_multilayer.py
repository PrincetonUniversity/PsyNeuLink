import numpy as np

from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.PathwayProjections.MappingProjection import MappingProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import LEARNING, SOFT_CLAMP
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.ObjectiveMechanisms import MSE
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import TransferMechanism


def test_multilayer():
    Input_Layer = TransferMechanism(
        name='Input Layer',
        function=Logistic,
        default_variable=np.zeros((2,)),
    )

    Hidden_Layer_1 = TransferMechanism(
        name='Hidden Layer_1',
        function=Logistic(),
        # default_variable=np.zeros((5,)),
        size=5
    )

    Hidden_Layer_2 = TransferMechanism(
        name='Hidden Layer_2',
        function=Logistic(),
        default_variable=[0, 0, 0, 0],
    )

    Output_Layer = TransferMechanism(
        name='Output Layer',
        function=Logistic,
        default_variable=[0, 0, 0],
    )

    Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
    Middle_Weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
    Output_Weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

    # TEST PROCESS.LEARNING WITH:
    # CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
    # INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
    # NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)

    # This projection will be used by the process below by referencing it in the process' pathway;
    #    note: sender and receiver args don't need to be specified
    Input_Weights = MappingProjection(
        name='Input Weights',
        matrix=Input_Weights_matrix,
    )

    # This projection will be used by the process below by assigning its sender and receiver args
    #    to mechanismss in the pathway
    Middle_Weights = MappingProjection(
        name='Middle Weights',
        sender=Hidden_Layer_1,
        receiver=Hidden_Layer_2,
        matrix=Middle_Weights_matrix,
    )

    # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
    Output_Weights = MappingProjection(
        name='Output Weights',
        sender=Hidden_Layer_2,
        receiver=Output_Layer,
        matrix=Output_Weights_matrix,
    )

    p = process(
        # default_variable=[0, 0],
        size=2,
        pathway=[
            Input_Layer,
            # The following reference to Input_Weights is needed to use it in the pathway
            #    since it's sender and receiver args are not specified in its declaration above
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
            REPORT_OUTPUT_PREF: False
        },
    )

    stim_list = {Input_Layer: [[-1, 30]]}
    target_list = {Output_Layer: [[0, 0, 1]]}

    def show_target():
        i = s.input
        t = s.target_input_states[0].value
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
        processes=[p],
        targets=[0, 0, 1],
        learning_rate=1.0,
    )

    s.reportOutputPref = True

    results = s.run(
        num_trials=10,
        inputs=stim_list,
        targets=target_list,
        call_after_trial=show_target,
    )

    objective_output_layer = s.mechanisms[4]

    results_list = []
    for elem in s.results:
        for nested_elem in elem:
            nested_elem = nested_elem.tolist()
            try:
                iter(nested_elem)
            except TypeError:
                nested_elem = [nested_elem]
            results_list.extend(nested_elem)

    expected_output = [
        (Output_Layer.output_states.values, [np.array([0.22686074, 0.25270212, 0.91542149])]),
        (objective_output_layer.output_states[MSE].value, np.array(0.04082589331852094)),
        (Input_Weights.matrix, np.array([
            [ 0.09900247, 0.19839653, 0.29785764, 0.39739191, 0.49700232],
            [ 0.59629092, 0.69403786, 0.79203411, 0.89030237, 0.98885379],
        ])),
        (Middle_Weights.matrix, np.array([
            [ 0.09490249, 0.10488719, 0.12074013, 0.1428774 ],
            [ 0.29677354, 0.30507726, 0.31949676, 0.3404652 ],
            [ 0.49857336, 0.50526254, 0.51830509, 0.53815062],
            [ 0.70029406, 0.70544225, 0.71717037, 0.73594383],
            [ 0.90192903, 0.90561554, 0.91609668, 0.93385292],
        ])),
        (Output_Weights.matrix, np.array([
            [-0.74447522, -0.71016859, 0.31575293],
            [-0.50885177, -0.47444784, 0.56676582],
            [-0.27333719, -0.23912033, 0.8178167 ],
            [-0.03767547, -0.00389039, 1.06888608],
        ])),
        (results, [
            [np.array([0.8344837 , 0.87072018, 0.89997433])],
            [np.array([0.77970193, 0.83263138, 0.90159627])],
            [np.array([0.70218502, 0.7773823 , 0.90307765])],
            [np.array([0.60279149, 0.69958079, 0.90453143])],
            [np.array([0.4967927 , 0.60030321, 0.90610082])],
            [np.array([0.4056202 , 0.49472391, 0.90786617])],
            [np.array([0.33763025, 0.40397637, 0.90977675])],
            [np.array([0.28892812, 0.33633532, 0.9117193 ])],
            [np.array([0.25348771, 0.28791896, 0.9136125 ])],
            [np.array([0.22686074, 0.25270212, 0.91542149])]
        ]),
    ]

    for i in range(len(expected_output)):
        val, expected = expected_output[i]
        # setting absolute tolerance to be in accordance with reference_output precision
        # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
        # which WILL FAIL unless you gather higher precision values to use as reference
        np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
