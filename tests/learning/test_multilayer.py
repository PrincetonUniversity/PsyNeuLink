import numpy as np

from psyneulink.components.functions.function import Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import LEARNING, SOFT_CLAMP
from psyneulink.globals.preferences.componentpreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import MSE


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

    p = Process(
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

    Hidden_Layer_2.log_items('Middle Weights')

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

    s = System(
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

    assert Hidden_Layer_2.log.nparray(entries='Middle Weights') == \
           np.array([[['Entry'] [0] [1] [2] [3] [4] [5] [6] [7] [8] [9]],
                     ['Middle Weights',
                      [[ 0.05,  0.1 ,  0.15,  0.2 ],
                       [ 0.25,  0.3 ,  0.35,  0.4 ],
                       [ 0.45,  0.5 ,  0.55,  0.6 ],
                       [ 0.65,  0.7 ,  0.75,  0.8 ],
                       [ 0.85,  0.9 ,  0.95,  1.  ]],
                      [[ 0.04579814,  0.08827665,  0.13268481,  0.17877847],
                       [ 0.24561623,  0.28776909,  0.33193516,  0.3778597 ],
                       [ 0.44544242,  0.48728418,  0.53121895,  0.5769819 ],
                       [ 0.6452775 ,  0.68682405,  0.73053934,  0.77614898],
                       [ 0.84512203,  0.88639026,  0.92989864,  0.97536374]],
                      [[ 0.04901996,  0.08046388,  0.11609331,  0.15558256],
                       [ 0.24897375,  0.27962724,  0.3146448 ,  0.35368673],
                       [ 0.44893023,  0.4788264 ,  0.51325768,  0.55187098],
                       [ 0.64888958,  0.6780649 ,  0.71193809,  0.75014332],
                       [ 0.84885192,  0.87734543,  0.9106907 ,  0.94850989]],
                      [[ 0.06316692,  0.08303935,  0.10870443,  0.14005193],
                       [ 0.26370305,  0.28230872,  0.30695176,  0.33751681],
                       [ 0.46422077,  0.48161006,  0.50527151,  0.53508491],
                       [ 0.66471785,  0.68094646,  0.70367107,  0.73276693],
                       [ 0.86519225,  0.88032021,  0.90215622,  0.93057134]],
                      [[ 0.07856911,  0.09139869,  0.11052933,  0.13611383],
                       [ 0.27973756,  0.29101125,  0.30885158,  0.33341703],
                       [ 0.48086506,  0.49064354,  0.50724358,  0.53082922],
                       [ 0.68194671,  0.6902972 ,  0.7057124 ,  0.72836178],
                       [ 0.88297817,  0.88997329,  0.90426356,  0.92602376]],
                      [[ 0.08977667,  0.0988261 ,  0.11428057,  0.13646287],
                       [ 0.29141092,  0.29874734,  0.31275872,  0.33378057],
                       [ 0.49298675,  0.49867675,  0.51130078,  0.53120673],
                       [ 0.6944973 ,  0.69861466,  0.70991316,  0.72875264],
                       [ 0.89593658,  0.898561  ,  0.90860082,  0.92642732]],
                      [[ 0.09811548,  0.10480778,  0.11790552,  0.13784395],
                       [ 0.30010061,  0.30498073,  0.31653621,  0.33521977],
                       [ 0.50201352,  0.50515192,  0.5152248 ,  0.53270175],
                       [ 0.70384584,  0.70532065,  0.71397705,  0.73030096],
                       [ 0.90559035,  0.90548595,  0.9127974 ,  0.92802619]],
                      [[ 0.10469417,  0.10974533,  0.12117048,  0.13948495],
                       [ 0.30695901,  0.3101282 ,  0.31993998,  0.33693054],
                       [ 0.50914015,  0.51050072,  0.51876169,  0.53447943],
                       [ 0.71122806,  0.71086127,  0.71764079,  0.73214239],
                       [ 0.91321454,  0.91120818,  0.91658122,  0.92992798]],
                      [[ 0.11011252,  0.11393871,  0.12409692,  0.14116272],
                       [ 0.31260971,  0.31450141,  0.32299192,  0.33868025],
                       [ 0.51501338,  0.51504615,  0.52193381,  0.53629805],
                       [ 0.71731298,  0.71557053,  0.72092725,  0.73402656],
                       [ 0.91949955,  0.91607229,  0.91997575,  0.93187411]],
                      [[ 0.11471403,  0.11758198,  0.12673709,  0.14280208],
                       [ 0.31741   ,  0.31830207,  0.32574614,  0.34039044],
                       [ 0.52000382,  0.51899735,  0.52479713,  0.53807598],
                       [ 0.72248405,  0.71966475,  0.72389421,  0.73586884],
                       [ 0.92484112,  0.92030151,  0.92304054,  0.93377713]]]])
