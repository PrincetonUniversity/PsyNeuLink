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

    Middle_Weights.log_items('matrix')

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

    # log_val = Middle_Weights.log.nparray(entries='matrix', header=False)
    # expected_log_val = np.array(
    #         [
    #             [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]],
    #             [ [[ 0.05,  0.1 ,  0.15,  0.2 ],
    #                [ 0.25,  0.3 ,  0.35,  0.4 ],
    #                [ 0.45,  0.5 ,  0.55,  0.6 ],
    #                [ 0.65,  0.7 ,  0.75,  0.8 ],
    #                [ 0.85,  0.9 ,  0.95,  1.  ]],
    #               [[ 0.04789907,  0.09413833,  0.14134241,  0.18938924],
    #                [ 0.24780811,  0.29388455,  0.34096758,  0.38892985],
    #                [ 0.44772121,  0.49364209,  0.54060947,  0.58849095],
    #                [ 0.64763875,  0.69341202,  0.74026967,  0.78807449],
    #                [ 0.84756101,  0.89319513,  0.93994932,  0.98768187]],
    #               [[ 0.04738148,  0.08891106,  0.13248753,  0.177898  ],
    #                [ 0.24726841,  0.28843403,  0.33173452,  0.37694783],
    #                [ 0.44716034,  0.48797777,  0.53101423,  0.57603893],
    #                [ 0.64705774,  0.6875443 ,  0.73032986,  0.77517531],
    #                [ 0.84696096,  0.88713512,  0.92968378,  0.97435998]],
    #               [[ 0.04937771,  0.08530344,  0.12439361,  0.16640433],
    #                [ 0.24934878,  0.28467436,  0.32329947,  0.36496974],
    #                [ 0.44932147,  0.48407216,  0.52225175,  0.56359587],
    #                [ 0.64929589,  0.68349948,  0.72125508,  0.76228876],
    #                [ 0.84927212,  0.88295836,  0.92031297,  0.96105307]],
    #               [[ 0.05440291,  0.08430585,  0.1183739 ,  0.15641064],
    #                [ 0.25458348,  0.28363519,  0.3170288 ,  0.35455942],
    #                [ 0.45475764,  0.48299299,  0.51573974,  0.55278488],
    #                [ 0.65492462,  0.68238209,  0.7145124 ,  0.75109483],
    #                [ 0.85508376,  0.88180465,  0.91335119,  0.94949538]],
    #               [[ 0.06177218,  0.0860581 ,  0.11525064,  0.14926369],
    #                [ 0.26225812,  0.28546004,  0.31377611,  0.34711631],
    #                [ 0.46272625,  0.48488774,  0.51236246,  0.54505667],
    #                [ 0.66317453,  0.68434373,  0.7110159 ,  0.74309381],
    #                [ 0.86360121,  0.88382991,  0.9097413 ,  0.94123489]],
    #               [[ 0.06989398,  0.08959148,  0.11465594,  0.14513241],
    #                [ 0.27071639,  0.2891398 ,  0.31315677,  0.34281389],
    #                [ 0.47150846,  0.48870843,  0.5117194 ,  0.54058946],
    #                [ 0.67226675,  0.68829929,  0.71035014,  0.73846891],
    #                [ 0.87298831,  0.88791376,  0.90905395,  0.93646   ]],
    #               [[ 0.07750784,  0.09371987,  0.11555569,  0.143181  ],
    #                [ 0.27864693,  0.29343991,  0.31409396,  0.3407813 ],
    #                [ 0.47974374,  0.49317377,  0.5126926 ,  0.53847878],
    #                [ 0.68079346,  0.69292265,  0.71135777,  0.73628353],
    #                [ 0.88179203,  0.89268732,  0.91009431,  0.93420362]],
    #               [[ 0.0841765 ,  0.09776672,  0.11711835,  0.14249779],
    #                [ 0.28559463,  0.29765609,  0.31572199,  0.34006951],
    #                [ 0.48695967,  0.49755273,  0.51438349,  0.5377395 ],
    #                [ 0.68826567,  0.69745713,  0.71310872,  0.735518  ],
    #                [ 0.88950757,  0.89736946,  0.91190228,  0.93341316]],
    #               [[ 0.08992499,  0.10150104,  0.11891032,  0.14250149],
    #                [ 0.29158517,  0.30154765,  0.31758943,  0.34007336],
    #                [ 0.49318268,  0.50159531,  0.51632339,  0.5377435 ],
    #                [ 0.69471052,  0.70164382,  0.71511777,  0.73552215],
    #                [ 0.8961628 ,  0.90169281,  0.91397691,  0.93341744]]]
    #         ]
    # )
    # np.testing.assert_allclose(log_val, expected_log_val,
    #                            atol=1e-08,
    #                            err_msg='Failed on test of logged values')
