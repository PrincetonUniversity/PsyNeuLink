from PsyNeuLink.Components.Functions.Function import Linear, Logistic
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.System import *
from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Projections.LearningProjection import TARGET_MSE
import numpy as np


class TestStroop:
    def test_stroop_model(self):
        process_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False
        }

        system_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False
        }

        colors = TransferMechanism(
            default_input_value=[0,0],
            function=Linear,
            name="Colors",
        )

        words = TransferMechanism(
            default_input_value=[0,0],
            function=Linear,
            name="Words",
        )

        response = TransferMechanism(
            default_input_value=[0,0],
            function=Logistic,
            name="Response",
        )

        color_naming_process = process(
            default_input_value=[1, 2.5],
            pathway=[colors, FULL_CONNECTIVITY_MATRIX, response],
            learning=LEARNING_PROJECTION,
            target=[0, 1],
            name='Color Naming',
            prefs=process_prefs,
        )

        word_reading_process = process(
            default_input_value=[.5, 3],
            pathway=[words, FULL_CONNECTIVITY_MATRIX, response],
            name='Word Reading',
            learning=LEARNING_PROJECTION,
            target=[1, 0],
            prefs=process_prefs,
        )

        s = system(
            processes=[color_naming_process, word_reading_process],
            name='Stroop Model',
            targets=[0,0],
            prefs=system_prefs,
        )

        # stim_dict = {
        #     colors: [
        #         [1,0],
        #         [0,1]
        #     ],
        #     words: [
        #         [0,1],
        #         [1,0]
        #     ]
        # }
        # target_dict = {
        #     response: [
        #         [1,0],
        #         [0,1]
        #     ]
        # }

        # results = s.run(
        #     num_executions=10,
        #     inputs=stim_dict,
        #     targets=target_dict,
        # )
        expected_color_results = [
            np.array([ 0.88079708,  0.88079708]),
            np.array([ 0.85997037,  0.88340023]),
            np.array([ 0.83312329,  0.88585176]),
            np.array([ 0.79839127,  0.88816536]),
            np.array([ 0.75384913,  0.89035312]),
            np.array([ 0.69835531,  0.89242571]),
            np.array([ 0.63303376,  0.89439259]),
            np.array([ 0.56245802,  0.8962622 ]),
            np.array([ 0.49357614,  0.89804208]),
            np.array([ 0.43230715,  0.89973899]),
        ]

        expected_word_results = [
            np.array([ 0.88079708,  0.88079708]),
            np.array([ 0.88340023,  0.85997037]),
            np.array([ 0.88585176,  0.83312329]),
            np.array([ 0.88816536,  0.79839127]),
            np.array([ 0.89035312,  0.75384913]),
            np.array([ 0.89242571,  0.69835531]),
            np.array([ 0.89439259,  0.63303376]),
            np.array([ 0.8962622 ,  0.56245802]),
            np.array([ 0.89804208,  0.49357614]),
            np.array([ 0.89973899,  0.43230715]),
        ]

        for i in range(10):
            cr = color_naming_process.execute(input=[1, 1], target=[0,1])
            wr = word_reading_process.execute(input=[1, 1], target=[1,0])

            np.testing.assert_allclose(cr, expected_color_results[i], atol=1e-08, err_msg='Failed on expected_color_results[{0}]'.format(i))
            np.testing.assert_allclose(wr, expected_word_results[i], atol=1e-08, err_msg='Failed on expected_word_results[{0}]'.format(i))

    def test_stroop_model_learning(self):
        process_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False,
        }
        system_prefs = {
            REPORT_OUTPUT_PREF: True,
            VERBOSE_PREF: False,
        }

        colors = TransferMechanism(
            default_input_value=[0,0],
            function=Linear,
            name="Colors",
        )
        words = TransferMechanism(
            default_input_value=[0,0],
            function=Linear,
            name="Words",
        )
        hidden = TransferMechanism(
            default_input_value=[0,0],
            function=Logistic,
            name="Hidden",
        )
        response = TransferMechanism(
            default_input_value=[0,0],
            function=Logistic(),
            name="Response",
        )
        output = TransferMechanism(
            default_input_value=[0,0],
            function=Logistic,
            name="Output",
        )

        CH_Weights_matrix = np.arange(4).reshape((2,2))
        WH_Weights_matrix = np.arange(4).reshape((2,2))
        HO_Weights_matrix = np.arange(4).reshape((2,2))

        CH_Weights = MappingProjection(
            name='Color-Hidden Weights',
            matrix=CH_Weights_matrix,
        )
        WH_Weights = MappingProjection(
            name='Word-Hidden Weights',
            matrix=WH_Weights_matrix,
        )
        HO_Weights = MappingProjection(
            name='Hidden-Output Weights',
            matrix=HO_Weights_matrix,
        )

        color_naming_process = process(
            default_input_value=[1, 2.5],
            pathway=[colors, CH_Weights, hidden, HO_Weights, response],
            learning=LEARNING,
            target=[2,2],
            name='Color Naming',
            prefs=process_prefs,
        )

        word_reading_process = process(
            default_input_value=[.5, 3],
            pathway=[words, WH_Weights, hidden],
            name='Word Reading',
            learning=LEARNING,
            target=[3,3],
            prefs=process_prefs,
        )

        s = system(
            processes=[color_naming_process, word_reading_process],
            targets=[20,20],
            name='Stroop Model',
            prefs=system_prefs,
        )

        def print_header():
            print("\n\n**** TRIAL: ", CentralClock.trial)

        def show_target():
            print ('\nColor Naming\n\tInput: {}\n\tTarget: {}'.
                   # format(color_naming_process.input, color_naming_process.target))
                   format(colors.inputValue, s.targets))
            print ('Wording Reading:\n\tInput: {}\n\tTarget: {}\n'.
                   # format(word_reading_process.input, word_reading_process.target))
                   format(words.inputValue, s.targets))
            print ('Response: \n', response.outputValue[0])
            print ('Hidden-Output:')
            print (HO_Weights.matrix)
            print ('Color-Hidden:')
            print (CH_Weights.matrix)
            print ('Word-Hidden:')
            print (WH_Weights.matrix)


        stim_list_dict = {
            colors:[[1, 1]],
            words:[[-2, -2]]
        }

        target_list_dict = {response:[[1, 1]]}

        s.run(
            num_executions=2,
            inputs=stim_list_dict,
            targets=target_list_dict,
            call_before_trial=print_header,
            call_after_trial=show_target,
        )
        # import code
        # code.interact(local=locals())


        objective_response = s.mechanisms[3]
        objective_hidden = s.mechanisms[7]
        expected_output = [
            (colors.outputState.value, np.array([1., 1.])),
            (words.outputState.value, np.array([-2., -2.])),
            (hidden.outputState.value, np.array([0.13227553,  0.01990677])),
            (response.outputState.value, np.array([0.51044657,  0.5483048])),
            (objective_response.outputState.value, np.array([0.48955343, 0.4516952])),
            (objective_response.outputStates[TARGET_MSE].value, np.array(0.22184555903789838)),
            (objective_hidden.outputState.value, np.array([0., 0.])),
            (CH_Weights.matrix, np.array([
                [0.01190129, 1.0103412],
                [2.01190129, 3.0103412]
            ])),
            (WH_Weights.matrix, np.array([
                [-0.02380258, 0.9793176],
                [1.97619742, 2.9793176]
            ])),
            (HO_Weights.matrix, np.array([
                [0.01462766, 1.01351195],
                [2.00220713, 3.00203878]
            ])),
        ]

        for i in range(len(expected_output)):
            val, expected = expected_output[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))
