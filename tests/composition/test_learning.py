import numpy as np
import pytest
import psyneulink as pnl

# np.set_printoptions(suppress=True)

import numpy as np
import pytest

from psyneulink.core.compositions.composition import Composition, CompositionError, RunError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.functions.learningfunctions import BackPropagation


class TestTargetSpecs:

    def test_target_spec_default_assignment(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp1 = Composition()
        p1 = comp1.add_backpropagation_learning_pathway(pathway=[A,B])
        # Call learn with default_variable specified for target (for comparison with missing target)
        comp1.learn(inputs={A: 1.0,
                            p1.target: 0.0},
                 num_trials=2)
        assert np.allclose(comp1.results, [[[1.]], [[0.9]]])

        # Repeat with no target assignment (should use default_variable)
        C = TransferMechanism(name="learning-process-mech-C")
        D = TransferMechanism(name="learning-process-mech-D")
        comp2 = Composition()
        comp2.add_backpropagation_learning_pathway(pathway=[C,D])
        # Call learn with no target specification
        comp2.learn(inputs={C: 1.0},
                   num_trials=2)
        # Should be same with default target specification
        assert np.allclose(comp2.results, comp1.results)

    def test_target_dict_spec_single_trial_scalar_and_lists_rl(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        p = comp.add_reinforcement_learning_pathway(pathway=[A,B])
        # Confirm that targets are ignored in run (vs learn)
        comp.run(inputs={A: 1.0,
                         p.target:2.0})
        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.]]])
        comp.learn(inputs={A: 1.0,
                           p.target:2.0})
        comp.learn(inputs={A: 1.0,
                           p.target:[2.0]})
        comp.learn(inputs={A: 1.0,
                           p.target:[[2.0]]})

        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.05]], [[1.0975]]])

    def test_target_dict_spec_single_trial_scalar_and_lists_bp(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        # Confirm that targets are ignored in run (vs learn)
        comp.run(inputs={A: 1.0,
                         p.target:2.0})
        assert np.allclose(comp.results, [[[1.]]])
        comp.learn(inputs={A: 1.0,
                           p.target:2.0})
        comp.learn(inputs={A: 1.0,
                           p.target:[2.0]})
        comp.learn(inputs={A: 1.0,
                           p.target:[[2.0]]})

        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.21]], [[1.40873161]]])

    def test_target_dict_spec_multi_trial_lists_rl(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B])
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                           p.target: [[4.0], [5.0], [6.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                           p.target: [[[4.0]], [[5.0]], [[6.0]]]})
        assert np.allclose(comp.results,
                           [[[1.]], [[2.6]], [[5.34]],
                            [[1.978]], [[4.3604]], [[6.92436]]])

    def test_target_dict_spec_multi_trial_lists_bp(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C",
                              default_variable=[[0.0, 0.0]])
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        comp.learn(inputs={A: 1.0,
                           p.target:[2.0, 3.0]})
        comp.learn(inputs={A: 1.0,
                           p.target:[[2.0, 3.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                           p.target: [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                           p.target: [[[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]})
        assert np.allclose(comp.results,
                           [[[1., 1.]],
                            [[1.2075, 1.265]],
                            [[1.41003122, 1.54413183]], [[3.64504691, 4.13165454]], [[8.1607109 , 9.54419477]],
                            [[1.40021212, 1.56636511]], [[3.61629564, 4.17586792]], [[8.11241026, 9.57222535]]])

    # DS: The following test fails the assert. The same value is returned whether a dict or function is used as input,
    # which is not the same as the expected values. Are the expected values incorrect? If not, there is a problem
    # at a deeper level than just the input handling. 5/18/2020
    #
    # def test_function_target_spec(self):
    #
    #     from psyneulink.core.compositions.composition import Composition
    #     A = pnl.TransferMechanism(name="learning-process-mech-A")
    #     B = pnl.TransferMechanism(name="learning-process-mech-B",
    #                           default_variable=np.array([[0.0, 0.0]]))
    #     comp = Composition()
    #     learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B], learning_rate=0.05)
    #     target = learning_pathway.target
    #     # global x
    #     # x = 1
    #
    #     # def input_function(a,b):
    #     #     global x
    #     #     x = x + 1
    #     #     y = 2 * x
    #     #     z = 3 * x
    #     #     target_value = {A:[x], target:[y,z]}
    #     #     print('trial')
    #     #     return target_value
    #     def input_function(trial):
    #         x = trial + 1
    #         y = 2 * x
    #         z = y + 2
    #         target_value = {A:[x], target:[y,z]}
    #         print(target_value)
    #         return target_value
    #
    #     target.log.set_log_conditions('variable')
    #
    #     comp.learn(inputs=input_function, num_trials=3)
    #     assert np.allclose(comp.results, [[[2., 2.]], [[2.4, 2.8]], [[2.72, 3.44]]])

    def test_dict_target_spec_converging_pathways(self):
        A = TransferMechanism(name="converging-learning-pathways-mech-A")
        B = TransferMechanism(name="converging-learning-pathways-mech-B")
        C = TransferMechanism(name="converging-learning-pathways-mech-C", size=2)
        D = TransferMechanism(name="converging-learning-pathways-mech-D")
        E = TransferMechanism(name="converging-learning-pathways-mech-E")
        comp = Composition()
        p1 = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p2 = comp.add_backpropagation_learning_pathway(pathway=[D,E,C])
        assert p1.target == p2.target
        comp.learn(inputs={A: 1.0,
                           D: 2.0,
                           p1.target: [3.0, 4.0]
                           })
        comp.learn(inputs={A: 5.0,
                           D: 6.0,
                           p1.target: [7.0, 8.0]
                           })
        assert np.allclose(comp.results,[[[3., 3.]], [[11.85  , 12.7725]]])

    def test_function_target_spec_converging_pathways(self):
        A = TransferMechanism(name="converging-learning-pathways-mech-A")
        B = TransferMechanism(name="converging-learning-pathways-mech-B")
        C = TransferMechanism(name="converging-learning-pathways-mech-C", size=2)
        D = TransferMechanism(name="converging-learning-pathways-mech-D")
        E = TransferMechanism(name="converging-learning-pathways-mech-E")
        comp = Composition()
        p1 = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p2 = comp.add_backpropagation_learning_pathway(pathway=[D,E,C])
        assert p1.target == p2.target
        inputs = {
            A: [1.0, 5.0],
            D: [2.0, 6.0],
            p1.target: [[3.0, 4.0], [7.0, 8.0]]
        }
        def input_function(trial_num):
            return {
                A: inputs[A][trial_num],
                D: inputs[D][trial_num],
                p1.target: inputs[p1.target][trial_num]
            }
        comp.learn(inputs=input_function,
                   num_trials=2)
        assert np.allclose(comp.results,[[[3., 3.]], [[11.85  , 12.7725]]])

    def test_dict_target_spec_diverging_pathways(self):
        A = TransferMechanism(name="diverging-learning-pathways-mech-A")
        B = TransferMechanism(name="diverging-learning-pathways-mech-B")
        C = TransferMechanism(name="diverging-learning-pathways-mech-C")
        D = TransferMechanism(name="diverging-learning-pathways-mech-D")
        E = TransferMechanism(name="diverging-learning-pathways-mech-E")
        comp = Composition()
        p1 = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p2 = comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
        inputs = {
            A: [1.0, 2.0],
            p1.target: [2.0, 2.0],
            p2.target: [4.0, 4.0]
        }
        def input_function(trial_num):
            return {
                A: inputs[A][trial_num],
                p1.target: inputs[p1.target][trial_num],
                p2.target: inputs[p2.target][trial_num]
            }
        comp.learn(inputs=input_function,
                   num_trials=2)
        assert np.allclose(comp.results,[[[1.], [1.]], [[2.42], [3.38]]])

    def test_function_target_spec_divergin_pathways(self):
        A = TransferMechanism(name="diverging-learning-pathways-mech-A")
        B = TransferMechanism(name="diverging-learning-pathways-mech-B")
        C = TransferMechanism(name="diverging-learning-pathways-mech-C")
        D = TransferMechanism(name="diverging-learning-pathways-mech-D")
        E = TransferMechanism(name="diverging-learning-pathways-mech-E")
        comp = Composition()
        p1 = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p2 = comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
        comp.learn(inputs={A: 1.0,
                         p1.target: 2.0,
                         p2.target: 4.0
                         })
        comp.learn(inputs={A: 2.0,
                         p1.target: 2.0,
                         p2.target: 4.0
                         })
        assert np.allclose(comp.results,[[[1.], [1.]], [[2.42], [3.38]]])

    def test_dict_target_spec_divering_pathways_with_only_one_target(self):
        # First test with both targets (but use default_variale for second for comparison with missing target)
        A = TransferMechanism(name="diverging-learning-pathways-mech-A")
        B = TransferMechanism(name="diverging-learning-pathways-mech-B")
        C = TransferMechanism(name="diverging-learning-pathways-mech-C")
        D = TransferMechanism(name="diverging-learning-pathways-mech-D")
        E = TransferMechanism(name="diverging-learning-pathways-mech-E")
        comp1 = Composition()
        p1 = comp1.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p2 = comp1.add_backpropagation_learning_pathway(pathway=[A,D,E])
        comp1.learn(inputs={A: 1.0,
                            p1.target: 0.0,
                            p2.target: 2.0
                            },
                    num_trials=2)
        assert np.allclose(comp1.results,[[[1.], [1.]], [[0.81], [1.21]]])

        F = TransferMechanism(name="diverging-learning-pathways-mech-F")
        G = TransferMechanism(name="diverging-learning-pathways-mech-G")
        H = TransferMechanism(name="diverging-learning-pathways-mech-H")
        I = TransferMechanism(name="diverging-learning-pathways-mech-I")
        J = TransferMechanism(name="diverging-learning-pathways-mech-J")
        comp2 = Composition()
        p3 = comp2.add_backpropagation_learning_pathway(pathway=[F,G,H])
        p4 = comp2.add_backpropagation_learning_pathway(pathway=[F,I,J])
        # Call learn with missing spec for p3.target;  should use default_variable
        comp2.learn(inputs={F: 1.0,
                            p4.target: 2.0
                            },
                    num_trials=2)
        assert np.allclose(comp2.results, comp1.results)

    def test_target_spec_over_nesting_of_items_in_target_value_error(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C",
                              default_variable=[[0.0, 0.0]])
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        # Elicit error with run
        with pytest.raises(RunError) as error_text:
            comp.run(inputs={A: [1.0, 2.0, 3.0],
                             p.target: [[[3.0], [4.0]], [[5.0], [6.0]], [[7.0], [8.0]]]})
        assert ("Input stimulus" in str(error_text.value) and
                "for Target is incompatible with its external_input_values" in str(error_text.value))
        # Elicit error with learn
        with pytest.raises(RunError) as error_text:
            comp.learn(inputs={A: [1.0, 2.0, 3.0],
                             p.target: [[[3.0], [4.0]], [[5.0], [6.0]], [[7.0], [8.0]]]})
        assert ("Input stimulus" in str(error_text.value) and
                "for Target is incompatible with its external_input_values" in str(error_text.value))

    # The input sizes were picked because the lengths conflict in set:
    # >>> print({10, 2}, {2, 10})
    #     {10, 2} {2, 10}
    # whereas:
    # >>> print({4, 2}, {2, 4})
    #     {2, 4} {2, 4}
    @pytest.mark.parametrize("input_A, target_B", [
        ([[[1.0]], [[2.0]]], [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0], [10.0]]),
        ([[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]], [[7.0]], [[8.0]], [[9.0]], [[10.0]]], [[1.0], [2.0]])],
        ids=["2,10", "10,2"])
    def test_different_number_of_stimuli_for_targets_and_other_input_mech_error(self, input_A, target_B):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B])
        with pytest.raises(CompositionError) as error:
            comp.run(inputs={A: input_A, p.target: target_B})
        error_text = str(error.value)
        assert 'The input dictionary' in error_text
        assert 'contains input specifications of different lengths ({10, 2})' in error_text or \
               'contains input specifications of different lengths ({2, 10})' in error_text
        assert 'The same number of inputs must be provided for each node in a Composition' in error_text


class TestLearningPathwayMethods:
    def test_multiple_of_same_learning_pathway(self):
        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))

        hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        output_comp = pnl.TransferMechanism(name='output_comp',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                    matrix=in_to_hidden_matrix.copy(),
                                    sender=input_comp,
                                    receiver=hidden_comp)

        hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                    matrix=hidden_to_out_matrix.copy(),
                                    sender=hidden_comp,
                                    receiver=output_comp)

        xor_comp = pnl.Composition()

        backprop_pathway = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                          in_to_hidden_comp,
                                                                          hidden_comp,
                                                                          hidden_to_out_comp,
                                                                          output_comp],
                                                                         learning_rate=10)
        # Try readd the same learning pathway (shouldn't error)
        backprop_pathway = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                          in_to_hidden_comp,
                                                                          hidden_comp,
                                                                          hidden_to_out_comp,
                                                                          output_comp],
                                                                         learning_rate=10)

    def test_run_no_targets(self):
        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))

        hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                    default_variable=np.zeros(10),
                                    function=pnl.Logistic())

        output_comp = pnl.TransferMechanism(name='output_comp',
                                    default_variable=np.zeros(1),
                                    function=pnl.Logistic())

        in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                    matrix=in_to_hidden_matrix.copy(),
                                    sender=input_comp,
                                    receiver=hidden_comp)

        hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                    matrix=hidden_to_out_matrix.copy(),
                                    sender=hidden_comp,
                                    receiver=output_comp)

        xor_comp = pnl.Composition()

        backprop_pathway = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                          in_to_hidden_comp,
                                                                          hidden_comp,
                                                                          hidden_to_out_comp,
                                                                          output_comp],
                                                                         learning_rate=10)
        # Try to run without any targets (non-learning
        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
            [0, 1],
            [1, 0],
            [1, 1]])
        xor_comp.run(inputs={input_comp:xor_inputs})

    def test_indepedence_of_learning_pathways_using_same_mechs_in_different_comps(self):
        A = TransferMechanism(name="Mech A")
        B = TransferMechanism(name="Mech B")

        comp1 = Composition(pathways=([A,B], BackPropagation))
        comp1.learn(inputs={A: 1.0,
                    comp1.pathways[0].target: 0.0},
                    num_trials=2)
        assert np.allclose(comp1.results, [[[1.]], [[0.9]]])

        comp2 = Composition()
        comp2.add_backpropagation_learning_pathway(pathway=[A,B], name='P1')
        comp2.learn(inputs={A: 1.0},
                    targets={B: 0.0},
                    num_trials=2)
        assert np.allclose(comp2.results, comp1.results)


class TestNoLearning:

    def test_multilayer(self):
        Input_Layer = pnl.TransferMechanism(
            name='Input Layer',
            function=pnl.Logistic,
            default_variable=np.zeros((2,)),
        )

        Hidden_Layer_1 = pnl.TransferMechanism(
            name='Hidden Layer_1',
            function=pnl.Logistic(),
            default_variable=np.zeros((5,)),
        )

        Hidden_Layer_2 = pnl.TransferMechanism(
            name='Hidden Layer_2',
            function=pnl.Logistic(),
            default_variable=[0, 0, 0, 0],
        )

        Output_Layer = pnl.TransferMechanism(
            name='Output Layer',
            function=pnl.Logistic,
            default_variable=[0, 0, 0],
        )

        Input_Weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)

        # TEST LEARNING WITH:
        # CREATION OF FREE STANDING PROJECTIONS THAT HAVE NO LEARNING (Input_Weights, Middle_Weights and Output_Weights)
        # INLINE CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)
        # NO EXPLICIT CREATION OF PROJECTIONS (Input_Weights, Middle_Weights and Output_Weights)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        Input_Weights = pnl.MappingProjection(
            name='Input Weights',
            matrix=Input_Weights_matrix,
        )

        c = pnl.Composition()
        learning_pathway = c.add_backpropagation_learning_pathway(pathway=[Input_Layer,
                                                                           Input_Weights,
                                                                           Hidden_Layer_1,
                                                                           Hidden_Layer_2,
                                                                           Output_Layer])
        target = learning_pathway.target
        stim_list = {Input_Layer: [[-1, 30]],
                     target: [0, 0, 1]}
        c.run(num_trials=10, inputs=stim_list, clamp_input=pnl.SOFT_CLAMP)

        expected_Output_Layer_output = [np.array([0.97988347, 0.97988347, 0.97988347])]

        np.testing.assert_allclose(expected_Output_Layer_output, Output_Layer.get_output_values(c))


class TestHebbian:

    def test_simple_hebbian(self):
        Hebb_C = pnl.Composition()
        size = 9

        Hebb2 = pnl.RecurrentTransferMechanism(
            size=size,
            function=pnl.Linear,
            enable_learning=True,
            hetero=0.,
            auto=0.,
            name='Hebb2',
        )

        Hebb_C.add_node(Hebb2)

        src = [1, 0, 0, 1, 0, 0, 1, 0, 0]

        inputs_dict = {Hebb2: np.array(src)}
        output = Hebb_C.learn(num_trials=5,
                   inputs=inputs_dict)

        activity = Hebb2.value

        assert np.allclose(activity, [[1.86643089, 0., 0., 1.86643089, 0., 0., 1.86643089, 0., 0.]])


class TestReinforcement:

    def test_rl(self):
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')
            input_layer.log.set_log_conditions(items=pnl.VALUE)
            action_selection = pnl.DDM(input_format=pnl.ARRAY,
                                       function=pnl.DriftDiffusionAnalytical(),
                                       output_ports=[pnl.SELECTED_INPUT_ARRAY],
                                       name='DDM')
            action_selection.log.set_log_conditions(items=pnl.SELECTED_INPUT_ARRAY)

            comp = pnl.Composition(name='comp')
            learning_pathway = comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                                                          learning_rate=0.05)
            learned_projection = learning_pathway.learning_components[pnl.LEARNED_PROJECTIONS]
            learning_mechanism = learning_pathway.learning_components[pnl.LEARNING_MECHANISMS]
            target_mechanism = learning_pathway.target
            comparator_mechanism = learning_pathway.learning_objective

            learned_projection.log.set_log_conditions(items=["matrix", "mod_matrix"])

            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]
                           }
            learning_mechanism.log.set_log_conditions(items=[pnl.VALUE])
            comparator_mechanism.log.set_log_conditions(items=[pnl.VALUE])

            target_mechanism.log.set_log_conditions(items=pnl.VALUE)
            comp.learn(inputs=inputs_dict)


            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

    def test_reinforcement_fixed_targets(self):
        input_layer = pnl.TransferMechanism(size=2,
                                        name='Input Layer',
        )

        action_selection = pnl.DDM(input_format=pnl.ARRAY,
                                   function=pnl.DriftDiffusionAnalytical(),
                                   output_ports=[pnl.SELECTED_INPUT_ARRAY],
                                   name='DDM')
        c = pnl.Composition()
        c.add_reinforcement_learning_pathway([input_layer, action_selection], learning_rate=0.05)

        # LEARN:
        c.learn(inputs={input_layer: [[1, 1], [1, 1]]},
                targets={action_selection: [[10.], [10.]]})

        assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                    [2.08614798], [1.85006765], [2.30401336], [2.08614798], [1.85006765]])

    def test_prediction_error_delta_first_run(self):
        learning_rate = 0.3

        stimulus_onset = 41
        sample = np.zeros(60)
        sample[stimulus_onset:] = 1

        reward_onset = 54
        target = np.zeros(60)
        target[reward_onset] = 1

        delta_function = pnl.PredictionErrorDeltaFunction()
        delta_vals = np.zeros((60, 60))

        weights = np.zeros(60)
        for t in range(60):
            print("Timestep {}".format(t))
            new_sample = sample * weights
            # print("sample = {}".format(new_sample))
            delta_vals[t] = delta_function(variable=[new_sample, target])
            print("delta: {}".format(delta_vals[t]))

            for i in range(59):
                weights[i] = weights[i] + learning_rate * sample[i] * \
                             delta_vals[t][i + 1]

        validation_array = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3,
                                      0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.09,
                                      0.42000000000000004, 0.49, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027, 0.189,
                                      0.44100000000000006, 0.34299999999999997, 0.0,
                                      0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0081, 0.0756,
                                      0.2646, 0.4116, 0.24009999999999998, 0.0, 0.0,
                                      0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.00243, 0.02835, 0.1323,
                                      0.3087, 0.3601500000000001,
                                      0.16806999999999994, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0007289999999999999,
                                      0.010206, 0.05953499999999999, 0.18522,
                                      0.32413500000000006, 0.30252599999999996,
                                      0.117649, 0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.00021869999999999998, 0.0035721,
                                      0.025004699999999998, 0.09724049999999998,
                                      0.2268945, 0.31765230000000005,
                                      0.24706289999999997, 0.08235429999999999, 0.0,
                                      0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 6.560999999999999e-05,
                                      0.0012247199999999999, 0.01000188, 0.04667544,
                                      0.1361367, 0.25412184, 0.29647548,
                                      0.19765032000000005, 0.05764800999999997, 0.0,
                                      0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.9682999999999998e-05, 0.000413343,
                                      0.003857868, 0.021003947999999998,
                                      0.073513818, 0.171532242, 0.26682793199999993,
                                      0.2668279320000001, 0.15564962699999996,
                                      0.040353607000000014, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      5.904899999999999e-06, 0.000137781,
                                      0.0014467005, 0.009001692,
                                      0.036756909000000004, 0.1029193452,
                                      0.200120949, 0.26682793199999993,
                                      0.2334744405000001, 0.12106082099999993,
                                      0.028247524900000043, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.7714699999999997e-06,
                                      4.5467729999999994e-05, 0.00053045685,
                                      0.0037131979500000002, 0.0173282571,
                                      0.05660563986, 0.13207982633999998,
                                      0.2201330439, 0.25682188454999993,
                                      0.19975035465000013, 0.09321683216999987,
                                      0.019773267430000074, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      5.314409999999999e-07, 1.4880347999999997e-05,
                                      0.00019096446599999996, 0.00148527918,
                                      0.0077977156950000005, 0.029111471928000003,
                                      0.07924789580399999, 0.15849579160799998,
                                      0.23113969609499996, 0.23970042558000004,
                                      0.16779029790600009, 0.07118376274799987,
                                      0.013841287201000085, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.5943229999999994e-07, 4.8361131e-06,
                                      6.770558339999998e-05, 0.0005792588802,
                                      0.0033790101345, 0.014191842564900003,
                                      0.044152399090799994, 0.1030222645452,
                                      0.18028896295409996, 0.23370791494049992,
                                      0.21812738727780012, 0.1388083373586,
                                      0.05398102008389993, 0.009688901040700082,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 1.61026623e-06,
                                      2.3696954189999994e-05, 0.00022117157244,
                                      0.00141918425649, 0.006622859863620001,
                                      0.023180009522670002, 0.06181335872711999,
                                      0.12620227406787, 0.19631464855001995,
                                      0.22903375664168996, 0.19433167230204007,
                                      0.11336014217619006, 0.040693384370939945,
                                      0.006782230728490046, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      8.719352486999997e-06, 8.2939339665e-05,
                                      0.000580575377655, 0.002980286938629,
                                      0.011590004761335003, 0.034770014284004995,
                                      0.08113003332934499, 0.14723598641251498,
                                      0.20613038097752096, 0.218623131339795,
                                      0.1700402132642851, 0.09156011483461501,
                                      0.03052003827820493, 0.0047475615099430435,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      3.3601154386499996e-05,
                                      0.00023223015106199995, 0.0013004888459472001,
                                      0.0055632022854408, 0.018544007618136,
                                      0.048678019997607, 0.100961819254296,
                                      0.16490430478201676, 0.20987820608620322,
                                      0.20404825591714193, 0.1464961837353841,
                                      0.07324809186769199, 0.02278829524772641,
                                      0.0033232930569601082, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.00010327019970509998, 0.0005527077595275599,
                                      0.0025793028777952804, 0.00945744388524936,
                                      0.0275842113319773, 0.0643631597746137,
                                      0.12014456491261222, 0.17839647517327273,
                                      0.2081292210354848, 0.18678263426261454,
                                      0.12452175617507655, 0.05811015288170229,
                                      0.016948794590496474, 0.002326305139872087,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.00026908252756336796, 0.0011606862950078762,
                                      0.004642745180031503, 0.014895474119267742,
                                      0.038617895864768215, 0.08109758131601326,
                                      0.13762013799081035, 0.18731629893193635,
                                      0.2017252450036237, 0.1681043708363532,
                                      0.10459827518706422, 0.045761745394340525,
                                      0.012562047755309225, 0.0016284135979104386,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0006172884160657308, 0.002205303960514964,
                                      0.007718563861802375, 0.022012200642917885,
                                      0.05136180150014173, 0.09805434831845238,
                                      0.15252898627314818, 0.1916389827534426,
                                      0.1916389827534425, 0.1490525421415665,
                                      0.08694731624924712, 0.03580183610263121,
                                      0.009281957508089578, 0.0011398895185372737,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.00127887960422022,
                                      0.0038592819309011877, 0.012006654896137028,
                                      0.030817080900085034, 0.06536956554563493,
                                      0.11439673970486111, 0.1642619852172365,
                                      0.1916389827534426, 0.1788630505698796,
                                      0.13042097437387068, 0.07160367220526243,
                                      0.027845872524268733, 0.006839337111223864,
                                      0.0007979226629760694, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0024366641834905763, 0.006303493820471939,
                                      0.01764978269732143, 0.04118282629375,
                                      0.08007771779340278, 0.12935631335857373,
                                      0.17247508447809834, 0.1878062030983737,
                                      0.16433042771107698, 0.11277578372328811,
                                      0.058476332300964384, 0.021543911900355206,
                                      0.005026912776749604, 0.0005585458640832153,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0043277123296321576, 0.009707380483526788,
                                      0.024709695776250002, 0.05285129374364584,
                                      0.09486129646295406, 0.1422919446944311,
                                      0.17707442006418095, 0.18076347048218466,
                                      0.1488640345147404, 0.09648594829659096,
                                      0.047396606180781564, 0.01658881216327357,
                                      0.0036864027029497315, 0.0003909821048582174,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.007239926474690194,
                                      0.014208075071343751, 0.03315217516646875,
                                      0.06545429455943831, 0.10909049093239716,
                                      0.15272668730535605, 0.17818113518958212,
                                      0.17119363969195134, 0.13315060864929562,
                                      0.08175914566184805, 0.03815426797552923,
                                      0.012718089325176374, 0.0026977765235223217,
                                      0.00027368747340072996, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.011502348996093318,
                                      0.01989130509988125, 0.04284281098435962,
                                      0.07854515347132598, 0.1221813498442848,
                                      0.16036302167062388, 0.17608488654029292,
                                      0.15978073037915452, 0.11773316975306136,
                                      0.0686776823559524, 0.030523414380423386,
                                      0.009711995484680158, 0.0019705498084858775,
                                      0.00019158123138052208, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.017469740526057695,
                                      0.02677675686522476, 0.053553513730449524,
                                      0.09163601238321362, 0.13363585139218653,
                                      0.1650795811315246, 0.17119363969195145,
                                      0.14716646219132656, 0.10301652353392865,
                                      0.05723140196329368, 0.02427998871170045,
                                      0.0073895617818218184, 0.0014368592353543042,
                                      0.00013410686196635435, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.02550276758562512,
                                      0.03480978392479219, 0.06497826332627876,
                                      0.10423596408590546, 0.14306897031398796,
                                      0.16691379869965267, 0.16398548644176403,
                                      0.13392148059410713, 0.08928098706273813,
                                      0.0473459779878157, 0.01921286063273686,
                                      0.005603751017881575, 0.0010460335233379858,
                                      9.387480337641474e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.035945702763062776,
                                      0.04386032774523817, 0.07675557355416678,
                                      0.1158858659543302, 0.1502224188296874,
                                      0.16603530502228608, 0.15496628468746698,
                                      0.12052933253469633, 0.07670048434026144,
                                      0.03890604278129206, 0.015130127748280264,
                                      0.004236435769518487, 0.0007603859073495034,
                                      6.571236236352362e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.04910380108663422,
                                      0.05372890148791675, 0.0884946612742158,
                                      0.12618683181693738, 0.15496628468746698,
                                      0.16271459892184037, 0.1446351990416358,
                                      0.10738067807636587, 0.06536215187257055,
                                      0.0317732682713886, 0.011862020154651653,
                                      0.0031936208108678255, 0.0005519838438536873,
                                      4.5998653654510946e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.06522247153300925,
                                      0.06415862942380646, 0.09980231243703228,
                                      0.13482066767809628, 0.157290778957779,
                                      0.157290778957779, 0.13345884275205488,
                                      0.09477512021522716, 0.05528548679221601,
                                      0.025799893836367493, 0.009261500351516516,
                                      0.002401129720763562, 0.000400188286794001,
                                      3.219905755813546e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.08447006036015119,
                                      0.07485173432777421, 0.1103078190093515,
                                      0.14156170106200106, 0.157290778957779,
                                      0.15014119809606175, 0.12185372599100663,
                                      0.08292823018832374, 0.04643980890546151,
                                      0.02083837579091219, 0.007203389162290574,
                                      0.0018008472905727269, 0.0002897915180232191,
                                      2.2539340290728127e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.10692558065848345,
                                      0.0854885597322474, 0.11968398362514637,
                                      0.1462804244307344, 0.15514590469926381,
                                      0.14165495646454518, 0.11017607725020184,
                                      0.07198170380346502, 0.038759378971096714,
                                      0.016747879802325727, 0.005582626600775242,
                                      0.0013475305588078745, 0.00020961586470347182,
                                      1.5777538203476382e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.13257214857815766,
                                      0.0957471869001171, 0.12766291586682277,
                                      0.14894006851129327, 0.15109862022884823,
                                      0.13221129270024212, 0.09871776521618081,
                                      0.062015006353754565, 0.032155929220465396,
                                      0.013398303841860582, 0.0043120977881849765,
                                      0.0010061561505766425, 0.00015146436675339547,
                                      1.1044276742477876e-05, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.16129630464819278,
                                      0.10532190559012883, 0.13404606166016392,
                                      0.14958763402655972, 0.1454324219702664,
                                      0.12216323445502375, 0.08770693755745296,
                                      0.05305728321376779, 0.02652864160688395,
                                      0.0106724420257579, 0.003320315296902465,
                                      0.0007497486154296462, 0.0001093383397501313,
                                      7.730993719756718e-06, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.19289287632523142,
                                      0.11393915241113936, 0.13870853337008265,
                                      0.1483410704096717, 0.13845166571569367,
                                      0.11182634538575253, 0.07731204125434732,
                                      0.045098690731702695, 0.021771781732546125,
                                      0.008466804007101203, 0.0025491452924606417,
                                      0.0005576255327258695, 7.885613594094121e-05,
                                      5.411695603863009e-06, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.22707462204857323,
                                      0.12136996669882236, 0.14159829448195937,
                                      0.14537424900147833, 0.1304640696167113,
                                      0.10147205414633098, 0.06764803609755388,
                                      0.038100618031955746, 0.017780288414912637,
                                      0.00669150639270899, 0.0019516893645402655,
                                      0.0004139947136904132, 5.682280383978444e-05,
                                      3.7881869227041065e-06, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.26348561205821996,
                                      0.12743846503376344, 0.14273108083781505,
                                      0.14090119518604827, 0.12176646497559718,
                                      0.09132484873169788, 0.058783810677874415,
                                      0.03200451914684277, 0.014453653808251588,
                                      0.005269561284258373, 0.001490380969285332,
                                      0.00030684314073514685, 4.091241876469365e-05,
                                      2.6517308459039768e-06, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.301717151568349,
                                      0.1320262497749789, 0.14218211514228507,
                                      0.13516077612291288, 0.11263398010242742,
                                      0.08156253731555085, 0.05075002321856492,
                                      0.026739259545265348, 0.011698426051053645,
                                      0.004135807189766472, 0.001135319620720332,
                                      0.00022706392414395538,
                                      2.9434212389101155e-05, 1.856211592099477e-06,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.3413250265008427,
                                      0.13507300938517075, 0.14007571343647335,
                                      0.12840273731676732, 0.10331254726636441,
                                      0.07231878308645512, 0.04354679411657503,
                                      0.02222700949700185, 0.009429640392667471,
                                      0.0032356609190525853, 0.0008628429117474301,
                                      0.000167775010617488, 2.116081215008947e-05,
                                      1.2993481144363273e-06, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.3818469293163939,
                                      0.1365738206005615, 0.13657382060056156,
                                      0.12087568030164642, 0.09401441801239163,
                                      0.06368718639549109, 0.03715085873070312,
                                      0.018387798765701513, 0.00757144655058295,
                                      0.0025238155168610943, 0.0006543225414084031,
                                      0.00012379075107726845,
                                      1.5202372939393527e-05, 9.09543680149838e-07,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.4228190754965624,
                                      0.13657382060056145, 0.13186437851088706,
                                      0.11281730161487002, 0.08491624852732138,
                                      0.05572628809605473, 0.031521940741202625,
                                      0.01514289310116601, 0.006057157240466293,
                                      0.0019629676242253202, 0.0004951630043090738,
                                      9.121423763591707e-05, 1.0914524161576011e-05,
                                      6.366805761492955e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.4637912216767308,
                                      0.13516098797365916, 0.12615025544208192,
                                      0.10444698568860544, 0.07615926039794141,
                                      0.04846498388959908, 0.026608226449191696,
                                      0.012417172342956029, 0.0048289003555940235,
                                      0.0015226262382503908, 0.0003739783743071934,
                                      6.712432359357035e-05, 7.831171085936894e-06,
                                      4.4567640333781355e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.5043395180688286,
                                      0.13245776821418598, 0.11963927451603895,
                                      0.09596066810140624, 0.06785097744543866,
                                      0.04190795665747693, 0.02235091021732094,
                                      0.010140690746747505, 0.003837018120390834,
                                      0.0011780318790675093, 0.00028192215909306206,
                                      4.933637784132472e-05, 5.6155226810794545e-06,
                                      3.119734823808784e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.5440768485330844,
                                      0.12861222010474194, 0.11253569259164908,
                                      0.087527760904616, 0.06006807120905011,
                                      0.03604084272543018, 0.0186878443761489,
                                      0.008249588958840537, 0.0030393222479937476,
                                      0.0009091989630751751, 0.00021214642471756306,
                                      3.6220121293228935e-05, 4.024457921469882e-06,
                                      2.183814377110238e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.582660514564507,
                                      0.12378926185081407, 0.1050333130855392,
                                      0.07928985399594612, 0.05285990266396423,
                                      0.030834943220645727, 0.015556367750956368,
                                      0.006686508945586533, 0.0024002852625182314,
                                      0.0007000832015678915, 0.00015936853369025172,
                                      2.6561422281634606e-05,
                                      2.8826349763866332e-06,
                                      1.5286700638661443e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.6197972931197512,
                                      0.11816247722123163, 0.09731027535866121,
                                      0.0713608685963516, 0.04625241483096865,
                                      0.02625137057973892, 0.012895410109345473,
                                      0.005400641840666021, 0.0018902246442331627,
                                      0.0005378688012045441, 0.00011952640026768879,
                                      1.9457786090026907e-05,
                                      2.0637045854421388e-06,
                                      1.0700690444842564e-07, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.6552460362861207,
                                      0.1119068166624605, 0.08952545332996831,
                                      0.0638283324667368, 0.04025210155559966,
                                      0.02224458243862093, 0.010646979628741615,
                                      0.004347516681736163, 0.0014845178913245327,
                                      0.000412366080923543, 8.950581601441243e-05,
                                      1.4239561638595966e-05,
                                      1.4766952811662293e-06, 7.490483311389795e-08,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.6888180812848588,
                                      0.10519240766271287, 0.0818163170709989,
                                      0.05675546319339564, 0.03484984582050599,
                                      0.018765301595657147, 0.008757140744639957,
                                      0.003488617044612674, 0.0011628723482043357,
                                      0.0003155080014507483, 6.69259397017008e-05,
                                      1.0410701731355942e-05,
                                      1.0561581467172232e-06,
                                      5.2433383190830796e-08, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.7203758035836726,
                                      0.0981795804851987, 0.07429806090771796,
                                      0.05018377798152873, 0.030024482553051346,
                                      0.01576285334035199, 0.007176583634631806,
                                      0.0027908936356900726, 0.0009086630441783594,
                                      0.00024093338292596744, 4.997136831064175e-05,
                                      7.604338655986531e-06, 7.550407176148966e-07,
                                      3.670336823358156e-08, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.7498296777292321,
                                      0.09101512461195449, 0.06706377602986124,
                                      0.04413598935298546, 0.025745993789241584,
                                      0.013186972428635979, 0.005860876634949275,
                                      0.002226224458236503, 0.0007083441458026751,
                                      0.00018364477854138084, 3.726125941427849e-05,
                                      5.549549274452836e-06, 5.395395127338887e-07,
                                      2.569235779681378e-08, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.7771342151128184,
                                      0.08382972003732658, 0.06018544002679849,
                                      0.03861899068386232, 0.02197828738105989,
                                      0.010989143690529946, 0.004770480981935443,
                                      0.0017708603645063548, 0.0005509343356242535,
                                      0.00013972972280329454,
                                      2.7747746372375204e-05, 4.046546345892743e-06,
                                      3.853853662860729e-07, 1.7984650435565186e-08,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8022831311240164,
                                      0.07673643603416813, 0.05371550522391766,
                                      0.033626779693021636, 0.018681544273900896,
                                      0.009123544877951528, 0.0038705947967068166,
                                      0.0014048825558417022, 0.00042757295177797694,
                                      0.00010613512987400764,
                                      2.0637386364374954e-05,
                                      2.9481980520218443e-06,
                                      2.7516515155312504e-07, 1.258925530489563e-08,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8253040619342669,
                                      0.06983015679109295, 0.04768888756464884,
                                      0.029143209067285403, 0.015814144455116086,
                                      0.0075476598535781925, 0.00313088112444726,
                                      0.0011116896746226068, 0.00033114160520675284,
                                      8.048580682107342e-05, 1.5330629870691226e-05,
                                      2.146288181847922e-06, 1.9639238268975845e-07,
                                      8.812478746733632e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8462531089715948,
                                      0.06318777602315973, 0.04212518401543974,
                                      0.025144489683634585, 0.013334199074654718,
                                      0.006222626234838935, 0.002525123689499864,
                                      0.0008775252537979172, 0.0002559448656910268,
                                      6.0939253735958765e-05,
                                      1.1375327364060439e-05,
                                      1.5613194420671661e-06, 1.401184115401577e-07,
                                      6.168735078304621e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8652094417785428,
                                      0.0568689984208437, 0.037030975715898196,
                                      0.02160140250094056, 0.011200727222710039,
                                      0.005113375471237247, 0.002030844158789291,
                                      0.0006910511373657835, 0.000197443182104462,
                                      4.607007582446698e-05, 8.431124987495764e-06,
                                      1.1349591328979614e-06, 9.993350857939731e-08,
                                      4.3181145326087744e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8822701413047959,
                                      0.05091759160936005, 0.03240210375141095,
                                      0.018481199917471325, 0.009374521697268268,
                                      0.004188616077502871, 0.0016289062523622277,
                                      0.0005429687507872982, 0.0001520312502205634,
                                      3.4778390573309004e-05, 6.242275231160832e-06,
                                      8.244514455579832e-07, 7.124889034315629e-08,
                                      3.022680217235063e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.8975454187876039,
                                      0.04536294525197537, 0.028225832601229017,
                                      0.015749196451410374, 0.007818750011338693,
                                      0.0034207031299606783, 0.0013031250018897822,
                                      0.00042568750061722227,
                                      0.00011685539232642039, 2.621755597065345e-05,
                                      4.616928095502182e-06, 5.984906790157396e-07,
                                      5.078102727207323e-08, 2.115876140962314e-09,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.9111543023631965,
                                      0.04022181145675141, 0.024482841756283458,
                                      0.013370062519388881, 0.006499335946925311,
                                      0.0027854296915393872, 0.0010398937515080364,
                                      0.0003330378681299928, 8.96640414196348e-05,
                                      1.9737367608074763e-05, 3.411396870545147e-06,
                                      4.341777835037419e-07, 3.6181481921637726e-08,
                                      1.4811133430825407e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.9232208458002219,
                                      0.035500120546611, 0.02114900798521513,
                                      0.011308844547649799, 0.005385164070309534,
                                      0.002261768909530004, 0.0008278369864945789,
                                      0.0002600257201169631, 6.868603927612238e-05,
                                      1.4839576386815878e-05,
                                      2.5182311443883165e-06,
                                      3.1477889306241735e-07, 2.577137137027563e-08,
                                      1.0367793290555483e-09, 0.0, 0.0, 0.0, 0.0,
                                      0.0],
                                     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.9338708819642052,
                                      0.031194786778192207, 0.018196958953945574,
                                      0.009531740404447708, 0.0044481455220757304,
                                      0.0018315893326192878, 0.0006574936065812942,
                                      0.0002026238158647775, 5.253210040934153e-05,
                                      1.1143172814032098e-05,
                                      1.8571954689683423e-06, 2.280766365769793e-07,
                                      1.8350993724602915e-08, 7.257455747478048e-10,
                                      0.0, 0.0, 0.0, 0.0, 0.0],
                                     ])

        for i in range(len(delta_vals)):
            deltas = delta_vals[i]
            validation_deltas = validation_array[i]

            np.testing.assert_allclose(deltas, validation_deltas, atol=1e-08,
                                       err_msg="mismatch on timestep {}".format(i))

    def test_rl_enable_learning_false(self):
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')
            input_layer.log.set_log_conditions(items=pnl.VALUE)
            action_selection = pnl.DDM(input_format=pnl.ARRAY,
                                       function=pnl.DriftDiffusionAnalytical(),
                                       output_ports=[pnl.SELECTED_INPUT_ARRAY],
                                       name='DDM')
            action_selection.log.set_log_conditions(items=pnl.SELECTED_INPUT_ARRAY)

            comp = pnl.Composition(name='comp')
            learning_pathway = comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                                                          learning_rate=0.05)
            learned_projection = learning_pathway.learning_components[pnl.LEARNED_PROJECTIONS]
            learning_mechanism = learning_pathway.learning_components[pnl.LEARNING_MECHANISMS]
            target_mechanism = learning_pathway.learning_components[pnl.TARGET_MECHANISM]
            comparator_mechanism = learning_pathway.learning_components[pnl.OBJECTIVE_MECHANISM]

            learned_projection.log.set_log_conditions(items=["matrix", "mod_matrix"])

            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]
                           }
            learning_mechanism.log.set_log_conditions(items=[pnl.VALUE])
            comparator_mechanism.log.set_log_conditions(items=[pnl.VALUE])

            target_mechanism.log.set_log_conditions(items=pnl.VALUE)
            comp.learn(inputs=inputs_dict)


            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

            # Pause learning -- values are the same as the previous trial (because we pass in the same inputs)
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]]}
            comp.run(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.4275, 0.]), np.array([0.4275, 0.])])
            assert np.allclose(action_selection.value, [[1.], [2.30401336], [0.97340301], [0.02659699], [2.30401336],
                                                        [2.08614798], [1.85006765], [2.30401336], [2.08614798],
                                                        [1.85006765]])

            # Resume learning
            inputs_dict = {input_layer: [[1., 1.], [1., 1.]],
                           target_mechanism: [[10.], [10.]]}
            comp.learn(inputs=inputs_dict)
            assert np.allclose(learning_mechanism.value, [np.array([0.38581875, 0.]), np.array([0.38581875, 0.])])
            assert np.allclose(action_selection.value, [[1.], [0.978989672], [0.99996], [0.0000346908466], [0.978989672],
                                                        [0.118109771], [1.32123733], [0.978989672], [0.118109771],
                                                        [1.32123733]])

    def test_td_enabled_learning_false(self):

        # create processing mechanisms
        sample_mechanism = pnl.TransferMechanism(default_variable=np.zeros(60),
                                       name=pnl.SAMPLE)

        action_selection = pnl.TransferMechanism(default_variable=np.zeros(60),
                                                 function=pnl.Linear(slope=1.0, intercept=0.01),
                                                 name='Action Selection')

        sample_to_action_selection = pnl.MappingProjection(sender=sample_mechanism,
                                                           receiver=action_selection,
                                                           matrix=np.zeros((60, 60)))

        comp = pnl.Composition(name='TD_Learning')
        pathway = [sample_mechanism, sample_to_action_selection, action_selection]
        learning_pathway = comp.add_td_learning_pathway(pathway, learning_rate=0.3)

        comparator_mechanism = learning_pathway.learning_objective
        comparator_mechanism.log.set_log_conditions(pnl.VALUE)
        target_mechanism = learning_pathway.target

        # comp.show_graph()

        stimulus_onset = 41
        reward_delivery = 54

        # build input dictionary
        samples = []
        targets = []
        for trial in range(50):
            target = [0.] * 60
            target[reward_delivery] = 1.
            # {14, 29, 44, 59, 74, 89}
            if trial in {14, 29, 44}:
                target[reward_delivery] = 0.
            targets.append(target)

            sample = [0.] * 60
            for i in range(stimulus_onset, 60):
                sample[i] =1.
            samples.append(sample)

        inputs1 = {sample_mechanism: samples[0:30],
                  target_mechanism: targets[0:30]}

        inputs2 = {sample_mechanism: samples[30:50],
                   target_mechanism: targets[30:50]}

        comp.learn(inputs=inputs1)

        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_1_expected = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0.,
                            0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,  0.]

        trial_30_expected = [0.] * 40
        trial_30_expected += [
            0.06521536244675225, 0.0640993870383315, 0.09944290863181729, 0.13325956499595726, 0.15232363406006394,
            0.14570077419644378, 0.11414216814982991, 0.07374140787058237, 0.04546975436471501, 0.036210519138262454,
            0.03355295938927161, 0.024201157062338496, 0.010573534379529015, -0.9979331317238949, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]

        assert np.allclose(trial_1_expected, delta_vals[0][0])
        assert np.allclose(trial_30_expected, delta_vals[29][0])

        # Pause Learning
        comp.run(inputs={sample_mechanism: samples[0:3]})

        # Resume Learning
        comp.learn(inputs=inputs2)
        delta_vals = comparator_mechanism.log.nparray_dictionary()['TD_Learning'][pnl.VALUE]

        trial_50_expected = [0.] * 40
        trial_50_expected += [
            0.7149863408177357, 0.08193033235388536, 0.05988592388364977, 0.03829793050401187, 0.01972582584273075,
            0.007198872281648616, 0.0037918828476545263, 0.009224297157983563, 0.015045769646998886,
            0.00034051016062952577, -0.040721638768680624, -0.03599485605332753, 0.0539151932684796,
            0.07237361605659998, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

        assert np.allclose(trial_50_expected, delta_vals[49][0])

    def test_reinforcement_too_many_nodes(self):
        A = TransferMechanism()
        B = TransferMechanism()
        C = TransferMechanism()
        with pytest.raises(CompositionError) as error:
            comp = Composition(([A,B,C],pnl.Reinforcement))
        error_text = str(error.value)
        assert "Too many Nodes in learning pathway" in error_text
        assert "Use BackPropagation LearningFunction or see AutodiffComposition for other learning models" in error_text


class TestNestedLearning:

    def test_nested_learning(self):
        stim_size = 10
        context_size = 2
        num_actions = 4

        def Concatenate(variable):
            return np.append(variable[0], variable[1])

        stim_in = pnl.ProcessingMechanism(name='Stimulus',
                                          size=stim_size)
        context_in = pnl.ProcessingMechanism(name='Context',
                                             size=context_size)
        reward_in = pnl.ProcessingMechanism(name='Reward',
                                            size=1)

        perceptual_state = pnl.ProcessingMechanism(name='Current Port',
                                                   function=Concatenate,
                                                   input_ports=[{pnl.NAME: 'STIM',
                                                                  pnl.SIZE: stim_size,
                                                                  pnl.PROJECTIONS: stim_in},
                                                                 {pnl.NAME: 'CONTEXT',
                                                                  pnl.SIZE: context_size,
                                                                  pnl.PROJECTIONS: context_in}])

        action = pnl.ProcessingMechanism(name='Action',
                                         size=num_actions)

        # Nested Composition
        rl_agent_state = pnl.ProcessingMechanism(name='RL Agent Port',
                                                 size=5)
        rl_agent_action = pnl.ProcessingMechanism(name='RL Agent Action',
                                                  size=5)
        rl_agent = pnl.Composition(name='RL Agent')
        rl_learning_components = rl_agent.add_reinforcement_learning_pathway([rl_agent_state,
                                                                              rl_agent_action])
        rl_agent._analyze_graph()

        model = pnl.Composition(name='Adaptive Replay Model')
        model.add_nodes([stim_in, context_in, reward_in, perceptual_state, rl_agent, action])
        model.add_projection(sender=perceptual_state, receiver=rl_agent_state)
        model.add_projection(sender=reward_in, receiver=rl_learning_components.target)
        model.add_projection(sender=rl_agent_action, receiver=action)
        model.add_projection(sender=rl_agent, receiver=action)

        # model.show_graph(show_controller=True, show_nested=True, show_node_structure=True)

        stimuli = {stim_in: np.array([1] * stim_size),
                   context_in: np.array([10] * context_size)}
        #
        # print(model.run(inputs=stimuli))

    def test_nested_learn_then_run(self):
        iSs = np.array(
            [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])
             ],
        )

        cSs = np.array(
            [np.array(
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
            [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        il = pnl.TransferMechanism(size=D_i, name='input')
        cl = pnl.TransferMechanism(size=D_c, name='control')
        hl = pnl.TransferMechanism(size=D_h, name='hidden',
                                   function=pnl.Logistic(bias=-2))
        ol = pnl.TransferMechanism(size=D_o, name='output',
                                   function=pnl.Logistic(bias=-2))
        pih = pnl.MappingProjection(matrix=wih)
        pch = pnl.MappingProjection(matrix=wch)
        pco = pnl.MappingProjection(matrix=wco)
        pho = pnl.MappingProjection(matrix=who)

        mnet = pnl.Composition()

        target_mech = mnet.add_backpropagation_learning_pathway(
            [il, pih, hl, pho, ol],
            learning_rate=100
        ).target

        mnet.add_backpropagation_learning_pathway(
            [cl, pch, hl, pho, ol],
            learning_rate=100
        )

        mnet.add_backpropagation_learning_pathway(
            [cl, pco, ol],
            learning_rate=100
        )

        mnet._analyze_graph()

        inputs = {
            il: iSs,
            cl: cSs,
            target_mech: oSs
        }

        outer = pnl.Composition(name="outer-composition")
        outer.add_node(mnet)
        mnet.learn(inputs=inputs)

        del inputs[target_mech]
        # This run should not error, as we are no longer in learning mode (and hence, we shouldn't need the target mech inputs)
        outer.run(inputs={mnet: inputs})

    def test_stranded_nested_target_mech_error(self):
        ia = pnl.ProcessingMechanism(name='ia')
        ib = pnl.ProcessingMechanism(name='ib')
        oa = pnl.ProcessingMechanism(name='oa')
        ot = pnl.ProcessingMechanism(name='ot')

        inner_comp = pnl.Composition(name='inner_comp', nodes=[ia, ib])
        inner_comp_learning_pathway = inner_comp.add_backpropagation_learning_pathway([ia, ib], learning_rate=0.005)
        inner_comp_target = inner_comp_learning_pathway.target
        outer_comp = pnl.Composition(name='outer_comp', nodes=[oa, ot, inner_comp])
        outer_comp.add_projection(pnl.MappingProjection(), sender=oa, receiver=ia)

        try:
            outer_comp.learn({oa: 1, ot: 1})
        except CompositionError as e:
            assert e.error_value == (
                   f'Target mechanism {inner_comp_target.name} of nested Composition {inner_comp.name} is not being projected to '
                    f'from its enclosing Composition {outer_comp.name}. For a call to {outer_comp.name}.learn, {inner_comp_target.name} '
                    f'must have an afferent projection with a target value so that an error term may be computed. '
                    f'A reference to {inner_comp_target.name}, with which you can create the needed projection, can be found '
                    f'as the target attribute of the relevant pathway in {inner_comp.name}.pathways. '
            )

class TestBackProp:

    def test_matrix_spec_and_learning_rate(self):
        T1 = pnl.TransferMechanism(size = 2,
                                   initial_value= [[0.0,0.0]],
                                   name = 'INPUT LAYER')
        T2 = pnl.TransferMechanism(size= 1,
                                   function =pnl.Logistic,
                                   name = 'OUTPUT LAYER')
        W = np.array([[0.1],[0.2]])
        C = pnl.Composition()
        learning_pathway = C.add_backpropagation_learning_pathway(pathway=[T1, W, T2])
        target = learning_pathway.target
        inputs = {T1:[1,0], target:[1]}
        C.learning_components[2].learning_rate.base = 0.5
        result = C.learn(inputs=inputs, num_trials=2)
        assert np.allclose(result, [[[0.52497919]], [[0.55439853]]])

    @pytest.mark.pytorch
    def test_back_prop(self):

        input_layer = pnl.TransferMechanism(name="input",
                                            size=2,
                                            function=pnl.Logistic())

        hidden_layer = pnl.TransferMechanism(name="hidden",
                                             size=2,
                                             function=pnl.Logistic())

        output_layer = pnl.TransferMechanism(name="output",
                                             size=2,
                                             function=pnl.Logistic())

        comp = pnl.Composition(name="backprop-composition")
        backprop_pathway = comp.add_backpropagation_learning_pathway(pathway=[input_layer, hidden_layer, output_layer],
                                                                learning_rate=0.5)
        # learned_projection = learning_components[pnl.LEARNED_PROJECTION]
        # learned_projection.log.set_log_conditions(pnl.MATRIX)
        learning_mechanism = backprop_pathway.learning_components[pnl.LEARNING_MECHANISMS]
        target_mechanism = backprop_pathway.target
        # comparator_mechanism = learning_components[pnl.OBJECTIVE_MECHANISM]
        for node in comp.nodes:
            node.log.set_log_conditions(pnl.VALUE)
        # comp.show_graph(show_node_structure=True)
        eid="eid"

        comp.learn(inputs={input_layer: [[1.0, 1.0]],
                         target_mechanism: [[1.0, 1.0]]},
                 num_trials=5,
                 context=eid)

        # for node in comp.nodes:
        #     try:
        #         log = node.log.nparray_dictionary()
        #     except ValueError:
        #         continue
        #     if eid in log:
        #         print(node.name, " values:")
        #         values = log[eid][pnl.VALUE]
        #         for i, val in enumerate(values):
        #             print("     Trial ", i, ":  ", val)
        #         print("\n - - - - - - - - - - - - - - - - - - \n")
        #     else:
        #         print(node.name, " EMPTY LOG!")

    def test_multilayer(self):

        input_layer = pnl.TransferMechanism(name='input_layer',
                                            function=pnl.Logistic,
                                            size=2)

        hidden_layer_1 = pnl.TransferMechanism(name='hidden_layer_1',
                                               function=pnl.Logistic,
                                               size=5)

        hidden_layer_2 = pnl.TransferMechanism(name='hidden_layer_2',
                                               function=pnl.Logistic,
                                               size=4)

        output_layer = pnl.TransferMechanism(name='output_layer',
                                             function=pnl.Logistic,
                                             size=3)

        input_weights_matrix = (np.arange(2 * 5).reshape((2, 5)) + 1) / (2 * 5)
        middle_weights_matrix = (np.arange(5 * 4).reshape((5, 4)) + 1) / (5 * 4)
        output_weights_matrix = (np.arange(4 * 3).reshape((4, 3)) + 1) / (4 * 3)

        # This projection will be used by the process below by referencing it in the process' pathway;
        #    note: sender and receiver args don't need to be specified
        input_weights = pnl.MappingProjection(
            name='Input Weights',
            matrix=input_weights_matrix,
        )

        # This projection will be used by the process below by assigning its sender and receiver args
        #    to mechanismss in the pathway
        middle_weights = pnl.MappingProjection(
            name='Middle Weights',
            sender=hidden_layer_1,
            receiver=hidden_layer_2,
            matrix=middle_weights_matrix,
        )

        # Commented lines in this projection illustrate variety of ways in which matrix and learning signals can be specified
        output_weights = pnl.MappingProjection(
            name='Output Weights',
            sender=hidden_layer_2,
            receiver=output_layer,
            matrix=output_weights_matrix,
        )

        comp = pnl.Composition(name='multilayer')

        p = [input_layer, input_weights, hidden_layer_1, middle_weights, hidden_layer_2, output_weights, output_layer]
        backprop_pathway = comp.add_backpropagation_learning_pathway(
            pathway=p,
            loss_function='sse',
            learning_rate=1.
        )

        input_dictionary = {backprop_pathway.target: [[0., 0., 1.]],
                            input_layer: [[-1., 30.]]}

        # comp.show_graph(show_learning=True)

        comp.learn(inputs=input_dictionary,
                 num_trials=10)

        objective_output_layer = comp.nodes[5]

        expected_output = [
            (output_layer.get_output_values(comp), [np.array([0.22686074, 0.25270212, 0.91542149])]),
            # error here? why still MSE
            (objective_output_layer.output_ports[pnl.MSE].parameters.value.get(comp), np.array(0.04082589331852094)),
            (input_weights.get_mod_matrix(comp), np.array([
                [ 0.09900247, 0.19839653, 0.29785764, 0.39739191, 0.49700232],
                [ 0.59629092, 0.69403786, 0.79203411, 0.89030237, 0.98885379],
            ])),
            (middle_weights.get_mod_matrix(comp), np.array([
                [ 0.09490249, 0.10488719, 0.12074013, 0.1428774 ],
                [ 0.29677354, 0.30507726, 0.31949676, 0.3404652 ],
                [ 0.49857336, 0.50526254, 0.51830509, 0.53815062],
                [ 0.70029406, 0.70544225, 0.71717037, 0.73594383],
                [ 0.90192903, 0.90561554, 0.91609668, 0.93385292],
            ])),
            (output_weights.get_mod_matrix(comp), np.array([
                [-0.74447522, -0.71016859, 0.31575293],
                [-0.50885177, -0.47444784, 0.56676582],
                [-0.27333719, -0.23912033, 0.8178167 ],
                [-0.03767547, -0.00389039, 1.06888608],
            ])),
            (comp.parameters.results.get(comp), [
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

        # Test nparray output of log for Middle_Weights

        for i in range(len(expected_output)):
            val, expected = expected_output[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    @pytest.mark.parametrize('models', [
        # [pnl.SYSTEM,pnl.COMPOSITION],
        # [pnl.SYSTEM,'AUTODIFF'],
        [pnl.COMPOSITION,'AUTODIFF']
    ])
    @pytest.mark.pytorch
    def test_xor_training_identicalness_standard_composition_vs_autodiff(self, models):
        """Test equality of results for running 3-layered xor network using System, Composition and Autodiff"""

        num_epochs=2

        xor_inputs = np.array(  # the inputs we will provide to the model
            [[0, 0],
             [0, 1],
             [1, 0],
             [1, 1]])

        xor_targets = np.array(  # the outputs we wish to see from the model
            [[0],
             [1],
             [1],
             [0]])

        in_to_hidden_matrix = np.random.rand(2,10)
        hidden_to_out_matrix = np.random.rand(10,1)

        # SET UP MODELS --------------------------------------------------------------------------------

       # STANDARD Composition
        if pnl.COMPOSITION in models:

            input_comp = pnl.TransferMechanism(name='input_comp',
                                       default_variable=np.zeros(2))

            hidden_comp = pnl.TransferMechanism(name='hidden_comp',
                                        default_variable=np.zeros(10),
                                        function=pnl.Logistic())

            output_comp = pnl.TransferMechanism(name='output_comp',
                                        default_variable=np.zeros(1),
                                        function=pnl.Logistic())

            in_to_hidden_comp = pnl.MappingProjection(name='in_to_hidden_comp',
                                        matrix=in_to_hidden_matrix.copy(),
                                        sender=input_comp,
                                        receiver=hidden_comp)

            hidden_to_out_comp = pnl.MappingProjection(name='hidden_to_out_comp',
                                        matrix=hidden_to_out_matrix.copy(),
                                        sender=hidden_comp,
                                        receiver=output_comp)

            xor_comp = pnl.Composition()

            backprop_pathway = xor_comp.add_backpropagation_learning_pathway([input_comp,
                                                                              in_to_hidden_comp,
                                                                              hidden_comp,
                                                                              hidden_to_out_comp,
                                                                              output_comp],
                                                                             learning_rate=10)
            target_mech = backprop_pathway.target
            inputs_dict = {"inputs": {input_comp:xor_inputs},
                           "targets": {output_comp:xor_targets},
                           "epochs": num_epochs}
            result_comp = xor_comp.learn(inputs=inputs_dict)

        # AutodiffComposition
        if 'AUTODIFF' in models:

            input_autodiff = pnl.TransferMechanism(name='input',
                                       default_variable=np.zeros(2))

            hidden_autodiff = pnl.TransferMechanism(name='hidden',
                                        default_variable=np.zeros(10),
                                        function=pnl.Logistic())

            output_autodiff = pnl.TransferMechanism(name='output',
                                        default_variable=np.zeros(1),
                                        function=pnl.Logistic())

            in_to_hidden_autodiff = pnl.MappingProjection(name='in_to_hidden',
                                        matrix=in_to_hidden_matrix.copy(),
                                        sender=input_autodiff,
                                        receiver=hidden_autodiff)

            hidden_to_out_autodiff = pnl.MappingProjection(name='hidden_to_out',
                                        matrix=hidden_to_out_matrix.copy(),
                                        sender=hidden_autodiff,
                                        receiver=output_autodiff)

            xor_autodiff = pnl.AutodiffComposition(learning_rate=10,
                                                   optimizer_type='sgd')

            xor_autodiff.add_node(input_autodiff)
            xor_autodiff.add_node(hidden_autodiff)
            xor_autodiff.add_node(output_autodiff)

            xor_autodiff.add_projection(sender=input_autodiff, projection=in_to_hidden_autodiff, receiver=hidden_autodiff)
            xor_autodiff.add_projection(sender=hidden_autodiff, projection=hidden_to_out_autodiff, receiver=output_autodiff)
            xor_autodiff.infer_backpropagation_learning_pathways()

            inputs_dict = {"inputs": {input_autodiff:xor_inputs},
                           "targets": {output_autodiff:xor_targets},
                           "epochs": num_epochs}
            result_autodiff = xor_autodiff.learn(inputs=inputs_dict)

        # COMPARE WEIGHTS FOR PAIRS OF MODELS ----------------------------------------------------------
        if all(m in models for m in {pnl.COMPOSITION, 'AUTODIFF'}):
            assert np.allclose(in_to_hidden_autodiff.parameters.matrix.get(xor_autodiff), in_to_hidden_comp.get_mod_matrix(xor_comp))
            assert np.allclose(hidden_to_out_autodiff.parameters.matrix.get(xor_autodiff), hidden_to_out_comp.get_mod_matrix(xor_comp))
            assert np.allclose(result_comp, result_autodiff)

    @pytest.mark.parametrize('configuration', [
        'Y UP',
        'BRANCH UP',
        'EXTEND UP',
        'EXTEND DOWN BRANCH UP',
        'CROSS',
        'Y UP AND DOWN',
        'BRANCH DOWN',
        'EXTEND DOWN',
        'BOW',
        'COMPLEX',
        'JOIN BY TERMINAL'
    ])
    def test_backprop_with_various_intersecting_pathway_configurations(self, configuration, show_graph=False):
        '''Test add_backpropgation using various configuration of intersecting pathways

        References in description are to attachment point of added pathway (always A)
        Branches created/added left to right

        '''

        if 'Y UP' == configuration:
            # 1) First mech is already an origin (Y UP)
            #
            #    E            C
            #     \         /
            #      D       B
            #       \     /
            #        A + A
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {A})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {E,C})
            print(f'Completed configuration: {configuration}')

        if 'BRANCH UP' == configuration:
            # 2) First mech is intermediate (BRANCH UP)
            #
            #            C
            #             \
            #         E   B
            #       /      \
            #      B   +    A
            #     /
            #    D
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,B,E])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {A,D})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {C})
            print(f'Completed configuration: {configuration}')

        if 'EXTEND UP' == configuration:
            # 3) First mech is already a terminal (EXTEND UP)
            #
            #                  C
            #                /
            #               B
            #              /
            #         A + A
            #       /
            #      E
            #     /
            #    D
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {D})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {C})
            print(f'Completed configuration: {configuration}')

        if 'EXTEND DOWN BRANCH UP' == configuration:
            # 4) Intermediate mech is already an origin (EXTEND DOWN BRANCH UP)
            #
            #    D       C
            #     \     /
            #      A + A
            #         /
            #        B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[A,D])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {B})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {D, C})
            print(f'Completed configuration: {configuration}')

        if 'CROSS' == configuration:
            # 5) Intermediate mech is already an intermediate (CROSS)
            #
            #    E       C
            #     \     /
            #      A + A
            #     /     \
            #    D       B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A,E])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {D,B})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {E,C})
            print(f'Completed configuration: {configuration}')

        if 'Y UP AND DOWN' == configuration:
            # 6) Intermediate mech is already a terminal (Y UP AND DOWN)
            #
            #          C
            #          \
            #      A + A
            #     /     \
            #    D      B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {D,B})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {C})
            print(f'Completed configuration: {configuration}')

        if 'BRANCH DOWN' == configuration:
            # 7) Last mech is already an intermediate (BRANCH DOWN)
            #
            #    D
            #     \
            #      A + A
            #     /     \
            #    C       B
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[C,A,D])
            comp.add_backpropagation_learning_pathway(pathway=[B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {C,B})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {D})
            print(f'Completed configuration: {configuration}')

        if 'EXTEND DOWN' == configuration:
            # 8) Last mech is already a terminal (EXTEND DOWN)
            #
            #        A + A
            #       /     \
            #      E       B
            #     /         \
            #    D           C
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[C,B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {D,C})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {A})
            print(f'Completed configuration: {configuration}')

        if 'BOW' == configuration:
            # 9) Bow
            #
            #            F
            #           /
            #      C + C
            #     /     \
            #    B       D
            #     \     /
            #      A + A
            #     /
            #    E
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[E,A,B,C])
            comp.add_backpropagation_learning_pathway(pathway=[A,D,C,F])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {E})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {F})
            print(f'Completed configuration: {configuration}')

        if 'COMPLEX' == configuration:
            # 10) Complex
            #
            #          C        I
            #          \         \
            #      A + A      F   G
            #     /     \    /     \
            #    D      B + B   +  D
            #              /        \
            #             E         H
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            G = pnl.ProcessingMechanism(name='G')
            H = pnl.ProcessingMechanism(name='H')
            I = pnl.ProcessingMechanism(name='I')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,A])
            comp.add_backpropagation_learning_pathway(pathway=[B,A,C])
            comp.add_backpropagation_learning_pathway(pathway=[E,B,F])
            comp.add_backpropagation_learning_pathway(pathway=[H,D,G,I])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {E,H})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {C,I})
            print(f'Completed configuration: {configuration}')

        if 'JOIN BY TERMINAL' == configuration:
            # 8) Last mech is already a terminal (EXTEND DOWN)
            #
            #        A     F   A
            #       /     /     \
            #      E  +  B   +   B
            #     /       \
            #    D         C
            #
            pnl.clear_registry(pnl.MechanismRegistry)
            A = pnl.ProcessingMechanism(name='A')
            B = pnl.ProcessingMechanism(name='B')
            C = pnl.ProcessingMechanism(name='C')
            D = pnl.ProcessingMechanism(name='D')
            E = pnl.ProcessingMechanism(name='E')
            F = pnl.ProcessingMechanism(name='F')
            comp = pnl.Composition(name=configuration)
            comp.add_backpropagation_learning_pathway(pathway=[D,E,A])
            comp.add_backpropagation_learning_pathway(pathway=[C,B,F])
            comp.add_backpropagation_learning_pathway(pathway=[B,A])
            if show_graph == True:
                comp.show_graph(show_learning=True)
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.INPUT) for n in {D,C})
            assert all(n in comp.get_nodes_by_role(pnl.NodeRole.OUTPUT) for n in {A,F})
            print(f'Completed configuration: {configuration}')


    @pytest.mark.parametrize('order', [
        'color_full',
        'word_partial',
        'word_full',
        'full_overlap'
    ])
    def test_stroop_model_learning(self, order):
        """Test backpropagation learning for simple convergent/overlapping pathways"""

        # CONSTRUCT MODEL ---------------------------------------------------------------------------

        num_trials = 2

        color_to_hidden_wts = np.arange(4).reshape((2, 2))
        word_to_hidden_wts = np.arange(4).reshape((2, 2))
        hidden_to_response_wts = np.arange(4).reshape((2, 2))

        color_comp = pnl.TransferMechanism(size=2, name='Color')
        word_comp = pnl.TransferMechanism(size=2, name='Word')
        hidden_comp = pnl.TransferMechanism(size=2, function=pnl.Logistic(), name='Hidden')
        response_comp = pnl.TransferMechanism(size=2, function=pnl.Logistic(), name='Response')

        if order == 'color_full':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp]
        elif order == 'word_full':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            hidden_to_response_wts.copy(),
                            response_comp]
        elif order == 'word_partial':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            # FIX: CROSSED_PATHWAYS 7/28/19 [JDC]: THE FOLLOWING LINES CRASHES:
                            # response_comp
                            ]
        elif order == 'full_overlap':
            color_pathway = [color_comp,
                             color_to_hidden_wts.copy(),
                             hidden_comp,
                             hidden_to_response_wts.copy(),
                             response_comp]
            word_pathway = [word_comp,
                            word_to_hidden_wts.copy(),
                            hidden_comp,
                            hidden_to_response_wts.copy(),
                            response_comp
                            ]
        else:
            assert False, 'Bad order specified for test_stroop_model_learning'

        comp = pnl.Composition(name='Stroop Model - Composition')
        comp.add_backpropagation_learning_pathway(pathway=color_pathway,
                                          learning_rate=1)
        comp.add_backpropagation_learning_pathway(pathway=word_pathway,
                                          learning_rate=1)
        # comp.show_graph(show_learning=True)

        # RUN MODEL ---------------------------------------------------------------------------

        # print('\nEXECUTING COMPOSITION-----------------------\n')
        target = comp.get_nodes_by_role(pnl.NodeRole.TARGET)[0]
        results_comp = comp.learn(inputs={color_comp: [[1, 1]],
                                          word_comp: [[-2, -2]],
                                          target: [[1, 1]]},
                                  num_trials=num_trials)
        # print('\nCOMPOSITION RESULTS')
        # print(f'Results: {comp.results}')
        # print(f'color_to_hidden_comp: {comp.projections[0].get_mod_matrix(comp)}')
        # print(f'word_to_hidden_comp: {comp.projections[15].get_mod_matrix(comp)}')

        # VALIDATE RESULTS ---------------------------------------------------------------------------
        # Note:  numbers based on test of System in tests/learning/test_stroop

        composition_and_expected_outputs = [
            (color_comp.output_ports[0].parameters.value.get(comp), np.array([1., 1.])),
            (word_comp.output_ports[0].parameters.value.get(comp), np.array([-2., -2.])),
            (hidden_comp.output_ports[0].parameters.value.get(comp), np.array([0.13227553, 0.01990677])),
            (response_comp.output_ports[0].parameters.value.get(comp), np.array([0.51044657, 0.5483048])),
            (comp.nodes['Comparator'].output_ports[0].parameters.value.get(comp), np.array([0.48955343, 0.4516952])),
            (comp.nodes['Comparator'].output_ports[pnl.MSE].parameters.value.get(comp), np.array(
                    0.22184555903789838)),
            (comp.projections['MappingProjection from Color[RESULT] to Hidden[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [ 0.02512045, 1.02167245],
                 [ 2.02512045, 3.02167245],
             ])),
            (comp.projections['MappingProjection from Word[RESULT] to Hidden[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [-0.05024091, 0.9566551 ],
                 [ 1.94975909, 2.9566551 ],
             ])),
            (comp.projections['MappingProjection from Hidden[RESULT] to Response[InputPort-0]'].get_mod_matrix(comp),
             np.array([
                 [ 0.03080958, 1.02830959],
                 [ 2.00464242, 3.00426575],
             ])),
            ([results_comp[-1][0]], [np.array([0.51044657, 0.5483048])]),
        ]

        for i in range(len(composition_and_expected_outputs)):
            val, expected = composition_and_expected_outputs[i]
            # setting absolute tolerance to be in accordance with reference_output precision
            # if you do not specify, assert_allcose will use a relative tolerance of 1e-07,
            # which WILL FAIL unless you gather higher precision values to use as reference
            np.testing.assert_allclose(val, expected, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

    def test_pytorch_equivalence_with_learning_enabled_composition(self):
        iSs = np.array(
            [np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.58185035, 0.4143686, 0.4746975]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.47360805, 0.8009108, 0.5204775, 0.53737324, 0.7586156,
                    0.1059076, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.9023486,
                    0.09928035, 0.96980906, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.14967486, 0.22232139, 0.38648897, 0.75068617,
                    0.60783064, 0.32504722, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.53737324, 0.7586156,
                    0.1059076, 0.14967486, 0.22232139, 0.38648897, 0.65314,
                    0.17090958, 0.35815218, 0.03842543, 0.63427407, 0.95894927]),
             np.array([0.95715517, 0.14035077, 0.87008727, 0.47360042, 0.18633235,
                    0.73691815, 0.21655035, 0.13521817, 0.324141, 0.75068617,
                    0.60783064, 0.32504722, 0.6527903, 0.6350589, 0.9952996]),
             np.array([0.33739617, 0.6481719, 0.36824155, 0.47360042, 0.18633235,
                    0.73691815, 0.9025985, 0.44994998, 0.61306345, 0.9023486,
                    0.09928035, 0.96980906, 0.6527903, 0.6350589, 0.9952996])
             ],
        )

        cSs = np.array(
            [np.array(
                [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]),
             np.array(
                 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array(
                 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])]
        )

        oSs = np.array(
            [np.array([0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([1., 0., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., -0., 0., 1., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., -0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., -0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 1., -0., 0., 0., 0., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., -0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]),
             np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])]
        )

        nf = 3
        nd = 5
        nh = 200

        D_i = nf * nd
        D_c = nd ** 2
        D_h = nh
        D_o = nf * nd

        wih = np.random.rand(D_i, D_h) * 0.02 - 0.01
        wch = np.random.rand(D_c, D_h) * 0.02 - 0.01
        wco = np.random.rand(D_c, D_o) * 0.02 - 0.01
        who = np.random.rand(D_h, D_o) * 0.02 - 0.01

        il = pnl.TransferMechanism(size=D_i, name='input')
        cl = pnl.TransferMechanism(size=D_c, name='control')
        hl = pnl.TransferMechanism(size=D_h, name='hidden',
                                   function=pnl.Logistic(bias=-2))
        ol = pnl.TransferMechanism(size=D_o, name='output',
                                   function=pnl.Logistic(bias=-2))
        pih = pnl.MappingProjection(matrix=wih)
        pch = pnl.MappingProjection(matrix=wch)
        pco = pnl.MappingProjection(matrix=wco)
        pho = pnl.MappingProjection(matrix=who)

        mnet = pnl.Composition()

        target_mech = mnet.add_backpropagation_learning_pathway(
            [il, pih, hl, pho, ol],
            learning_rate=100
        ).target

        mnet.add_backpropagation_learning_pathway(
            [cl, pch, hl, pho, ol],
            learning_rate=100
        )

        mnet.add_backpropagation_learning_pathway(
            [cl, pco, ol],
            learning_rate=100
        )

        mnet._analyze_graph()

        inputs = {
            il: iSs,
            cl: cSs,
            target_mech: oSs
        }

        mnet.learn(inputs=inputs)
        mnet.run(inputs=inputs)

        comparator = np.array([0.02288846, 0.11646781, 0.03473711, 0.0348004, 0.01679579,
                             0.04851733, 0.05857743, 0.04819957, 0.03004438, 0.05113508,
                             0.06849843, 0.0442623, 0.00967315, 0.06998125, 0.03482444,
                             0.05856816, 0.00724313, 0.03676571, 0.03668758, 0.01761947,
                             0.0516829, 0.06260267, 0.05160782, 0.03140498, 0.05462971,
                             0.07360401, 0.04687923, 0.00993319, 0.07662302, 0.03687142,
                             0.0056837, 0.03411045, 0.03615285, 0.03606166, 0.01774354,
                             0.04700402, 0.09696857, 0.06843472, 0.06108671, 0.0485631,
                             0.07194324, 0.04485926, 0.00526768, 0.07442083, 0.0364541,
                             0.02819926, 0.03804169, 0.04091214, 0.04091113, 0.04246229,
                             0.05583883, 0.06643675, 0.05630667, 0.01540373, 0.05948422,
                             0.07721549, 0.05081813, 0.01205326, 0.07998289, 0.04084186,
                             0.02859247, 0.03794089, 0.04111452, 0.04139213, 0.01222424,
                             0.05677404, 0.06736114, 0.05614553, 0.03573626, 0.05983103,
                             0.07867571, 0.09971621, 0.01203033, 0.08107789, 0.04110497,
                             0.02694072, 0.03592752, 0.03878366, 0.03895513, 0.01852774,
                             0.05097689, 0.05753834, 0.05090328, 0.03405996, 0.05293719,
                             0.07037981, 0.03474316, 0.02861534, 0.12504038, 0.0387827,
                             0.02467716, 0.03373265, 0.03676382, 0.03677551, 0.00758558,
                             0.089832, 0.06330426, 0.0514472, 0.03120581, 0.05535174,
                             0.07494839, 0.04169744, 0.00698747, 0.0771042, 0.03659954,
                             0.03008443, 0.0393799, 0.0423592, 0.04237004, 0.00965198,
                             0.09863199, 0.06813933, 0.05675321, 0.03668943, 0.0606036,
                             0.07898065, 0.04662618, 0.00954765, 0.08093391, 0.04218842,
                             0.02701085, 0.03660227, 0.04058368, 0.04012464, 0.02030738,
                             0.047633, 0.06693405, 0.055821, 0.03456592, 0.10166267,
                             0.07870758, 0.04935871, 0.01065449, 0.08012213, 0.04036544,
                             0.02576563, 0.03553382, 0.03920509, 0.03914452, 0.01907667,
                             0.05106766, 0.06555857, 0.05434728, 0.03335726, 0.05074808,
                             0.07715102, 0.04839309, 0.02494798, 0.08001304, 0.03921895,
                             0.00686952, 0.03941704, 0.04128484, 0.04117602, 0.02217508,
                             0.05152296, 0.10361618, 0.07488737, 0.0707186, 0.05289282,
                             0.07557573, 0.04978292, 0.00705783, 0.07787788, 0.04164007,
                             0.00574239, 0.03437231, 0.03641445, 0.03631848, 0.01795791,
                             0.04723996, 0.09732232, 0.06876138, 0.06156679, 0.04878423,
                             0.07214104, 0.04511085, 0.00535038, 0.07459818, 0.0367153,
                             0.02415251, 0.03298647, 0.03586635, 0.0360273, 0.01624523,
                             0.04829838, 0.05523439, 0.04821285, 0.03115052, 0.05034625,
                             0.06836408, 0.03264844, 0.0241706, 0.12190507, 0.03585727,
                             0.02897192, 0.03925683, 0.04250414, 0.04253885, 0.02175426,
                             0.05683923, 0.06547528, 0.05705267, 0.03742978, 0.05951711,
                             0.12675475, 0.05216411, 0.00181494, 0.08218002, 0.04234364,
                             0.02789848, 0.036924, 0.03976586, 0.03993866, 0.01932489,
                             0.05186586, 0.05829845, 0.05179337, 0.03504668, 0.05379566,
                             0.07103772, 0.03544133, 0.03019486, 0.12605846, 0.03976812])

        assert np.allclose(comparator, np.array(mnet.parameters.results.get(mnet)[-15:]).reshape(225))


def validate_learning_mechs(comp):

    def get_learning_mech(name):
        return next(lm for lm in comp.get_nodes_by_role(pnl.NodeRole.LEARNING) if lm.name == name)

    REP_IN_to_REP_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_IN to REP_HIDDEN')
    REP_HIDDEN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REP_HIDDEN to REL_HIDDEN')
    REL_IN_to_REL_HIDDEN_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_IN to REL_HIDDEN')
    REL_HIDDEN_to_REP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to REP_OUT')
    REL_HIDDEN_to_PROP_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to PROP_OUT')
    REL_HIDDEN_to_QUAL_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to QUAL_OUT')
    REL_HIDDEN_to_ACT_OUT_LM = get_learning_mech('LearningMechanism for MappingProjection from REL_HIDDEN to ACT_OUT')

    # Validate error_signal Projections for REP_IN to REP_HIDDEN
    assert len(REP_IN_to_REP_HIDDEN_LM.input_ports) == 3
    assert REP_IN_to_REP_HIDDEN_LM.input_ports[pnl.ERROR_SIGNAL].path_afferents[0].sender.owner == \
           REP_HIDDEN_to_REL_HIDDEN_LM

    # Validate error_signal Projections to LearningMechanisms for REP_HIDDEN_to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REP_HIDDEN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})

    # Validate error_signal Projections to LearningMechanisms for REL_IN to REL_HIDDEN Projections
    assert all(lm in [input_port.path_afferents[0].sender.owner for input_port in
                      REL_IN_to_REL_HIDDEN_LM.input_ports]
               for lm in {REL_HIDDEN_to_REP_OUT_LM, REL_HIDDEN_to_PROP_OUT_LM,
                          REL_HIDDEN_to_QUAL_OUT_LM, REL_HIDDEN_to_ACT_OUT_LM})


class TestRumelhartSemanticNetwork:
    r"""
    Tests construction and training of network with both convergent and divergent pathways
    with the following structure:

    # Semantic Network:
    #                        __
    #    REP PROP QUAL ACT     |
    #      \   \  /   /   __   | Output Processes
    #       REL_HIDDEN      |__|
    #          /   \        |
    #  REP_HIDDEN  REL_IN   |  Input Processes
    #       /               |
    #   REP_IN           ___|
    """

    def test_rumelhart_semantic_network_sequential(self):

        rep_in = pnl.TransferMechanism(size=10, name='REP_IN')
        rel_in = pnl.TransferMechanism(size=11, name='REL_IN')
        rep_hidden = pnl.TransferMechanism(size=4, function=pnl.Logistic, name='REP_HIDDEN')
        rel_hidden = pnl.TransferMechanism(size=5, function=pnl.Logistic, name='REL_HIDDEN')
        rep_out = pnl.TransferMechanism(size=10, function=pnl.Logistic, name='REP_OUT')
        prop_out = pnl.TransferMechanism(size=12, function=pnl.Logistic, name='PROP_OUT')
        qual_out = pnl.TransferMechanism(size=13, function=pnl.Logistic, name='QUAL_OUT')
        act_out = pnl.TransferMechanism(size=14, function=pnl.Logistic, name='ACT_OUT')

        comp = pnl.Composition()

        # comp.add_backpropagation_learning_pathway(pathway=[rep_in, rep_hidden, rel_hidden])
        comp.add_backpropagation_learning_pathway(pathway=[rel_in, rel_hidden])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, rep_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, prop_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, qual_out])
        comp.add_backpropagation_learning_pathway(pathway=[rel_hidden, act_out])
        comp.add_backpropagation_learning_pathway(pathway=[rep_in, rep_hidden, rel_hidden])

        # gv = comp.show_graph(show_learning=True, show_node_structure=pnl.ALL)
        # gv = comp.show_graph(show_learning=True, show_node_structure=pnl.ALL, output_fmt='source')
        # assert gv == 'digraph "Composition-0" {\n\tgraph [overlap=False rankdir=BT]\n\tnode [color=black fontname=arial fontsize=12 penwidth=1 shape=record]\n\tedge [fontname=arial fontsize=10]\n\tREP_IN [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_IN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_IN</font></b><br/><i>INPUT,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tTarget [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target</font></b><br/><i>TARGET,INPUT,LEARNING,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREL_IN [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REL_IN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REL_IN</font></b><br/><i>INPUT,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREL_HIDDEN [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REL_HIDDEN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REL_HIDDEN</font></b><br/><i>INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=black penwidth=1 rank=same shape=plaintext]\n\t"MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_IN:"OutputPort-RESULT" -> "MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" -> REL_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" -> REL_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" -> REP_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" -> PROP_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-1" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-1" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-1</font></b><br/><i>TARGET,INPUT,LEARNING,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\t"MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" -> QUAL_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-2" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-2" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-2</font></b><br/><i>TARGET,INPUT,LEARNING,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\t"MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" -> ACT_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-3" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-3" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-3</font></b><br/><i>TARGET,INPUT,LEARNING,ORIGIN</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREP_HIDDEN [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_HIDDEN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_HIDDEN</font></b><br/><i>INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=black penwidth=1 rank=same shape=plaintext]\n\t"MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREP_IN:"OutputPort-RESULT" -> "MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" -> REP_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-1"><b>error_signal-1</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-2"><b>error_signal-2</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-3"><b>error_signal-3</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_IN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-2" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-3" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-1" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tComparator [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator</font></b><br/><i>LEARNING_OBJECTIVE,LEARNING,INTERNAL</i><br/><i>LinearCombination(offset=None, scale=None, exponents=None, weights=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tTarget:"OutputPort-OutputPort-0" -> Comparator:"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_OUT:"OutputPort-RESULT" -> Comparator:"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\tComparator:"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-1" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-1" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-1</font></b><br/><i>LEARNING_OBJECTIVE,LEARNING,INTERNAL</i><br/><i>LinearCombination(offset=None, scale=None, exponents=None, weights=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tPROP_OUT:"OutputPort-RESULT" -> "Comparator-1":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Target-1":"OutputPort-OutputPort-0" -> "Comparator-1":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tPROP_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-1":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-2" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-2" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-2</font></b><br/><i>LEARNING_OBJECTIVE,LEARNING,INTERNAL</i><br/><i>LinearCombination(offset=None, scale=None, exponents=None, weights=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tQUAL_OUT:"OutputPort-RESULT" -> "Comparator-2":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Target-2":"OutputPort-OutputPort-0" -> "Comparator-2":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-2":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tQUAL_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-3" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-3" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-3</font></b><br/><i>LEARNING_OBJECTIVE,LEARNING,INTERNAL</i><br/><i>LinearCombination(offset=None, scale=None, exponents=None, weights=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\t"Target-3":"OutputPort-OutputPort-0" -> "Comparator-3":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\tACT_OUT:"OutputPort-RESULT" -> "Comparator-3":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\tACT_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-3":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]</font></b><br/><i>LEARNING,INTERNAL</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-1"><b>error_signal-1</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-2"><b>error_signal-2</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-3"><b>error_signal-3</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-2" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-3" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-1" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]</font></b><br/><i>TERMINAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.]]), array([0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREP_IN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_OUT</font></b><br/><i>OUTPUT,INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tPROP_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="PROP_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >PROP_OUT</font></b><br/><i>OUTPUT,INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tQUAL_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="QUAL_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >QUAL_OUT</font></b><br/><i>OUTPUT,INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tACT_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="ACT_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >ACT_OUT</font></b><br/><i>OUTPUT,INTERNAL</i><br/><i>Logistic(gain=None, bias=None, offset=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(offset=None, scale=None, offset=None, scale=None, exponents=None, weights=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n}'
        # assert gv == 'digraph "Composition-0" {\n\tgraph [overlap=False rankdir=BT]\n\tnode [color=black fontname=arial fontsize=12 penwidth=1 shape=record]\n\tedge [fontname=arial fontsize=10]\n\tREP_IN [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_IN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_IN</font></b><br/><i>ORIGIN,INPUT</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tTarget [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target</font></b><br/><i>ORIGIN,INPUT,LEARNING,TARGET</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREL_IN [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REL_IN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REL_IN</font></b><br/><i>ORIGIN,INPUT</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREL_HIDDEN [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REL_HIDDEN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REL_HIDDEN</font></b><br/><i>INTERNAL</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=black penwidth=1 rank=same shape=plaintext]\n\t"MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" -> REL_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_IN:"OutputPort-RESULT" -> "MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" -> REL_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" -> REP_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" -> PROP_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-1" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-1" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-1</font></b><br/><i>ORIGIN,INPUT,LEARNING,TARGET</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\t"MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" -> QUAL_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-2" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-2" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-2</font></b><br/><i>ORIGIN,INPUT,LEARNING,TARGET</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\t"MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" -> ACT_OUT:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Target-3" [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OutputPort-0"><b>OutputPort-0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Target-3" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Target-3</font></b><br/><i>ORIGIN,INPUT,LEARNING,TARGET</i><br/><i>Linear(intercept=None, slope=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-intercept"><b>intercept</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-slope"><b>slope</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=green penwidth=3 rank=source shape=plaintext]\n\tREP_HIDDEN [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_HIDDEN" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_HIDDEN</font></b><br/><i>INTERNAL</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=black penwidth=1 rank=same shape=plaintext]\n\t"MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [color=black penwidth=1 shape=diamond]\n\tREP_IN:"OutputPort-RESULT" -> "MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [arrowhead=none color=black penwidth=1]\n\t"MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" -> REP_HIDDEN:"InputPort-InputPort-0" [color=black penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"OutputPort-LearningSignal" -> "MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [label="" color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-1"><b>error_signal-1</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-2"><b>error_signal-2</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-3"><b>error_signal-3</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-1" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_IN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-2" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REL_IN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-3" [label="" arrowhead=normal color=orange penwidth=1]\n\tComparator [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator</font></b><br/><i>INTERNAL,LEARNING_OBJECTIVE,LEARNING</i><br/><i>LinearCombination(weights=None, exponents=None, offset=None, scale=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREP_OUT:"OutputPort-RESULT" -> Comparator:"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\tTarget:"OutputPort-OutputPort-0" -> Comparator:"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREP_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tComparator:"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-1" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-1" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-1</font></b><br/><i>INTERNAL,LEARNING_OBJECTIVE,LEARNING</i><br/><i>LinearCombination(weights=None, exponents=None, offset=None, scale=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\t"Target-1":"OutputPort-OutputPort-0" -> "Comparator-1":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\tPROP_OUT:"OutputPort-RESULT" -> "Comparator-1":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-1":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tPROP_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-2" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-2" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-2</font></b><br/><i>INTERNAL,LEARNING_OBJECTIVE,LEARNING</i><br/><i>LinearCombination(weights=None, exponents=None, offset=None, scale=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tQUAL_OUT:"OutputPort-RESULT" -> "Comparator-2":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Target-2":"OutputPort-OutputPort-0" -> "Comparator-2":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tQUAL_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-2":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Comparator-3" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-OUTCOME"><b>OUTCOME</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td><td port="OutputPort-MSE"><b>MSE</b><br/><i>UserDefinedFunction()</i><br/>=[1.]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Comparator-3" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Comparator-3</font></b><br/><i>INTERNAL,LEARNING_OBJECTIVE,LEARNING</i><br/><i>LinearCombination(weights=None, exponents=None, offset=None, scale=None)</i><br/>=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-SAMPLE"><b>SAMPLE</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-TARGET"><b>TARGET</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\t"Target-3":"OutputPort-OutputPort-0" -> "Comparator-3":"InputPort-TARGET" [label="" arrowhead=normal color=orange penwidth=1]\n\tACT_OUT:"OutputPort-RESULT" -> "Comparator-3":"InputPort-SAMPLE" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\t"Comparator-3":"OutputPort-OUTCOME" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tACT_OUT:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]\n [0. 0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]</font></b><br/><i>INTERNAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.],\n       [0., 0., 0., 0., 0.]]), array([0., 0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-1"><b>error_signal-1</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-2"><b>error_signal-2</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-error_signal-3"><b>error_signal-3</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREL_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to PROP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-1" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to QUAL_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-2" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to REP_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REL_HIDDEN[RESULT] to ACT_OUT[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"InputPort-error_signal-3" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" [label=<<table border=\'1\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-error_signal"><b>error_signal</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0. 0. 0. 0.]</td><td port="OutputPort-LearningSignal"><b>LearningSignal</b><br/><i>Linear(intercept=None, slope=None, intercept=None, slope=None)</i><br/>=[[0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]\n [0. 0. 0. 0.]]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]</font></b><br/><i>TERMINAL,LEARNING</i><br/><i>BackPropagation(learning_rate=None)</i><br/>=[array([[0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.],\n       [0., 0., 0., 0.]]), array([0., 0., 0., 0.])]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-learning_rate"><b>learning_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.05]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-activation_input"><b>activation_input</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td><td port="InputPort-activation_output"><b>activation_output</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0.5 0.5 0.5 0.5]</td><td port="InputPort-error_signal"><b>error_signal</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=orange penwidth=1 rank=min]\n\tREP_IN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-activation_input" [label="" arrowhead=normal color=orange penwidth=1]\n\t"Learning Mechanism for MappingProjection from REP_HIDDEN[RESULT] to REL_HIDDEN[InputPort-0]":"OutputPort-error_signal" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-error_signal" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_HIDDEN:"OutputPort-RESULT" -> "Learning Mechanism for MappingProjection from REP_IN[RESULT] to REP_HIDDEN[InputPort-0]":"InputPort-activation_output" [label="" arrowhead=normal color=orange penwidth=1]\n\tREP_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="REP_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >REP_OUT</font></b><br/><i>INTERNAL,OUTPUT</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tPROP_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="PROP_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >PROP_OUT</font></b><br/><i>INTERNAL,OUTPUT</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tQUAL_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="QUAL_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >QUAL_OUT</font></b><br/><i>INTERNAL,OUTPUT</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n\tACT_OUT [label=<<table border=\'3\' cellborder="0" cellspacing="1" bgcolor="#FFFFF0"><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="OutputPort-RESULT"><b>RESULT</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]</td></tr></table></td></tr> <tr><td colspan="1" valign="middle"><b><i>OutputPorts</i></b></td></tr> </table></td></tr><tr><td port="ACT_OUT" colspan="1"><b><b><i>Mechanism</i></b>:<br/><font point-size="16" >ACT_OUT</font></b><br/><i>INTERNAL,OUTPUT</i><br/><i>Logistic(offset=None, bias=None, gain=None, x_0=None, scale=None)</i><br/>=[[0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]]</td><td> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td rowspan="1" valign="middle"><b><i>ParameterPorts</i></b></td> <td> <table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="ParameterPort-bias"><b>bias</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-gain"><b>gain</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-integration_rate"><b>integration_rate</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.5]</td></tr><tr><td port="ParameterPort-noise"><b>noise</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-offset"><b>offset</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr><tr><td port="ParameterPort-scale"><b>scale</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[1.]</td></tr><tr><td port="ParameterPort-x_0"><b>x_0</b><br/><i>Linear(intercept=None, slope=None)</i><br/>=[0.]</td></tr></table></td></tr></table></td></tr><tr><td colspan="2"> <table border="0" cellborder="0" bgcolor="#FAFAD0"> <tr><td colspan="1" valign="middle"><b><i>InputPorts</i></b></td></tr><tr><td><table border="0" cellborder="2" cellspacing="0" color="LIGHTGOLDENRODYELLOW" bgcolor="PALEGOLDENROD"><tr><td port="InputPort-InputPort-0"><b>InputPort-0</b><br/><i>LinearCombination(weights=None, exponents=None, offset=None, offset=None, scale=None, scale=None)</i><br/>=[0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]</td></tr></table></td></tr></table></td></tr></table>> color=red penwidth=3 rank=max shape=plaintext]\n}'

        # validate_learning_mechs(comp)

        comp.learn(
              num_trials=2,
              inputs={rel_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      rep_in: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
              # targets={rep_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          prop_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          qual_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
              #          act_out: [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}
              )
        print(comp.results)
