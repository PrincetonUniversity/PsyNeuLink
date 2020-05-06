import numpy as np
import pytest

from psyneulink.core.compositions.composition import Composition, RunError
from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.learningfunctions import BackPropagation
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.system import System
from psyneulink.core.globals.keywords import ENABLED

class TestSimpleLearningPathways:

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

    # def test_target_function_spec(self):
    #     A = TransferMechanism(name="learning-process-mech-A")
    #     B = TransferMechanism(name="learning-process-mech-B")
    #     C = TransferMechanism(name="learning-process-mech-C",
    #                           default_variable=[[0.0, 0.0]])
    #     comp = Composition()
    #     learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
    #     target = learning_pathway.target
    #     input_fct = FUNCTION()
    #     comp.learn(inputs=input_fct)
    #
    #     assert np.allclose(comp.results, ???)

class TestDivergingLearningPathways:

    def test_dict_target_spec(self):
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

    def test_dict_target_spec_with_only_one_target(self):
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

    # def test_target_function_spec(self):
    #     A = TransferMechanism(name="diverging-learning-pathways-mech-A")
    #     B = TransferMechanism(name="diverging-learning-pathways-mech-B")
    #     C = TransferMechanism(name="diverging-learning-pathways-mech-C")
    #     D = TransferMechanism(name="diverging-learning-pathways-mech-D")
    #     E = TransferMechanism(name="diverging-learning-pathways-mech-E")
    #     comp = Composition()
    #     p1 = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
    #     p1_target = p1.target
    #     p2 = comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
    #     p2_target = p2.target
    #     inputs_fct = FUNCTION()
    #     comp.learn(inputs=inputs_fct)
    #     comp.learn(inputs=inputs_fct)
    #     assert np.allclose(comp.results,???)

class TestConvergingLearningPathways:

    def test_dict_target_spec(self):
        A = TransferMechanism(name="diverging-learning-pathways-mech-A")
        B = TransferMechanism(name="diverging-learning-pathways-mech-B")
        C = TransferMechanism(name="diverging-learning-pathways-mech-C", size=2)
        D = TransferMechanism(name="diverging-learning-pathways-mech-D")
        E = TransferMechanism(name="diverging-learning-pathways-mech-E")
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

    # def test_target_function_spec(self):
    #     A = TransferMechanism(name="diverging-learning-pathways-mech-A")
    #     B = TransferMechanism(name="diverging-learning-pathways-mech-B")
    #     C = TransferMechanism(name="diverging-learning-pathways-mech-C", size=2)
    #     D = TransferMechanism(name="diverging-learning-pathways-mech-D")
    #     E = TransferMechanism(name="diverging-learning-pathways-mech-E")
    #     comp = Composition()
    #     p = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
    #     p = comp.add_backpropagation_learning_pathway(pathway=[D,E,C])
    #     target = p.target
    #     inputs_fct = FUNCTION()
    #     comp.learn(inputs=inputs_fct)
    #     comp.learn(inputs=inputs_fct)
    #     assert np.allclose(comp.results,???)


class TestInvalidTargetSpecs:

    def test_target_spec_over_nesting_of_items_in_target_value(self):
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

    def test_different_number_of_stimuli_for_targets_and_other_input_mech(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B])
        with pytest.raises(RunError) as error_text:
            comp.run(inputs={A: [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
                             p.target: [[1.0], [2.0], [3.0]]})
        assert ('The input dictionary' in str(error_text.value) and
                'contains input specifications of different lengths ({3, 4})' in str(error_text.value) and
                'The same number of inputs must be provided for each node in a Composition' in str(error_text.value))
