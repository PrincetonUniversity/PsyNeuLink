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

    def test_target_dict_spec_single_trial_scalar_and_lists_rl(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        learning_pathway = comp.add_reinforcement_learning_pathway(pathway=[A,B])
        target = learning_pathway.target
        # Confirm that targets are ignored in run (vs learn)
        comp.run(inputs={A: 1.0,
                      target:2.0})
        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.]]])
        comp.learn(inputs={A: 1.0,
                      target:2.0})
        comp.learn(inputs={A: 1.0,
                      target:[2.0]})
        comp.learn(inputs={A: 1.0,
                      target:[[2.0]]})

        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.05]], [[1.0975]]])

    def test_target_dict_spec_single_trial_scalar_and_lists_bp(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")
        comp = Composition()
        learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        target = learning_pathway.target
        # Confirm that targets are ignored in run (vs learn)
        comp.run(inputs={A: 1.0,
                      target:2.0})
        assert np.allclose(comp.results, [[[1.]]])
        comp.learn(inputs={A: 1.0,
                      target:2.0})
        comp.learn(inputs={A: 1.0,
                      target:[2.0]})
        comp.learn(inputs={A: 1.0,
                      target:[[2.0]]})

        assert np.allclose(comp.results, [[[1.]], [[1.]], [[1.21]], [[1.40873161]]])

    def test_target_dict_spec_multi_trial_lists_rl(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B])
        target = learning_pathway.target
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                         target: [[4.0], [5.0], [6.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                         target: [[[4.0]], [[5.0]], [[6.0]]]})
        assert np.allclose(comp.results,
                            [[[1.]], [[2.6]], [[5.34]],
                             [[1.978]], [[4.3604]], [[6.92436]]])

    def test_target_dict_spec_multi_trial_lists_bp(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C",
                              default_variable=[[0.0, 0.0]])
        comp = Composition()
        learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        target = learning_pathway.target
        comp.learn(inputs={A: 1.0,
                      target:[2.0, 3.0]})
        comp.learn(inputs={A: 1.0,
                      target:[[2.0, 3.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                         target: [[3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]})
        comp.learn(inputs={A: [1.0, 2.0, 3.0],
                         target: [[[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]})
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
        p1_target = p1.target
        p2 = comp.add_backpropagation_learning_pathway(pathway=[A,D,E])
        p2_target = p2.target
        comp.learn(inputs={A: 1.0,
                         p1_target: 2.0,
                         p2_target: 4.0
                         })
        comp.learn(inputs={A: 2.0,
                         p1_target: 2.0,
                         p2_target: 4.0
                         })
        assert np.allclose(comp.results,[[[1.], [1.]], [[2.42], [3.38]]])

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
        p = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        p = comp.add_backpropagation_learning_pathway(pathway=[D,E,C])
        target = p.target
        comp.learn(inputs={A: 1.0,
                         D: 2.0,
                         target: [3.0, 4.0]
                         })
        comp.learn(inputs={A: 5.0,
                         D: 6.0,
                         target: [7.0, 8.0]
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
        learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B,C])
        target = learning_pathway.target
        # Elicit test with run
        with pytest.raises(RunError) as error_text:
            comp.run(inputs={A: [1.0, 2.0, 3.0],
                             target: [[[3.0], [4.0]], [[5.0], [6.0]], [[7.0], [8.0]]]})
        assert ("Input stimulus" in str(error_text.value) and
                "for Target is incompatible with its external_input_values" in str(error_text.value))
        # Elicit test with learn
        with pytest.raises(RunError) as error_text:
            comp.learn(inputs={A: [1.0, 2.0, 3.0],
                             target: [[[3.0], [4.0]], [[5.0], [6.0]], [[7.0], [8.0]]]})
        assert ("Input stimulus" in str(error_text.value) and
                "for Target is incompatible with its external_input_values" in str(error_text.value))

    def test_3_targets_4_inputs(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        comp = Composition()
        learning_pathway = comp.add_backpropagation_learning_pathway(pathway=[A,B])
        target = learning_pathway.target
        with pytest.raises(RunError) as error_text:
            comp.run(inputs={A: [[[1.0]], [[2.0]], [[3.0]], [[4.0]]],
                             target: [[1.0], [2.0], [3.0]]})
        assert ('The input dictionary' in str(error_text.value) and
                'contains input specifications of different lengths ({3, 4})' in str(error_text.value) and
                'The same number of inputs must be provided for each node in a Composition' in str(error_text.value))

    def test_2_target_mechanisms_1_dict_entry(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     learning=ENABLED)
        LP2 = Process(name="learning-process2",
                     pathway=[A, C],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP, LP2],
                   )
        with pytest.raises(RunError) as error_text:

            S.run(inputs={A: [[[1.0]]]},
                  targets={B: [[1.0]]})

        assert 'missing from specification of targets for run' in str(error_text.value)

    def test_1_target_mechanisms_2_dict_entries(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")

        LP = Process(name="learning-process",
                     pathway=[A, B, C],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP],
                   )

        with pytest.raises(RunError) as error_text:
            S.run(inputs={A: [[[1.0]]]},
                  targets={B: [[1.0]],
                           C: [[1.0]]})

        assert 'does not project to a target Mechanism in' in str(error_text.value)

    def test_2_target_mechanisms_list_spec(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     learning=ENABLED)
        LP2 = Process(name="learning-process2",
                     pathway=[A, C],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP, LP2],
                   )
        with pytest.raises(RunError) as error_text:

            S.run(inputs={A: [[[1.0]]]},
                  targets=[[1.0]])

        assert 'Target values for' in str(error_text.value) and \
               'must be specified in a dictionary' in str(error_text.value)

    def test_2_target_mechanisms_fn_spec(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")
        C = TransferMechanism(name="learning-process-mech-C")

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     learning=ENABLED)
        LP2 = Process(name="learning-process2",
                     pathway=[A, C],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP, LP2],
                   )

        def target_function():
            val_1 = NormalDist(mean=3.0)()
            val_2 = NormalDist(mean=3.0)()
            return [val_1, val_2]

        with pytest.raises(RunError) as error_text:

            S.run(inputs={A: [[[1.0]]]},
                  targets=target_function)

        assert 'Target values for' in str(error_text.value) and \
               'must be specified in a dictionary' in str(error_text.value)
