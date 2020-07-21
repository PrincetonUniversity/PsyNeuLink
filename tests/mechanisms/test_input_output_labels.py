import numpy as np
import pytest

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.globals.keywords import ENABLED, INPUT_LABELS_DICT, OUTPUT_LABELS_DICT

# FIX 5/8/20 ELIMINATE SYSTEM [JDC] -- CONVERTED TO COMPOSITION, BUT REQUIRE REFACTORING OF LABEL HANDLING
# class TestMechanismInputLabels:
#     def test_dict_of_floats(self):
#         input_labels_dict = {"red": 1,
#                              "green":0}
#
#         M = ProcessingMechanism(params={INPUT_LABELS_DICT: input_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_input_labels = []
#
#         def call_after_trial():
#             store_input_labels.append(M.get_input_labels(C))
#
#         C.run(inputs={M:['red', 'green', 'green', 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.]], [[0.]], [[0.]], [[1.]]])
#         assert store_input_labels == [['red'], ['green'], ['green'], ['red']]
#         C.run(inputs={M:[1, 'green', 0, 'red']})
#         assert np.allclose(C.results, [[[1.]], [[0.]], [[0.]], [[1.]], [[1.]], [[0.]], [[0.]], [[1.]]])
#
#     def test_dict_of_arrays(self):
#         input_labels_dict = {"red": [1, 0, 0],
#                              "green": [0, 1, 0],
#                              "blue": [0, 0, 1]}
#         M = ProcessingMechanism(default_variable=[[0, 0, 0]],
#                                 params={INPUT_LABELS_DICT: input_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_input_labels = []
#
#         def call_after_trial():
#             store_input_labels.append(M.get_input_labels(C))
#
#         C.run(inputs={M:['red', 'green', 'blue', 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]]])
#         assert store_input_labels == [['red'], ['green'], ['blue'], ['red']]
#
#         C.run(inputs={M:'red'})
#         assert np.allclose(C.results, [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]], [[1, 0, 0]]])
#
#         C.run(inputs={M:['red']})
#         assert np.allclose(C.results, [[[1, 0, 0]], [[0, 1, 0]], [[0, 0, 1]], [[1, 0, 0]], [[1, 0, 0]], [[1, 0, 0]]])
#
#     # def test_dict_of_arrays_2_input_ports(self):
#     #     input_labels_dict = {"red": [0],
#     #                          "green": [1]}
#     #
#     #     M = ProcessingMechanism(default_variable=[[0], [0]],
#     #                             params={INPUT_LABELS_DICT: input_labels_dict})
#     #     P = Process(pathway=[M])
#     #     S = System(processes=[P])
#     #
#     #     M_output = []
#     #     store_input_labels = []
#     #
#     #     def call_after_trial():
#     #         M_output.append(M.value)
#     #         store_input_labels.append(M.get_input_labels(S))
#     #
#     #     S.run(inputs=[['red', 'green'], ['green', 'red']],
#     #           call_after_trial=call_after_trial)
#     #
#     #     assert np.allclose(M_output, [[[0], [1]], [[1], [0]]])
#     #     assert store_input_labels == [['red', 'green'], ['green', 'red']]
#     #
#     #     S.run(inputs=[[[0], 'green'], [[1], 'red']],
#     #           call_after_trial=call_after_trial)
#     #
#     #     assert np.allclose(M_output, [[[0], [1]], [[1], [0]], [[0], [1]], [[1], [0]]])
#
#     # no longer valid:
#     # def test_dict_of_2d_arrays(self):
#     #     input_labels_dict = {"red": [[1, 0], [1, 0]],
#     #                          "green": [[0, 1], [0, 1]],
#     #                          "blue": [[0, 1], [1, 0]]}
#     #     M = TransferMechanism(default_variable=[[0, 0], [0, 0]],
#     #                             params={INPUT_LABELS_DICT: input_labels_dict})
#     #     P = Process(pathway=[M])
#     #     S = System(processes=[P])
#     #
#     #     store_input_labels = []
#     #
#     #     def call_after_trial():
#     #         store_input_labels.append(M.get_input_labels(S))
#     #
#     #     S.run(inputs=['red', 'green', 'blue'],
#     #           call_after_trial=call_after_trial)
#     #     assert np.allclose(S.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])
#     #     assert store_input_labels == ['red', 'green', 'blue']
#     #
#     #     S.run(inputs='red')
#     #     assert np.allclose(S.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [1, 0]]])
#
#     def test_dict_of_dicts_1_input_port(self):
#         input_labels_dict = {0: {"red": [1, 0],
#                                  "green": [0, 1]}}
#
#         M = TransferMechanism(default_variable=[[0, 0]],
#                               params={INPUT_LABELS_DICT: input_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_input_labels = []
#
#         def call_after_trial():
#             store_input_labels.append(M.get_input_labels(C))
#
#         C.run(inputs={M:[['red'], ['green'], ['green']]},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1, 0]], [[0, 1]], [[0, 1]]])
#         assert [['red'], ['green'], ['green']] == store_input_labels
#
#         C.run(inputs={M:'red'})
#         assert np.allclose(C.results, [[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]]])
#
#         C.run(inputs={M:['red']})
#         assert np.allclose(C.results, [[[1, 0]], [[0, 1]], [[0, 1]], [[1, 0]], [[1, 0]]])
#
#     def test_dict_of_dicts(self):
#         input_labels_dict = {0: {"red": [1, 0],
#                                  "green": [0, 1]},
#                              1: {"red": [0, 1],
#                                  "green": [1, 0]}}
#
#
#         M = TransferMechanism(default_variable=[[0, 0], [0, 0]],
#                               params={INPUT_LABELS_DICT: input_labels_dict})
#         C = Composition(pathways=[M])
#
#         C.run(inputs={M:[['red', 'green'], ['green', 'red'], ['green', 'green']]})
#         assert np.allclose(C.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])
#
#         C.run(inputs={M:[['red', [1, 0]], ['green', 'red'], [[0,1], 'green']]})
#         assert np.allclose(C.results, [[[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [1, 0]], [[0, 1], [0, 1]], [[0, 1], [1, 0]]])
#
#     def test_3_input_ports_2_label_dicts(self):
#         input_labels_dict = {0: {"red": [1, 0],
#                                  "green": [0, 1]},
#                              2: {"red": [0, 1],
#                                  "green": [1, 0]}}
#
#
#         M = TransferMechanism(default_variable=[[0, 0], [0, 0], [0, 0]],
#                               params={INPUT_LABELS_DICT: input_labels_dict})
#         C = Composition(pathways=[M])
#
#         C.run(inputs={M:[['red', [0, 0], 'green'], ['green', [1, 1], 'red'], ['green', [2, 2], 'green']]})
#         assert np.allclose(C.results, [[[1, 0], [0, 0], [1, 0]], [[0, 1], [1, 1], [0, 1]], [[0, 1], [2, 2], [1, 0]]])
#
#         C.run(inputs={M:[['red', [0, 0], [1, 0]], ['green', [1, 1], 'red'], [[0,1], [2, 2], 'green']]})
#         assert np.allclose(C.results, [[[1, 0], [0, 0], [1, 0]], [[0, 1], [1, 1], [0, 1]], [[0, 1], [2, 2], [1, 0]], [[1, 0], [0, 0], [1, 0]], [[0, 1], [1, 1], [0, 1]], [[0, 1], [2, 2], [1, 0]]])
#
# class TestMechanismTargetLabels:
#     def test_dict_of_floats(self):
#         input_labels_dict_M1 = {"red": 1,
#                                 "green": 0}
#         output_labels_dict_M2 = {"red": 0,
#                                 "green": 1}
#         M1 = ProcessingMechanism(params={INPUT_LABELS_DICT: input_labels_dict_M1})
#         M2 = ProcessingMechanism(params={OUTPUT_LABELS_DICT: output_labels_dict_M2})
#         C = Composition()
#         learning_pathway = C.add_backpropagation_learning_pathway(pathway=[M1, M2], learning_rate=0.25)
#         target = learning_pathway.target
#         learned_matrix = []
#
#         def record_matrix_after_trial():
#             learned_matrix.append(M2.path_afferents[0].get_mod_matrix(C))
#
#         C.learn(inputs={M1: ['red', 'green', 'green', 'red'],
#                         target:['red', 'green', 'green', 'red']},
#                 call_after_trial=record_matrix_after_trial)
#
#         assert np.allclose(C.results, [[[1.]], [[0.]], [[0.]], [[0.75]]])
#         assert np.allclose(learned_matrix, [[[0.75]], [[0.75]], [[0.75]], [[0.5625]]])
#
#     def test_dict_of_arrays(self):
#         input_labels_dict_M1 = {"red": [1, 1],
#                                 "green": [0, 0]}
#         output_labels_dict_M2 = {"red": [0, 0],
#                                 "green": [1, 1]}
#         M1 = ProcessingMechanism(size=2,
#                                  params={INPUT_LABELS_DICT: input_labels_dict_M1})
#         M2 = ProcessingMechanism(size=2,
#                                  params={OUTPUT_LABELS_DICT: output_labels_dict_M2})
#         C = Composition()
#         learning_pathway = C.add_backpropagation_learning_pathway(pathway=[M1, M2], learning_rate=0.25)
#         target = learning_pathway.target
#         learned_matrix = []
#         count = []
#
#         def record_matrix_after_trial():
#             learned_matrix.append(M2.path_afferents[0].get_mod_matrix(C))
#             count.append(1)
#
#         C.learn(inputs={M1: ['red', 'green', 'green', 'red'],
#                         target: ['red', 'green', 'green', 'red']},
#                 call_after_trial=record_matrix_after_trial)
#         assert np.allclose(C.results, [[[1, 1]], [[0., 0.]], [[0., 0.]], [[0.5, 0.5]]])
#         assert np.allclose(learned_matrix, [np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.625, -0.375], [-0.375,  0.625]])])
#
#     def test_dict_of_subdicts(self):
#         input_labels_dict_M1 = {"red": [1, 1],
#                                 "green": [0, 0]}
#         output_labels_dict_M2 = {0: {"red": [0, 0],
#                                        "green": [1, 1]}
#                                  }
#         M1 = ProcessingMechanism(size=2,
#                                  params={INPUT_LABELS_DICT: input_labels_dict_M1})
#         M2 = ProcessingMechanism(size=2,
#                                  params={OUTPUT_LABELS_DICT: output_labels_dict_M2})
#         C = Composition()
#
#         learning_pathway = C.add_backpropagation_learning_pathway(pathway=[M1, M2], learning_rate=0.25)
#         target = learning_pathway.target
#         learned_matrix = []
#         count = []
#
#         def record_matrix_after_trial():
#             learned_matrix.append(M2.path_afferents[0].get_mod_matrix(C))
#             count.append(1)
#
#         C.learn(inputs={M1: ['red', 'green', 'green', 'red'],
#                         target: ['red', 'green', 'green', 'red']},
#                 call_after_trial=record_matrix_after_trial)
#         assert np.allclose(C.results, [[[1, 1]], [[0., 0.]], [[0., 0.]], [[0.5, 0.5]]])
#         assert np.allclose(learned_matrix, [np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.75, -0.25], [-0.25,  0.75]]),
#                                             np.array([[0.625, -0.375], [-0.375,  0.625]])])
#
#
# class TestMechanismOutputLabels:
#
#     def test_dict_of_floats(self):
#         input_labels_dict = {"red": 1,
#                               "green": 0}
#         output_labels_dict = {"red": 1,
#                              "green":0}
#         M = ProcessingMechanism(params={INPUT_LABELS_DICT: input_labels_dict,
#                                         OUTPUT_LABELS_DICT: output_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_output_labels = []
#
#         def call_after_trial():
#             store_output_labels.append(M.get_output_labels(C))
#
#         C.run(inputs={M: ['red', 'green', 'green', 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.]], [[0.]], [[0.]], [[1.]]])
#         assert store_output_labels == [['red'], ['green'], ['green'], ['red']]
#
#         store_output_labels = []
#         C.run(inputs={M: [1, 'green', 0, 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.]], [[0.]], [[0.]], [[1.]], [[1.]], [[0.]], [[0.]], [[1.]]])
#         assert store_output_labels == [['red'], ['green'], ['green'], ['red']]
#
#     def test_dict_of_arrays(self):
#         input_labels_dict = {"red": [1.0, 0.0],
#                              "green": [0.0, 1.0]}
#         output_labels_dict = {"red": [1.0, 0.0],
#                               "green": [0.0, 1.0]}
#         M = ProcessingMechanism(size=2,
#                                 params={INPUT_LABELS_DICT: input_labels_dict,
#                                         OUTPUT_LABELS_DICT: output_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_output_labels = []
#
#         def call_after_trial():
#             store_output_labels.append(M.get_output_labels(S))
#
#         C.run(inputs={M:['red', 'green', 'green', 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[1.0, 0.0]]])
#         assert store_output_labels == [['red'], ['green'], ['green'], ['red']]
#
#         store_output_labels = []
#         C.run(inputs={M: [[1.0, 0.0], 'green', [0.0, 1.0], 'red']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], [[0.0, 1.0]], [[1.0, 0.0]]])
#         assert store_output_labels == [['red'], ['green'], ['green'], ['red']]
#         # S.show_graph(show_mechanism_structure="labels")
#
#     def test_not_all_output_port_values_have_label(self):
#         input_labels_dict = {"red": [1.0, 0.0],
#                              "green": [0.0, 1.0],
#                              "blue": [2.0, 2.0]}
#         output_labels_dict = {"red": [1.0, 0.0],
#                               "green": [0.0, 1.0]}
#         M = ProcessingMechanism(size=2,
#                                 params={INPUT_LABELS_DICT: input_labels_dict,
#                                         OUTPUT_LABELS_DICT: output_labels_dict})
#         C = Composition(pathways=[M])
#
#         store_output_labels = []
#
#         def call_after_trial():
#             store_output_labels.append(M.get_output_labels(C))
#
#         C.run(inputs={M: ['red', 'blue', 'green', 'blue']},
#               call_after_trial=call_after_trial)
#         assert np.allclose(C.results, [[[1.0, 0.0]], [[2.0, 2.0]], [[0.0, 1.0]], [[2.0, 2.0]]])
#
#         assert store_output_labels[0] == ['red']
#         assert np.allclose(store_output_labels[1], [[2.0, 2.0]])
#         assert store_output_labels[2] == ['green']
#         assert np.allclose(store_output_labels[3], [[2.0, 2.0]])
