import numpy as np

from psyneulink.components.functions.function import Linear, Logistic
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import Process
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.system import System
from psyneulink.globals.keywords import FULL_CONNECTIVITY_MATRIX, LEARNING, LEARNING_PROJECTION, ENABLED
from psyneulink.globals.preferences.componentpreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.library.mechanisms.processing.objective.comparatormechanism import MSE

class TestSimpleLearningPathway:

    def test_dict_target_spec(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     # target=[3.0],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP],
                   # targets={B: [4.0]}
                   )

        S.run(inputs={A: 1.0},
              targets={B: 2.0})

        S.run(inputs={A: 1.0},
              targets={B: [2.0]})

        S.run(inputs={A: 1.0},
              targets={B: [[2.0]]})

    def test_dict_target_spec_length2(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B",
                              default_variable=[[0.0, 0.0]])

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     # target=[3.0],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP],
                   # targets={B: [4.0]}
                   )

        S.run(inputs={A: 1.0},
              targets={B: [2.0, 3.0]})

        S.run(inputs={A: 1.0},
              targets={B: [[2.0, 3.0]]})

    def test_list_target_spec(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B")

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP])

        S.run(inputs={A: 1.0},
              targets=2.0)

        S.run(inputs={A: 1.0},
              targets=[2.0])

        S.run(inputs={A: 1.0},
              targets=[[2.0]])

    def test_list_target_spec_length2(self):
        A = TransferMechanism(name="learning-process-mech-A")
        B = TransferMechanism(name="learning-process-mech-B",
                              default_variable=[[0.0, 0.0]])

        LP = Process(name="learning-process",
                     pathway=[A, B],
                     # target=[3.0],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[LP],
                   # targets={B: [4.0]}
                   )

        S.run(inputs={A: 1.0},
              targets=[2.0, 3.0])

        S.run(inputs={A: 1.0},
              targets=[[2.0, 3.0]])
class TestMultilayerLearning:

    def test_dict_target_spec(self):
        A = TransferMechanism(name="multilayer-mech-A")
        B = TransferMechanism(name="multilayer-mech-B")
        C = TransferMechanism(name="multilayer-mech-C")
        P = Process(name="multilayer-process",
                     pathway=[A, B, C],
                     # target=[3.0],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[P]
                   )

        S.run(inputs={A: 1.0},
              targets={C: 2.0})

        S.run(inputs={A: 1.0},
              targets={C: [2.0]})

        S.run(inputs={A: 1.0},
              targets={C: [[2.0]]})

    def test_dict_target_spec_length2(self):
        A = TransferMechanism(name="multilayer-mech-A")
        B = TransferMechanism(name="multilayer-mech-B")
        C = TransferMechanism(name="multilayer-mech-C",
                              default_variable=[[0.0, 0.0]])
        P = Process(name="multilayer-process",
                     pathway=[A, B, C],
                     # target=[3.0],
                     learning=ENABLED)

        S = System(name="learning-system",
                   processes=[P]
                   )

        S.run(inputs={A: 1.0},
              targets={C: [2.0, 3.0]})

        S.run(inputs={A: 1.0},
              targets={C: [[2.0, 3.0]]})