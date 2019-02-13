import psyneulink as pnl
import numpy as np

class TestHebbian:

    def test_simple_course_example(self):
        Hebb_C = pnl.Composition()
        size = 9

        Hebb2 = pnl.RecurrentTransferMechanism(
            size=size,
            function=pnl.Linear,
            enable_learning=True,
            name='Hebb2',
        )

        Hebb_C.add_node(Hebb2)
        def print_info():
            print('\nWeight matrix:\n', Hebb2.matrix, '\nActivity: ', Hebb2.value, "\n")

        src = [1, 0, 0, 1, 0, 0, 1, 0, 0]

        inputs_dict = {Hebb2: np.array(src)}

        Hebb_C.run(num_trials=5,
                   call_after_trial=print_info,
                   inputs=inputs_dict)

    def test_simple_course_example_old(self):
        size = 9

        Hebb2 = pnl.RecurrentTransferMechanism(
            size=size,
            function=pnl.Linear,
            enable_learning=True,
            name='Hebb2',
        )
        proc = pnl.Process(pathway=[Hebb2])
        Hebb_C = pnl.System(processes=[proc])

        def print_info():
            print('\nWeight matrix:\n', Hebb2.matrix, '\nActivity: ', Hebb2.value, "\n")
            # pprint(Hebb2.recurrent_projection.mod_matrix)

        src = [1, 0, 0, 1, 0, 0, 1, 0, 0]

        inputs_dict = {Hebb2: np.array(src)}
        Hebb_C.run(num_trials=5,
                   call_after_trial=print_info,
                   inputs=inputs_dict)