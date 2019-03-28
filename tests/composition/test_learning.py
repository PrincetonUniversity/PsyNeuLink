import matplotlib
# matplotlib.use('TkAgg')
import psyneulink as pnl
import numpy as np
import pytest

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

        Hebb_C.run(num_trials=5,
                   inputs=inputs_dict)
        activity = Hebb2.value

        assert np.allclose(activity, [[1.86643089, 0., 0., 1.86643089, 0., 0., 1.86643089, 0., 0.]])

class TestReinforcement:
    def test_ddm(self):
        example_DDM = pnl.DDM(function=pnl.DriftDiffusionIntegrator(noise=2.0,
                                                                    rate=1.5),
                              name='DDM')
        example_DDM.log.set_log_conditions(items=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])
        comp = pnl.Composition(name='DDM_composition')

        comp.add_node(example_DDM)

        comp.run(inputs={example_DDM: [[[1.0]]]},
                 num_trials=10,
                 execution_id='execid')

        example_DDM.log.print_entries()

        log = example_DDM.log.nparray_dictionary()['execid']
        decision_variables = log[pnl.DECISION_VARIABLE]
        time_since_start = log[pnl.RESPONSE_TIME]

        print(decision_variables)
        print(time_since_start)
    def test_rl(self):
            input_layer = pnl.TransferMechanism(size=2,
                                                name='Input Layer')

            action_selection =  pnl.DDM(input_format=pnl.ARRAY,
                                        function=pnl.DriftDiffusionAnalytical(),
                                        output_states=[pnl.SELECTED_INPUT_ARRAY],
                                        name='DDM')

            comp = pnl.Composition(name='comp')
            # comp.add_linear_processing_pathway([input_layer, action_selection])
            learned_projection = comp.add_reinforcement_learning_pathway(pathway=[input_layer, action_selection],
                                                                         learning_rate=0.05)
            learned_projection.log.set_log_conditions(items=["matrix", "mod_matrix"])

            inputs_dict = {input_layer: [[1., 1.], [2., 2.], [3., 3.]],
                           comp.target_mechanism: [[10., 20.], [30., 10.], [20., 30.]]
                           }
            print("\n\n\n\n\nRUN ---------------------------")
            comp.run(inputs=inputs_dict)
            comp.show_graph()

            learned_projection.log.print_entries()
