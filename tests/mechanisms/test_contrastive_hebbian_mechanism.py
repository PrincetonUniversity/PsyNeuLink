import psyneulink as pnl
import numpy as np

class TestContrastiveHebbian:

    # def test_contrastive_hebbian_learning(self):
    #
    #     m = pnl.ContrastiveHebbianMechanism(enable_learning=True,
    #                                         size=2,
    #                                         )
    #
    #     s = pnl.sys(m)
    #     m.reinitialize_when = pnl.Never()
    #     output = s.run(inputs=[2, 2], num_trials=4)
    #
    #     expected_output = [[np.array([2., 2.]), np.array([2., 2.])], [np.array([2.4, 2.4]), np.array([2.4, 2.4])],
    #                        [np.array([5.5712, 5.5712]), np.array([5.5712, 5.5712])],
    #                        [np.array([16.93596594, 16.93596594]), np.array([16.93596594, 16.93596594])]]
    #     # print(m.output_states)
    #     # print(m.output_values)
    #     # print("sender =", m.recurrent_projection.sender)
    #     assert np.allclose(output, expected_output)

    def test_scheduled_contrastive_hebbian(self):
        o = pnl.TransferMechanism()
        m = pnl.ContrastiveHebbianMechanism(
            enable_learning=False,
            auto=0,
            hetero=-1,
            size=2,
        )

        s = pnl.sys(m, o)
        ms = pnl.Scheduler(system=s)
        ms.add_condition(o, pnl.WhenFinished(m))
        s.scheduler_processing = ms
        # m.reinitialize_when=pnl.Never()
        print('matrix:\n', m.afferents[1].matrix)
        results = s.run(inputs=[2, 2], num_trials=4)
        print(results)
