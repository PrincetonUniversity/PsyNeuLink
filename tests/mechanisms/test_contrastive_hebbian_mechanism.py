import psyneulink as pnl
import numpy as np

class TestContrastiveHebbian:

    def test_contrastive_hebbian_learning(self):
        m = pnl.ContrastiveHebbianMechanism(enable_learning=True,
                                            size=2,
                                            )

        s = pnl.sys(m)
        m.reinitialize_when = pnl.Never()
        output = s.run(inputs=[2, 2], num_trials=4)

        expected_output = [[np.array([2., 2.]), np.array([2., 2.])], [np.array([2.4, 2.4]), np.array([2.4, 2.4])],
                           [np.array([5.5712, 5.5712]), np.array([5.5712, 5.5712])],
                           [np.array([16.93596594, 16.93596594]), np.array([16.93596594, 16.93596594])]]

        assert np.allclose(output, expected_output)