import psyneulink as pnl
import numpy as np

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
