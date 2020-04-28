import psyneulink as pnl
import numpy as np

# Test matrix specification as np.array with learning enabled
import psyneulink.core.components.functions.transferfunctions


def test_matrix_spec_and_learning():
    T1 = pnl.TransferMechanism(size = 2,
                               initial_value= [[0.0,0.0]],
                               name = 'INPUT LAYER')
    T2 = pnl.TransferMechanism(size= 1,
                               function =psyneulink.core.components.functions.transferfunctions.Logistic,
                               name = 'OUTPUT LAYER')
    W = np.array([[0.1],[0.2]])

    C = pnl.Composition()
    learning_pathway = C.add_backpropagation_learning_pathway(pathway=[T1, W, T2])
    target = learning_pathway.target
    inputs = {T1:[1,0], target:[1]}
    result = C.learn(inputs=inputs, num_trials=2)
    assert np.allclose(result, [[[0.52497919]], [[0.52793236]]])

