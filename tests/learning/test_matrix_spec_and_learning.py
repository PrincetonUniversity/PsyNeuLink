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
    P = pnl.Process(pathway=[T1, W, T2], learning = pnl.ENABLED, target= 1.0)
    S = pnl.System(processes = [P])
    IN = {T1:[1,0]}
    TARG = {T2:[[1]]}
    S.run(IN, targets=TARG, num_trials= 2, learning = True)
