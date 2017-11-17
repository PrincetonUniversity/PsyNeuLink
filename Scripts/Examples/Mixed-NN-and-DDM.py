import numpy as np
import psyneulink as pnl

myInputLayer = pnl.TransferMechanism(
    name='Input Layer',
    function=pnl.Linear(),
    default_variable=[0, 0]
)

myHiddenLayer = pnl.TransferMechanism(
    name='Hidden Layer 1',
    function=pnl.Logistic(gain=1.0, bias=0),
    default_variable=np.zeros((5,))
)

myDDM = pnl.DDM(
    name='My_DDM',
    function=pnl.BogaczEtAl(
        drift_rate=0.5,
        threshold=1,
        starting_point=0.0
    )
)

myProcess = pnl.Process(
    name='Neural Network DDM Process',
    default_variable=[0, 0],
    pathway=[
        myInputLayer,
        pnl.get_matrix(pnl.RANDOM_CONNECTIVITY_MATRIX, 2, 5),
        myHiddenLayer,
        pnl.FULL_CONNECTIVITY_MATRIX,
        myDDM
    ]
)

myProcess.reportOutputPref = True
myInputLayer.reportOutputPref = True
myHiddenLayer.reportOutputPref = True
myDDM.reportOutputPref = pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)

pnl.run(myProcess, [[-1, 2], [2, 3], [5, 5]])
