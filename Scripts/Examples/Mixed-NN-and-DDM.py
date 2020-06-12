import numpy as np
import psyneulink as pnl
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.transferfunctions

myInputLayer = pnl.TransferMechanism(
    name='Input Layer',
    function=psyneulink.core.components.functions.transferfunctions.Linear(),
    default_variable=[0, 0]
)

myHiddenLayer = pnl.TransferMechanism(
    name='Hidden Layer 1',
    function=psyneulink.core.components.functions.transferfunctions.Logistic(gain=1.0, x_0=0),
    default_variable=np.zeros((5,))
)

myDDM = pnl.DDM(
    name='My_DDM',
    function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
        drift_rate=0.5,
        threshold=1,
        starting_point=0.0
    )
)

comp = pnl.Composition(
    name='Neural Network DDM Process',
    pathways=[
        [myInputLayer,
         pnl.get_matrix(pnl.RANDOM_CONNECTIVITY_MATRIX, 2, 5),
         myHiddenLayer,
         pnl.FULL_CONNECTIVITY_MATRIX,
         myDDM]
    ]
)

comp.reportOutputPref = True
myInputLayer.reportOutputPref = True
myHiddenLayer.reportOutputPref = True
myDDM.reportOutputPref = pnl.PreferenceEntry(True, pnl.PreferenceLevel.INSTANCE)

comp.run([[-1, 2], [2, 3], [5, 5]])
