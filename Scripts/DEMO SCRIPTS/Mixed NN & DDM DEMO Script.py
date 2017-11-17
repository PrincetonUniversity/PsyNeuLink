<<<<<<< HEAD
from psyneulink.components.mechanisms.ProcessingMechanisms.TransferMechanism import *
from psyneulink.components.Process import process
from psyneulink.components.System import system
=======
from psyneulink.components.mechanisms.processing.transfermechanism import *
from psyneulink.components.process import Process
from psyneulink.components.system import System
>>>>>>> devel

myInputLayer = TransferMechanism(name='Input Layer',
                        function=Linear(),
                        default_variable = [0,0])

myHiddenLayer = TransferMechanism(name='Hidden Layer 1',
                         function=Logistic(gain=1.0, bias=0),
                         default_variable = np.zeros((5,)))

myDDM = DDM(name='My_DDM',
            function=BogaczEtAl(drift_rate=0.5,
                                threshold=1,
                                starting_point=0.0,))

myProcess = Process(name='Neural Network DDM Process',
                    default_variable=[0, 0],
                    pathway=[myInputLayer,
                                   # RANDOM_CONNECTIVITY_MATRIX,
                                   myHiddenLayer,
                                   # FULL_CONNECTIVITY_MATRIX,
                                   myDDM])

#region
myProcess.reportOutputPref = True
myInputLayer.reportOutputPref = True
myHiddenLayer.reportOutputPref = True
myDDM.reportOutputPref = PreferenceEntry(True, PreferenceLevel.INSTANCE)
#endregion
input_set = {myInputLayer:[[-1,2],[2,3],[5,5]]}

myProcess.run(inputs=input_set,
              num_trials=2)

mySystem = System(processes=[myProcess])

mySystem.show_graph()

mySystem.reportOutputPref = True
mySystem.run(inputs=input_set)
