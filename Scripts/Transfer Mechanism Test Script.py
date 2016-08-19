from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

my_Transfer = Transfer(name='my_Transfer',
                       params={kwExecuteMethod:kwLogistic,
                               # kwExecuteMethodParams:{kwTransfer_Gain:(1, kwControlSignal)}},
                               kwExecuteMethodParams:{kwTransfer_Gain: 1}},
                       default_input_value = [0,0])

z = Process_Base(default_input_value=[0, 0],
                 params={kwConfiguration:[my_Transfer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

z.execute([-1, 30])
