from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

my_Transfer = Transfer(name='my_Transfer',
                       params={kwExecuteMethod:kwLogistic},
                       default_input_value = [0,0])

z = Process_Base(default_input_value=[0, 0],
                 params={kwConfiguration:[my_Transfer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

z.execute([-1, 30])
