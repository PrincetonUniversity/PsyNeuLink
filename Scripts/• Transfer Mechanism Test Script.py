from Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer

from Functions.Mechanisms.ProcessingMechanisms.DDM import *
from Functions.Process import Process_Base
from Globals.Keywords import *

my_Transfer = Transfer(name='my_Transfer',
                       params={kwExecuteMethod:kwLogistic},
                       default_input_value = [0,0])

z = Process_Base(default_input_value=[0, 0],
                 params={kwConfiguration:[my_Transfer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

z.execute([-1, 30])
