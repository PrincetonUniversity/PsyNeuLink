from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Globals.Keywords import *

my_Transfer = Transfer(name='my_Transfer',
                       default_input_value = [0,0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=0.0,
                       rate = .1,
                       time_scale=TimeScale.REAL_TIME
                       # function=Linear(slope=2, intercept=10)
                       )

z = Process_Base(default_input_value=[1, 1],
                 params={CONFIGURATION:[my_Transfer]},
                 prefs={kpVerbosePref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

# z.execute([-1, 30])

for i in range(10):
    z.execute([-1, 30])
