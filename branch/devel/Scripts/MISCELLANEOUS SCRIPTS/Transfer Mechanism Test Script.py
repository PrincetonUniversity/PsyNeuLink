from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Globals.Keywords import *

my_Transfer = TransferMechanism(name='my_Transfer',
                       default_variable = [0,0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=0.0,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       # function=Linear(slope=2, intercept=10)
                       )

z = process(default_variable=[1, 1],
                 params={PATHWAY:[my_Transfer]},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

# z.execute([-1, 30])
z.prefs.verbosePref = False
z.prefs.reportOutputPref = False
my_Transfer.prefs.verbosePref = False
my_Transfer.prefs.reportOutputPref = False

for i in range(10):
    z.execute([-1, 30])
