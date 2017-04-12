from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import Logistic
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.mechanismGenerator import mechanismGenerator
from PsyNeuLink.Globals.Keywords import *

mechanism1 = TransferMechanism(name='my_Transfer1',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=NormalDist(mean=1.0).function,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       )


mechanism2 = TransferMechanism(name='my_Transfer2',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=NormalDist(mean=2.0).function,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       )


mechanism3 = TransferMechanism(name='my_Transfer3',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=NormalDist(mean=-1.0).function,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       )


mechanism4 = TransferMechanism(name='my_Transfer4',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=NormalDist(mean=-0.5).function,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       )


mechanism5 = TransferMechanism(name='my_Transfer5',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=NormalDist(mean=0.5).function,
                       time_constant = .1,
                       time_scale=TimeScale.TIME_STEP
                       )
path = [mechanism1, mechanism2, mechanism3, mechanism4, mechanism5]
process1 = process(default_input_value=[1],
                 params={PATHWAY:path},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

system1 = system(processes=[process1], scheduler= mechanismGenerator(path))

print(system1.execute())
