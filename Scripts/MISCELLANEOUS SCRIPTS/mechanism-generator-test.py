from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import Logistic, Linear
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.mechanismGenerator import mechanismGenerator
from PsyNeuLink.Globals.Keywords import *
import graphviz

mechanism1 = TransferMechanism(name='my_Transfer1',
                       default_input_value = [100],
                       function=Linear(),
                       noise=5.0,
                       time_constant = .1,

                       )


mechanism2 = TransferMechanism(name='my_Transfer2',
                       default_input_value = [0],
                       function=Linear(),
                       noise=5.0,
                       time_constant = .1,

                       )


mechanism3 = TransferMechanism(name='my_Transfer3',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=5.0,
                       time_constant = .1,

                       )


mechanism4 = TransferMechanism(name='my_Transfer4',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=5.0,
                       time_constant = .1,

                       )


mechanism5 = TransferMechanism(name='my_Transfer5',
                       default_input_value = [0],
                       function=Logistic(gain=1.0, bias=0),
                       noise=5.0,
                       time_constant = .1,

                       )
path = [mechanism1, mechanism2, mechanism3, mechanism4, mechanism5]
process1 = process(default_input_value=[100],
                 params={PATHWAY:path},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

system1 = system(processes=[process1], scheduler= mechanismGenerator(path))
system1.show_graph()
from pprint import pprint
pprint(system1.__dict__)
print(system1.execute())
