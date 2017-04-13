from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import Logistic, Linear
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.mechanismGenerator import mechanismGenerator
from PsyNeuLink.Globals.Keywords import *
import graphviz

mechanism1 = TransferMechanism(name='my_Transfer1',
                       default_input_value = [0],
                       function=Linear(),
                       time_constant =0.0

                       )
print(mechanism1.execute(100)," = mech1 independently")

mechanism2 = TransferMechanism(name='my_Transfer2',
                               default_input_value=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )

print(mechanism2.execute(200), " = mech2 independently")

mechanism3 = TransferMechanism(name='my_Transfer3',
                               default_input_value=[0],
                               function=Linear(slope=3.0),
                               time_constant=0.0

                               )

print(mechanism3.execute(600), " = mech3 independently")

mechanism4 = TransferMechanism(name='my_Transfer4',
                               default_input_value=[0],
                               function=Linear(slope=3.0),
                               time_constant=0.0

                               )
print(mechanism4.execute(1800), " = mech4 independently")

mechanism5 = TransferMechanism(name='my_Transfer5',
                               default_input_value=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )

print(mechanism5.execute(3600), " = mech5 independently")

path = [mechanism1, mechanism2, mechanism3, mechanism4, mechanism5]
process1 = process(default_input_value=[100],
                 params={PATHWAY:path},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

system1 = system(processes=[process1], scheduler= mechanismGenerator(path))

print(system1.execute([[100]]))
