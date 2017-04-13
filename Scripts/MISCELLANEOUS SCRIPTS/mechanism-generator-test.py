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
print(mechanism1.execute(100)," = value of executing mechanism1 independently with input of 100")

mechanism2 = TransferMechanism(name='my_Transfer2',
                               default_input_value=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )

print(mechanism2.execute(100)," = value of executing mechanism2 independently with input of 100")

mechanism3 = TransferMechanism(name='my_Transfer3',
                               default_input_value=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )
print(mechanism3.execute(200)," = value of executing mechanism3 independently with input of 200")

mechanism4 = TransferMechanism(name='my_Transfer4',
                               default_input_value=[0],
                               function=Linear(slope=2.0),
                               time_constant=0.0

                               )
print(mechanism4.execute(400)," = value of executing mechanism4 independently with input of 400")


mechanism5 = TransferMechanism(name='my_Transfer5',
                               default_input_value=[0],
                               function=Linear(),
                               time_constant=0.0

                               )

print(mechanism5.execute(800)," = value of executing mechanism5 independently with input of 800")

path = [mechanism1, mechanism2, mechanism3, mechanism4, mechanism5]
process1 = process(default_input_value=[100],
                 params={PATHWAY:path},
                 prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE),
                        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.INSTANCE)})

system1 = system(processes=[process1], scheduler= mechanismGenerator(path))

print(system1.execute([[100]]), " = value of executing system with input of 100")
