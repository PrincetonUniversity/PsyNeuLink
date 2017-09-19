from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.System import System_Base, system
from PsyNeuLink.Components.Process import Process_Base, process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms.DDM import DDM
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.AGT.LCMechanism import LCMechanism
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.AGT.ITCMechanism import ITCMechanism
from PsyNeuLink.Components.Functions.Function import FHNIntegrator, UtilityIntegrator

my_mech_1 = TransferMechanism(name='Processing_Mech')

# my_ITC = ITCMechanism(monitored_output_states=my_mech,)
# my_LC = LCMechanism(function=(FHNIntegrator(mode=(1.0, my_ITC))),
#                     objective_mechanism=[my_mech])

my_LC = LCMechanism(objective_mechanism=[my_mech_1],
                    modulated_mechanisms=[my_mech_1],
                    name='LC')
my_ITC = ITCMechanism(monitored_output_states=my_mech_1,
                      control_signals=(FHNIntegrator.MODE,my_LC),
                      name='ITC')

# my_main_process = process(pathway=[my_mech_1], name='Main_process')
# my_LC_process = process(pathway=[my_LC], name='LC_process')
# my_ITC_process = process(pathway=[my_ITC], name='ITC_process')
# my_system = system(processes=[my_main_process, my_LC_process, my_ITC_process], name='my_system')

my_main_process = process(pathway=[my_mech_1], name='Main_process')
my_LC_process = process(pathway=[my_LC], name='LC_process')
my_ITC_process = process(pathway=[my_ITC], name='ITC_process')
my_system = system(processes=[my_main_process, my_LC_process, my_ITC_process], name='my_system')

my_system.show()
my_system.show_graph(show_control=True)


inputs={my_mech_1:[0]}
print(my_system.run(inputs=inputs))






