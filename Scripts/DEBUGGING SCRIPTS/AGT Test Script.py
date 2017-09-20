from PsyNeuLink.Components.Functions.Function import FHNIntegrator
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Library.Subsystems.AGT.AGTControlMechanism import AGTControlMechanism
from PsyNeuLink.Library.Subsystems.AGT.LCControlMechanism import LCControlMechanism

decision_mech = TransferMechanism(name='Decision_Mech')

# my_AGT = AGTControlMechanism(monitored_output_states=decision_mech,)
# my_LC = LCControlMechanism(function=(FHNIntegrator(mode=(1.0, my_AGT))),
#                            objective_mechanism=[decision_mech])

my_LC = LCControlMechanism(objective_mechanism=[decision_mech],
                    modulated_mechanisms=[decision_mech],
                    name='LC')

my_AGT = AGTControlMechanism(monitored_output_states=decision_mech,
                      control_signals=(FHNIntegrator.MODE,my_LC),
                      name='ITC')

my_main_process = process(pathway=[decision_mech], name='Decision_process')
my_AGT_process = process(pathway=[decision_mech, my_AGT], name='AGT_process')
my_LC_process = process(pathway=[decision_mech, my_LC], name='LC_process')
my_system = system(processes=[my_main_process, my_LC_process, my_AGT_process], name='my_system')

my_system.show()
my_system.show_graph()

inputs={decision_mech:[[1],[1],[1]]}
print(my_system.run(inputs=inputs))