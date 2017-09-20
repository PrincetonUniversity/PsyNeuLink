from PsyNeuLink.Components.Functions.Function import FHNIntegrator
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Library.Subsystems.AGT.AGTControlMechanism import AGTControlMechanism
from PsyNeuLink.Library.Subsystems.AGT.LCControlMechanism import LCControlMechanism

my_mech_1 = TransferMechanism(name='Processing_Mech')

# my_AGT = AGTControlMechanism(monitored_output_states=my_mech,)
# my_LC = LCControlMechanism(function=(FHNIntegrator(mode=(1.0, my_AGT))),
#                     objective_mechanism=[my_mech])

my_LC = LCControlMechanism(objective_mechanism=[my_mech_1],
                    modulated_mechanisms=[my_mech_1],
                    name='LC')
my_AGT = AGTControlMechanism(monitored_output_states=my_mech_1,
                      control_signals=(FHNIntegrator.MODE,my_LC),
                      name='ITC')

# my_main_process = process(pathway=[my_mech_1], name='Main_process')
# my_LC_process = process(pathway=[my_LC], name='LC_process')
# my_AGT_process = process(pathway=[my_AGT], name='AGT_process')
# my_system = system(processes=[my_main_process, my_LC_process, my_AGT_process], name='my_system')

my_main_process = process(pathway=[my_mech_1], name='Main_process')
my_AGT_process = process(pathway=[my_mech_1, my_AGT], name='AGT_process')
my_LC_process = process(pathway=[my_mech_1, my_LC], name='LC_process')
my_system = system(processes=[my_main_process, my_LC_process, my_AGT_process], name='my_system')

my_system.show()
my_system.show_graph()
# my_system.show_graph(show_control=True)


inputs={my_mech_1:[[1],[1],[1]]}
print(my_system.run(inputs=inputs))

# my_mech_2 = TransferMechanism(name='mech_2')
# my_mech_3 = TransferMechanism(name='mech_3')
#
# process_1 = process(pathway=[my_mech_1])
# process_2 = process(pathway=[my_mech_2])
# process_3 = process(pathway=[my_mech_3])
# system_2 = system(processes=[process_1, process_2, process_3])
#
# TEST = True


