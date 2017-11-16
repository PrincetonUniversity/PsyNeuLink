from psyneulink.components.functions.Function import Linear, Logistic
from psyneulink.components.functions.Function import ModulationParam
from psyneulink.components.mechanisms.AdaptiveMechanisms.GatingMechanism.GatingMechanism import GatingMechanism
from psyneulink.components.mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Library.Subsystems.AGT.LCControlMechanism import LCControlMechanism


my_mech_1 = TransferMechanism(function=Linear, name='my_linear_mechanism')
my_mech_2 = TransferMechanism(function=Logistic, name='my_logistic_mechanism')

LC = LCControlMechanism(modulated_mechanisms=[my_mech_1, my_mech_2], name='my_LC')

LC.show()

my_gating_mechanism = GatingMechanism(gating_signals=[{'GATE_ALL': [my_mech_1,
                                                                    my_mech_2]}],
                                      modulation=ModulationParam.ADDITIVE)


my_gating_mechanism.show()