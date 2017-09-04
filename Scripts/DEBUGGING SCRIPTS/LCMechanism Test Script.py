from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanism.ControlMechanism import ControlMechanism_Base
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.LCMechanism import LCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.GatingMechanism.GatingMechanism import GatingMechanism
from PsyNeuLink.Components.Functions.Function import ModulationParam
from PsyNeuLink.Components.Functions.Function import Linear, Logistic

from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *

my_mech_1 = TransferMechanism(function=Linear, name='my_linear_mechanism')
my_mech_2 = TransferMechanism(function=Logistic, name='my_logistic_mechanism')

LC = LCMechanism(modulated_mechanisms=[my_mech_1, my_mech_2], name='my_LC')

LC.show()

my_gating_mechanism = GatingMechanism(gating_signals=[{'GATE_ALL': [my_mech_1,
                                                                    my_mech_2]}],
                                      modulation=ModulationParam.ADDITIVE)


my_gating_mechanism.show()