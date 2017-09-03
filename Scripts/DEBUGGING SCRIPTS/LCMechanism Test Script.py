from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.LCMechanism import LCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *

mech_1 = TransferMechanism()
mech_2 = TransferMechanism()
mech_3 = TransferMechanism()

LC = LCMechanism(modulated_mechanisms=[mech_1, mech_2, mech_3])
