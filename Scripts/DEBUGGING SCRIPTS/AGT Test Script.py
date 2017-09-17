from PsyNeuLink.Globals.Keywords import *
from PsyNeuLink.Components.System import System_Base, system
from PsyNeuLink.Components.Process import Process_Base, process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.IntegratorMechanisms.DDM import DDM
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.AGT.LCMechanism import LCMechanism
from PsyNeuLink.Library.Mechanisms.AdaptiveMechanisms.ControlMechanisms.AGT.ITCMechanism import ITCMechanism
from PsyNeuLink.Components.Functions.Function import FHNIntegrator, UtilityIntegrator

my_mech = TransferMechanism()
my_ITC = ITCMechanism(monitored_output_states=my_mech)
my_LC = LCMechanism(function=(FHNIntegrator(mode=my_ITC)),
                    objective_mechanism=[my_mech])