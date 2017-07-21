from PsyNeuLink.Components.Process import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Function import Linear, Logistic
import numpy as np

linear_transfer_mechanism = TransferMechanism(function=Linear(slope = 1, intercept = 0))

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)
linear_transfer_process.execute([1])

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)

linear_transfer_process.execute([1])