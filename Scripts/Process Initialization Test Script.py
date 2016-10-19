from PsyNeuLink.Functions.Process import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Utility import Linear, Logistic
import numpy as np

linear_transfer_mechanism = Transfer(function=Linear(slope = 1, intercept = 0))

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)
linear_transfer_process.execute([1])

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process_Base(params = process_params)

linear_transfer_process.execute([1])