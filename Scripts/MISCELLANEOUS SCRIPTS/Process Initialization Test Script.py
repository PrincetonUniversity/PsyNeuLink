from psyneulink.components.Function import Linear

from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.process import *

linear_transfer_mechanism = TransferMechanism(function=Linear(slope = 1, intercept = 0))

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process(params = process_params)
linear_transfer_process.execute([1])

process_params = {PATHWAY:[linear_transfer_mechanism]}
linear_transfer_process = Process(params = process_params)

linear_transfer_process.execute([1])