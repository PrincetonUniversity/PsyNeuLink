from PsyNeuLink.Functions.Process import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *

# simple_ddm_process = process('Simple DDM Process')
# simple_ddm_process.execute([1])

my_transfer_mechanism = Transfer(name="My Transfer Mechanism",
                                 params={FUNCTION:Logistic,
                                         FUNCTION_PARAMS:{
                                             kwTransfer_Gain:5,
                                             kwTransfer_Bias:0
                                         }})

# simple_transfer_process = Process_Base(name='Simple Transfer Process',
#                                   params={PATHWAY:[Transfer]})
# simple_transfer_process.execute([0.5])

simple_transfer_process = Process_Base(name='Simple Transfer Process',
                                  params={PATHWAY:[my_transfer_mechanism]})
simple_transfer_process.execute([1])
