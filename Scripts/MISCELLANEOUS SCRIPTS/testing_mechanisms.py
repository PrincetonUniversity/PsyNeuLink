import numpy as np
import pytest
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import Linear, SimpleIntegrator, ConstantIntegrator, AccumulatorIntegrator
from PsyNeuLink.Components.Functions.Function import AdaptiveIntegrator, DriftDiffusionIntegrator, OrnsteinUhlenbeckIntegrator
from PsyNeuLink.Components.Functions.Function import FunctionError
from PsyNeuLink.Globals.Keywords import ADAPTIVE, CONSTANT, DIFFUSION, SIMPLE
from PsyNeuLink.Globals.TimeScale import TimeScale


I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=SimpleIntegrator(
        ),
        context="EXECUTING",
        time_scale=TimeScale.TIME_STEP
    )
# P = process(pathway=[I])

#  returns previous_value + rate*variable + noise
# so in this case, returns 10.0
# print("print1, ", I.context)
val = float(I.execute(10, ignore_execution_id=True))
# testing initializer
I.function_object.reset_initializer = 5.0
val2 = float(I.execute(0,  ignore_execution_id=True))
print([val, val2])
print([val, val2] == [10.0, 5.0])

#
# I2 = IntegratorMechanism(
#         name='IntegratorMechanism',
#         function=AdaptiveIntegrator(
#             rate=0.5),
#         context = "EXECUTING",
#         time_scale=TimeScale.TIME_STEP
#     )
# # val = float(I.execute(10)[0])
# P = process(pathway=[I2])
# val = float(P.execute(10))
# I2.context = None
# print("after setting context to None [1/2] ", I2.context)
# # returns (rate)*variable + (1-rate*previous_value) + noise
# # rate = 1, noise = 0, so in this case, returns 10.0
# # testing initializer
# I2.function_object.reset_initializer = 1.0
# val2 = float(P.execute(1))
# print("after setting context to None [2/2] ", I2.context)
# print(val2)
# assert [val, val2] == [5.0, 1.0]
# print([val, val2] == [5.0, 1.0])
#
#
# T = TransferMechanism(
#         name='TransferMechanism',
#         context = "EXECUTING",
#         time_scale=TimeScale.TIME_STEP
#     )
# # val = float(I.execute(10)[0])
# # P = process(pathway=[T])
# val = float(T.execute(10))
# # returns (rate)*variable + (1-rate*previous_value) + noise
# # rate = 1, noise = 0, so in this case, returns 10.0
# T.context = "EXECUTING"
# print(T.context)
# # testing initializer
# T.function_object.reset_initializer = 1.0
# val2 = float(T.execute(1))
# print(val2)
# assert [val, val2] == [5.0, 1.0]
# print([val, val2] == [5.0, 1.0])