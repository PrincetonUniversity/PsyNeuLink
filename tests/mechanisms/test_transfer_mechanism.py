from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import Logistic
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest


#  MECHANISMS WITH VALID NOISE:

def test_transfer_mech_array_var_float_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=5.0,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[5.0,5.0,5.0,5.0]]

def test_transfer_mech_array_var_normal_len_1_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=NormalDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()

    assert val == [[2.240893199201458, 2.240893199201458, 2.240893199201458, 2.240893199201458]]

def test_transfer_mech_array_var_normal_array_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=[NormalDist().function, NormalDist().function, NormalDist().function, NormalDist().function] ,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()

    assert val == [[0.7610377251469934, 0.12167501649282841, 0.44386323274542566, 0.33367432737426683]]

def test_transfer_mech_integer_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                   default_input_value = [0,0],
                                   function=Logistic(gain=0.1, bias=0.2),
                                   noise=5,
                                   time_constant = 0.1,
                                   time_scale=TimeScale.TIME_STEP
                                   )
        T.execute([0,0])
    assert 'noise parameter' in str(error_text.value)

def test_transfer_mech_mismatched_shape_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                   default_input_value = [0,0],
                                   function=Logistic(gain=0.1, bias=0.2),
                                   noise=[5.0,5.0,5.0],
                                   time_constant = 0.1,
                                   time_scale=TimeScale.TIME_STEP
                                   )
        T.execute()
    assert 'noise parameter' in str(error_text.value)

def test_transfer_mech_mismatched_shape_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                   default_input_value = [0,0],
                                   function=Logistic(gain=0.1, bias=0.2),
                                   noise=[5.0,5.0,5.0],
                                   time_constant = 0.1,
                                   time_scale=TimeScale.TIME_STEP
                                   )
        T.execute()
    assert 'noise parameter' in str(error_text.value)


#     T.execute([1,2])
#     except MechanismError as error_text:
#
#     T2 = TransferMechanism(name='my_Transfer_Test2',
#                             default_input_value = [0,0],
#                             function=Logistic(gain=0.1, bias=0.2),
#                             noise=2.0,
#                             time_constant = 0.1,
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test2.execute([1,1]))
#
#     my_Transfer_Test3 = TransferMechanism(name='my_Transfer_Test3',
#                             default_input_value = [0,0,0,0,0],
#                             function=Logistic(gain=1.0, bias=0.0),
#                             time_constant = 0.2,
#                             time_scale=TimeScale.TIME_STEP
#                             )
#
#     print(my_Transfer_Test3.execute([10,20,30,40,50]))
#
#     my_Transfer_Test4 = TransferMechanism(name='my_Transfer_Test4',
#                             default_input_value = [0,0,0,0,0],
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = [1.0,2.0,3.0,4.0,5.0],
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test4.execute([10,20,30,40,50]))
#
#     my_Transfer_Test5 = TransferMechanism(name='my_Transfer_Test5',
#                             default_input_value = [0,0,0,0,0],
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = [NormalDist().function, UniformDist().function, ExponentialDist().function, WaldDist().function, GammaDist().function ],
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test5.execute([10,20,30,40,50]))
#
#     my_Transfer_Test6 = TransferMechanism(name='my_Transfer_Test6',
#                             default_input_value = [0,0,0,0,0],
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = NormalDist().function,
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test6.execute([10,10,10,10,10]))
#
#     my_Transfer_Test8 = TransferMechanism(name='my_Transfer_Test8',
#                             default_input_value = [0,0,0,0,0],
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = 5.0,
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test8.execute([10,10,10,10,10]))
#
#     my_Transfer_Test9 = TransferMechanism(name='my_Transfer_Test9',
#                             default_input_value = 0.0,
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = 5.0,
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test9.execute(1.0))
#
#     try:
#         my_Transfer_Test10 = TransferMechanism(name='my_Transfer_Test10',
#                                 default_input_value = 0.0,
#                                 function=Logistic(gain=0.1, bias=0.2),
#                                 time_constant = 0.2,
#                                 noise = [5.0, 5.0],
#                                 time_scale=TimeScale.TIME_STEP
#                                 )
#         print(my_Transfer_Test10.execute(1.0))
#     except MechanismError as error_text:
#
#     try:
#         my_Transfer_Test11 = TransferMechanism(name='my_Transfer_Test11',
#                                 default_input_value = 0.0,
#                                 function=Logistic(gain=0.1, bias=0.2),
#                                 time_constant = 0.2,
#                                 noise = [NormalDist().function, UniformDist().function],
#                                 time_scale=TimeScale.TIME_STEP
#                                 )
#         print(my_Transfer_Test11.execute(1.0))
#     except MechanismError as error_text:
# =
#     try:
#         my_Transfer_Test12 = TransferMechanism(name='my_Transfer_Test12',
#                                 default_input_value = [0, 0, 0],
#                                 function=Logistic(gain=0.1, bias=0.2),
#                                 time_constant = 0.2,
#                                 noise = [1,2,3],
#                                 time_scale=TimeScale.TIME_STEP
#                                 )
#         print(my_Transfer_Test12.execute([1,1,1]))
#     except MechanismError as error_text:
#     my_Transfer_Test13 = TransferMechanism(name='my_Transfer_Test13',
#                             default_input_value = 0.0,
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = [1.0],
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test13.execute(1.0))
#     print("Passed")
#
#     print("")
#
#     print("-------------------------------------------------")
#
#     print("Transfer Test #14: Execute Transfer with noise= list of len 1 function, input = float ")
#     my_Transfer_Test14 = TransferMechanism(name='my_Transfer_Test14',
#                             default_input_value = 0.0,
#                             function=Logistic(gain=0.1, bias=0.2),
#                             time_constant = 0.2,
#                             noise = [NormalDist().function],
#                             time_scale=TimeScale.TIME_STEP
#                             )
#     print(my_Transfer_Test14.execute(1.0))
#
# if run_distribution_test:
#     print("Distribution Test #1: Execute Transfer with noise = WaldDist(scale = 2.0, mean = 2.0).function")
#
#
#     my_Transfer = TransferMechanism(name='my_Transfer',
#                            default_input_value = [0,0],
#                            function=Logistic(gain=0.1, bias=0.2),
#                            noise=WaldDist(scale = 2.0, mean = 2.0).function,
#                            time_constant = 0.1,
#                            time_scale=TimeScale.TIME_STEP
#                            )
#     my_Transfer.execute([1,1])
#
#     print("Passed")
#     print("")
#
#     print("-------------------------------------------------")
#
#     print("Distribution Test #2: Execute Transfer with noise = GammaDist(scale = 1.0, shape = 1.0).function")
#
#
#     my_Transfer2 = TransferMechanism(name='my_Transfer2',
#                            default_input_value = [0,0],
#                            function=Logistic(gain=0.1, bias=0.2),
#                            noise=GammaDist(scale = 1.0, shape = 1.0).function,
#                            time_constant = 0.1,
#                            time_scale=TimeScale.TIME_STEP
#                            )
#     my_Transfer2.execute([1,1])
#
#     print("Passed")
#     print("")
#
#     print("-------------------------------------------------")
#
#     print("Distribution Test #3: Execute Transfer with noise = UniformDist(low = 2.0, high = 3.0).function")
#
#     my_Transfer3 = TransferMechanism(name='my_Transfer3',
#                            default_input_value = [0,0],
#                            function=Logistic(gain=0.1, bias=0.2),
#                            noise=UniformDist(low = 2.0, high = 3.0).function,
#                            time_constant = 0.1,
#                            time_scale=TimeScale.TIME_STEP
#                            )
#     my_Transfer3.execute([1,1])
#
#     print("Passed")
#     print("")
#
#     print("-------------------------------------------------")
#
#     print("Distribution Test #4: Execute Transfer with noise = ExponentialDist(beta=1.0).function")
#
#     my_Transfer4 = TransferMechanism(name='my_Transfer4',
#                            default_input_value = [0,0],
#                            function=Logistic(gain=0.1, bias=0.2),
#                            noise=ExponentialDist(beta=1.0).function,
#                            time_constant = 0.1,
#                            time_scale=TimeScale.TIME_STEP
#                            )
#     my_Transfer4.execute([1,1])
#
#     print("Passed")
#     print("")
#
#     print("-------------------------------------------------")
#
#     print("Distribution Test #5: Execute Transfer with noise = NormalDist(mean=1.0, standard_dev = 2.0).function")
#
#     my_Transfer5 = TransferMechanism(name='my_Transfer5',
#                            default_input_value = [0,0],
#                            function=Logistic(gain=0.1, bias=0.2),
#                            noise=NormalDist(mean=1.0, standard_dev = 2.0).function,
#                            time_constant = 0.1,
#                            time_scale=TimeScale.TIME_STEP
#                            )
#     my_Transfer5.execute([1,1])
#
#     print("Passed")
#     print("")
#
