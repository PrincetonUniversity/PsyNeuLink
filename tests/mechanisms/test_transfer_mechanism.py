from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import *
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest


# ======================================= NOISE TESTS ============================================

# VALID NOISE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = List
# noise = Single float

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

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = List
# noise = Single function
def test_transfer_mech_array_var_normal_len_1_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=NormalDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[0.4001572083672233, 0.9787379841057392, 2.240893199201458, 1.8675579901499675]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = List
# noise = List of functions

def test_transfer_mech_array_var_normal_array_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=[NormalDist().function, NormalDist().function, NormalDist().function, NormalDist().function] ,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[1.8675579901499675, -0.977277879876411, 0.9500884175255894, -0.1513572082976979]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# variable = List
# noise = List of floats

def test_transfer_mech_array_var_normal_array_noise2():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=[5.0,5.0,5.0,5.0] ,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[5.0,5.0,5.0,5.0]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Normal

def test_transfer_mech_normal_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=NormalDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[0.4001572083672233, 0.9787379841057392, 2.240893199201458, 1.8675579901499675]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Exponential

def test_transfer_mech_exponential_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=ExponentialDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[1.2559307629658378, 0.9232231458040688, 0.7872011523172707, 0.5510484910954992]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Uniform

def test_transfer_mech_Uniform_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=UniformDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[0.7151893663724195, 0.6027633760716439, 0.5448831829968969, 0.4236547993389047]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Gamma

def test_transfer_mech_Gamma_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=GammaDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[1.2559307629658378, 0.9232231458040688, 0.7872011523172707, 0.5510484910954992]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Wald

def test_transfer_mech_Wald_noise():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Linear(),
                                   noise=WaldDist().function,
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[0.671974843101774, 0.18871271022979674, 2.5652458745028266, 1.108683289464279]]

# ------------------------------------------------------------------------------------------------

# INVALID NOISE:

# ------------------------------------------------------------------------------------------------

# TEST 1
# variable = List
# noise = Int

def test_transfer_mech_integer_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                   default_input_value = [0,0],
                                   function=Linear(),
                                   noise=5,
                                   time_constant = 0.1,
                                   time_scale=TimeScale.TIME_STEP
                                   )
        T.execute([0,0])
    assert 'noise parameter' in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 2
# variable = List
# noise = List of incorrect shape (>var)

def test_transfer_mech_mismatched_shape_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                   default_input_value = [0,0],
                                   function=Linear(),
                                   noise=[5.0,5.0,5.0],
                                   time_constant = 0.1,
                                   time_scale=TimeScale.TIME_STEP
                                   )
        T.execute()
    assert 'noise parameter' in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 3
# variable = List
# noise = List of incorrect shape (<var)


def test_transfer_mech_mismatched_shape_noise_2():
    with pytest.raises(MechanismError) as error_text:

        T = TransferMechanism(name='T',
                            default_input_value = [0,0,0],
                            function=Linear(),
                            noise=[5.0,5.0],
                            time_constant = 0.1,
                            time_scale=TimeScale.TIME_STEP
                            )
        T.execute()
    assert 'noise parameter' in str(error_text.value)

# ------------------------------------------------------------------------------------------------


# ====================================== FUNCTION TESTS ==========================================

# VALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = logistic

def test_transfer_mech_logistic_fun():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Logistic(),
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[0.5,0.5,0.5,0.5]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# function = exponential

def test_transfer_mech_exponential_fun():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=Exponential(),
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[1.0,1.0,1.0,1.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# function = soft max

def test_transfer_mech_softmax_fun():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   function=SoftMax(),
                                   time_constant = 1.0,
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([0,0,0,0]).tolist()
    assert val == [[1.0,1.0,1.0,1.0]]

# # ------------------------------------------------------------------------------------------------
# # TEST 4
# # function = linear matrix
#
# def test_transfer_mech_linearmatrix_fun():
#
#     T = TransferMechanism(name='T',
#                                    default_input_value = [0,0,0,0],
#                                    function=LinearMatrix(),
#                                    time_constant = 1.0,
#                                    time_scale=TimeScale.TIME_STEP
#                                    )
#     val = T.execute([0,0,0,0]).tolist()
#     assert val == [[1.0,1.0,1.0,1.0]]

# ------------------------------------------------------------------------------------------------

# INVALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Distribution Function

def test_transfer_mech_normal_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       function=NormalDist(),
                                       time_constant = 1.0,
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([0,0,0,0]).tolist()

    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


# ------------------------------------------------------------------------------------------------

# TEST 2
# function = Learning Function

def test_transfer_mech_reinforcement_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       function=Reinforcement(),
                                       time_constant = 1.0,
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([0,0,0,0]).tolist()

    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 3
# function = Integrator Function

def test_transfer_mech_integrator_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       function=Integrator(),
                                       time_constant = 1.0,
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([0,0,0,0]).tolist()

    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 4
# function = Combination Function

def test_transfer_mech_reduce_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       function=Reduce(),
                                       time_constant = 1.0,
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([0,0,0,0]).tolist()

    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


# ------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------

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
