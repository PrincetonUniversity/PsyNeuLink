from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Functions.Function import *
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest


# ======================================= INPUT TESTS ============================================

        # VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = list of ints

def test_transfer_mech_inputs_list_of_ints():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([10,10,10,10]).tolist()
    assert val == [[10.0,10.0,10.0,10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = list of floats

def test_transfer_mech_inputs_list_of_floats():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([10.0,10.0,10.0,10.0]).tolist()
    assert val == [[10.0,10.0,10.0,10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = list of fns

# ******
# Should transfer mech specifically support taking the output of a PNL function as input & reformat it to make sense?
# AND/OR Should transfer mech allow functions to be passed in as input and know how to execute them
# ******

def test_transfer_mech_inputs_list_of_fns():

    T = TransferMechanism(name='T',
                                   default_input_value = [0,0,0,0],
                                   time_scale=TimeScale.TIME_STEP
                                   )
    val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    assert val == [[np.array([ 0.]), 0.4001572083672233, np.array([ 1.]), 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------

        # INVALID INPUTS

# ------------------------------------------------------------------------------------------------

# TEST 1
# variable = list of strings

def test_transfer_mech_inputs_list_of_strings():
    with pytest.raises(UtilitiesError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute(["one", "two", "three", "four"]).tolist()

    assert "has non-numeric entries" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 2
# variable = list of greater length than default input

def test_transfer_mech_inputs_mismatched_with_default():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0],
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([1,2,3,4,5]).tolist()

    assert "does not match required length" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 3
# variable = list of shorter length than default input

def test_transfer_mech_inputs_mismatched_with_default():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(name='T',
                                       default_input_value = [0,0,0,0,0,0],
                                       time_scale=TimeScale.TIME_STEP
                                       )
        val = T.execute([1,2,3,4,5]).tolist()

    assert "does not match required length" in str(error_text.value)

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
    assert val == [[2.240893199201458, 2.240893199201458, 2.240893199201458, 2.240893199201458]]

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
    assert val == [[0.7610377251469934, 0.12167501649282841, 0.44386323274542566, 0.33367432737426683]]

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
    assert val == [[2.240893199201458, 2.240893199201458, 2.240893199201458, 2.240893199201458]]

# ------------------------------------------------------------------------------------------------
# TEST 6
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
    assert val == [[0.7872011523172707, 0.7872011523172707, 0.7872011523172707, 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 7
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
    assert val == [[0.5448831829968969, 0.5448831829968969, 0.5448831829968969, 0.5448831829968969]]

# ------------------------------------------------------------------------------------------------
# TEST 8
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
    assert val == [[0.7872011523172707, 0.7872011523172707, 0.7872011523172707, 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 9
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
    assert val == [[2.5652458745028266, 2.5652458745028266, 2.5652458745028266, 2.5652458745028266]]

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


# ======================================= TIME_CONSTANT TESTS ============================================

    # VALID TIME_CONSTANT PARAMS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# time_constant = 0.8

def test_transfer_mech_time_constant_0_8():
    T = TransferMechanism(name='T',
                          default_input_value=[0, 0, 0, 0],
                          function=Linear(),
                          time_constant=0.8,
                          time_scale=TimeScale.TIME_STEP
                          )
    val = T.execute([1,1,1,1]).tolist()
    assert val == [[0.8, 0.8, 0.8, 0.8]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 1.0

def test_transfer_mech_time_constant_1_0():
    T = TransferMechanism(name='T',
                          default_input_value=[0, 0, 0, 0],
                          function=Linear(),
                          time_constant=1.0,
                          time_scale=TimeScale.TIME_STEP
                          )
    val = T.execute([1,1,1,1]).tolist()
    assert val == [[1.0,1.0,1.0,1.0]]

# ------------------------------------------------------------------------------------------------
# TEST 1
# time_constant = 0.0

def test_transfer_mech_time_constant_0_0():
    T = TransferMechanism(name='T',
                          default_input_value=[0, 0, 0, 0],
                          function=Linear(),
                          time_constant=0.0,
                          time_scale=TimeScale.TIME_STEP
                          )
    val = T.execute([1,1,1,1]).tolist()
    assert val == [[0.0,0.0,0.0,0.0]]

# ------------------------------------------------------------------------------------------------

    # INVALID TIME_CONSTANT PARAMS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# time_constant = [0.8,0.8,0.8,0.8]

# *******
# Should PNL support a time_constant list with a separate value for each input?
# *******
def test_transfer_mech_time_constant_0_8_list():
    with pytest.raises(ComponentError) as error_text:
        T = TransferMechanism(name='T',
                              default_input_value=[0, 0, 0, 0],
                              function=Linear(),
                              time_constant=[0.8,0.8,0.8,0.8],
                              time_scale=TimeScale.TIME_STEP
                              )
        val = T.execute([1,1,1,1]).tolist()
    assert "Value of time_constant param" in str(error_text.value) \
           and "must be a float" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 2

def test_transfer_mech_time_constant_2():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                              default_input_value=[0, 0, 0, 0],
                              function=Linear(),
                              time_constant=2,
                              time_scale=TimeScale.TIME_STEP
                              )
        val = T.execute([1,1,1,1]).tolist()
    assert "time_constant parameter" in str(error_text.value) \
           and "must be a float between 0 and 1" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# time_constant = 1

def test_transfer_mech_time_constant_1():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                              default_input_value=[0, 0, 0, 0],
                              function=Linear(),
                              time_constant=1,
                              time_scale=TimeScale.TIME_STEP
                              )
        val = T.execute([1, 1, 1, 1]).tolist()
    assert "time_constant parameter" in str(error_text.value) \
           and "must be a float between 0 and 1" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# time_constant = 1

def test_transfer_mech_time_constant_0():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(name='T',
                              default_input_value=[0, 0, 0, 0],
                              function=Linear(),
                              time_constant=0,
                              time_scale=TimeScale.TIME_STEP
                              )
        val = T.execute([1, 1, 1, 1]).tolist()
    assert "time_constant parameter" in str(error_text.value) \
           and "must be a float between 0 and 1" in str(error_text.value)
# ------------------------------------------------------------------------------------------------
