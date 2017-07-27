import numpy as np
import pytest

from PsyNeuLink.Components.Component import ComponentError
from PsyNeuLink.Components.Functions.Function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax
from PsyNeuLink.Components.Functions.Function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Globals.Utilities import UtilitiesError
from PsyNeuLink.Scheduling.TimeScale import TimeScale

# ======================================= INPUT TESTS ============================================

# VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = list of ints


def test_transfer_mech_inputs_list_of_ints():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([10, 10, 10, 10]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]
    assert len(T.size) == 1 and T.size[0] == 4 and type(T.size[0]) == np.int64
    # this test assumes size is returned as a 1D array: if it's not, then several tests in this file must be changed

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = list of floats


def test_transfer_mech_inputs_list_of_floats():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = list of fns

# ******
# Should transfer mech specifically support taking the output of a PNL function as input & reformat it to make sense?
# AND/OR Should transfer mech allow functions to be passed in as input and know how to execute them
# ******


def test_transfer_mech_inputs_list_of_fns():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# variable = 3D array

def test_transfer_mech_variable_3D_array():

    T = TransferMechanism(
        name='T',
        default_variable=[[[0, 0, 0, 0]],[[1,1,1,1]]],
        time_scale=TimeScale.TIME_STEP
    )
    assert len(T.variable) == 1 and len(T.variable[0]) == 4 and (T.variable[0] == 0).all()

# ------------------------------------------------------------------------------------------------
# TEST 5
# variable = none and size = none; this should yield variable = [[0]]

def test_transfer_mech_variable_none_size_none():

    T = TransferMechanism(
        name='T'
    )
    assert len(T.variable) == 1 and len(T.variable[0]) == 1 and T.variable[0][0] == 0

# ------------------------------------------------------------------------------------------------

# INVALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = list of strings


def test_transfer_mech_inputs_list_of_strings():
    with pytest.raises(UtilitiesError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            time_scale=TimeScale.TIME_STEP
        )
        T.execute(["one", "two", "three", "four"]).tolist()
    assert "has non-numeric entries" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = list of greater length than default input


def test_transfer_mech_inputs_mismatched_with_default_longer():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 2, 3, 4, 5]).tolist()
    assert "does not match required length" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# TEST 3
# variable = list of shorter length than default input


def test_transfer_mech_inputs_mismatched_with_default_shorter():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0, 0, 0],
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 2, 3, 4, 5]).tolist()
    assert "does not match required length" in str(error_text.value)

# ======================================= NOISE TESTS ============================================

# VALID NOISE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = List
# noise = Single float


def test_transfer_mech_array_var_float_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=5.0,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[5.0, 5.0, 5.0, 5.0]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = List
# noise = Single function


def test_transfer_mech_array_var_normal_len_1_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=NormalDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[2.240893199201458, 2.240893199201458, 2.240893199201458, 2.240893199201458]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = List
# noise = List of functions


def test_transfer_mech_array_var_normal_array_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=[NormalDist().function, NormalDist().function, NormalDist().function, NormalDist().function],
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[0.7610377251469934, 0.12167501649282841, 0.44386323274542566, 0.33367432737426683]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# variable = List
# noise = List of floats


def test_transfer_mech_array_var_normal_array_noise2():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=[5.0, 5.0, 5.0, 5.0],
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[5.0, 5.0, 5.0, 5.0]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# Noise distribution: Normal


def test_transfer_mech_normal_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=NormalDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[2.240893199201458, 2.240893199201458, 2.240893199201458, 2.240893199201458]]

# ------------------------------------------------------------------------------------------------
# TEST 6
# Noise distribution: Exponential


def test_transfer_mech_exponential_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=ExponentialDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[0.7872011523172707, 0.7872011523172707, 0.7872011523172707, 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 7
# Noise distribution: Uniform


def test_transfer_mech_Uniform_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=UniformDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[0.5448831829968969, 0.5448831829968969, 0.5448831829968969, 0.5448831829968969]]

# ------------------------------------------------------------------------------------------------
# TEST 8
# Noise distribution: Gamma


def test_transfer_mech_Gamma_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=GammaDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[0.7872011523172707, 0.7872011523172707, 0.7872011523172707, 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 9
# Noise distribution: Wald


def test_transfer_mech_Wald_noise():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        noise=WaldDist().function,
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[2.5652458745028266, 2.5652458745028266, 2.5652458745028266, 2.5652458745028266]]

# ------------------------------------------------------------------------------------------------

# INVALID NOISE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable = List
# noise = Int


def test_transfer_mech_integer_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0],
            function=Linear(),
            noise=5,
            time_constant=0.1,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([0, 0])
    assert 'noise parameter' in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable = List
# noise = List of incorrect shape (>var)


def test_transfer_mech_mismatched_shape_noise():
    with pytest.raises(MechanismError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0],
            function=Linear(),
            noise=[5.0, 5.0, 5.0],
            time_constant=0.1,
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

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0],
            function=Linear(),
            noise=[5.0, 5.0],
            time_constant=0.1,
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

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Logistic(),
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[0.5, 0.5, 0.5, 0.5]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# function = exponential


def test_transfer_mech_exponential_fun():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Exponential(),
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[1.0, 1.0, 1.0, 1.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# function = soft max


def test_transfer_mech_softmax_fun():

    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=SoftMax(),
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([0, 0, 0, 0]).tolist()
    assert val == [[1.0, 1.0, 1.0, 1.0]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# function = linear matrix


# def test_transfer_mech_linearmatrix_fun():

#     T = TransferMechanism(
#         name='T',
#         default_variable=[0, 0, 0, 0],
#         function=LinearMatrix(),
#         time_constant=1.0,
#         time_scale=TimeScale.TIME_STEP
#     )
#     val = T.execute([0, 0, 0, 0]).tolist()
#     assert val == [[1.0, 1.0, 1.0, 1.0]]

# ------------------------------------------------------------------------------------------------

# INVALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Distribution Function


def test_transfer_mech_normal_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=NormalDist(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


# ------------------------------------------------------------------------------------------------
# TEST 2
# function = Learning Function

def test_transfer_mech_reinforcement_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Reinforcement(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# function = Integrator Function


def test_transfer_mech_integrator_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=ConstantIntegrator(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# function = Combination Function


def test_transfer_mech_reduce_fun():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Reduce(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


# ======================================= TIME_CONSTANT TESTS ============================================

# VALID TIME_CONSTANT PARAMS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# time_constant = 0.8

def test_transfer_mech_time_constant_0_8():
    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        time_constant=0.8,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([1, 1, 1, 1]).tolist()
    assert val == [[0.8, 0.8, 0.8, 0.8]]
    val = T.execute([1, 1, 1, 1]).tolist()
    assert val == [[0.96, 0.96, 0.96, 0.96]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 1.0


def test_transfer_mech_time_constant_1_0():
    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        time_constant=1.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([1, 1, 1, 1]).tolist()
    assert val == [[1.0, 1.0, 1.0, 1.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# time_constant = 0.0


def test_transfer_mech_time_constant_0_0():
    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        time_constant=0.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([1, 1, 1, 1]).tolist()
    assert val == [[0.0, 0.0, 0.0, 0.0]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 0.8, initial_value=0.5

def test_transfer_mech_time_constant_0_8_initial_0_5():
    T = TransferMechanism(
        name='T',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        time_constant=0.8,
        initial_value=np.array([[.5, .5, .5, .5]]),
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([1, 1, 1, 1]).tolist()
    assert val == [[0.9, 0.9, 0.9, 0.9]]
    T.noise = 10
    val = T.execute([1, 2, -3, 0]).tolist()
    assert val == [[10.98, 11.78, 7.779999999999999, 10.18]] # testing noise changes to an integrator

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
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=[0.8, 0.8, 0.8, 0.8],
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 1, 1, 1]).tolist()
    assert (
        "Value of time_constant param" in str(error_text.value)
        and "must be compatible with float" in str(error_text.value)
    )

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 2


def test_transfer_mech_time_constant_2():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=2,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 1, 1, 1]).tolist()
    assert (
        "time_constant parameter" in str(error_text.value)
        and "must be a float between 0 and 1" in str(error_text.value)
    )

# ------------------------------------------------------------------------------------------------
# TEST 3
# time_constant = 1


def test_transfer_mech_time_constant_1():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=1,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 1, 1, 1]).tolist()
    assert (
        "time_constant parameter" in str(error_text.value)
        and "must be a float between 0 and 1" in str(error_text.value)
    )

# ------------------------------------------------------------------------------------------------
# TEST 4
# time_constant = 1


def test_transfer_mech_time_constant_0():
    with pytest.raises(TransferError) as error_text:
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0,
            time_scale=TimeScale.TIME_STEP
        )
        T.execute([1, 1, 1, 1]).tolist()
    assert (
        "time_constant parameter" in str(error_text.value)
        and "must be a float between 0 and 1" in str(error_text.value)
    )

# ------------------------------------------------------------------------------------------------


# ======================================= SIZE_INITIALIZATION TESTS ============================================

# VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# size = int, check if variable is an array of zeros


def test_transfer_mech_size_int_check_var():
    T = TransferMechanism(
        name='T',
        size=4
    )
    assert len(T.variable) == 1 and (T.variable[0] == [0., 0., 0., 0.]).all()
    assert len(T.size) == 1 and T.size[0] == 4 and type(T.size[0]) == np.int64

# ------------------------------------------------------------------------------------------------
# TEST 2
# size = int, variable = list of ints


def test_transfer_mech_size_int_inputs_ints():
    T = TransferMechanism(
        name='T',
        size=4
    )
    val = T.execute([10, 10, 10, 10]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# size = int, variable = list of floats


def test_transfer_mech_size_int_inputs_floats():
    T = TransferMechanism(
        name='T',
        size=4
    )
    val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# size = int, variable = list of functions


def test_transfer_mech_size_int_inputs_fns():
    T = TransferMechanism(
        name='T',
        size=4,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# size = float, check if variable is an array of zeros


def test_transfer_mech_size_float_inputs_check_var():
    T = TransferMechanism(
        name='T',
        size=4.0,
    )
    assert len(T.variable) == 1 and (T.variable[0] == [0., 0., 0., 0.]).all()
    assert len(T.size == 1) and T.size[0] == 4.0 and type(T.size[0]) == np.int64

# ------------------------------------------------------------------------------------------------
# TEST 6
# size = float, variable = list of ints


def test_transfer_mech_size_float_inputs_ints():
    T = TransferMechanism(
        name='T',
        size=4.0
    )
    val = T.execute([10, 10, 10, 10]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 7
# size = float, variable = list of floats


def test_transfer_mech_size_float_inputs_floats():
    T = TransferMechanism(
        name='T',
        size=4.0
    )
    val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 8
# size = float, variable = list of functions


def test_transfer_mech_size_float_inputs_fns():
    T = TransferMechanism(
        name='T',
        size=4.0,
        time_scale=TimeScale.TIME_STEP
    )
    val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 9
# size = list of ints, check that variable is correct


def test_transfer_mech_size_list_of_ints():
    T = TransferMechanism(
        name='T',
        size=[2, 3, 4]
    )
    assert len(T.variable) == 3 and len(T.variable[0]) == 2 and len(T.variable[1]) == 3 and len(T.variable[2]) == 4

# ------------------------------------------------------------------------------------------------
# TEST 10
# size = list of floats, check that variable is correct


def test_transfer_mech_size_list_of_floats():
    T = TransferMechanism(
        name='T',
        size=[2., 3., 4.]
    )
    assert len(T.variable) == 3 and len(T.variable[0]) == 2 and len(T.variable[1]) == 3 and len(T.variable[2]) == 4

# ------------------------------------------------------------------------------------------------
# TEST 11
# size = list of floats, variable = a compatible 2D array: check that variable is correct
# note that this output under the Linear function is useless/odd, but the purpose of allowing this configuration
# is for possible user-defined functions.


def test_transfer_mech_size_var_both_lists():
    T = TransferMechanism(
        name='T',
        size=[2., 3.],
        default_variable=[[1, 2], [3, 4, 5]]
    )
    assert len(T.variable) == 2 and (T.variable[0] == [1, 2]).all() and (T.variable[1] == [3, 4, 5]).all()

# ------------------------------------------------------------------------------------------------
# TEST 12
# size = int, variable = a compatible 2D array: check that variable is correct


def test_transfer_mech_size_scalar_var_2d():
    T = TransferMechanism(
        name='T',
        size=2,
        default_variable=[[1, 2], [3, 4]]
    )
    assert len(T.variable) == 2 and (T.variable[0] == [1, 2]).all() and (T.variable[1] == [3, 4]).all()
    assert len(T.size) == 2 and T.size[0] == 2 and T.size[1] == 2

# ------------------------------------------------------------------------------------------------
# TEST 13
# variable = a 2D array: check that variable is correct


def test_transfer_mech_var_2d_array():
    T = TransferMechanism(
        name='T',
        default_variable=[[1, 2], [3, 4]]
    )
    assert len(T.variable) == 2 and (T.variable[0] == [1, 2]).all() and (T.variable[1] == [3, 4]).all()

# ------------------------------------------------------------------------------------------------
# TEST 14
# variable = a 1D array, size does not match: check that variable and output are correct


def test_transfer_mech_var_1D_size_wrong():
    T = TransferMechanism(
        name='T',
        default_variable=[1, 2, 3, 4],
        size=2
    )
    assert len(T.variable) == 1 and (T.variable[0] == [1, 2, 3, 4]).all()
    val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 15
# variable = a 1D array, size does not match again: check that variable and output are correct


def test_transfer_mech_var_1D_size_wrong_2():
    T = TransferMechanism(
        name='T',
        default_variable=[1, 2, 3, 4],
        size=[2, 3, 4]
    )
    assert len(T.variable) == 1 and (T.variable[0] == [1, 2, 3, 4]).all()
    val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 16
# size = int, variable = incompatible array, check variable


def test_transfer_mech_size_var_incompatible1():
    T = TransferMechanism(
        name='T',
        size=2,
        default_variable=[[1, 2], [3, 4, 5]]
    )
    assert (T.variable[0] == [1, 2]).all() and (T.variable[1] == [3, 4, 5]).all() and len(T.variable) == 2

# ------------------------------------------------------------------------------------------------
# TEST 17
# size = array, variable = incompatible array, check variable


def test_transfer_mech_size_var_incompatible1():
    T = TransferMechanism(
        name='T',
        size=[2, 2],
        default_variable=[[1, 2], [3, 4, 5]]
    )
    assert (T.variable[0] == [1, 2]).all() and (T.variable[1] == [3, 4, 5]).all() and len(T.variable) == 2

# ------------------------------------------------------------------------------------------------

# INVALID SIZE INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# size = 0, check less-than-one error


def test_transfer_mech_size_zero():
    with pytest.raises(ComponentError) as error_text:
        T = TransferMechanism(
            name='T',
            size=0,
        )
    assert "is not a positive number" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# size = -1.0, check less-than-one error


def test_transfer_mech_size_negative_one():
    with pytest.raises(ComponentError) as error_text:
        T = TransferMechanism(
            name='T',
            size=-1.0,
        )
    assert "is not a positive number" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# size = non-integer float, check integer-cast value-change warning
# this test and the (currently commented) test immediately after it _may_ be deprecated if we ever fix
# warnings to be no longer fatal. At the time of writing (6/30/17, CW), warnings are always fatal.


# the test commented out here is similar to what we'd want if we got warnings to be non-fatal
# and error_text was correctly representing the warning. For now, the warning is hidden under
# a verbosity preference
# def test_transfer_mech_size_bad_float():
#     with pytest.raises(UserWarning) as error_text:
#         T = TransferMechanism(
#             name='T',
#             size=3.5,
#         )
#     assert "cast to integer, its value changed" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# size = 2D array, check too-many-dimensions warning


# def test_transfer_mech_size_2d():
#     with pytest.raises(UserWarning) as error_text:
#         T = TransferMechanism(
#             name='T',
#             size=[[2]],
#         )
#     assert "had more than one dimension" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 5
# size = 2D array, check variable is correctly instantiated


# for now, since the test above doesn't work, we use this test. 6/30/17 (CW)
def test_transfer_mech_size_2d():
    T = TransferMechanism(
        name='T',
        size=[[2]],
    )
    assert len(T.variable) == 1 and len(T.variable[0]) == 2
    assert len(T.size) == 1 and T.size[0] == 2 and len(T.params['size']) == 1 and T.params['size'][0] == 2
