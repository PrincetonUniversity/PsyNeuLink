import numpy as np
import pytest

from PsyNeuLink.Components.Component import ComponentError
from PsyNeuLink.Components.Functions.Function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax
from PsyNeuLink.Components.Functions.Function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferError
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Scheduling.TimeScale import TimeScale

# ======================================= INPUT TESTS ============================================

# VALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# check attributes


def test_recurrent_mech_check_attrs():

    R = RecurrentTransferMechanism(
        name='R',
        size=3
    )
    assert R.value is None
    assert R.variable.tolist() == [[0., 0., 0.]]
    assert R.matrix.tolist() == [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# check recurrent projection attributes


def test_recurrent_mech_check_proj_attrs():

    R = RecurrentTransferMechanism(
        name='R',
        size=3
    )
    assert R.recurrent_projection.matrix is R.matrix
    assert R.recurrent_projection.sender is R.output_state
    assert R.recurrent_projection.receiver is R.input_state

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = list of ints


def test_recurrent_mech_inputs_list_of_ints():

    R = RecurrentTransferMechanism(
        name='R',
        default_variable=[0, 0, 0, 0]
    )
    val = R.execute([10, 10, 10, 10]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 4
# variable = list of floats


def test_recurrent_mech_inputs_list_of_floats():

    R = RecurrentTransferMechanism(
        name='R',
        size=4
    )
    val = R.execute([10.0, 10.0, 10.0, 10.0]).tolist()
    assert val == [[10.0, 10.0, 10.0, 10.0]]

# ------------------------------------------------------------------------------------------------
# TEST 5
# variable = list of fns


def test_recurrent_mech_inputs_list_of_fns():

    R = RecurrentTransferMechanism(
        name='R',
        size=4,
        time_scale=TimeScale.TIME_STEP
    )
    val = R.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

# ------------------------------------------------------------------------------------------------
# TEST 6
# no variable and no size


def test_recurrent_mech_no_inputs():

    R = RecurrentTransferMechanism(
        name='R'
    )
    assert R.variable.tolist() == [[0]]
    val = R.execute([10]).tolist()
    assert val == [[10.]]

# ------------------------------------------------------------------------------------------------

# INVALID INPUTS

# ------------------------------------------------------------------------------------------------
# TEST 1
# variable is list of ints, input is list of strings


def test_recurrent_mech_inputs_list_of_strings():
    with pytest.raises(UtilitiesError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            time_scale=TimeScale.TIME_STEP
        )
        R.execute(["one", "two", "three", "four"]).tolist()
    assert "has non-numeric entries" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# variable is list of strings


def test_recurrent_mech_var_list_of_strings():
    with pytest.raises(UtilitiesError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=['a', 'b', 'c', 'd'],
            time_scale=TimeScale.TIME_STEP
        )
    assert "has non-numeric entries" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# variable = list of greater length than default input


def test_recurrent_mech_inputs_mismatched_with_default_longer():
    with pytest.raises(MechanismError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            size=4
        )
        R.execute([1, 2, 3, 4, 5]).tolist()
    assert "does not match required length" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# variable = list of shorter length than default input


def test_recurrent_mech_inputs_mismatched_with_default_shorter():
    with pytest.raises(MechanismError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            size=6
        )
        R.execute([1, 2, 3, 4, 5]).tolist()
    assert "does not match required length" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# ======================================= MATRIX TESTS ===========================================

# VALID MATRICES

# consider adding a different kind of validation to these matrix tests:
# the current assertion/validation doesn't really execute the recurrent projection

# ------------------------------------------------------------------------------------------------
# TEST 1
# matrix is a string specification


def test_recurrent_mech_matrix_str_spec():

    for m in MATRIX_KEYWORD_VALUES:
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            matrix=m
        )
        val = R.execute([10, 10, 10, 10]).tolist()
        assert val == [[10., 10., 10., 10.]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# matrix is a matrix, array, or list specification


def test_recurrent_mech_matrix_other_spec():

    specs = [np.matrix('1 2; 3 4'), np.atleast_2d([[1, 2], [3, 4]]), [[1, 2], [3, 4]]]
    for m in specs:
        R = RecurrentTransferMechanism(
            name='R',
            size=2,
            matrix=m
        )
        val = R.execute([10, 10]).tolist()
        assert val == [[10., 10.]]
        assert R.matrix.tolist() == [[1, 2], [3, 4]] and isinstance(R.matrix, np.ndarray)

# ------------------------------------------------------------------------------------------------

# INVALID MATRICES

# ------------------------------------------------------------------------------------------------
# TEST 1
# matrix = larger than default input


def test_recurrent_mech_matrix_too_large():
    with pytest.raises(RecurrentTransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        )

    assert "must be same as the size of variable" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# matrix = smaller than default input


def test_recurrent_mech_matrix_too_small():
    with pytest.raises(RecurrentTransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            size=5,
            matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        )
    assert "must be same as the size of variable" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# matrix = 2D list of strings


def test_recurrent_mech_matrix_strings():
    with pytest.raises(UtilitiesError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            matrix=[['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']]
        )
    assert "has non-numeric entries" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# ====================================== FUNCTION TESTS ==========================================

# VALID FUNCTIONS

# ------------------------------------------------------------------------------------------------
# TEST 1
# function is Logistic


def test_recurrent_mech_function_logistic():

    R = RecurrentTransferMechanism(
        name='R',
        size=10,
        function=Logistic(gain=2, bias=1)
    )
    val = R.execute(np.ones(10)).tolist()
    assert val == [np.full(10, 0.7310585786300049).tolist()]

# ------------------------------------------------------------------------------------------------
# TEST 2
# function is user-assigned PsyNeuLink function


def test_recurrent_mech_function_psyneulink():

    a = Logistic(gain=2, bias=1)

    R = RecurrentTransferMechanism(
        name='R',
        size=7,
        function=a
    )
    val = R.execute(np.zeros(7)).tolist()
    assert val == [np.full(7, 0.2689414213699951).tolist()]

# ------------------------------------------------------------------------------------------------
# TEST 3
# function is user-defined custom function


def test_recurrent_mech_function_custom():
    pass

# I don't know how to do this at the moment but it seems important.

# ------------------------------------------------------------------------------------------------

# INVALID FUNCTIONS

# ------------------------------------------------------------------------------------------------
# TEST 1
# function is normal distribution


def test_recurrent_mech_normal_fun():
    with pytest.raises(TransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=NormalDist(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        R.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 2
# function = Learning Function


def test_recurrent_mech_reinforcement_fun():
    with pytest.raises(TransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Reinforcement(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        R.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 3
# function = Integrator Function


def test_recurrent_mech_integrator_fun():
    with pytest.raises(TransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=ConstantIntegrator(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        R.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------
# TEST 4
# function = Combination Function


def test_recurrent_mech_reduce_fun():
    with pytest.raises(TransferError) as error_text:
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Reduce(),
            time_constant=1.0,
            time_scale=TimeScale.TIME_STEP
        )
        R.execute([0, 0, 0, 0]).tolist()
    assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

# ------------------------------------------------------------------------------------------------

# =================================== TIME_CONSTANT TESTS ========================================

# VALID TIME_CONSTANT PARAMS

# ------------------------------------------------------------------------------------------------
# TEST 1
# time_constant = 0.8

def test_recurrent_mech_time_constant_0_8():
    R = RecurrentTransferMechanism(
        name='R',
        default_variable=[0, 0, 0, 0],
        function=Linear(),
        time_constant=0.8,
        time_scale=TimeScale.TIME_STEP
    )
    val = R.execute([1, 1, 1, 1]).tolist()
    assert val == [[0.8, 0.8, 0.8, 0.8]]

# ------------------------------------------------------------------------------------------------
# TEST 2
# time_constant = 0.8, initial_value=0.5

# def test_recurrent_mech_time_constant_0_8_initial_0_5():
#     R = RecurrentTransferMechanism(
#         name='R',
#         default_variable=[0, 0, 0, 0],
#         function=Linear(),
#         time_constant=0.8,
#         initial_value=np.atleast_2d([0.5, 0.5, 0.5, 0.5]),
#         time_scale=TimeScale.TIME_STEP
#     )
#     val = R.execute([1, 1, 1, 1]).tolist()
#     assert val == [[0.9, 0.9, 0.9, 0.9]]

# ------------------------------------------------------------------------------------------------
# TEST 3
# time_constant = 0.8