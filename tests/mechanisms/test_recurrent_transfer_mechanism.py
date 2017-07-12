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
        default_input_value=[0, 0, 0, 0]
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
# TEST 7
# matrix is a string specification

# consider adding a different kind of validation to these matrix tests:
# the current assertion/validation doesn't really execute the recurrent projection
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
# TEST 8
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

