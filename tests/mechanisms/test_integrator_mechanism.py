import numpy as np
import pytest

from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import Integrator
# from PsyNeuLink.Components.Functions.Function import FunctionError
from PsyNeuLink.Globals.Keywords import ADAPTIVE, CONSTANT, DIFFUSION, SIMPLE
from PsyNeuLink.Globals.TimeScale import TimeScale

# ======================================= FUNCTION TESTS ============================================

# VALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# integration_type = simple


def test_integrator_simple():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=SIMPLE,
        ),
        time_scale=TimeScale.TIME_STEP
    )
    # val = float(I.execute(10)[0])
    P = process(pathway=[I])
    val = float(P.execute(10))
    #  returns previous_value + rate*variable + noise
    # so in this case, returns 10.0
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 2
# integration_type = adaptive


def test_integrator_adaptive():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=ADAPTIVE,
        ),
        time_scale=TimeScale.TIME_STEP
    )
    # val = float(I.execute(10)[0])
    P = process(pathway=[I])
    val = float(P.execute(10))
    # returns (rate)*variable + (1-rate*previous_value) + noise
    # rate = 1, noise = 0, so in this case, returns 10.0
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 3
# integration_type = constant


def test_integrator_constant():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=CONSTANT,
        ),
        time_scale=TimeScale.TIME_STEP
    )
    # val = float(I.execute(10)[0])
    P = process(pathway=[I])
    val = float(P.execute(10))
    # returns previous_value + rate + noise
    # rate = 1.0, noise = 0, so in this case returns 1.0
    assert val == 1.0

# ------------------------------------------------------------------------------------------------
# TEST 4
# integration_type = diffusion


def test_integrator_diffusion():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=DIFFUSION,
        ),
        time_scale=TimeScale.TIME_STEP
    )
    # val = float(I.execute(10)[0])
    P = process(pathway=[I])
    val = float(P.execute(10))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------

# INVALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Linear


# def test_integrator_linear():

#     # SHOULD CAUSE AN ERROR??

#     with pytest.raises(FunctionError) as error_text:
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=Linear(),
#             time_scale=TimeScale.TIME_STEP
#         )
#         # val = float(I.execute(10)[0])
#         P = process(pathway=[I])
#         val = float(P.execute(10))
#     assert val == 10


# ======================================= INPUT TESTS ============================================

# VALID INPUT:

# ------------------------------------------------------------------------------------------------
# TEST 1
# input = float

def test_integrator_input_float():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=SIMPLE,
        )
    )
    P = process(pathway=[I])
    val = float(P.execute(10.0))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 2
# input = list of length 1


def test_integrator_input_list():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=SIMPLE,
        )
    )
    P = process(pathway=[I])
    val = float(P.execute([10.0]))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 3
# input = list of length 5


def test_integrator_input_list_len_5():

    I = IntegratorMechanism(
        name='IntegratorMechanism',
        default_input_value=[0, 0, 0, 0, 0],
        function=Integrator(
            integration_type=SIMPLE,
        )
    )
    P = process(pathway=[I])
    val = P.execute([10.0, 5.0, 2.0, 1.0, 0.0])
    expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

    for i in range(len(expected_output)):
        v = val[i]
        e = expected_output[i]
        np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

# ------------------------------------------------------------------------------------------------
# TEST 4
# input = numpy array of length 5


def test_integrator_input_array_len_5():

    I = IntegratorMechanism(
        name='IntegratorMechanism',
        default_input_value=[0, 0, 0, 0, 0],
        function=Integrator(
            integration_type=SIMPLE,
        )
    )
    P = process(pathway=[I])
    input_array = np.array([10.0, 5.0, 2.0, 1.0, 0.0])
    val = P.execute(input_array)
    expected_output = [10.0, 5.0, 2.0, 1.0, 0.0]

    for i in range(len(expected_output)):
        v = val[i]
        e = expected_output[i]
        np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

# ------------------------------------------------------------------------------------------------

# INVALID INPUT:

# ------------------------------------------------------------------------------------------------
# TEST 1
# input = list of length > default length


def test_integrator_input_array_greater_than_default():

    with pytest.raises(ValueError) as error_text:
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_input_value=[0, 0, 0]
        )
        P = process(pathway=[I])
        P.execute([10.0, 5.0, 2.0, 1.0, 0.0])
    assert "shapes" in str(error_text) and "not aligned" in str(error_text)

# ------------------------------------------------------------------------------------------------
# TEST 2
# input = list of length < default length


def test_integrator_input_array_less_than_default():

    with pytest.raises(ValueError) as error_text:
        I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_input_value=[0, 0, 0, 0, 0]
        )
        P = process(pathway=[I])
        P.execute([10.0, 5.0, 2.0])
    assert "shapes" in str(error_text) and "not aligned" in str(error_text)

# ======================================= RATE TESTS ============================================

# VALID RATE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# rate = float, integration_type = simple


def test_integrator_type_simple_rate_float():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=SIMPLE,
            rate=5.0
        )
    )
    P = process(pathway=[I])
    val = float(P.execute(10.0))
    assert val == 50.0

# ------------------------------------------------------------------------------------------------
# TEST 2
# rate = float, integration_type = constant


def test_integrator_type_constant_rate_float():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=CONSTANT,
            rate=5.0
        )
    )
    P = process(pathway=[I])
    val = float(P.execute(10.0))
    assert val == 5.0

# ------------------------------------------------------------------------------------------------
# TEST 3
# rate = float, integration_type = diffusion


def test_integrator_type_diffusion_rate_float():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=DIFFUSION,
            rate=5.0
        )
    )
    P = process(pathway=[I])
    val = float(P.execute(10.0))
    assert val == 50.0

# ------------------------------------------------------------------------------------------------
# # TEST 4
# # rate = list, integration_type = simple


# def test_integrator_type_simple_rate_list():
#     I = IntegratorMechanism(
#         name='IntegratorMechanism',
#         default_input_value=[0, 0, 0],
#         function=Integrator(
#             integration_type=SIMPLE,
#             rate=[5.0, 5.0, 5.0]
#         )
#     )
#     P = process(pathway=[I])
#     val = list(P.execute([10.0, 10.0, 10.0]))
#     assert val == [50.0, 50.0, 50.0]
# # ------------------------------------------------------------------------------------------------
# # TEST 5
# # rate = list, integration_type = constant


# def test_integrator_type_constant_rate_list():
#     I = IntegratorMechanism(
#         default_input_value=[0, 0, 0],
#         name='IntegratorMechanism',
#         function=Integrator(
#             integration_type=CONSTANT,
#             rate=[5.0, 5.0, 5.0]
#         )
#     )
#     P = process(pathway=[I])
#     val = list(P.execute([10.0, 10.0, 10.0]))
#     assert val == [5.0, 5.0, 5.0]

# # ------------------------------------------------------------------------------------------------
# # TEST 6
# # rate = list, integration_type = diffusion


# def test_integrator_type_diffusion_rate_list():
#     I = IntegratorMechanism(
#         default_input_value=[0, 0, 0],
#         name='IntegratorMechanism',
#         function=Integrator(
#             integration_type=DIFFUSION,
#             rate=[5.0, 5.0, 5.0]
#         )
#     )
#     P = process(pathway=[I])
#     val = list(P.execute([10.0, 10.0, 10.0]))
#     assert val == [50.0, 50.0, 50.0]

# # ------------------------------------------------------------------------------------------------
# # TEST 7
# # rate = list, integration_type = diffusion


# def test_integrator_type_adaptive_rate_list():
#     I = IntegratorMechanism(
#         default_input_value=[0, 0, 0],
#         name='IntegratorMechanism',
#         function=Integrator(
#             integration_type=ADAPTIVE,
#             rate=[0.5, 0.5, 0.5]
#         )
#     )
#     P = process(pathway=[I])
#     val = list(P.execute([10.0, 10.0, 10.0]))
#     assert val == [5.0, 5.0, 5.0]

# ------------------------------------------------------------------------------------------------
# TEST 8
# rate = float, integration_type = adaptive


def test_integrator_type_adaptive_rate_float_input_list():
    I = IntegratorMechanism(
        default_input_value=[0, 0, 0],
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=ADAPTIVE,
            rate=0.5
        )
    )
    P = process(pathway=[I])
    val = list(P.execute([10.0, 10.0, 10.0]))
    assert val == [5.0, 5.0, 5.0]

# ------------------------------------------------------------------------------------------------
# TEST 9
# rate = float, integration_type = adaptive


def test_integrator_type_adaptive_rate_float():
    I = IntegratorMechanism(
        name='IntegratorMechanism',
        function=Integrator(
            integration_type=ADAPTIVE,
            rate=0.5
        )
    )
    P = process(pathway=[I])
    val = list(P.execute(10.0))
    assert val == [5.0]

# ------------------------------------------------------------------------------------------------

# INVALID INPUT:

# ------------------------------------------------------------------------------------------------
# TEST 1
# rate = list of length > default length
# integration_type = SIMPLE


# def test_integrator_type_simple_rate_list_input_float():
#     with pytest.raises(FunctionError) as error_text:
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=Integrator(
#                 integration_type=SIMPLE,
#                 rate=[5.0, 5.0, 5.0]
#             )
#         )
#         P = process(pathway=[I])
#         float(P.execute(10.0))
#     assert (
#         "array specified for the rate parameter" in str(error_text)
#         and "must match the length" in str(error_text)
#         and "of the default input" in str(error_text)
#     )
# # ------------------------------------------------------------------------------------------------
# # TEST 2
# # rate = list of length > default length
# # integration_type = CONSTANT


# def test_integrator_type_constant_rate_list_input_float():
#     with pytest.raises(FunctionError) as error_text:
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=Integrator(
#                 integration_type=CONSTANT,
#                 rate=[5.0, 5.0, 5.0]
#             )
#         )
#         P = process(pathway=[I])
#         float(P.execute(10.0))
#     assert (
#         "array specified for the rate parameter" in str(error_text)
#         and "must match the length" in str(error_text)
#         and "of the default input" in str(error_text)
#     )

# # ------------------------------------------------------------------------------------------------
# # TEST 3
# # rate = list of length > default length
# # integration_type = DIFFUSION


# def test_integrator_type_diffusion_rate_list_input_float():
#     with pytest.raises(FunctionError) as error_text:
#         I = IntegratorMechanism(
#             name='IntegratorMechanism',
#             function=Integrator(
#                 integration_type=DIFFUSION,
#                 rate=[5.0, 5.0, 5.0]
#             )
#         )
#         P = process(pathway=[I])
#         float(P.execute(10.0))
#     assert (
#         "array specified for the rate parameter" in str(error_text)
#         and "must match the length" in str(error_text)
#         and "of the default input" in str(error_text)
#     )
