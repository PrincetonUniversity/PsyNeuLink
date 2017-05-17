from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import *
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest
from PsyNeuLink.Globals.Utilities import *

# ======================================= FUNCTION TESTS ============================================

# VALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# integration_type = simple

def test_integrator_simple():
    I = IntegratorMechanism(
            name='IntegratorMechanism',
            function = Integrator(
                                    integration_type= SIMPLE,
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    # val = float(I.execute(10)[0])
    P = process( pathway= [I])
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
            function = Integrator(
                                    integration_type= ADAPTIVE,
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    # val = float(I.execute(10)[0])
    P = process( pathway= [I])
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
            function = Integrator(
                                    integration_type= CONSTANT,
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    # val = float(I.execute(10)[0])
    P = process( pathway= [I])
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
            function = Integrator(
                                    integration_type= DIFFUSION,
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    # val = float(I.execute(10)[0])
    P = process( pathway= [I])
    val = float(P.execute(10))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------

# INVALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Linear

# def test_integrator_linear():
#
#     # SHOULD CAUSE AN ERROR??
#
#     with pytest.raises(FunctionError) as error_text:
#         I = IntegratorMechanism(
#                 name='IntegratorMechanism',
#                 function = Linear(),
#                 time_scale=TimeScale.TIME_STEP
#                )
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
            function = Integrator(
                                    integration_type= SIMPLE,
                                  )
           )
    P = process( pathway= [I])
    val = float(P.execute(10.0))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 2
# input = list of length 1
def test_integrator_input_list():
    I = IntegratorMechanism(
            name='IntegratorMechanism',
            function = Integrator(
                                    integration_type= SIMPLE,
                                  )
           )
    P = process( pathway= [I])
    val = float(P.execute([10.0]))
    assert val == 10.0

# ------------------------------------------------------------------------------------------------
# TEST 3
# input = list of length 5

def test_integrator_input_list_len_5():

    I = IntegratorMechanism(
            name='IntegratorMechanism',
            default_input_value= [0,0,0,0,0],
            function = Integrator(
                                integration_type= SIMPLE,
                                )
           )
    P = process( pathway= [I])
    val = P.execute([10.0, 5.0, 2.0, 1.0, 0.0])
    expected_output= [10.0, 5.0, 2.0, 1.0, 0.0]

    for i in range(len(expected_output)):
        v = val[i]
        e = expected_output[i]
        np.testing.assert_allclose(v, e, atol=1e-08, err_msg='Failed on expected_output[{0}]'.format(i))

# ------------------------------------------------------------------------------------------------