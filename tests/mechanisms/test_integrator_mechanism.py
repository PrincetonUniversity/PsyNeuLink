from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.IntegratorMechanism import IntegratorMechanism

from PsyNeuLink.Components.Functions.Function import *
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest
from PsyNeuLink.Globals.Utilities import *

# ======================================= FUNCTION TESTS ============================================

# VALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Integrator

def test_integrator():
    I = IntegratorMechanism(
            name='IntegratorMechanism',
            function = Integrator(
                                    integration_type= DIFFUSION,
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    val = float(I.execute(10)[0])
    assert val == 0

# INVALID FUNCTIONS:

# ------------------------------------------------------------------------------------------------
# TEST 1
# function = Linear

def test_integrator2():
    # with pytest.raises(FunctionError) as error_text:
    I = IntegratorMechanism(
            name='IntegratorMechanism',
            function = Linear(),
            time_scale=TimeScale.TIME_STEP
           )
    val = float(I.execute(10)[0])
    assert val == 0