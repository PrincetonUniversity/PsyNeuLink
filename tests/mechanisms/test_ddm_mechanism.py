from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
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

def test_DDM():
    stim = 100
    T = DDM(
            name='T',
            function = Integrator(
                                    integration_type= DIFFUSION,
                                    noise = 0.0,
                                    rate = 1.0,
                                    time_step_size=1.0
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    val = float(T.execute(stim)[0])
    assert val == -0.10702570836876696