from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Functions.Function import *
import numpy as np
from PsyNeuLink.Globals.Keywords import *
import pytest


# ======================================= NOISE TESTS ============================================

# VALID NOISE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# noise = Single float

def test_DDM_zero_noise():
    stim = 10
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
    assert val == 10

# ------------------------------------------------------------------------------------------------
# TEST 2
# noise = Single float

def test_DDM_noise_0_5():
    stim = 10
    T = DDM(
            name='T',
            function = Integrator(
                                    integration_type= DIFFUSION,
                                    noise = 0.5,
                                    rate = 1.0,
                                    time_step_size=1.0
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    val = float(T.execute(stim)[0])
    assert val == 9.892974291631234

# ------------------------------------------------------------------------------------------------
# TEST 3
# noise = Single float

def test_DDM_noise_2_0():
    stim = 10
    T = DDM(
        name='T',
        function=Integrator(
            integration_type=DIFFUSION,
            noise=2.0,
            rate=1.0,
            time_step_size=1.0
        ),
        time_scale=TimeScale.TIME_STEP
    )
    val = float(T.execute(stim)[0])
    assert val == 9.785948583262465
# ------------------------------------------------------------------------------------------------

# INVALID NOISE:

# ------------------------------------------------------------------------------------------------
# TEST 1
# noise = Single int

def test_DDM_noise_2():
    with pytest.raises(FunctionError) as error_text:
        stim = 10
        T = DDM(
            name='T',
            function=Integrator(
                integration_type=DIFFUSION,
                noise=2,
                rate=1.0,
                time_step_size=1.0
            ),
            time_scale=TimeScale.TIME_STEP
        )
        val = float(T.execute(stim)[0])
    assert "noise parameter" in str(error_text.value) \
           and "must be a float, function, array or list of floats, or array or list of functions" in str(error_text.value)