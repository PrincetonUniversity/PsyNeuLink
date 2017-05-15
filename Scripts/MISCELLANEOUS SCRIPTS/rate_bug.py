from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import Integrator
from PsyNeuLink.Globals.Utilities import *


def test_DDM_rate_int():
    stim = 10
    D = DDM(
            name='DDM',
            function = Integrator(
                                    integration_type= DIFFUSION,
                                    noise = 0.0,
                                    rate = 5,
                                    time_step_size=1.0
                                  ),
            time_scale=TimeScale.TIME_STEP
           )
    P = process(pathway=[D])
    # val = float(D.execute(stim)[0])
    val = P.execute(stim)
    return val

def test_DDM_rate_list_len_1():
    stim = 10
    D_2 = DDM(
        name='DDM',
        function=Integrator(
            integration_type=DIFFUSION,
            noise=0.0,
            rate=[5],
            time_step_size=1.0
        ),
        time_scale=TimeScale.TIME_STEP
    )
    P_2 = process(pathway=[D_2])
    # val = float(D_2.execute(stim)[0])
    val = P_2.execute(stim)
    return val

print(test_DDM_rate_int())
print(test_DDM_rate_list_len_1())
print(test_DDM_rate_int())