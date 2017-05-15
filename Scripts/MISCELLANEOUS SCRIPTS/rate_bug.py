from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Functions.Function import Integrator,BogaczEtAl
from PsyNeuLink.Globals.Utilities import *
from pprint import pprint


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
    print('BEFORE EXECUTION ')
    pprint(D.function_object.__dict__)
    val = P.execute(stim)
    print("AFTER EXECUTION ")
    pprint(D.function_object.__dict__)
    print("=======================================================================================")

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
    print('BEFORE EXECUTION ')
    pprint(D_2.function_object.__dict__)
    val = P_2.execute(stim)
    print("AFTER EXECUTION ")
    pprint(D_2.function_object.__dict__)
    print("=======================================================================================")


def test_DDM_rate_int_bog():
    stim = 10
    D_bog = DDM(
            name='DDM',
            function = BogaczEtAl(drift_rate=5)
           )
    P_bog = process(pathway=[D_bog])
    # val = float(D.execute(stim)[0])
    print('BEFORE EXECUTION ')
    pprint(D_bog.function_object.__dict__)
    val = P_bog.execute(stim)
    print("AFTER EXECUTION ")
    pprint(D_bog.function_object.__dict__)
    print("=======================================================================================")

    return val

def test_DDM_rate_list_len_1_bog():
    stim = 10
    D_2_bog = DDM(
        name='DDM',
        function=BogaczEtAl(drift_rate=[5, 5])
    )
    P_2_bog = process(pathway=[D_2_bog])
    # val = float(D_2_bog.execute(stim)[0])
    print('BEFORE EXECUTION ')
    pprint(D_2_bog.function_object.__dict__)
    val = P_2_bog.execute(stim)
    print("AFTER EXECUTION ")
    pprint(D_2_bog.function_object.__dict__)
    print("=======================================================================================")

    return val
#
# print("Bog 1 = ", test_DDM_rate_int_bog())
# print("Bog 2 = ",test_DDM_rate_list_len_1_bog())
# print("Bog 3 = ",test_DDM_rate_int_bog())

print("Integrator 1 = ",test_DDM_rate_int())
print("Integrator 2 = ", test_DDM_rate_list_len_1())
print("Integrator 3 = ", test_DDM_rate_int())