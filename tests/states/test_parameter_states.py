from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.functions.function import Linear

class TestParameterStates:
    def test_inspect_function_params_slope(self):
        # A = IntegratorMechanism(function=Linear)
        A = TransferMechanism()
        print()
        print(A.function)
        print(A._function)
        print("A.function_object.slope --> ", A.function_object.slope)
        print("A.function_object._slope --> ", A.function_object._slope)
        print("A.function_object.mod_slope --> ", A.function_object.mod_slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - SETTING A.function_object.slope = 0.2 - - - - -")
        A.function_object.slope = 0.2
        print("A.user_params -->", A.user_params)
        print("A.function_object.slope --> ", A.function_object.slope)
        print("A.function_object._slope --> ", A.function_object._slope)
        print("A.function_object.mod_slope --> ", A.function_object.mod_slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - EXECUTING A - - - - -")
        A.execute()
        print("A.function_object.slope --> ", A.function_object.slope)
        print("A.function_object._slope --> ", A.function_object._slope)
        print("A.function_object.mod_slope --> ", A.function_object.mod_slope)
        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)

    def test_inspect_mechanism_params_noise(self):
        print("starting second test:")
        B = TransferMechanism()
        # B = TransferMechanism()
        print()
        print(B.function)
        print(B._function)
        print("B.noise --> ", B.noise)
        print("B._noise --> ", B._noise)
        print("B.mod_noise --> ", B.mod_noise)
        # print("B.base_value_noise --> ", B.base_value_noise)
        print("- - - - - SETTING B.noise = 0.2 - - - - -")
        B.noise = 0.2
        print("B.user_params -->", B.user_params)
        print("B.noise --> ", B.noise)
        print("B._noise --> ", B._noise)
        print("B.mod_noise --> ", B.mod_noise)
        # print("B.base_value_noise --> ", B.base_value_noise)
        print("- - - - - EXECUTING A - - - - -")
        B.execute()
        print("B.noise --> ", B.noise)
        print("B._noise --> ", B._noise)
        print("B.mod_noise --> ", B.mod_noise)
        # print("B.base_value_noise --> ", B.base_value_noise)