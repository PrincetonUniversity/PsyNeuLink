from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.functions.function import Linear

class TestParameterStates:
    def test_inspect_function_params(self):
        A = TransferMechanism()
        # B = TransferMechanism()
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