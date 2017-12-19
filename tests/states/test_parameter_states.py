from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.components.functions.function import Linear
from psyneulink.components.functions.function import Logistic

class TestParameterStates:
    def test_inspect_function_params_slope(self):

        A = TransferMechanism()
        B = TransferMechanism()
        # C = TransferMechanism(function=Linear(slope=2))

        assert A.function_object.slope == 1.0
        assert B.function_object.slope == 1.0
        assert A.function_object._slope == 1.0
        assert B.function_object._slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        assert A.noise == 0.0
        assert B.noise == 0.0
        assert A._noise == 0.0
        assert B._noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        A.function_object.slope = 0.2

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 1.0
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 1.0
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        A.noise = 0.5

        assert A.noise == 0.5
        assert B.noise == 0.0
        assert A._noise == 0.5
        assert B._noise == 0.0
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        B.function_object.slope = 0.7

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 0.7
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 0.7
        assert A.mod_slope == [1.0]
        assert B.mod_slope == [1.0]

        B.noise = 0.6

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A._noise == 0.5
        assert B._noise == 0.6
        assert A.mod_noise == 0.0
        assert B.mod_noise == 0.0

        # print("A.function_object.base_value_slope --> ", A.function_object.base_value_slope)
        print("- - - - - EXECUTING A ", A.name, "- - - - -")
        print(A.execute(1.0))
        assert A.mod_slope == [0.2]
        print("- - - - - EXECUTING B - - - - -")
        print(B.execute(1.0))

        assert A.function_object.slope == 0.2
        assert B.function_object.slope == 0.7
        assert A.function_object._slope == 0.2
        assert B.function_object._slope == 0.7
        assert A.mod_slope == [0.2]
        assert B.mod_slope == [0.7]

        assert A.noise == 0.5
        assert B.noise == 0.6
        assert A._noise == 0.5
        assert B._noise == 0.6
        assert A.mod_noise == 0.5
        assert B.mod_noise == 0.6

    def test_inspect_mechanism_params_noise(self):
        print("\n\n========================== starting second test ==========================\n\n")
        C = TransferMechanism(function=Linear(slope=2.5))
        print(C.function_object)
        print(C.paramClassDefaults)
        print(C.function_object.paramClassDefaults)
        # C = TransferMechanism()
        print("C.function_object.slope --> ", C.function_object.slope)
        print("C.function_object._slope --> ", C.function_object._slope)
        print("executing: ", C.execute(1.0))
        C.function_object.slope=2.0
        print("setting slope to 2.0")
        print("C.function_object.slope --> ", C.function_object.slope)
        print("C.function_object._slope --> ", C.function_object._slope)
        print("executing: ", C.execute(1.0))
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)
        # print("C.base_value_noise --> ", C.base_value_noise)
        print("- - - - - SETTING C.noise = 0.2 - - - - -")
        C.noise = 0.2
        print("C.user_params -->", C.user_params)
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)
        # print("C.base_value_noise --> ", C.base_value_noise)
        print("- - - - - EXECUTING A - - - - -")
        C.execute(1.0)
        print("C.noise --> ", C.noise)
        print("C._noise --> ", C._noise)
        # print("C.mod_noise --> ", C.mod_noise)

    def test_make_property(self):
        print("making mechanism A")
        A = TransferMechanism(function=Logistic)
        print(A.function_object.gain)
        print("\n\n ============================== \n\n")
        print("making mechanism B")
        B = TransferMechanism()