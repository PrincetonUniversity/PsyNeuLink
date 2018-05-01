import numpy as np
import pytest

from psyneulink.components.component import ComponentError
from psyneulink.components.functions.function import FunctionError
from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, Reinforcement, SoftMax, UserDefinedFunction
from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist, UniformToNormalDist
from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.transfermechanism import TransferError, TransferMechanism
from psyneulink.globals.utilities import UtilitiesError
from psyneulink.components.process import Process
from psyneulink.components.system import System

VECTOR_SIZE=4

class TestTransferMechanismInputs:
    # VALID INPUTS

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    def test_transfer_mech_inputs_list_of_ints(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [10 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
        assert len(T.size) == 1 and T.size[0] == VECTOR_SIZE and isinstance(T.size[0], np.integer)
        # this test assumes size is returned as a 1D array: if it's not, then several tests in this file must be changed

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    def test_transfer_mech_inputs_list_of_floats(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [10.0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])

    #@pytest.mark.mechanism
    #@pytest.mark.transfer_mechanism
    # def test_transfer_mech_inputs_list_of_fns(self):
    #
    #     T = TransferMechanism(
    #         name='T',
    #         default_variable=[0, 0, 0, 0],
    #         integrator_mode=True
    #     )
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()])
    #     assert np.allclose(val, [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

    # @pytest.mark.mechanism
    # @pytest.mark.transfer_mechanism
    # def test_transfer_mech_variable_3D_array(self):
    #
    #     T = TransferMechanism(
    #         name='T',
    #         default_variable=[[[0, 0, 0, 0]], [[1, 1, 1, 1]]],
    #         integrator_mode=True
    #     )
    #     np.testing.assert_array_equal(T.instance_defaults.variable, np.array([[[0, 0, 0, 0]], [[1, 1, 1, 1]]]))

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_variable_none_size_none(self):

        T = TransferMechanism(
            name='T'
        )
        assert len(T.instance_defaults.variable) == 1 and T.instance_defaults.variable[0] == 0

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_inputs_list_of_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                integrator_mode=True
            )
            T.execute(["one", "two", "three", "four"])
        assert "has non-numeric entries" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_inputs_mismatched_with_default_longer(self):
        with pytest.raises(MechanismError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                integrator_mode=True
            )
            T.execute([1, 2, 3, 4, 5])
        assert "does not match required length" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_inputs_mismatched_with_default_shorter(self):
        with pytest.raises(MechanismError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0, 0, 0],
                integrator_mode=True
            )
            T.execute([1, 2, 3, 4, 5])
        assert "does not match required length" in str(error_text.value)


class TestTransferMechanismNoise:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise")
    def test_transfer_mech_array_var_float_noise(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=5.0,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[5.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_len_1_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=NormalDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.41059850193837233, 0.144043571160878, 1.454273506962975, 0.7610377251469934]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_array_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=[NormalDist().function, NormalDist().function, NormalDist().function, NormalDist().function],
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        expected = [0.7610377251469934, 0.12167501649282841, 0.44386323274542566, 0.33367432737426683]
        for i in range(len(val[0])):
            assert val[0][i] ==  expected[i]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise2")
    def test_transfer_mech_array_var_normal_array_noise2(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=[5.0 for i in range(VECTOR_SIZE)],
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[5.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_mismatched_shape_noise(self):
        with pytest.raises(MechanismError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0],
                function=Linear(),
                noise=[5.0, 5.0, 5.0],
                smoothing_factor=0.1,
                integrator_mode=True
            )
            T.execute()
        assert 'Noise parameter' in str(error_text.value) and "does not match default variable" in str(
                error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_mismatched_shape_noise_2(self):
        with pytest.raises(MechanismError) as error_text:

            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0],
                function=Linear(),
                noise=[5.0, 5.0],
                smoothing_factor=0.1,
                integrator_mode=True
            )
            T.execute()
        assert 'Noise parameter' in str(error_text.value) and "does not match default variable" in str(error_text.value)


class TestDistributionFunctions:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=NormalDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.41059850193837233, 0.144043571160878, 1.454273506962975, 0.7610377251469934]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_noise_standard_dev_error(self):
        with pytest.raises(FunctionError) as error_text:
            standard_deviation = -2.0
            T = TransferMechanism(
                name="T",
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                noise=NormalDist(standard_dev=standard_deviation).function,
                smoothing_factor=1.0,
                integrator_mode=True
            )

        assert "The standard_dev parameter" in str(error_text) and "must be greater than zero" in str(error_text)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_exponential_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=ExponentialDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.4836021009022533, 1.5688961399691683, 0.7526741095365884, 0.8394328467388229]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_uniform_to_normal_noise(self):
        try:
            import scipy
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                noise=UniformToNormalDist().function,
                smoothing_factor=1.0
            )
            np.random.seed(22)
            val = T.execute([0, 0, 0, 0])
            assert np.allclose(val, [[-0.81177443, -0.04593492, -0.20051725, 1.07665147]])
        except:
            with pytest.raises(FunctionError) as error_text:
                T = TransferMechanism(
                    name='T',
                    default_variable=[0, 0, 0, 0],
                    function=Linear(),
                    noise=UniformToNormalDist().function,
                    smoothing_factor=1.0
                )
            assert "The UniformToNormalDist function requires the SciPy package." in str(error_text)


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Uniform_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=UniformDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.3834415188257777, 0.7917250380826646, 0.5288949197529045, 0.5680445610939323]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Gamma_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=GammaDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.4836021009022533, 1.5688961399691683, 0.7526741095365884, 0.8394328467388229]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Wald_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=WaldDist().function,
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[1.3939555850782692, 0.25118783985272053, 1.2272797824363235, 0.1190661760253029]])


class TestTransferMechanismFunctions:

    def tests_valid_udf_1d_variable(self):
        def double_all_elements(variable):
            return np.array(variable)*2

        T = TransferMechanism(name='T-udf',
                              default_variable=[[0.0, 0.0]],
                              function=UserDefinedFunction(custom_function=double_all_elements))
        result = T.execute([[1.0, 2.0]])
        assert np.allclose(result, [[2.0, 4.0]])

    def tests_valid_udf_2d_variable(self):
        def double_all_elements(variable):
            return np.array(variable)*2

        T = TransferMechanism(name='T-udf',
                              default_variable=[[0.0, 0.0], [0.0, 0.0]],
                              function=UserDefinedFunction(custom_function=double_all_elements))
        result = T.execute([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(result, [[2.0, 4.0], [6.0, 8.0]])

    def tests_invalid_udf(self):
        def sum_all_elements(variable):
            return sum(np.array(variable))

        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(name='T-udf',
                                  default_variable=[[0.0, 0.0]],
                                  function=UserDefinedFunction(custom_function=sum_all_elements))
        assert "value returned by the Python function, method, or UDF specified" in str(error_text.value) \
               and "must be the same shape" in str(error_text.value) \
               and "as its 'variable'" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Logistic")
    def test_transfer_mech_logistic_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Logistic(),
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[0.5 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Exponential")
    def test_transfer_mech_exponential_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Exponential(),
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[1.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism SoftMax")
    def test_transfer_mech_softmax_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=SoftMax(),
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[1.0/VECTOR_SIZE for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=NormalDist(),
                smoothing_factor=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_reinforcement_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Reinforcement(),
                smoothing_factor=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_integrator_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=ConstantIntegrator(),
                smoothing_factor=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_reduce_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Reduce(),
                smoothing_factor=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


class TestTransferMechanismTimeConstant:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_smoothing_factor_0_8(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            smoothing_factor=0.8,
            integrator_mode=True
        )
        val = T.execute([1 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[0.8 for i in range(VECTOR_SIZE)]])
        val = T.execute([1 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[0.96 for i in range(VECTOR_SIZE)]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=1")
    def test_transfer_mech_smoothin_factor_1_0(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            smoothing_factor=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[1.0 for i in range(VECTOR_SIZE)]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=0")
    def test_transfer_mech_smoothing_factor_0_0(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            smoothing_factor=0.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[0.0 for i in range(VECTOR_SIZE)]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_smoothing_factor_0_8_initial_0_5(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            smoothing_factor=0.8,
            initial_value=np.array([[.5, .5, .5, .5]]),
            integrator_mode=True
        )
        val = T.execute([1, 1, 1, 1])
        assert np.allclose(val, [[0.9, 0.9, 0.9, 0.9]])
        T.noise = 10
        val = T.execute([1, 2, -3, 0])
        assert np.allclose(val, [[10.98, 11.78, 7.779999999999999, 10.18]]) # testing noise changes to an integrator


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_smoothing_factor_0_8_list(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                smoothing_factor=[0.8, 0.8, 0.8, 0.8],
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1])
        assert (
            "smoothing_factor parameter" in str(error_text.value)
            and "must be a float" in str(error_text.value)
        )


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_smoothing_factor_2(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                smoothing_factor=2,
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1])
        assert (
            "smoothing_factor parameter" in str(error_text.value)
            and "must be a float between 0 and 1" in str(error_text.value)
        )


class TestTransferMechanismSize:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_int_check_var(self):
        T = TransferMechanism(
            name='T',
            size=4
        )
        assert len(T.instance_defaults.variable) == 1 and (T.instance_defaults.variable[0] == [0., 0., 0., 0.]).all()
        assert len(T.size) == 1 and T.size[0] == 4 and isinstance(T.size[0], np.integer)


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_int_inputs_ints(self):
        T = TransferMechanism(
            name='T',
            size=4
        )
        val = T.execute([10, 10, 10, 10])
        assert np.allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    # ------------------------------------------------------------------------------------------------
    # TEST 3
    # size = int, variable = list of floats

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_int_inputs_floats(self):
        T = TransferMechanism(
            name='T',
            size=VECTOR_SIZE
        )
        val = T.execute([10.0 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])

    # ------------------------------------------------------------------------------------------------
    # TEST 4
    # size = int, variable = list of functions

    #@pytest.mark.mechanism
    #@pytest.mark.transfer_mechanism
    # def test_transfer_mech_size_int_inputs_fns(self):
    #     T = TransferMechanism(
    #         name='T',
    #         size=4,
    #         integrator_mode=True
    #     )
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()])
    #     assert np.allclose(val, [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

    # ------------------------------------------------------------------------------------------------
    # TEST 5
    # size = float, check if variable is an array of zeros

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_float_inputs_check_var(self):
        T = TransferMechanism(
            name='T',
            size=4.0,
        )
        assert len(T.instance_defaults.variable) == 1 and (T.instance_defaults.variable[0] == [0., 0., 0., 0.]).all()
        assert len(T.size == 1) and T.size[0] == 4.0 and isinstance(T.size[0], np.integer)

    # ------------------------------------------------------------------------------------------------
    # TEST 6
    # size = float, variable = list of ints

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_float_inputs_ints(self):
        T = TransferMechanism(
            name='T',
            size=4.0
        )
        val = T.execute([10, 10, 10, 10])
        assert np.allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    # ------------------------------------------------------------------------------------------------
    # TEST 7
    # size = float, variable = list of floats

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_float_inputs_floats(self):
        T = TransferMechanism(
            name='T',
            size=4.0
        )
        val = T.execute([10.0, 10.0, 10.0, 10.0])
        assert np.allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    # ------------------------------------------------------------------------------------------------
    # TEST 8
    # size = float, variable = list of functions

    #@pytest.mark.mechanism
    #@pytest.mark.transfer_mechanism
    # def test_transfer_mech_size_float_inputs_fns(self):
    #     T = TransferMechanism(
    #         name='T',
    #         size=4.0,
    #         integrator_mode=True
    #     )
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()])
    #     assert np.allclose(val, [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # size = list of ints, check that variable is correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_list_of_ints(self):
        T = TransferMechanism(
            name='T',
            size=[2, 3, 4]
        )
        assert len(T.instance_defaults.variable) == 3 and len(T.instance_defaults.variable[0]) == 2 and len(T.instance_defaults.variable[1]) == 3 and len(T.instance_defaults.variable[2]) == 4

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # size = list of floats, check that variable is correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_list_of_floats(self):
        T = TransferMechanism(
            name='T',
            size=[2., 3., 4.]
        )
        assert len(T.instance_defaults.variable) == 3 and len(T.instance_defaults.variable[0]) == 2 and len(T.instance_defaults.variable[1]) == 3 and len(T.instance_defaults.variable[2]) == 4

    # note that this output under the Linear function is useless/odd, but the purpose of allowing this configuration
    # is for possible user-defined functions that do use unusual shapes.

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_var_both_lists(self):
        T = TransferMechanism(
            name='T',
            size=[2., 3.],
            default_variable=[[1, 2], [3, 4, 5]]
        )
        assert len(T.instance_defaults.variable) == 2 and (T.instance_defaults.variable[0] == [1, 2]).all() and (T.instance_defaults.variable[1] == [3, 4, 5]).all()

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # size = int, variable = a compatible 2D array: check that variable is correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_scalar_var_2d(self):
        T = TransferMechanism(
            name='T',
            size=2,
            default_variable=[[1, 2], [3, 4]]
        )
        assert len(T.instance_defaults.variable) == 2 and (T.instance_defaults.variable[0] == [1, 2]).all() and (T.instance_defaults.variable[1] == [3, 4]).all()
        assert len(T.size) == 2 and T.size[0] == 2 and T.size[1] == 2

    # ------------------------------------------------------------------------------------------------
    # TEST 13
    # variable = a 2D array: check that variable is correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_var_2d_array(self):
        T = TransferMechanism(
            name='T',
            default_variable=[[1, 2], [3, 4]]
        )
        assert len(T.instance_defaults.variable) == 2 and (T.instance_defaults.variable[0] == [1, 2]).all() and (T.instance_defaults.variable[1] == [3, 4]).all()

    # ------------------------------------------------------------------------------------------------
    # TEST 14
    # variable = a 1D array, size does not match: check that variable and output are correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_var_1D_size_wrong(self):
        T = TransferMechanism(
            name='T',
            default_variable=[1, 2, 3, 4],
            size=2
        )
        assert len(T.instance_defaults.variable) == 1 and (T.instance_defaults.variable[0] == [1, 2, 3, 4]).all()
        val = T.execute([10.0, 10.0, 10.0, 10.0])
        assert np.allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    # ------------------------------------------------------------------------------------------------
    # TEST 15
    # variable = a 1D array, size does not match again: check that variable and output are correct

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_var_1D_size_wrong_2(self):
        T = TransferMechanism(
            name='T',
            default_variable=[1, 2, 3, 4],
            size=[2, 3, 4]
        )
        assert len(T.instance_defaults.variable) == 1 and (T.instance_defaults.variable[0] == [1, 2, 3, 4]).all()
        val = T.execute([10.0, 10.0, 10.0, 10.0])
        assert np.allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    # ------------------------------------------------------------------------------------------------
    # TEST 16
    # size = int, variable = incompatible array, check variable

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_var_incompatible1(self):
        T = TransferMechanism(
            name='T',
            size=2,
            default_variable=[[1, 2], [3, 4, 5]]
        )
        assert (T.instance_defaults.variable[0] == [1, 2]).all() and (T.instance_defaults.variable[1] == [3, 4, 5]).all() and len(T.instance_defaults.variable) == 2

    # ------------------------------------------------------------------------------------------------
    # TEST 17
    # size = array, variable = incompatible array, check variable

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_var_incompatible2(self):
        T = TransferMechanism(
            name='T',
            size=[2, 2],
            default_variable=[[1, 2], [3, 4, 5]]
        )
        assert (T.instance_defaults.variable[0] == [1, 2]).all() and (T.instance_defaults.variable[1] == [3, 4, 5]).all() and len(T.instance_defaults.variable) == 2

    # ------------------------------------------------------------------------------------------------

    # INVALID INPUTS

    # ------------------------------------------------------------------------------------------------
    # TEST 1
    # size = 0, check less-than-one error

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_zero(self):
        with pytest.raises(ComponentError) as error_text:
            T = TransferMechanism(
                name='T',
                size=0,
            )
        assert "is not a positive number" in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 2
    # size = -1.0, check less-than-one error

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_negative_one(self):
        with pytest.raises(ComponentError) as error_text:
            T = TransferMechanism(
                name='T',
                size=-1.0,
            )
        assert "is not a positive number" in str(error_text.value)

    # this test below and the (currently commented) test immediately after it _may_ be deprecated if we ever fix
    # warnings to be no longer fatal. At the time of writing (6/30/17, CW), warnings are always fatal.

    # the test commented out here is similar to what we'd want if we got warnings to be non-fatal
    # and error_text was correctly representing the warning. For now, the warning is hidden under
    # a verbosity preference
    # def test_transfer_mech_size_bad_float(self):
    #     with pytest.raises(UserWarning) as error_text:
    #         T = TransferMechanism(
    #             name='T',
    #             size=3.5,
    #         )
    #     assert "cast to integer, its value changed" in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 4
    # size = 2D array, check too-many-dimensions warning

    # def test_transfer_mech_size_2d(self):
    #     with pytest.raises(UserWarning) as error_text:
    #         T = TransferMechanism(
    #             name='T',
    #             size=[[2]],
    #         )
    #     assert "had more than one dimension" in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 5
    # size = 2D array, check variable is correctly instantiated

    # for now, since the test above doesn't work, we use this tesT.6/30/17 (CW)
    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_2d(self):
        T = TransferMechanism(
            name='T',
            size=[[2]],
        )
        assert len(T.instance_defaults.variable) == 1 and len(T.instance_defaults.variable[0]) == 2
        assert len(T.size) == 1 and T.size[0] == 2 and len(T.params['size']) == 1 and T.params['size'][0] == 2

class TestTransferMechanismMultipleInputStates:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.mimo
    def test_transfer_mech_2d_variable(self):
        from psyneulink.globals.keywords import MEAN
        T = TransferMechanism(
            name='T',
            function=Linear(slope=2.0, intercept=1.0),
            default_variable=[[0.0, 0.0], [0.0, 0.0]],
            output_states=[MEAN]
        )
        val = T.execute([[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_2d_variable_noise(self):
        T = TransferMechanism(
            name='T',
            function=Linear(slope=2.0, intercept=1.0),
            noise=NormalDist().function,
            default_variable=[[0.0, 0.0], [0.0, 0.0]]
        )
        val = T.execute([[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.mimo
    def test_multiple_output_states_for_multiple_input_states(self):
        T = TransferMechanism(input_states=['a','b','c'])
        val = T.execute([[1],[2],[3]])
        assert len(T.variable)==3
        assert all(a==b for a,b in zip(val, [[ 1.],[ 2.],[ 3.]]))
        assert len(T.output_states)==3
        assert all(a==b for a,b in zip(T.output_values,val))

    # @pytest.mark.mechanism
    # @pytest.mark.transfer_mechanism
    # @pytest.mark.mimo
    # def test_OWNER_VALUE_standard_output_state(self):
    #     from psyneulink.globals.keywords import OWNER_VALUE
    #     T = TransferMechanism(input_states=[[[0],[0]],'b','c'],
    #                               output_states=OWNER_VALUE)
    #     print(T.value)
    #     val = T.execute([[[1],[4]],[2],[3]])
    #     expected_val = [[[1],[4]],[2],[3]]
    #     assert len(T.output_states)==1
    #     assert len(T.output_states[OWNER_VALUE].value)==3
    #     assert all(all(a==b for a,b in zip(x,y)) for x,y in zip(val, expected_val))

class TestIntegratorMode:
    def test_previous_value_persistence_execute(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)

        assert np.allclose(T.previous_value, 0.5)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        T.execute(1.0)
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        assert np.allclose(T.previous_value, 0.55)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        T.execute(1.0)
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, 0.595)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

    def test_previous_value_persistence_run(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)
        P = Process(name="P",
                    pathway=[T])
        S = System(name="S",
                   processes=[P])

        assert np.allclose(T.previous_value, 0.5)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        S.run(inputs={T: 1.0}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, 0.595)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        S.run(inputs={T: 2.0}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.595 + 0.1*2.0 + 0.0 = 0.7355  --->  previous value = 0.7355
        # linear fn: 0.7355*1.0 = 0.7355
        # Trial 4
        # integration: 0.9*0.7355 + 0.1*2.0 + 0.0 = 0.86195  --->  previous value = 0.86195
        # linear fn: 0.86195*1.0 = 0.86195

        assert np.allclose(T.previous_value, 0.86195)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

    def test_previous_value_reinitialize_execute(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)

        assert np.allclose(T.previous_value, 0.5)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)
        T.execute(1.0)
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        assert np.allclose(T.previous_value, 0.55)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)
        assert np.allclose(T.value, 0.55)

        # Reset integrator_function ONLY
        T.integrator_function.reinitialize(0.6)

        assert np.allclose(T.previous_value, 0.6)   # previous_value is a property that looks at integrator_function
        assert np.allclose(T.initial_value, 0.5)    # initial_value is on mechanism only, does not update with int_fun
        assert np.allclose(T.integrator_function.initializer, 0.6)  # initializer is on integrator_function
        assert np.allclose(T.value, 0.55)           # on mechanism only, so does not update until execution

        T.execute(1.0)
        # integration: 0.9*0.6 + 0.1*1.0 + 0.0 = 0.64  --->  previous value = 0.55
        # linear fn: 0.64*1.0 = 0.64
        assert np.allclose(T.previous_value, 0.64)   # property that looks at integrator_function
        assert np.allclose(T.initial_value, 0.5)     # initial_value is on mechanism only, and does not update with exec
        assert np.allclose(T.integrator_function.initializer, 0.5)     # initializer does not change with execution
        assert np.allclose(T.value, 0.64)            # on mechanism, but updates with execution

        T.reinitialize(0.4)
        # linear fn: 0.4*1.0 = 0.4
        assert np.allclose(T.previous_value, 0.4)   # property that looks at integrator, which updated with mech reset
        assert np.allclose(T.initial_value, 0.4)    # updates because mechanism was reset
        assert np.allclose(T.integrator_function.initializer, 0.4)  # on integrator fun, but updates when mech resets
        assert np.allclose(T.value, 0.4)  # on mechanism, but updates with mech reset

        T.execute(1.0)
        # integration: 0.9*0.4 + 0.1*1.0 + 0.0 = 0.46  --->  previous value = 0.46
        # linear fn: 0.46*1.0 = 0.46
        assert np.allclose(T.previous_value, 0.46)  # property that looks at integrator, which updated with mech exec
        assert np.allclose(T.initial_value, 0.4)                    # on mech, does not update with exec
        assert np.allclose(T.integrator_function.initializer, 0.4)  # initializer does not change with execution
        assert np.allclose(T.value, 0.46)  # on mechanism, but updates with exec

    def test_reinitialize_run(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)
        P = Process(name="P",
                    pathway=[T])
        S = System(name="S",
                   processes=[P])

        assert np.allclose(T.previous_value, 0.5)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        S.run(inputs={T: 1.0}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, 0.595)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

        T.integrator_function.reinitialize(0.9)

        assert np.allclose(T.previous_value, 0.9)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.9)
        assert np.allclose(T.value, 0.595)

        T.reinitialize(0.5)

        assert np.allclose(T.previous_value, 0.5)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)
        assert np.allclose(T.value, 0.5)

        S.run(inputs={T: 1.0}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, 0.595)
        assert np.allclose(T.initial_value, 0.5)
        assert np.allclose(T.integrator_function.initializer, 0.5)

    def test_reinitialize_run_array(self):
        T = TransferMechanism(name="T",
                              default_variable=[0.0, 0.0, 0.0],
                              initial_value=[0.5, 0.5, 0.5],
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)
        P = Process(name="P",
                    pathway=[T])
        S = System(name="S",
                   processes=[P])

        assert np.allclose(T.previous_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.initial_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.integrator_function.initializer, [0.5, 0.5, 0.5])

        S.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, [0.595, 0.595, 0.595])
        assert np.allclose(T.initial_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.integrator_function.initializer, [0.5, 0.5, 0.5])

        T.integrator_function.reinitialize([0.9, 0.9, 0.9])

        assert np.allclose(T.previous_value, [0.9, 0.9, 0.9])
        assert np.allclose(T.initial_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.integrator_function.initializer, [0.9, 0.9, 0.9])
        assert np.allclose(T.value, [0.595, 0.595, 0.595])

        T.reinitialize([0.5, 0.5, 0.5])

        assert np.allclose(T.previous_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.initial_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.integrator_function.initializer, [0.5, 0.5, 0.5])
        assert np.allclose(T.value, [0.5, 0.5, 0.5])

        S.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, [0.595, 0.595, 0.595])
        assert np.allclose(T.initial_value, [0.5, 0.5, 0.5])
        assert np.allclose(T.integrator_function.initializer, [0.5, 0.5, 0.5])

    def test_reinitialize_run_2darray(self):

        initial_val = [[0.5, 0.5, 0.5]]
        T = TransferMechanism(name="T",
                              default_variable=[[0.0, 0.0, 0.0]],
                              initial_value=initial_val,
                              integrator_mode=True,
                              smoothing_factor=0.1,
                              noise=0.0)
        P = Process(name="P",
                    pathway=[T])
        S = System(name="S",
                   processes=[P])

        assert np.allclose(T.previous_value, initial_val)
        assert np.allclose(T.initial_value, initial_val)
        assert np.allclose(T.integrator_function.initializer, initial_val)

        S.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, [0.595, 0.595, 0.595])
        assert np.allclose(T.initial_value, initial_val)
        assert np.allclose(T.integrator_function.initializer, initial_val)

        T.integrator_function.reinitialize([0.9, 0.9, 0.9])

        assert np.allclose(T.previous_value, [0.9, 0.9, 0.9])
        assert np.allclose(T.initial_value, initial_val)
        assert np.allclose(T.integrator_function.initializer, [0.9, 0.9, 0.9])
        assert np.allclose(T.value, [0.595, 0.595, 0.595])

        T.reinitialize(initial_val)

        assert np.allclose(T.previous_value, initial_val)
        assert np.allclose(T.initial_value, initial_val)
        assert np.allclose(T.integrator_function.initializer, initial_val)
        assert np.allclose(T.value, initial_val)

        S.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.previous_value, [0.595, 0.595, 0.595])
        assert np.allclose(T.initial_value, initial_val)
        assert np.allclose(T.integrator_function.initializer, initial_val)

    def test_reinitialize_not_integrator(self):

        with pytest.raises(MechanismError) as err_txt:
            T_not_integrator = TransferMechanism()
            T_not_integrator.execute(1.0)
            T_not_integrator.reinitialize(0.0)
        assert "not allowed because this Mechanism is not stateful." in str(err_txt) \
               and "try setting the integrator_mode argument to True." in str(err_txt)

    def test_switch_mode(self):
        T = TransferMechanism(integrator_mode=True)
        P = Process(pathway=[T])
        S = System(processes=[P])
        integrator_function = T.integrator_function

        # T starts with integrator_mode = True; confirm that T behaves correctly
        S.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.value, [[0.875]])

        assert T.integrator_mode is True
        assert T.integrator_function is integrator_function

        # Switch integrator_mode to False; confirm that T behaves correctly
        T.integrator_mode = False

        assert T.integrator_mode is False
        assert T.integrator_function is None

        S.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.value, [[1.0]])

        # Switch integrator_mode BACK to True; confirm that T picks up where it left off
        T.integrator_mode = True

        assert T.integrator_mode is True
        assert T.integrator_function is integrator_function

        S.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.value, [[0.984375]])



class TestClip:
    def test_clip_float(self):
        T = TransferMechanism(clip=[-2.0, 2.0])
        assert np.allclose(T.execute(3.0), 2.0)
        assert np.allclose(T.execute(-3.0), -2.0)

    def test_clip_array(self):
        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        assert np.allclose(T.execute([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])

    def test_clip_2d_array(self):
        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        assert np.allclose(T.execute([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
                           [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])

