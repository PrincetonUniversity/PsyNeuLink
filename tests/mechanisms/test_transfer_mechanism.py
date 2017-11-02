import numpy as np
import pytest
from psyneulink.components.mechanisms.processing.transfermechanism import TransferError

from psyneulink.components.component import ComponentError
from psyneulink.components.functions.function import ConstantIntegrator, Exponential, Linear, Logistic, Reduce, \
    Reinforcement, SoftMax
from psyneulink.components.functions.function import ExponentialDist, GammaDist, NormalDist, UniformDist, WaldDist
from psyneulink.components.mechanisms.mechanism import MechanismError
from psyneulink.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.components.states.inputstate import InputStateError
from psyneulink.globals.utilities import UtilitiesError
from psyneulink.globals.keywords import NAME, MECHANISM, INPUT_STATES, OUTPUT_STATES, PROJECTIONS
from psyneulink.scheduling.timescale import TimeScale

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
            integrator_mode=True
        )
        val = benchmark(T.execute, [10 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[10.0 for i in range(VECTOR_SIZE)]]
        assert len(T.size) == 1 and T.size[0] == VECTOR_SIZE and isinstance(T.size[0], np.integer)
        # this test assumes size is returned as a 1D array: if it's not, then several tests in this file must be changed

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    def test_transfer_mech_inputs_list_of_floats(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True
        )
        val = benchmark(T.execute, [10.0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[10.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    def test_transfer_mech_inputs_list_of_floats_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True
        )
        val = benchmark(T.execute, [10.0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[10.0 for i in range(VECTOR_SIZE)]]

    #@pytest.mark.mechanism
    #@pytest.mark.transfer_mechanism
    # def test_transfer_mech_inputs_list_of_fns(self):
    #
    #     T = TransferMechanism(
    #         name='T',
    #         default_variable=[0, 0, 0, 0],
    #         integrator_mode=True
    #     )
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    #     assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_variable_3D_array(self):

        T = TransferMechanism(
            name='T',
            default_variable=[[[0, 0, 0, 0]],[[1,1,1,1]]],
            integrator_mode=True
        )
        assert len(T.instance_defaults.variable) == 1 and len(T.instance_defaults.variable[0]) == 4 and (T.instance_defaults.variable[0] == 0).all()

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_variable_none_size_none(self):

        T = TransferMechanism(
            name='T'
        )
        assert len(T.instance_defaults.variable) == 1 and len(T.instance_defaults.variable[0]) == 1 and T.instance_defaults.variable[0][0] == 0

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_inputs_list_of_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                integrator_mode=True
            )
            T.execute(["one", "two", "three", "four"]).tolist()
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
            T.execute([1, 2, 3, 4, 5]).tolist()
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
            T.execute([1, 2, 3, 4, 5]).tolist()
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
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[5.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise")
    def test_transfer_mech_array_var_float_noise_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=5.0,
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[5.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_len_1_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=NormalDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[0.41059850193837233, 0.144043571160878, 1.454273506962975, 0.7610377251469934]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_array_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=[NormalDist().function, NormalDist().function, NormalDist().function, NormalDist().function],
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[1.8675579901499675, -0.977277879876411, 0.9500884175255894, -0.1513572082976979]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise2")
    def test_transfer_mech_array_var_normal_array_noise2(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=[5.0 for i in range(VECTOR_SIZE)],
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[5.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise2")
    def test_transfer_mech_array_var_normal_array_noise2_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=[5.0 for i in range(VECTOR_SIZE)],
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[5.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_mismatched_shape_noise(self):
        with pytest.raises(MechanismError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0],
                function=Linear(),
                noise=[5.0, 5.0, 5.0],
                time_constant=0.1,
                integrator_mode=True
            )
            T.execute()
        assert 'noise parameter' in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_mismatched_shape_noise_2(self):
        with pytest.raises(MechanismError) as error_text:

            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0],
                function=Linear(),
                noise=[5.0, 5.0],
                time_constant=0.1,
                integrator_mode=True
            )
            T.execute()
        assert 'noise parameter' in str(error_text.value)


class TestDistributionFunctions:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=NormalDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[0.41059850193837233, 0.144043571160878, 1.454273506962975, 0.7610377251469934]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_exponential_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=ExponentialDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[0.4836021009022533, 1.5688961399691683, 0.7526741095365884, 0.8394328467388229]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Uniform_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=UniformDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[0.3834415188257777, 0.7917250380826646, 0.5288949197529045, 0.5680445610939323]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Gamma_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=GammaDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[0.4836021009022533, 1.5688961399691683, 0.7526741095365884, 0.8394328467388229]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Wald_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=WaldDist().function,
            time_constant=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0]).tolist()
        assert val == [[1.3939555850782692, 0.25118783985272053, 1.2272797824363235, 0.1190661760253029]]


class TestTransferMechanismFunctions:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Logistic")
    def test_transfer_mech_logistic_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Logistic(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[0.5 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Logistic")
    def test_transfer_mech_logistic_fun_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Logistic(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[0.5 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Exponential")
    def test_transfer_mech_exponential_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Exponential(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[1.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Exponential")
    def test_transfer_mech_exponential_fun_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Exponential(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[1.0 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism SoftMax")
    def test_transfer_mech_softmax_fun(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=SoftMax(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[0.25 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism SoftMax")
    def test_transfer_mech_softmax_fun_llvm(self, benchmark):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=SoftMax(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[0.25 for i in range(VECTOR_SIZE)]]

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=NormalDist(),
                time_constant=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_reinforcement_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Reinforcement(),
                time_constant=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_integrator_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=ConstantIntegrator(),
                time_constant=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_reduce_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Reduce(),
                time_constant=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


class TestTransferMechanismTimeConstant:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0_8(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=0.8,
            integrator_mode=True
        )
        val = T.execute([1 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[0.8 for i in range(VECTOR_SIZE)]]
        val = T.execute([1 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[0.96 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0_8_llvm(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=0.8,
            integrator_mode=True
        )
        val = T.execute([1 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[0.8 for i in range(VECTOR_SIZE)]]
        val = T.execute([1 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[0.96 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=1")
    def test_transfer_mech_time_constant_1_0(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[1.0 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=1")
    def test_transfer_mech_time_constant_1_0_llvm(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=1.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1.0 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[1.0 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=0")
    def test_transfer_mech_time_constant_0_0(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=0.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[0.0 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=0")
    def test_transfer_mech_time_constant_0_0_llvm(self, benchmark):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            time_constant=0.0,
            integrator_mode=True
        )
        val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)], bin_execute=True).tolist()
        assert val == [[0.0 for i in range(VECTOR_SIZE)]]


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0_8_initial_0_5(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            initial_value=np.array([[.5, .5, .5, .5]]),
            integrator_mode=True
        )
        val = T.execute([1, 1, 1, 1]).tolist()
        assert val == [[0.9, 0.9, 0.9, 0.9]]
        T.noise = 10
        val = T.execute([1, 2, -3, 0]).tolist()
        assert val == [[10.98, 11.78, 7.779999999999999, 10.18]]  # testing noise changes to an integrator


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0_8_initial_0_5_llvm(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            initial_value=np.array([[.5, .5, .5, .5]]),
            integrator_mode=True
        )
        val = T.execute([1, 1, 1, 1], bin_execute=True).tolist()
        assert val == [[0.9, 0.9, 0.9, 0.9]]
        T.noise = 10
        val = T.execute([1, 2, -3, 0], bin_execute=True).tolist()
        assert val == [[10.98, 11.78, 7.779999999999999, 10.18]]  # testing noise changes to an integrator


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0_8_list(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                time_constant=[0.8, 0.8, 0.8, 0.8],
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1]).tolist()
        assert (
            "time_constant parameter" in str(error_text.value)
            and "must be a float" in str(error_text.value)
        )


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_2(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                time_constant=2,
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1]).tolist()
        assert (
            "time_constant parameter" in str(error_text.value)
            and "must be a float between 0 and 1" in str(error_text.value)
        )


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_1(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                time_constant=1,
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1]).tolist()
        assert (
            "time_constant parameter" in str(error_text.value)
            and "must be a float between 0 and 1" in str(error_text.value)
        )


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_time_constant_0(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                time_constant=0,
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1]).tolist()
        assert (
            "time_constant parameter" in str(error_text.value)
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
        val = T.execute([10, 10, 10, 10]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

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
        val = T.execute([10.0 for i in range(VECTOR_SIZE)]).tolist()
        assert val == [[10.0 for i in range(VECTOR_SIZE)]]

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
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    #     assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

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
        val = T.execute([10, 10, 10, 10]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

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
        val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

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
    #     val = T.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
    #     assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

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
        val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

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
        val = T.execute([10.0, 10.0, 10.0, 10.0]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

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
    def test_transfer_mech_size_var_incompatible1(self):
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


    # ------------------------------------------------------------------------------------------------

    # InputState SPECIFICATIONS

    # ------------------------------------------------------------------------------------------------
    # TEST 1
    # Match of default_variable and specification of multiple InputStates by value and string

    def test_transfer_mech_input_states_match_with_default_variable(self):

        T = TransferMechanism(default_variable=[[0,0],[0]],
                                      input_states=[[32, 24], 'HELLO'])
        assert T.input_states[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 2
    # Mismatch between InputState variable specification and corresponding item of owner Mechanism's variable

    def test_transfer_mech_input_states_mismatch_with_default_variable_error(self):

        with pytest.raises(InputStateError) as error_text:
            T = TransferMechanism(default_variable=[[0],[0]],
                                  input_states=[[32, 24], 'HELLO'])
        assert "Value specified for" in str(error_text.value) and "with its expected format" in str(error_text.value)

    # ------------------------------------------------------------------------------------------------
    # TEST 3
    # Override of input_states (mis)specification by INPUT_STATES entry in params specification dict

    def test_transfer_mech_input_states_override_by_dict_spec(self):

        T = TransferMechanism(default_variable=[[0,0],[0]],
                              input_states=[[32], 'HELLO'],
                              params = {INPUT_STATES:[[32, 24], 'HELLO']}
                              )
        assert T.input_states[1].name == 'HELLO'
        # # PROBLEM WITH input FOR RUN:
        # my_mech_2.execute()

    # # ------------------------------------------------------------------------------------------------
    # # TEST 4
    # # Specification using input_states without default_variable
    #
    # def test_transfer_mech_input_states_no_default_variable(self):
    #
    #     # PROBLEM: SHOULD GENERATE TWO INPUT_STATES (
    #     #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
    #     #                INSTEAD, SEEM TO IGNORE InputState SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
    #     #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
    #     T = TransferMechanism(input_states=[[32, 24], 'HELLO'])
    #     assert len(T.input_states)==2
    #     assert T.input_states[1].name == 'HELLO'
    #     assert len(T.variable[0])==2
    #     assert len(T.variable[1])==1

    # # ------------------------------------------------------------------------------------------------
    # # TEST 5
    # # Specification using INPUT_STATES entry in params specification dict without default_variable
    #
    # def test_transfer_mech_input_states_specification_dict_no_default_variable(self):
    #
    #     # PROBLEM: SHOULD GENERATE TWO INPUT_STATES (
    #     #                ONE WITH [[32],[24]] AND OTHER WITH [[0]] AS VARIABLE INSTANCE DEFAULT
    #     #                INSTEAD, SEEM TO IGNORE InputState SPECIFICATIONS AND JUST USE DEFAULT_VARIABLE
    #     #                NOTE:  WORKS FOR ObjectiveMechanism, BUT NOT TransferMechanism
    #     T = TransferMechanism(params = {INPUT_STATES:[[32, 24], 'HELLO']})
    #     assert len(T.input_states)==2
    #     assert T.input_states[1].name == 'HELLO'
    #     assert len(T.variable[0])==2
    #     assert len(T.variable[1])==1

    # ------------------------------------------------------------------------------------------------
    # TEST 6
    # Mechanism specification

    def test_transfer_mech_input_states_mech_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[[0]],
                                  input_states=[R1])
        assert T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 7
    # Mechanism specification outside of a list

    def test_transfer_mech_input_states_standalone_mech_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        # Mechanism outside of list specification
        T = TransferMechanism(default_variable=[[0]],
                                      input_states=R1)
        assert T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 8
    # OutputState specification

    def test_transfer_mech_input_states_output_state_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[[0],[0]],
                                      input_states=[R1.output_states['FIRST'],
                                                    R1.output_states['SECOND']])
        assert T.input_states.names[0] == 'InputState'
        assert T.input_states.names[1] == 'InputState-1'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 9
    # OutputState specification outside of a list

    def test_transfer_mech_input_states_stand_alone_output_state_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[0],
                                      input_states=R1.output_states['FIRST'])
        assert T.input_states.names[0] == 'InputState'
        T.input_state.path_afferents[0].sender == R1.output_state
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 10
    # OutputStates in PROJECTIONS entries of a specification dictiontary, using with names (and one outside of a list)

    def test_transfer_mech_input_states_specification_dict_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(default_variable=[[0],[0]],
                                      input_states=[{NAME: 'FROM DECISION',
                                                     PROJECTIONS: [R1.output_states['FIRST']]},
                                                    {NAME: 'FROM RESPONSE_TIME',
                                                     PROJECTIONS: R1.output_states['SECOND']}])
        assert T.input_states.names[0] == 'FROM DECISION'
        assert T.input_states.names[1] == 'FROM RESPONSE_TIME'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 11
    # default_variable override of value of OutputState specification

    def test_transfer_mech_input_states_default_variable_override(self):

        R2 = TransferMechanism(size=3)

        # default_variable override of OutputState.value
        T = TransferMechanism(default_variable=[[0,0]],
                                      input_states=[R2])
        assert len(T.input_state.path_afferents[0].sender.variable)==3
        assert len(T.input_state.variable)==2
        assert len(T.variable)==1
        assert len(T.variable[0])==2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 12
    # 2-item tuple specification with default_variable override of OutputState.value

    def test_transfer_mech_input_states_2_item_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_states=[(R2, np.zeros((3,2)))])
        assert len(T.input_state.path_afferents[0].sender.variable)==3
        assert len(T.input_state.variable)==2
        assert len(T.variable)==1
        assert len(T.variable[0])==2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 13
    # ConnectionTuple Specification

    def test_transfer_mech_input_states_connection_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        T = TransferMechanism(size=2, input_states=[(R2, None, None, np.zeros((3,2)))])
        assert len(T.input_state.path_afferents[0].sender.variable)==3
        assert len(T.input_state.variable)==2
        assert len(T.variable)==1
        assert len(T.variable[0])==2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 14
    # Standalone Projection specification

    def test_transfer_mech_input_states_projection_spec(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(size=2,
                              input_states=[P])
        assert len(T.input_state.path_afferents[0].sender.variable)==3
        assert len(T.input_state.variable)==2
        assert len(T.variable)==1
        assert len(T.variable[0])==2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 15
    # Projection specification in Tuple

    def test_transfer_mech_input_states_projection_in_tuple_spec(self):
        R2 = TransferMechanism(size=3)
        P = MappingProjection(sender=R2)
        T = TransferMechanism(size=2,
                              input_states=[(R2, None, None, P)])
        assert len(T.input_state.path_afferents[0].sender.variable)==3
        assert len(T.input_state.variable)==2
        assert len(T.variable)==1
        assert len(T.variable[0])==2
        T.execute()

    # ------------------------------------------------------------------------------------------------
    # TEST 16
    # PROJECTIONS specification in InputState specification dictionary

    def test_transfer_mech_input_states_projection_in_specification_dict_spec(self):
        R1 = TransferMechanism(output_states=['FIRST', 'SECOND'])
        T = TransferMechanism(input_states=[{NAME: 'My InputState with Two Projections',
                                                     PROJECTIONS:[R1.output_states['FIRST'],
                                                                  R1.output_states['SECOND']]}])
        assert T.input_state.name == 'My InputState with Two Projections'
        for input_state in T.input_states:
            for projection in input_state.path_afferents:
                assert projection.sender.owner is R1
        T.execute()

    # # ------------------------------------------------------------------------------------------------
    # # TEST 17
    # METHOD OF SPECIFICATION NOT YET IMPLEMENTED:
    # # MECHANISMS/OUTPUT_STATES entries in params specification dict
    #
    # def test_transfer_mech_input_states_mech_output_state_in_specification_dict_spec(self):
    #
    #     # NOT YET IMPLEMENTED [10/29/17]:
    #     T = TransferMechanism(input_states=[{MECHANISM: R1,
    #                                                  OUTPUT_STATES: ['FIRST', 'SECOND']}])
    #     assert len(T.input_states)==2
    #     assert all(name in T.input_states.names for name in {'FIRST', 'SECOND'})
    #     for input_state in T.input_states:
    #         for projection in input_state.path_afferents:
    #             assert projection.sender.owner is R1
