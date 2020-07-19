import numpy as np
import pytest

import psyneulink.core.llvm as pnlvm
from psyneulink.core.components.component import ComponentError
from psyneulink.core.components.functions.learningfunctions import Reinforcement
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AccumulatorIntegrator, AdaptiveIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear, Exponential, Logistic, ReLU, SoftMax
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.functions.userdefinedfunction import UserDefinedFunction
from psyneulink.core.components.functions.distributionfunctions import NormalDist, UniformToNormalDist, \
    ExponentialDist, \
    UniformDist, GammaDist, WaldDist
from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferError, TransferMechanism
from psyneulink.library.components.mechanisms.processing.transfer.lcamechanism import LCAMechanism
from psyneulink.core.components.ports.inputport import InputPort
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.keywords import \
    INSTANTANEOUS_MODE_VALUE, INTEGRATOR_MODE_VALUE, RESET, COMBINE, RESULT, GREATER_THAN
from psyneulink.core.globals.utilities import UtilitiesError, set_global_seed
from psyneulink.core.globals.parameters import ParameterError
from psyneulink.core.scheduling.condition import Never
from psyneulink.core.scheduling.time import TimeScale

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
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        val = benchmark(T.execute, [10 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[10.0 for i in range(VECTOR_SIZE)]])
        assert len(T.size) == 1 and T.size[0] == VECTOR_SIZE and isinstance(T.size[0], np.integer)
        # this test assumes size is returned as a 1D array: if it's not, then several tests in this file must be changed

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_inputs_list_of_floats(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        var = [10.0 for i in range(VECTOR_SIZE)]
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        val = EX(var)
        benchmark(EX, var)
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
    #     np.testing.assert_array_equal(T.defaults.variable, np.array([[[0, 0, 0, 0]], [[1, 1, 1, 1]]]))

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_variable_none_size_none(self):

        T = TransferMechanism(
            name='T'
        )
        assert len(T.defaults.variable) == 1 and T.defaults.variable[0] == 0

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_inputs_list_of_strings(self):
        with pytest.raises(FunctionError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                integrator_mode=True
            )
            T.execute(["one", "two", "three", "four"])
        assert "Unrecognized type" in str(error_text.value)

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
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_var_float_noise(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=5.0,
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [0 for i in range(VECTOR_SIZE)]
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[5.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_len_1_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=NormalDist(),
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[-1.56404341, -0.51529709, -2.55352727, 0.30284592]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_array_var_normal_array_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=[NormalDist(), NormalDist(), NormalDist(), NormalDist()],
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        val = T.execute([0, 0, 0, 0])
        expected = [0.6202001216069017, 0.840166034615641, 0.7279826246296707, -1.5678942459349325]
        assert np.allclose(np.asfarray(val[0]), expected)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear noise2")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_var_normal_array_noise2(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            noise=[5.0 for i in range(VECTOR_SIZE)],
            integration_rate=1.0,
            integrator_mode=True
        )
        T.reset_stateful_function_when = Never()
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [0 for i in range(VECTOR_SIZE)]
        val = EX(var)
        benchmark(EX, var)
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
                integration_rate=0.1,
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
                integration_rate=0.1,
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
            noise=NormalDist(),
            integration_rate=1.0,
            integrator_mode=True
        )

        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[-1.56404341, -0.51529709, -2.55352727, 0.30284592]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_noise_standard_deviation_error(self):
        with pytest.raises(FunctionError) as error_text:
            standard_deviation = -2.0
            T = TransferMechanism(
                name="T",
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                noise=NormalDist(standard_deviation=standard_deviation),
                integration_rate=1.0,
                integrator_mode=True
            )

        assert "The standard_deviation parameter" in str(error_text.value)
        assert "must be greater than zero" in str(error_text.value)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_exponential_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=ExponentialDist(),
            integration_rate=1.0,
            integrator_mode=True,
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.54571315, 0.29964231, 0.71595475, 0.51908319]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_uniform_to_normal_noise(self):
        try:
            import scipy
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                noise=UniformToNormalDist(),
                integration_rate=1.0
            )
            T.noise.parameters.random_state.get(None).seed(22)
            val = T.execute([0, 0, 0, 0])
            assert np.allclose(val, [[-0.81177443, -0.04593492, -0.20051725, 1.07665147]])

        except:
            with pytest.raises(FunctionError) as error_text:
                T = TransferMechanism(
                    name='T',
                    default_variable=[0, 0, 0, 0],
                    function=Linear(),
                    noise=UniformToNormalDist(),
                    integration_rate=1.0
                )
            assert "The UniformToNormalDist function requires the SciPy package." in str(error_text.value)


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Uniform_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=UniformDist(),
            integration_rate=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.42057158, 0.25891675, 0.51127472, 0.40493414]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Gamma_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=GammaDist(),
            integration_rate=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.54571315, 0.29964231, 0.71595475, 0.51908319]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_Wald_noise(self):

        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            noise=WaldDist(),
            integration_rate=1.0,
            integrator_mode=True
        )
        val = T.execute([0, 0, 0, 0])
        assert np.allclose(val, [[0.11902672, 0.73955962, 8.38161237, 0.49600183]])


class TestTransferMechanismFunctions:

    def tests_valid_udf_1d_variable(self):
        def double_all_elements(variable):
            return np.array(variable) * 2

        T = TransferMechanism(name='T-udf',
                              default_variable=[[0.0, 0.0]],
                              function=UserDefinedFunction(custom_function=double_all_elements))
        result = T.execute([[1.0, 2.0]])
        assert np.allclose(result, [[2.0, 4.0]])

    def tests_valid_udf_2d_variable(self):
        def double_all_elements(variable):
            return np.array(variable) * 2

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
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_logistic_fun(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Logistic(),
            integration_rate=1.0,
            integrator_mode=True
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [0 for i in range(VECTOR_SIZE)]
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[0.5 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism ReLU")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_relu_fun(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=ReLU(),
            integration_rate=1.0,
            integrator_mode=True
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        val1 = EX([0 for i in range(VECTOR_SIZE)])
        val2 = EX([1 for i in range(VECTOR_SIZE)])
        val3 = EX([-1 for i in range(VECTOR_SIZE)])
        benchmark(EX, [0 for i in range(VECTOR_SIZE)])

        assert np.allclose(val1, [[0.0 for i in range(VECTOR_SIZE)]])
        assert np.allclose(val2, [[1.0 for i in range(VECTOR_SIZE)]])
        assert np.allclose(val3, [[0.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Exponential")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_exponential_fun(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Exponential(),
            integration_rate=1.0,
            integrator_mode=True
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [0 for i in range(VECTOR_SIZE)]
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[1.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism SoftMax")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_softmax_fun(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=SoftMax(),
            integration_rate=1.0,
            integrator_mode=True
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [0 for i in range(VECTOR_SIZE)]
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[1.0 / VECTOR_SIZE for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_normal_fun(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=NormalDist(),
                integration_rate=1.0,
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
                integration_rate=1.0,
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
                function=AccumulatorIntegrator(),
                integration_rate=1.0,
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
                integration_rate=1.0,
                integrator_mode=True
            )
            T.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


class TestTransferMechanismIntegratorFunctionParams:

    # integration_rate array on mech: assigned to mech value
    # integration_rate array on fct: assigned to fct value
    # integration_rate array on both: assinged to fct value
    # integration_rate array wrong array size error


    # RATE TESTS ---------------------------------------------------------------------------

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_mech_rate(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator,
            integration_rate=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0., 0.19, 0.36, 0.51]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_fct_rate(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator(rate=[i / 10 for i in range(VECTOR_SIZE)])
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0., 0.19, 0.36, 0.51]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_fct_over_mech_rate(self, benchmark, mode):

        T = TransferMechanism(
                name='T',
                default_variable=[0 for i in range(VECTOR_SIZE)],
                integrator_mode=True,
                integrator_function=AdaptiveIntegrator(rate=[i / 20 for i in range(VECTOR_SIZE)]),
                integration_rate=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0., 0.0975, 0.19, 0.2775]])

    def test_transfer_mech_array_assignments_wrong_size_mech_rate(self):

        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    integration_rate=[i / 10 for i in range(VECTOR_SIZE + 1)]
            )
        assert (
            "integration_rate' arg for" in str(error_text.value)
            and "must be either an int or float, or have the same shape as its variable" in str(error_text.value)
        )

    def test_transfer_mech_array_assignments_wrong_size_fct_rate(self):

        with pytest.raises(FunctionError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    integrator_function=AdaptiveIntegrator(rate=[i / 10 for i in range(VECTOR_SIZE + 1)])
            )
        assert (
            "The following parameters with len>1 specified" in str(error_text.value)
            and "don't have the same length as its 'default_variable' (4): ['rate']." in str(error_text.value)
        )

    # INITIAL_VALUE / INITALIZER TESTS -------------------------------------------------------

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_mech_init_val(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            initial_value=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75,  0.775,  0.8, 0.825]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_fct_initzr(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator(
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    initializer=[i / 10 for i in range(VECTOR_SIZE)]
            ),
        )
        if mode == 'Python':
            EX = T.execute
            assert True
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75,  0.775,  0.8, 0.825]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_fct_initlzr_over_mech_init_val(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator(
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    initializer=[i / 10 for i in range(VECTOR_SIZE)]
            ),
            initial_value=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75,  0.775,  0.8, 0.825]])

    # def test_transfer_mech_array_assignments_wrong_size_mech_init_val(self, benchmark, mode):
    def test_transfer_mech_array_assignments_wrong_size_mech_init_val(self):

        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    initial_value=[i / 10 for i in range(VECTOR_SIZE + 1)]
            )
        assert (
            "The format of the initial_value parameter" in str(error_text.value)
            and "must match its variable" in str(error_text.value)
        )

    def test_transfer_mech_array_assignment_matches_size_integrator_fct_param_def(self):

        T = TransferMechanism(
                name='T',
                default_variable=[0 for i in range(VECTOR_SIZE)],
                integrator_mode=True,
                integrator_function=AdaptiveIntegrator(rate=[.1 + i / 10 for i in range(VECTOR_SIZE)])
        )
        result1 = T.execute([range(VECTOR_SIZE)])
        result2 = T.execute([range(VECTOR_SIZE)])
        assert np.allclose(result1, [[0., 0.2, 0.6, 1.2]])
        assert np.allclose(result2, [[0., 0.36, 1.02, 1.92]])

    def test_transfer_mech_array_assignment_wrong_size_integrator_fct_default_variable(self):

        with pytest.raises(ParameterError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    integrator_function=AdaptiveIntegrator(default_variable=[0 for i in range(VECTOR_SIZE + 1)])
            )
        assert 'Variable shape incompatibility between (TransferMechanism T) and its integrator_function Parameter' in str(error_text.value)

    def test_transfer_mech_array_assignment_wrong_size_integrator_fct_param(self):

        with pytest.raises(FunctionError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    integrator_function=AdaptiveIntegrator(rate=[0 for i in range(VECTOR_SIZE + 1)])
            )
        assert (
            "The following parameters with len>1 specified" in str(error_text.value)
            and "don't have the same length as its 'default_variable' (4): ['rate']." in str(error_text.value)
        )

    # FIX: CAN'T RUN THIS YET:  CRASHES W/O ERROR MESSAGE SINCE DEFAULT_VARIABLE OF FCT DOESN'T MATCH ITS INITIALIZER
    # # def test_transfer_mech_array_assignments_wrong_size_fct_initlzr(self, benchmark, mode):
    # def test_transfer_mech_array_assignments_wrong_size_fct_initlzr(self):
    #
    #     with pytest.raises(TransferError) as error_text:
    #         T = TransferMechanism(
    #                 name='T',
    #                 default_variable=[0 for i in range(VECTOR_SIZE)],
    #                 integrator_mode=True,
    #                 integrator_function=AdaptiveIntegrator(initializer=[i / 10 for i in range(VECTOR_SIZE + 1)])
    #         )
    #     assert (
    #         "initializer' arg for" in str(error_text.value)
    #         and "must be either an int or float, or have the same shape as its variable" in str(error_text.value)
    #     )


    # NOISE TESTS ---------------------------------------------------------------------------

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_array_assignments_mech_noise(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator,
            noise=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75, 0.9, 1.05, 1.2 ]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=[pytest.mark.llvm, pytest.mark.skip]),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda, pytest.mark.skip])])
    # FIXME: Incorrect T.integrator_function.defaults.variable reported
    def test_transfer_mech_array_assignments_fct_noise(self, benchmark, mode):

        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            integrator_mode=True,
            integrator_function=AdaptiveIntegrator(noise=[i / 10 for i in range(VECTOR_SIZE)])
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75, 0.9, 1.05, 1.2 ]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Parameter Array Assignments")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=[pytest.mark.llvm, pytest.mark.skip]),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda, pytest.mark.skip])])
    # FIXME: Incorrect T.integrator_function.defaults.variable reported
    def test_transfer_mech_array_assignments_fct_over_mech_noise(self, benchmark, mode):

        T = TransferMechanism(
                name='T',
                default_variable=[0 for i in range(VECTOR_SIZE)],
                integrator_mode=True,
                integrator_function=AdaptiveIntegrator(noise=[i / 20 for i in range(VECTOR_SIZE)]),
                noise=[i / 10 for i in range(VECTOR_SIZE)]
        )
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            EX = e.cuda_execute

        var = [1 for i in range(VECTOR_SIZE)]
        EX(var)
        val = EX(var)
        benchmark(EX, var)
        assert np.allclose(val, [[ 0.75, 0.825, 0.9, 0.975]])

    # def test_transfer_mech_array_assignments_wrong_size_mech_noise(self, benchmark, mode):
    def test_transfer_mech_array_assignments_wrong_size_mech_noise(self):

        with pytest.raises(MechanismError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_mode=True,
                    noise=[i / 10 for i in range(VECTOR_SIZE + 1)]
            )
        assert (
            "Noise parameter ([0.0, 0.1, 0.2, 0.3, 0.4])" in str(error_text.value) and
            "does not match default variable ([[0 0 0 0]])." in str(error_text.value) and
            "must be specified as a float, a function, or an array of the appropriate shape ((1, 4))." in str(error_text.value)
        )

    # def test_transfer_mech_array_assignments_wrong_size_fct_noise(self, benchmark, mode):
    def test_transfer_mech_array_assignments_wrong_size_fct_noise(self):

        with pytest.raises(FunctionError) as error_text:
            T = TransferMechanism(
                    name='T',
                    default_variable=[0 for i in range(VECTOR_SIZE)],
                    integrator_function=AdaptiveIntegrator(noise=[i / 10 for i in range(VECTOR_SIZE + 1)]),
                    integrator_mode=True
            )
        assert (
            "Noise parameter" in str(error_text.value) and
            "does not match default variable" in str(error_text.value) and
            "must be specified as a float, a function, or an array of the appropriate shape ((1, 4))" in str(error_text.value)
        )


class TestTransferMechanismTimeConstant:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=1")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_integration_rate_0_8(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            integration_rate=0.8,
            integrator_mode=True
        )
        if mode == 'Python':
            val1 = T.execute([1 for i in range(VECTOR_SIZE)])
            val2 = T.execute([1 for i in range(VECTOR_SIZE)])
            benchmark(T.execute, [0 for i in range(VECTOR_SIZE)])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val1 = e.execute([1 for i in range(VECTOR_SIZE)])
            val2 = e.execute([1 for i in range(VECTOR_SIZE)])
            benchmark(e.execute, [0 for i in range(VECTOR_SIZE)])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            val1 = e.cuda_execute([1 for i in range(VECTOR_SIZE)])
            val2 = e.cuda_execute([1 for i in range(VECTOR_SIZE)])
            benchmark(e.cuda_execute, [0 for i in range(VECTOR_SIZE)])

        assert np.allclose(val1, [[0.8 for i in range(VECTOR_SIZE)]])
        assert np.allclose(val2, [[0.96 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=1")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_smoothin_factor_1_0(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            integration_rate=1.0,
            integrator_mode=True
        )
        if mode == 'Python':
            val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.execute, [1 for i in range(VECTOR_SIZE)])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.cuda_execute, [1 for i in range(VECTOR_SIZE)])

        assert np.allclose(val, [[1.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism Linear TimeConstant=0")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_integration_rate_0_0(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0 for i in range(VECTOR_SIZE)],
            function=Linear(),
            integration_rate=0.0,
            integrator_mode=True
        )
        if mode == 'Python':
            val = benchmark(T.execute, [1 for i in range(VECTOR_SIZE)])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.execute, [1 for i in range(VECTOR_SIZE)])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.cuda_execute, [1 for i in range(VECTOR_SIZE)])
        assert np.allclose(val, [[0.0 for i in range(VECTOR_SIZE)]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=[pytest.mark.llvm, pytest.mark.skip])])
    def test_transfer_mech_integration_rate_0_8_initial_0_5(self, mode):
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=0.8,
            initial_value=np.array([[.5, .5, .5, .5]]),
            integrator_mode=True
        )
        if mode == 'Python':
            val = T.execute([1, 1, 1, 1])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = e.execute([1, 1, 1, 1])
        assert np.allclose(val, [[0.9, 0.9, 0.9, 0.9]])

        T.noise = 10

        if mode == 'Python':
            val = T.execute([1, 2, -3, 0])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = e.execute([1, 2, -3, 0])
        assert np.allclose(val, [[10.98, 11.78, 7.779999999999999, 10.18]]) # testing noise changes to an integrator

    # @pytest.mark.mechanism
    # @pytest.mark.transfer_mechanism
    # def test_transfer_mech_integration_rate_0_8_list(self):
    #     with pytest.raises(TransferError) as error_text:
    #         T = TransferMechanism(
    #             name='T',
    #             default_variable=[0, 0, 0, 0],
    #             function=Linear(),
    #             integration_rate=[0.8, 0.8, 0.8, 0.8],
    #             integrator_mode=True
    #         )
    #         T.execute([1, 1, 1, 1])
    #     assert (
    #         "integration_rate parameter" in str(error_text.value)
    #         and "must be a float" in str(error_text.value)
    #     )
    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_integration_rate_0_8_list(self):
        T = TransferMechanism(
            name='T',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=[0.8, 0.8, 0.8, 0.8],
            integrator_mode=True
        )
        T.execute([1, 1, 1, 1])
        val = T.execute([1, 1, 1, 1])
        assert np.allclose(val, [[ 0.96,  0.96,  0.96,  0.96]])


    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_integration_rate_2(self):
        with pytest.raises(TransferError) as error_text:
            T = TransferMechanism(
                name='T',
                default_variable=[0, 0, 0, 0],
                function=Linear(),
                integration_rate=2,
                integrator_mode=True
            )
            T.execute([1, 1, 1, 1])
        assert (
            "Value(s) in \'integration_rate\' arg for" in str(error_text.value)
            and "must be an int or float in the interval [0,1]" in str(error_text.value)
        )


class TestTransferMechanismSize:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_size_int_check_var(self):
        T = TransferMechanism(
            name='T',
            size=4
        )
        assert len(T.defaults.variable) == 1 and (T.defaults.variable[0] == [0., 0., 0., 0.]).all()
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
        assert len(T.defaults.variable) == 1 and (T.defaults.variable[0] == [0., 0., 0., 0.]).all()
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
        assert len(T.defaults.variable) == 3 and len(T.defaults.variable[0]) == 2 and len(T.defaults.variable[1]) == 3 and len(T.defaults.variable[2]) == 4

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
        assert len(T.defaults.variable) == 3 and len(T.defaults.variable[0]) == 2 and len(T.defaults.variable[1]) == 3 and len(T.defaults.variable[2]) == 4

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
        assert len(T.defaults.variable) == 2 and (T.defaults.variable[0] == [1, 2]).all() and (T.defaults.variable[1] == [3, 4, 5]).all()

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
        assert len(T.defaults.variable) == 2 and (T.defaults.variable[0] == [1, 2]).all() and (T.defaults.variable[1] == [3, 4]).all()
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
        assert len(T.defaults.variable) == 2 and (T.defaults.variable[0] == [1, 2]).all() and (T.defaults.variable[1] == [3, 4]).all()

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
        assert len(T.defaults.variable) == 1 and (T.defaults.variable[0] == [1, 2, 3, 4]).all()
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
        assert len(T.defaults.variable) == 1 and (T.defaults.variable[0] == [1, 2, 3, 4]).all()
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
        assert (T.defaults.variable[0] == [1, 2]).all() and (T.defaults.variable[1] == [3, 4, 5]).all() and len(T.defaults.variable) == 2

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
        assert (T.defaults.variable[0] == [1, 2]).all() and (T.defaults.variable[1] == [3, 4, 5]).all() and len(T.defaults.variable) == 2

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
        assert len(T.defaults.variable) == 1 and len(T.defaults.variable[0]) == 2
        assert len(T.size) == 1 and T.size[0] == 2


class TestTransferMechanismMultipleInputPorts:

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.mimo
    def test_transfer_mech_2d_variable_mean(self):
        from psyneulink.core.globals.keywords import MEAN
        T = TransferMechanism(
            name='T',
            function=Linear(slope=2.0, intercept=1.0),
            default_variable=[[0.0, 0.0], [0.0, 0.0]],
            output_ports=[MEAN]
        )
        val = T.execute([[1.0, 2.0], [3.0, 4.0]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.mimo
    @pytest.mark.benchmark(group="MIMO")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_transfer_mech_2d_variable(self, benchmark, mode):
        T = TransferMechanism(
            name='T',
            function=Linear(slope=2.0, intercept=1.0),
            default_variable=[[0.0, 0.0], [0.0, 0.0]],
        )
        if mode == 'Python':
            val = benchmark(T.execute, [[1.0, 2.0], [3.0, 4.0]])
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.execute, [[1.0, 2.0], [3.0, 4.0]])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.cuda_execute, [[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(val, [[3., 5.], [7., 9.]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    def test_transfer_mech_2d_variable_noise(self):
        T = TransferMechanism(
            name='T',
            function=Linear(slope=2.0, intercept=1.0),
            noise=NormalDist(),
            default_variable=[[0.0, 0.0], [0.0, 0.0]]
        )
        val = T.execute([[1.0, 2.0], [3.0, 4.0]])
        assert np.allclose(val, [[1.6136458, 7.00036006], [12.09938081, 7.56874402]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.mimo
    @pytest.mark.benchmark(group="MIMO")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_multiple_output_ports_for_multiple_input_ports(self, benchmark, mode):
        T = TransferMechanism(input_ports=['a','b','c'])
        if mode == 'Python':
            val = benchmark(T.execute, [[1], [2], [3]])
            assert all(a==b for a,b in zip(T.output_values,val))
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.execute, [[1], [2], [3]])
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(T)
            val = benchmark(e.cuda_execute, [[1], [2], [3]])

        assert len(T.variable)==3
        assert all(a==b for a,b in zip(val, [[ 1.],[ 2.],[ 3.]]))
        assert len(T.output_ports)==3

    # @pytest.mark.mechanism
    # @pytest.mark.transfer_mechanism
    # @pytest.mark.mimo
    # def test_OWNER_VALUE_standard_output_port(self):
    #     from psyneulink.core.globals.keywords import OWNER_VALUE
    #     T = TransferMechanism(input_ports=[[[0],[0]],'b','c'],
    #                               output_ports=OWNER_VALUE)
    #     print(T.value)
    #     val = T.execute([[[1],[4]],[2],[3]])
    #     expected_val = [[[1],[4]],[2],[3]]
    #     assert len(T.output_ports)==1
    #     assert len(T.output_ports[OWNER_VALUE].value)==3
    #     assert all(all(a==b for a,b in zip(x,y)) for x,y in zip(val, expected_val))


class TestIntegratorMode:
    def test_integrator_mode_simple_on_and_off(self):
        T = TransferMechanism(size=2)
        assert np.allclose(T.execute([0.5, 1]), [[0.5, 1]])
        T.integrator_mode=True
        assert np.allclose(T.execute([0.5, 1]), [[0.25, 0.5 ]])
        assert np.allclose(T.execute([0.5, 1]), [[0.375, 0.75 ]])
        T.integrator_mode=False
        assert np.allclose(T.execute([0.5, 1]), [[0.5, 1]])

    def test_previous_value_persistence_execute(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        T.reset_stateful_function_when = Never()
        assert np.allclose(T.integrator_function.previous_value, 0.5)

        T.execute(1.0)
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        assert np.allclose(T.integrator_function.previous_value, 0.55)

        T.execute(1.0)
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.previous_value, 0.595)

    def test_previous_value_persistence_run(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        C = Composition(pathways=[T])
        T.reset_stateful_function_when = Never()

        assert np.allclose(T.integrator_function.previous_value, 0.5)

        C.run(inputs={T: 1.0}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.595)

        C.run(inputs={T: 2.0}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.595 + 0.1*2.0 + 0.0 = 0.7355  --->  previous value = 0.7355
        # linear fn: 0.7355*1.0 = 0.7355
        # Trial 4
        # integration: 0.9*0.7355 + 0.1*2.0 + 0.0 = 0.86195  --->  previous value = 0.86195
        # linear fn: 0.86195*1.0 = 0.86195

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.86195)

    def test_previous_value_reset_execute(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        T.reset_stateful_function_when = Never()
        assert np.allclose(T.integrator_function.previous_value, 0.5)
        T.execute(1.0)
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        assert np.allclose(T.integrator_function.previous_value, 0.55)
        assert np.allclose(T.value, 0.55)

        # Reset integrator_function ONLY
        T.integrator_function.reset(0.6)

        assert np.allclose(T.integrator_function.previous_value, 0.6)   # previous_value is a property that looks at integrator_function
        assert np.allclose(T.value, 0.55)           # on mechanism only, so does not update until execution

        T.execute(1.0)
        # integration: 0.9*0.6 + 0.1*1.0 + 0.0 = 0.64  --->  previous value = 0.55
        # linear fn: 0.64*1.0 = 0.64
        assert np.allclose(T.integrator_function.previous_value, 0.64)   # property that looks at integrator_function
        assert np.allclose(T.value, 0.64)            # on mechanism, but updates with execution

        T.reset(0.4)
        # linear fn: 0.4*1.0 = 0.4
        assert np.allclose(T.integrator_function.previous_value, 0.4)   # property that looks at integrator, which updated with mech reset
        assert np.allclose(T.value, 0.4)  # on mechanism, but updates with mech reset

        T.execute(1.0)
        # integration: 0.9*0.4 + 0.1*1.0 + 0.0 = 0.46  --->  previous value = 0.46
        # linear fn: 0.46*1.0 = 0.46
        assert np.allclose(T.integrator_function.previous_value, 0.46)  # property that looks at integrator, which updated with mech exec
        assert np.allclose(T.value, 0.46)  # on mechanism, but updates with exec

    def test_reset_run(self):
        T = TransferMechanism(name="T",
                              initial_value=0.5,
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        C = Composition(pathways=[T])

        T.reset_stateful_function_when = Never()

        assert np.allclose(T.integrator_function.previous_value, 0.5)

        C.run(inputs={T: 1.0}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.595)

        T.integrator_function.reset(0.9, context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.9)
        assert np.allclose(T.parameters.value.get(C), 0.595)

        T.reset(0.5, context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.5)
        assert np.allclose(T.parameters.value.get(C), 0.5)

        C.run(inputs={T: 1.0}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), 0.595)

    def test_reset_run_array(self):
        T = TransferMechanism(name="T",
                              default_variable=[0.0, 0.0, 0.0],
                              initial_value=[0.5, 0.5, 0.5],
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        C = Composition(pathways=[T])
        T.reset_stateful_function_when = Never()

        assert np.allclose(T.integrator_function.previous_value, [0.5, 0.5, 0.5])

        C.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.595, 0.595, 0.595])

        T.integrator_function.reset([0.9, 0.9, 0.9], context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.9, 0.9, 0.9])
        assert np.allclose(T.parameters.value.get(C), [0.595, 0.595, 0.595])

        T.reset([0.5, 0.5, 0.5], context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.5, 0.5, 0.5])
        assert np.allclose(T.parameters.value.get(C), [0.5, 0.5, 0.5])

        C.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.595, 0.595, 0.595])

    def test_reset_run_2darray(self):

        initial_val = [[0.5, 0.5, 0.5]]
        T = TransferMechanism(name="T",
                              default_variable=[[0.0, 0.0, 0.0]],
                              initial_value=initial_val,
                              integrator_mode=True,
                              integration_rate=0.1,
                              noise=0.0)
        C = Composition(pathways=[T])
        T.reset_stateful_function_when = Never()

        assert np.allclose(T.integrator_function.previous_value, initial_val)

        C.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 1
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.595, 0.595, 0.595])

        T.integrator_function.reset([0.9, 0.9, 0.9], context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.9, 0.9, 0.9])
        assert np.allclose(T.parameters.value.get(C), [0.595, 0.595, 0.595])

        T.reset(initial_val, context=C)

        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), initial_val)
        assert np.allclose(T.parameters.value.get(C), initial_val)

        C.run(inputs={T: [1.0, 1.0, 1.0]}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 4
        # integration: 0.9*0.55 + 0.1*1.0 + 0.0 = 0.595  --->  previous value = 0.595
        # linear fn: 0.595*1.0 = 0.595
        assert np.allclose(T.integrator_function.parameters.previous_value.get(C), [0.595, 0.595, 0.595])

    def test_reset_not_integrator(self):

        with pytest.raises(MechanismError) as err_txt:
            T_not_integrator = TransferMechanism()
            T_not_integrator.execute(1.0)
            T_not_integrator.reset(0.0)
        assert "not allowed because this Mechanism is not stateful;" in str(err_txt.value)
        assert "try setting the integrator_mode argument to True." in str(err_txt.value)

    def test_switch_mode(self):
        T = TransferMechanism(integrator_mode=True,
                              on_resume_integrator_mode=INTEGRATOR_MODE_VALUE)
        C = Composition(pathways=[T])
        integrator_function = T.integrator_function
        T.reset_stateful_function_when = Never()
        # T starts with integrator_mode = True; confirm that T behaves correctly
        C.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.parameters.value.get(C), [[0.875]])

        assert T.parameters.integrator_mode.get(C) is True
        assert T.integrator_function is integrator_function

        # Switch integrator_mode to False; confirm that T behaves correctly
        T.parameters.integrator_mode.set(False, context=C)

        assert T.parameters.integrator_mode.get(C) is False

        C.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.parameters.value.get(C), [[1.0]])

        # Switch integrator_mode BACK to True; confirm that T picks up where it left off
        T.parameters.integrator_mode.set(True, context=C)

        assert T.parameters.integrator_mode.get(C) is True
        assert T.has_integrated is True
        assert T.integrator_function is integrator_function

        C.run({T: [[1.0], [1.0], [1.0]]})
        assert np.allclose(T.parameters.value.get(C), [[0.984375]])

    def test_initial_values_softmax(self):
        T = TransferMechanism(default_variable=[[0.0, 0.0], [0.0, 0.0]],
                              function=SoftMax(),
                              integrator_mode=True,
                              integration_rate=0.5,
                              initial_value=[[1.0, 2.0], [3.0, 4.0]])
        T2 = TransferMechanism()
        C = Composition(pathways=[T, T2])

        C.run(inputs={T: [[1.5, 2.5], [3.5, 4.5]]})

        result = T.parameters.value.get(C)
        # Expected results
        # integrator function:
        # input = [[1.5, 2.5], [3.5, 4.5]]  |  output = [[1.25, 2.25]], [3.25, 4.25]]
        integrator_fn = AdaptiveIntegrator(rate=0.5,
                                           default_variable=[[0.0, 0.0], [0.0, 0.0]],
                                           initializer=[[1.0, 2.0], [3.0, 4.0]])
        expected_result_integrator = integrator_fn([[1.5, 2.5], [3.5, 4.5]])

        S1 = SoftMax()
        expected_result_s1 = S1([[1.25, 2.25]])

        S2 = SoftMax()
        expected_result_s2 = S2([[3.25, 4.25]])

        assert np.allclose(expected_result_integrator, T.parameters.integrator_function_value.get(C))
        assert np.allclose(expected_result_s1, result[0])
        assert np.allclose(expected_result_s2, result[1])

    def test_set_integrator_mode_after_init(self):
        T = TransferMechanism()
        T.integrator_mode = True
        T.execute(1)


class TestOnResumeIntegratorMode:

    def test_integrator_mode_value_spec(self):
        T = TransferMechanism(on_resume_integrator_mode=INTEGRATOR_MODE_VALUE,
                              integration_rate=0.5,
                              integrator_mode=True)
        C = Composition()
        C.add_node(T)

        C.run(inputs={T: [[1.0], [2.0]]})                   # Run in "integrator mode"
        # Trial 0: 0.5*0.0 + 0.5*1.0 = 0.5 * 1.0 = 0.5
        # Trial 1: 0.5*0.5 + 0.5*2.0 = 1.25 * 1.0 = 1.25
        assert np.allclose(T.parameters.value.get(C), [[1.25]])

        T.parameters.integrator_mode.set(False, context=C)    # Switch to "instantaneous mode"

        C.run(inputs={T: [[1.0], [2.0]]})                               # Run in "instantaneous mode"
        # Trial 0: 1.0 * 1.0 = 1.0
        # Trial 1: 1.0 * 2.0 = 2.0
        assert np.allclose(T.parameters.value.get(C), [[2.0]])

        T.parameters.integrator_mode.set(True, context=C)     # Switch back to "integrator mode"

        C.run(inputs={T: [[1.0], [2.0]]})                               # Run in "integrator mode" and pick up at 1.25
        # Trial 0: 0.5*1.25 + 0.5*1.0 = 1.125 * 1.0 = 1.125
        # Trial 1: 0.5*1.125 + 0.5*2.0 = 1.5625 * 1.0 = 1.5625
        assert np.allclose(T.parameters.value.get(C), [[1.5625]])

    def test_instantaneous_mode_value_spec(self):
        T = TransferMechanism(on_resume_integrator_mode=INSTANTANEOUS_MODE_VALUE,
                              integration_rate=0.5,
                              integrator_mode=True)
        C = Composition()
        C.add_node(T)

        C.run(inputs={T: [[1.0], [2.0]]})                   # Run in "integrator mode"
        # Trial 0: 0.5*0.0 + 0.5*1.0 = 0.5 * 1.0 = 0.5
        # Trial 1: 0.5*0.5 + 0.5*2.0 = 1.25 * 1.0 = 1.25
        assert np.allclose(T.parameters.value.get(C), [[1.25]])

        T.parameters.integrator_mode.set(False, context=C)     # Switch to "instantaneous mode"

        C.run(inputs={T: [[1.0], [2.0]]})                                # Run in "instantaneous mode"
        # Trial 0: 1.0 * 1.0 = 1.0
        # Trial 1: 1.0 * 2.0 = 2.0
        assert np.allclose(T.parameters.value.get(C), [[2.0]])

        T.parameters.integrator_mode.set(True, context=C)      # Switch back to "integrator mode"

        C.run(inputs={T: [[1.0], [2.0]]})                                # Run in "integrator mode" and pick up at 2.0
        # Trial 0: 0.5*2.0 + 0.5*1.0 = 1.5 * 1.0 = 1.5
        # Trial 1: 0.5*1.5 + 0.5*2.0 = 1.75 * 1.0 = 1.75
        assert np.allclose(T.parameters.value.get(C), [[1.75]])

    def test_reset_spec(self):
        T = TransferMechanism(on_resume_integrator_mode=RESET,
                              integrator_mode=True)
        C = Composition()
        C.add_node(T)

        C = Composition()
        C.add_node(T)

        C.run(inputs={T: [[1.0], [2.0]]})                        # Run in "integrator mode"
        # Trial 0: 0.5*0.0 + 0.5*1.0 = 0.5 * 1.0 = 0.5
        # Trial 1: 0.5*0.5 + 0.5*2.0 = 1.25 * 1.0 = 1.25
        assert np.allclose(T.parameters.value.get(C), [[1.25]])

        T.parameters.integrator_mode.set(False, context=C)                               # Switch to "instantaneous mode"

        C.run(inputs={T: [[1.0], [2.0]]})                       # Run in "instantaneous mode"
        # Trial 0: 1.0 * 1.0 = 1.0
        # Trial 1: 1.0 * 2.0 = 2.0
        assert np.allclose(T.parameters.value.get(C), [[2.0]])

        T.parameters.integrator_mode.set(True, context=C)                                # Switch back to "integrator mode"

        C.run(inputs={T: [[1.0], [2.0]]})                       # Run in "integrator mode", pick up at 0.0
        # Trial 0: 0.5*0.0 + 0.5*1.0 = 0.5 * 1.0 = 0.5
        # Trial 1: 0.5*0.5 + 0.5*2.0 = 1.25 * 1.0 = 1.25
        assert np.allclose(T.parameters.value.get(C), [[1.25]])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.benchmark(group="TransferMechanism")
    @pytest.mark.parametrize('bin_execute', ['Python',
                                             # 'LLVM' mode is not supported
                                             # the comparison values and checks
                                             # are not synced between binary
                                             # and Python structures
                                             pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                             pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                             pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                             pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                            ])
    def test_termination_measures(self, bin_execute):
        stim_input = ProcessingMechanism(size=2, name='Stim Input')
        stim_percept = TransferMechanism(name='Stimulus', size=2, function=Logistic)
        instruction_input = ProcessingMechanism(size=2, function=Linear(slope=10))
        attention = LCAMechanism(name='Attention', size=2, function=Logistic,
                                 leak=8, competition=8, self_excitation=0,
                                 noise=0, time_step_size=.1,
                                 termination_threshold=3,
                                 termination_measure=TimeScale.TRIAL)
        decision = TransferMechanism(name='Decision', size=2,
                                     integrator_mode=True,
                                     execute_until_finished=False,
                                     termination_threshold=0.65,
                                     termination_measure=max,
                                     termination_comparison_op=GREATER_THAN)
        response = ProcessingMechanism(size=2, name='Response')

        comp = Composition()
        comp.add_linear_processing_pathway([stim_input, [[1,-1],[-1,1]], stim_percept, decision, response])
        comp.add_linear_processing_pathway([instruction_input, attention, stim_percept])
        inputs = {stim_input: [[1, 1], [1, 1]],
                  instruction_input: [[1, -1], [-1, 1]]}
        result = comp.run(inputs=inputs, bin_execute=bin_execute)

        assert np.allclose(result, [[0.43636140750487973, 0.47074475219780554]])
        if bin_execute == 'Python':
            assert decision.num_executions.time_step == 1
            assert decision.num_executions.pass_ == 2
            assert decision.num_executions.trial== 1
            assert decision.num_executions.run == 2


class TestClip:
    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_clip_float(self, mode):
        T = TransferMechanism(clip=[-2.0, 2.0])
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            EX = pnlvm.execution.MechExecution(T).execute
        elif mode == 'PTX':
            EX = pnlvm.execution.MechExecution(T).cuda_execute

        assert np.allclose(EX(3.0), 2.0)
        assert np.allclose(EX(1.0), 1.0)
        assert np.allclose(EX(-3.0), -2.0)

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_clip_array(self, mode):
        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            EX = pnlvm.execution.MechExecution(T).execute
        elif mode == 'PTX':
            EX = pnlvm.execution.MechExecution(T).cuda_execute
        assert np.allclose(EX([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])

    @pytest.mark.mechanism
    @pytest.mark.transfer_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_clip_2d_array(self, mode):
        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        if mode == 'Python':
            EX = T.execute
        elif mode == 'LLVM':
            EX = pnlvm.execution.MechExecution(T).execute
        elif mode == 'PTX':
            EX = pnlvm.execution.MechExecution(T).cuda_execute

        assert np.allclose(EX([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
                           [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])


class TestOutputPorts:
    def test_output_ports_match_input_ports(self):
        T = TransferMechanism(default_variable=[[0], [0], [0]])
        assert len(T.input_ports) == 3
        assert len(T.output_ports) == 3

        T.execute(input=[[1.0], [2.0], [3.0]])

        assert np.allclose(T.value, [[1.0], [2.0], [3.0]])
        assert np.allclose(T.output_ports[0].value, [1.0])
        assert np.allclose(T.output_ports[1].value, [2.0])
        assert np.allclose(T.output_ports[2].value, [3.0])

    def test_add_input_ports(self):
        T = TransferMechanism(default_variable=[[0], [0], [0]])
        I = InputPort(owner=T,
                      variable=[4.0],
                      reference_value=[4.0],
                      name="extra InputPort")
        T.add_ports([I])
        print("Number of input ports: ", len(T.input_ports))
        print(T.input_ports, "\n\n")
        print("Number of output ports: ", len(T.output_ports))
        print(T.output_ports)

    def test_combine_standard_output_port(self):
        T = TransferMechanism(default_variable=[[0,0,0],[0,0,0]], output_ports=[COMBINE])
        T.execute([[1,2,1],[5,0,4]])
        assert np.allclose(T.output_ports[0].value, [6,2,5])

        # assert len(T.input_ports) == 4
        # assert len(T.output_ports) == 4
        #
        # T.execute(input=[[1.0], [2.0], [3.0], [4.0]])
        #
        # assert np.allclose(T.value, [[1.0], [2.0], [3.0], [4.0]])
        # assert np.allclose(T.output_ports[0].value, [1.0])
        # assert np.allclose(T.output_ports[1].value, [2.0])
        # assert np.allclose(T.output_ports[2].value, [3.0])
        # assert np.allclose(T.output_ports[3].value, [4.0])
