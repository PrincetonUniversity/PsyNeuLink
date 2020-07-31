import numpy as np
import pytest

import psyneulink as pnl
import psyneulink.core.llvm as pnlvm

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.combinationfunctions import Reduce
from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.function import FunctionError, get_matrix
from psyneulink.core.components.functions.learningfunctions import Reinforcement
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AccumulatorIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear, Logistic
from psyneulink.core.components.mechanisms.mechanism import MechanismError
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferError, TransferMechanism
from psyneulink.core.globals.keywords import MATRIX_KEYWORD_VALUES, RANDOM_CONNECTIVITY_MATRIX, RESULT
from psyneulink.core.globals.preferences.basepreferenceset import REPORT_OUTPUT_PREF, VERBOSE_PREF
from psyneulink.core.globals.utilities import UtilitiesError
from psyneulink.core.globals.parameters import ParameterError
from psyneulink.core.scheduling.condition import Never
from psyneulink.library.components.mechanisms.processing.transfer.recurrenttransfermechanism import \
    RecurrentTransferError, RecurrentTransferMechanism
from psyneulink.library.components.projections.pathway.autoassociativeprojection import AutoAssociativeProjection

class TestMatrixSpec:
    def test_recurrent_mech_matrix(self):

        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0]])
        recurrent_mech = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                          matrix=[[1.0, 2.0, 3.0],
                                                                  [2.0, 1.0, 2.0],
                                                                  [3.0, 2.0, 1.0]])
        c = Composition(pathways=[T, recurrent_mech])

        results = []
        def record_trial():
            results.append(recurrent_mech.parameters.value.get(c))
        c.run(inputs=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
              call_after_trial=record_trial)
        assert True

    def test_recurrent_mech_auto_associative_projection(self):

        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0]])
        recurrent_mech = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                          matrix=AutoAssociativeProjection)
        c = Composition(pathways=[T, recurrent_mech])

        results = []
        def record_trial():
            results.append(recurrent_mech.parameters.value.get(c))
        c.run(inputs=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
              call_after_trial=record_trial)

    def test_recurrent_mech_auto_auto_hetero(self):

        T = TransferMechanism(default_variable=[[0.0, 0.0, 0.0]])
        recurrent_mech = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                                    auto=3.0,
                                                    hetero=-7.0)

        c = Composition(pathways=[T, recurrent_mech])

        results = []
        def record_trial():
            results.append(recurrent_mech.parameters.value.get(c))
        c.run(inputs=[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
              call_after_trial=record_trial)

class TestRecurrentTransferMechanismInputs:

    def test_recurrent_mech_empty_spec(self):
        R = RecurrentTransferMechanism(auto=1.0)
        np.testing.assert_allclose(R.value, R.defaults.value)
        np.testing.assert_allclose(R.defaults.variable, [[0]])
        np.testing.assert_allclose(R.matrix, [[1]])

    def test_recurrent_mech_check_attrs(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            auto=1.0
        )
        print("matrix = ", R.matrix)
        print("auto = ", R.auto)
        print("hetero = ", R.hetero)
        # np.testing.assert_allclose(R.value, R.defaults.value)
        # np.testing.assert_allclose(R.defaults.variable, [[0., 0., 0.]])
        # np.testing.assert_allclose(R.matrix, [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])

    def test_recurrent_mech_check_proj_attrs(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3
        )
        np.testing.assert_allclose(R.recurrent_projection.matrix, R.matrix)
        assert R.recurrent_projection.sender is R.output_port
        assert R.recurrent_projection.receiver is R.input_port

    @pytest.mark.mechanism
    @pytest.mark.recurrent_transfer_mechanism
    @pytest.mark.benchmark(group="RecurrentTransferMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_recurrent_mech_inputs_list_of_ints(self, benchmark, mode):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0]
        )
        if mode == 'Python':
            EX = R.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(R)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(R)
            EX = e.cuda_execute

        val1 = EX([10, 12, 0, -1])
        val2 = EX([1, 2, 3, 0])
        benchmark(EX, [1, 2, 3, 0])

        np.testing.assert_allclose(val1, [[10.0, 12.0, 0, -1]])
        np.testing.assert_allclose(val2, [[1, 2, 3, 0]])  # because recurrent projection is not used when executing: mech is reset each time

    @pytest.mark.mechanism
    @pytest.mark.recurrent_transfer_mechanism
    @pytest.mark.benchmark(group="RecurrentTransferMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_recurrent_mech_inputs_list_of_floats(self, benchmark, mode):
        R = RecurrentTransferMechanism(
            name='R',
            size=4
        )
        if mode == 'Python':
            EX = R.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(R)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(R)
            EX = e.cuda_execute

        val = benchmark(EX, [10.0, 10.0, 10.0, 10.0])

        np.testing.assert_allclose(val, [[10.0, 10.0, 10.0, 10.0]])

    @pytest.mark.mechanism
    @pytest.mark.recurrent_transfer_mechanism
    @pytest.mark.benchmark(group="RecurrentTransferMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_recurrent_mech_integrator(self, benchmark, mode):
        R = RecurrentTransferMechanism(size=2,
                                       function=Logistic(),
                                       hetero=-2.0,
                                       integrator_mode=True,
                                       integration_rate=0.01,
                                       output_ports = [RESULT])
        if mode == 'Python':
            EX = R.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(R)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(R)
            EX = e.cuda_execute

        val1 = EX([[1.0, 2.0]])
        val2 = EX([[1.0, 2.0]])
        # execute 10 times
        for i in range(10):
            val = EX([[1.0, 2.0]])
        benchmark(EX, [[1.0, 2.0]])

        assert np.allclose(val1, [[0.50249998, 0.50499983]])
        assert np.allclose(val2, [[0.50497484, 0.50994869]])
        assert np.allclose(val, [[0.52837327, 0.55656439]])

    # def test_recurrent_mech_inputs_list_of_fns(self):
    #     R = RecurrentTransferMechanism(
    #         name='R',
    #         size=4,
    #         integrator_mode=True
    #     )
    #     val = R.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()])
    #     expected = [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]
    #     assert len(val) == len(expected) == 1
    #     assert len(val[0]) == len(expected[0])
    #     for i in range(len(val[0])):
    #         np.testing.assert_allclose(val[0][i], expected[0][i])

    @pytest.mark.mechanism
    @pytest.mark.recurrent_transfer_mechanism
    @pytest.mark.benchmark(group="RecurrentTransferMechanism")
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVM', marks=pytest.mark.llvm),
                                      pytest.param('PTX', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    def test_recurrent_mech_no_inputs(self, benchmark, mode):
        R = RecurrentTransferMechanism(
            name='R'
        )
        np.testing.assert_allclose(R.defaults.variable, [[0]])
        if mode == 'Python':
            EX = R.execute
        elif mode == 'LLVM':
            e = pnlvm.execution.MechExecution(R)
            EX = e.execute
        elif mode == 'PTX':
            e = pnlvm.execution.MechExecution(R)
            EX = e.cuda_execute

        val = EX([10])
        benchmark(EX, [1])
        np.testing.assert_allclose(val, [[10.]])

    def test_recurrent_mech_inputs_list_of_strings(self):
        with pytest.raises(FunctionError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                integrator_mode=True
            )
            R.execute(["one", "two", "three", "four"])
        assert "Unrecognized type" in str(error_text.value)

    def test_recurrent_mech_var_list_of_strings(self):
        with pytest.raises(ParameterError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=['a', 'b', 'c', 'd'],
                integrator_mode=True
            )
        assert "non-numeric entries" in str(error_text.value)

    def test_recurrent_mech_inputs_mismatched_with_default_longer(self):
        with pytest.raises(MechanismError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=4
            )
            R.execute([1, 2, 3, 4, 5])
        assert "does not match required length" in str(error_text.value)

    def test_recurrent_mech_inputs_mismatched_with_default_shorter(self):
        with pytest.raises(MechanismError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=6
            )
            R.execute([1, 2, 3, 4, 5])
        assert "does not match required length" in str(error_text.value)


class TestRecurrentTransferMechanismMatrix:

    @pytest.mark.parametrize("matrix", MATRIX_KEYWORD_VALUES)
    def test_recurrent_mech_matrix_keyword_spec(self, matrix):

        if matrix == RANDOM_CONNECTIVITY_MATRIX:
            pytest.skip("Random test")
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            matrix=matrix
        )
        val = R.execute([10, 10, 10, 10])
        np.testing.assert_allclose(val, [[10., 10., 10., 10.]])
        np.testing.assert_allclose(R.recurrent_projection.matrix, get_matrix(matrix, R.size[0], R.size[0]))

    @pytest.mark.parametrize("matrix", [np.matrix('1 2; 3 4'), np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]], '1 2; 3 4'])
    def test_recurrent_mech_matrix_other_spec(self, matrix):

        R = RecurrentTransferMechanism(
            name='R',
            size=2,
            matrix=matrix
        )
        val = R.execute([10, 10])

        # np.testing.assert_allclose(val, [[10., 10.]])
        # assert isinstance(R.matrix, np.ndarray)
        # np.testing.assert_allclose(R.matrix, [[1, 2], [3, 4]])
        # np.testing.assert_allclose(R.recurrent_projection.matrix, [[1, 2], [3, 4]])
        # assert isinstance(R.recurrent_projection.matrix, np.ndarray)

    def test_recurrent_mech_matrix_auto_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            auto=2
        )
        assert isinstance(R.matrix, np.ndarray)
        np.testing.assert_allclose(R.matrix, [[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        np.testing.assert_allclose(run_twice_in_composition(R, [1, 2, 3], [10, 11, 12]), [17, 19, 21])

    def test_recurrent_mech_matrix_hetero_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            hetero=-1
        )
        # (7/28/17 CW) these numbers assume that execute() leaves its value in the outputPort of the mechanism: if
        # the behavior of execute() changes, feel free to change these numbers
        val = R.execute([-1, -2, -3])
        np.testing.assert_allclose(val, [[-1, -2, -3]])
        assert isinstance(R.matrix, np.ndarray)
        np.testing.assert_allclose(R.matrix, [[0, -1, -1], [-1, 0, -1], [-1, -1, 0]])
        # Execution 1:
        # Recurrent input = [5, 4, 3] | New input = [1, 2, 3] | Total input = [6, 6, 6]
        # Output 1 = [6, 6, 6]
        # Execution 2:
        # Recurrent input =[-12, -12, -12] | New input =  [10, 11, 12] | Total input = [-2, -1, 0]
        # Output 2 =  [-2, -1, 0]
        np.testing.assert_allclose(run_twice_in_composition(R, [1, 2, 3], [10, 11, 12]), [-2., -1.,  0.])

    def test_recurrent_mech_matrix_auto_hetero_spec_size_1(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=1,
            auto=-2,
            hetero=4.4
        )
        val = R.execute([10])
        np.testing.assert_allclose(val, [[10.]])
        assert isinstance(R.matrix, np.ndarray)
        np.testing.assert_allclose(R.matrix, [[-2]])

    def test_recurrent_mech_matrix_auto_hetero_spec_size_4(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=2.2,
            hetero=-3
        )
        val = R.execute([10, 10, 10, 10])
        np.testing.assert_allclose(val, [[10., 10., 10., 10.]])
        np.testing.assert_allclose(R.matrix, [[2.2, -3, -3, -3], [-3, 2.2, -3, -3], [-3, -3, 2.2, -3], [-3, -3, -3, 2.2]])
        assert isinstance(R.matrix, np.ndarray)

    def test_recurrent_mech_matrix_auto_hetero_matrix_spec(self):
        # when auto, hetero, and matrix are all specified, auto and hetero should take precedence
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=2.2,
            hetero=-3,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([10, 10, 10, 10])
        np.testing.assert_allclose(val, [[10., 10., 10., 10.]])
        np.testing.assert_allclose(R.matrix, [[2.2, -3, -3, -3], [-3, 2.2, -3, -3], [-3, -3, 2.2, -3], [-3, -3, -3, 2.2]])
        assert isinstance(R.matrix, np.ndarray)

    def test_recurrent_mech_auto_matrix_spec(self):
        # auto should override the diagonal only
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=2.2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([10, 11, 12, 13])
        np.testing.assert_allclose(val, [[10., 11., 12., 13.]])
        np.testing.assert_allclose(R.matrix, [[2.2, 2, 3, 4], [1, 2.2, 3, 4], [1, 2, 2.2, 4], [1, 2, 3, 2.2]])

    def test_recurrent_mech_auto_array_matrix_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=[1.1, 2.2, 3.3, 4.4],
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([10, 11, 12, 13])
        np.testing.assert_allclose(val, [[10., 11., 12., 13.]])
        np.testing.assert_allclose(R.matrix, [[1.1, 2, 3, 4], [1, 2.2, 3, 4], [1, 2, 3.3, 4], [1, 2, 3, 4.4]])

    def test_recurrent_mech_hetero_float_matrix_spec(self):
        # hetero should override off-diagonal only
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            hetero=-2.2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[1., 2., 3., 4.]])
        np.testing.assert_allclose(
            R.matrix,
            [[1, -2.2, -2.2, -2.2], [-2.2, 2, -2.2, -2.2], [-2.2, -2.2, 3, -2.2], [-2.2, -2.2, -2.2, 4]]
        )

    def test_recurrent_mech_hetero_matrix_matrix_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[1., 2., 3., 4.]])
        np.testing.assert_allclose(
            R.matrix,
            [[1, -3, -2, -1], [-4, 2, -2, -1], [-4, -3, 3, -1], [-4, -3, -2, 4]]
        )

    def test_recurrent_mech_auto_hetero_matrix_spec_v1(self):
        # auto and hetero should override matrix
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=[1, 3, 5, 7],
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[1., 2., 3., 4.]])
        np.testing.assert_allclose(
            R.matrix,
            [[1, -3, -2, -1], [-4, 3, -2, -1], [-4, -3, 5, -1], [-4, -3, -2, 7]]
        )

    def test_recurrent_mech_auto_hetero_matrix_spec_v2(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=[3],
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[1., 2., 3., 4.]])
        np.testing.assert_allclose(
            R.matrix,
            [[3, -3, -2, -1], [-4, 3, -2, -1], [-4, -3, 3, -1], [-4, -3, -2, 3]]
        )

    def test_recurrent_mech_auto_hetero_matrix_spec_v3(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=[3],
            hetero=2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[1., 2., 3., 4.]])
        np.testing.assert_allclose(
            R.matrix,
            [[3, 2, 2, 2], [2, 3, 2, 2], [2, 2, 3, 2], [2, 2, 2, 3]]
        )

    def test_recurrent_mech_matrix_too_large(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=3,
                matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
            )

        assert "must be the same as its variable" in str(error_text.value)

    def test_recurrent_mech_matrix_too_small(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=5,
                matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
            )
        assert "must be the same as its variable" in str(error_text.value)

    def test_recurrent_mech_matrix_strings(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=4,
                matrix=[['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd'], ['a', 'b', 'c', 'd']]
            )
        assert "has non-numeric entries" in str(error_text.value)

    def test_recurrent_mech_matrix_nonsquare(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=4,
                matrix=[[1, 3]]
            )
        assert "must be square" in str(error_text.value)

    def test_recurrent_mech_matrix_3d(self):
        with pytest.raises(FunctionError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=2,
                matrix=[[[1, 3], [2, 4]], [[5, 7], [6, 8]]]
            )
        assert "more than 2d" in str(error_text.value)


class TestRecurrentTransferMechanismFunction:

    def test_recurrent_mech_function_logistic(self):

        R = RecurrentTransferMechanism(
            name='R',
            size=10,
            function=Logistic(gain=2, offset=1)
        )
        val = R.execute(np.ones(10))
        np.testing.assert_allclose(val, [np.full(10, 0.7310585786300049)])

    def test_recurrent_mech_function_psyneulink(self):

        a = Logistic(gain=2, offset=1)

        R = RecurrentTransferMechanism(
            name='R',
            size=7,
            function=a
        )
        val = R.execute(np.zeros(7))
        np.testing.assert_allclose(val, [np.full(7, 0.2689414213699951)])

    def test_recurrent_mech_function_custom(self):
        # I don't know how to do this at the moment but it seems highly important.
        pass

    def test_recurrent_mech_normal_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=NormalDist(),
                integration_rate=1.0,
                integrator_mode=True
            )
            R.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_reinforcement_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=Reinforcement(),
                integration_rate=1.0,
                integrator_mode=True
            )
            R.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_integrator_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=AccumulatorIntegrator(),
                integration_rate=1.0,
                integrator_mode=True
            )
            R.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_reduce_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=Reduce(),
                integration_rate=1.0,
                integrator_mode=True
            )
            R.execute([0, 0, 0, 0])
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)


class TestRecurrentTransferMechanismTimeConstant:

    def test_recurrent_mech_integration_rate_0_8(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=0.8,
            integrator_mode=True
        )
        val = R.execute([1, 1, 1, 1])
        np.testing.assert_allclose(val, [[0.8, 0.8, 0.8, 0.8]])
        val = R.execute([1, 1, 1, 1])
        np.testing.assert_allclose(val, [[.96, .96, .96, .96]])

    def test_recurrent_mech_integration_rate_0_8_initial_0_5(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=0.8,
            initial_value=np.array([[0.5, 0.5, 0.5, 0.5]]),
            integrator_mode=True
        )
        val = R.execute([1, 1, 1, 1])
        np.testing.assert_allclose(val, [[0.9, 0.9, 0.9, 0.9]])
        val = R.execute([1, 2, 3, 4])
        np.testing.assert_allclose(val, [[.98, 1.78, 2.5800000000000005, 3.3800000000000003]])  # due to inevitable floating point errors

    def test_recurrent_mech_integration_rate_0_8_initial_1_8(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=0.8,
            initial_value=np.array([[1.8, 1.8, 1.8, 1.8]]),
            integrator_mode=True
        )
        val = R.execute([1, 1, 1, 1])
        np.testing.assert_allclose(val, [[1.16, 1.16, 1.16, 1.16]])
        val = R.execute([2, 2, 2, 2])
        np.testing.assert_allclose(val, [[1.832, 1.832, 1.832, 1.832]])
        val = R.execute([-4, -3, 0, 1])
        np.testing.assert_allclose(val, [[-2.8336, -2.0336000000000003, .36639999999999995, 1.1663999999999999]])

    def test_recurrent_mech_integration_rate_0_8_initial_1_2(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            integration_rate=0.8,
            initial_value=np.array([[-1, 1, -2, 2]]),
            integrator_mode=True
        )
        val = R.execute([3, 2, 1, 0])
        np.testing.assert_allclose(val, [[2.2, 1.8, .40000000000000013, .3999999999999999]])

# (7/28/17 CW): the below are used because it's good to test Composition anyways, and because the recurrent Projection
# won't get executed if we only use the execute() method of Mechanism: thus, to test it we must use a Composition


def run_twice_in_composition(mech, input1, input2=None):
    if input2 is None:
        input2 = input1
    c = Composition(pathways=[mech])
    c.run(inputs={mech:[input1]})
    result = c.run(inputs={mech:[input2]})
    return result[0]


class TestRecurrentTransferMechanismInProcess:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_recurrent_mech_transfer_mech_process_three_runs(self):
        # this test ASSUMES that the ParameterPort for auto and hetero is updated one run-cycle AFTER they are set by
        # lines by `R.auto = 0`. If this (potentially buggy) behavior is changed, then change these values
        R = RecurrentTransferMechanism(
            size=4,
            auto=0,
            hetero=-1
        )
        T = TransferMechanism(
            size=3,
            function=Linear
        )
        c = Composition(pathways=[R, T], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        c.run(inputs={R: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1., 2., 3., 4.]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[10., 10., 10.]])
        c.run(inputs={R: [[5, 6, 7, 8]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-4, -2, 0, 2]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[-4, -4, -4]])
        c.run(inputs={R: [[-1, 2, -2, 5.5]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-1.0, 4.0, 2.0, 11.5]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[16.5, 16.5, 16.5]])

    def test_transfer_mech_process_matrix_change(self):
        from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
        T1 = TransferMechanism(
            size=4,
            function=Linear)
        proj = MappingProjection(matrix=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])
        T2 = TransferMechanism(
            size=4,
            function=Linear)
        c = Composition(pathways=[[T1, proj, T2]])
        c.run(inputs={T1: [[1, 2, 3, 4]]})
        proj.matrix = [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
        assert np.allclose(proj.matrix, [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])
        # c.run(inputs={T1: [[1, 2, 3, 4]]})
        T1.execute([[1, 2, 3, 4]])
        proj.execute()
        # removed this assert, because before the changes of most_recent_execution_id -> most_recent_context
        # proj.matrix referred to the 'Process-0' execution_id, even though it was last executed with None
        # assert np.allclose(proj.matrix, np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]))

    def test_recurrent_mech_process_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        c = Composition(pathways=[T, R], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        R.matrix = [[2, 0, 1, 3]] * 4

        c.run(inputs={T: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 2, 3, 4]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[1, 2, 3, 4]])
        c.run(inputs={T: [[1, 3, 2, 5]]})
        np.testing.assert_allclose(R.recurrent_projection.matrix, [[2, 0, 1, 3]] * 4)
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 3, 2, 5]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[21, 3, 12, 35]])

    # this test must wait until we create a property such that R.recurrent_projection.matrix sets R.auto and R.hetero
    def test_recurrent_mech_process_proj_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        c = Composition(pathways=[T, R], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        R.recurrent_projection.matrix = [[2, 0, 1, 3]] * 4
        c.run(inputs={T: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 2, 3, 4]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[1, 2, 3, 4]])
        c.run(inputs={T: [[1, 3, 2, 5]]})
        np.testing.assert_allclose(R.recurrent_projection.matrix, [[2, 0, 1, 3]] * 4)
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 3, 2, 5]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[21, 3, 12, 35]])


class TestRecurrentTransferMechanismInComposition:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_recurrent_mech_transfer_mech_composition_three_runs(self):
        # this test ASSUMES that the ParameterPort for auto and hetero is updated one run-cycle AFTER they are set by
        # lines by `R.auto = 0`. If this (potentially buggy) behavior is changed, then change these values
        R = RecurrentTransferMechanism(
            size=4,
            auto=0,
            hetero=-1)
        T = TransferMechanism(
            size=3,
            function=Linear)
        c = Composition(pathways=[R,T])

        c.run(inputs={R: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1., 2., 3., 4.]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[10., 10., 10.]])
        c.run(inputs={R: [[5, 6, 7, 8]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-4, -2, 0, 2]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[-4, -4, -4]])
        c.run(inputs={R: [[-1, 2, -2, 5.5]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-1.0, 4.0, 2.0, 11.5]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[16.5, 16.5, 16.5]])

    @pytest.mark.xfail(reason='Unsure if this is correct behavior - see note for _recurrent_transfer_mechanism_matrix_setter')
    def test_recurrent_mech_composition_auto_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=-1)
        T = TransferMechanism(
            size=3,
            function=Linear)
        c = Composition(pathways=[R, T], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        c.run(inputs={R: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1., 2., 3., 4.]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[10., 10., 10.]])
        R.parameters.auto.set(0, c)
        c.run(inputs={R: [[5, 6, 7, 8]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-4, -2, 0, 2]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[-4, -4, -4]])
        R.recurrent_projection.parameters.auto.set([1, 1, 2, 4], c)
        c.run(inputs={R: [[12, 11, 10, 9]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[8, 11, 14, 23]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[56, 56, 56]])

    @pytest.mark.xfail(reason='Unsure if this is correct behavior - see note for _recurrent_transfer_mechanism_matrix_setter')
    def test_recurrent_mech_composition_hetero_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=[[-1, -2, -3, -4]] * 4)
        T = TransferMechanism(
            size=5,
            function=Linear)
        c = Composition(pathways=[R, T], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        c.run(inputs={R: [[1, 2, 3, -0.5]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1., 2., 3., -0.5]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[5.5, 5.5, 5.5, 5.5, 5.5]])
        R.parameters.hetero.set(0, c)
        c.run(inputs={R: [[-1.5, 0, 1, 2]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-.5, 4, 10, 0]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[13.5, 13.5, 13.5, 13.5, 13.5]])
        R.parameters.hetero.set(np.array([[-1, 2, 3, 1.5]] * 4), c)
        c.run(inputs={R: [[12, 11, 10, 9]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-2.5, 38, 50.5, 29.25]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[115.25, 115.25, 115.25, 115.25, 115.25]])

    @pytest.mark.xfail(reason='Unsure if this is correct behavior - see note for _recurrent_transfer_mechanism_matrix_setter')
    def test_recurrent_mech_composition_auto_and_hetero_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=[[-1, -2, -3, -4]] * 4)
        T = TransferMechanism(
            size=5,
            function=Linear)
        c = Composition(pathways=[R,T], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        c.run(inputs={R: [[1, 2, 3, -0.5]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1., 2., 3., -0.5]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[5.5, 5.5, 5.5, 5.5, 5.5]])
        R.parameters.hetero.set(0, c)
        c.run(inputs={R: [[-1.5, 0, 1, 2]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[-.5, 4, 10, 0]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[13.5, 13.5, 13.5, 13.5, 13.5]])
        R.parameters.auto.set([0, 0, 0, 0], c)
        c.run(inputs={R: [[12, 11, 10, 9]]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[12, 11, 10, 9]])
        np.testing.assert_allclose(T.parameters.value.get(c), [[42, 42, 42, 42, 42]])

    @pytest.mark.xfail(reason='Unsure if this is correct behavior - see note for _recurrent_transfer_mechanism_matrix_setter')
    def test_recurrent_mech_composition_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        c = Composition(pathways=[T, R], prefs=TestRecurrentTransferMechanismInComposition.simple_prefs)
        R.parameters.matrix.set([[2, 0, 1, 3]] * 4, c)
        c.run(inputs={T: [[1, 2, 3, 4]]})
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 2, 3, 4]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[1, 2, 3, 4]])
        c.run(inputs={T: [[1, 3, 2, 5]]})
        np.testing.assert_allclose(R.recurrent_projection.parameters.matrix.get(c), [[2, 0, 1, 3]] * 4)
        np.testing.assert_allclose(T.parameters.value.get(c), [[1, 3, 2, 5]])
        np.testing.assert_allclose(R.parameters.value.get(c), [[21, 3, 12, 35]])

    def test_recurrent_mech_with_learning(self):
        R = RecurrentTransferMechanism(size=4,
                                       function=Linear,
                                       matrix=np.full((4, 4), 0.1),
                                       enable_learning=True
                                       )
        # Test that all of these are the same:
        np.testing.assert_allclose(
            R.recurrent_projection.mod_matrix,
            [
                [0.1,  0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1],
                [0.1, 0.1, 0.1, 0.1]
            ]
        )
        np.testing.assert_allclose(R.recurrent_projection.matrix, R.matrix)
        np.testing.assert_allclose(R.input_port.path_afferents[0].matrix, R.matrix)

        # Test that activity is properly computed prior to learning
        # p = Process(pathway=[R])
        c = Composition(pathways=[R])
        R.learning_enabled = False
        c.learn(inputs={R:[1, 1, 0, 0]})
        c.learn(inputs={R:[1, 1, 0, 0]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1.2, 1.2, 0.2, 0.2]])

        # Test that activity and weight changes are properly computed with learning
        R.learning_enabled = True
        c.learn(inputs={R:[1, 1, 0, 0]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1.28, 1.28, 0.28, 0.28]])
        np.testing.assert_allclose(
            R.recurrent_projection.get_mod_matrix(c),
            [
                [0.1, 0.18192000000000003, 0.11792000000000001, 0.11792000000000001],
                [0.18192000000000003, 0.1, 0.11792000000000001, 0.11792000000000001],
                [0.11792000000000001, 0.11792000000000001, 0.1, 0.10392000000000001],
                [0.11792000000000001, 0.11792000000000001, 0.10392000000000001, 0.1]
            ]
        )
        c.learn(inputs={R:[1, 1, 0, 0]})
        np.testing.assert_allclose(R.parameters.value.get(c), [[1.4268928, 1.4268928, 0.3589728, 0.3589728]])
        np.testing.assert_allclose(
            R.recurrent_projection.get_mod_matrix(c),
            [
                [0.1, 0.28372115, 0.14353079, 0.14353079],
                [0.28372115, 0.1, 0.14353079, 0.14353079],
                [0.14353079, 0.14353079, 0.1, 0.11036307],
                [0.14353079, 0.14353079, 0.11036307, 0.1]
            ]
        )

    def test_recurrent_mech_change_learning_rate(self):
        R = RecurrentTransferMechanism(size=4,
                                       function=Linear,
                                       enable_learning=True,
                                       learning_rate=0.1
                                       )

        c = Composition(pathways=[R])
        assert R.learning_rate == 0.1
        assert R.learning_mechanism.learning_rate == 0.1
        # assert R.learning_mechanism.function.learning_rate == 0.1
        c.learn(inputs={R:[[1.0, 1.0, 1.0, 1.0]]})
        matrix_1 = [[0., 1.1, 1.1, 1.1],
                    [1.1, 0., 1.1, 1.1],
                    [1.1, 1.1, 0., 1.1],
                    [1.1, 1.1, 1.1, 0.]]
        assert np.allclose(R.recurrent_projection.mod_matrix, matrix_1)
        print(R.recurrent_projection.mod_matrix)
        R.learning_rate = 0.9

        assert R.learning_rate == 0.9
        assert R.learning_mechanism.learning_rate == 0.9
        # assert R.learning_mechanism.function.learning_rate == 0.9
        c.learn(inputs={R:[[1.0, 1.0, 1.0, 1.0]]})
        matrix_2 = [[0., 1.911125, 1.911125, 1.911125],
                    [1.911125, 0., 1.911125, 1.911125],
                    [1.911125, 1.911125, 0., 1.911125],
                    [1.911125, 1.911125, 1.911125, 0.]]
        # assert np.allclose(R.recurrent_projection.mod_matrix, matrix_2)
        print(R.recurrent_projection.mod_matrix)

    def test_learning_of_orthognal_inputs(self):
        size=4
        R = RecurrentTransferMechanism(
            size=size,
            function=Linear,
            enable_learning=True,
            auto=0,
            hetero=np.full((size,size),0.0)
            )
        C=Composition(pathways=[R])

        inputs_dict = {R:[1,0,1,0]}
        C.learn(num_trials=4,
                inputs=inputs_dict)
        np.testing.assert_allclose(
            R.recurrent_projection.get_mod_matrix(C),
            [
                [0.0,        0.0,  0.23700501,  0.0],
                [0.0,        0.0,  0.0,         0.0],
                [0.23700501, 0.0,  0.0,         0.0],
                [0.0,        0.0,  0.0,         0.0]
            ]
        )
        np.testing.assert_allclose(R.output_port.parameters.value.get(C), [1.18518086, 0.0, 1.18518086, 0.0])

        # Reset state so learning of new pattern is "uncontaminated" by activity from previous one
        R.output_port.parameters.value.set([0, 0, 0, 0], C, override=True)
        inputs_dict = {R:[0,1,0,1]}
        C.learn(num_trials=4,
                inputs=inputs_dict)
        np.testing.assert_allclose(
                R.recurrent_projection.get_mod_matrix(C),
                [
                    [0.0,        0.0,        0.23700501, 0.0       ],
                    [0.0,        0.0,        0.0,        0.23700501],
                    [0.23700501, 0.0,        0.0,        0.        ],
                    [0.0,        0.23700501, 0.0,        0.        ]
                ]
        )
        np.testing.assert_allclose(R.output_port.parameters.value.get(C),[0.0, 1.18518086, 0.0, 1.18518086])


class TestRecurrentTransferMechanismReset:

    def test_reset_run(self):

        R = RecurrentTransferMechanism(name="R",
                 initial_value=0.5,
                 integrator_mode=True,
                 integration_rate=0.1,
                 auto=1.0,
                 noise=0.0)
        R.reset_stateful_function_when = Never()
        C = Composition(pathways=[R])
        assert np.allclose(R.integrator_function.previous_value, 0.5)

        # S.run(inputs={R: 1.0},
        #       num_trials=2,
        #       initialize=True,
        #       initial_values={R: 0.0})
        from psyneulink.core.scheduling.condition import AtTrialStart, AtRunStart
        C.run(inputs={R: 1.0},
              num_trials=2,
              initialize_cycle_values={R: [0.0]}
        )

        # Trial 1    |   variable = 1.0 + 0.0
        # integration: 0.9*0.5 + 0.1*1.0 + 0.0 = 0.55  --->  previous value = 0.55
        # linear fn: 0.55*1.0 = 0.55
        # Trial 2    |   variable = 1.0 + 0.55
        # integration: 0.9*0.55 + 0.1*1.55 + 0.0 = 0.65  --->  previous value = 0.65
        # linear fn: 0.65*1.0 = 0.65
        assert np.allclose(R.integrator_function.parameters.previous_value.get(C), 0.65)

        R.integrator_function.reset(0.9, context=C)

        assert np.allclose(R.integrator_function.parameters.previous_value.get(C), 0.9)
        assert np.allclose(R.parameters.value.get(C), 0.65)

        R.reset(0.5, context=C)

        assert np.allclose(R.integrator_function.parameters.previous_value.get(C), 0.5)
        assert np.allclose(R.parameters.value.get(C), 0.5)

        C.run(inputs={R: 1.0}, num_trials=2)
        # Trial 3
        # integration: 0.9*0.5 + 0.1*1.5 + 0.0 = 0.6  --->  previous value = 0.6
        # linear fn: 0.6*1.0 = 0.6
        # Trial 4
        # integration: 0.9*0.6 + 0.1*1.6 + 0.0 = 0.7 --->  previous value = 0.7
        # linear fn: 0.7*1.0 = 0.7
        assert np.allclose(R.integrator_function.parameters.previous_value.get(C), 0.7)

class TestClip:
    def test_clip_float(self):
        R = RecurrentTransferMechanism(clip=[-2.0, 2.0])
        assert np.allclose(R.execute(3.0), 2.0)
        assert np.allclose(R.execute(-3.0), -2.0)

    def test_clip_array(self):
        R = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        assert np.allclose(R.execute([3.0, 0.0, -3.0]), [2.0, 0.0, -2.0])

    def test_clip_2d_array(self):
        R = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                              clip=[-2.0, 2.0])
        assert np.allclose(R.execute([[-5.0, -1.0, 5.0], [5.0, -5.0, 1.0], [1.0, 5.0, 5.0]]),
                           [[-2.0, -1.0, 2.0], [2.0, -2.0, 1.0], [1.0, 2.0, 2.0]])

class TestRecurrentInputPort:

    def test_ris_simple(self):
        R2 = RecurrentTransferMechanism(default_variable=[[0.0, 0.0, 0.0]],
                                            matrix=[[1.0, 2.0, 3.0],
                                                    [2.0, 1.0, 2.0],
                                                    [3.0, 2.0, 1.0]],
                                            has_recurrent_input_port=True)
        R2.execute(input=[1, 3, 2])
        c = Composition(pathways=[R2])
        c.run(inputs=[[1, 3, 2]])
        np.testing.assert_allclose(R2.parameters.value.get(c), [[14., 12., 13.]])
        assert len(R2.input_ports) == 2
        assert "Recurrent Input Port" not in R2.input_port.name  # make sure recurrent InputPort isn't primary


class TestCustomCombinationFunction:

    def test_rt_without_custom_comb_fct(self):
        R1 = RecurrentTransferMechanism(
                has_recurrent_input_port=True,
                size=2,
        )
        result = R1.execute([1,2])
        np.testing.assert_allclose(result, [[1,2]])

    def test_rt_with_custom_comb_fct(self):
        def my_fct(x):
            return x[0] * x[1] if len(x) == 2 else x[0]
        R2 = RecurrentTransferMechanism(
                has_recurrent_input_port=True,
                size=2,
                combination_function=my_fct
        )
        result = R2.execute([1,2])
        np.testing.assert_allclose(result, [[0,0]])

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    @pytest.mark.parametrize('cond0, cond1, expected', [
        (pnl.Never(), pnl.AtTrial(2),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.75]), np.array([0.75])],
          [np.array([0.875]), np.array([0.5])],   # I2 resets at Trial 2
          [np.array([0.9375]), np.array([0.75])],
          [np.array([0.96875]), np.array([0.875])],
          [np.array([0.984375]), np.array([0.9375])],
          [np.array([0.9921875]), np.array([0.96875])]]),
        (pnl.Never(), pnl.AtTrialStart(),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.75]), np.array([0.5])],
          [np.array([0.875]), np.array([0.5])],
          [np.array([0.9375]), np.array([0.5])],
          [np.array([0.96875]), np.array([0.5])],
          [np.array([0.984375]), np.array([0.5])],
          [np.array([0.9921875]), np.array([0.5])]]),
        (pnl.AtPass(0), pnl.AtTrial(2),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.5])],   # I2 resets at Trial 2
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.875])],
          [np.array([0.5]), np.array([0.9375])],
          [np.array([0.5]), np.array([0.96875])]]),
        ], ids=lambda x: str(x) if isinstance(x, pnl.Condition) else "")
    def test_reset_stateful_function_when_composition(self, mode, cond0, cond1, expected):
        I1 = pnl.RecurrentTransferMechanism(integrator_mode=True,
                                            integration_rate=0.5)
        I2 = pnl.RecurrentTransferMechanism(integrator_mode=True,
                                            integration_rate=0.5)
        I1.reset_stateful_function_when = cond0
        I2.reset_stateful_function_when = cond1
        C = pnl.Composition()
        C.add_node(I1)
        C.add_node(I2)

        C.run(inputs={I1: [[1.0]], I2: [[1.0]]}, num_trials=7, bin_execute=mode)

        assert np.allclose(expected, C.results)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    @pytest.mark.parametrize('cond0, cond1, expected', [
        (pnl.AtPass(0), pnl.AtTrial(2),
         [[np.array([0.5]), np.array([0.5])],
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.5])],   # I2 resets at Trial 2
          [np.array([0.5]), np.array([0.75])],
          [np.array([0.5]), np.array([0.875])],
          [np.array([0.5]), np.array([0.9375])],
          [np.array([0.5]), np.array([0.96875])]]),
        ], ids=lambda x: str(x) if isinstance(x, pnl.Condition) else "")
    @pytest.mark.parametrize('has_initializers2', [True, False],
                             ids=lambda x: "initializers1" if x else "NO initializers1")
    @pytest.mark.parametrize('has_initializers1', [True, False],
                             ids=lambda x: "initializers2" if x else "NO initializers2")
    def test_reset_stateful_function_when_has_initializers_composition(self, mode, cond0, cond1, expected,
                                           has_initializers1, has_initializers2):
        I1 = pnl.RecurrentTransferMechanism(integrator_mode=True,
                                            integration_rate=0.5)
        I2 = pnl.RecurrentTransferMechanism(integrator_mode=True,
                                            integration_rate=0.5)
        I1.reset_stateful_function_when = cond0
        I2.reset_stateful_function_when = cond1
        I1.has_initializers = has_initializers1
        I2.has_initializers = has_initializers2
        C = pnl.Composition()
        C.add_node(I1)
        C.add_node(I2)
        exp = expected.copy()
        def_res = [np.array([0.5]), np.array([0.75]), np.array([0.875]),
                   np.array([0.9375]), np.array([0.96875]),
                   np.array([0.984375]), np.array([0.9921875])]
        if not has_initializers1:
            exp = list(zip(def_res, (x[1] for x in exp)))
        if not has_initializers2:
            exp = list(zip((x[0] for x in exp), def_res))

        C.run(inputs={I1: [[1.0]], I2: [[1.0]]}, num_trials=7, bin_execute=mode)

        assert np.allclose(exp, C.results)

    @pytest.mark.mechanism
    @pytest.mark.integrator_mechanism
    @pytest.mark.parametrize('mode', ['Python',
                                      pytest.param('LLVMExec', marks=pytest.mark.llvm),
                                      pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                      pytest.param('PTXExec', marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                      pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])])
    @pytest.mark.parametrize('until_finished, expected', [
        (True, [[[[0.96875]]], [[[0.9990234375]]]]), # The 5th and the 10th iteration
        (False, [[[[0.5]]], [[[0.75]]]]), # The first and the second iteration
    ], ids=['until_finished', 'oneshot'])
    def test_max_executions_before_finished(self, mode, until_finished, expected):
        I1 = pnl.RecurrentTransferMechanism(integrator_mode=True,
                                            integration_rate=0.5,
                                            termination_threshold=0.0,
                                            max_executions_before_finished=5,
                                            execute_until_finished=until_finished)
        C = pnl.Composition()
        C.add_node(I1)

        results = C.run(inputs={I1: [[1.0]]}, num_trials=1, bin_execute=mode)
        if mode == 'Python':
            assert I1.parameters.is_finished_flag.get(C) is until_finished
        results2 = C.run(inputs={I1: [[1.0]]}, num_trials=1, bin_execute=mode)
        assert np.allclose(expected[0], results)
        assert np.allclose(expected[1], results2)

class TestDebugProperties:

    def test_defaults(self):
        R = RecurrentTransferMechanism(name='R',
                                       size=3)
        print("\n\nTEST DEFAULTS")
        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get() = ", R.parameters.auto.get())

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get() = ", R.parameters.hetero.get())

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get() = ", R.parameters.matrix.get())

        comp = pnl.Composition()
        comp.add_node(R)
        print("\n\n---------------------------- Run -------------------------- ")
        eid = "eid"
        inputs = {R: [[1.0, 1.0, 1.0]]}
        comp.run(inputs=inputs,
                 context=eid)

        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get(eid) = ", R.parameters.auto.get(eid))

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get(eid) = ", R.parameters.hetero.get(eid))

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get(eid) = ", R.parameters.matrix.get(eid))

    def test_auto(self):
        auto_val = 10.0
        R = RecurrentTransferMechanism(name='R',
                                       size=3,
                                       auto=auto_val)

        print("\n\nTEST AUTO     [auto = ", auto_val, "]")
        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get() = ", R.parameters.auto.get())

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get() = ", R.parameters.hetero.get())

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get() = ", R.parameters.matrix.get())

        comp = pnl.Composition()
        comp.add_node(R)
        print("\n\n---------------------------- Run -------------------------- ")
        eid = "eid"
        inputs = {R: [[1.0, 1.0, 1.0]]}
        comp.run(inputs=inputs,
                 context=eid)

        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get(eid) = ", R.parameters.auto.get(eid))

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get(eid) = ", R.parameters.hetero.get(eid))

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get(eid) = ", R.parameters.matrix.get(eid))

    def test_hetero(self):
        hetero_val = 10.0
        R = RecurrentTransferMechanism(name='R',
                                       size=3,
                                       hetero=hetero_val)
        print("\n\nTEST HETERO    [hetero = ", hetero_val, "]")
        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get() = ", R.parameters.auto.get())

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get() = ", R.parameters.hetero.get())

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get() = ", R.parameters.matrix.get())

        comp = pnl.Composition()
        comp.add_node(R)

        print("\n\n---------------------------- Run -------------------------- ")

        eid = "eid"
        inputs = {R: [[1.0, 1.0, 1.0]]}
        comp.run(inputs=inputs,
                 context=eid)

        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get(eid) = ", R.parameters.auto.get(eid))

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get(eid) = ", R.parameters.hetero.get(eid))

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get(eid) = ", R.parameters.matrix.get(eid))

    def test_auto_and_hetero(self):

        auto_val = 10.0
        hetero_val = 5.0

        R = RecurrentTransferMechanism(name='R',
                                       size=3,
                                       auto=auto_val,
                                       hetero=hetero_val)
        print("\n\nTEST AUTO AND HETERO\n [auto = ", auto_val, " | hetero = ", hetero_val, "] ")
        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get() = ", R.parameters.auto.get())

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get() = ", R.parameters.hetero.get())

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get() = ", R.parameters.matrix.get())

        comp = pnl.Composition()
        comp.add_node(R)
        print("\n\nRun")
        eid = "eid"
        inputs = {R: [[1.0, 1.0, 1.0]]}
        comp.run(inputs=inputs,
                 context=eid)

        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get(eid) = ", R.parameters.auto.get(eid))

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get(eid) = ", R.parameters.hetero.get(eid))

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get(eid) = ", R.parameters.matrix.get(eid))

    def test_matrix(self):
        matrix_val = [[ 5.0,  10.0,  10.0],
                      [10.0,   5.0,  10.0],
                      [10.0,  10.0,   5.0]]

        R = RecurrentTransferMechanism(name='R',
                                       size=3,
                                       matrix=matrix_val)
        print("\n\nTEST MATRIX\n", matrix_val)
        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get() = ", R.parameters.auto.get())

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get() = ", R.parameters.hetero.get())

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get() = ", R.parameters.matrix.get())

        comp = pnl.Composition()
        comp.add_node(R)
        print("\n\nRun")
        eid = "eid"
        inputs = {R: [[1.0, 1.0, 1.0]]}
        comp.run(inputs=inputs,
                 context=eid)

        print("\n\nAuto Values -----------------------------------")
        print("R.auto = ", R.auto)
        print("R.parameters.auto.get(eid) = ", R.parameters.auto.get(eid))

        print("\n\nHetero Values ----------------------------------")
        print("R.hetero = ", R.hetero)
        print("R.parameters.hetero.get(eid) = ", R.parameters.hetero.get(eid))

        print("\n\nMatrix Values ----------------------------------")
        print("R.matrix = ", R.matrix)
        print("R.parameters.matrix.get(eid) = ", R.parameters.matrix.get(eid))
