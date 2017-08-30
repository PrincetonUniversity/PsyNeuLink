import pytest
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.RecurrentTransferMechanism import RecurrentTransferError, \
    RecurrentTransferMechanism

from PsyNeuLink.Components.Functions.Function import ConstantIntegrator, Exponential, ExponentialDist, FunctionError, \
    Linear, Logistic, NormalDist, Reduce, Reinforcement, get_matrix
from PsyNeuLink.Components.Mechanisms.Mechanism import MechanismError
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import RANDOM_CONNECTIVITY_MATRIX
from PsyNeuLink.Globals.Preferences.ComponentPreferenceSet import REPORT_OUTPUT_PREF, VERBOSE_PREF
from PsyNeuLink.Globals.Utilities import *
from PsyNeuLink.Library.Mechanisms.ProcessingMechanisms.TransferMechanisms.TransferMechanism import TransferError, \
    TransferMechanism
from PsyNeuLink.Scheduling.TimeScale import TimeScale


class TestRecurrentTransferMechanismInputs:

    def test_recurrent_mech_empty_spec(self):
        R = RecurrentTransferMechanism()
        assert R.value is None
        assert R.instance_defaults.variable.tolist() == [[0]]
        assert R.matrix.tolist() == [[1]]

    def test_recurrent_mech_check_attrs(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3
        )
        assert R.value is None
        assert R.instance_defaults.variable.tolist() == [[0., 0., 0.]]
        assert R.matrix.tolist() == [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]

    def test_recurrent_mech_check_proj_attrs(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3
        )
        assert R.recurrent_projection.matrix.tolist() == R.matrix.tolist()
        assert R.recurrent_projection.sender is R.output_state
        assert R.recurrent_projection.receiver is R.input_state

    def test_recurrent_mech_inputs_list_of_ints(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0]
        )
        val = R.execute([10, 12, 0, -1]).tolist()
        assert val == [[10.0, 12.0, 0, -1]]
        val = R.execute([1, 2, 3, 0]).tolist()
        assert val == [[1, 2, 3, 0]] # because recurrent projection is not used when executing: mech is reset each time

    def test_recurrent_mech_inputs_list_of_floats(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4
        )
        val = R.execute([10.0, 10.0, 10.0, 10.0]).tolist()
        assert val == [[10.0, 10.0, 10.0, 10.0]]

    def test_recurrent_mech_inputs_list_of_fns(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            time_scale=TimeScale.TIME_STEP
        )
        val = R.execute([Linear().execute(), NormalDist().execute(), Exponential().execute(), ExponentialDist().execute()]).tolist()
        assert val == [[np.array([0.]), 0.4001572083672233, np.array([1.]), 0.7872011523172707]]

    def test_recurrent_mech_no_inputs(self):
        R = RecurrentTransferMechanism(
            name='R'
        )
        assert R.instance_defaults.variable.tolist() == [[0]]
        val = R.execute([10]).tolist()
        assert val == [[10.]]

    def test_recurrent_mech_inputs_list_of_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                time_scale=TimeScale.TIME_STEP
            )
            R.execute(["one", "two", "three", "four"]).tolist()
        assert "has non-numeric entries" in str(error_text.value)

    def test_recurrent_mech_var_list_of_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=['a', 'b', 'c', 'd'],
                time_scale=TimeScale.TIME_STEP
            )
        assert "has non-numeric entries" in str(error_text.value)

    def test_recurrent_mech_inputs_mismatched_with_default_longer(self):
        with pytest.raises(MechanismError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=4
            )
            R.execute([1, 2, 3, 4, 5]).tolist()
        assert "does not match required length" in str(error_text.value)

    def test_recurrent_mech_inputs_mismatched_with_default_shorter(self):
        with pytest.raises(MechanismError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=6
            )
            R.execute([1, 2, 3, 4, 5]).tolist()
        assert "does not match required length" in str(error_text.value)

class TestRecurrentTransferMechanismMatrix:

    def test_recurrent_mech_matrix_keyword_spec(self):

        for m in MATRIX_KEYWORD_VALUES:
            if m == RANDOM_CONNECTIVITY_MATRIX:
                continue
            R = RecurrentTransferMechanism(
                name='R',
                size=4,
                matrix=m
            )
            val = R.execute([10, 10, 10, 10]).tolist()
            assert val == [[10., 10., 10., 10.]]
            assert R.recurrent_projection.matrix.tolist() == get_matrix(m, R.size[0], R.size[0]).tolist()

    def test_recurrent_mech_matrix_other_spec(self):

        specs = [np.matrix('1 2; 3 4'), np.array([[1, 2], [3, 4]]), [[1, 2], [3, 4]], '1 2; 3 4']
        for m in specs:
            R = RecurrentTransferMechanism(
                name='R',
                size=2,
                matrix=m
            )
            val = R.execute([10, 10]).tolist()
            assert val == [[10., 10.]]
            assert R.matrix.tolist() == [[1, 2], [3, 4]] and isinstance(R.matrix, np.ndarray)
            assert R.recurrent_projection.matrix.tolist() == [[1, 2], [3, 4]]
            assert isinstance(R.recurrent_projection.matrix, np.ndarray)

    def test_recurrent_mech_matrix_auto_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            auto=2
        )
        assert R.matrix.tolist() == [[2, 1, 1], [1, 2, 1], [1, 1, 2]] and isinstance(R.matrix, np.ndarray)
        assert run_twice_in_system(R, [1, 2, 3], [10, 11, 12]) == [17, 19, 21]

    def test_recurrent_mech_matrix_hetero_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=3,
            hetero=-1
        )
        # (7/28/17 CW) these numbers assume that execute() leaves its value in the outputState of the mechanism: if
        # the behavior of execute() changes, feel free to change these numbers
        val = R.execute([-1, -2, -3]).tolist()
        assert val == [[-1, -2, -3]]
        assert R.matrix.tolist() == [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]] and isinstance(R.matrix, np.ndarray)
        assert run_twice_in_system(R, [1, 2, 3], [10, 11, 12]) == [8, 7, 6]

    def test_recurrent_mech_matrix_auto_hetero_spec_size_1(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=1,
            auto=-2,
            hetero=4.4
        )
        val = R.execute([10]).tolist()
        assert val == [[10.]]
        assert R.matrix.tolist() == [[-2]] and isinstance(R.matrix, np.ndarray)

    def test_recurrent_mech_matrix_auto_hetero_spec_size_4(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=2.2,
            hetero=-3
        )
        val = R.execute([10, 10, 10, 10]).tolist()
        assert val == [[10., 10., 10., 10.]]
        assert R.matrix.tolist() == [[2.2, -3, -3, -3], [-3, 2.2, -3, -3], [-3, -3, 2.2, -3], [-3, -3, -3, 2.2]]
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
        val = R.execute([10, 10, 10, 10]).tolist()
        assert val == [[10., 10., 10., 10.]]
        assert R.matrix.tolist() == [[2.2, -3, -3, -3], [-3, 2.2, -3, -3], [-3, -3, 2.2, -3], [-3, -3, -3, 2.2]]
        assert isinstance(R.matrix, np.ndarray)

    def test_recurrent_mech_auto_matrix_spec(self):
        # auto should override the diagonal only
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=2.2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([10, 11, 12, 13]).tolist()
        assert val == [[10., 11., 12., 13.]]
        assert R.matrix.tolist() == [[2.2, 2, 3, 4], [1, 2.2, 3, 4], [1, 2, 2.2, 4], [1, 2, 3, 2.2]]

    def test_recurrent_mech_auto_array_matrix_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto=[1.1, 2.2, 3.3, 4.4],
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([10, 11, 12, 13]).tolist()
        assert val == [[10., 11., 12., 13.]]
        assert R.matrix.tolist() == [[1.1, 2, 3, 4], [1, 2.2, 3, 4], [1, 2, 3.3, 4], [1, 2, 3, 4.4]]

    def test_recurrent_mech_hetero_float_matrix_spec(self):
        # hetero should override off-diagonal only
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            hetero=-2.2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[1., 2., 3., 4.]]
        assert R.matrix.tolist() == [[1, -2.2, -2.2, -2.2], [-2.2, 2, -2.2, -2.2], [-2.2, -2.2, 3, -2.2],
                                     [-2.2, -2.2, -2.2, 4]]

    def test_recurrent_mech_hetero_matrix_matrix_spec(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[1., 2., 3., 4.]]
        assert R.matrix.tolist() == [[1, -3, -2, -1], [-4, 2, -2, -1], [-4, -3, 3, -1],
                                     [-4, -3, -2, 4]]

    def test_recurrent_mech_auto_hetero_matrix_spec_v1(self):
        # auto and hetero should override matrix
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto = [1, 3, 5, 7],
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[1., 2., 3., 4.]]
        assert R.matrix.tolist() == [[1, -3, -2, -1], [-4, 3, -2, -1], [-4, -3, 5, -1],
                                     [-4, -3, -2, 7]]

    def test_recurrent_mech_auto_hetero_matrix_spec_v2(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto = [3],
            hetero=np.array([[-4, -3, -2, -1]] * 4),
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[1., 2., 3., 4.]]
        assert R.matrix.tolist() == [[3, -3, -2, -1], [-4, 3, -2, -1], [-4, -3, 3, -1],
                                     [-4, -3, -2, 3]]

    def test_recurrent_mech_auto_hetero_matrix_spec_v3(self):
        R = RecurrentTransferMechanism(
            name='R',
            size=4,
            auto = [3],
            hetero=2,
            matrix=[[1, 2, 3, 4]] * 4
        )
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[1., 2., 3., 4.]]
        assert R.matrix.tolist() == [[3, 2, 2, 2], [2, 3, 2, 2], [2, 2, 3, 2],
                                     [2, 2, 2, 3]]

    def test_recurrent_mech_matrix_too_large(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=3,
                matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
            )

        assert "must be same as the size of variable" in str(error_text.value)

    def test_recurrent_mech_matrix_too_small(self):
        with pytest.raises(RecurrentTransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                size=5,
                matrix=[[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
            )
        assert "must be same as the size of variable" in str(error_text.value)

    def test_recurrent_mech_matrix_strings(self):
        with pytest.raises(UtilitiesError) as error_text:
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

    def test_recurrent_mech_matrix_none_auto_none(self):
        for a in [None, 1, [1, 2, 5]]:
                with pytest.raises(RecurrentTransferError) as error_text:
                    R = RecurrentTransferMechanism(
                        name = 'R',
                        size = 3,
                        matrix = None,
                        auto = a
                    )
                assert "failed to produce a suitable matrix" in str(error_text.value)

class TestRecurrentTransferMechanismFunction:

    def test_recurrent_mech_function_logistic(self):

        R = RecurrentTransferMechanism(
            name='R',
            size=10,
            function=Logistic(gain=2, bias=1)
        )
        val = R.execute(np.ones(10)).tolist()
        assert val == [np.full(10, 0.7310585786300049).tolist()]

    def test_recurrent_mech_function_psyneulink(self):

        a = Logistic(gain=2, bias=1)

        R = RecurrentTransferMechanism(
            name='R',
            size=7,
            function=a
        )
        val = R.execute(np.zeros(7)).tolist()
        assert val == [np.full(7, 0.2689414213699951).tolist()]

    def test_recurrent_mech_function_custom(self):
        # I don't know how to do this at the moment but it seems highly important.
        pass

    def test_recurrent_mech_normal_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=NormalDist(),
                time_constant=1.0,
                time_scale=TimeScale.TIME_STEP
            )
            R.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_reinforcement_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=Reinforcement(),
                time_constant=1.0,
                time_scale=TimeScale.TIME_STEP
            )
            R.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_integrator_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=ConstantIntegrator(),
                time_constant=1.0,
                time_scale=TimeScale.TIME_STEP
            )
            R.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

    def test_recurrent_mech_reduce_fun(self):
        with pytest.raises(TransferError) as error_text:
            R = RecurrentTransferMechanism(
                name='R',
                default_variable=[0, 0, 0, 0],
                function=Reduce(),
                time_constant=1.0,
                time_scale=TimeScale.TIME_STEP
            )
            R.execute([0, 0, 0, 0]).tolist()
        assert "must be a TRANSFER FUNCTION TYPE" in str(error_text.value)

class TestRecurrentTransferMechanismTimeConstant:

    def test_recurrent_mech_time_constant_0_8(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            time_scale=TimeScale.TIME_STEP
        )
        val = R.execute([1, 1, 1, 1]).tolist()
        assert val == [[0.8, 0.8, 0.8, 0.8]]
        val = R.execute([1, 1, 1, 1]).tolist()
        assert val == [[.96, .96, .96, .96]]

    def test_recurrent_mech_time_constant_0_8_initial_0_5(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            initial_value=np.array([[0.5, 0.5, 0.5, 0.5]]),
            time_scale=TimeScale.TIME_STEP
        )
        val = R.execute([1, 1, 1, 1]).tolist()
        assert val == [[0.9, 0.9, 0.9, 0.9]]
        val = R.execute([1, 2, 3, 4]).tolist()
        assert val == [[.98, 1.78, 2.5800000000000005, 3.3800000000000003]] # due to inevitable floating point errors

    def test_recurrent_mech_time_constant_0_8_initial_1_8(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            initial_value=np.array([[1.8, 1.8, 1.8, 1.8]]),
            time_scale=TimeScale.TIME_STEP
        )
        val = R.execute([1, 1, 1, 1]).tolist()
        assert val == [[1.16, 1.16, 1.16, 1.16]]
        val = R.execute([2, 2, 2, 2]).tolist()
        assert val == [[1.832, 1.832, 1.832, 1.832]]
        val = R.execute([-4, -3,0, 1]).tolist()
        assert val == [[-2.8336, -2.0336000000000003, .36639999999999995, 1.1663999999999999]]

    def test_recurrent_mech_time_constant_0_8_initial_1_2(self):
        R = RecurrentTransferMechanism(
            name='R',
            default_variable=[ 0, 0, 0, 0],
            function=Linear(),
            time_constant=0.8,
            initial_value=np.array([[-1, 1, -2, 2]]),
         time_scale=TimeScale.TIME_STEP
        )
        val = R.execute([3, 2,1, 0]).tolist()
        assert val == [[2.2,1.8, .40000000000000013, .3999999999999999]]

# (7/28/17 CW): the below are used because it's good to test System and Process anyways, and because the recurrent
# projection won't get executed if we only use the execute() method of Mechanism: thus, to test it we must use a System

def run_twice_in_system(mech, input1, input2=None):
    if input2 is None:
        input2 = input1
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}
    simple_process = process(size = mech.size[0], pathway = [mech], name = 'simple_process')
    simple_system = system(processes = [simple_process], name='simple_system', prefs = simple_prefs)

    first_output = simple_system.run(inputs = {mech: input1})
    second_output = simple_system.run(inputs = {mech: input2})
    return second_output[ 1][0].tolist()

class TestRecurrentTransferMechanismInProcess:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_recurrent_mech_transfer_mech_process_three_runs(self):
        # this test ASSUMES that the parameter state for auto and hetero is updated one run-cycle AFTER they are set by
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
        p = process(size=4, pathway=[R, T], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        p.run(inputs=[[[1, 2, 3, 4]]])
        assert (R.value.tolist()== [[1., 2., 3., 4.]])
        assert (T.value.tolist() == [[10., 10., 10.]])
        p.run(inputs=[[[5, 6, 7, 8]]])
        assert (R.value.tolist() == [[-4, -2,0, 2]])
        assert (T.value.tolist() == [[-4, -4, -4]])
        p.run(inputs=[[[-1, 2, -2, 5.5]]])
        assert (R.value.tolist() == [[-1.0, 4.0, 2.0, 11.5]])
        assert (T.value.tolist() == [[16.5, 16.5, 16.5]])

    def test_recurrent_mech_process_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        p = process(size=4, pathway=[T, R], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        R.matrix = [[2, 0, 1, 3]] * 4
        p.run(inputs = [[[1, 2, 3, 4]]])
        assert(T.value.tolist() == [[1, 2, 3, 4]])
        assert(R.value.tolist() == [[1, 2, 3, 4]])
        p.run(inputs = {T: [1, 3, 2, 5]})
        assert(R.recurrent_projection.matrix.tolist() == [[2, 0, 1, 3]] * 4)
        assert(T.value.tolist() == [[1, 3, 2, 5]])
        assert(R.value.tolist() == [[21, 3, 12, 35]])

    # this test must wait until we create a property such that R.recurrent_projection.matrix sets R.auto and R.hetero
    def test_recurrent_mech_process_proj_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        p = process(size=4, pathway=[T, R], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        R.recurrent_projection.matrix = [[2, 0, 1, 3]] * 4
        p.run(inputs = [[[1, 2, 3, 4]]])
        assert(T.value.tolist() == [[1, 2, 3, 4]])
        assert(R.value.tolist() == [[1, 2, 3, 4]])
        p.run(inputs = {T: [1, 3, 2, 5]})
        assert(R.recurrent_projection.matrix.tolist() == [[2, 0, 1, 3]] * 4)
        assert(T.value.tolist() == [[1, 3, 2, 5]])
        assert(R.value.tolist() == [[21, 3, 12, 35]])



class TestRecurrentTransferMechanismInSystem:
    simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}

    def test_recurrent_mech_transfer_mech_system_three_runs(self):
        # this test ASSUMES that the parameter state for auto and hetero is updated one run-cycle AFTER they are set by
        # lines by `R.auto = 0`. If this (potentially buggy) behavior is changed, then change these values
        R = RecurrentTransferMechanism(
            size=4,
            auto=0,
            hetero=-1)
        T = TransferMechanism(
            size = 3,
            function = Linear)
        p = process(size = 4, pathway = [R, T], prefs = TestRecurrentTransferMechanismInSystem.simple_prefs)
        s = system(processes = [p], prefs = TestRecurrentTransferMechanismInSystem.simple_prefs)
        s.run(inputs = {R: [1, 2, 3, 4]})
        assert(R.value.tolist() == [[1., 2., 3., 4.]])
        assert(T.value.tolist() == [[10., 10., 10.]])
        s.run(inputs = {R: [5, 6, 7, 8]})
        assert(R.value.tolist() == [[-4, -2, 0, 2]])
        assert(T.value.tolist() == [[-4, -4, -4]])
        s.run(inputs={R: [-1, 2, -2, 5.5]})
        assert (R.value.tolist() == [[-1.0, 4.0, 2.0, 11.5]])
        assert (T.value.tolist() == [[16.5, 16.5, 16.5]])

    def test_recurrent_mech_system_auto_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=-1)
        T = TransferMechanism(
            size=3,
            function=Linear)
        p = process(size=4, pathway=[R, T], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s = system(processes=[p], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s.run(inputs={R: [1, 2, 3, 4]})
        assert (R.value.tolist() == [[1., 2., 3., 4.]])
        assert (T.value.tolist() == [[10., 10., 10.]])
        R.auto = 0
        s.run(inputs={R: [5, 6, 7, 8]})
        assert (R.value.tolist() == [[-4, -2, 0, 2]])
        assert (T.value.tolist() == [[-4, -4, -4]])
        R.recurrent_projection.auto = [1, 1, 2, 4]
        s.run(inputs={R: [12, 11, 10,9]})
        assert (R.value.tolist() == [[8, 11, 14, 23]])
        assert (T.value.tolist() == [[56, 56, 56]])

    def test_recurrent_mech_system_hetero_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=[[-1, -2, -3, -4]] * 4)
        T = TransferMechanism(
            size=5,
            function=Linear)
        p = process(size=4, pathway=[R, T], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s = system(processes=[p], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s.run(inputs={R: [1, 2, 3, -0.5]})
        assert (R.value.tolist() == [[1., 2., 3., -0.5]])
        assert (T.value.tolist() == [[5.5, 5.5, 5.5, 5.5, 5.5]])
        R.hetero = 0
        s.run(inputs={R: [-1.5, 0, 1, 2]})
        assert (R.value.tolist() == [[-.5, 4, 10, 0]])
        assert (T.value.tolist() == [[13.5, 13.5, 13.5, 13.5, 13.5]])
        R.hetero = [[-1, 2, 3, 1.5]] * 4
        s.run(inputs={R: [12, 11, 10,9]})
        assert (R.value.tolist() == [[-2.5, 38, 50.5, 29.25]])
        assert (T.value.tolist() == [[115.25, 115.25, 115.25, 115.25, 115.25]])

    def test_recurrent_mech_system_auto_and_hetero_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=[1, 2, 3, 4],
            hetero=[[-1, -2, -3, -4]] * 4)
        T = TransferMechanism(
            size=5,
            function=Linear)
        p = process(size=4, pathway=[R, T], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s = system(processes=[p], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s.run(inputs={R: [1, 2, 3, -0.5]})
        assert (R.value.tolist() == [[1., 2., 3., -0.5]])
        assert (T.value.tolist() == [[5.5, 5.5, 5.5, 5.5, 5.5]])
        R.hetero = 0
        s.run(inputs={R: [-1.5, 0, 1, 2]})
        assert (R.value.tolist() == [[-.5, 4, 10, 0]])
        assert (T.value.tolist() == [[13.5, 13.5, 13.5, 13.5, 13.5]])
        R.auto = [0, 0, 0, 0]
        s.run(inputs={R: [12, 11, 10,9]})
        assert (R.value.tolist() == [[12, 11, 10, 9]])
        assert (T.value.tolist() == [[42, 42, 42, 42, 42]])

    def test_recurrent_mech_system_matrix_change(self):
        R = RecurrentTransferMechanism(
            size=4,
            auto=1,
            hetero=-1)
        T = TransferMechanism(
            size=4,
            function=Linear)
        p = process(size=4, pathway=[T, R], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        s = system(processes=[p], prefs=TestRecurrentTransferMechanismInSystem.simple_prefs)
        R.matrix = [[2, 0, 1, 3]] * 4
        s.run(inputs = {T: [1, 2, 3, 4]})
        assert(T.value.tolist() == [[1, 2, 3, 4]])
        assert(R.value.tolist() == [[1, 2, 3, 4]])
        s.run(inputs = {T: [1, 3, 2, 5]})
        assert(R.recurrent_projection.matrix.tolist() == [[2, 0, 1, 3]] * 4)
        assert(T.value.tolist() == [[1, 3, 2, 5]])
        assert(R.value.tolist() == [[21, 3, 12, 35]])


# this doesn't work consistently due to EVC's issue with the scheduler

# class TestRecurrentTransferMechanismControl:
#     simple_prefs = {REPORT_OUTPUT_PREF: False, VERBOSE_PREF: False}
#     def test_recurrent_mech_EVC(self):
#         R = RecurrentTransferMechanism(
#             size=4,
#             auto=1,
#             hetero=-1)
#         T = TransferMechanism(
#             size=3,
#             function=Linear)
#         p = process(size=4, pathway=[R, T], prefs=TestRecurrentTransferMechanismControl.simple_prefs)
#         s = system(processes=[p], prefs=TestRecurrentTransferMechanismControl.simple_prefs, controller = EVCMechanism,
#            enable_controller = True, monitor_for_control = [T.output_state], control_signals=[('auto', R), ('hetero', R)])
#         s.run(inputs = {R: [1, 3, 2, 5]})
#         print('T.value: ', T.value)
#         assert(T.value.tolist() == [[-.09645391388158941, -.09645391388158941, -.09645391388158941]])
#         s.run(inputs = {})
