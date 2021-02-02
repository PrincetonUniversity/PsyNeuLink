import numpy as np
import pytest
from collections import deque

from psyneulink.core.compositions.composition import Composition
from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import Buffer
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.scheduling.condition import Never

class TestBuffer():

    def test_buffer_standalone(self):
        B = Buffer()
        val = B.execute(1.0)
        assert np.allclose(deque(np.atleast_1d(1.0)), val)

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_rate_float(self, benchmark):
        B = Buffer(history=3, rate = 0.1)
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04,  0.05,  0.06], [ 0.7,  0.8,  0.9], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_rate_list(self, benchmark):
        B = Buffer(history=3, rate = [0.1, 0.5, 0.9])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04, 1.25, 4.86], [ 0.7,  4. , 8.1], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_rate_ndarray(self, benchmark):
        B = Buffer(history=3, rate = np.array([0.1, 0.5, 0.9]))
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04, 1.25, 4.86], [ 0.7,  4. , 8.1], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_noise_float(self, benchmark):
        B = Buffer(history=3, rate = 1.0, noise=10.0)
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24.,  25.,  26.], [ 17.,  18.,  19.], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_noise_list(self, benchmark):
        B = Buffer(history=3, rate = 1.0, noise=[10.0, 20.0, 30.0])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24., 45., 66.], [ 17., 28., 39.], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_noise_ndarray(self, benchmark):
        B = Buffer(history=3, rate = 1.0, noise=[10.0, 20.0, 30.0])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24., 45., 66.], [ 17., 28., 39.], [10, 11, 12])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_noise_function(self, benchmark):
        B = Buffer(history=3, rate = 1.0, noise=NormalDist(standard_deviation=0.1))
        B.execute([1, 2, 3])
        B.execute([4, 5, 6])
        B.execute([7, 8, 9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([[4.02430687, 4.91927251, 5.95087965],
                                                [7.09586966, 7.91823773, 8.86077491],
                                                [10, 11, 12]])), val)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_standalone_noise_function_in_array(self, benchmark):
        B = Buffer(history=3)
        # Set noise parameter ouside of a constructor to avoid problems
        # with extra copying
        B.parameters.noise.set([10, NormalDist(standard_deviation=0.1), 20])
        B.execute([1, 2, 3])
        B.execute([4, 5, 6])
        B.execute([7, 8, 9])
        val = B.execute([10, 11, 12])
        expected_val = [[24, 4.693117564500052, 46], [17, 7.744647273059847, 29], [10, 11, 12]]
        for v_v, v_e in zip(val, expected_val):
            for v, e in zip(v_v, v_e):
                assert np.allclose(v, e)
        if benchmark.enabled:
            benchmark(B.execute, [1, 2, 3])

    def test_buffer_standalone_noise_function_invocation(self):
        class CallCount:
            def __init__(self):
                self.count = 0
            def __call__(self):
                self.count += 1
                return self.count

        counter_f = CallCount()
        # Set noise parameter ouside of a constructor to avoid problems
        # with extra copying. This test fails if noise is passed to constructor
        B = Buffer(history=3)
        B.parameters.noise.set([10, counter_f, 20])
        B.execute([1, 2, 3])
        B.execute([4, 5, 6])
        B.execute([7, 8, 9])
        val = B.execute([10, 11, 12])

        assert counter_f.count == 4
        expected_val = [[24, 12.0, 46], [17, 12.0, 29], [10, 11, 12]]
        for v_v, v_e in zip(val, expected_val):
            for v, e in zip(v_v, v_e):
                assert np.allclose(v, e)

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_initializer_len_3(self, benchmark):
        B = Buffer(default_variable=[[0.0], [1.0], [2.0]],
                   initializer=[[0.0], [1.0], [2.0]],
                   history=3)
        assert np.allclose(B.execute(3.0), deque([[1.0], [2.0], np.array([3.])]))
        assert np.allclose(B.execute(4.0), deque([[2.0], np.array([3.]), np.array([4.])]))
        assert np.allclose(B.execute(5.0), deque([np.array([3.]), np.array([4.]), np.array([5.])]))
        if benchmark.enabled:
            benchmark(B.execute, 5.0)

    @pytest.mark.benchmark(group="BufferFunction")
    def test_buffer_as_function_of_processing_mech(self, benchmark):

        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0]],
                                                initializer=[0.0],
                                                history=3))
        val = P.execute(1.0)

        # NOTE: actual output is [0, [[1]]]
        assert np.allclose(np.asfarray(val), [[0., 1.]])
        if benchmark.enabled:
            benchmark(P.execute, 5.0)
        # fails due to value and variable problems when Buffer is the function of a mechanism
        # P = ProcessingMechanism(function=Buffer(default_variable=[[0.0], [1.0], [2.0]],
        #                                         initializer=[[0.0], [1.0], [2.0]],
        #                                         history=3))
        # P.execute(1.0)


    def test_buffer_as_function_of_origin_mech_in_composition(self):
        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0]],
                                initializer=[[0.0]],
                                history=3))

        C = Composition(pathways=[P])
        P.reset_stateful_function_when = Never()
        full_result = []

        def assemble_full_result():
            full_result.append(P.parameters.value.get(C))

        C.run(inputs={P: [[1.0], [2.0], [3.0], [4.0], [5.0]]},
              call_after_trial=assemble_full_result)
        # only returns index 0 item of the deque on each trial  (OutputPort value)
        assert np.allclose(np.asfarray(C.results), [[[0.0]], [[0.0]], [[1.0]], [[2.0]], [[3.0]]])

        # stores full mechanism value (full deque) on each trial
        expected_full_result = [np.array([[0.], [1.]]),
                                np.array([[0.], [1.], [2.]]),
                                np.array([[1.], [2.], [3.]]),   # Shape change
                                np.array([[2.], [3.], [4.]]),
                                np.array([[3.], [4.], [5.]])]
        for i in range(5):
            assert np.allclose(expected_full_result[i],
                               np.asfarray(full_result[i]))
