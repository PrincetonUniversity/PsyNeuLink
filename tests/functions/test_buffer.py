import numpy as np

from collections import deque
from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.statefulfunctions.memoryfunctions import Buffer
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.process import Process
from psyneulink.core.components.system import System
from psyneulink.core.scheduling.condition import Never

class TestBuffer():

    def test_buffer_standalone(self):
        B = Buffer()
        val = B.execute(1.0)
        assert np.allclose(deque(np.atleast_1d(1.0)), val)

    def test_buffer_standalone_rate_float(self):
        B = Buffer(history=3, rate = 0.1)
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04,  0.05,  0.06], [ 0.7,  0.8,  0.9], [10, 11, 12])), val)

    def test_buffer_standalone_rate_list(self):
        B = Buffer(history=3, rate = [0.1, 0.5, 0.9])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04, 1.25, 4.86], [ 0.7,  4. , 8.1], [10, 11, 12])), val)

    def test_buffer_standalone_rate_ndarray(self):
        B = Buffer(history=3, rate = np.array([0.1, 0.5, 0.9]))
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 0.04, 1.25, 4.86], [ 0.7,  4. , 8.1], [10, 11, 12])), val)

    def test_buffer_standalone_noise_float(self):
        B = Buffer(history=3, rate = 1.0, noise=10.0)
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24.,  25.,  26.], [ 17.,  18.,  19.], [10, 11, 12])), val)

    def test_buffer_standalone_noise_list(self):
        B = Buffer(history=3, rate = 1.0, noise=[10.0, 20.0, 30.0])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24., 45., 66.], [ 17., 28., 39.], [10, 11, 12])), val)

    def test_buffer_standalone_noise_ndarray(self):
        B = Buffer(history=3, rate = 1.0, noise=[10.0, 20.0, 30.0])
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([ 24., 45., 66.], [ 17., 28., 39.], [10, 11, 12])), val)

    def test_buffer_standalone_noise_function(self):
        np.random.seed(22)
        B = Buffer(history=3, rate = 1.0, noise=NormalDist(standard_deviation=0.1))
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        assert np.allclose(deque(np.atleast_1d([[ 3.8925223, 5.03957263, 6.00262384],
                                                [ 7.00288551, 7.97692328, 9.05877522],
                                                [10, 11, 12]])), val)

    def test_buffer_standalone_noise_function_in_array(self):
        B = Buffer(history=3, noise=[10, NormalDist(standard_deviation=0.1), 20])
        np.random.seed(22)
        B.execute([1,2,3])
        B.execute([4,5,6])
        B.execute([7,8,9])
        val = B.execute([10,11,12])
        expected_val = [[24, 5.0800314416734444, 46], [17, 8.040015720836722, 29], [10, 11, 12]]
        for i in range(len(val)):
            for j in range(len(val[i])):
                assert np.allclose(expected_val[i][j], val[i][j])

    def test_buffer_initializer_len_3(self):
        B = Buffer(default_variable=[[0.0], [1.0], [2.0]],
                   initializer=[[0.0], [1.0], [2.0]],
                   history=3)
        assert np.allclose(B.execute(3.0), deque([[1.0], [2.0], np.array([3.])]))
        assert np.allclose(B.execute(4.0), deque([[2.0], np.array([3.]), np.array([4.])]))
        assert np.allclose(B.execute(5.0), deque([np.array([3.]), np.array([4.]), np.array([5.])]))

    def test_buffer_as_function_of_processing_mech(self):

        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0]],
                                                initializer=[0.0],
                                                history=3))
        val = P.execute(1.0)

        assert np.allclose(val, [[0., 1.]])
        # fails due to value and variable problems when Buffer is the function of a mechanism
        # P = ProcessingMechanism(function=Buffer(default_variable=[[0.0], [1.0], [2.0]],
        #                                         initializer=[[0.0], [1.0], [2.0]],
        #                                         history=3))
        # P.execute(1.0)


    def test_buffer_as_function_of_origin_mech_in_system(self):
        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0]],
                                initializer=[[0.0]],
                                history=3))

        process = Process(pathway=[P])
        system = System(processes=[process])
        P.reinitialize_when = Never()
        full_result = []

        def assemble_full_result():
            full_result.append(P.parameters.value.get(system))

        result = system.run(inputs={P: [[1.0], [2.0], [3.0], [4.0], [5.0]]},
                            call_after_trial=assemble_full_result)
        # only returns index 0 item of the deque on each trial  (output state value)
        assert np.allclose(result, [[[0.0]], [[0.0]], [[1.0]], [[2.0]], [[3.0]]])

        # stores full mechanism value (full deque) on each trial
        expected_full_result = [np.array([[0.], [1.]]),
                                np.array([[0.], [1.], [2.]]),
                                np.array([[1.], [2.], [3.]]),   # Shape change
                                np.array([[2.], [3.], [4.]]),
                                np.array([[3.], [4.], [5.]])]
        for i in range(5):
            assert np.allclose(expected_full_result[i], full_result[i])

