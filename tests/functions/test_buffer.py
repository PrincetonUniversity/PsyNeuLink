from psyneulink.components.functions.function import Buffer
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.process import Process
from psyneulink.components.system import System
from collections import deque
import numpy as np

class TestBuffer():

    def test_buffer_standalone(self):
        B = Buffer()
        val = B.execute(1.0)
        assert np.allclose(deque(np.atleast_1d(1.0)), val)

    def test_buffer_initializer_len_3(self):
        B = Buffer(default_variable=[[0.0], [1.0], [2.0]],
                   initializer=[[0.0], [1.0], [2.0]],
                   history=3)
        assert np.allclose(B.execute(3.0), deque([[1.0], [2.0], np.array([3.])]))
        assert np.allclose(B.execute(4.0), deque([[2.0], np.array([3.]), np.array([4.])]))
        assert np.allclose(B.execute(5.0), deque([np.array([3.]), np.array([4.]), np.array([5.])]))


    def test_buffer_as_function_of_processing_mech(self):
        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0], [1.0], [2.0]],
                                                initializer=[[0.0], [1.0], [2.0]],
                                                history=3))
        P.execute(1.0)

        process = Process(pathway=[P])
        system = System(processes=[process])
        system.run(inputs={[1.0]})

    def test_buffer_as_function_of_origin_mech_in_system(self):
        P = ProcessingMechanism(function=Buffer(default_variable=[[0.0]],
                                initializer=[[0.0]],
                                history=3))

        process = Process(pathway=[P])
        system = System(processes=[process])
        print(system.run(inputs={P: [1.0]},
                         num_trials=5),
              )
