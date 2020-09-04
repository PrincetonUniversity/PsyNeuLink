import numpy as np

from psyneulink.core.components.functions.distributionfunctions import NormalDist
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import AccumulatorIntegrator
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.globals.keywords import MATRIX


class TestAccumulator():

    def test_accumulator_standalone(self):
        A = AccumulatorIntegrator()
        val = A()
        assert np.allclose([[0]], val)

    def test_accumulator_initializer_len_3(self):
        A = AccumulatorIntegrator(default_variable=[[0.0], [1.0], [2.0]],
                                  initializer=[[0.0], [1.0], [2.0]],
                                  increment=1)
        assert np.allclose(A(), [[1],[2],[3]])
        assert np.allclose(A(), [[2],[3],[4]])
        assert np.allclose(A(), [[3],[4],[5]])

    # RATE ------------------------------------------------------------

    def test_accumulator_standalone_rate_float(self):
        A = AccumulatorIntegrator(rate = 0.1, increment=1)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[1.111]], val)

    def test_accumulator_standalone_rate_list(self):
        A = AccumulatorIntegrator(rate = [0.1, 0.5, 0.9],
                                  increment=1)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[1.111, 1.875, 3.439]], val)

    def test_accumulator_standalone_rate_ndarray(self):
        A = AccumulatorIntegrator(rate = [0.1, 0.5, 0.9],
                                  increment=1)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[1.111, 1.875, 3.439]], val)

    # INCREMENT ------------------------------------------------------------

    def test_accumulator_standalone_increment_float(self):
        A = AccumulatorIntegrator(increment = 0.1)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[0.4]], val)

    def test_accumulator_standalone_increment_list(self):
        A = AccumulatorIntegrator(increment = [0.1, 0.5, 0.9])
        A()
        A()
        A()
        val = A()
        assert np.allclose([[0.4, 2., 3.6]], val)

    def test_accumulator_standalone_increment_ndarray(self):
        A = AccumulatorIntegrator(increment = np.array([0.1, 0.5, 0.9]))
        A()
        A()
        A()
        val = A()
        assert np.allclose([[0.4, 2., 3.6]], val)

    # RATE X INCRMEMNT -------------------------------------------------

    def test_accumulator_standalone_rate_float_incement_float(self):
        A = AccumulatorIntegrator(
                rate=2.0,
                increment = 3.0)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[45]], val)

    def test_accumulator_standalone_rate_float_incement_list(self):
        A = AccumulatorIntegrator(
                rate=2.0,
                increment = [3.5, 4.0, 5.0])
        A()
        A()
        A()
        val = A()
        assert np.allclose([[52.5, 60., 75.]], val)

    def test_accumulator_standalone_rate_list_incement_float(self):
        A = AccumulatorIntegrator(
                rate=[2.0, 2.5, -3.0],
                increment = -1.4)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[-21., -35.525, 28.]], val)

    def test_accumulator_standalone_rate_list_incement_list(self):
        A = AccumulatorIntegrator(
                rate=[-.5, 2.0, -3.5],
                increment = [-1.0, .5, -2.0])
        A()
        A()
        A()
        val = A()
        assert np.allclose([[-0.625, 7.5, 66.25]], val)

    # NOISE ------------------------------------------------------------

    def test_accumulator_standalone_noise_float(self):
        A = AccumulatorIntegrator(rate = 1.0, noise=10.0)
        A()
        A()
        A()
        val = A()
        assert np.allclose([[40]], val)

    def test_accumulator_standalone_noise_list(self):
        A = AccumulatorIntegrator(rate = 1.0, noise=[10.0, 20.0, 30.0])
        A()
        A()
        A()
        val = A()
        assert np.allclose([[40., 80., 120.]], val)

    def test_accumulator_standalone_noise_ndarray(self):
        A = AccumulatorIntegrator(rate = 1.0, noise=[10.0, 20.0, 30.0])
        A()
        A()
        A()
        val = A()
        assert np.allclose([[40., 80., 120.]], val)

    def test_accumulator_standalone_noise_function(self):
        A = AccumulatorIntegrator(rate = 1.0, noise=NormalDist(standard_deviation=0.1))
        A()
        A()
        A()
        val = A()
        assert np.allclose([[-0.34591555]], val)

    def test_accumulator_standalone_noise_function_in_array(self):
        A = AccumulatorIntegrator(noise=[10, NormalDist(standard_deviation=0.1), 20])
        A()
        A()
        A()
        val = A()
        expected_val = [[40.0, 0.2480800486427607, 80.0]]
        for i in range(len(val)):
            for j in range(len(val[i])):
                assert np.allclose(expected_val[i][j], val[i][j])

    def test_accumulator_as_function_of_processing_mech(self):

        P = ProcessingMechanism(function=AccumulatorIntegrator(initializer=[0.0],
                                                               rate=.02,
                                                               increment=1))
        P.execute(1.0)
        P.execute(1.0)
        val = P.execute(1.0)
        assert np.allclose(val, [[1.0204]])

    def test_accumulator_as_function_of_matrix_param_of_mapping_projection(self):
        # Test that accumulator is function of parameter_port of mapping project,
        # and that its increment param works properly (used as modulatory param by LearningProjetion)

        T1 = TransferMechanism(size=3)
        T2 = TransferMechanism(size=3)
        M = MappingProjection(sender=T1, receiver=T2)
        C = Composition()
        C.add_linear_processing_pathway([T1, M, T2])
        C.run(inputs={T1: [1.0, 1.0, 1.0]})
        assert np.allclose(M.matrix.base, [[ 1.,  0.,  0.], [ 0.,  1.,  0.],[ 0.,  0.,  1.]])
        M.parameter_ports[MATRIX].function.parameters.increment.set(2, C)
        C.run(inputs={T1: [1.0, 1.0, 1.0]})
        assert np.allclose(M.matrix.base, [[ 3.,  2.,  2.], [ 2.,  3.,  2.], [ 2.,  2.,  3.]])
        C.run(inputs={T1: [1.0, 1.0, 1.0]})
        assert np.allclose(M.matrix.base, [[ 5.,  4.,  4.], [ 4.,  5.,  4.], [ 4.,  4.,  5.]])
