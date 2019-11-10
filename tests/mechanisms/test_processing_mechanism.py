import numpy as np
import pytest

from psyneulink.core.components.functions.function import FunctionError
from psyneulink.core.components.functions.learningfunctions import Hebbian, Reinforcement, TDLearning
from psyneulink.core.components.functions.objectivefunctions import Stability, Distance
from psyneulink.core.components.functions.distributionfunctions import NormalDist, ExponentialDist, \
    UniformDist, GammaDist, WaldDist, DriftDiffusionAnalytical
from psyneulink.core.components.functions.statefulfunctions.integratorfunctions import SimpleIntegrator, \
    AdaptiveIntegrator, DriftDiffusionIntegrator, OrnsteinUhlenbeckIntegrator, FitzHughNagumoIntegrator, \
    AccumulatorIntegrator, DualAdaptiveIntegrator
from psyneulink.core.components.functions.transferfunctions import Linear, Exponential, Logistic, SoftMax, LinearMatrix
from psyneulink.core.components.functions.combinationfunctions import Reduce, LinearCombination, CombineMeans
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.core.globals.keywords import \
    MAX_ABS_INDICATOR, MAX_ABS_ONE_HOT, MAX_ABS_VAL, MAX_INDICATOR, MAX_ONE_HOT, MAX_VAL, \
    MEAN, MEDIAN, PROB, STANDARD_DEVIATION, VARIANCE

class TestProcessingMechanismFunctions:

    def test_processing_mechanism_linear_function(self):

        PM1 = ProcessingMechanism()
        PM1.execute(1.0)
        assert np.allclose(PM1.value, 1.0)

        PM2 = ProcessingMechanism(function=Linear(slope=2.0,
                                                  intercept=1.0))
        PM2.execute(1.0)
        assert np.allclose(PM2.value, 3.0)

    def test_processing_mechanism_LinearCombination_function(self):

        PM1 = ProcessingMechanism(function=LinearCombination)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Reduce_function(self):
        PM1 = ProcessingMechanism(function=Reduce)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_CombineMeans_function(self):
        PM1 = ProcessingMechanism(function=CombineMeans)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Exponential_function(self):
        PM1 = ProcessingMechanism(function=Exponential)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Logistic_function(self):
        PM1 = ProcessingMechanism(function=Logistic)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_SoftMax_function(self):
        PM1 = ProcessingMechanism(function=SoftMax(per_item=False))
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_SimpleIntegrator_function(self):
        PM1 = ProcessingMechanism(function=SimpleIntegrator)
        PM1.execute(1.0)

    def test_processing_mechanism_AdaptiveIntegrator_function(self):
        PM1 = ProcessingMechanism(function=AdaptiveIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_DriftDiffusionIntegrator_function(self):
        PM1 = ProcessingMechanism(function=DriftDiffusionIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_OrnsteinUhlenbeckIntegrator_function(self):
        PM1 = ProcessingMechanism(function=OrnsteinUhlenbeckIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_AccumulatorIntegrator_function(self):
        PM1 = ProcessingMechanism(function=AccumulatorIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_FitzHughNagumoIntegrator_function(self):
        PM1 = ProcessingMechanism(function=FitzHughNagumoIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_DualAdaptiveIntegrator_function(self):
        PM1 = ProcessingMechanism(function=DualAdaptiveIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_BogaczEtAl_function(self):
        PM1 = ProcessingMechanism(function=DriftDiffusionAnalytical)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    # COMMENTED OUT BECAUSE OF MATLAB ENGINE:
    # def test_processing_mechanism_NavarroAndFuss_function(self):
    #     PM1 = ProcessingMechanism(function=NavarroAndFuss)
    #     PM1.execute(1.0)
    #     # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_NormalDist_function(self):
        PM1 = ProcessingMechanism(function=NormalDist)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_ExponentialDist_function(self):
        PM1 = ProcessingMechanism(function=ExponentialDist)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_UniformDist_function(self):
        PM1 = ProcessingMechanism(function=UniformDist)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_GammaDist_function(self):
        PM1 = ProcessingMechanism(function=GammaDist)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_WaldDist_function(self):
        PM1 = ProcessingMechanism(function=WaldDist)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Stability_function(self):
        PM1 = ProcessingMechanism(function=Stability)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Distance_function(self):
        PM1 = ProcessingMechanism(function=Distance,
                                  default_variable=[[0,0], [0,0]])
        PM1.execute([[1, 2], [3, 4]])
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Hebbian_function(self):
        PM1 = ProcessingMechanism(function=Hebbian,
                                  default_variable=[[0.0], [0.0], [0.0]])
        PM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_Reinforcement_function(self):
        PM1 = ProcessingMechanism(function=Reinforcement,
                                  default_variable=[[0.0], [0.0], [0.0]])
        PM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(PM1.value, 1.0)

    # COMMENTING OUT BECAUSE BACK PROP FN DOES NOT WORK WITH UNRESTRICTED MECHANISM
    # def test_processing_mechanism_BackPropagation_function(self):
    #     PM1 = ProcessingMechanism(function=BackPropagation,
    #                                          default_variable=[[0.0], [0.0], [0.0]])
    #     PM1.execute([[1.0], [2.0], [3.0]])
    #     PM1.execute(1.0)
    #     # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_TDLearning_function(self):
        PM1 = ProcessingMechanism(function=TDLearning,
                                  default_variable=[[0.0], [0.0], [0.0]])
        PM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_multiple_input_ports(self):
        PM1 = ProcessingMechanism(size=[4, 4], function=LinearCombination, input_ports=['input_1', 'input_2'])
        PM2 = ProcessingMechanism(size=[2, 2, 2], function=LinearCombination, input_ports=['1', '2', '3'])
        PM1.execute([[1, 2, 3, 4], [5, 4, 2, 2]])
        PM2.execute([[2, 0], [1, 3], [1, 0]])
        assert np.allclose(PM1.value, [6, 6, 5, 6])
        assert np.allclose(PM2.value, [4, 3])

class TestLinearMatrixFunction:

    def test_valid_matrix_specs(self):
        # Note: default matrix specification is None

        PM_default = ProcessingMechanism(function=LinearMatrix())
        PM_default.execute(1.0)

        assert np.allclose(PM_default.value, 1.0)

        PM_default_len_2_var = ProcessingMechanism(function=LinearMatrix(default_variable=[[0.0, 0.0]]),
                                                   default_variable=[[0.0, 0.0]])
        PM_default_len_2_var.execute([[1.0, 2.0]])

        assert np.allclose(PM_default_len_2_var.value, [[1.0, 2.0]])

        PM_default_2d_var = ProcessingMechanism(function=LinearMatrix(default_variable=[[0.0, 0.0],
                                                                                        [0.0, 0.0],
                                                                                        [0.0, 0.0]]),
                                                   default_variable=[[0.0, 0.0],
                                                                     [0.0, 0.0],
                                                                     [0.0, 0.0]])

        #  [1.0   0.0]           [1.0   0.0]
        #  [0.0   2.0]    *      [0.0   1.0]
        #  [3.0   0.0]
        PM_default_2d_var.execute([[1.0, 0.0],
                                   [0.0, 2.0],
                                   [3.0, 0.0]])

        assert np.allclose(PM_default_2d_var.value, [[1.0, 0.0],
                                                     [0.0, 2.0],
                                                     [3.0, 0.0]])

        # PM_float = ProcessingMechanism(function=LinearMatrix(matrix=4.0))
        # PM_float.execute(1.0)
        #
        # assert np.allclose(PM_float.value, 4.0)

        PM_1d_list = ProcessingMechanism(function=LinearMatrix(matrix=[4.0]))
        PM_1d_list.execute(1.0)

        assert np.allclose(PM_1d_list.value, 4.0)

        PM_2d_list = ProcessingMechanism(function=LinearMatrix(matrix=[[4.0, 5.0],
                                                                       [6.0, 7.0],
                                                                       [8.0, 9.0],
                                                                       [10.0, 11.0]],
                                                               default_variable=[[0.0, 0.0, 0.0, 0.0],
                                                                                 [0.0, 0.0, 0.0, 0.0],
                                                                                 [0.0, 0.0, 0.0, 0.0]]),
                                         default_variable=[[0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0, 0.0]])
        PM_2d_list.execute([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        assert np.allclose(PM_2d_list.value, [[4.0, 5.0],
                                              [8.0, 9.0],
                                              [10.0, 11.0]])

        PM_1d_array = ProcessingMechanism(function=LinearMatrix(matrix=np.array([4.0])))
        PM_1d_array.execute(1.0)

        assert np.allclose(PM_1d_array.value, 4.0)

        PM_2d_array = ProcessingMechanism(function=LinearMatrix(matrix=np.array([[4.0]])))
        PM_2d_array.execute(1.0)

        assert np.allclose(PM_2d_array.value, 4.0)

        PM_matrix = ProcessingMechanism(function=LinearMatrix(matrix=np.matrix([[4.0]])))
        PM_matrix.execute(1.0)

        assert np.allclose(PM_matrix.value, 4.0)

    def test_invalid_matrix_specs(self):

        with pytest.raises(FunctionError) as error_text:
            PM_mismatched_float = ProcessingMechanism(function=LinearMatrix(default_variable=0.0,
                                                                        matrix=[[1.0, 0.0, 0.0, 0.0],
                                                                                [0.0, 2.0, 0.0, 0.0],
                                                                                [0.0, 0.0, 3.0, 0.0],
                                                                                [0.0, 0.0, 0.0, 4.0]]),
                                                  default_variable=0.0)
        assert "Specification of matrix and/or default_variable" in str(error_text.value) and \
               "not compatible for multiplication" in str(error_text.value)

        with pytest.raises(FunctionError) as error_text:
            PM_mismatched_matrix = ProcessingMechanism(function=LinearMatrix(default_variable=[[0.0, 0.0],
                                                                                          [0.0, 0.0],
                                                                                          [0.0, 0.0]],
                                                                        matrix=[[1.0, 0.0, 0.0, 0.0],
                                                                                [0.0, 2.0, 0.0, 0.0],
                                                                                [0.0, 0.0, 3.0, 0.0],
                                                                                [0.0, 0.0, 0.0, 4.0]]),
                                                  default_variable=[[0.0, 0.0],
                                                                    [0.0, 0.0],
                                                                    [0.0, 0.0]])
        assert "Specification of matrix and/or default_variable" in str(error_text.value) and \
               "not compatible for multiplication" in str(error_text.value)

class TestProcessingMechanismStandardOutputPorts:

    def test_mean(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MEAN])
        PM1.execute([1,2,4])
        assert np.allclose(PM1.output_ports[0].value,[2.33333333])

    def test_median(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MEDIAN])
        PM1.execute([1,2,4])
        assert np.allclose(PM1.output_ports[0].value,[2])

    def test_std_dev(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[STANDARD_DEVIATION])
        PM1.execute([1,2,4])
        assert np.allclose(PM1.output_ports[0].value,[1.24721913])

    def test_variance(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[VARIANCE])
        PM1.execute([1,2,4])
        assert np.allclose(PM1.output_ports[0].value,[1.55555556])

    def test_max_val(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_VAL])
        PM1.execute([1,2,-4])
        # assert np.allclose(PM1.output_ports[0].value,[0,2,0])
        assert np.allclose(PM1.output_ports[0].value,[2])

    def test_max_abs_val(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_ABS_VAL])
        PM1.execute([1,2,-4])
        # assert np.allclose(PM1.output_ports[0].value,[0,0,-4])
        assert np.allclose(PM1.output_ports[0].value,[4])

    def test_max_one_hot(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_ONE_HOT])
        PM1.execute([1,2,-4])
        assert np.allclose(PM1.output_ports[0].value,[0,2,0])

    def test_max_abs_one_hot(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_ABS_ONE_HOT])
        PM1.execute([1,2,-4])
        assert np.allclose(PM1.output_ports[0].value,[0,0,4])

    def test_max_indicator(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_INDICATOR])
        PM1.execute([1,2,-4])
        assert np.allclose(PM1.output_ports[0].value,[0,1,0])

    def test_max_abs_indicator(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[MAX_ABS_INDICATOR])
        PM1.execute([1,2,-4])
        assert np.allclose(PM1.output_ports[0].value,[0,0,1])

    def test_prob(self):
        PM1 = ProcessingMechanism(default_variable=[0,0,0], output_ports=[PROB])
        PM1.execute([1,2,4])
        assert np.allclose(PM1.output_ports[0].value,[0,0,4])
