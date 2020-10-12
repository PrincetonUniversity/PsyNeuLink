import numpy as np
import pytest

from psyneulink.core import llvm as pnlvm
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

    @pytest.mark.benchmark(group="ProcessingMechanism[DefaultFunction]")
    @pytest.mark.parametrize("variable", [[1, 2, 3, 4],
                                          [1., 2., 3., 4.],
                                          np.asarray([1., 2., 3., 4.], dtype=np.int8),
                                          np.asarray([1., 2., 3., 4.], dtype=np.int16),
                                          np.asarray([1., 2., 3., 4.], dtype=np.int32),
                                          np.asarray([1., 2., 3., 4.], dtype=np.int64),
                                          np.asarray([1., 2., 3., 4.], dtype=np.float32),
                                          np.asarray([1., 2., 3., 4.], dtype=np.float64),
                                          [[1, 2, 3, 4]],
                                          [[1., 2., 3., 4.]],
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.int8),
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.int16),
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.int32),
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.int64),
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.float32),
                                          np.asarray([[1., 2., 3., 4.]], dtype=np.float64),
                                         ],
                             ids=["list.int", "list.float", "np.1d.i8", "np.1d.i16", "np.1d.i32", "np.1d.i64", "np.1d.f32", "np.1d.f64",
                                  "list2d.int", "list2d.float", "np.2d.i8", "np.2d.i16", "np.2d.i32", "np.2d.i64", "np.2d.f32", "np.2d.f64",
                             ])
    @pytest.mark.parametrize("mode", ["Python",
                                      pytest.param("LLVM", marks=[pytest.mark.llvm]),
                                      pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_processing_mechanism_default_function(self, mode, variable, benchmark):
        PM = ProcessingMechanism(default_variable=[0, 0, 0, 0])
        if mode == "Python":
            ex = PM.execute
        elif mode == "LLVM":
            ex = pnlvm.MechExecution(PM).execute
        elif mode == "PTX":
            ex = pnlvm.MechExecution(PM).cuda_execute
        res = benchmark(ex, variable)
        assert np.allclose(res, [[1., 2., 3., 4.]])

    def test_processing_mechanism_linear_function(self):

        PM1 = ProcessingMechanism()
        PM1.execute(1.0)
        assert np.allclose(PM1.value, 1.0)

        PM2 = ProcessingMechanism(function=Linear(slope=2.0,
                                                  intercept=1.0))
        PM2.execute(1.0)
        assert np.allclose(PM2.value, 3.0)

    @pytest.mark.parametrize("function,expected", [(LinearCombination, [[1.]]),
                                                   (Reduce, [[1.]]),
                                                   (CombineMeans, [1.0]),
                                                   (Exponential, [[2.71828183]]),
                                                   (Logistic, [[0.73105858]]),
                                                   (SoftMax, [[1,]]),
                                                   (SimpleIntegrator, [[1.]]),
                                                   (AdaptiveIntegrator, [[1.]]),
                                                   (DriftDiffusionIntegrator, [[[1.]], [[1.]]]),
                                                   (OrnsteinUhlenbeckIntegrator, [[[-1.]], [[1.]]]),
                                                   (AccumulatorIntegrator, [[0.]]),
                                                   (FitzHughNagumoIntegrator, [[[0.05127053]], [[0.00279552]], [[0.05]]]),
                                                   (DualAdaptiveIntegrator, [[0.1517455]]),
                                                   (DriftDiffusionAnalytical, [[1.19932930e+00],
                                                                               [3.35350130e-04],
                                                                               [1.19932930e+00],
                                                                               [2.48491374e-01],
                                                                               [1.48291009e+00],
                                                                               [1.19932930e+00],
                                                                               [2.48491374e-01],
                                                                               [1.48291009e+00]]),
                                                   (NormalDist, [[-0.51529709]]),
                                                   (ExponentialDist, [[0.29964231]]),
                                                   (UniformDist, [[0.25891675]]),
                                                   (GammaDist, [[0.29964231]]),
                                                   (WaldDist, [[0.73955962]]),
                                                  ],
                             ids=lambda x: getattr(x, "componentName", ""))
    def test_processing_mechanism_function(self, function, expected):
        PM = ProcessingMechanism(function=function)
        res = PM.execute(1.0)
        assert np.allclose(np.asfarray(res), expected)

    # COMMENTED OUT BECAUSE OF MATLAB ENGINE:
    # def test_processing_mechanism_NavarroAndFuss_function(self):
    #     PM1 = ProcessingMechanism(function=NavarroAndFuss)
    #     PM1.execute(1.0)
    #     # assert np.allclose(PM1.value, 1.0)

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

    @pytest.mark.benchmark
    @pytest.mark.parametrize("op, expected", [(MAX_ONE_HOT, [0, 2, 0]),
                                              (MAX_INDICATOR, [0, 1, 0]),
                                              (MAX_ABS_INDICATOR, [0, 0, 1]),
                                             ],
                             ids=lambda x: x if isinstance(x, str) else "")
    @pytest.mark.parametrize("mode", ["Python",
                                      pytest.param("LLVM", marks=[pytest.mark.llvm]),
                                      pytest.param("PTX", marks=[pytest.mark.llvm, pytest.mark.cuda]),
                                     ])
    def test_output_ports(self, mode, op, expected, benchmark):
        benchmark.group = "Output Port Op: {}".format(op)
        PM1 = ProcessingMechanism(default_variable=[0, 0, 0], output_ports=[op])
        var = [1, 2, 4] if op in {MEAN, MEDIAN, STANDARD_DEVIATION, VARIANCE} else [1, 2, -4]
        if mode == "Python":
            ex = PM1.execute
        elif mode == "LLVM":
            ex = pnlvm.MechExecution(PM1).execute
        elif mode == "PTX":
            ex = pnlvm.MechExecution(PM1).cuda_execute
        res = benchmark(ex, var)
        res = PM1.output_ports[0].value if mode == "Python" else res
        assert np.allclose(res, expected)

    # FIXME: These variants don't compile (use UDFs)
    @pytest.mark.parametrize("op, expected", [(MEAN, [2.33333333]),
                                              (MEDIAN, [2]),
                                              (STANDARD_DEVIATION, [1.24721913]),
                                              (VARIANCE, [1.55555556]),
                                              (MAX_VAL, [2]),
                                              (MAX_ABS_VAL, [4]),
                                              (MAX_ABS_ONE_HOT, [0, 0, 4]),
                                              (PROB, [0, 2, 0]),
                                             ],
                             ids=lambda x: x if isinstance(x, str) else "")
    def test_output_ports2(self, op, expected):
        PM1 = ProcessingMechanism(default_variable=[0, 0, 0], output_ports=[op])
        var = [1, 2, 4] if op in {MEAN, MEDIAN, STANDARD_DEVIATION, VARIANCE} else [1, 2, -4]
        PM1.execute(var)
        assert np.allclose(PM1.output_ports[0].value, expected)
