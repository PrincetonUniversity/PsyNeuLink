import numpy as np
import pytest
from psyneulink.components.mechanisms.processing.processingmechanism import ProcessingMechanism
from psyneulink.components.functions.function import Reduce, LinearCombination, CombineMeans, Linear, Exponential, Logistic, SoftMax, LinearMatrix, Integrator, SimpleIntegrator, ConstantIntegrator, AdaptiveIntegrator, DriftDiffusionIntegrator, OrnsteinUhlenbeckIntegrator, AccumulatorIntegrator, FHNIntegrator, AGTUtilityIntegrator, BogaczEtAl, NavarroAndFuss, NormalDist, UniformToNormalDist, ExponentialDist, UniformDist, GammaDist, WaldDist, Stability, Distance, Hebbian, Reinforcement, BackPropagation, TDLearning

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
        PM1 = ProcessingMechanism(function=SoftMax)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_LinearMatrix_function(self):
        PM1 = ProcessingMechanism(function=LinearMatrix()
                                  )
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_SimpleIntegrator_function(self):
        PM1 = ProcessingMechanism(function=SimpleIntegrator)
        PM1.execute(1.0)

    def test_processing_mechanism_ConstantIntegrator_function(self):
        PM1 = ProcessingMechanism(function=ConstantIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

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

    def test_processing_mechanism_FHNIntegrator_function(self):
        PM1 = ProcessingMechanism(function=FHNIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_AGTUtilityIntegrator_function(self):
        PM1 = ProcessingMechanism(function=AGTUtilityIntegrator)
        PM1.execute(1.0)
        # assert np.allclose(PM1.value, 1.0)

    def test_processing_mechanism_BogaczEtAl_function(self):
        PM1 = ProcessingMechanism(function=BogaczEtAl)
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

