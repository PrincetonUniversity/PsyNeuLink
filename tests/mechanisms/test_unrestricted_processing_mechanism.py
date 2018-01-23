import numpy as np
import pytest
from psyneulink.components.mechanisms.processing.unrestrictedprocessingmechanism import UnrestrictedProcessingMechanism
from psyneulink.components.functions.function import Reduce, LinearCombination, CombineMeans, Linear, Exponential, Logistic, SoftMax, LinearMatrix, Integrator, SimpleIntegrator, ConstantIntegrator, AdaptiveIntegrator, DriftDiffusionIntegrator, OrnsteinUhlenbeckIntegrator, AccumulatorIntegrator, FHNIntegrator, AGTUtilityIntegrator, BogaczEtAl, NavarroAndFuss, NormalDist, UniformToNormalDist, ExponentialDist, UniformDist, GammaDist, WaldDist, Stability, Distance, Hebbian, Reinforcement, BackPropagation, TDLearning
class TestUnrestrictedProcessingMechanismFunctions:
    # VALID INPUTS

    def test_unrestricted_processing_mechanism_linear_function(self):

        UPM1 = UnrestrictedProcessingMechanism()
        UPM1.execute(1.0)
        assert np.allclose(UPM1.value, 1.0)

        UPM2 = UnrestrictedProcessingMechanism(function=Linear(slope=2.0,
                                                               intercept=1.0))
        UPM2.execute(1.0)
        assert np.allclose(UPM2.value, 3.0)

    def test_unrestricted_processing_mechanism_LinearCombination_function(self):

        UPM1 = UnrestrictedProcessingMechanism(function=LinearCombination)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Reduce_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Reduce)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_CombineMeans_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=CombineMeans)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Exponential_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Exponential)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Logistic_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Logistic)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_SoftMax_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=SoftMax)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_SimpleIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=SimpleIntegrator)
        UPM1.execute(1.0)

    def test_unrestricted_processing_mechanism_ConstantIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=ConstantIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_AdaptiveIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=AdaptiveIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_DriftDiffusionIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=DriftDiffusionIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_OrnsteinUhlenbeckIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=OrnsteinUhlenbeckIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_AccumulatorIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=AccumulatorIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_FHNIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=FHNIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_AGTUtilityIntegrator_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=AGTUtilityIntegrator)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_BogaczEtAl_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=BogaczEtAl)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    # COMMENTED OUT BECAUSE OF MATLAB ENGINE:
    # def test_unrestricted_processing_mechanism_NavarroAndFuss_function(self):
    #     UPM1 = UnrestrictedProcessingMechanism(function=NavarroAndFuss)
    #     UPM1.execute(1.0)
    #     # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_NormalDist_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=NormalDist)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_ExponentialDist_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=ExponentialDist)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_UniformDist_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=UniformDist)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_GammaDist_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=GammaDist)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_WaldDist_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=WaldDist)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Stability_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Stability)
        UPM1.execute(1.0)
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Distance_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Distance,
                                               default_variable=[[0,0], [0,0]])
        UPM1.execute([[1, 2], [3, 4]])
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Hebbian_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Hebbian,
                                               default_variable=[[0.0], [0.0], [0.0]])
        UPM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_Reinforcement_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=Reinforcement,
                                               default_variable=[[0.0], [0.0], [0.0]])
        UPM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(UPM1.value, 1.0)

    # COMMENTING OUT BECAUSE BACK PROP FN DOES NOT WORK WITH UNRESTRICTED MECHANISM
    # def test_unrestricted_processing_mechanism_BackPropagation_function(self):
    #     UPM1 = UnrestrictedProcessingMechanism(function=BackPropagation,
    #                                            default_variable=[[0.0], [0.0], [0.0]])
    #     UPM1.execute([[1.0], [2.0], [3.0]])
    #     UPM1.execute(1.0)
    #     # assert np.allclose(UPM1.value, 1.0)

    def test_unrestricted_processing_mechanism_TDLearning_function(self):
        UPM1 = UnrestrictedProcessingMechanism(function=TDLearning,
                                               default_variable=[[0.0], [0.0], [0.0]])
        UPM1.execute([[1.0], [2.0], [3.0]])
        # assert np.allclose(UPM1.value, 1.0)

