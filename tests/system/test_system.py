import numpy as np
import random

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Projections.MappingProjection import MappingProjection
from PsyNeuLink.Components.Functions.Function import Linear, Logistic, BogaczEtAl, Integrator
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import *

def test_danglingControlledMech():
    #
    #   first section is from Stroop Demo
    #
    Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
    Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

    # Processing Mechanisms (Control)
    Color_Hidden = TransferMechanism(
        name='Colors Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Word_Hidden = TransferMechanism(
        name='Words Hidden',
        function=Logistic(gain=(1.0, ControlProjection)),
    )
    Output = TransferMechanism(
        name='Output',
        function=Logistic(gain=(1.0, ControlProjection)),
    )

    # Decision Mechanisms
    Decision = DDM(
        function=BogaczEtAl(
            drift_rate=(1.0),
            threshold=(0.1654),
            noise=(0.5),
            starting_point=(0),
            t0=0.25,
        ),
        name='Decision',
    )
    # Outcome Mechanisms:
    Reward = TransferMechanism(name='Reward')

    # Processes:
    ColorNamingProcess = process(
        default_input_value = [0],
        pathway = [Color_Input, Color_Hidden, Output, Decision],
        name = 'Color Naming Process',
    )

    WordReadingProcess = process(
        default_input_value = [0],
        pathway = [Word_Input, Word_Hidden, Output, Decision],
        name = 'Word Reading Process',
    )

    RewardProcess = process(
        default_input_value = [0],
        pathway = [Reward],
        name = 'RewardProcess',
    )

    # add another DDM but do not add to system
    second_DDM = DDM(
        function=BogaczEtAl(
            drift_rate=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            threshold=(
                1.0,
                ControlProjection(
                    function=Linear,
                    control_signal={
                        ALLOCATION_SAMPLES:np.arange(0.1, 1.01, 0.3)
                    },
                ),
            ),
            noise=(0.5),
            starting_point=(0),
            t0=0.45
        ),
        name='second_DDM',
    )

    # System:
    mySystem = system(
        processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
        controller=EVCMechanism,
        enable_controller=True,
        monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
        name='EVC Gratton System',
    )

    # no assert, should only complete without error

class TestDocumentationExamples:
    def test_mechs_in_pathway(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = DDM()
        some_params = {PARAMETER_STATE_PARAMS:{THRESHOLD:2,NOISE:0.1}}
        my_process = process(pathway=[mechanism_1, TransferMechanism, (mechanism_2, some_params, 0)])
        result = my_process.execute()

        assert(result == np.array([2]))

    def test_default_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_inline_projection_using_existing_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        projection_A = MappingProjection()
        my_process = process(pathway=[mechanism_1, projection_A, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_inline_projection_using_keyword(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        my_process = process(pathway=[mechanism_1, RANDOM_CONNECTIVITY_MATRIX, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_standalone_projection(seed0):
        mechanism_1 = TransferMechanism()
        mechanism_2 = TransferMechanism()
        mechanism_3 = DDM()
        projection_A = MappingProjection(sender=mechanism_1, receiver=mechanism_2)
        my_process = process(pathway=[mechanism_1, mechanism_2, mechanism_3])
        result = my_process.execute()

        assert(result == np.array([1.]))

    def test_process_learning(seed0):
        mechanism_1 = TransferMechanism(function=Logistic)
        mechanism_2 = TransferMechanism(function=Logistic)
        mechanism_3 = TransferMechanism(function=Logistic)
        my_process = process(
            pathway=[mechanism_1, mechanism_2, mechanism_3],
            learning=LEARNING_PROJECTION,
            target=[0],
        )
        result = my_process.execute()

        np.testing.assert_allclose(result, np.array([0.65077768]))
