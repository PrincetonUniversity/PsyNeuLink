import logging

from PsyNeuLink.Components.System import system
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import DDM, DDM_PROBABILITY_UPPER_THRESHOLD
from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.Functions.Function import Linear, Logistic, BogaczEtAl
from PsyNeuLink.scheduling.Scheduler import Scheduler
from PsyNeuLink.scheduling.condition import *


logger = logging.getLogger(__name__)

class TestInit:
    def test_create_scheduler_from_system_StroopDemo(self):
        Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
        Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

        # Processing Mechanisms (Control)
        Color_Hidden = TransferMechanism(name='Colors Hidden',
                                       function=Logistic(gain=(1.0, ControlProjection)))
        Word_Hidden = TransferMechanism(name='Words Hidden',
                                       function=Logistic(gain=(1.0, ControlProjection)))
        Output = TransferMechanism(name='Output',
                                       function=Logistic(gain=(1.0, ControlProjection)))

        # Decision Mechanisms
        Decision = DDM(function=BogaczEtAl(drift_rate=(1.0),
                                           threshold=(0.1654),
                                           noise=(0.5),
                                           starting_point=(0),
                                           t0=0.25),
                       name='Decision')
        # Outcome Mechanisms:
        Reward = TransferMechanism(name='Reward')

        # Processes:
        ColorNamingProcess = process(
            default_input_value = [0],
            pathway = [Color_Input, Color_Hidden, Output, Decision],
            name = 'Color Naming Process')

        WordReadingProcess = process(
            default_input_value = [0],
            pathway = [Word_Input, Word_Hidden, Output, Decision],
            name = 'Word Reading Process')

        RewardProcess = process(
            default_input_value = [0],
            pathway = [(Reward, 1)],
            name = 'RewardProcess')

        # System:
        mySystem = system(processes=[ColorNamingProcess, WordReadingProcess, RewardProcess],
                          controller=EVCMechanism,
                          enable_controller=True,
                          monitor_for_control=[Reward, (DDM_PROBABILITY_UPPER_THRESHOLD, 1, -1)],
                          name='EVC Gratton System')

        sched = Scheduler(system=mySystem)

        integrator_ColorInputPrediction = mySystem.mechanisms[7]
        integrator_RewardPrediction = mySystem.mechanisms[8]
        integrator_WordInputPrediction = mySystem.mechanisms[9]

        expected_consideration_queue = [
            {Color_Input, Word_Input, Reward},
            {Color_Hidden, Word_Hidden, integrator_ColorInputPrediction, integrator_WordInputPrediction, integrator_RewardPrediction},
            {Output},
            {Decision}
        ]

        assert sched.consideration_queue == expected_consideration_queue