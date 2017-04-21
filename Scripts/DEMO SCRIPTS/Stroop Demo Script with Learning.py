# from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
import numpy as np
import random as rand

from PsyNeuLink.Components.Mechanisms.AdaptiveMechanisms.ControlMechanisms.EVCMechanism import EVCMechanism
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import *
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ControlProjection import ControlProjection
from PsyNeuLink.Components.System import system
from PsyNeuLink.Globals.Keywords import *

# Stimulus Mechanisms
Color_Input = TransferMechanism(name='Color Input', function=Linear(slope = 0.2995))
Word_Input = TransferMechanism(name='Word Input', function=Linear(slope = 0.2995))

# Processing Mechanisms (Control)
Color_Hidden = TransferMechanism(name='Colors Hidden',
                               function=Logistic(gain=(1.0, ControlProjection)))
Word_Hidden = TransferMechanism(name='Words Hidden',
                               function=Logistic(gain=(1.0, ControlProjection)))
Output = TransferMechanism(name='Output',
                               function=Logistic(gain=(1.0, ControlProjection)))

# Processes:
ColorNamingProcess = process(
        default_input_value=[0],
        pathway=[Color_Input, Color_Hidden, Output],
        target=[0],
        learning=LEARNING,
        name = 'Color Naming Process')

WordReadingProcess = process(
        default_input_value=[0],
        pathway=[Word_Input, Word_Hidden, Output],
        learning=LEARNING,
        target=[0],
        name = 'Word Reading Process')


# System:
mySystem = system(processes=[ColorNamingProcess, WordReadingProcess],
                  controller=EVCMechanism,
                  enable_controller=True,
                  monitor_for_control=[Output],
                  targets=[0],
                  name='Stroop Learning')
# Show characteristics of system:
mySystem.show()
# mySystem.controller.show()
# mySystem.show_graph(direction='LR')
# mySystem.show_graph_with_learning(direction='LR')
# mySystem.show_graph_with_control(direction='LR')


stim_list_dict = {Color_Input:[1, 1],
                  Word_Input:[-1, -1]}

target_list_dict = {Output:[1]}

# Run system:
Color_Hidden.reportOutputPref = True
mySystem.reportOutputPref = True
mySystem.controller.reportOutputPref = True
mySystem.run(num_executions=2,
             inputs=stim_list_dict,
             targets=target_list_dict)
