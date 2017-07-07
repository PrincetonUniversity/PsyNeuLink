import numpy as np

from PsyNeuLink.Components.Functions.Function import PROB
from PsyNeuLink.Components.Functions.Function import Reinforcement, SoftMax
from PsyNeuLink.Components.Mechanisms.ProcessingMechanisms.TransferMechanism import TransferMechanism
from PsyNeuLink.Components.Process import process
from PsyNeuLink.Components.Projections.ModulatoryProjections.LearningProjection import LearningProjection
from PsyNeuLink.Components.System import System, system
from PsyNeuLink.Scheduling.TimeScale import CentralClock

import random
random.seed(0)
np.random.seed(0)

input_layer = TransferMechanism(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = TransferMechanism(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=1.0),
                            name='Action Selection')

p = process(default_input_value=[0, 0, 0],
            pathway=[input_layer,action_selection],
            learning=LearningProjection(learning_function=Reinforcement(learning_rate=.05)),
            target=0)

print ('reward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
print ('target_mechanism weights: \n', action_selection.output_state.efferents[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.output_state.value = [0, 0, 1]
# Get reward value for selected action)
reward = lambda : [reward_values[int(np.nonzero(action_selection.output_state.value)[0])]]

def print_header():
    print("\n\n**** TRIAL: ", CentralClock.trial)

def show_weights():
    print ('\nreward prediction weights: \n', action_selection.input_state.path_afferents[0].matrix)
    print ('action selected: ', action_selection.output_state.value)

p.run(num_executions=10,
      # inputs=[[[1, 1, 1]]],
      # inputs=[ [ [1, 1, 1] ],[ [.2, 1, .2] ]],
      inputs={input_layer:[[1, 1, 1],[.2, 1, .2]]},
      targets=reward,
      call_before_trial=print_header,
      call_after_trial=show_weights
      )
