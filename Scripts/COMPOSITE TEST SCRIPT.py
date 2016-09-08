from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.AdaptiveIntegrator import *
# from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Deprecated.LinearMechanism import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import *
from PsyNeuLink.Functions.Projections.ControlSignal import ControlSignal
from PsyNeuLink.Functions.Utility import Exponential, Linear

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.System import System_Base
from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Globals.Keywords import *

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.Transfer import Transfer
from PsyNeuLink.Functions.Projections.Mapping import Mapping
from PsyNeuLink.Functions.Projections.LearningSignal import LearningSignal
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Functions.Utility import Logistic, LinearMatrix

# **********************************************************************************************************************
# EVC Laming Validation ************************************************************************************************
# **********************************************************************************************************************

#region Preferences
DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(False,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(False,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))
#endregion

#region Mechanisms
Input = Transfer(name='Input')
Reward = Transfer(name='Reward')
Decision = DDM(function=BogaczEtAl(drift_rate=(1.0, ControlSignal(function=Linear)),
                                   threshold=(1.0),
                                   noise=(0.5),
                                   starting_point=(0),
                                   T0=0.45),
               prefs = DDM_prefs,
               name='Decision')
#endregion

#region Processes
TaskExecutionProcess = process(default_input_value=[0],
                               configuration=[(Input, 0), IDENTITY_MATRIX, (Decision, 0)],
                               prefs = process_prefs,
                               name = 'TaskExecutionProcess')

RewardProcess = process(default_input_value=[0],
                        configuration=[(Reward, 1)],
                        prefs = process_prefs,
                        name = 'RewardProcess')
#endregion

#region System
mySystem = System_Base(processes=[TaskExecutionProcess, RewardProcess],
                       monitored_output_states=[Reward, kwDDM_Probability_upperBound,(kwDDM_RT_Mean, -1, 1)],
                       name='EVC Test System')
#endregion

#region Inspect
mySystem.inspect()
mySystem.controller.inspect()
#endregion

#region Run

inputList = [0.5, 0.123]
rewardList = [20, 20]

for i in range(0,2):

    print("\n############################ TRIAL {} ############################".format(i));

    stimulusInput = inputList[i]
    rewardInput = rewardList[i]

    # Present stimulus:
    CentralClock.time_step = 0
    mySystem.execute([[stimulusInput],[0]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateNames,
                               mySystem.terminalMechanisms.outputStateValues))

    # Present feedback:
    CentralClock.time_step = 1
    mySystem.execute([[0],[rewardInput]])
    print ('\n{0}\n{1}'.format(mySystem.terminalMechanisms.outputStateNames,
                               mySystem.terminalMechanisms.outputStateValues))

#endregion

# output states in EVCMechanism DDM_Error_Rate and DDM_RT_Mean are flipped
# first control intensity in allocation list is 0 but appears to be 1 when multiplied times drift
# how to specify stimulus learning rate? currently there appears to be no learning
# no learning rate


# **********************************************************************************************************************
# MULTILAYER LEARNING **************************************************************************************************
# **********************************************************************************************************************

Input_Layer = Transfer(name='Input Layer',
                       function=Logistic(),
                       default_input_value = np.zeros((2,)))

Hidden_Layer_1 = Transfer(name='Hidden Layer_1',
                          function=Logistic(),
                          default_input_value = np.zeros((5,)))

Hidden_Layer_2 = Transfer(name='Hidden Layer_2',
                          function=Logistic(),
                          default_input_value = [0,0,0,0])

Output_Layer = Transfer(name='Output Layer',
                        function=Logistic(),
                        default_input_value = [0,0,0])

randomized_matrix = lambda sender, receiver, range, offset: ((range * np.random.rand(sender, receiver)) + offset)
random_weight_matrix = lambda sender, receiver : randomized_matrix(sender, receiver, .2, -.1)

Input_Weights = Mapping(name='Input Weights',
                        sender=Input_Layer,
                        receiver=Hidden_Layer_1,
                        matrix=(random_weight_matrix, LearningSignal()),
                        )

Middle_Weights = Mapping(name='Middle Weights',
                         sender=Hidden_Layer_1,
                         receiver=Hidden_Layer_2,
                        matrix=(FULL_CONNECTIVITY_MATRIX, LearningSignal())
                         )
Output_Weights = Mapping(name='Output Weights',
                         sender=Hidden_Layer_2,
                         receiver=Output_Layer,
                         params={FUNCTION_PARAMS: {MATRIX: (FULL_CONNECTIVITY_MATRIX, LEARNING_SIGNAL)}}
                         )

z = process(default_input_value=[0, 0],
            configuration=[Input_Layer,
                           # Input_Weights,
                           Hidden_Layer_1,
                           # Middle_Weights,
                           Hidden_Layer_2,
                           # Output_Weights,
                           Output_Layer],
            learning=LearningSignal,
            prefs={kpVerbosePref: PreferenceEntry(False, PreferenceLevel.INSTANCE)})

# print ('Input Weights: \n', Input_Weights.matrix)
# print ('Middle Weights: \n', Middle_Weights.matrix)
# print ('Output Weights: \n', Output_Weights.matrix)

for i in range(10):

    # z.execute([-1, 30])
    z.execute([[-1, 30],[0, 0, 1]])
    # z.execute([[-1, 30],[0]])

    # print ('Input Weights: \n', Input_Weights.matrix)
    # print ('Middle Weights: \n', Middle_Weights.matrix)
    # print ('Output Weights: \n', Output_Weights.matrix)


# **********************************************************************************************************************
# RL *******************************************************************************************************************
# **********************************************************************************************************************

from PsyNeuLink import *
from PsyNeuLink.Functions.Utility import SoftMax, Reinforcement
import numpy as np

input_layer = Transfer(default_input_value=[0,0,0],
                       name='Input Layer')

action_selection = Transfer(default_input_value=[0,0,0],
                            function=SoftMax(output=PROB,
                                             gain=1.0),
                            name='Action Selection')

p = process(default_input_value=[0, 0, 0],
            configuration=[input_layer,action_selection],
            learning=LearningSignal(function=Reinforcement(learning_rate=.05)))

print ('reward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)
print ('comparator weights: \n', action_selection.outputState.sendsToProjections[0].matrix)

actions = ['left', 'middle', 'right']
reward_values = [15, 7, 13]
first_reward = 0

# Must initialize reward (won't be used, but needed for declaration of lambda function)
action_selection.outputState.value = [0, 0, 1]
# Get reward value for selected action)
reward = lambda : [reward_values[int(np.nonzero(action_selection.outputState.value)[0])]]

# Run process with RL
for i in range(100):

    # # Execute process, including weight adjustment based on last reward
    result = p.execute([[1, 1, 1], reward])

    print ('result: ', result)

    # Note: this shows weights updated on prior trial, not current one
    #       (this is a result of parameterState "lazy updating" -- only updated when called)
    print ('\nreward prediction weights: \n', action_selection.inputState.receivesFromProjections[0].matrix)


# **********************************************************************************************************************
# DDM ******************************************************************************************************************
# **********************************************************************************************************************

from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.DDM import *
# from PsyNeuLink.Functions.Process import Process_Base
from PsyNeuLink.Functions.Process import process
from PsyNeuLink.Globals.Keywords import *

DDM_prefs = FunctionPreferenceSet(
                prefs = {
                    kpVerbosePref: PreferenceEntry(True,PreferenceLevel.INSTANCE),
                    kpReportOutputPref: PreferenceEntry(True,PreferenceLevel.INSTANCE)})

# my_DDM = DDM(name='my_DDM')

my_DDM = DDM(function=BogaczEtAl(drift_rate=(1.0, CONTROL_SIGNAL),
                                 threshold=(10.0, CONTROL_SIGNAL),
                                 starting_point=0.0),
             name='My_DDM',
             prefs = DDM_prefs
             )


my_DDM.prefs = DDM_prefs
my_DDM.prefs.level = PreferenceLevel.SYSTEM

my_DDM.prefs.inspect()

myMechanism = DDM(function=BogaczEtAl(drift_rate=(1.0, CONTROL_SIGNAL),
                                      threshold=(10.0, CONTROL_SIGNAL),
                                      starting_point=0.0),
                  prefs = DDM_prefs,
                  name='My_DDM'
                  )

myMechanism_2 = DDM(function=BogaczEtAl(drift_rate=2.0,
                                        threshold=20.0),
                    prefs = DDM_prefs,
                    name='My_DDM_2'
                    )

myMechanism_3 = DDM(function=BogaczEtAl(drift_rate=3.0,
                                        threshold=30.0),
                    prefs = DDM_prefs,
                    name='My_DDM_3'
                    )

process_prefs = FunctionPreferenceSet(reportOutput_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE),
                                      verbose_pref=PreferenceEntry(True,PreferenceLevel.INSTANCE))

process_prefs.inspect()

z = process(default_input_value=[[30], [10]],
            params={CONFIGURATION:[myMechanism,
                                   (IDENTITY_MATRIX),
                                   myMechanism_2,
                                   (FULL_CONNECTIVITY_MATRIX),
                                   myMechanism_3]},
            prefs = process_prefs)

z.execute([[30], [10]])

myMechanism.log.print_entries(ALL_ENTRIES, kwTime, kwValue)
