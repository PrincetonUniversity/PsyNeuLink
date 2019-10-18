import numpy as np
import psyneulink as pnl


# here we implement a test demo as in the EVC paper example:
#in v2 we add control signals and a EVC mechanism to the model

# EVC params for Umemoto et al
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.transferfunctions

w_t = 0.065
w_d = 0.065
f_t = 1
f_d = 1


# EVC params for Umemoto et al
t0 = 0.2
c = 0.19
thresh = 0.21
x_0 = 0 # starting point

#wTarget = 0.065 # I think this has to do with learning and is constant over trials in Umemoto
costParam1 = 0.35
reconfCostParam1 = 5
rewardTaskA = 50
rewardTaskBToA = 0.7


# Control Parameters
signalSearchRange = np.arange(0.0, 4.1, 0.2) #like in MATLAB Umemoto[0.0:0.2:4.0]# needs to be adjusted

print(signalSearchRange)

# Stimulus Mechanisms
Target_Stim = pnl.TransferMechanism(name='Target Stimulus', function=psyneulink.core.components.functions
                                    .transferfunctions.Linear)
Target_Stim.set_log_conditions('value') # Log Target_Rep

Distractor_Stim = pnl.TransferMechanism(name='Distractor Stimulus', function=psyneulink.core.components.functions.transferfunctions.Linear)
Distractor_Stim.set_log_conditions('value') # Log Target_Rep

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(name='Target Representation',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(
                                        slope=(1.0)))#, pnl.ControlProjection(
                                           # control_signal_params={
                                           #    pnl.ALLOCATION_SAMPLES: signalSearchRange}))))
Target_Rep.set_log_conditions('value') # Log Target_Rep
Target_Rep.loggable_items

Distractor_Rep = pnl.TransferMechanism(name='Distractor Representation',
                                    function=psyneulink.core.components.functions.transferfunctions.Linear(
                                         slope=(1.0)))#, pnl.ControlProjection(
                                            # control_signal_params={
                                            #    pnl.ALLOCATION_SAMPLES: signalSearchRange}))))

Distractor_Rep.set_log_conditions('value') # Log Flanker_Rep
Distractor_Rep.loggable_items
# Processing Mechanism (Automatic)
Automatic_Component_Target = pnl.TransferMechanism(name='Automatic Component Target', function=psyneulink.core.components.functions.transferfunctions.Linear)
Automatic_Component_Target.loggable_items
Automatic_Component_Target.set_log_conditions('value')

# Markus october 25 2018: I think we need 2 automatic components

Automatic_Component_Flanker = pnl.TransferMechanism(name='Automatic Component Flanker', function=psyneulink.core.components.functions.transferfunctions.Linear)
Automatic_Component_Flanker.loggable_items
Automatic_Component_Flanker.set_log_conditions('value')
#


# Decision Mechanisms
Decision = pnl.DDM(function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
       # drift_rate=(0.3),
        threshold=(thresh),
        noise=(c),
        starting_point=(x_0),
        t0=t0
    ),name='Decision',
    output_ports=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
        {
            pnl.NAME: 'OFFSET RT',
            pnl.VARIABLE: (pnl.OWNER_VALUE, 2),
            pnl.FUNCTION: psyneulink.core.components.functions.transferfunctions.Linear(0, slope=1.0, intercept=1)
        }
    ],) #drift_rate=(1.0),threshold=(0.2645),noise=(0.5),starting_point=(0), t0=0.15

print(Decision.execute([1]))

# Decision.set_log_conditions('DECISION_VARIABLE')
# Decision.set_log_conditions('value')
# Decision.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')
Decision.set_log_conditions('InputPort-0')
# Decision.set_log_conditions('RESPONSE_TIME')

# Decision.loggable_items

# Outcome Mechanisms:
Reward = pnl.TransferMechanism(size = 1,
                               name='Reward')

# Processes:
TargetControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    name='Target Control Process'
)

FlankerControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Distractor_Stim, Distractor_Rep, Decision],
    name='Flanker Control Process'
)

TargetAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Automatic_Component_Target, Decision],
    name='Target Automatic Process'
)

FlankerAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Distractor_Stim, Automatic_Component_Flanker, Decision], #
    name='Flanker1 Automatic Process'
)

RewardProcess = pnl.Process(
    pathway=[Reward],
    name='RewardProcess'
)

# System:
mySystem = pnl.System(processes=[TargetControlProcess,
                                 FlankerControlProcess,
                                 TargetAutomaticProcess,
                                 FlankerAutomaticProcess,
                                 RewardProcess],
                      controller=pnl.EVCControlMechanism(
                              control_signals=pnl.ControlSignal(modulates=[(pnl.SLOPE, Target_Rep),
                                                                           (pnl.SLOPE, Distractor_Rep)
                                                                           ],
                                                                function=psyneulink.core.components.functions.transferfunctions.Logistic,
                                                                cost_options=[pnl.CostFunctions.INTENSITY,
                                                                               pnl.CostFunctions.ADJUSTMENT],
                                                                allocation_samples=signalSearchRange
                                                                )),
                      enable_controller=True,
                      monitor_for_control=[
                          # (None, None, np.ones((2,1))), # what the **** is this for? Markus October 25 2018
                          Reward,
                          Decision.PROBABILITY_UPPER_THRESHOLD,
                          ('OFFSET RT', 1, -1),
                      ],
                      name='EVC Markus System')

# log controller

mySystem.loggable_items


# Show characteristics of system:
mySystem.show()
# mySystem.controller.show()

# Show graph of system
mySystem.show_graph(show_control=True, show_dimensions=True)

#Markus: incongruent trial weights:

# f = np.array([1,1])
# W_inc = np.array([[1.0, 0.0],[0.0, 1.5]])
# W_con = np.array([[1.0, 0.0],[1.5, 0.0]])


# generate stimulus environment
nTrials = 3
targetFeatures = [w_t]
flankerFeatures_inc = [w_d]
reward = [100]


targetInputList = targetFeatures
flankerInputList = flankerFeatures_inc
rewardList = reward

stim_list_dict = {
    Target_Stim: targetInputList,
    Distractor_Stim: flankerInputList,
     Reward: rewardList
}

def x():
    #print(mySystem.conroller.)
    # print(mySystem.controller.control_signals.values)
    print("============== ")
    print("decision input vale:", Decision.input_values)
    print("============== ")

    # print(Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD].value)
    # print(Decision.output_ports[pnl.DECISION_VARIABLE].value)
    # print(Decision.output_ports[pnl.RESPONSE_TIME].value)
    # print(Target_Rep.input_values)
    # print("target rep variable:", Target_Rep.input_ports[0].variable)
    # print("target rep input ports:", Target_Rep.input_ports)
    # print("output target stim", Target_Stim.output_values)
    #
    # print(Target_Rep.path_afferents)
    # print("control proj sender value:", Target_Rep.mod_afferents[0].sender.value)
    #
    # # print(Target_Rep.path_afferents)
    #
    #
    # print("distractor rep input: ", Distractor_Rep.input_values)
    # print("my system controller: ", mySystem.controller.control_signals.values)
    # print("my system controller SLOPE: ", mySystem.controller.control_signals.values)
    #
    # print("InputPort bla bla:", Target_Rep.input_ports[0].function.exponents)
    # print("============== ")
    # print("my system  stuff: ", mySystem.controller.control_signals.values)
    #





    # print(Target_Rep.output_values)
    # print(Automatic_Component_Target.output_values)
    #
    # print(Distractor_Rep.output_values)
    # print(Automatic_Component_Flanker.output_values)



mySystem.run(num_trials=nTrials,
             inputs=stim_list_dict,
              call_after_trial=x)

# Flanker_Rep.log.print_entries()
# Target_Rep.log.print_entries()
from pprint import pprint
a = Decision.log.nparray_dictionary()
pprint(a)
# Target_Stim.log.print_entries()
# Distractor_Stim.log.print_entries()
# Target_Rep.log.print_entries()
# Distractor_Rep.log.print_entries()
#
Decision.log.print_entries()
# mySystem.controller.control_signals.values

