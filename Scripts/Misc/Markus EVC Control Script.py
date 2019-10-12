import numpy as np
import psyneulink as pnl

# Control Parameters
import psyneulink.core.components.functions.distributionfunctions
import psyneulink.core.components.functions.transferfunctions

signalSearchRange = np.arange(1.0,3.1,0.5) # why 0.8 to 2.0 in increments of 0.2 np.array([1.0])#


test_mech = pnl.TransferMechanism(size=1)

# Stimulus Mechanisms
Target_Stim = pnl.TransferMechanism(name='Target Stimulus', function=psyneulink.core.components.functions
                                    .transferfunctions.Linear(slope=0.3324))
Target_Stim.set_log_conditions('value')
Flanker_Stim = pnl.TransferMechanism(name='Flanker Stimulus', function=psyneulink.core.components.functions.transferfunctions.Linear(slope=0.3545))
Flanker_Stim.set_log_conditions('value')

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(name='Target Representation',
                                   function=psyneulink.core.components.functions.transferfunctions.Linear(
                                       slope=(1.0, pnl.ControlProjection(
                                           control_signal_params={
                                               pnl.ALLOCATION_SAMPLES: signalSearchRange}))),
                                   prefs = {pnl.LOG_PREF: pnl.PreferenceEntry(pnl.LogCondition.INITIALIZATION, pnl.PreferenceLevel.INSTANCE)})
Target_Rep.set_log_conditions('value') # Log Target_Rep
Target_Rep.set_log_conditions('slope') # Log Target_Rep
Target_Rep.loggable_items

#log initialization

Target_Rep.log.LogCondition =2

Flanker_Rep = pnl.TransferMechanism(name='Flanker Representation',
                                    function=psyneulink.core.components.functions.transferfunctions.Linear(
                                        slope=(1.0, pnl.ControlProjection(
                                            control_signal_params={
                                                pnl.ALLOCATION_SAMPLES: signalSearchRange}))))
Flanker_Rep.set_log_conditions('value') # Log Flanker_Rep
Flanker_Rep.set_log_conditions('slope') # Log Flanker_Rep
Flanker_Rep.loggable_items

Target_Rep.log.LogCondition =2

# Processing Mechanism (Automatic)
Automatic_Component = pnl.TransferMechanism(name='Automatic Component', function=psyneulink.core.components.functions.transferfunctions.Linear)
Automatic_Component.loggable_items
Automatic_Component.set_log_conditions('value')

# Decision Mechanisms
Decision = pnl.DDM(function=psyneulink.core.components.functions.distributionfunctions.DriftDiffusionAnalytical(
        drift_rate=1.0,
        threshold=0.2645,
        # noise=(0.5),
        starting_point=0,
        t0=0.15
    ),name='Decision',
    output_ports=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
        {
            pnl.NAME: 'OFFSET_RT',
            pnl.VARIABLE: (pnl.OWNER_VALUE, 1),
                           pnl.FUNCTION: psyneulink.core.components.functions.transferfunctions.Linear(0, slope=0.0, intercept=1).function
        }
    ],) #drift_rate=(1.0),threshold=(0.2645),noise=(0.5),starting_point=(0), t0=0.15
Decision.set_log_conditions('DECISION_VARIABLE')
Decision.set_log_conditions('value')
Decision.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')
Decision.set_log_conditions('InputPort-0')
Decision.set_log_conditions('drift_rate')

Decision.set_log_conditions('OFFSET_RT')

Decision.set_log_conditions('RESPONSE_TIME')

Decision.loggable_items

# Outcome Mechanisms:
Reward = pnl.TransferMechanism(name='Reward')
Reward.set_log_conditions('value')
# Processes:
TargetControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Target_Rep, Decision],
    name='Target Control Process'
)

FlankerControlProcess = pnl.Process(
    default_variable=[0],
    pathway=[Flanker_Stim, Flanker_Rep, Decision],
    name='Flanker Control Process'
)

TargetAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Target_Stim, Automatic_Component, Decision],
    name='Target Automatic Process'
)

FlankerAutomaticProcess = pnl.Process(
    default_variable=[0],
    pathway=[Flanker_Stim, Automatic_Component, Decision],
    name='Flanker1 Automatic Process'
)

RewardProcess = pnl.Process(
    default_variable=[0],
    pathway=[Reward, test_mech],
    name='RewardProcess'
)


# System:
mySystem = pnl.System(processes=[TargetControlProcess,
        FlankerControlProcess,
        TargetAutomaticProcess,
        FlankerAutomaticProcess,
        RewardProcess],
    controller=pnl.EVCControlMechanism(prefs={pnl.LOG_PREF: pnl.PreferenceEntry(pnl.LogCondition.INITIALIZATION, pnl.PreferenceLevel.INSTANCE)}),
    enable_controller=True,
    monitor_for_control=[
        # (None, None, np.ones((1,1))),
        Reward,
        Decision.PROBABILITY_UPPER_THRESHOLD,
        ('OFFSET_RT', 1, -1),
    ],
    name='EVC Markus System')

# Show characteristics of system:
mySystem.show()
# mySystem.controller.show()

# Show graph of system
# mySystem.show_graph(show_control=pnl.ALL, show_dimensions=pnl.ALL)# show_control=True,show_dimensions=True)
mySystem.show_graph(show_control=pnl.ALL, show_mechanism_structure=True)# show_control=True,show_dimensions=True)


#log InputPort of mySystem
mySystem.controller.loggable_items
mySystem.controller.set_log_conditions('InputPort-0')
mySystem.controller.set_log_conditions('value')

mySystem.controller.set_log_conditions('Flanker Representation[slope] ControlSignal')
mySystem.controller.set_log_conditions('Target Representation[slope] ControlSignal')

mySystem.controller.objective_mechanism.set_log_conditions('value')
mySystem.controller.objective_mechanism.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')
mySystem.controller.objective_mechanism.set_log_conditions('OFFSET_RT')

# print('current input value',mySystem.controller.input_ports.values)
# print('current objective mech output value',mySystem.controller.objective_mechanism.output_ports.values)
#


# configure EVC components
mySystem.controller.control_signals[0].intensity_cost_function = psyneulink.core.components.functions.transferfunctions.Exponential(rate=0.8046).function
mySystem.controller.control_signals[1].intensity_cost_function = psyneulink.core.components.functions.transferfunctions.Exponential(rate=0.8046).function
#
# #change prediction mechanism function.rate for all 3 prediction mechanisms
#
mySystem.controller.prediction_mechanisms.mechanisms[0].function.rate = 1.0
mySystem.controller.prediction_mechanisms.mechanisms[1].function.rate = 0.0  # reward rate
mySystem.controller.prediction_mechanisms.mechanisms[2].function.rate = 1.0





# log predictions:
#how to log this??? with prefs??
#mySystem.controller.prediction_mechanisms.mechanisms.


# add weight matrix for input updates here ! ??? ask Sebastian on march 9!

# W_new = W_hat_old + alpha*(W_hat_predicted - W_actual)




# for mech in mySystem.controller.prediction_mechanisms.mechanisms:
#     if mech.name == 'Flanker Stimulus Prediction Mechanism' or mech.name == 'Target Stimulus Prediction Mechanism':
#         # when you find a key mechanism (transfer mechanism) with the correct name, print its name
#         # print(mech.name)
#         mech.function.rate = 1.0
#
#     if 'Reward' in mech.name:
#         # print(mech.name)
#         mech.function.rate = 1.0
#         # mySystem.controller.prediction_mechanisms[mech].parameterPorts['rate'].base_value = 1.0
#



#Markus: incongruent trial weights:

# f = np.array([1,1])
# W_inc = np.array([[1.0, 0.0],[0.0, 1.5]])
# W_con = np.array([[1.0, 0.0],[1.5, 0.0]])


# generate stimulus environment: remember that we add one congruent stimulus infront of actuall stimulus list
# compatible with MATLAB stimulus list for initialization
nTrials = 7
targetFeatures = [1.0, 1.0, 1.0, 1.0,1.0, 1.0, 1.0, 1.0]
flankerFeatures = [1.0, 1.0, 1.0, 1.0,-1.0, -1.0, 1.0, -1.0]
reward = [100, 100, 100, 100, 100, 100, 100, 100]

stim_list_dict = {
    Target_Stim: targetFeatures,
    Flanker_Stim: flankerFeatures,
    Reward: reward

}
Target_Rep.set_log_conditions('slope')
# mySystem.controller.objective_mechanism.loggable_items
mySystem.run(num_trials=nTrials,inputs=stim_list_dict)

# Flanker_Rep.log.print_entries()
# Target_Rep.log.print_entries()
# Decision.log.print_entries()

# print('OutputPort of objective mechanism', mySystem.controller.objective_mechanism.output_ports.values)
#
# print('InputPort of EVC Control mechanism', mySystem.controller.input_port.value)
#
# print('mapping projection from objective mechanism to EVC Control mechanism',mySystem.controller.projections[0].matrix)

# mySystem.controller.log.print_entries()

# Reward.log.print_entries()
D = Decision.log.nparray_dictionary()
print(D['drift_rate'])

Flanker_Rep.log.print_entries()

mySystem.controller.log.print_entries()


d = mySystem.controller.objective_mechanism.log.nparray_dictionary()
mySystem.controller.objective_mechanism.log.print_entries()
print(d['PROBABILITY_UPPER_THRESHOLD'])


# assert np.allclose([94.81, 47.66, 94.81, 94.81, 47.66, 47.66, 94.81, 47.66],
