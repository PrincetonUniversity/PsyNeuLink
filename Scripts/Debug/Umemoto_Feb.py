import psyneulink as pnl


# here we implement a test demo as in the EVC paper example:
#in v2 we add control signals and a EVC mechanism to the model

# EVC params for Umemoto et al

w_t = 0.065
w_d = 0.065 # made negative here to match -1 values for distractor
f_t = 1
f_d = 1


# EVC params for Umemoto et al
non_decision_time = 0.2
c = 0.19
thresh = 0.21
x_0 = 0 # starting point

#wTarget = 0.065 # I think this has to do with learning and is constant over trials in Umemoto
costParam1 = 0.35
reconfCostParam1 = 5
#rewardTaskA = 50
#rewardTaskBToA = 0.7


# Control Parameters
# signalSearchRange = pnl.SampleSpec(start=1, stop=1.8, step=0.2)
# signalSearchRange = pnl.SampleSpec(start=0.0, stop=0.4, step=0.2)
signalSearchRange = pnl.SampleSpec(start=0.0, stop=0.5, step=0.2)

# Stimulus Mechanisms
Target_Stim = pnl.TransferMechanism(name='Target Stimulus', function=pnl.Linear)
Target_Stim.set_log_conditions('value')#, log_condition=pnl.PROCESSING) # Log Target_Rep

Distractor_Stim = pnl.TransferMechanism(name='Distractor Stimulus', function=pnl.Linear)
Distractor_Stim.set_log_conditions('value')#, log_condition=pnl.PROCESSING) # Log Target_Rep

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(name='Target Representation')

Target_Rep.set_log_conditions('value')#, log_condition=pnl.PROCESSING) # Log Target_Rep
Target_Rep.set_log_conditions('mod_slope')#, log_condition=pnl.PROCESSING)
Target_Rep.set_log_conditions('InputPort-0')#, log_condition=pnl.PROCESSING)

Distractor_Rep = pnl.TransferMechanism(name='Distractor Representation')

Distractor_Rep.set_log_conditions('value')#, log_condition=pnl.PROCESSING) # Log Flanker_Rep
Distractor_Rep.set_log_conditions('mod_slope')#, log_condition=pnl.PROCESSING)

# Processing Mechanism (Automatic)
Automatic_Component = pnl.TransferMechanism(name='Automatic Component')

Automatic_Component.set_log_conditions('value')#, log_condition=pnl.PROCESSING)

# Decision Mechanisms
Decision = pnl.DDM(function=pnl.DriftDiffusionAnalytical(
       # drift_rate=(0.1170),
        threshold=(thresh),
        noise=(c),
        starting_value=(x_0),
        non_decision_time=non_decision_time
    ),name='Decision',
    output_ports=[
        pnl.DECISION_VARIABLE,
        pnl.RESPONSE_TIME,
        pnl.PROBABILITY_UPPER_THRESHOLD,
        {
            pnl.NAME: 'OFFSET RT',
            pnl.VARIABLE: (pnl.OWNER_VALUE, 2),
            pnl.FUNCTION: pnl.Linear(0, slope=1.0, intercept=1)
        }
    ],) #drift_rate=(1.0),threshold=(0.2645),noise=(0.5),non_decision_time=(0), non_decision_time=0.15

Decision.set_log_conditions('InputPort-0')#, log_condition=pnl.PROCESSING)

# Outcome Mechanisms:
Reward = pnl.TransferMechanism(name='Reward')

# Composition
Umemoto_comp = pnl.Composition(name="Umemoto_System")

# ADD pathways
TargetControl_pathway = [Target_Stim, Target_Rep, Decision]
Umemoto_comp.add_linear_processing_pathway(TargetControl_pathway)

FlankerControl_pathway = [Distractor_Stim, Distractor_Rep, Decision]
Umemoto_comp.add_linear_processing_pathway(FlankerControl_pathway)

TargetAutomatic_pathway = [Target_Stim, Automatic_Component, Decision]
Umemoto_comp.add_linear_processing_pathway(TargetAutomatic_pathway)

FlankerAutomatic_pathway = [Distractor_Stim, Automatic_Component, Decision]
Umemoto_comp.add_linear_processing_pathway(FlankerAutomatic_pathway)

Reward_pathway = [Reward]
Umemoto_comp.add_linear_processing_pathway(Reward_pathway)

Umemoto_comp.add_node(Decision,
                      # required_roles=pnl.NodeRole.OUTPUT
                      )

# COMPOSITION
Target_Rep_Control_Signal = pnl.ControlSignal(modulates=[(pnl.SLOPE, Target_Rep)],
                                              function=pnl.Linear,
                                              variable=1.0,
                                              cost_options=[pnl.CostFunctions.INTENSITY, pnl.CostFunctions.ADJUSTMENT],
                                              intensity_cost_function=pnl.Exponential(scale=1, rate=1),
                                              adjustment_cost_function=pnl.Exponential(scale=1, rate=1, offset=-1),
                                              allocation_samples=signalSearchRange)

Distractor_Rep_Control_Signal = pnl.ControlSignal(modulates=[(pnl.SLOPE, Distractor_Rep)],
                                                  function=pnl.Linear,
                                                  variable=1.0,
                                                  cost_options=[pnl.CostFunctions.INTENSITY, pnl.CostFunctions.ADJUSTMENT],
                                                  intensity_cost_function=pnl.Exponential(rate=0.8046),
                                                  adjustment_cost_function=pnl.Exponential(scale=1, rate=1, offset=-1),
                                                  allocation_samples=signalSearchRange)

Umemoto_comp.add_model_based_optimizer(optimizer=pnl.OptimizationControlMechanism(agent_rep=Umemoto_comp,
                                                                                  state_features=[Target_Stim.input_port, Distractor_Stim.input_port, Reward.input_port],
                                                                                  state_feature_function=pnl.AdaptiveIntegrator(rate=1.0),
                                                                                  objective_mechanism=pnl.ObjectiveMechanism(monitor_for_control=[Reward,
                                                                                                                                                 (Decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD], 1, -1)],
                                                                                                                             ),
                                                                                  function=pnl.GridSearch(),
                                                                                  control_signals=[Target_Rep_Control_Signal, Distractor_Rep_Control_Signal]
                                                                                  )
                                                                                )
Umemoto_comp.enable_model_based_optimizer = True
Umemoto_comp.model_based_optimizer.set_log_conditions('value')

nTrials = 2
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

Umemoto_comp.run(num_trials=nTrials,
                 inputs=stim_list_dict)


# print("\n\n---------  DISTRACTOR REP  ---------")
# Distractor_Rep.log.print_entries()
# print("\n\n---------  TARGET REP  ---------")
# Target_Rep.log.print_entries()
# print("\n\n---------  AUTOMATIC COMPONENT  ---------")
# Automatic_Component.log.print_entries()
print("\n\n---------  DECISION  ---------")
Decision.log.print_entries()
# print("\n\n---------  MODEL BASED OPTIMIZER  ---------")
# Umemoto_comp.model_based_optimizer.log.print_entries()
# print("\n\n---------  Target_Stim  ---------")
#
# Target_Stim.log.print_entries()