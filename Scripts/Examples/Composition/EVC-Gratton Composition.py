from psyneulink import *

# Stimulus Mechanisms
target_stim = TransferMechanism(name='Target Stimulus',
                                    function=Linear(slope=0.3324))
flanker_stim = TransferMechanism(name='Flanker Stimulus',
                                     function=Linear(slope=0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = TransferMechanism(name='Target Representation')
Flanker_Rep = TransferMechanism(name='Flanker Representation')

# Processing Mechanism (Automatic)
Automatic_Component = TransferMechanism(name='Automatic Component')

# Decision Mechanism
Decision = DDM(name='Decision',
                   function=DriftDiffusionAnalytical(drift_rate=(1.0),
                                                         threshold=(0.2645),
                                                         noise=(0.5),
                                                         starting_point=(0),
                                                         t0=0.15),
                   output_states=[DECISION_VARIABLE,
                                  RESPONSE_TIME,
                                  PROBABILITY_UPPER_THRESHOLD]
                   )

# Outcome Mechanism
reward = TransferMechanism(name='reward')

# Pathways
target_control_pathway = [target_stim, Target_Rep, Decision]
flanker_control_pathway = [flanker_stim, Flanker_Rep, Decision]
target_automatic_pathway = [target_stim, Automatic_Component, Decision]
flanker_automatic_pathway = [flanker_stim, Automatic_Component, Decision]
pathways = [target_control_pathway, flanker_control_pathway, target_automatic_pathway,
            flanker_automatic_pathway]

# Composition
evc_gratton = Composition(name="EVCGratton")
evc_gratton.add_node(Decision, required_roles=NodeRole.OUTPUT)
for path in pathways:
    evc_gratton.add_linear_processing_pathway(path)
evc_gratton.add_node(reward, required_roles=NodeRole.OUTPUT)

# Control Signals
signalSearchRange = SampleSpec(start=1.0, stop=1.8, step=0.2)

target_rep_control_signal = ControlSignal(projections=[(SLOPE, Target_Rep)],
                                              function=Linear,
                                              variable=1.0,
                                              intensity_cost_function=Exponential(rate=0.8046),
                                              allocation_samples=signalSearchRange)

flanker_rep_control_signal = ControlSignal(projections=[(SLOPE, Flanker_Rep)],
                                               function=Linear,
                                               variable=1.0,
                                               intensity_cost_function=Exponential(rate=0.8046),
                                               allocation_samples=signalSearchRange)

objective_mech = ObjectiveMechanism(name='EVC Objective Mechanism', function=LinearCombination(operation=PRODUCT),
                                        monitor=[reward,
                                                                 (Decision.output_states[
                                                                      PROBABILITY_UPPER_THRESHOLD], 1, -1)])
# Model Based OCM (formerly controller)
evc_gratton.add_controller(controller=OptimizationControlMechanism(name='EVC OCM',
                                                                  agent_rep=evc_gratton,
                                                                  features=[target_stim.input_state,
                                                                            flanker_stim.input_state,
                                                                            reward.input_state],
                                                                  feature_function=AdaptiveIntegrator(
                                                                          rate=1.0),
                                                                  objective_mechanism=objective_mech,
                                                                  function=GridSearch(),
                                                                  control_signals=[
                                                                      target_rep_control_signal,
                                                                      flanker_rep_control_signal]))
evc_gratton.enable_controller = True

targetFeatures = [1, 1, 1]
flankerFeatures = [1, -1, 1]
rewardValues = [100, 100, 100]

stim_list_dict = {target_stim: targetFeatures,
                  flanker_stim: flankerFeatures,
                  reward: rewardValues}

evc_gratton.show_graph(show_model_based_optimizer=True,
                       show_node_structure=ALL)
# evc_gratton.show_graph(show_model_based_optimizer=True, show_node_structure=ALL)

# evc_gratton.run(inputs=stim_list_dict)

