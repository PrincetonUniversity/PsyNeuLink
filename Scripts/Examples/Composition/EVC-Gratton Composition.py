import psyneulink as pnl
import numpy as np

target_stim = pnl.TransferMechanism(name='Target Stimulus',
                                    function=pnl.Linear(slope=0.3324))
flanker_stim = pnl.TransferMechanism(name='Flanker Stimulus',
                                     function=pnl.Linear(slope=0.3545221843))

# Processing Mechanisms (Control)
Target_Rep = pnl.TransferMechanism(name='Target Representation')
Flanker_Rep = pnl.TransferMechanism(name='Flanker Representation')

# Processing Mechanism (Automatic)
Automatic_Component = pnl.TransferMechanism(name='Automatic Component')

# Decision Mechanism
Decision = pnl.DDM(name='Decision',
                   function=pnl.DriftDiffusionAnalytical(drift_rate=(1.0),
                                                         threshold=(0.2645),
                                                         noise=(0.5),
                                                         starting_point=(0),
                                                         t0=0.15),
                   output_ports=[pnl.DECISION_VARIABLE,
                                  pnl.RESPONSE_TIME,
                                  pnl.PROBABILITY_UPPER_THRESHOLD]
                   )

# Outcome Mechanism
reward = pnl.TransferMechanism(name='reward')

# Pathways
target_control_pathway = [target_stim, Target_Rep, Decision]
flanker_control_pathway = [flanker_stim, Flanker_Rep, Decision]
target_automatic_pathway = [target_stim, Automatic_Component, Decision]
flanker_automatic_pathway = [flanker_stim, Automatic_Component, Decision]
pathways = [target_control_pathway, flanker_control_pathway, target_automatic_pathway,
            flanker_automatic_pathway]

# Composition
evc_gratton = pnl.Composition(name="EVCGratton")
evc_gratton.add_node(Decision, required_roles=pnl.NodeRole.OUTPUT)
for path in pathways:
    evc_gratton.add_linear_processing_pathway(path)
evc_gratton.add_node(reward, required_roles=pnl.NodeRole.OUTPUT)

# Control Signals
signalSearchRange = pnl.SampleSpec(start=1.0, stop=1.8, step=0.2)

target_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Target_Rep)],
                                              variable=1.0,
                                              intensity_cost_function=pnl.Exponential(rate=0.8046),
                                              allocation_samples=signalSearchRange)

flanker_rep_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, Flanker_Rep)],
                                               variable=1.0,
                                               intensity_cost_function=pnl.Exponential(rate=0.8046),
                                               allocation_samples=signalSearchRange)

objective_mech = pnl.ObjectiveMechanism(function=pnl.LinearCombination(operation=pnl.PRODUCT),
                                        monitor=[reward,
                                                                 (Decision.output_ports[
                                                                      pnl.PROBABILITY_UPPER_THRESHOLD], 1, -1)])
# Model Based OCM (formerly controller)
evc_gratton.add_controller(controller=pnl.OptimizationControlMechanism(agent_rep=evc_gratton,
                                                                                 features=[target_stim.input_port,
                                                                                           flanker_stim.input_port,
                                                                                           reward.input_port],
                                                                                 feature_function=pnl.AdaptiveIntegrator(
                                                                                     rate=1.0),
                                                                                 objective_mechanism=objective_mech,
                                                                                 function=pnl.GridSearch(),
                                                                                 control_signals=[
                                                                                     target_rep_control_signal,
                                                                                     flanker_rep_control_signal]))

evc_gratton.show_graph(show_controller=True)

evc_gratton.enable_controller = True

targetFeatures = [1, 1, 1]
flankerFeatures = [1, -1, 1]
rewardValues = [100, 100, 100]

stim_list_dict = {target_stim: targetFeatures,
                  flanker_stim: flankerFeatures,
                  reward: rewardValues}


evc_gratton.run(inputs=stim_list_dict)

expected_results_array = [[[0.32257752863413636], [0.9481940753514433], [100.]],
                          [[0.42963678062444666], [0.47661180945923376], [100.]],
                          [[0.300291026852769], [0.97089165101931], [100.]]]

expected_sim_results_array = [
    [[0.32257753], [0.94819408], [100.]],
    [[0.31663196], [0.95508757], [100.]],
    [[0.31093566], [0.96110142], [100.]],
    [[0.30548947], [0.96633839], [100.]],
    [[0.30029103], [0.97089165], [100.]],
    [[0.3169957], [0.95468427], [100.]],
    [[0.31128378], [0.9607499], [100.]],
    [[0.30582202], [0.96603252], [100.]],
    [[0.30060824], [0.9706259], [100.]],
    [[0.29563774], [0.97461444], [100.]],
    [[0.31163288], [0.96039533], [100.]],
    [[0.30615555], [0.96572397], [100.]],
    [[0.30092641], [0.97035779], [100.]],
    [[0.2959409], [0.97438178], [100.]],
    [[0.29119255], [0.97787196], [100.]],
    [[0.30649004], [0.96541272], [100.]],
    [[0.30124552], [0.97008732], [100.]],
    [[0.29624499], [0.97414704], [100.]],
    [[0.29148205], [0.97766847], [100.]],
    [[0.28694892], [0.98071974], [100.]],
    [[0.30156558], [0.96981445], [100.]],
    [[0.29654999], [0.97391021], [100.]],
    [[0.29177245], [0.97746315], [100.]],
    [[0.28722523], [0.98054192], [100.]],
    [[0.28289958], [0.98320731], [100.]],
    [[0.42963678], [0.47661181], [100.]],
    [[0.42846471], [0.43938586], [100.]],
    [[0.42628176], [0.40282965], [100.]],
    [[0.42314468], [0.36732207], [100.]],
    [[0.41913221], [0.333198], [100.]],
    [[0.42978939], [0.51176048], [100.]],
    [[0.42959394], [0.47427693], [100.]],
    [[0.4283576], [0.43708106], [100.]],
    [[0.4261132], [0.40057958], [100.]],
    [[0.422919], [0.36514906], [100.]],
    [[0.42902209], [0.54679323], [100.]],
    [[0.42980788], [0.50942101], [100.]],
    [[0.42954704], [0.47194318], [100.]],
    [[0.42824656], [0.43477897], [100.]],
    [[0.42594094], [0.3983337], [100.]],
    [[0.42735293], [0.58136855], [100.]],
    [[0.42910149], [0.54447221], [100.]],
    [[0.42982229], [0.50708112], [100.]],
    [[0.42949608], [0.46961065], [100.]],
    [[0.42813159], [0.43247968], [100.]],
    [[0.42482049], [0.61516258], [100.]],
    [[0.42749136], [0.57908829], [100.]],
    [[0.42917687], [0.54214925], [100.]],
    [[0.42983261], [0.50474093], [100.]],
    [[0.42944107], [0.46727945], [100.]],
    [[0.32257753], [0.94819408], [100.]],
    [[0.31663196], [0.95508757], [100.]],
    [[0.31093566], [0.96110142], [100.]],
    [[0.30548947], [0.96633839], [100.]],
    [[0.30029103], [0.97089165], [100.]],
    [[0.3169957], [0.95468427], [100.]],
    [[0.31128378], [0.9607499], [100.]],
    [[0.30582202], [0.96603252], [100.]],
    [[0.30060824], [0.9706259], [100.]],
    [[0.29563774], [0.97461444], [100.]],
    [[0.31163288], [0.96039533], [100.]],
    [[0.30615555], [0.96572397], [100.]],
    [[0.30092641], [0.97035779], [100.]],
    [[0.2959409], [0.97438178], [100.]],
    [[0.29119255], [0.97787196], [100.]],
    [[0.30649004], [0.96541272], [100.]],
    [[0.30124552], [0.97008732], [100.]],
    [[0.29624499], [0.97414704], [100.]],
    [[0.29148205], [0.97766847], [100.]],
    [[0.28694892], [0.98071974], [100.]],
    [[0.30156558], [0.96981445], [100.]],
    [[0.29654999], [0.97391021], [100.]],
    [[0.29177245], [0.97746315], [100.]],
    [[0.28722523], [0.98054192], [100.]],
    [[0.28289958], [0.98320731], [100.]],
]

for trial in range(len(evc_gratton.results)):
    assert np.allclose(expected_results_array[trial],
                       # Note: Skip decision variable OutputPort
                       evc_gratton.results[trial][1:])
for simulation in range(len(evc_gratton.simulation_results)):
    assert np.allclose(expected_sim_results_array[simulation],
                       # Note: Skip decision variable OutputPort
                       evc_gratton.simulation_results[simulation][1:])