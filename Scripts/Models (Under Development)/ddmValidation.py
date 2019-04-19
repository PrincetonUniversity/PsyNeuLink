import psyneulink as pnl

Drift = 1 # Drift Rate
SP = 0.0 # Starting Point
Z = 0.0475 # Threshold
C = 0.04 # Noise
T0 = 0.2 # T0

# Create Stimuli Info 1 layer
stimulusInfo1 = pnl.TransferMechanism(size=1,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  name = 'S1 Input')
# Create Stimuli Info 2 layer
stimulusInfo2 = pnl.TransferMechanism(size=1,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  name = 'S2 Input')
# Create Task Layer 1
taskInfo1 = pnl.TransferMechanism(size=1,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  name = 'act1')

# Create Task Layer 2
taskInfo2 = pnl.TransferMechanism(size=1,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  name = 'act2')

# Create layer to integrate the DDM inputs
controlTask1 = pnl.TransferMechanism(size = 1,
                                  function=pnl.Linear(slope=1, intercept= 0),
                                  input_states=pnl.InputState(combine=pnl.PRODUCT),
                                  name = 'S1 Act1')

# Create layer to integrate the DDM inputs
controlTask2 = pnl.TransferMechanism(size = 1,
                                  function=pnl.Linear(slope=1, intercept= 0),
                                  input_states=pnl.InputState(combine=pnl.PRODUCT),
                                  name = 'S2 Act2')

# Integrate inputs
DDM_InputIntegrator = pnl.TransferMechanism(size = 1,
                                  function=pnl.Linear(slope=1, intercept= 0),
                                  #input_states=pnl.InputState(combine=pnl.SUM),
                                  name = 'DDM Input Mechanism')


decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate = Drift,
                                                              starting_point = SP,
                                                              threshold = Z,
                                                              noise = C,
                                                              t0 = T0),
                        output_states=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
                                                  pnl.PROBABILITY_UPPER_THRESHOLD,
                                                  pnl.PROBABILITY_LOWER_THRESHOLD],
                        name = 'DDM ')
decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
                                  pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])


decisionModule = pnl.Composition()

decisionModule.add_node(stimulusInfo1)
decisionModule.add_node(stimulusInfo2)
decisionModule.add_node(taskInfo1)
decisionModule.add_node(taskInfo2)
decisionModule.add_node(controlTask1)
decisionModule.add_node(controlTask2)
decisionModule.add_node(DDM_InputIntegrator)
decisionModule.add_node(decisionMaker)

decisionModule.add_projection(sender=taskInfo1, receiver=controlTask1)
decisionModule.add_projection(sender=taskInfo2, receiver=controlTask2)
decisionModule.add_projection(sender=stimulusInfo1, receiver=controlTask1)
decisionModule.add_projection(sender=stimulusInfo2, receiver=controlTask2)

decisionModule.add_projection(sender=stimulusInfo1, receiver=DDM_InputIntegrator)
decisionModule.add_projection(sender=stimulusInfo2, receiver=DDM_InputIntegrator)
decisionModule.add_projection(sender=controlTask1, receiver=DDM_InputIntegrator)
decisionModule.add_projection(sender=controlTask2, receiver=DDM_InputIntegrator)

decisionModule.add_projection(sender=DDM_InputIntegrator, receiver=decisionMaker)

decisionModule.show_graph()

act1 = [[0.52248482], [0.58757666], [0.58246114], [0.58122223]] # Color Task
act2 = [[0.5], [0.48576476], [0.49170455], [0.49304476]] # Motion Task
S1 = [[1], [1], [1], [1]]  # Color coherence
S2 = [[-1], [-1], [-1], [-1]]  # Motion coherence


inputs = {taskInfo1: act1, taskInfo2: act2, stimulusInfo1: S1, stimulusInfo2: S2}
results = decisionModule.run(inputs)

print(results)
decisionMaker.log.print_entries()


