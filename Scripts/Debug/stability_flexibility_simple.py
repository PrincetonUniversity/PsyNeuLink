import psyneulink as pnl
import numpy as np

def computeAccuracy(variable):

	# variable is the list of values given by the monitored output ports in the Objective Mechanism

	print("\n\nInputs to ComputeAccuracy Function: ", variable)

	taskInputs = variable[0]
	stimulusInputs = variable[1]
	upperThreshold = variable[2]
	lowerThreshold = variable[3]

	accuracy = []
	colorTrial = (taskInputs[0] == 1)
	motionTrial = (taskInputs[1] == 1)

	# during color trials

	if colorTrial:
		# if the correct answer is the upper threshold
		if stimulusInputs[0] == 1:
			accuracy.append(upperThreshold)
			print('Color Trial: 1')

		# if the correct answer is the lower threshold
		elif stimulusInputs[0] == -1:
			accuracy.append(lowerThreshold)
			print('Color Trial: -1')

	if motionTrial:
		# if the correct answer is the upper threshold
		if stimulusInputs[1] == 1:
			accuracy.append(upperThreshold)
			print('Motion Trial: 1')

		# if the correct answer is the lower threshold
		elif stimulusInputs[1] == -1:
			accuracy.append(lowerThreshold)
			print('Motion Trial: -1')

	# added in after original "concept" for function to account for simulations where the variable does pass in inputs
	# as expected, which is exactly the problem we're currently trying to solve
	if len(accuracy) == 0:
		accuracy = [0]
		# print('No Input')

	print("accuracy = ", accuracy)
	return [accuracy]

##### BEGIN STABILITY FLEXIBILITY MODEL CONSTRUCTION

tau = 0.9  # time constant
g = 1

# testing input
INPUT = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]
stimulusInput = [[1, -1], [1, -1], [1, -1], [1, -1], [1, -1], [1, -1]]
runs = len(INPUT)

excitatoryWeight = np.asarray([[1]])
inhibitoryWeight = np.asarray([[-1]])
gain = np.asarray([[g]])

DRIFT = 1  # Drift Rate
STARTING_POINT = 0.0  # Starting Point
THRESHOLD = 0.0475  # Threshold
NOISE = 0.04  # Noise
T0 = 0.2  # T0

# first element is color task attendance, second element is motion task attendance
inputLayer = pnl.TransferMechanism(  # default_variable=[[0.0, 0.0]],
	size=2,
	function=pnl.Linear(slope=1, intercept=0),
	output_ports=[pnl.RESULT],
	name='Input')
inputLayer.set_log_conditions([pnl.RESULT])

# Recurrent Transfer Mechanism that models the recurrence in the activation between the two stimulus and action
# dimensions. Positive self excitation and negative opposite inhibition with an integrator rate = tau
# Modulated variable in simulations is the GAIN variable of this mechanism
activation = pnl.RecurrentTransferMechanism(default_variable=[[0.0, 0.0]],
											function=pnl.Logistic(gain=1.0),
											matrix=[[1.0, -1.0],
													[-1.0, 1.0]],
											integrator_mode=True,
											integrator_function=pnl.AdaptiveIntegrator(rate=(tau)),
											initial_value=np.array([[0.0, 0.0]]),
											output_ports=[pnl.RESULT],
											name='Activity')

activation.set_log_conditions([pnl.RESULT, "mod_gain"])

stimulusInfo = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
									 size=2,
									 function=pnl.Linear(slope=1, intercept=0),
									 output_ports=[pnl.RESULT],
									 name="Stimulus Info")

stimulusInfo.set_log_conditions([pnl.RESULT])

controlledElement = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                          size=2,
                                          function=pnl.Linear(slope=1, intercept=0),
                                          input_ports=pnl.InputPort(combine=pnl.PRODUCT),
                                          output_ports=[pnl.RESULT],
                                          name='Stimulus Info * Activity')

controlledElement.set_log_conditions([pnl.RESULT])

ddmCombination = pnl.TransferMechanism(size=1,
									   function=pnl.Linear(slope=1, intercept=0),
									   output_ports=[pnl.RESULT],
									   name="DDM Integrator")
ddmCombination.set_log_conditions([pnl.RESULT])

decisionMaker = pnl.DDM(function=pnl.DriftDiffusionAnalytical(drift_rate=DRIFT,
															  starting_point=STARTING_POINT,
															  threshold=THRESHOLD,
															  noise=NOISE,
															  t0=T0),

						output_ports=[pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME,
									   pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD],
						name='DDM')

decisionMaker.set_log_conditions([pnl.PROBABILITY_UPPER_THRESHOLD, pnl.PROBABILITY_LOWER_THRESHOLD,
								  pnl.DECISION_VARIABLE, pnl.RESPONSE_TIME])

########### Composition

stabilityFlexibility = pnl.Composition()

### NODE CREATION

stabilityFlexibility.add_node(inputLayer)
stabilityFlexibility.add_node(activation)
stabilityFlexibility.add_node(controlledElement)
stabilityFlexibility.add_node(stimulusInfo)
stabilityFlexibility.add_node(ddmCombination)
stabilityFlexibility.add_node(decisionMaker)

stabilityFlexibility.add_projection(sender=inputLayer, receiver=activation)
stabilityFlexibility.add_projection(sender=activation, receiver=controlledElement)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=controlledElement)
stabilityFlexibility.add_projection(sender=stimulusInfo, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=controlledElement, receiver=ddmCombination)
stabilityFlexibility.add_projection(sender=ddmCombination, receiver=decisionMaker)

# beginning of Controller

search_range = pnl.SampleSpec(start=0.1, stop=1.0, num=10)

signal = pnl.ControlSignal(modulates=[(pnl.GAIN, activation)],
						   function=pnl.Linear,
						   variable=1.0,
						   allocation_samples=search_range)

objective_mech = pnl.ObjectiveMechanism(monitor=[inputLayer, stimulusInfo,
												 (pnl.PROBABILITY_UPPER_THRESHOLD, decisionMaker),
												 (pnl.PROBABILITY_LOWER_THRESHOLD, decisionMaker)],
										function=computeAccuracy)

meta_controller = pnl.OptimizationControlMechanism(agent_rep=stabilityFlexibility,
												   features=[inputLayer.input_port, stimulusInfo.input_port],
												   feature_function=pnl.Buffer(history=100),
												   objective_mechanism=objective_mech,
												   function=pnl.GridSearch(),
												   control_signals=[signal])

inputs = {inputLayer: INPUT, stimulusInfo: stimulusInput}
stabilityFlexibility.add_model_based_optimizer(meta_controller)
stabilityFlexibility.enable_model_based_optimizer = True

# stabilityFlexibility.show_graph(show_node_structure=[pnl.FUNCTIONS,
# 													 pnl.PORT_FUNCTION_PARAMS,
# 													 pnl.MECH_FUNCTION_PARAMS],
# 								show_controller=True)
stabilityFlexibility.show_graph(show_node_structure=pnl.ALL, show_controller=True, show_cim=True)


# print("Beginning of Run")
# for i in range(1, len(stabilityFlexibility.model_based_optimizer.input_ports)):
# 	stabilityFlexibility.model_based_optimizer.input_ports[i].function.reset()
#
# 	stabilityFlexibility.run(inputs)
#
# 	activation.log.print_entries()