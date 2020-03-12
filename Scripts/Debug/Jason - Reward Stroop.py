import psyneulink as pnl

unit_noise_std=.01
dec_noise_std=.1
integration_rate = 1
# n_conditions = len(CONDITIONS)
# n_tasks = len(TASKS)
# n_colors = len(COLORS)
N_UNITS = 2

def object_function(x):
    return (x[0] * x[2] - x[1] * x[3])/(x[4])

def power_func(input=1,power=2):
    return input ** power

hidden_func = pnl.Logistic(gain=1.0, x_0=4.0)

# input layer, color and word
reward = pnl.TransferMechanism(name='reward')

punish = pnl.TransferMechanism(name='punish')

inp_clr = pnl.TransferMechanism(
    size=N_UNITS, function=pnl.Linear, name='COLOR INPUT'
)
inp_wrd = pnl.TransferMechanism(
    size=N_UNITS, function=pnl.Linear, name='WORD INPUT'
)
# task layer, represent the task instruction; color naming / word reading
inp_task = pnl.TransferMechanism(
    size=N_UNITS, function=pnl.Linear, name='TASK'
)
# hidden layer for color and word
hid_clr = pnl.TransferMechanism(
    size=N_UNITS,
    function=hidden_func,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='COLORS HIDDEN'
)
hid_wrd = pnl.TransferMechanism(
    size=N_UNITS,
    function=hidden_func,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='WORDS HIDDEN'
)
# output layer
output = pnl.TransferMechanism(
    size=N_UNITS,
    function=pnl.Logistic,
    integrator_mode=True,
    integration_rate=integration_rate,
    noise=pnl.NormalDist(standard_deviation=unit_noise_std).function,
    name='OUTPUT'
)
# decision layer, some accumulator

signalSearchRange = pnl.SampleSpec(start=0.05, stop=5, step=0.05)

decision = pnl.DDM(name='Decision',
                   input_format=pnl.ARRAY,
                   function=pnl.DriftDiffusionAnalytical(drift_rate=1,
                                                         threshold =1,
                                                         noise=1,
                                                         starting_point=0,
                                                         t0=0.35),
                   output_ports=[pnl.RESPONSE_TIME,
                                 pnl.PROBABILITY_UPPER_THRESHOLD,
                                 pnl.PROBABILITY_LOWER_THRESHOLD]
                   )

driftrate_control_signal = pnl.ControlSignal(projections=[(pnl.SLOPE, inp_clr)],
                                             variable=1.0,
                                             # intensity_cost_function=pnl.Exponential(rate=1),#pnl.Exponential(rate=0.8),#pnl.Exponential(rate=1),
                                             intensity_cost_function=power_func,
                                             allocation_samples=signalSearchRange)


threshold_control_signal = pnl.ControlSignal(projections=[(pnl.THRESHOLD, decision)],
                                             variable=1.0,
                                             intensity_cost_function=pnl.Linear(slope=0),
                                             allocation_samples=signalSearchRange)



objective_mech = pnl.ObjectiveMechanism(function=object_function,
                                        monitor=[reward,
                                                 punish,
                                                 decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                                                 decision.output_ports[pnl.PROBABILITY_LOWER_THRESHOLD],
                                                 (decision.output_ports[pnl.RESPONSE_TIME])])



# PROJECTIONS, weights copied from cohen et al (1990)
wts_clr_ih = pnl.MappingProjection(
    matrix=[[2.2, -2.2], [-2.2, 2.2]], name='COLOR INPUT TO HIDDEN')
wts_wrd_ih = pnl.MappingProjection(
    matrix=[[2.6, -2.6], [-2.6, 2.6]], name='WORD INPUT TO HIDDEN')
wts_clr_ho = pnl.MappingProjection(
    matrix=[[1.3, -1.3], [-1.3, 1.3]], name='COLOR HIDDEN TO OUTPUT')
wts_wrd_ho = pnl.MappingProjection(
    matrix=[[2.5, -2.5], [-2.5, 2.5]], name='WORD HIDDEN TO OUTPUT')
wts_tc = pnl.MappingProjection(
    matrix=[[4.0, 4.0], [0, 0]], name='COLOR NAMING')
wts_tw = pnl.MappingProjection(
    matrix=[[0, 0], [4.0, 4.0]], name='WORD READING')


# build the model
model = pnl.Composition(name='STROOP model')


model.add_node(inp_clr)
model.add_node(inp_wrd)
model.add_node(hid_clr)
model.add_node(hid_wrd)
model.add_node(inp_task)
model.add_node(output)
model.add_node(decision, required_roles=pnl.NodeRole.OUTPUT)


model.add_node(reward, required_roles=pnl.NodeRole.OUTPUT)
model.add_node(punish, required_roles=pnl.NodeRole.OUTPUT)


model.add_linear_processing_pathway([inp_clr, wts_clr_ih, hid_clr])
model.add_linear_processing_pathway([inp_wrd, wts_wrd_ih, hid_wrd])
model.add_linear_processing_pathway([hid_clr, wts_clr_ho, output])
model.add_linear_processing_pathway([hid_wrd, wts_wrd_ho, output])
model.add_linear_processing_pathway([inp_task, wts_tc, hid_clr])
model.add_linear_processing_pathway([inp_task, wts_tw, hid_wrd])
model.add_linear_processing_pathway([output, pnl.IDENTITY_MATRIX, decision])

controller = pnl.OptimizationControlMechanism(agent_rep=model,
                                              features=[inp_clr.input_port,
                                                        inp_wrd.input_port,
                                                        inp_task.input_port,
                                                        reward.input_port,
                                                        punish.input_port],
                                              feature_function=pnl.AdaptiveIntegrator(rate=0.1),
                                              objective_mechanism=objective_mech,
                                              function=pnl.GridSearch(),
                                              control_signals=[driftrate_control_signal,
                                                               threshold_control_signal])

model.add_controller(controller=controller)
model.show_graph(show_controller=True)
