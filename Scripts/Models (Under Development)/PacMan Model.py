from psyneulink import *

prey_len = 5
predator_len = 5

prey = TransferMechanism(size=prey_len, function=Gaussian(variance=1), name="PREY")
predator = TransferMechanism(size=predator_len, function=Gaussian(variance=1), name="PREDATOR")

motor = ProcessingMechanism(name='MOTOR OUTPUT')


# SYSTEM VERSION ----------------------------------------------------------------------------
p_prey = Process(pathway=[prey, motor], name='PREY PROCESS')
p_predator = Process(pathway=[predator, motor], name='PREDATOR PROCESS')
s = System(processes=[p_prey, p_predator],
           controller=EVCControlMechanism(monitor_for_control=motor,
                                          control_signals=[(VARIANCE, prey),
                                                           (OFFSET,prey),
                                                           (VARIANCE, predator)],
                                          name='EVC'),
           name='PACMAN SYSTEM')
s.show_graph(show_control=True)

# COMPOSITION VERSION ----------------------------------------------------------------------

def objective_function(x):
    return x

# svoc = LVOCControlMechanism(name='EVC',
#                             feature_predictors=[prey, predator],
#                             objective_mechanism=ObjectiveMechanism(name='LVOC ObjectiveMechanism',
#                                                                    monitored_output_states=motor,
#                                                                    function=objective_function),
#                             terminal_objective_mechanism=True,
#                             function=GridSearch,
#                             # function=??SAMPLING FUNCTION HERE??,
#                             control_signals=ControlSignal(projections=[(VARIANCE, prey),
#                                                                        (VARIANCE, predator)],
#                                                           # function=Logistic,
#                                                           cost_options=[ControlSignalCosts.INTENSITY,
#                                                                         ControlSignalCosts.ADJUSTMENT],
#                                                           intensity_cost_function=Exponential(rate=0.25, bias=-3),
#                                                           adjustment_cost_function=Exponential(rate=0.25, bias=-3),
#                                                           allocation_samples=[i/2 for i in list(range(0,50,1))]))

evc = EVCControlMechanism(objective_mechanism=ObjectiveMechanism(monitored_output_states=motor,
                                                                 # function=
                                                                 ),
                          control_signals=[ControlSignal(projections=(VARIANCE, prey),
                                                         allocation_samples=[0,1,2]),
                                           (OFFSET,prey),
                                           (VARIANCE, predator)],
                          function=GridSearch,
                          name='EVC'),

c = Composition(name='PACMAN COMPOSITION')
c.add_c_node(prey)
c.add_c_node(predator)
c.add_c_node(motor)
c.add_c_node(svoc)
c.add_projection(sender=prey, receiver=motor)
c.add_projection(sender=predator, receiver=motor)

c._analyze_graph()
c.show_graph(show_control=True)

# print(p.execute([[5,0,100,0,0],[1,0,0,0,0]]))
