from psyneulink import *

prey_len = 5
predator_len = 5

prey = TransferMechanism(size=prey_len, noise=NormalDist(), name="PREY")
predator = TransferMechanism(size=predator_len, noise=NormalDist(), name="PREDATOR")

motor = ProcessingMechanism(name='MOTOR OUTPUT')


def objective_function(x):
    return x

ocm = OptimizationControlMechanism(name='EVC',
                                   features=[prey, predator],
                                   objective_mechanism=ObjectiveMechanism(name='Performance ObjectiveMechanism',
                                                                          monitored_output_states=motor,
                                                                          function=objective_function),
                                   terminal_objective_mechanism=True,
                                   function=GridSearch,
                                   # function=??SAMPLING FUNCTION HERE??,
                                   control_signals=ControlSignal(projections=[(NOISE, prey),
                                                                              (NOISE, predator)],
                                                                 # function=Logistic,
                                                                 # cost_options=[ControlSignalCosts.INTENSITY,
                                                                 #               ControlSignalCosts.ADJUSTMENT],
                                                                 # intensity_cost_function=Exponential(rate=0.25, bias=-3),
                                                                 # adjustment_cost_function=Exponential(rate=0.25, bias=-3),
                                                                 allocation_samples=[i/2 for i in list(range(0,50,1))])
                                   )

c = Composition(name='PREDATOR-PREY COMPOSITION')
c.add_c_node(prey)
c.add_c_node(predator)
c.add_c_node(motor)
c.add_c_node(ocm)
c.add_projection(sender=prey, receiver=motor)
c.add_projection(sender=predator, receiver=motor)

# c._analyze_graph()
c.show_graph(show_control=True)

# print(p.execute([[5,0,100,0,0],[1,0,0,0,0]]))
