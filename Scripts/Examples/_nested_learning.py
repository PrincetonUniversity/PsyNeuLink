from psyneulink import *

ia = ProcessingMechanism(name='INNER INPUT')
ib = ProcessingMechanism(name='INNER OUTPUT')
input_mech = ProcessingMechanism(name='OUTER INPUT')
internal_mech = ProcessingMechanism(name='INTERNAL')
output_mech = ProcessingMechanism(name='OUTER OUTPUT')
ctl_mech = ControlMechanism(name='CONTROL',
                            control=[(SLOPE, input_mech),
                                     (SLOPE, output_mech)])
target = ProcessingMechanism(name='TARGET')
icomp = Composition(name="NESTED COMPOSITION")
p = icomp.add_backpropagation_learning_pathway(pathway=[ia, ib])
ocomp = Composition(name='COMPOSITION',
                    pathways=[
                        [input_mech,
                        internal_mech,
                        icomp,
                        output_mech],
                        # [target,p.target]
                    ],
                    controller=OptimizationControlMechanism(
                        name='CONTROLLER',
                        objective_mechanism=ObjectiveMechanism(name='OBJECTIVE MECHANISM',
                                                               monitor=[input_mech,
                                                                        output_mech]),
                        control=(SLOPE, internal_mech))
                    )

ocomp.show_graph(show_nested=NESTED,
                 show_learning=True,
                 # show_cim=True
                 )