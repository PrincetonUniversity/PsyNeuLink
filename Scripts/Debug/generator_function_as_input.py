import psyneulink as pnl

# instantiate mechanisms and inner comp
ia = pnl.TransferMechanism(name='ia')
ib = pnl.TransferMechanism(name='ib')
ic = pnl.TransferMechanism(name='ic')
icomp = pnl.Composition(name='icomp', controller_mode=pnl.BEFORE)

# set up structure of inner comp
icomp.add_node(ia, required_roles=pnl.NodeRole.INPUT)
icomp.add_node(ib, required_roles=pnl.NodeRole.INPUT)
icomp.add_node(ic, required_roles=pnl.NodeRole.OUTPUT)
icomp.add_projection(pnl.MappingProjection(), sender=ia, receiver=ic)
icomp.add_projection(pnl.MappingProjection(), sender=ib, receiver=ic)

# set up inner comp controller and add to comp
icomp.add_controller(
        pnl.OptimizationControlMechanism(
                agent_rep=icomp,
                features=[ia.input_port, ib.input_port],
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                        monitor=ic.output_port,
                        function=pnl.SimpleIntegrator,
                        name="iController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
)

# instantiate outer comp
ocomp = pnl.Composition(name='ocomp', controller_mode=pnl.BEFORE)

# setup structure for outer comp
ocomp.add_node(icomp)

# add controller to outer comp
ocomp.add_controller(
        pnl.OptimizationControlMechanism(
                agent_rep=ocomp,
                features=[ia.input_port, ib.input_port],
                name="Controller",
                objective_mechanism=pnl.ObjectiveMechanism(
                        monitor=ic.output_port,
                        function=pnl.SimpleIntegrator,
                        name="oController Objective Mechanism"
                ),
                function=pnl.GridSearch(direction=pnl.MAXIMIZE),
                control_signals=[pnl.ControlSignal(projections=[(pnl.SLOPE, ia)],
                                                   variable=1.0,
                                                   intensity_cost_function=pnl.Linear(slope=0.0),
                                                   allocation_samples=pnl.SampleSpec(start=1.0,
                                                                                     stop=5.0,
                                                                                     num=5))])
)

# set up input using three different formats:
#  1) generator function
#  2) instance of generator function
#  3) inputs dict
def inputs_generator_function():
    for i in range(2):
        yield {
            icomp:
                {
                    ia: 5,
                    ib: 5
                }
        }
inputs_generator_instance = inputs_generator_function()
inputs_dict = {
    icomp:
        {
            ia: [[5],[5]],
            ib: [[5],[5]]
        }
}

# run Composition with all three input types and assert that results are equal.
# NOTE: in some cases, results using the generator function input type may differ
# from the results of the generator or input dict type.
results_generator_function = ocomp.run(inputs=inputs_generator_function)
results_generator_instance = ocomp.run(inputs=inputs_generator_instance)
results_dict = ocomp.run(inputs=inputs_dict)
assert results_generator_function == results_generator_instance == results_dict == [[130]]
