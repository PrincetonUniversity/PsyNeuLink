import psyneulink as pnl
import numpy as np

input_a = pnl.ProcessingMechanism(
    name='oa',
    function=pnl.Linear(slope=1)
)
input_b = pnl.ProcessingMechanism(
    name='ob',
    function=pnl.Linear(slope=1)
)
output = pnl.ProcessingMechanism(name='oc')
comp = pnl.Composition(
    name='ocomp',
    controller_mode=pnl.BEFORE,
    retain_old_simulation_data=True
)
comp.add_linear_processing_pathway([input_a, output])
comp.add_linear_processing_pathway([input_b, output])
comp.add_controller(
    pnl.OptimizationControlMechanism(
        agent_rep=comp,
        features=[input_b.input_port, input_a.input_port],
        name="Controller",
        objective_mechanism=pnl.ObjectiveMechanism(
            monitor=output.output_port,
            function=pnl.SimpleIntegrator,
            name="Output Objective Mechanism"
        ),
        function=pnl.GridSearch(direction=pnl.MAXIMIZE),
        control_signals=[
            pnl.ControlSignal(modulates=[(pnl.SLOPE, input_a)],
                              intensity_cost_function=pnl.Linear(slope=0),
                              allocation_samples=[-1, 1]),
            pnl.ControlSignal(modulates=[(pnl.SLOPE, input_b)],
                              intensity_cost_function=pnl.Linear(slope=0),
                              allocation_samples=[-1, 1])
        ]
    )
)
results = comp.run(
    inputs={
        input_a: [[5]],
        input_b: [[-2]]
    },
    bin_execute=True
)

# The controller of this model uses two control signals: one that modulates the slope of input_a and one that modulates
# the slope of input_b. Both control signals have two possible values: -1 or 1.
#
# In the correct case, input_a receives a control signal with value 1 and input_b receives a control signal with value
# -1 to maximize the output of the model given their respective input values of 5 and -2.
#
# In the errant case, the control signals are flipped so that input_b receives a control signal with value -1 and
# input_a receives a control signal with value 1.
#
# Thus, in the correct case, the output of the model is 7 ((5*1)+(-2*-1)) and in the errant case the output of the model is
# -7 ((5*-1)+(-2*1))

assert np.allclose(results, [[7]])