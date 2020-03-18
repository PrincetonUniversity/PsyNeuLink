import psyneulink as pnl
import numpy as np
import pytest

cueInterval = pnl.TransferMechanism(default_variable=[[0.0]],
                                    size=1,
                                    function=pnl.Linear(slope=1, intercept=0),
                                    output_ports=[pnl.RESULT],
                                    name='Cue-Stimulus Interval')

taskLayer = pnl.TransferMechanism(default_variable=[[0.0, 0.0]],
                                  size=2,
                                  function=pnl.Linear(slope=1, intercept=0),
                                  output_ports=[pnl.RESULT],
                                  name='Task Input [I1, I2]')

activation = pnl.LCAMechanism(default_variable=[[0.0, 0.0]],
                              size=2,
                              function=pnl.Logistic(gain=1),
                              leak=.5,
                              competition=2,
                              noise=0,
                              time_step_size=.1,
                              termination_measure=pnl.TimeScale.TRIAL,
                              termination_threshold=3,
                              name='Task Activations [Act 1, Act 2]')

# response = pnl.ProcessingMechanism()

# Create controller
csiController = pnl.ControlMechanism(
        # objective_mechanism=pnl.ObjectiveMechanism(monitor=[cueInterval]),
        monitor_for_control=cueInterval,
        control_signals=[(pnl.TERMINATION_THRESHOLD, activation)],
        modulation=pnl.OVERRIDE
)

comp = pnl.Composition(
        # controller_mode=pnl.BEFORE
)
# comp.add_node(cueInterval)
# comp.add_node(taskLayer)
# comp.add_node(activation)
# comp.add_projection(sender=taskLayer, receiver=activation)
# comp.add_node(csiController)
# comp.add_controller(csiController)
# comp.scheduler.add_condition(activation,pnl.WhenFinished(csiController))
# comp.enable_controller=True
comp.add_linear_processing_pathway(pathway=[taskLayer,activation])
comp.add_node(cueInterval)
comp.add_node(csiController)

cueInterval.set_log_conditions([pnl.VALUE])
activation.set_log_conditions([pnl.RESULT, 'termination_threshold'])
csiController.set_log_conditions([pnl.VALUE])
taskLayer.set_log_conditions([pnl.VALUE])

# csiController.control_signals[0].set_log_conditions([pnl.VALUE])


def print_ctl(context):
    print('\ncontrol_signal:', csiController.control_signals[0].parameters.value.get(context))
    print('parameter_port:', activation.parameter_ports['termination_threshold'].parameters.value.get(context))
    print('mod_term_thresh:', activation.get_mod_termination_threshold(context)),
    print('term_thresh w/ context:', activation.parameters.termination_threshold.get(context))

comp.show_graph()
inputs = {taskLayer: [[1, 0], [1, 0], [1, 0], [1, 0]],
          cueInterval: [[1], [5], [1], [5]]}

comp.run(inputs, bin_execute=False, call_after_trial=print_ctl)

activation.log.print_entries()
csiController.log.print_entries()
cueInterval.log.print_entries()
taskLayer.log.print_entries()
# csiController.control_signals[0].log.print_entries()