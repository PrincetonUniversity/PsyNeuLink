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

comp = pnl.Composition(controller_mode=pnl.BEFORE)
comp.add_node(cueInterval)
comp.add_node(taskLayer)
comp.add_node(activation)
comp.add_projection(sender=taskLayer, receiver=activation)

activation.set_log_conditions([pnl.RESULT])

# Create controller
csiController = pnl.ControlMechanism(objective_mechanism=pnl.ObjectiveMechanism(monitor=[cueInterval]),
                                     control_signals=[(pnl.TERMINATION_THRESHOLD, activation)],
                                     # modulation=pnl.OVERRIDE
                                     )

# comp.add_controller(csiController)
comp.add_node(csiController)
comp.enable_controller=True

def print_ctl(context):
    print('\ncontrol_signal:', csiController.control_signals[0].value)
    print('parameter_port:', activation.parameter_ports['termination_threshold'].value)
    print('mod_term_thresh:', activation.mod_termination_threshold),
    print('term_thresh w/ context:', activation.parameters.termination_threshold.get(context))

inputs = {taskLayer: [[1, 0], [1, 0], [1, 0]],
          cueInterval: [[1], [5], [1]]}

comp.run(inputs, bin_execute=False, call_after_trial=print_ctl)

activation.log.print_entries()