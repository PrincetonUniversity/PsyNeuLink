import psyneulink as pnl

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
        monitor_for_control=cueInterval,
        control_signals=[(pnl.TERMINATION_THRESHOLD, activation)],
        modulation=pnl.OVERRIDE
)

comp = pnl.Composition(
        # controller_mode=pnl.BEFORE
)
comp.add_linear_processing_pathway(pathway=[taskLayer,activation])
comp.add_node(cueInterval)
comp.add_node(csiController)
# csiController.control_signals[0].set_log_conditions([pnl.VALUE])

# comp.show_graph()


cueInterval.set_log_conditions([pnl.VALUE])
activation.set_log_conditions([pnl.RESULT, 'mod_termination_threshold'])
csiController.set_log_conditions([pnl.VALUE])
taskLayer.set_log_conditions([pnl.VALUE])


inputs = {taskLayer: [[1, 0], [1, 0], [1, 0], [1, 0]],
          cueInterval: [[1], [5], [1], [5]]}

comp.run(inputs, bin_execute=False)

activation.log.print_entries()
csiController.log.print_entries()
cueInterval.log.print_entries()
taskLayer.log.print_entries()
# csiController.control_signals[0].log.print_entries()
