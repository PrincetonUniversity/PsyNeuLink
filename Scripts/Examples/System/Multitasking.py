import psyneulink as pnl
import psyneulink.core.components.functions.transferfunctions

stimulus_layer = pnl.TransferMechanism(size=4)
task_layer = pnl.TransferMechanism(size=4)
hidden_layer = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions.Logistic)
output_layer = pnl.TransferMechanism(size=4, function=psyneulink.core.components.functions.transferfunctions.Logistic)

network_process = pnl.Process(pathway=[stimulus_layer, hidden_layer, output_layer])
hidden_control_process = pnl.Process(pathway=[task_layer, hidden_layer])
output_control_process = pnl.Process(pathway=[task_layer, output_layer])

multitasking_system = pnl.System(processes=[network_process, hidden_control_process, output_control_process])

# WEIGHTS TO COME FROM SEBASTIAN

example_stimulus_inputs = [[1,0,0,1],[1,0,1,0]]
example_task_inputs = [[0,0,0,1],[1,0,0,0]]
example_training_pattern = [[0,0,0,1],[1,0,0,0]]

# RUN THIS TO GET SPACE OF INPUTS ON WHICH TO OPTIMIZE LCAMechanism PARAMS:
inputs_to_LCA = multitasking_system.run(inputs={stimulus_layer:example_stimulus_inputs,
                                                task_layer:example_task_inputs})

# SOME PYTHON ALGORITHM HERE THAT SELECTS THE 2-UNIT SUBVECTOR FROM inputs_to_LCA CORRESPONDING TO THE RELEVANT TASK
# AS INPUT TO optimization_system BELOW, AND THEN RUN THE SYSTEM FOR EACH INPUT, USING EVC TO OPTIMIZE LCAMechanism PARAMETERS
#  FOR EACH, BASED CONTROL PARAMETERS AND OBJECTIVE FUNCTION

input_layer = pnl.TransferMechanism(size=2)
decision_layer = pnl.LCAMechanism(size=2,
                                  # INCLUDE TERMINATION CONDITION USING THREHSOLD = ControlSignal)
                                  )
decision_process = pnl.Process(pathway=[input_layer, decision_layer])
optimization_system = pnl.System(processes=[decision_process],
                                 monitor_for_control=[decision_layer.output_ports[pnl.RESULT]])
# ADD COMPARATOR MECHANISM FOR ACCURACY DETERMINATION


# EVC PARAMS:
# - number of simulations to run per LCAMechanism
# - which inputs to provide (i.e., *NOT* using typical PredictionMechanisms)
