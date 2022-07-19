import psyneulink as pnl
import numpy as np

# CONSTRUCT THE MODEL ***********************************

# Construct the color naming pathway:
color_input = pnl.ProcessingMechanism(
    name="color_input", size=2
)  # Note:  default function is Linear
color_input_to_hidden_wts = np.array([[2, -2], [-2, 2]])
color_hidden = pnl.ProcessingMechanism(
    name="color_hidden", size=2, function=pnl.Logistic(bias=-4)
)
color_hidden_to_output_wts = np.array([[2, -2], [-2, 2]])
output = pnl.ProcessingMechanism(name="OUTPUT", size=2, function=pnl.Logistic)
color_pathway = [
    color_input,
    color_input_to_hidden_wts,
    color_hidden,
    color_hidden_to_output_wts,
    output,
]

# Construct the word reading pathway (using the same output_layer)
word_input = pnl.ProcessingMechanism(name="word_input", size=2)
word_input_to_hidden_wts = np.array([[3, -3], [-3, 3]])
word_hidden = pnl.ProcessingMechanism(
    name="word_hidden", size=2, function=pnl.Logistic(bias=-4)
)
word_hidden_to_output_wts = np.array([[3, -3], [-3, 3]])
word_pathway = [
    word_input,
    word_input_to_hidden_wts,
    word_hidden,
    word_hidden_to_output_wts,
    output,
]

# Construct the task specification pathways
task_input = pnl.ProcessingMechanism(name="task_input", size=2)
task = pnl.LCAMechanism(name="TASK", size=2, initial_value=[0.5, 0.5])
task_color_wts = np.array([[4, 4], [0, 0]])
task_word_wts = np.array([[0, 0], [4, 4]])
task_color_pathway = [task_input, task, task_color_wts, color_hidden]
task_word_pathway = [task_input, task, task_word_wts, word_hidden]

# Construct the decision pathway:
decision = pnl.DDM(name="DECISION", input_format=pnl.ARRAY)
decision_pathway = [output, decision]

# Construct control mechanism
control = pnl.ControlMechanism(
    name="CONTROL",
    objective_mechanism=pnl.ObjectiveMechanism(
        name="Conflict Monitor",
        function=pnl.Energy(size=2, matrix=[[0, -2.5], [-2.5, 0]]),
        monitor=output,
    ),
    default_allocation=[0.5],
    control_signals=[(pnl.GAIN, task)],
)

# Construct the Composition:
Stroop_model = pnl.Composition(name="Stroop_model")
Stroop_model.add_linear_processing_pathway(color_pathway)
Stroop_model.add_linear_processing_pathway(word_pathway)
Stroop_model.add_linear_processing_pathway(task_color_pathway)
Stroop_model.add_linear_processing_pathway(task_word_pathway)
Stroop_model.add_linear_processing_pathway(decision_pathway)
Stroop_model.add_controller(control)

# Assign conditions:
settling_time = 10
Stroop_model.scheduler.add_condition(
    color_hidden, pnl.EveryNCalls(task, settling_time)
)
Stroop_model.scheduler.add_condition(
    word_hidden, pnl.EveryNCalls(task, settling_time)
)
Stroop_model.scheduler.add_condition(
    output,
    pnl.All(pnl.EveryNCalls(color_hidden, 1), pnl.EveryNCalls(word_hidden, 1)),
)
Stroop_model.scheduler.add_condition(decision, pnl.EveryNCalls(output, 1))
