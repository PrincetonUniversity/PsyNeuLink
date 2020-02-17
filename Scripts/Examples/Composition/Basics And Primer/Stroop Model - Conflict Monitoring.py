from psyneulink import *
import numpy as np

# CONSTRUCT THE MODEL ***********************************

# Construct the color naming pathway:
color_input = ProcessingMechanism(name='COLOR INPUT', size=2) # Note:  default function is Linear
color_input_to_hidden_wts = np.array([[2, -2], [-2, 2]])
color_hidden = ProcessingMechanism(name='COLOR HIDDEN', size=2, function=Logistic(bias=-4))
color_hidden_to_output_wts = np.array([[2, -2], [-2, 2]])
output = ProcessingMechanism(name='OUTPUT', size=2, function=Logistic)
color_pathway = [color_input, color_input_to_hidden_wts, color_hidden, color_hidden_to_output_wts, output]

# Construct the word reading pathway (using the same output_layer)
word_input = ProcessingMechanism(name='WORD INPUT', size=2)
word_input_to_hidden_wts = np.array([[3, -3], [-3, 3]])
word_hidden = ProcessingMechanism(name='WORD HIDDEN', size=2, function=Logistic(bias=-4))
word_hidden_to_output_wts = np.array([[3, -3], [-3, 3]])
word_pathway = [word_input, word_input_to_hidden_wts, word_hidden, word_hidden_to_output_wts, output]

# Construct the task specification pathways
task_input = ProcessingMechanism(name='TASK INPUT', size=2)
task = LCAMechanism(name='TASK', size=2, initial_value=[0.5,0.5])
task_color_wts = np.array([[4,4],[0,0]])
task_word_wts = np.array([[0,0],[4,4]])
task_color_pathway = [task_input, task, task_color_wts, color_hidden]
task_word_pathway = [task_input, task, task_word_wts, word_hidden]

# Construct the decision pathway:
decision = DDM(name='DECISION', input_format=ARRAY)
decision_pathway = [output, decision]

# Construct control mechanism
control = ControlMechanism(name='CONTROL',
                           objective_mechanism=ObjectiveMechanism(name='Conflict Monitor',
                                                                  function=Energy(size=2,
                                                                                  matrix=[[0,-2.5],[-2.5,0]]),
                                                                  monitor=output),
                           default_allocation=[0.5],
                           control_signals=[(GAIN, task)])

# Construct the Composition:
Stroop_model = Composition(name='Stroop Model - Conflict Monitoring')
Stroop_model.add_linear_processing_pathway(color_pathway)
Stroop_model.add_linear_processing_pathway(word_pathway)
Stroop_model.add_linear_processing_pathway(task_color_pathway)
Stroop_model.add_linear_processing_pathway(task_word_pathway)
Stroop_model.add_linear_processing_pathway(decision_pathway)
Stroop_model.add_controller(control)

# Assign conditions:
settling_time = 10
Stroop_model.scheduler.add_condition(color_hidden, EveryNCalls(task, settling_time))
Stroop_model.scheduler.add_condition(word_hidden, EveryNCalls(task, settling_time))
Stroop_model.scheduler.add_condition(output,All(EveryNCalls(color_hidden, 1),
                                                           EveryNCalls(word_hidden, 1)))
Stroop_model.scheduler.add_condition(decision, EveryNCalls(output, 1))

# SHOW_GRAPH ***********************************
Stroop_model.show_graph(show_controller=True,
                        # show_node_structure=ALL
                        )

# RUN THE MODEL *********************************

# Label inputs
red = [1,0]
green = [0,1]
word = [0,1]
color = [1,0]

np.set_printoptions(precision=2)
global t
t = 0
def print_after():
    global t
    print(f'\nEnd of trial {t}:')
    print(f'\t\t\t\tcolor  word')
    print(f'\ttask:\t\t{task.value[0]}')
    print(f'\ttask gain:\t   {task.parameter_ports[GAIN].value}')
    # print(f'\ttask gain:\t   {task.function.parameters.gain}')
    print(f'\t\t\t\tred   green')
    print(f'\toutput:\t\t{output.value[0]}')
    print(f'\tdecision:\t{decision.value[0]}{decision.value[1]}')
    print(f'\tconflict:\t  {control.objective_mechanism.value[0]}')
    t += 1

task.log.set_log_conditions(VALUE)
control.log.set_log_conditions(VALUE)

task.initial_value = [0.5,0.5]
num_trials = 4
stimuli = {color_input:[red] * num_trials,
           word_input:[green] * num_trials,
           task_input:[color] * num_trials}

Stroop_model.run(inputs=stimuli,
                 # animate={'show_controller':True,
                 #          # 'show_cim':True
                 #          },
                 call_after_trial=print_after)

Stroop_model.log.print_entries(display=[TIME, VALUE])
