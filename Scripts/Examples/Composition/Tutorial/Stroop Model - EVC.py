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
decision = DDM(name='DECISION',
               input_format=ARRAY,
               function=DriftDiffusionAnalytical(drift_rate=(1.0),
                                                 threshold=(0.2645),
                                                 noise=(0.5),
                                                 starting_point=(0),
                                                 t0=0.15),
               output_ports=[DECISION_VARIABLE,
                              RESPONSE_TIME,
                              PROBABILITY_UPPER_THRESHOLD]
               )
decision_pathway = [output, decision]

# Reward Mechanism - should receive input from environment based on DECISION_VARIABLE of decision Mechanism
reward = TransferMechanism(name='reward')

# Construct the Composition:
# Stroop_model = Composition(name='Stroop Model', controller=control)
Stroop_model = Composition(name='Stroop Model - EVC')
Stroop_model.add_linear_processing_pathway(color_pathway)
Stroop_model.add_linear_processing_pathway(word_pathway)
Stroop_model.add_linear_processing_pathway(task_color_pathway)
Stroop_model.add_linear_processing_pathway(task_word_pathway)
Stroop_model.add_linear_processing_pathway(decision_pathway)
Stroop_model.add_node(reward)

# Assign Scheduler Conditions:
settling_time = 1
# scheduler = Scheduler(composition=Stroop_model)
Stroop_model.scheduler.add_condition(color_hidden, EveryNCalls(task, settling_time))
Stroop_model.scheduler.add_condition(word_hidden, EveryNCalls(task, settling_time))

# Construct and add ControlMechanism
control_signal_search_range = SampleSpec(start=1.0, stop=1.8, step=0.2)
evc = OptimizationControlMechanism(name='EVC',
                                   agent_rep=Stroop_model,
                                   features=[color_input.input_port, word_input.input_port, reward.input_port],
                                   feature_function=AdaptiveIntegrator(rate=1.0),
                                   # feature_function=AdaptiveIntegrator,
                                   objective_mechanism= \
                                       ObjectiveMechanism(
                                               name='EVC Objective Mechanism',
                                               function=LinearCombination(operation=PRODUCT),
                                               monitor=[reward,
                                                        (decision.output_ports[PROBABILITY_UPPER_THRESHOLD], 1, -1)]),
                                   function=GridSearch,
                                   control_signals=[ControlSignal(modulates=[(GAIN, color_hidden)],
                                                                  function=Linear,
                                                                  variable=1.0,
                                                                  intensity_cost_function=Exponential(rate=0.8046),
                                                                  allocation_samples=control_signal_search_range),
                                                    ControlSignal(modulates=[(GAIN, word_hidden)],
                                                                  function=Linear,
                                                                  variable=1.0,
                                                                  intensity_cost_function=Exponential(rate=0.8046),
                                                                  allocation_samples=control_signal_search_range)])
Stroop_model.add_controller(evc)

# SHOW_GRAPH ***********************************

# Stroop_model.show_graph()
Stroop_model.show_graph(show_controller=True)
# Stroop_model.show_graph(show_node_structure=ALL)
# Stroop_model.show_graph(show_dimensions=True, show_node_structure=ALL)

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
    if t==0:
        return
    print(f'\nEnd of trial {t}:')
    print(f'\t\t\t\tcolor  word')
    print(f'\ttask:\t\t{task.value[0]}')
    print(f'\ttask gain:\t   {task.parameter_ports[GAIN].value}')
    print(f'\t\t\t\tred   green')
    print(f'\toutput:\t\t{output.value[0]}')
    print(f'\tdecision:\t{decision.value[0]}{decision.value[1]}')
    print(f'\tconflict:\t  {control._objective_mechanism.value[0]}')
    t += 1

task.log.set_log_conditions(VALUE)
evc.log.set_log_conditions(VARIABLE)
evc.log.set_log_conditions(VALUE)

task.initial_value = [0.5,0.5]
task.reset_stateful_function_when=AtPass(n=0)
# task.reset_stateful_function_when=AtTrialStart()

num_trials = 2
stimuli = {color_input:[red] * num_trials,
           word_input:[green] * num_trials,
           task_input:[color] * num_trials}
Stroop_model.run(inputs=stimuli,
                 # animate=True,
                 animate={'show_controller':True,
                          'show_cim':True},
                 call_after_trial=print_after
                 )
Stroop_model.log.print_entries()
