from psyneulink import *
import numpy as np

# Construct the color naming pathway:
color_input = ProcessingMechanism(name='COLOR INPUT', size=2) # Note:  default function is Linear
color_input_to_hidden_wts = np.array([[1, -1], [-1, 1]])
color_hidden = ProcessingMechanism(name='COLOR HIDDEN', size=2, function=Logistic(bias=-4))
color_hidden_to_output_wts = np.array([[1, -1], [-1, 1]])
output = ProcessingMechanism(name='OUTPUT', size=2, function=Logistic)
color_pathway = [color_input, color_input_to_hidden_wts, color_hidden, color_hidden_to_output_wts, output]

# Construct the word reading pathway (using the same output_layer)
word_input = ProcessingMechanism(name='WORD INPUT', size=2)
word_input_to_hidden_wts = np.array([[2, -2], [-2, 2]])
word_hidden = ProcessingMechanism(name='WORD HIDDEN', size=2, function=Logistic(bias=-4))
word_hidden_to_output_wts = np.array([[2, -2], [-2, 2]])
word_pathway = [word_input, word_input_to_hidden_wts, word_hidden, word_hidden_to_output_wts, output]

# Construct the task specification pathways
task_input = ProcessingMechanism(name='TASK INPUT', size=2)
task_color_wts = np.array([[4,4],[0,0]])
task_word_wts = np.array([[0,0],[4,4]])
task_color_pathway = [task_input, task_color_wts, color_hidden]
task_word_pathway = [task_input, task_word_wts, word_hidden]

# Construct the decision pathway:
decision = DDM(name='DECISION',
               input_format=ARRAY,
               function=DriftDiffusionIntegrator(noise=0.5, threshold=20)
               )
decision_pathway = [output, decision]

# Construct the Composition:
Stroop_model = Composition(name='Stroop Model - Basic')
Stroop_model.add_linear_processing_pathway(color_pathway)
Stroop_model.add_linear_processing_pathway(word_pathway)
Stroop_model.add_linear_processing_pathway(task_color_pathway)
Stroop_model.add_linear_processing_pathway(task_word_pathway)
Stroop_model.add_linear_processing_pathway(decision_pathway)

# Label inputs
red = [1,0]
green = [0,1]
word = [0,1]
color = [1,0]

Stroop_model.show_graph()

Stroop_model.run(inputs={color_input:red, word_input:green, task_input:color},
                 num_trials=2,
                 termination_processing={TimeScale.TRIAL: WhenFinished(decision)},
                 animate={'show_cim':True}
                 )

print (Stroop_model.results)