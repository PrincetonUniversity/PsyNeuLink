import psyneulink as pnl
import numpy as np


colors_input_layer = pnl.TransferMechanism(size=3,
                                           function=pnl.Linear,
                                           name='COLORS_INPUT')

words_input_layer = pnl.TransferMechanism(size=3,
                                          function=pnl.Linear,
                                          name='WORDS_INPUT')

task_input_layer = pnl.TransferMechanism(size=2,
                                         function=pnl.Linear,
                                         name='TASK_INPUT')

#   Task layer, tasks: ('name the color', 'read the word')
task_layer = pnl.RecurrentTransferMechanism(size=2,
                                            function=pnl.Logistic(),
                                            hetero=-2,
                                            integrator_mode=True,
                                            integration_rate=0.01,
                                            name='TASK_LAYER')

# Hidden layer
# colors: ('red','green', 'neutral') words: ('RED','GREEN', 'NEUTRAL')
colors_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                     function=pnl.Logistic(x_0=4.0),  # bias 4.0 is -4.0 in the paper see Docs for description
                                                     integrator_mode=True,
                                                     hetero=-2,
                                                     integration_rate=0.01,  # cohen-huston text says 0.01
                                                     name='COLORS_HIDDEN')

words_hidden_layer = pnl.RecurrentTransferMechanism(size=3,
                                                    function=pnl.Logistic(x_0=4.0),
                                                    integrator_mode=True,
                                                    hetero=-2,
                                                    integration_rate=0.01,
                                                    name='WORDS_HIDDEN')

#   Response layer, responses: ('red', 'green')
response_layer = pnl.RecurrentTransferMechanism(size=2,
                                                function=pnl.Logistic(),
                                                hetero=-2.0,
                                                integrator_mode=True,
                                                integration_rate=0.01,
                                                output_ports = [pnl.RESULT,
                                                                 {pnl.NAME: 'DECISION_ENERGY',
                                                                  pnl.VARIABLE: (pnl.OWNER_VALUE,0),
                                                                  pnl.FUNCTION: pnl.Stability(
                                                                          default_variable = np.array([0.0, 0.0]),
                                                                          metric = pnl.ENERGY,
                                                                          matrix = np.array([[0.0, -4.0],
                                                                                             [-4.0, 0.0]]))}],
                                                name='RESPONSE', )

# Mapping projections---------------------------------------------------------------------------------------------------

color_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                             [0.0, 1.0, 0.0],
                                                             [0.0, 0.0, 1.0]]))

word_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0, 0.0],
                                                            [0.0, 1.0, 0.0],
                                                            [0.0, 0.0, 1.0]]))

task_input_weights = pnl.MappingProjection(matrix=np.array([[1.0, 0.0],
                                                            [0.0, 1.0]]))

color_task_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 0.0],
                                                             [4.0, 0.0],
                                                             [4.0, 0.0]]))

task_color_weights  = pnl.MappingProjection(matrix=np.array([[4.0, 4.0, 4.0],
                                                             [0.0, 0.0, 0.0]]))

response_color_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0, 0.0],
                                                                [0.0, 1.5, 0.0]]))

response_word_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0, 0.0],
                                                                [0.0, 2.5, 0.0]]))

color_response_weights = pnl.MappingProjection(matrix=np.array([[1.5, 0.0],
                                                                [0.0, 1.5],
                                                                [0.0, 0.0]]))

word_response_weights  = pnl.MappingProjection(matrix=np.array([[2.5, 0.0],
                                                                [0.0, 2.5],
                                                                [0.0, 0.0]]))

word_task_weights = pnl.MappingProjection(matrix=np.array([[0.0, 4.0],
                                                           [0.0, 4.0],
                                                           [0.0, 4.0]]))

task_word_weights = pnl.MappingProjection(matrix=np.array([[0.0, 0.0, 0.0],
                                                           [4.0, 4.0, 4.0]]))

# CREATE Composition
comp = pnl.Composition()

# Add mechanisms
comp.add_node(colors_input_layer)
comp.add_node(colors_hidden_layer)

comp.add_node(words_input_layer)
comp.add_node(words_hidden_layer)

comp.add_node(task_input_layer)
comp.add_node(task_layer)
comp.add_node(response_layer)

# Add projections
comp.add_projection(task_input_weights, task_input_layer, task_layer)

# Color process
comp.add_projection(color_input_weights, colors_input_layer, colors_hidden_layer)
comp.add_projection(color_response_weights, colors_hidden_layer, response_layer)
comp.add_projection(response_color_weights, response_layer, colors_hidden_layer)

# Word process
comp.add_projection(word_input_weights, words_input_layer, words_hidden_layer)
comp.add_projection(word_response_weights, words_hidden_layer, response_layer)
comp.add_projection(response_word_weights, response_layer, words_hidden_layer)

# Color task process
comp.add_projection(task_color_weights, task_layer, colors_hidden_layer)
comp.add_projection(color_task_weights, colors_hidden_layer, task_layer)

# Word task process
comp.add_projection(task_word_weights, task_layer, words_hidden_layer)
comp.add_projection(word_task_weights, words_hidden_layer, task_layer)

def trial_dict(red_color, green_color, neutral_color, red_word, green_word, neutral_word, CN, WR):
    trialdict = {
    colors_input_layer: [red_color, green_color, neutral_color],
    words_input_layer: [red_word, green_word, neutral_word],
    task_input_layer: [CN, WR]
    }
    return trialdict

# Define initialization trials separately
CN_trial_initialize_input = trial_dict(0, 0, 0, 0, 0, 0, 1, 0)
  #red_color, green color, red_word, green word, CN, WR
CN_incongruent_trial_input = trial_dict(1, 0, 0, 0, 1, 0, 1, 0)
 #red_color, green color, red_word, green word, CN, WR
CN_congruent_trial_input = trial_dict(1, 0, 0, 1, 0, 0, 1, 0)
   #red_color, green color, red_word, green word, CN, WR
CN_control_trial_input = trial_dict(1, 0, 0, 0, 0, 1, 1, 0)
     #red_color, green color, red_word, green word, CN, WR

Stimulus = [[CN_trial_initialize_input, CN_congruent_trial_input],
            [CN_trial_initialize_input, CN_incongruent_trial_input],
            [CN_trial_initialize_input, CN_control_trial_input]]

# should be 500 and 1000
ntrials0 = 5
ntrials = 10
comp._analyze_graph()

comp.show_graph()

def run(bin_execute):
    results = []
    for stim in Stimulus:
        # RUN the SYSTEM to initialize ---------------------------------------
        comp.run(inputs=stim[0], num_trials=ntrials0, bin_execute=bin_execute)
        comp.run(inputs=stim[1], num_trials=ntrials, bin_execute=bin_execute)
        # reset after condition was run
        colors_hidden_layer.reset([[0, 0, 0]], context=comp)
        words_hidden_layer.reset([[0, 0, 0]], context=comp)
        response_layer.reset([[0, 0]], context=comp)
        task_layer.reset([[0, 0]], context=comp)
        # Comp results include concatenation of both the above runs
        results.append(comp.results.copy())
        comp.reset()
        comp.results = []

    return results
pnlv_graphics_spec = {
    "components": {
        "nodes": {
            "COLORS_HIDDEN": {
                "x": 399,
                "y": 145
            },
            "WORDS_INPUT": {
                "x": 887,
                "y": 398
            },
            "TASK_INPUT": {
                "x": 670,
                "y": 435
            },
            "WORDS_HIDDEN": {
                "x": 877,
                "y": 141
            },
            "RESPONSE": {
                "x": 674,
                "y": 50
            },
            "COLORS_INPUT": {
                "x": 412,
                "y": 403
            },
            "TASK_LAYER": {
                "x": 674,
                "y": 278
            }
        }
    }
}
