
# this scrip implements Gilbert and Shallice 2002 PDP Task Switching Model
import numpy as np
import psyneulink as pnl


### LAYERS
WORD_INPUT_LAYER = pnl.TransferMechanism(size = 3,
                                         function=pnl.Linear,
                                         name='WORD INPUT LAYER')

COLOR_INPUT_LAYER = pnl.TransferMechanism(size = 3,
                                         function=pnl.Linear,
                                         name='COLOR INPUT LAYER')

WORD_OUTPUT_LAYER = pnl.IntegratorMechanism(size = 3,
                                                   # auto= 0.0,
                                                   # hetero= -2.0,
                                                   function= pnl.InteractiveActivationIntegrator(decay= 0.0015, rest=-6),
                                                   name='WORD OUTPUT LAYER')
WORD_OUTPUT_LAYER.set_log_conditions('value')


COLOR_OUTPUT_LAYER = pnl.IntegratorMechanism(size = 3,
                                                   # auto= 0.0,
                                                   # hetero= -2.0,
                                                   function= pnl.InteractiveActivationIntegrator(decay= 0.0015, rest=-6, ),
                                             #  (rest= -6),
                                                   name='COLOR OUTPUT LAYER')

COLOR_OUTPUT_LAYER.set_log_conditions('value')


TASK_DEMAND_LAYER = pnl.IntegratorMechanism(size = 2,
                                                   # auto= 0.0,
                                                   # hetero= -2.0,
                                                   function= pnl.InteractiveActivationIntegrator(decay= 0.0015, max_val=1,
                                                                                       min_val= 1, rest= -4),
                                                   name='TASK DEMAND LAYER')

WORD_RECURRENT_LAYER = pnl.TransferMechanism(size = 3,
                                             function=pnl.Linear,
                                             name = 'WORD RECURRENT LAYER')

COLOR_RECURRENT_LAYER = pnl.TransferMechanism(size = 3,
                                             function=pnl.Linear,
                                             name = 'COLOR RECURRENT LAYER')

### WEIGHTS

#   WORD INPUT TO WORD OUTPUT
word_weights = pnl.MappingProjection(matrix=np.matrix([[3.5, 0.0, 0.0],
                                                        [0.0, 3.5, 0.0],
                                                        [0.0, 0.0, 3.5]]),
                                     name='WORD_WEIGHTS')

#   COLOR INPUT TO COLOR OUTPUT
color_weights = pnl.MappingProjection(matrix=np.matrix([[1.9, 0.0, 0.0],
                                                        [0.0, 1.9, 0.0],
                                                        [0.0, 0.0, 1.9]]),
                                      name='COLOR_WEIGHTS')

#    WORD INPUT to TASK DEMAND LAYER
word_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[1.0, 1.0],
                                                                    [1.0, 1.0],
                                                                    [1.0, 1.0]]),
                                                 name='WORD_TASK_DEMAND_WEIGHTS')

#    COLOR INPUT to TASK DEMAND LAYER
color_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[1.0, 1.0],
                                                                    [1.0, 1.0],
                                                                    [1.0, 1.0]]),
                                                  name='COLOR_TASK_DEMAND_WEIGHTS')


#   TASK DEMAND TO WORD OUTPUT
task_demand_word_output_weights = pnl.MappingProjection(matrix=np.matrix([[2.5, 2.5, 2.5],
                                                                          [-2.5, -2.5, -2.5]]),
                                                        name='TASK_DEMAND_WORD_OUTPUT_WEIGHTS')

#   TASK DEMAND TO COLOR OUTPUT
task_demand_color_output_weights = pnl.MappingProjection(matrix=np.matrix([[-2.5, -2.5, -2.5],
                                                                           [2.5, 2.5, 2.5]]),
                                                         name='TASK_DEMAND_COLOR_OUTPUT_WEIGHTS')

#   WORD OUTPUT TO TASK DEMAND
word_output_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[1.0, -1.0],
                                                                          [1.0, -1.0],
                                                                          [1.0, -1.0]]),
                                                        name='WORD_OUTPUT_TASK_DEMAND_WEIGHTS')

#   WORD OUTPUT TO TASK DEMAND
color_output_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[-1.0, 1.0],
                                                                           [-1.0, 1.0],
                                                                           [-1.0, 1.0]]),
                                                        name='COLOR_OUTPUT_TASK_DEMAND_WEIGHTS')

#   WORD OUTPUT to COLOR OUTPUT
word_output_color_output_weights = pnl.MappingProjection(matrix=np.matrix([[2.0, -2.0, -2.0],
                                                                           [-2.0, 2.0, -2.0],
                                                                           [-2.0, -2.0, 2.0]]),
                                                         name='WORD_OUTPUT_COLOR_OUTPUT_WEIGHTS')
# WORD OUTPUT TO TASK DEMAND
word_output_output_to_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[1.0, 1.0],
                                                                                          [1.0, 1.0],
                                                                                          [1.0, 1.0]]),
                                                                        name='WORD_COLOR_OUTPUT_TASK_DEMAND_WEIGHTS')

# COLOR OUTPUT TO TASK DEMAND
color_output_output_to_task_demand_weights = pnl.MappingProjection(matrix=np.matrix([[1.0, 1.0],
                                                                                          [1.0, 1.0],
                                                                                          [1.0, 1.0]]),
                                                                        name='COLOR_COLOR_OUTPUT_TASK_DEMAND_WEIGHTS')

# RECURRENT WORD weights
word_recurrent = pnl.MappingProjection(matrix=np.matrix([[0.0, -2.0, -2.0],
                                                         [-2.0, 0.0, -2.0],
                                                         [-2.0, -2.0, 0.0]]),
                                       name='WORD_RECURRENT_WEIGHTS')

# RECURRENT COLOR weights
color_recurrent = pnl.MappingProjection(matrix=np.matrix([[0.0, -2.0, -2.0],
                                                          [-2.0, 0.0, -2.0],
                                                          [-2.0, -2.0, 0.0]]),
                                       name='TASK_RECURRENT_WEIGHTS')

# RECURRENT TASK weights
task_recurrent = pnl.MappingProjection(matrix=np.matrix([[0.0, -2.0, -2.0],
                                                         [-2.0, 0.0, -2.0],
                                                         [-2.0, -2.0, 0.0]]),
                                       name='TASK_RECURRENT_WEIGHTS')

### COMPOSITION

Gilbert_Shallice_System = pnl.Composition(name="Gilbert_Shallice_System")

### ADD pathways
### pathway word input word output
words_input_output_pathway = [WORD_INPUT_LAYER,
                              word_weights,
                              WORD_OUTPUT_LAYER]

Gilbert_Shallice_System.add_linear_processing_pathway(pathway=words_input_output_pathway)

### pathway color input color output
color_input_output_pathway = [COLOR_INPUT_LAYER,
                              color_weights,
                              COLOR_OUTPUT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=color_input_output_pathway)

# PATHWAY: TASK WORD OUTPUT
task_word_output_pathway = [TASK_DEMAND_LAYER,
                              task_demand_word_output_weights,
                              WORD_OUTPUT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=task_word_output_pathway)

# Pathway: Task demand color output pathway
task_color_output_pathway = [TASK_DEMAND_LAYER,
                             task_demand_color_output_weights,
                             COLOR_OUTPUT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=task_color_output_pathway)

### Pathway: word input task demand
word_input_task_pathway = [WORD_INPUT_LAYER,
                           word_task_demand_weights,
                           TASK_DEMAND_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=word_input_task_pathway)

### Pathway: color input task demand
color_input_task_pathway = [COLOR_INPUT_LAYER,
                            color_task_demand_weights,
                            TASK_DEMAND_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=color_input_task_pathway)

### Pathway: Word output Task
word_output_task_pathway = [WORD_OUTPUT_LAYER,
                            word_output_output_to_task_demand_weights,
                            TASK_DEMAND_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=word_output_task_pathway)

### Pathway: Color output Task
color_output_task_pathway = [COLOR_OUTPUT_LAYER,
                             color_output_output_to_task_demand_weights,
                             TASK_DEMAND_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=color_output_task_pathway)

### Pathway: word output - recurrent word
word_output_word_recurrent_pathway = [WORD_OUTPUT_LAYER,
                                       WORD_RECURRENT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=word_output_word_recurrent_pathway)

### Pathway: color output - recurrent color
color_output_color_recurrent_pathway = [COLOR_OUTPUT_LAYER,
                                       COLOR_RECURRENT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=color_output_color_recurrent_pathway)


### Pathway: recurrent word - word output
word_recurrent_word_output_pathway = [WORD_RECURRENT_LAYER,
                                      word_recurrent,
                                      WORD_OUTPUT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=word_recurrent_word_output_pathway,
                                                      feedback=True)

### Pathway:  recurrent color - color output
color_recurrent_color_output_pathway = [COLOR_RECURRENT_LAYER,
                                        color_recurrent,
                                        COLOR_OUTPUT_LAYER]
Gilbert_Shallice_System.add_linear_processing_pathway(pathway=color_recurrent_color_output_pathway,
                                                      feedback=True)


## specify terminal layers (can only specify one terminal layer at a time)
# # MODIFIED 4/25/20 OLD:
# Gilbert_Shallice_System.add_required_node_role(WORD_OUTPUT_LAYER, pnl.NodeRole.TERMINAL)
# Gilbert_Shallice_System.add_required_node_role(COLOR_OUTPUT_LAYER, pnl.NodeRole.TERMINAL)
# MODIFIED 4/25/20 NEW:
Gilbert_Shallice_System.require_node_roles(WORD_OUTPUT_LAYER, pnl.NodeRole.TERMINAL)
Gilbert_Shallice_System.require_node_roles(COLOR_OUTPUT_LAYER, pnl.NodeRole.TERMINAL)
# MODIFIED 4/25/20 END

Gilbert_Shallice_System._analyze_graph()
# Gilbert_Shallice_System.show_graph()

input_dict = {COLOR_INPUT_LAYER: [1, 0, 0],
              WORD_INPUT_LAYER: [1, 0, 0],
              # TASK_DEMAND_LAYER: [1, 0]
              }

Gilbert_Shallice_System.run(num_trials=3,
                            inputs=input_dict)

WORD_OUTPUT_LAYER.log.print_entries()
COLOR_OUTPUT_LAYER.log.print_entries()

### SYSTEM
