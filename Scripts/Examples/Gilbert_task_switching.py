
# this scrip implements Gilbert and Shallice 2002 PDP Task Switching Model
import numpy as np
import psyneulink as pnl


### LAYERS
WORD_INPUT_LAYER = pnl.TransferMechanism(size = 3,
                                         function=pnl.Linear,
                                         name='WORD INPUT LAYER')

COLOR_INPUT_LAYER = pnl.TransferMechanism(size = 3,
                                         function=pnl.Linear,
                                         name='WORD INPUT LAYER')

WORD_OUTPUT_LAYER = pnl.RecurrentTransferMechanism(size = 3,
                                                   auto= 0.0,
                                                   hetero= -2.0,
                                                   function=pnl.Logistic(offset= -6),
                                                   name='WORD INPUT LAYER')

COLOR_OUTPUT_LAYER = pnl.IntegratorMechanism(size = 3,
                                                   auto= 0.0,
                                                   hetero= -2.0,
                                                   function= pnl.InteractiveActivation #(offset= -6),
                                                   name='COLOR OUTPUT LAYER')

TASK_DEMAND_LAYER = pnl.RecurrentTransferMechanism(size = 3,
                                                   auto= 0.0,
                                                   hetero= -2.0,
                                                   function=pnl.Logistic(offset= -4),
                                                   name='TASK DEMAND LAYER')


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






### PROCESSES
#   Create pathways as processes

#   Words pathway
words_process = pnl.Process(pathway=[WORD_INPUT_LAYER,
                                     word_weights,
                                     WORD_OUTPUT_LAYER], name='WORDS_PROCESS')

#   Colors pathway
colors_process = pnl.Process(pathway=[COLOR_INPUT_LAYER,
                                      color_weights,
                                      COLOR_OUTPUT_LAYER], name='COLORS_PROCESS')

#   Task demand word  pathway
task_WR_process = pnl.Process(pathway=[TASK_DEMAND_LAYER,
                                       task_demand_word_output_weights,
                                       WORD_OUTPUT_LAYER],
                              name='TASK_WR_PROCESS')

#   Task demand color  pathway
task_CN_process = pnl.Process(pathway=[TASK_DEMAND_LAYER,
                                       task_demand_color_output_weights,
                                       COLOR_OUTPUT_LAYER],
                              name='TASK_CN_PROCESS')

#   Word input Task pathway
words_task_process = pnl.Process(pathway=[WORD_INPUT_LAYER,
                                     word_weights,
                                     WORD_OUTPUT_LAYER], name='WORDS_PROCESS')



#   CREATE SYSTEM
my_Stroop = pnl.System(processes=[colors_process,
                                  words_process,
                                  task_CN_process,
                                  task_WR_process,
                                  respond_red_process,
                                  respond_green_process],
                       name='FEEDFORWARD_STROOP_SYSTEM')

my_Stroop.show()
# my_Stroop.show_graph(show_dimensions=pnl.ALL)

### SYSTEM



