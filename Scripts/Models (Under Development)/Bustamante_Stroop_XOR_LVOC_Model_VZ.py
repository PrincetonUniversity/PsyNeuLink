# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Bustamante_Stroop_XOR_LVOC_Model ***************************************

# along with str cast in log line 1516 vv
# next_eid_entry_data += ", \'" + "\', \'".join(str(i[0]) if isinstance(i, list) else i for i in data[0]) + "\'"

'''
Implements a model of the `Stroop XOR task
<https://scholar.google.com/scholar?hl=en&as_sdt=0%2C31&q=laura+bustamante+cohen+musslick&btnG=>`_
using a version of the `Learned Value of Control Model
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev=2>`_
'''
import importlib

import numpy as np
import psyneulink as pnl
import timeit

import psyneulink.core.components.functions.learningfunctions
import psyneulink.core.components.functions.optimizationfunctions
import psyneulink.core.components.functions.transferfunctions

from build_stimuli_VZ import xor_dict
import csv
import math

np.random.seed(0)

def w_fct(stim, color_control):
    '''function for word_task, to modulate strength of word reading based on 1-strength of color_naming ControlSignal'''
    # print('stim: ', stim)
    # print("Color control: ", color_control)
    return stim * (1 - color_control)


w_fct_UDF = pnl.UserDefinedFunction(custom_function=w_fct, color_control=1)


def objective_function(v):
    '''function used for ObjectiveMechanism of lvoc
     v[0] = probability of color naming (DDM output)
     v[1] = probability of word reading (DDM output)
     v[2] = reward: [color naming rewarded, word reading rewarded]'''
    prob_upper = v[0]
    prob_lower = v[1]
    reward_upper = v[2][0]
    reward_lower = v[2][1]
    # print('prob upper: ', prob_upper)
    # print('prob lower: ', prob_lower)
    # print("reward: ", prob_upper * reward_upper + prob_lower * reward_lower)
    return prob_upper * reward_upper + prob_lower * reward_lower


color_stim = pnl.TransferMechanism(name='Color Stimulus', size=8)
word_stim = pnl.TransferMechanism(name='Word Stimulus', size=8)

color_task = pnl.TransferMechanism(name='Color Task')
word_task = pnl.ProcessingMechanism(name='Word Task', function=w_fct_UDF)

reward = pnl.TransferMechanism(name='Reward', size=2)

task_decision = pnl.DDM(
    name='Task Decision',
    # function=pnl.NavarroAndFuss,
    output_states=[
        pnl.DDM_OUTPUT.PROBABILITY_UPPER_THRESHOLD,
        pnl.DDM_OUTPUT.PROBABILITY_LOWER_THRESHOLD
    ]
)

# print("Task decision loggable: ", task_decision.loggable_items)
task_decision.set_log_conditions('InputState-0')  
# task_decision.set_log_conditions('func_drift_rate')     
# task_decision.set_log_conditions('mod_drift_rate')      
task_decision.set_log_conditions('PROBABILITY_LOWER_THRESHOLD')     
task_decision.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')     
color_task.set_log_conditions('value')      
word_task.set_log_conditions('value')

control_signal_range = (0,1)
default_control_signal = np.mean(control_signal_range)

c = pnl.Composition(name='Stroop XOR Model')
c.add_node(color_stim)
c.add_node(word_stim)
c.add_node(color_task, required_roles=pnl.NodeRole.ORIGIN)
c.add_node(word_task, required_roles=pnl.NodeRole.ORIGIN)
c.add_node(reward)
c.add_node(task_decision)
c.add_projection(sender=color_task, receiver=task_decision)
c.add_projection(sender=word_task, receiver=task_decision)

lvoc = pnl.OptimizationControlMechanism(
    name='LVOC ControlMechanism',
    features=[color_stim.input_state, word_stim.input_state],
    # features={pnl.SHADOW_EXTERNAL_INPUTS: [color_stim, word_stim]},

    # computes value of processing, reward received
    objective_mechanism=pnl.ObjectiveMechanism(
        name='LVOC ObjectiveMechanism',
        monitor=[task_decision.output_states[pnl.PROBABILITY_UPPER_THRESHOLD],
                 task_decision.output_states[pnl.PROBABILITY_LOWER_THRESHOLD],
                 reward],
        function=objective_function
    ),
    # posterior weight distribution
    agent_rep=pnl.RegressionCFA(
        update_weights=pnl.BayesGLM(mu_0=-0.17, sigma_0=0.11), #sigma_0=math.sqrt(0.11))
        prediction_terms=[pnl.PV.FC, pnl.PV.COST]
    ),
    # sample control allocs, and return best
    # evaluate() computes outcome (obj mech) - costs given state (features) and sample ctrl alloc
    function=pnl.GradientOptimization(
            convergence_criterion=pnl.VALUE,
            convergence_threshold=0.001,
            step_size=1, #1
            # bounds=control_signal_range,
            bounds = ([[3],[2]], None),
            # Note: Falk used 10 in the denom below, but indexed sample numbers from 1;
            #       but sample_num passed to _follow_gradient is indexed from 0, so use 11 below
            annealing_function=lambda x, y: x / np.sqrt(11+y),
            max_iterations=100
            # save_samples=True,
            # save_values=True,
            # direction=pnl.ASCENT
    ),
    # opt control alloc used to compute ctrl sigs
    control_signals=pnl.ControlSignal(
            variable=default_control_signal,
            projections=[(pnl.SLOPE, color_task), ('color_control', word_task)],
            # function=pnl.ReLU,
            # function=pnl.Logistic,
            cost_options=[pnl.ControlSignalCosts.INTENSITY, pnl.ControlSignalCosts.ADJUSTMENT],
            # intensity_cost_function=pnl.Exponential(rate=0, bias=0), # 0.25, -3
            # adjustment_cost_function=pnl.Exponential(rate=0, bias=0) # 0.25, -3
            intensity_cost_function=pnl.Linear(slope=0, intercept=0), # 0.25, -3
            adjustment_cost_function=pnl.Linear(slope=0, intercept=0) # 0.25, -3
            # allocation_samples=[i / 2 for i in list(range(0, 50, 1))]
    )
)

lvoc.set_log_conditions('value')
# lvoc.set_log_conditions('features')
# print("LVOC loggable: ", lvoc.loggable_items)
# lvoc.set_log_conditions('variable')     
# lvoc.agent_rep.set_log_conditions('regression_weights')

lvoc.reportOutputPref=True

c.add_node(lvoc)

# c.show_graph(show_node_structure=pnl.ALL)


for i in range(len(xor_dict)): # 30 subjects, 520 trials
# for i in range(2):
    input_dict = {color_stim: xor_dict[i][0],
              word_stim: xor_dict[i][1],
              color_task: xor_dict[i][2],
              word_task: xor_dict[i][3],
              reward:    xor_dict[i][4]}

    # start_time = time.time()
    def print_weights():
        print()
        # print("----------------ENDED TRIAL-------------------")

    # duration = timeit.timeit(c.run(inputs=input_dict, execution_id=i), number=1) #number=2
    c.run(inputs=input_dict,
          execution_id=i,
          call_after_trial=print_weights) #num_trials
    # duration = time.time() - start_time

    print('\n')
    print('Subject: ', i+1)
    print('--------------------')
    print('ControlSignal variables: ', [sig.parameters.variable.get(i) for sig in lvoc.control_signals])
    print('ControlSignal values: ', [sig.parameters.value.get(i) for sig in lvoc.control_signals])
    # print('features: ', lvoc.feature_values)
    # print('lvoc: ', lvoc.evaluation_function([sig.parameters.variable.get(i) for sig in lvoc.control_signals], execution_id=i))
    # print('time: ', duration)
    print('--------------------')

# ------------------------------------------------------------------------------------------------------------------

# print('\n\n\nLVOC Log\n')
print(lvoc.log.csv()) 

# print('\n\n\nTask Decision Log\n')
print(task_decision.log.csv())

# print('\n\n\nColor Task Log\n')
# # print(color_task.log.csv())

# print('\n\n\nWord Task Log\n')
# # print(word_task.log.csv())


file_lvoc = open("lvoc_5_29.csv", 'w')
file_lvoc.write(lvoc.log.csv()) 
file_lvoc.close()

file_task = open("task_5_29.csv", 'w')
file_task.write(task_decision.log.csv())
file_task.close()
