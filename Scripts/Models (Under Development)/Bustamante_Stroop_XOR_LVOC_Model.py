# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ********************************************* Bustamante_Stroop_XOR_LVOC_Model ***************************************

# along with str cast in log line 1516 vv
# next_eid_entry_data += ", \'" + "\', \'".join(str(i[0]) if isinstance(i, list) else i for i in data[0]) + "\'"

"""
Implements a model of the `Stroop XOR task
<https://scholar.google.com/scholar?hl=en&as_sdt=0%2C31&q=laura+bustamante+cohen+musslick&btnG=>`_
using a version of the `Learned Value of Control Model
<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006043&rev=2>`_
"""
import importlib

import numpy as np
import psyneulink as pnl
import timeit

import psyneulink.core.components.functions.learningfunctions
import psyneulink.core.components.functions.optimizationfunctions
import psyneulink.core.components.functions.transferfunctions

# from build_input import xor_dict
from build_stimuli_VZ import xor_dict
import csv

np.random.seed(0)


def w_fct(stim, color_control):
    """function for word_task, to modulate strength of word reading based on 1-strength of color_naming ControlSignal"""
    print('color control: ', color_control)
    return stim * (1 - color_control)


w_fct_UDF = pnl.UserDefinedFunction(custom_function=w_fct, color_control=1)


def objective_function(v):
    """function used for ObjectiveMechanism of lvoc
     v[0] = output of DDM: [probability of color naming, probability of word reading]
     v[1] = reward:        [color naming rewarded, word reading rewarded]
     """
    prob_upper = v[0]
    prob_lower = v[1]
    reward_upper = v[2][0]
    reward_lower = v[2][1]
    return prob_upper * reward_upper + prob_lower * reward_lower
    # return np.sum(v[0] * v[1])


color_stim = pnl.TransferMechanism(name='Color Stimulus', size=8)
word_stim = pnl.TransferMechanism(name='Word Stimulus', size=8)

color_task = pnl.TransferMechanism(name='Color Task')
word_task = pnl.ProcessingMechanism(name='Word Task', function=w_fct_UDF)

reward = pnl.TransferMechanism(name='Reward', size=2)

task_decision = pnl.DDM(
    name='Task Decision',
    # function=pnl.NavarroAndFuss,
    output_ports=[
        pnl.PROBABILITY_UPPER_THRESHOLD,
        pnl.PROBABILITY_LOWER_THRESHOLD
    ]
)

task_decision.set_log_conditions('func_drift_rate')
task_decision.set_log_conditions('mod_drift_rate')
task_decision.set_log_conditions('PROBABILITY_LOWER_THRESHOLD')
task_decision.set_log_conditions('PROBABILITY_UPPER_THRESHOLD')
color_task.set_log_conditions('value')
word_task.set_log_conditions('value')


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
    features=[color_stim.input_port, word_stim.input_port],
    # features={pnl.SHADOW_EXTERNAL_INPUTS: [color_stim, word_stim]},
    objective_mechanism=pnl.ObjectiveMechanism(
        name='LVOC ObjectiveMechanism',
        monitor=[task_decision.output_ports[pnl.PROBABILITY_UPPER_THRESHOLD],
                 task_decision.output_ports[pnl.PROBABILITY_LOWER_THRESHOLD],
                 reward],
        # monitored_output_ports=[task_decision, reward],
        function=objective_function
    ),
    agent_rep=pnl.RegressionCFA(
        update_weights=pnl.BayesGLM(mu_0=-0.17, sigma_0=9.0909), # -0.17, 9.0909 precision = 0.11; 1/p = v
        prediction_terms=[pnl.PV.FC, pnl.PV.COST]
    ),
    function=pnl.GradientOptimization(
        convergence_criterion=pnl.VALUE,
        convergence_threshold=0.001, #0.001
        step_size=1, #1
        annealing_function=lambda x, y: x / np.sqrt(y)
        # direction=pnl.ASCENT
    ),
    control_signals=pnl.ControlSignal(
        modulates=[(pnl.SLOPE, color_task), ('color_control', word_task)],
        # function=pnl.ReLU,
        function=pnl.Logistic,
        cost_options=[pnl.CostFunctions.INTENSITY, pnl.CostFunctions.ADJUSTMENT],
        intensity_cost_function=pnl.Exponential(rate=0.25, bias=-3),
        adjustment_cost_function=pnl.Exponential(rate=0.25, bias=-3),
        # allocation_samples=[i / 2 for i in list(range(0, 50, 1))]
    )
    )

# print(lvoc.loggable_items)
lvoc.set_log_conditions('value')
# print(lvoc.loggable_items)
# lvoc.set_log_conditions('variable')
# lvoc.agent_rep.set_log_conditions('regression_weights')

lvoc.reportOutputPref=True

c.add_node(lvoc)

# c.show_graph(show_node_structure=pnl.ALL)

    # input_dict = {
    #     color_stim: [[1, 0, 0, 0, 0, 0, 0, 0]],
    #     word_stim: [[1, 0, 0, 0, 0, 0, 0, 0]],
    #     color_task: [[1]],
    #     word_task: [[-1]],
    #     reward: [[1, 0]]
    # }


# print('PREDICTION WEIGHTS T1', lvoc.agent_rep.parameters.regression_weights.get(None))

# for i in range(len(xor_dict)): # run on all 30 subjects
for i in range(3): # testing for three subjects, 200 trials per subject
    input_dict = {color_stim: xor_dict[i][0],
              word_stim: xor_dict[i][1],
              color_task: xor_dict[i][2],
              word_task: xor_dict[i][3],
              reward:    xor_dict[i][4]}

    # print("input dict: ", input_dict)

    # start_time = time.time()

    def print_weights():
        print("OUTCOME = ", lvoc.objective_mechanism.output_ports[pnl.OUTCOME].value)
        print("WEIGHTS = ", lvoc.agent_rep.parameters.regression_weights.get(i))
        print("LVOC VALUE = ", lvoc.value)
    # duration = timeit.timeit(c.run(inputs=input_dict, context=i), number=1) #number=2
    c.run(inputs=input_dict,
          context=i,
          call_after_trial=print_weights) #number=2, num_trials
    # duration = time.time() - start_time
    # print('PREDICTION WEIGHTS T2', lvoc.agent_rep.parameters.regression_weights.get(i))
    # print("WEIGHTS = ", lvoc.agent_rep.parameters.regression_weights.get(i))
    # print("OUTCOME = ", lvoc.objective_mechanism.output_ports[pnl.OUTCOME].value)
    # print('LVOC Log\n')
    # print(lvoc.log.csv)
    # print('Task Decision Log\n')
    # print(task_decision.log.csv())
    # print('Color Task Log\n')
    # print(color_task.log.csv())
    # print('Word Task Log\n')
    # print(word_task.log.csv())

    print('\n')
    print('Subject: ', i + 1)
    print('--------------------')
    print('ControlSignal variables: ', [sig.parameters.variable.get(i) for sig in lvoc.control_signals])
    print('ControlSignal values: ', [sig.parameters.value.get(i) for sig in lvoc.control_signals])
    # print('features: ', lvoc.get_feature_values(context=c))
    print('lvoc: ', lvoc.evaluation_function([sig.parameters.variable.get(i) for sig in lvoc.control_signals], context=i))
    # print('time: ', duration)
    print('--------------------')

# ------------------------------------------------------------------------------------------------------------------

print('\n\n\nLVOC Log\n')
print(lvoc.log.csv())
# print(lvoc.log.nparray())
# # print(lvoc.log.print_entries())

# print('\n\n\nTask Decision Log\n')
# # print(task_decision.log.csv())
# print(task_decision.log.nparray())
# print('\n\n\nColor Task Log\n')
# # print(color_task.log.csv())
# print(color_task.log.nparray())
# print('\n\n\nWord Task Log\n')
# # print(word_task.log.csv())
# print(word_task.log.nparray())

# file_lvoc = open("LVOC_200_LOG.csv", 'w')
# file_lvoc.write(lvoc.log.csv())
# file_lvoc.close()

# file_task = open("LVOC_200_TASK.csv", 'w')
# # file_task.write(task_decision.log.csv())
# file_task.close()


# def run():
#     c.run(inputs=input_dict, num_trials=1)


# duration = timeit.timeit(run, number=2)

# print('\n')
# print('--------------------')
# print('ControlSignal variables: ', [sig.parameters.variable.get(c) for sig in lvoc.control_signals])
# print('ControlSignal values: ', [sig.parameters.value.get(c) for sig in lvoc.control_signals])
# # print('features: ', lvoc.get_feature_values(context=c))
# print('lvoc: ', lvoc.evaluation_function([sig.parameters.variable.get(c) for sig in lvoc.control_signals], context=c))
# print('time: ', duration)
# print('--------------------')
