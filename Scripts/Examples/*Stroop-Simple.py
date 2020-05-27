import functools
import numpy as np
import psyneulink as pnl

# NOTE:  This implements the two stimulus processing pathways (color naming and word reading)
#        but not an "attention" mechanism that selects between them... stay tuned!
import psyneulink.core.components.functions.transferfunctions

pathway_prefs = {
    pnl.REPORT_OUTPUT_PREF: True,
    pnl.VERBOSE_PREF: False
}

comp_prefs = {
    pnl.REPORT_OUTPUT_PREF: True,
    pnl.VERBOSE_PREF: False
}

colors = pnl.TransferMechanism(
    default_variable=[0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Linear,
    name="Colors"
)

words = pnl.TransferMechanism(
    default_variable=[0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Linear,
    name="Words"
)

hidden = pnl.TransferMechanism(
    default_variable=[0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Logistic,
    name="Hidden"
)

response = pnl.TransferMechanism(
    default_variable=[0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Logistic(),
    name="Response"
)

output = pnl.TransferMechanism(
    default_variable=[0, 0],
    function=psyneulink.core.components.functions.transferfunctions.Logistic,
    name="Output"
)

CH_Weights_matrix = np.arange(4).reshape((2, 2))
WH_Weights_matrix = np.arange(4).reshape((2, 2))
HO_Weights_matrix = np.arange(4).reshape((2, 2))

CH_Weights = pnl.MappingProjection(
    name='Color-Hidden Weights',
    matrix=CH_Weights_matrix
)
WH_Weights = pnl.MappingProjection(
    name='Word-Hidden Weights',
    matrix=WH_Weights_matrix
)
HO_Weights = pnl.MappingProjection(
    name='Hidden-Output Weights',
    matrix=HO_Weights_matrix
)

color_naming_pathway = pnl.Pathway(
    pathway=[colors, CH_Weights, hidden, HO_Weights, response],
    name='Color Naming',
    # prefs=pathway_prefs
)

word_reading_pathway = pnl.Pathway(
    pathway=[words, WH_Weights, hidden],
    name='Word Reading',
    # prefs=pathway_prefs
)

# color_naming_pathway.execute()
# word_reading_pathway.execute()

comp = pnl.Composition(pathways=[(color_naming_pathway, pnl.BackPropagation),
                                 (word_reading_pathway, pnl.BackPropagation)],
                       name='Stroop Model',
                       prefs=comp_prefs)

comp.show_graph(
    show_learning=pnl.ALL
)

def print_header(comp):
    print("\n\n**** Time: ", comp.scheduler.get_clock(comp).simple_time)

def show_target(context):
    print(f'\nColor Naming\n\tInput: {colors.input_ports.get_values_as_lists(context)}.')
    print(f'Wording Reading:\n\tInput: {words.input_ports.get_values_as_lists(context)}.')
    print(f'Target: {comp.pathways[0].target.value}.')
    print('Response: \n', response.get_output_values(context)[0])
    print('Hidden-Output:')
    print(HO_Weights.get_mod_matrix(context))
    print('Color-Hidden:')
    print(CH_Weights.get_mod_matrix(context))
    print('Word-Hidden:')
    print(WH_Weights.get_mod_matrix(context))


# FROM SYSTEM:
#     COLOR     default_variable=[1, 2.5],
#               target=[2, 2],
#     WORD:      default_variable=[.5, 3],
#                target=[3, 3],
#     COMPOSITION:  targets=[20, 20],

stim_list_dict = {
    colors: [[1, 1]],
    words: [[-2, -2]]
}

target_list_dict = {response: [[1, 1]]}

# comp.show_graph(show_learning=True)

comp.learn(
    num_trials=2,
    inputs=stim_list_dict,
    targets=target_list_dict,
    call_before_trial=functools.partial(print_header, comp),
    call_after_trial=show_target
)

# PsyNeuLink response & weights after 1st trial:
#
# Response:
#  [ 0.50899214  0.54318254]
# Hidden-Output:
# [[ 0.01462766  1.01351195]
#  [ 2.00220713  3.00203878]]
# Color-Hidden:
# [[ 0.01190129  1.0103412 ]
#  [ 2.01190129  3.0103412 ]]
# Word-Hidden:
# [[-0.02380258  0.9793176 ]
#  [ 1.97619742  2.9793176 ]]

# Matlab validated response & weights after 1st trial:
#
# response
#     0.5090    0.5432
# hidden-output weights
#     0.0146    1.0135
#     2.0022    3.0020
# color-hidden weights
#     0.0119    1.0103
#     2.0119    3.0103
# words-hidden weights
#    -0.0238    0.9793
#     1.9762    2.9793
