# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# CONTROL FLOW:
#   - EM EXECUTES FIRST:
#     - RETRIEVES USING PREVIOUS STATE NODE AND CONTEXT (PRE-INTEGRATION) TO RETRIEVE PREDICTED CURRENT STATE
#     - STORES VALUES OF PREVIOUS STATE, CURRENT STATE (INPUT) AND CONTEXT (PRE-INTEGRATION) INTO EM
#   - THEN:
#     - PREVIOUS_STATE EXECUTES TO GET CURRENT_STATE_INPUT (FOR RETRIEVAL ON NEXT TRIAL)
#     - INTEGRATOR LAYER EXECUTES, INTEGRATING CURRENT_STATE_INPUT INTO MEMORY
#     - CONTEXT LAYER EXECUTES TO GET LEARNED CONTEXT (FOR RETRIEVAL ON NEXT TRIAL)
#   - PREDICTED CURRENT STATE IS COMPARED WITH ACTUAL CURRENT STATE (TARGET) TO UPDATE INTEGRATOR -> CONTEXT WEIGHTS

# ISSUES:
#   * Using TransferMechanism (to avoid recurrent in PyTorch):
#     -> input is always just linearly integrated, and the integral is tanh'd
#        (not sure tanh is even necessary, since integral is always between 0 and 1)
#     -> how is recurrence implemented in PyTorch?
#   * ??Possible bug:  for nodes in nested composition (such as EMComposition):  calling of execute_node on the
#                      nested Composition rather than the outer one to which they now belong in
#                      PytorchCompositionWrapper

# TODO:
#
# SCRIPT STUFF:
# √ REPLACE INTEGRATOR RECURRENTTRANSFERMECHANISM WITH TRANSFERMECHANISM IN INTEGRATOR MODE
#   OR TRY USING LCA with DECAY?
# - CHECK THAT VERSION WITH TRANSFERMECHANISM FOR CONTEXT PRODUCES CORRECT EM ENTRIES PER PREVOUS BENCHMARKING
# - DEBUG LEARNING
#

"""
QUESTIONS:

NOTES:
    *MUST* run Experience before Predict, as the latter requires retrieved_reward to be non-zero
           (from last trial of Experience) in order to know to encode the next state (see control policy)

**Overview**
------------

This implements a model of...

The model is an example of...

The script contains methods to construct, train, and run the model, and analyze the results of its execution:

* `construct_model <EGO.construct_model>`:
  takes as arguments parameters used to construct the model;  for convenience, defaults are defined below,
  (under "Construction parameters")

* `train_network <EGO.train_network>`:
   ...

* `run_model <EGO.run_model>`:
   ...

* `analyze_results <EGO.analyze_results>`:
  takes as arguments the results of executing the model, and optionally a number of trials and EGO_level to analyze;
  returns...


**The Model**
-------------

The model is comprised of...

.. _EGO_Fig:

.. figure:: _static/<FIG FILE
   :align: left
   :alt: EGO Model for Revaluation Experiment

.. _EGO_model_composition:

*EGO_model Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~

This is comprised of...  three input Mechanisms, and the nested `ffn <EGO_ffn_composition>` `Composition`.


**Construction and Execution**
------------------------------

.. _EGO_settings:

*Settings*
~~~~~~~~~~

The default parameters are ones that have been fit to empirical data concerning human performance
(taken from `Kane et al., 2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_).

See "Settings for running the script" to specify whether the model is trained and/or executed when the script is run,
and whether a graphic display of the network is generated when it is constructed.

.. _EGO_stimuli:

*Stimuli*
~~~~~~~~~

Sequences of stimuli are constructed either using `SweetPea <URL HERE>`_
(using the script in stim/SweetPea) or replicate those used in...

    .. note::
       Use of SweetPea for stimulus generation requires it be installed::
       >> pip install sweetpea


.. _EGO_training:

*Training*
~~~~~~~~~~

MORE HERE

.. _EGO_execution:

*Execution*
~~~~~~~~~~~

MORE HERE

.. _EGO_methods_reference:

**Methods Reference**
---------------------


"""
import matplotlib.pyplot as plt
import numpy as np
import graph_scheduler as gs
from enum import IntEnum

import torch
torch.manual_seed(0)

from psyneulink import *
from psyneulink._typing import Union, Literal


#region   SCRIPT SETTINGS
# ======================================================================================================================
#                                                   SCRIPT SETTINGS
# ======================================================================================================================
# Settings for running script:

CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL =  (                     # Only one of the following can be uncommented:
    # None                             # suppress display of model
    {                                  # show simple visual display of model
    'show_pytorch': True,            # show pytorch graph of model
     'show_learning': True
    # # 'show_projections_not_in_composition': True,
    # # 'exclude_from_gradient_calc_style': 'dashed'# show target mechanisms for learning
    # # {'show_node_structure': True     # show detailed view of node structures and projections
    }
)
# RUN_MODEL = True,                       # True => run the model
RUN_MODEL = False                      # False => don't run the model
# EXECUTION_MODE = ExecutionMode.Python
EXECUTION_MODE = ExecutionMode.PyTorch
# REPORT_OUTPUT = ReportOutput.FULL  # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_OUTPUT = ReportOutput.OFF     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_PROGRESS = ReportProgress.OFF # Sets console progress bar during run
# PRINT_RESULTS = False                # print model.results to console after execution
SAVE_RESULTS = True                  # save model.results to disk
PLOT_RESULTS = True                  # plot results (PREDICTIONS) vs. TARGETS
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution
#endregion

#region ENVIRONMENT
# ======================================================================================================================
#                                                   ENVIRONMENT
# ======================================================================================================================

# Task environment:
import Environment
# CURRICULUM_TYPE = 'Blocked'     # 'Blocked' or 'Interleaved'
CURRICULUM_TYPE = 'Interleaved'     # 'Blocked' or 'Interleaved'
NUM_STIMS = 5
# dataset = Environment.generate_dataset(condition=CURRICULUM_TYPE)
dataset = Environment.generate_dataset(condition=CURRICULUM_TYPE)
if NUM_STIMS is ALL:
    INPUTS = dataset.xs.numpy()
    TARGETS = dataset.ys.numpy()
else:
    INPUTS = dataset.xs.numpy()[:NUM_STIMS]
    TARGETS = dataset.ys.numpy()[:NUM_STIMS]
TOTAL_NUM_STIMS = len(INPUTS)

#endregion

#region   PARAMETERS
# ======================================================================================================================
#                                                   MODEL PARAMETERS
# ======================================================================================================================

model_params = dict(

    # Names:
    name = "EGO Model CSW",
    state_input_layer_name = "STATE",
    previous_state_layer_name = "PREVIOUS STATE",
    context_layer_name = 'CONTEXT',
    em_name = "EM",
    prediction_layer_name = "PREDICTION",

    # Structral
    state_d = 11, # length of state vector
    previous_state_d = 11, # length of state vector
    context_d = 11, # length of context vector
    memory_capacity = TOTAL_NUM_STIMS, # number of entries in EM memory
    # memory_init = (0,.0001),  # Initialize memory with random values in interval
    memory_init = None,  # Initialize with zeros
    concatenate_keys = False,

    # Processing
    integration_rate = .69, # rate at which state is integrated into new context
    state_weight = 1, # weight of the state used during memory retrieval
    context_weight = 1, # weight of the context used during memory retrieval
    normalize_field_weights = True, # whether to normalize the field weights during memory retrieval
    # softmax_temperature = None, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    softmax_temperature = .1, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = ADAPTIVE, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_temperature = CONTROL, # temperature of the softmax used during memory retrieval (smaller means more argmax-like
    # softmax_threshold = None, # threshold used to mask out small values in softmax
    softmax_threshold = .001, # threshold used to mask out small values in softmax
    enable_learning=[True, False, False], # Enable learning for PREDICTION (STATE) but not CONTEXT or PREVIOUS STATE
    learn_field_weights = False,
    loss_spec = Loss.BINARY_CROSS_ENTROPY,
    learning_rate = .5,
    device = CPU,
    # device = MPS,
)

# EM structdural params:
EMFieldsIndex = IntEnum('EMFields',
                        ['STATE',
                         'CONTEXT',
                         'PREVIOUS_STATE'],
                        start=0)
STATE_RETRIEVAL_WEIGHT = 0
RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

if is_numeric_scalar(model_params['softmax_temperature']):      # translate to gain of softmax retrieval function
    RETRIEVAL_SOFTMAX_GAIN = 1/model_params['softmax_temperature']
else:                                                           # pass along ADAPTIVE or CONTROL spec
    RETRIEVAL_SOFTMAX_GAIN = model_params['softmax_temperature']
#endregion

#region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model(model_name:str=model_params['name'],

                    # Input layer:
                    state_input_name:str=model_params['state_input_layer_name'],
                    state_size:int=model_params['state_d'],

                    # Previous state
                    previous_state_input_name:str=model_params['previous_state_layer_name'],

                    # Context representation (learned):
                    context_name:str=model_params['context_layer_name'],
                    context_size:Union[float,int]=model_params['context_d'],
                    integration_rate:float=model_params['integration_rate'],

                    # EM:
                    em_name:str=model_params['em_name'],
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    retrieval_softmax_threshold=model_params['softmax_threshold'],
                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    previous_state_retrieval_weight:Union[float,int]=model_params['state_weight'],
                    context_retrieval_weight:Union[float,int]=model_params['context_weight'],
                    normalize_field_weights = model_params['normalize_field_weights'],
                    concatenate_keys = model_params['concatenate_keys'],
                    learn_field_weights = model_params['learn_field_weights'],
                    memory_capacity = model_params['memory_capacity'],
                    memory_init=model_params['memory_init'],

                    # Output:
                    prediction_layer_name:str=model_params['prediction_layer_name'],

                    # Learning
                    loss_spec=model_params['loss_spec'],
                    enable_learning=model_params['enable_learning'],
                    learning_rate = model_params['learning_rate'],
                    device=model_params['device']

                    )->Composition:

    assert 0 <= integration_rate <= 1,\
        f"integrator_retrieval_weight must be a number from 0 to 1"

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, size=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_input_name, size=state_size)
    # context_layer = ProcessingMechanism(name=context_name, size=context_size)
    context_layer = TransferMechanism(name=context_name,
                                      size=context_size,
                                      function=Tanh,
                                      integrator_mode=True,
                                      integration_rate=integration_rate)

    em = EMComposition(name=em_name,
                       memory_template=[[0] * state_size,   # state
                                        [0] * state_size,   # previous state
                                        [0] * state_size],  # context
                       memory_fill=memory_init,
                       memory_capacity=memory_capacity,
                       memory_decay_rate=0,
                       softmax_gain=retrieval_softmax_gain,
                       softmax_threshold=retrieval_softmax_threshold,
                       # Input Nodes:
                       field_names=[state_input_name,
                                    previous_state_input_name,
                                    context_name,
                                    ],
                       field_weights=(state_retrieval_weight,
                                      previous_state_retrieval_weight,
                                      context_retrieval_weight
                                      ),
                       normalize_field_weights=normalize_field_weights,
                       concatenate_keys=concatenate_keys,
                       learn_field_weights=learn_field_weights,
                       learning_rate=learning_rate,
                       enable_learning=enable_learning,
                       device=device
                       )

    prediction_layer = ProcessingMechanism(name=prediction_layer_name, size=state_size)

    
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
    state_to_previous_state_pathway = [state_input_layer,
                                       MappingProjection(matrix=IDENTITY_MATRIX,
                                                         learnable=False),
                                       previous_state_layer]
    state_to_context_pathway = [state_input_layer,
                                MappingProjection(matrix=IDENTITY_MATRIX,
                                                  learnable=False),
                                context_layer]
    state_to_em_pathway = [state_input_layer,
                           MappingProjection(sender=state_input_layer,
                                             receiver=em.nodes[state_input_name+VALUE],
                                             matrix=IDENTITY_MATRIX,
                                             learnable=False),
                           em]
    previous_state_to_em_pathway = [previous_state_layer,
                                    MappingProjection(sender=previous_state_layer,
                                                      receiver=em.nodes[previous_state_input_name+QUERY],
                                                      matrix=IDENTITY_MATRIX,
                                                      learnable=False),
                                    em]
    context_learning_pathway = [context_layer,
                                MappingProjection(sender=context_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  receiver=em.nodes[context_name + QUERY],
                                                  learnable=True),
                                em,
                                MappingProjection(sender=em.nodes[state_input_name + RETRIEVED],
                                                  receiver=prediction_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  learnable=False),
                                prediction_layer]

    # Composition
    EGO_comp = AutodiffComposition([state_to_previous_state_pathway,
                                    state_to_context_pathway,
                                    state_to_em_pathway,
                                    previous_state_to_em_pathway,
                                    context_learning_pathway],
                                   learning_rate=learning_rate,
                                   loss_spec=loss_spec,
                                   name=model_name,
                                   device=device)

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    EGO_comp.add_projection(MappingProjection(sender=state_input_layer,
                                              receiver=learning_components[0],
                                              learnable=False))

    # Ensure EM is executed (to encode previous state and context, and predict current state)
    #     before updating state and context
    EGO_comp.scheduler.add_condition(em, BeforeNodes(previous_state_layer, context_layer))

    # # Validate construction
    # print(EGO_comp.scheduler.consideration_queue)
    # import graph_scheduler
    # graph_scheduler.output_graph_image(EGO_comp.scheduler.graph, 'EGO_comp-scheduler.png')

    return EGO_comp
#endregion


#region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

if __name__ == '__main__':
    model = None

    if CONSTRUCT_MODEL:
        print(f'Constructing {model_params["name"]}')
        model = construct_model()
        assert 'DEBUGGING BREAK POINT'
        # print(model.scheduler.consideration_queue)
        # gs.output_graph_image(model.scheduler.graph, 'EGO_comp-scheduler.png')

    if DISPLAY_MODEL is not None:
        if model:
            model.show_graph(**DISPLAY_MODEL)
        else:
            print("Model not yet constructed")

    if RUN_MODEL:
        import timeit
        def print_stuff(**kwargs):
            print(f"\n**************\n BATCH: {kwargs['batch']}\n**************\n")
            print(kwargs)
            print('\nContext internal: \n', model.nodes['CONTEXT'].function.parameters.value.get(kwargs['context']))
            print('\nContext hidden: \n', model.nodes['CONTEXT'].parameters.value.get(kwargs['context']))
            print('\nContext for EM: \n',
                  model.nodes['EM'].nodes['CONTEXT [QUERY]'].parameters.value.get(kwargs['context']))
            print('\nPrediction: \n',
                  model.nodes['PREDICTION'].parameters.value.get(kwargs['context']))
            # print('\nLoss: \n',
            #       model.parameters.tracked_loss._get(kwargs['context']))
            print('\nProjections from context to EM: \n', model.projections[7].parameters.matrix.get(kwargs['context']))
            print('\nEM Memory: \n', model.nodes['EM'].parameters.memory.get(model.name))

        # print("MODEL NOT YET FULLY EXECUTABLE")
        print(f"Running {model_params['name']}")
        context = model_params['name']
        start_time = timeit.default_timer()
        model.learn(inputs={model_params['state_input_layer_name']:INPUTS},
                  # report_output=REPORT_OUTPUT,
                  # report_progress=REPORT_PROGRESS
                  #   call_after_minibatch=print('Projections from context to EM: ',
                  #                              model.projections[7].parameters.matrix.get(context)),
                  #                              # model.projections[7].matrix)
                  #   call_after_minibatch=print_stuff,
                    optimizations_per_minibatch=1,
                    learning_rate=model_params['learning_rate'],
                    execution_mode=ExecutionMode.PyTorch,
                    # minibatch_size=3,
                  )
        stop_time = timeit.default_timer()
        print(f"Elapsed time: {stop_time - start_time}")
        if DISPLAY_MODEL is not None:
            model.show_graph(**DISPLAY_MODEL)
        if PRINT_RESULTS:
            print("MEMORY:")
            print(model.nodes['EM'].parameters.memory.get(model.name))
            model.run(inputs={model_params["state_input_layer_name"]:INPUTS[4]},
                      # report_output=REPORT_OUTPUT,
                      # report_progress=REPORT_PROGRESS
                      )
            print("CONTEXT INPUT:")
            print(model.nodes['CONTEXT'].parameters.variable.get(model.name))
            print("CONTEXT OUTPUT:")
            print(model.nodes['CONTEXT'].parameters.value.get(model.name))
            print("PREDICTION OUTPUT:")
            print(model.nodes['PREDICTION'].parameters.value.get(model.name))
            print("CONTEXT WEIGHTS:")
            print(model.projections[7].parameters.matrix.get(model.name))
            plt.imshow(model.projections[7].parameters.matrix.get(model.name))
            def test_weights(weight_mat):
                # checks whether only 5 weights are updated.
                weight_mat -= np.eye(11)
                col_sum = weight_mat.sum(1)
                row_sum = weight_mat.sum(0)
                return np.max([(row_sum != 0).sum(), (col_sum != 0).sum()])
            print(test_weights(model.projections[7].parameters.matrix.get(model.name)))

        if SAVE_RESULTS:
            np.save('EGO PREDICTIONS', model.results)
            np.save('EGO INPUTS', INPUTS)
            np.save('EGO TARGETS', TARGETS)

        if PLOT_RESULTS:
            plt.plot(1 - np.abs(model.results[2:998,2]-TARGETS[:996]))
            plt.show()
            plt.savefig('EGO PLOT.png')

    #endregion
