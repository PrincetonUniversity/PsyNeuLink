# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""

CONTROL FLOW:
  - EM EXECUTES FIRST:
    - RETRIEVES USING PREVIOUS STATE NODE AND CONTEXT (PRE-INTEGRATION) TO RETRIEVE PREDICTED CURRENT STATE
    - STORES VALUES OF PREVIOUS STATE, CURRENT STATE (INPUT) AND CONTEXT (PRE-INTEGRATION) INTO EM
  - THEN:
    - PREVIOUS_STATE EXECUTES TO GET CURRENT_STATE_INPUT (FOR RETRIEVAL ON NEXT TRIAL)
    - INTEGRATOR LAYER EXECUTES, INTEGRATING CURRENT_STATE_INPUT INTO MEMORY
    - CONTEXT LAYER EXECUTES TO GET LEARNED CONTEXT (FOR RETRIEVAL ON NEXT TRIAL)
  - PREDICTED CURRENT STATE IS COMPARED WITH ACTUAL CURRENT STATE (TARGET) TO UPDATE INTEGRATOR -> CONTEXT WEIGHTS

ISSUES:
  * Using TransferMechanism (to avoid recurrent in PyTorch):
    -> input is always just linearly integrated, and the integral is tanh'd
       (not sure tanh is even necessary, since integral is always between 0 and 1)
    -> how is recurrence implemented in PyTorch?
  * ??Possible bug:  for nodes in nested composition (such as EMComposition):  calling of execute_node on the
                     nested Composition rather than the outer one to which they now belong in
                     PytorchCompositionWrapper

TODO:

SCRIPT STUFF:
√ REPLACE INTEGRATOR RECURRENTTRANSFERMECHANISM WITH TRANSFERMECHANISM IN INTEGRATOR MODE
  OR TRY USING LCA with DECAY?
- CHECK THAT VERSION WITH TRANSFERMECHANISM FOR CONTEXT PRODUCES CORRECT EM ENTRIES PER PREVOUS BENCHMARKING
- DEBUG LEARNING

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

import numpy as np
import graph_scheduler as gs
from importlib import import_module
from enum import IntEnum
import matplotlib.pyplot as plt
import torch
torch.manual_seed(0)
from psyneulink import *
from psyneulink._typing import Union, Literal

from ScriptControl import (MODEL_PARAMS, CONSTRUCT_MODEL, DISPLAY_MODEL, RUN_MODEL,
                           REPORT_OUTPUT, REPORT_PROGRESS, PRINT_RESULTS, SAVE_RESULTS, PLOT_RESULTS)
import Environment
import_module(MODEL_PARAMS)
model_params = import_module(MODEL_PARAMS).model_params

#region  TASK ENVIRONMENT
# ======================================================================================================================
#                                                   TASK ENVIRONMENT
# ======================================================================================================================

dataset = Environment.generate_dataset(condition=model_params['curriculum_type'],)
if model_params['num_stims'] is ALL:
    INPUTS = dataset.xs.numpy()
    TARGETS = dataset.ys.numpy()
else:
    INPUTS = dataset.xs.numpy()[:model_params['num_stims']]
    TARGETS = dataset.ys.numpy()[:model_params['num_stims']]
TOTAL_NUM_STIMS = len(INPUTS)

#endregion

#region  MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

# EM structural params:
EMFieldsIndex = IntEnum('EMFields',
                        ['STATE',
                         'CONTEXT',
                         'PREVIOUS STATE'],
                        start=0)
state_retrieval_weight = 0
RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

if is_numeric_scalar(model_params['softmax_temperature']):      # translate to gain of softmax retrieval function
    retrieval_softmax_gain = 1/model_params['softmax_temperature']
else:                                                           # pass along ADAPTIVE or CONTROL spec
    retrieval_softmax_gain = model_params['softmax_temperature']

if model_params['memory_capacity'] is ALL:
    memory_capacity =  TOTAL_NUM_STIMS
elif not isinstance(model_params['memory_capacity'], int):
    raise ValueError(f"memory_capacity must be an integer or ALL; got {model_params['memory_capacity']}")

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
                    retrieval_softmax_gain=retrieval_softmax_gain,
                    retrieval_softmax_threshold=model_params['softmax_threshold'],
                    state_retrieval_weight:Union[float,int]=state_retrieval_weight,
                    previous_state_retrieval_weight:Union[float,int]=model_params['state_weight'],
                    context_retrieval_weight:Union[float,int]=model_params['context_weight'],
                    normalize_field_weights = model_params['normalize_field_weights'],
                    concatenate_queries = model_params['concatenate_queries'],
                    learn_field_weights = model_params['learn_field_weights'],
                    memory_capacity = memory_capacity,
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

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_input_name, input_shapes=state_size)
    # context_layer = ProcessingMechanism(name=context_name, input_shapes=context_size)
    context_layer = TransferMechanism(name=context_name,
                                      input_shapes=context_size,
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
                       # field_names=[state_input_name,
                       #              previous_state_input_name,
                       #              context_name,
                       #              ],
                       # field_weights=(state_retrieval_weight,
                       #                previous_state_retrieval_weight,
                       #                context_retrieval_weight
                       #                ),
                       field_names=[previous_state_input_name,
                                    context_name,
                                    state_input_name,
                                    ],
                       field_weights=(previous_state_retrieval_weight,
                                      context_retrieval_weight,
                                      state_retrieval_weight,
                                      ),
                       normalize_field_weights=normalize_field_weights,
                       concatenate_queries=concatenate_queries,
                       learn_field_weights=learn_field_weights,
                       learning_rate=learning_rate,
                       enable_learning=enable_learning,
                       device=device
                       )

    # # TO GET SHOW_GRAPH FOR PNL LEARNING:
    # inputs = {em.nodes['CONTEXT [QUERY]']: [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
    #           em.nodes['PREVIOUS STATE [QUERY]']: [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
    #           em.nodes['STATE [VALUE]']: [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]}
    # em.learn(inputs=inputs, execution_mode=ExecutionMode.Python)
    # em.show_graph(show_learning=True)


    prediction_layer = ProcessingMechanism(name=prediction_layer_name, input_shapes=state_size)

    
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
        print(f"Constructing '{model_params['name']}'...")
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
            print(f"\n**************\n BATCH: {kwargs['minibatch']}\n**************\n")
            print(kwargs)
            print('\nContext internal: \n', model.nodes['CONTEXT'].function.parameters.value.get(kwargs['context']))
            print('\nContext hidden: \n', model.nodes['CONTEXT'].parameters.value.get(kwargs['context']))
            print('\nContext for EM: \n',
                  model.nodes['EM'].nodes['CONTEXT [QUERY]'].parameters.value.get(kwargs['context']))
            print('\nPrediction: \n',
                  model.nodes['PREDICTION'].parameters.value.get(kwargs['context']))
            # print('\nLoss: \n',
            #       model.parameters.minibatch_loss._get(kwargs['context']))
            print('\nProjections from context to EM: \n', model.projections[7].parameters.matrix.get(kwargs['context']))
            print('\nEM Memory: \n', model.nodes['EM'].parameters.memory.get(model.name))

        if INPUTS[0][9]:
            sequence_context = 'context 1'
        else:
            sequence_context = 'context 2'
        if INPUTS[1][1]:
            sequence_state = 'state 1'
        else:
            sequence_state = 'state 2'

        print(f"Running '{model_params['name']}' with {MODEL_PARAMS} for {model_params['num_stims']} stims "
              f"using {model_params['curriculum_type']} training starting with {sequence_context}, {sequence_state}...")
        context = model_params['name']
        start_time = timeit.default_timer()
        model.learn(inputs={model_params['state_input_layer_name']:INPUTS},
                  # report_output=REPORT_OUTPUT,
                  # report_progress=REPORT_PROGRESS
                  #   call_after_minibatch=print('Projections from context to EM: ',
                  #                              model.projections[7].parameters.matrix.get(context)),
                  #                              # model.projections[7].matrix)
                  #   call_after_minibatch=print_stuff,
                  #   optimizations_per_minibatch=model_params['num_optimization_steps'],
                    synch_projection_matrices_with_torch=model_params['synch_weights'],
                    synch_node_values_with_torch=model_params['synch_values'],
                    synch_results_with_torch=model_params['synch_results'],
                    learning_rate=model_params['learning_rate'],
                    execution_mode= model_params['execution_mode'],
                    # minibatch_size=1,
                    # epochs=1
                  )
        stop_time = timeit.default_timer()
        print(f"Elapsed time: {stop_time - start_time}")
        # if DISPLAY_MODEL is not None:
        #     model.show_graph(**DISPLAY_MODEL)
        if PRINT_RESULTS:
            print("MEMORY:")
            print(np.round(model.nodes['EM'].parameters.memory.get(model.name),3))
            # model.run(inputs={model_params["state_input_layer_name"]:INPUTS[TOTAL_NUM_STIMS-1]},
            #           # report_output=REPORT_OUTPUT,
            #           # report_progress=REPORT_PROGRESS
            #           )
            print("CONTEXT INPUT:")
            print(np.round(model.nodes['CONTEXT'].parameters.variable.get(model.name),3))
            print("CONTEXT OUTPUT:")
            print(np.round(model.nodes['CONTEXT'].parameters.value.get(model.name),3))
            print("STATE:")
            print(np.round(model.nodes['STATE'].parameters.value.get(model.name),3))
            print("PREDICTION:")
            print(np.round(model.nodes['PREDICTION'].parameters.value.get(model.name),3))
            # print("CONTEXT WEIGHTS:")
            # print(model.projections[7].parameters.matrix.get(model.name))


            def eval_weights(weight_mat):
                # checks whether only 5 weights are updated.
                weight_mat -= np.eye(11)
                col_sum = weight_mat.sum(1)
                row_sum = weight_mat.sum(0)
                return np.max([(row_sum != 0).sum(), (col_sum != 0).sum()])
            print(eval_weights(model.projections[7].parameters.matrix.get(model.name)))

        if SAVE_RESULTS:
            np.save('EGO PREDICTIONS', model.results)
            np.save('EGO INPUTS', INPUTS)
            np.save('EGO TARGETS', TARGETS)

        if PLOT_RESULTS:
            fig, axes = plt.subplots(3, 1, figsize=(5, 12))
            # Weight matrix
            axes[0].imshow(model.projections[7].parameters.matrix.get(model.name), interpolation=None)
            # L1 of loss
            axes[1].plot((1 - np.abs(model.results[1:TOTAL_NUM_STIMS,2]-TARGETS[:TOTAL_NUM_STIMS-1])).sum(-1))
            axes[1].set_xlabel('Stimuli')
            axes[1].set_ylabel(model_params['loss_spec'])
            # Logit of loss
            axes[2].plot( (model.results[1:TOTAL_NUM_STIMS,2]*TARGETS[:TOTAL_NUM_STIMS-1]).sum(-1) )
            axes[2].set_xlabel('Stimuli')
            axes[2].set_ylabel('Correct Logit')
            plt.suptitle(f"{model_params['curriculum_type']} Training")
            plt.show()
            # plt.savefig('../show_graph OUTPUT/EGO PLOT.png')
    #endregion
