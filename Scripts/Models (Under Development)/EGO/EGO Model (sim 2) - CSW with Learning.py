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

# Settings for running script:

MEMORY_CAPACITY = 5
CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL =  (                     # Only one of the following can be uncommented:
    None                             # suppress display of model
    # {                                  # show simple visual display of model
    # 'show_pytorch': True,            # show pytorch graph of model
    #  'show_learning': True
    # # 'show_projections_not_in_composition': True,
    # # 'exclude_from_gradient_calc_style': 'dashed'# show target mechanisms for learning
    # # {'show_node_structure': True     # show detailed view of node structures and projections
    # }
)
RUN_MODEL = True                       # True => run the model
# RUN_MODEL = False                      # False => don't run the model
# EXECUTION_MODE = ExecutionMode.Python
EXECUTION_MODE = ExecutionMode.PyTorch
# ANALYZE_RESULTS = False                # True => output analysis of results of run
# REPORT_OUTPUT = ReportOutput.FULL     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_OUTPUT = ReportOutput.OFF     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
PRINT_RESULTS = True                  # print model.results after execution
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution


#region   PARAMETERS
# ======================================================================================================================
#                                                   PARAMETERS
# ======================================================================================================================

# PyTorch Version Parameters:
model_params = dict(
    state_d = 11, # length of state vector
    previous_state_d = 11, # length of state vector
    context_d = 11, # length of context vector
    integration_rate = .69, # rate at which state is integrated into new context
    state_weight = .5, # weight of the state used during memory retrieval
    context_weight = .5, # weight of the context used during memory retrieval
    temperature = .1 # temperature of the softmax used during memory retrieval (smaller means more argmax-like
)

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model CSW"
STATE_INPUT_LAYER_NAME = "STATE"
PREVIOUS_STATE_LAYER_NAME = "PREVIOUS STATE"
CONTEXT_LAYER_NAME = 'CONTEXT'

EM_NAME = "EM"
PREDICTION_LAYER_NAME = "PREDICTION"

EMFieldsIndex = IntEnum('EMFields',
                        ['STATE',
                         'CONTEXT',
                         'PREVIOUS_STATE'],
                        start=0)


# CONSTRUCTION PARAMETERS

# Layer sizes:
STATE_SIZE = model_params['state_d']  # length of state vector
CONTEXT_SIZE = model_params['context_d']  # length of state vector

# Context processing:
INTEGRATION_RATE = model_params['integration_rate']  # rate at which state is integrated into integrator layer

# EM retrieval
STATE_RETRIEVAL_WEIGHT = 0
PREVIOUS_STATE_RETRIEVAL_WEIGHT = model_params['state_weight']     # weight of state field in retrieval from EM
CONTEXT_RETRIEVAL_WEIGHT = model_params['context_weight'] # weight of context field in retrieval from EM
RETRIEVAL_SOFTMAX_GAIN = 1/model_params['temperature']    # gain on softmax retrieval function


RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

#endregion

#region ENVIRONMENT
# ======================================================================================================================
#                                                   ENVIRONMENT
# ======================================================================================================================

# Task environment:
import Environment
CURRICULUM_TYPE = 'Blocked'     # 'Blocked' or 'Interleaved'
INPUTS = Environment.generate_dataset(condition=CURRICULUM_TYPE).xs.numpy()[:5]

#endregion

#region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model(model_name:str=MODEL_NAME,

                    # Input layer:
                    state_input_name:str=STATE_INPUT_LAYER_NAME,
                    state_size:int=STATE_SIZE,

                    # Previous state
                    previous_state_input_name:str=PREVIOUS_STATE_LAYER_NAME,

                    # Context representation (learned):
                    context_name:str=CONTEXT_LAYER_NAME,
                    context_size:Union[float,int]=CONTEXT_SIZE,
                    integration_rate:float=INTEGRATION_RATE,

                    # EM:
                    em_name:str=EM_NAME,
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    previous_state_retrieval_weight:Union[float,int]=PREVIOUS_STATE_RETRIEVAL_WEIGHT,
                    context_retrieval_weight:Union[float,int]=CONTEXT_RETRIEVAL_WEIGHT,

                    # Output / decision processing:
                    prediction_layer_name:str=PREDICTION_LAYER_NAME,

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
                       # memory_fill=(0,.01),
                       memory_capacity=MEMORY_CAPACITY,
                       memory_decay_rate=0,
                       softmax_gain=retrieval_softmax_gain,
                       # Input Nodes:
                       field_names=[state_input_name,
                                    previous_state_input_name,
                                    context_name,
                                    ],
                       field_weights=(state_retrieval_weight,
                                      previous_state_retrieval_weight,
                                      context_retrieval_weight
                                      ),
                       learn_field_weights=False,
                       # enable_learning=True,
                       enable_learning=[True, False, False]
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
                                   learning_rate=.5,
                                   name=model_name)

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
        print(f'Constructing {MODEL_NAME}')
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
        def print_stuff(**kwargs):
            print(kwargs)
            print('Context internal: \n', model.nodes['CONTEXT'].parameters.value.get(kwargs['context']))
            print('Context for EM: \n',
                  model.nodes['EM'].nodes['CONTEXT [QUERY]'].parameters.value.get(kwargs['context']))
            print('Prediction: \n',
                  model.nodes['PREDICTION'].parameters.value.get(kwargs['context']))
            print('Projections from context to EM: \n', model.projections[7].parameters.matrix.get(kwargs['context']))

        # print("MODEL NOT YET FULLY EXECUTABLE")
        print(f'Running {MODEL_NAME}')
        context = MODEL_NAME
        model.learn(inputs={STATE_INPUT_LAYER_NAME:INPUTS},
                  # report_output=REPORT_OUTPUT,
                  # report_progress=REPORT_PROGRESS
                  #   call_after_minibatch=print('Projections from context to EM: ',
                  #                              model.projections[7].parameters.matrix.get(context)),
                  #                              # model.projections[7].matrix)
                    call_after_minibatch=print_stuff,
                    optimizations_per_minibatch=1,
                    # minibatch_size=3,
                    learning_rate=.5
                  )
        if DISPLAY_MODEL is not None:
            model.show_graph(**DISPLAY_MODEL)
        if PRINT_RESULTS:
            print("MEMORY:")
            print(model.nodes['EM'].parameters.memory.get(model.name))
            model.run(inputs={STATE_INPUT_LAYER_NAME:INPUTS[4]},
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

    #endregion
