# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# TODO:

# REPLACE INTEGRATOR RECURRENTTRANSFERMECHANISM WITH TRANSFERMECHANISM IN INTEGRATOR MODE

# ADD PREVIOUS STATES
# ADD next_state to EM and control to support that
# - CONTROL FLOW:
#   - UPDATE CONTEXT LAYER:  INTEGRATE CURRENT STATE IN CONTEXT LAYER
#   - USE UPDATED CONTEXT + CURRENT STATE TO RETRIEVE PREDICTED NEXT STATE
#   - GET NEXT STATE
#   - ENCODE "CURRENT" (I.E., PREVIOUS) STATE + "NEXT" (NOW ACTUALLY CURRENT) STATE + CONTEXT (PRIOR TO
#           INTEGRATION OF "NEXT") INTO EM

# - CONTROL FLOW (FROM VANTAGE OF "NEXT" STATE):
#   - USE PREVIOUS STATE + CONTEXT (PRE-INTEGRATION) TO RETRIEVE PREDICTION OF CURRENT STATE
#   - ENCODE PREVIOUS STATE + CURRENT STATE (INPUT) + CONTEXT (PRE-INTEGRATION) INTO EM
#   - UPDATE CONTEXT LAYER:  INTEGRATE CURRENT STATE (INPUT) IN CONTEXT LAYER:
#   SO:
#   - EM SHOULD EXECUTE FIRST:
#     - RETRIEVE USING PREVIOUS STATE NODE AND CONTEXT (PRE-INTEGRATION) TO RETRIEVE PREDICTED CURRENT STATE
#     - STORE VALUES OF PREVIOUS STATE, CURRENT STATE (INPUT) AND CONTEXT (PRE-INTEGRATION) INTO EM
#   - THEN CONTEXT LAYER SHOULD EXECUTE, USING CURRENT INPUT TO INTEGRATE INTO CONTEXT LAYER



# FIX: TERMINATION CONDITION IS GETTING TRIGGED AFTER 1st TRIAL

# FOR INPUT NODES: scheduler.add_condition(A, BeforeNCalls(A,1)
# Termination: AfterNCalls(Ctl,2)

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

import numpy as np
from enum import IntEnum

from psyneulink import *
from psyneulink._typing import Union

# Settings for running script:

MEMORY_CAPACITY = 5
CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = (                      # Only one of the following can be uncommented:
    # None                             # suppress display of model
    {}                               # show simple visual display of model
    # {'show_node_structure': True}    # show detailed view of node structures and projections
)
RUN_MODEL = True                       # True => run the model
# RUN_MODEL = False                      # False => don't run the model
EXECUTION_MODE = ExecutionMode.Python
# EXECUTION_MODE = ExecutionMode.PyTorch
ANALYZE_RESULTS = False                # True => output analysis of results of run
# REPORT_OUTPUT = ReportOutput.FULL     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_OUTPUT = ReportOutput.OFF     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
PRINT_RESULTS = False                  # print model.results after execution
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution


#region   PARAMETERS
# ======================================================================================================================
#                                                   PARAMETERS
# ======================================================================================================================

# PyTorch Version Parameters:
model_params = dict(
    state_d = 11, # length of state vector
    integrator_d = 11, # length of integrator vector
    context_d = 11, # length of context vector
    integration_rate = .69, # rate at which state is integrated into new context
    state_weight = .5, # weight of the state used during memory retrieval
    context_weight = .5, # weight of the context used during memory retrieval
    temperature = .01 # temperature of the softmax used during memory retrieval (smaller means more argmax-like
)

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model CSW"
STATE_INPUT_LAYER_NAME = "STATE"
PREVIOUS_STATE_LAYER_NAME = "PREVIOUS STATE"
INTEGRATOR_LAYER_NAME = 'INTEGRATOR'
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
INTEGRATOR_SIZE = model_params['integrator_d']  # length of state vector
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
# INPUTS = [env_inputs[i][:10] for i in range(len(env_inputs))]


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
                    previous_state_size:int=STATE_SIZE,

                    # Integrator:
                    integrator_name:str=INTEGRATOR_LAYER_NAME,
                    integrator_size:int=INTEGRATOR_SIZE,
                    integration_rate:Union[float,int]=INTEGRATION_RATE,

                    # Context representation (learned):
                    context_name:str=CONTEXT_LAYER_NAME,
                    context_size:Union[float,int]=CONTEXT_SIZE,

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

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_input_name, input_shapes=state_size)
    integrator_layer = RecurrentTransferMechanism(name=integrator_name,
                                                  function=Tanh,
                                                  input_shapes=integrator_size,
                                                  auto=1-integration_rate,
                                                  hetero=0.0)
    context_layer = ProcessingMechanism(name=context_name, input_shapes=context_size)

    em = EMComposition(name=em_name,
                       memory_template=[[0] * state_size,   # state
                                        [0] * state_size,   # previous state
                                        [0] * state_size],  # context
                       # memory_fill=(0,.01),
                       memory_capacity=MEMORY_CAPACITY,
                       memory_decay_rate=0,
                       softmax_gain=1.0,
                       # Input Nodes:
                       field_names=[state_input_name,
                                    previous_state_input_name,
                                    context_name,
                                    ],
                       field_weights=(state_retrieval_weight,
                                      previous_state_retrieval_weight,
                                      context_retrieval_weight
                                      )
                       )

    prediction_layer = ProcessingMechanism(name=prediction_layer_name,
                                           input_shapes=state_size)

    
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    EGO_comp = Composition(name=model_name,
                           # # Terminate a Task.PREDICT trial after prediction_layer executes if a reward is retrieved
                           # termination_processing={
                           #     # TimeScale.TRIAL: And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                      Condition(lambda: retrieved_reward_layer.value),
                           #     #                      JustRan(prediction_layer))}
                           #     # CRASHES:
                           #     # TimeScale.TRIAL: Any(And(Condition(lambda: task_input_layer.value == Task.EXPERIENCE),
                           #     #                          JustRan(em)),
                           #     #                      And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                          Condition(lambda: retrieved_reward_layer.value),
                           #     #                          JustRan(prediction_layer)))}
                           #     TimeScale.TRIAL: Any(And(Condition(lambda: task_input_layer.value == Task.EXPERIENCE),
                           #                              AllHaveRun()),
                           #                          And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #                              Condition(lambda: retrieved_reward_layer.value),
                           #                              AllHaveRun()))}
                           )

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([state_input_layer,
                        previous_state_layer,
                        integrator_layer,
                        context_layer,
                        em,
                        prediction_layer])

    # Projections:
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding & retrieval  --------------------------------------------------------------------------------
    # state_input -> em (retrieval)
    EGO_comp.add_projection(MappingProjection(state_input_layer,
                                              em.nodes[state_input_name + VALUE],
                                              matrix=IDENTITY_MATRIX))

    # previous_state -> em (retrieval)
    EGO_comp.add_projection(MappingProjection(previous_state_layer,
                                              em.nodes[previous_state_input_name + QUERY],
                                              matrix=IDENTITY_MATRIX))
    # context -> em (retrieval)
    EGO_comp.add_projection(MappingProjection(context_layer,
                                              em.nodes[context_name + QUERY],
                                              matrix=IDENTITY_MATRIX))

    # Inputs to previous_state and context -------------------------------------------------------------------
    # state -> previous_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer,
                                              previous_state_layer,
                                              matrix=IDENTITY_MATRIX))
    # state -> integrator_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer,
                                              integrator_layer,
                                              matrix=np.eye(STATE_SIZE) * integration_rate))

    # integrator_layer -> context_layer (learnable)
    EGO_comp.add_projection(MappingProjection(integrator_layer,
                                              context_layer,
                                              matrix=IDENTITY_MATRIX))

    # Response pathway ---------------------------------------------------------------------------------------
    # retrieved state -> prediction_layer
    EGO_comp.add_projection(MappingProjection(em.nodes[state_input_name + RETRIEVED],
                                              prediction_layer))


    EGO_comp.scheduler.add_condition(em, BeforeNodes(previous_state_layer, integrator_layer))

    # Validate construction
    assert integrator_layer.input_port.path_afferents[0].sender.owner == integrator_layer # recurrent projection
    assert integrator_layer.input_port.path_afferents[0].parameters.matrix.get()[0][0] == 1-integration_rate
    assert integrator_layer.input_port.path_afferents[1].sender.owner == state_input_layer  #

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
        # gs.output_graph_image(model.scheduler.graph, 'show_graph OUTPUT/EGO_comp-scheduler.png')

    if DISPLAY_MODEL is not None:
        if model:
            model.show_graph(**DISPLAY_MODEL)
        else:
            print("Model not yet constructed")

    if RUN_MODEL:
        # print("MODEL NOT YET FULLY EXECUTABLE")
        print(f'Running {MODEL_NAME}')
        model.run(inputs={STATE_INPUT_LAYER_NAME:INPUTS},
                  # report_output=REPORT_OUTPUT,
                  # report_progress=REPORT_PROGRESS
                  )
        print(model.nodes['EM'].parameters.memory.get(context=MODEL_NAME))
        if PRINT_RESULTS:
            print("MODEL NOT YET FULLY EXECUTABLE SO NO RESULTS")
    #endregion
