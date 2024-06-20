# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# TODO:

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
from psyneulink._typing import Union, Literal
from psyneulink.core.scheduling.condition import Any, And, AllHaveRun, AtRunStart

# Settings for running script:

NUM_EXP_SEQS = 5               # Number of sequences to run in EXPERIENCE Phase (includes baseline + revaluation)
NUM_PRED_TRIALS = 10           # Number of trials (ROLL OUTS) to run in PREDICTION Phase

CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = (                      # Only one of the following can be uncommented:
    None                             # suppress display of model
    # {}                               # show simple visual display of model
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
    n_participants=58,
    n_simulations = 100, # number of rollouts per participant
    n_steps = 3, # number of steps per rollout
    state_d = 7, # length of state vector
    context_d = 7, # length of context vector
    time_d = 25, # length of time vector
    self_excitation = .25, # rate at which old context is carried over to new context
    input_weight = .5, # rate at which state is integrated into new context
    retrieved_context_weight = .25, # rate at which context retrieved from EM is integrated into new context
    time_noise=.01,# noise std for time integrator (drift is set to 0)
    state_weight = .5, # weight of the state used during memory retrieval
    context_weight = .3, # weight of the context used during memory retrieval
    time_weight = .2, # weight of the time used during memory retrieval
    temperature = .05 # temperature of the softmax used during memory retrieval (smaller means more argmax-like
)

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model CSW"
STATE_INPUT_LAYER_NAME = "STATE"
CONTEXT_LAYER_NAME = 'CONTEXT'
EM_NAME = "EM"
RESPONSE_LAYER_NAME = "RESPONSE"

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
STATE_WEIGHT = model_params['input_weight']  # rate at which external vs. memory state are integrated in context_layer
CONTEXT_INTEGRATION_RATE = model_params['retrieved_context_weight']  # rate at which retrieved context (from EM)
                                                                     # is integrated into context_layer
assert (model_params['retrieved_context_weight'] + STATE_WEIGHT + CONTEXT_INTEGRATION_RATE) == 1,\
    (f"Sum of STATE_WEIGHT ({STATE_WEIGHT}), CONTEXT_INTEGRATION_RATE ({CONTEXT_INTEGRATION_RATE}), "
     f"and RETRIEVED_CONTEXT_WEIGHT ({model_params['retrieved_context_weight']}) must equal 1")

# EM retrieval
STATE_RETRIEVAL_WEIGHT = model_params['state_weight']     # weight of state field in retrieval from EM
CONTEXT_RETRIEVAL_WEIGHT = model_params['context_weight'] # weight of context field in retrieval from EM
RETRIEVAL_SOFTMAX_GAIN = 1/model_params['temperature']    # gain on softmax retrieval function

RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

#endregion

#region ENVIRONMENT
# ======================================================================================================================
#                                                   ENVIRONMENT
# ======================================================================================================================

# Task environment:
NUM_STIM_PER_SEQ = model_params['n_steps'] # number of stimuli in a sequence
NUM_SEQS = model_params['num_seqs']  # total number of sequences to be executed (to set size of EM)

STIM_SEQS = [list(range(1,NUM_STIM_PER_SEQ*2,2)),
            list(range(2,NUM_STIM_PER_SEQ*2+1,2))]
CURRICULUM_TYE = 'blocked'     # 'blocked' or 'interleaved'

#endregion

#region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model(model_name:str=MODEL_NAME,

                    # Inputs:
                    state_input_name:str=STATE_INPUT_LAYER_NAME,
                    state_size:int=STATE_SIZE,

                    # Context processing:
                    context_name:str=CONTEXT_LAYER_NAME,
                    state_weight:Union[float,int]=STATE_WEIGHT,
                    context_integration_rate:Union[float,int]=CONTEXT_INTEGRATION_RATE,

                    # EM:
                    em_name:str=EM_NAME,
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    context_retrieval_weight:Union[float,int]=CONTEXT_RETRIEVAL_WEIGHT,

                    # Output / decision processing:
                    response_layer_name:str=RESPONSE_LAYER_NAME,

                    )->Composition:

    # Apportionment of contributions of state (actual or em) vs. context (em) to context_layer integration:

    # FIX: THIS IS FOR MDP;  NEEDS TO BE REVISED FOR CSW
    # state input (EXPERIENCE) -\
    #                            --> state_weight -------\
    # state from em (PREDICT)---/                         -> * (context_integration_rate) -----\
    #                          /-----> context_weight ---/                                      --> context
    # context from em --------/      (=1- state_weight)                                        /
    #                                                    /---> 1 - context_integration_rate --/
    # context from prev. cycle -------------------------/

    assert 0 <= context_integration_rate <= 1,\
        f"context_retrieval_weight must be a number from 0 to 1"
    assert 0 <= state_weight <= 1,\
        f"context_retrieval_weight must be a number from 0 to 1"
    context_weight = 1 - state_weight
    state_weight *= context_integration_rate
    context_weight *= context_integration_rate

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, size=state_size)
    context_layer = RecurrentTransferMechanism(name=context_name,
                                               size=state_size,
                                               auto=1-context_integration_rate,
                                               hetero=0.0)
    em = EMComposition(name=em_name,
                       memory_template=[[0] * state_size,   # state
                                        [0] * state_size],   # context
                       memory_fill=(0,.01),
                       memory_capacity=NUM_SEQS,
                       softmax_gain=1.0,
                       # Input Nodes:
                       field_names=[state_input_name,
                                    context_name],
                       field_weights=(state_retrieval_weight,
                                      context_retrieval_weight))

    response_layer = ProcessingMechanism(name=response_layer_name)

    
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    

    EGO_comp = Composition(name=model_name,
                           # # Terminate a Task.PREDICT trial after response_layer executes if a reward is retrieved
                           # termination_processing={
                           #     # TimeScale.TRIAL: And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                      Condition(lambda: retrieved_reward_layer.value),
                           #     #                      JustRan(response_layer))}
                           #     # CRASHES:
                           #     # TimeScale.TRIAL: Any(And(Condition(lambda: task_input_layer.value == Task.EXPERIENCE),
                           #     #                          JustRan(em)),
                           #     #                      And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                          Condition(lambda: retrieved_reward_layer.value),
                           #     #                          JustRan(response_layer)))}
                           #     TimeScale.TRIAL: Any(And(Condition(lambda: task_input_layer.value == Task.EXPERIENCE),
                           #                              AllHaveRun()),
                           #                          And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #                              Condition(lambda: retrieved_reward_layer.value),
                           #                              AllHaveRun()))}
                           )

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([task_input_layer,
                        state_input_layer,
                        time_input_layer,
                        attend_external_layer,
                        attend_memory_layer,
                        context_layer,
                        reward_input_layer,
                        em,
                        retrieved_reward_layer,
                        control_layer,
                        response_layer
                        ])
    EGO_comp.exclude_node_roles(task_input_layer, NodeRole.OUTPUT)

    # Projections:
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding --------------------------------------------------------------------------------
    # state -> em
    EGO_comp.add_projection(MappingProjection(state_input_layer, em.nodes[state_input_name + QUERY]))
    # time -> em
    EGO_comp.add_projection(MappingProjection(time_input_layer, em.nodes[time_input_name + QUERY]))
    # context -> em
    EGO_comp.add_projection(MappingProjection(context_layer, em.nodes[context_name + QUERY]))
    # reward -> em
    EGO_comp.add_projection(MappingProjection(reward_input_layer, em.nodes[reward_input_name + VALUE]))

    # Inputs to Context ---------------------------------------------------------------------------
    # actual state -> attend_external_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer, attend_external_layer))
    # retrieved state -> attend_memory_layer
    EGO_comp.add_projection(MappingProjection(em.nodes[STATE_INPUT_LAYER_NAME + RETRIEVED],
                                              attend_memory_layer))
    # attend_external_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attend_external_layer, context_layer,
                                              matrix=np.eye(STATE_SIZE) * state_weight))
    # attend_memory_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attend_memory_layer, context_layer,
                                              matrix=np.eye(STATE_SIZE) * state_weight))
    # retrieved context -> context_layer
    EGO_comp.add_projection(MappingProjection(em.nodes[CONTEXT_LAYER_NAME + RETRIEVED], context_layer,
                                              matrix=np.eye(STATE_SIZE) * context_weight))

    # Decision pathway ---------------------------------------------------------------------------
    # retrieved reward -> retrieved reward
    EGO_comp.add_projection(MappingProjection(em.nodes[REWARD_INPUT_LAYER_NAME + RETRIEVED],
                                              retrieved_reward_layer))
    # retrieved reward -> decision layer
    EGO_comp.add_projection(MappingProjection(retrieved_reward_layer, response_layer))

    # Validate construction
    proj_from_retrieved_reward_to_control = control_layer.input_ports[1].path_afferents[0]
    assert proj_from_retrieved_reward_to_control._feedback == True
    assert proj_from_retrieved_reward_to_control in EGO_comp.feedback_projections # retrieved_reward feedback
    assert context_layer.input_port.path_afferents[0].sender.owner == context_layer # recurrent projection
    assert context_layer.input_port.path_afferents[0].parameters.matrix.get()[0][0] == 1-context_integration_rate
    assert context_layer.input_port.path_afferents[1].sender.owner == attend_external_layer # external state
    assert context_layer.input_port.path_afferents[1].parameters.matrix.get()[0][0] == state_weight
    assert context_layer.input_port.path_afferents[2].sender.owner == attend_memory_layer # memory of state
    assert context_layer.input_port.path_afferents[2].parameters.matrix.get()[0][0] == state_weight
    assert context_layer.input_port.path_afferents[3].sender.owner == em.nodes[CONTEXT_LAYER_NAME + RETRIEVED]  # memory of
    # context
    assert context_layer.input_port.path_afferents[3].parameters.matrix.get()[0][0] == context_weight
    assert control_layer.input_ports[0].path_afferents[0].sender.owner == task_input_layer
    assert control_layer.input_ports[1].path_afferents[0].sender.owner == retrieved_reward_layer

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

    if DISPLAY_MODEL is not None:
        if model:
            model.show_graph(**DISPLAY_MODEL)
        else:
            print("Model not yet constructed")

    if RUN_MODEL:
        experience_inputs = build_experience_inputs(state_size=STATE_SIZE,
                                                    time_drift_rate=TIME_DRIFT_RATE,
                                                    num_baseline_seqs=NUM_BASELINE_SEQS,
                                                    num_revaluation_seqs=NUM_REVALUATION_SEQS,
                                                    reward_vals=REWARD_VALS,
                                                    CURRICULUM_TYE=CURRICULUM_TYE,
                                                    ratio=RATIO,
                                                    stim_seqs=STIM_SEQS)
        input_layers = [TIME_INPUT_LAYER_NAME,
                        TASK_INPUT_LAYER_NAME,
                        STATE_INPUT_LAYER_NAME,
                        REWARD_INPUT_LAYER_NAME]

        # Experience Phase
        print(f"Presenting {model.name} with {TOTAL_NUM_EXPERIENCE_STIMS} EXPERIENCE stimuli")
        model.run(inputs={k: v for k, v in zip(input_layers, experience_inputs)},
                  execution_mode=EXECUTION_MODE,
                  report_output=REPORT_OUTPUT,
                  report_progress=REPORT_PROGRESS)

        # Prediction Phase
        prediction_inputs = build_prediction_inputs(state_size=STATE_SIZE,
                                                    time_drift_rate=TIME_DRIFT_RATE,
                                                    num_roll_outs_per_stim=int(NUM_ROLL_OUTS / 2),
                                                    stim_seqs=STIM_SEQS,
                                                    reward_vals=REWARD_VALS,
                                                    seq_type=PREDICT_SEQ_TYPE)
        print(f"Running {model.name} for {NUM_ROLL_OUTS} PREDICT (ROLL OUT) trials")
        model.termination_processing = {
            TimeScale.TRIAL: And(Condition(lambda: model.nodes[TASK_INPUT_LAYER_NAME].value == Task.PREDICT),
                                 Condition(lambda: model.nodes[RETRIEVED_REWARD_NAME].value),
                                 # JustRan(model.nodes[RESPONSE_LAYER_NAME])
                                 AllHaveRun()
                                 )
        }
        model.run(inputs={k: v for k, v in zip(input_layers, prediction_inputs)},
                  report_output=REPORT_OUTPUT,
                  report_progress=REPORT_PROGRESS
                  )

        if PRINT_RESULTS:
            print(f"Predicted reward for last stimulus: {model.results}")
    #endregion