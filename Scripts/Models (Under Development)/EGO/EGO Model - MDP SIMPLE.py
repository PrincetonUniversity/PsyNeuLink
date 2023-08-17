"""
QUESTIONS:

-

**Overview**
------------

This implements a model of...

The model is an example of...

The script contains methods to construct, train, and run the model, and analyze the results of its execution:

* `construct_model <EGO.construct_model>`:
  takes as arguments parameters used to construct the model;  for convenience, defaults are defined below,
  (under "Construction parameters")

* `train_network <EGO.train_network>`:
  takes as arguments the feedforward neural network Composition (FFN_COMPOSITION) and number of epochs to train.
  Note: learning_rate is set at construction (can specify using LEARNING_RATE under "Training parameters" below).

* `run_model <EGO.run_model>`:
  takes as arguments the drift rate in the temporal context vector to be applied on each trial,
  and the number of trials to execute, as well as reporting and animation specifications
  (see "Execution parameters").

* `analyze_results <EGO.analyze_results>`:
  takes as arguments the results of executing the model, and optionally a number of trials and EGO_level to analyze;
  returns d-prime statistics and plots results for different conditions at each EGO_level executed.


**The Model**
-------------

The model is comprised of...

.. _EGO_Fig:

.. figure:: _static/N-Back_Model_movie.gif
   :align: left
   :alt: N-Back Model Animation

.. _EGO_model_composition:

*EGO_model Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~

This is comprised of three input Mechanisms, and the nested `ffn <EGO_ffn_composition>` `Composition`.

.. _EGO_ffn_composition:

*FFN Composition*
~~~~~~~~~~~~~~~~~

The temporal context is provided by a randomly drifting high dimensional vector that maintains a constant norm (i.e.,
drifts on a sphere).  The FFN is trained, given an n-back level of *n*, to identify when the current stimulus matches
one stored in EM with a temporal context vector that differs by an amount corresponding to *n* time steps of drift.
During n-back performance, the model encodes the current stimulus and temporal context, retrieves an item from EM
that matches the current stimulus, weighted by the similarity of its temporal context vector (i.e., most recent), and
then uses the FFN to evaluate whether it is an n-back match.  The model responds "match" if the FFN detects a match;
otherwise, it either uses the current stimulus and temporal context to retrieve another sample from EM and repeat the
evaluation or, with a fixed probability (hazard rate), it responds "non-match".

The ffn Composition is trained using the train_network() method


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

Sequences of stimuli are constructed either using `SweetPea <https://sites.google.com/view/sweetpea-ai?pli=1>`_
(using the script in stim/SweetPea) or replicate those used in the study by `Kane et al.,
2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_ (from stimulus files in stim/Kane_et_al).

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

import os
import random
import time
import timeit
import numpy as np
from typing import Union
from enum import Enum, IntEnum
from pathlib import Path

from graph_scheduler import *

from psyneulink import *

# Settings for running script:
CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = (                      # Only one of the following can be uncommented:
    # None                             # suppress display of model
    {}                               # show simple visual display of model
    # {'show_node_structure': True}    # show detailed view of node structures and projections
)
RUN_MODEL = True                      # True => run the model
ANALYZE_RESULTS = False                # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.ON       # Sets console output during run
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

#region ========================================= PARAMETERS ===========================================================

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model"
TASK_INPUT_LAYER_NAME = "TASK"
STATE_INPUT_LAYER_NAME = "STATE"
TIME_INPUT_LAYER_NAME = "TIME"
ATTENTION_LAYER_NAME = "STATE ATTENTION"
ATTENTION_LAYER = "ATTENTION"
ACTUAL_STATE_INPUT = 'ACTUAL_STATE_INPUT'
RETRIEVED_STATE_INPUT = 'RETRIEVED_STATE'
CONTEXT_LAYER_NAME = 'CONTEXT'
REWARD_INPUT_LAYER_NAME = "REWARD"
RETRIEVED_TIME_NAME = "RETRIEVED TIME"
RETRIEVED_REWARD_NAME = "RETRIEVED REWARD"
EM_NAME = "EPISODIC MEMORY"
DECISION_LAYER_NAME = "DECISION"

class Task(IntEnum):
    EXPERIENCE = 0
    PREDICT = 1

# CONSTRUCTION PARAMETERS

# Layer sizes:
TASK_SIZE = 1                  # length of task vector
STATE_SIZE = 8                 # length of state vector
TIME_SIZE = 25                 # length of time vector
REWARD_SIZE = 1                # length of reward vector
DECISION_SIZE = 2              # length of decision vector

# Context processing:
STATE_WEIGHT = .1    # rate at which actual vs. retrieved state (from EM) are integrated in context_layer
CONTEXT_INTEGRATION_RATE = .1  # rate at which retrieved context (from EM) is integrated into context_layer
TIME_DRIFT_NOISE = 0.0         # noise used by DriftOnASphereIntegrator (function of Context mech)

# EM retrieval
STATE_RETRIEVAL_WEIGHT = 1     # weight of state field in retrieval from EM
TIME_RETRIEVAL_WEIGHT = 1      # weight of time field in retrieval from EM
CONTEXT_RETRIEVAL_WEIGHT = 1   # weight of context field in retrieval from EM
REWARD_RETRIEVAL_WEIGHT = 0    # weight of reward field in retrieval from EM
RETRIEVAL_SOFTMAX_GAIN = 10    # gain on softmax retrieval function
# RETRIEVAL_HAZARD_RATE = 0.04   # rate of re=sampling of em following non-match determination in a pass through ffn

RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

# Execution parameters:

# Temporal context vector generation as input to time_input_layer of model
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
time_fct = DriftOnASphereIntegrator(initializer=np.random.random(TIME_SIZE - 1),
                                    noise=TIME_DRIFT_NOISE,
                                    dimension=TIME_SIZE)
# Task environment:
NUM_EXPERIENCE_TRIALS = 9      # number of trials for Task.EXPERIENCE (passive encoding into EM)
NUM_PREDICT_TRIALS = 9         # number of trials Task.PREDICT (active retrieval from EM and reward prediction)
NUM_ROLL_OUT = 3               # number of trials of roll-out under OCM control
NUM_TRIALS = NUM_EXPERIENCE_TRIALS + NUM_PREDICT_TRIALS # total number of trials
assert NUM_PREDICT_TRIALS % NUM_ROLL_OUT == 0, \
    f"NUM_PREDICT_TRIALS ({NUM_PREDICT_TRIALS}) " \
    f"must be evenly divisible by NUM_ROLL_OUT ({NUM_ROLL_OUT})"

inputs = {STATE_INPUT_LAYER_NAME: [[1],[2],[3]] * STATE_SIZE * NUM_TRIALS,
          TIME_INPUT_LAYER_NAME: np.array([time_fct(i) for i in range(NUM_TRIALS)]).reshape(NUM_TRIALS,TIME_SIZE,1),
          REWARD_INPUT_LAYER_NAME: [[0],[0],[1]] * REWARD_SIZE * NUM_TRIALS,
          TASK_INPUT_LAYER_NAME: [[Task.EXPERIENCE.value]] * NUM_EXPERIENCE_TRIALS
                                 + [[Task.PREDICT.value]] * NUM_PREDICT_TRIALS}
def gen_baseline_trials_exp1(dim=STATE_SIZE, num_trials=NUM_EXPERIENCE_TRIALS):
    # Generate one-hots
    state_reps = np.eye(dim)
    visited_states, rewards = [], []

    for trial_idx in range(num_trials):
        if np.random.random()<.5:
            visited_states.extend([1,3,5])
            rewards.extend([0,0,10])
        else:
            visited_states.extend([2,4,6])
            rewards.extend([0,0,1])

    # Pick one-hots corresponding to each state
    visited_states = state_reps[visited_states]
    rewards = np.array(rewards)

    return visited_states, rewards
def gen_reward_revaluation_trials_exp1(dim=STATE_SIZE, num_trials=NUM_PREDICT_TRIALS):
    # Generate one-hots
    state_reps = np.eye(dim)
    visited_states, rewards = [], []

    for trial_idx in range(num_trials):
        if np.random.random()<.5:
            visited_states.extend([3,5])
            rewards.extend([0,1])
        else:
            visited_states.extend([4,6])
            rewards.extend([0,10])

    # Pick one-hots corresponding to each state
    visited_states = state_reps[visited_states]
    rewards = np.array(rewards)

    return visited_states, rewards

assert True

def construct_model(model_name:str=MODEL_NAME,

                    # Inputs:
                    task_input_name:str=TASK_INPUT_LAYER_NAME,
                    task_size:int=1,
                    state_input_name:str=STATE_INPUT_LAYER_NAME,
                    state_size:int=STATE_SIZE,
                    time_input_name:str=TIME_INPUT_LAYER_NAME,
                    time_size:int=TIME_SIZE,
                    reward_input_name = REWARD_INPUT_LAYER_NAME,
                    reward_size:int=REWARD_SIZE,

                    # Context processing:
                    attention_layer_name=ATTENTION_LAYER_NAME,
                    context_name:str=CONTEXT_LAYER_NAME,
                    state_weight:Union[float,int]=STATE_WEIGHT,
                    context_integration_rate:Union[float,int]=CONTEXT_INTEGRATION_RATE,

                    # EM:
                    em_name:str=EM_NAME,
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    # retrieval_hazard_rate=RETRIEVAL_HAZARD_RATE,
                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    time_retrieval_weight:Union[float,int]=TIME_RETRIEVAL_WEIGHT,
                    context_retrieval_weight:Union[float,int]=CONTEXT_RETRIEVAL_WEIGHT,
                    reward_retrieval_weight:Union[float,int]=REWARD_RETRIEVAL_WEIGHT,
                    # retrieved_time_name:str=RETRIEVED_TIME_NAME,
                    retrieved_reward_name:str=RETRIEVED_REWARD_NAME,

                    # Output / decision processing:
                    decision_layer_name:str=DECISION_LAYER_NAME,
                    decision_size:int=DECISION_SIZE,

                    )->Composition:

    # Apportionment of contributions of state (actual or em) vs. context (em) to context_layer integration:

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


    task_input_layer = ProcessingMechanism(name=task_input_name,
                                           size=task_size)

    state_input_layer = ProcessingMechanism(name=state_input_name,
                                            size=state_size)

    time_input_layer = ProcessingMechanism(name=time_input_name,
                                           size=time_size)

    attention_layer = ProcessingMechanism(name=ATTENTION_LAYER_NAME,
                                          input_ports=[ACTUAL_STATE_INPUT, RETRIEVED_STATE_INPUT],
                                          function=LinearCombination)

    context_layer = RecurrentTransferMechanism(name=context_name,
                                               size=state_size,
                                               auto=1-context_integration_rate,
                                               hetero=0.0)

    reward_input_layer = ProcessingMechanism(name=reward_input_name,
                                              size=reward_size)

    # retrieved_time_layer = TransferMechanism(name=retrieved_time_name,
    #                                    size=time_size)

    retrieved_reward_layer = TransferMechanism(name=retrieved_reward_name,
                                         size=reward_size)

    em = EpisodicMemoryMechanism(name=em_name,
                                 input_ports=[{NAME:state_input_name, SIZE:state_size},
                                              {NAME:time_input_name, SIZE:time_size},
                                              {NAME:context_name, SIZE:state_size},
                                              {NAME:reward_input_name, SIZE:reward_size}
                                              ],
                                 function=ContentAddressableMemory(
                                     initializer=[[0] * state_size,   # state
                                                  [0] * time_size,    # time
                                                  [0] * state_size,   # context
                                                  [0] * reward_size], # reward
                                     distance_field_weights=[state_retrieval_weight,
                                                             time_retrieval_weight,
                                                             context_retrieval_weight,
                                                             reward_retrieval_weight],
                                     selection_function=SoftMax(gain=retrieval_softmax_gain)))

    decision_layer = TransferMechanism(name=decision_layer_name,
                                       size=decision_size,
                                       function=SoftMax(output=PROB))

    def encoding_control_function(variable,context):
        """Used by attention_layer to control encoding of state info in context_layer and storing in EM

        If task is:

         Task.EXPERIENCE (0):
          - stores state info in em on every trial (control_signal[0]=1)
          - always attend to actual state (control_signal[1]=1, control_signal[2]=0)

         Task.PREDICT: (1):
          - never store info in em (control_signal[0]=0)
          - attend to actual state on first trial (control_signal[1]=1, control_signal[2]=0)
          - attend to retrieved state on all subsequent trials (control_signal[1]=0, control_signal[2]=1)

        Returns:
            control_signal[0]: 1 if store, 0 otherwise
            control_signal[1]: 1 if attend to actual state, 0 otherwise
            control_signal[2]: 1 if attend to retrieved state, 0 otherwise
        """

        # Tas
        task = int(variable)

        # Trial Number:
        if context and context.composition:
            trial = int([context.composition.get_current_execution_time(context)[TimeScale.TRIAL]])
        else:
            trial = 0

        if task == Task.EXPERIENCE:
            attend_actual = 1

        elif task == Task.PREDICT:
            attend_actual = 1 if not (trial % NUM_ROLL_OUT) else 1
            attend_retrieved = 1 if (trial % NUM_ROLL_OUT) else 0
        else:
            raise ValueError(f"Unrecognized task value in encoding_control_function: {task}")

        # Store to EM to
        store = 1 if task == Task.EXPERIENCE.value else 0

        control_signals = [store, attend_actual, attend_retrieved]

        return control_signals

    # Control Mechanism
    # Uses the encoding_control_function (see above) to control:
    #   - encoding of state info in context_layer (from stimulus vs. em)
    #   - storage of info in em
    attention_layer = ControlMechanism(name=attention_layer_name,
                                                 monitor_for_control=task_input_layer,
                                                 function = encoding_control_function,
                                                 control=[(STORAGE_PROB, em),
                                                          attention_layer.input_ports[ACTUAL_STATE_INPUT],
                                                          attention_layer.input_ports[RETRIEVED_STATE_INPUT]])

    EGO_comp = Composition(name=model_name,
                           pathways=[retrieved_reward_layer, decision_layer], # Decision
                           # # Use this to terminate a Task.PREDICT trial
                           termination_processing={
                               TimeScale.TRIAL: And(WhenFinished(decision_layer),
                                                    )}
                           )

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([task_input_layer,
                        state_input_layer,
                        time_input_layer,
                        attention_layer,
                        context_layer,
                        reward_input_layer,
                        # retrieved_time_layer,
                        em])
    EGO_comp.exclude_node_roles(task_input_layer, NodeRole.OUTPUT)

    # Projections not included in (decision output) Pathway specified above
    # EM encoding
    EGO_comp.add_projection(MappingProjection(state_input_layer, em.input_ports[STATE_INPUT_LAYER_NAME]))
    EGO_comp.add_projection(MappingProjection(time_input_layer, em.input_ports[TIME_INPUT_LAYER_NAME]))
    EGO_comp.add_projection(MappingProjection(context_layer, em.input_ports[CONTEXT_LAYER_NAME]))
    EGO_comp.add_projection(MappingProjection(reward_input_layer, em.input_ports[REWARD_INPUT_LAYER_NAME]))

    # Inputs to Context
    # actual state -> attention_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer,
                                              attention_layer.input_ports[ACTUAL_STATE_INPUT]))
    # retrieved state -> attention_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{STATE_INPUT_LAYER_NAME}'],
                                              attention_layer.input_ports[RETRIEVED_STATE_INPUT]))
    # attention_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attention_layer,
                                              context_layer,
                                              state_weight))
    # retrieved context -> context_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{CONTEXT_LAYER_NAME}'],
                                              context_layer,
                                              weight=context_weight))

    # Rest of EM retrieval
    # EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{TIME_INPUT_LAYER_NAME}'],
    #                                           retrieved_time_layer)),
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{REWARD_INPUT_LAYER_NAME}'],
                                              retrieved_reward_layer))
    EGO_comp.add_node(attention_layer)

    print(f'{model_name} constructed')
    return EGO_comp

# Script execution:

model = None

if CONSTRUCT_MODEL:
    model = construct_model()

if DISPLAY_MODEL is not None:
    if model:
        model.show_graph(**DISPLAY_MODEL)
    else:
        print("Model not yet constructed")

if RUN_MODEL:
    model.run(inputs=inputs)
