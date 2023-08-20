# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

# TODO:
# - Refactor to use only Reward and not Counter (though that may not work for transition devaluation?)

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

import numpy as np
from typing import Union
from enum import IntEnum

from psyneulink import *

# Settings for running script:
CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = (                      # Only one of the following can be uncommented:
    None                             # suppress display of model
    # {}                               # show simple visual display of model
    # {'show_node_structure': True}    # show detailed view of node structures and projections
)
RUN_MODEL = True                       # True => run the model
# RUN_MODEL = False                      # False => don't run the model
ANALYZE_RESULTS = False                # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.TERSE     # Sets console output during run [alt: ReportOutput.FULL or .ON]
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
PRINT_RESULTS = False                  # print model.results after execution
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution


#region   PARAMETERS
# ======================================================================================================================
#                                                   PARAMETERS
# ======================================================================================================================

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model"
TASK_INPUT_LAYER_NAME = "TASK"
STATE_INPUT_LAYER_NAME = "STATE"
TIME_INPUT_LAYER_NAME = "TIME"
# ATTENTION_LAYER_NAME = "ENCODE\nSTATE"
ATTENTION_LAYER_NAME = "ENCODE STATE"
CONTROL_LAYER_NAME = "CONTROL"
ACTUAL_STATE_INPUT = 'ACTUAL_STATE_INPUT'
RETRIEVED_STATE_INPUT = 'RETRIEVED_STATE'
CONTEXT_LAYER_NAME = 'CONTEXT'
REWARD_INPUT_LAYER_NAME = "REWARD"
# RETRIEVED_TIME_NAME = "RETRIEVED\nTIME"
RETRIEVED_TIME_NAME = "RETRIEVED TIME"
# RETRIEVED_REWARD_NAME = "RETRIEVED\nREWARD"
RETRIEVED_REWARD_NAME = "RETRIEVED REWARD"
EM_NAME = "EM"
DECISION_LAYER_NAME = "DECISION"
RESPONSE_LAYER_NAME = "RESPONSE"


Task = IntEnum('Task', ['EXPERIENCE', 'PREDICT'],
               start=0)


StateFeatureIndex = IntEnum('StateFeatureIndex',
                            ['TASK',
                             'REWARD',
                             'SIM_STEP',
                             'SIM_TOTAL'],
                            start=0)

ControlSignalIndex = IntEnum('ControlSignalIndex',
                             ['ATTEND_ACTUAL',
                              'ATTEND_RETRIEVED',
                              'EM_FIELD_WEIGHTS',
                              'STORAGE_PROB',
                              'COUNTER_RESET',
                              'DECISION_GATE',
                              'RESPONSE_GATE'],
                             start=0)

# CONSTRUCTION PARAMETERS

# Layer sizes:
TASK_SIZE = 1       # length of task vector
STATE_SIZE = 8      # length of state vector
TIME_SIZE = 25      # length of time vector
REWARD_SIZE = 1     # length of reward vector
DECISION_SIZE = 1   # length of decision vector
RESPONSE_SIZE = 1   # length of response vector

# Context processing:
STATE_WEIGHT = .1              # rate at which actual vs. retrieved state (from EM) are integrated in context_layer
CONTEXT_INTEGRATION_RATE = .1  # rate at which retrieved context (from EM) is integrated into context_layer

# EM retrieval
STATE_RETRIEVAL_WEIGHT = 1     # weight of state field in retrieval from EM
TIME_RETRIEVAL_WEIGHT = 1      # weight of time field in retrieval from EM
CONTEXT_RETRIEVAL_WEIGHT = 1   # weight of context field in retrieval from EM
REWARD_RETRIEVAL_WEIGHT = 0    # weight of reward field in retrieval from EM
RETRIEVAL_SOFTMAX_GAIN = 10    # gain on softmax retrieval function
# RETRIEVAL_HAZARD_RATE = 0.04   # rate of re=sampling of em following non-match determination in a pass through ffn

RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

#endregion

#region ENVIRONMENT
# ======================================================================================================================
#                                                   ENVIRONMENT
# ======================================================================================================================


# Temporal context vector generation as input to time_input_layer of model
TIME_DRIFT_RATE = 0.1          # noise used by DriftOnASphereIntegrator (function of Context mech)
TIME_DRIFT_NOISE = 0.0         # noise used by DriftOnASphereIntegrator (function of Context mech)
time_fct = DriftOnASphereIntegrator(initializer=np.random.random(TIME_SIZE - 1),
                                    noise=TIME_DRIFT_NOISE,
                                    dimension=TIME_SIZE)
# Task environment:
NUM_BASELINE_SEQS = 20       # number of trials for Task.EXPERIENCE (passive encoding into EM) BEFORE revaluation
NUM_REVALUATION_SEQS = 10    # number of trials for Task.EXPERIENCE (passive encoding into EM) AFTER revaluation 
NUM_EXPERIENCE_SEQS = NUM_BASELINE_SEQS + NUM_REVALUATION_SEQS # total number of trials for Task.EXPERIENCE
NUM_PREDICT_TRIALS = 9       # number of trials Task.PREDICT (active retrieval from EM and reward prediction)
NUM_STIM_PER_SEQ = 3         # number of stimuli in a sequence
NUM_ROLL_OUTS = 3            # number of times to roll out each sequence
TOTAL_NUM_TRIALS = NUM_EXPERIENCE_SEQS * NUM_STIM_PER_SEQ + NUM_PREDICT_TRIALS # total number of trials
assert NUM_PREDICT_TRIALS % NUM_ROLL_OUTS == 0, \
    f"NUM_PREDICT_TRIALS ({NUM_PREDICT_TRIALS}) " \
    f"must be evenly divisible by NUM_ROLL_OUTS ({NUM_ROLL_OUTS})"

def build_inputs(state_size:int=STATE_SIZE,
                 time_drift_rate:float=TIME_DRIFT_RATE,
                 num_baseline_seqs:int=NUM_BASELINE_SEQS,
                 num_revaluation_seqs:int=NUM_REVALUATION_SEQS,
                 num_predict_trials:int=NUM_PREDICT_TRIALS)->tuple:
    """Build inputs for full sequence of trials (with one stim per trial) for EGO MDP model
    Return tuple in which each item is list of all trials for a layer of the model: (time, task, state, reward)
    """
    def gen_baseline_states_and_rewards_exp1(dim:int=state_size,
                                             num_seqs:int=NUM_BASELINE_SEQS)->tuple:
        """Generate states and rewards for reward revaluation phase of Experiment 1
        Return tuple with one-hot representations of (states, rewards, length of a single sequence)
        """
        # Generate one-hots
        state_reps = np.eye(dim)
        visited_states, rewards = [], []
        seq_len = 3

        for trial_idx in range(num_seqs):
            if np.random.random()<.5:
                visited_states.extend([1,3,5])
                rewards.extend([0,0,10])
            else:
                visited_states.extend([2,4,6])
                rewards.extend([0,0,1])

        # Pick one-hots corresponding to each state
        visited_states = state_reps[visited_states]
        rewards = np.array(rewards)

        return visited_states, rewards, seq_len

    def gen_reward_revaluation_states_and_reward_exp1(dim:int=STATE_SIZE,
                                                      num_seqs:int=NUM_REVALUATION_SEQS)->tuple:
        """Generate states and rewards for reward revaluation phase of Experiment 1
        Return tuple with one-hot representations of (states, rewards, length of a single sequence)
        """

        # Generate one-hots
        state_reps = np.eye(dim)
        visited_states, rewards = [], []
        seq_len = 2

        for trial_idx in range(num_seqs):
            if np.random.random()<.5:
                visited_states.extend([3,5])
                rewards.extend([0,1])
            else:
                visited_states.extend([4,6])
                rewards.extend([0,10])

        # Pick one-hots corresponding to each state
        visited_states = state_reps[visited_states]
        rewards = np.array(rewards)

        return visited_states, rewards, seq_len

    # Get sequences of states and rewards for baseline trials
    baseline_states, baseline_rewards, num_stim_per_baseline_seq = (
        gen_baseline_states_and_rewards_exp1(state_size, num_baseline_seqs))

    # Get sequences of states and rewards for reward revaluation trials
    reward_revaluation_states, reward_revaluation_rewards, num_stim_per_revaluation_seq =(
        gen_reward_revaluation_states_and_reward_exp1(state_size,num_revaluation_seqs))

    # FIX: NEED TO GENERATE CORRECT states and rewards for num_predict_trials
    experience_trials_states = np.concatenate((baseline_states, reward_revaluation_states))
    experience_trials_rewards = np.concatenate((baseline_rewards, reward_revaluation_rewards))
    states = np.concatenate((experience_trials_states, np.zeros((num_predict_trials, state_size))))
    rewards = np.concatenate((experience_trials_rewards, np.zeros(num_predict_trials)))

    # Get sequences of task and time inputs
    num_experience_trials = (num_baseline_seqs * num_stim_per_baseline_seq
                             + num_revaluation_seqs * num_stim_per_revaluation_seq)
    total_num_trials = num_experience_trials + num_predict_trials
    tasks = np.array([Task.EXPERIENCE.value] * num_experience_trials + [Task.PREDICT.value] * num_predict_trials)
    times = np.array([time_fct(time_drift_rate) for i in range(total_num_trials)])

    assert len(times) == total_num_trials
    assert len(tasks) == total_num_trials
    assert len(states) == total_num_trials
    assert len(rewards) == total_num_trials

    return times, tasks, states, rewards


inputs = {STATE_INPUT_LAYER_NAME: [[1],[2],[3]] * STATE_SIZE * TOTAL_NUM_TRIALS,
          TIME_INPUT_LAYER_NAME: np.array([time_fct(i) for i in range(TOTAL_NUM_TRIALS)]).reshape(TOTAL_NUM_TRIALS,TIME_SIZE,1),
          REWARD_INPUT_LAYER_NAME: [[0],[0],[1]] * REWARD_SIZE * TOTAL_NUM_TRIALS,
          TASK_INPUT_LAYER_NAME: [[Task.EXPERIENCE.value]] * NUM_EXPERIENCE_SEQS
                                 + [[Task.PREDICT.value]] * NUM_PREDICT_TRIALS}

#endregion

#region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

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
                    attentional_control_name=CONTROL_LAYER_NAME,
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

                    response_layer_name:str=RESPONSE_LAYER_NAME,
                    response_size:int=RESPONSE_SIZE,

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

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Mechanisms  -------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    task_input_layer = ProcessingMechanism(name=task_input_name,
                                           size=task_size)

    state_input_layer = ProcessingMechanism(name=state_input_name,
                                            size=state_size)

    time_input_layer = ProcessingMechanism(name=time_input_name,
                                           size=time_size)

    context_layer = RecurrentTransferMechanism(name=context_name,
                                               size=state_size,
                                               auto=1-context_integration_rate,
                                               hetero=0.0)

    reward_input_layer = ProcessingMechanism(name=reward_input_name,
                                              size=reward_size)

    attention_layer = ProcessingMechanism(name=ATTENTION_LAYER_NAME,
                                          size=(state_size,state_size),
                                          input_ports=[ACTUAL_STATE_INPUT, RETRIEVED_STATE_INPUT],
                                          function=LinearCombination)

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
                                                             reward_retrieval_weight]))

    counter_layer = IntegratorMechanism(function=SimpleIntegrator,
                                        default_variable=1,
                                        reset_default=1,
                                        name='COUNTER')

    decision_layer = DDM(function=DriftDiffusionIntegrator(noise=0.0, rate=1.0),
                         execute_until_finished=False,
                         name=DECISION_LAYER_NAME)

    response_layer = TransferMechanism(name=response_layer_name,
                                       size=response_size)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Control  ----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    tot = NUM_ROLL_OUTS * NUM_STIM_PER_SEQ
    #                +--------------------------------------------------------------------------------------------+
    #                |         KEYS            |                        VALUES                                    |
    #                +--------------------------------------------------------------------------------------------+
    #                 TASK  COUNT  DDM  REWARD | ACTUAL EM | TIME STATE CONTEXT REWARD | STORE | CT  | DDM | RESP
    control_policy = [[[0], [-1], [-1], [-1],     [1], [0],  [0,    0,     0,      0],    [1],   [0],  [0],  [0]],
                      [[1],  [0], [-1],  [0],     [1], [0],  [1,    1,     1,      0],    [0],   [1],  [0],  [0]],
                      [[1],  [1], [-1],  [0],     [0], [1],  [1,    0,     1,      0],    [0],   [0],  [1],  [0]],
                      [[1],  [2], [-1],  [0],     [0], [1],  [1,    0,     1,      0],    [0],   [0],  [1],  [0]],
                      [[1],  [2], [-1],  [1],     [0], [1],  [1,    0,     1,      0],    [0],   [1],  [1],  [1]],
                      [[1],  [2], [tot], [1],     [0], [1],  [1,    0,     1,      0],    [0],   [1],  [1],  [1]]]
    control_em = ContentAddressableMemory(initializer=control_policy,
                                          distance_function=Distance(metric=EUCLIDEAN),
                                          storage_prob=0.0)
    num_keys = len(StateFeatureIndex)
    num_vals = len(ControlSignalIndex)
    num_fields = num_keys + num_vals

    def control_function(variable, context=None, **kwargs):
        """Use by control_layer to govern EM storage & retrieval, access to context, and gating decision and response

        FIX:
        - Use control_layer variable ("state_features" per ocm) to get control_allocation from control_em:
            - for Task EXPERIENCE, set distance_field_weights to [1] + [0] * 9
            - for Task PREDICT, set distance_field_weights to [1] * 4 + [0] * 6
        - Use retrieved values from control_em to set control_signals for control_layer

        CONTROL PROTOCOL FOR PREDICT TRIALS:
          EM STORAGE PROB  CONTROL SIGNAL = 0
          ON COUNT 0:
            - STATE ATTENTION CONTROL SIGNAL -> STATE 1, CONTEXT 0
            - EM FIELD WEIGHTS CONTROL SIGNAL:  TIME, STATE, REWARD, CONTEXT
            - EM STORAGE PROB  CONTROL SIGNAL = 0
            - COUNTER RESET CONTROL SIGNAL = 0
            - TERMINATE CONTROL SIGNAL = 0
          ON COUNT > 0:
            - EM STORAGE PROB  CONTROL SIGNAL = 0
            - STATE ATTENTION CONTROL SIGNAL -> STATE 0, CONTEXT 1
            - EM FIELD WEIGHTS CONTROL SIGNAL:  TIME, CONTEXT
            - COUNTER RESET CONTROL SIGNAL = 0
            - TERMINATE CONTROL SIGNAL = 0
          ON RETRIEVED REWARD > 0:
            - COUNTER RESET CONTROL SIGNAL = 1
            - DECISION INPUT GATE CONTROL SIGNAL (NEEDED, IF OTHERWISE REWARDS ARE ALWAYS 0?)
            - INCREMENT DECISION COUNTER (OR JUST USE DECISION TIME IF DDM?)
          TERMINATION @  NUM_ROLL_OUTS * NUM_STIM_PER_SEQ, USING EITHER:
            - CONDITION FOR DDM DECISION TIME AS COUNTER
            - OR INPUT_GATE RESPONSE LAYER USING DDM DECISION TIME AS COUNTER,
              AND TERMINATION ON RESPONSE.VALUE > 0

        .. table::
             :align: left
             :alt: Player Piano (aka Production System)
           +-----------------------------------------------------------------------------------------------------------+
           |                                       **CONTROL POLICY**                                                  |
           +--------------------------+--------------------------------------------------------------------------------+
           | "STATE FEATURE" VECTOR   |                      CONTROL ALLOCATION VECTOR                                 |
           |  (monitor_for_control)   +-------------+---------------------------------------+-----------+--------------+
           |                          | STATE ATTN  |                  EM CONTROL           | CTR RESET |    GATES     |
           |                          |             +-------------------------------+-------+-----------+--------------+
           |                          |             |            MATCH              | STORE |           |              |
           |                          |             |        (field_weights)        |       |           |              |
           | TASK  COUNT  DDM  REWARD | ACTUAL  EM  |   TIME  STATE  CONTEXT REWARD |       |           | DDM  | RESP  |
           +--------------------------+-------------+-------------------------------+-------+-----------+------+-------+
           |  EXP   ANY  !=TOT  ANY   |   1     0   |   0      0       0       0    |   1   |     0     |   0  |   0   |
           |  PRED   0   !=TOT   0    |   1     0   |   1      1       1       0    |   0   |     1     |   0  |   0   |
           |  PRED   1   !=TOT   0    |   0     1   |   1      0       1       0    |   0   |     0     |   1  |   0   |
           |  PRED   2   !=TOT   0    |   0     1   |   1      0       1       0    |   0   |     0     |   1  |   0   |
           |  PRED   2   !=TOT   1    |   0     1   |   1      0       1       0    |   0   |     1     |   1  |   1   |
           |  PRED   2   ==TOT   1    |   0     1   |   1      0       1       0    |   0   |     1     |   1  |   1   |
           +--------------------------+-------------+-------------------------------+-------+-----------+------+-------+
           NOTES:
           - DDM is open gated on each PREDICT step because REWARD is 0 so it won't accumulate,
                     but it will increment its counter (RESPONSE TIME) that can be used to determine when to terminate
           - RO: NUM_ROLL_OUTS * NUM_STIM_PER_SEQ

        """

        task = variable[StateFeatureIndex.TASK]
        query = np.array(list(variable) + [[0],[0],[0,0,0,0],[0],[0],[0],[0]], dtype=object)
        if task == Task.EXPERIENCE:
            # Set distance_field_weights for EXPERIENCE
            # control_em.parameters.distance_field_weights.set([1] + [0] * (num_fields - 1), context)
            control_em.distance_field_weights = [1] + [0] * (num_fields - 1)
            # Get control_signals for EXPERIENCE
            control_signals = control_em(query)[num_keys:]

        elif task == Task.PREDICT:
            # Set distance_field_weights for PREDICT
            control_em.parameters.distance_field_weights.set([1] * num_keys + [0] * num_vals, context)
            # Set control_signals for PREDICT
            control_signals = control_em.execute(query)[num_keys:]

        return control_signals

    # Monitored for control
    state_features = [None] * len(StateFeatureIndex)
    state_features[StateFeatureIndex.TASK] = task_input_layer
    state_features[StateFeatureIndex.REWARD] = retrieved_reward_layer
    state_features[StateFeatureIndex.SIM_STEP] = counter_layer
    state_features[StateFeatureIndex.SIM_TOTAL] = decision_layer.output_ports[RESPONSE_TIME]

    # Control signals
    control_signals = [None] * len(ControlSignalIndex)
    control_signals[ControlSignalIndex.ATTEND_ACTUAL] = attention_layer.input_ports[ACTUAL_STATE_INPUT]
    control_signals[ControlSignalIndex.ATTEND_RETRIEVED] = attention_layer.input_ports[RETRIEVED_STATE_INPUT]
    control_signals[ControlSignalIndex.EM_FIELD_WEIGHTS] = ('distance_field_weights', em)
    control_signals[ControlSignalIndex.STORAGE_PROB] = (STORAGE_PROB, em)
    control_signals[ControlSignalIndex.COUNTER_RESET] = (RESET, counter_layer)
    control_signals[ControlSignalIndex.DECISION_GATE] = decision_layer.input_port
    control_signals[ControlSignalIndex.RESPONSE_GATE] = response_layer.input_port
    control_layer = ControlMechanism(name=attentional_control_name,
                                     monitor_for_control=state_features,
                                     function = control_function,
                                     control=control_signals)
    
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    EGO_comp = Composition(name=model_name,
                           # Decision output pathway
                           pathways=[retrieved_reward_layer, decision_layer, response_layer], # Decision
                           # # Use this to terminate a Task.PREDICT trial
                           # FIX: NEEDS TO BE PROPERLY CONFIGURED
                           # termination_processing={TimeScale.TRIAL: WhenFinished(response_layer)}
                           )

    # Nodes not included in (decision output) Pathway specified above
    EGO_comp.add_nodes([task_input_layer,
                        state_input_layer,
                        time_input_layer,
                        attention_layer,
                        context_layer,
                        counter_layer,
                        reward_input_layer,
                        em,
                        control_layer,
                        ])
    EGO_comp.exclude_node_roles(task_input_layer, NodeRole.OUTPUT)
    EGO_comp.exclude_node_roles(counter_layer, [NodeRole.INPUT, NodeRole.OUTPUT])

    # Projections not included in (decision output) Pathway specified above

    # EM encoding --------------------------------------------------------------------------------
    # state -> em
    EGO_comp.add_projection(MappingProjection(state_input_layer, em.input_ports[STATE_INPUT_LAYER_NAME]))
    # time -> em
    EGO_comp.add_projection(MappingProjection(time_input_layer, em.input_ports[TIME_INPUT_LAYER_NAME]))
    # context -> em
    EGO_comp.add_projection(MappingProjection(context_layer, em.input_ports[CONTEXT_LAYER_NAME]))
    # reward -> em
    EGO_comp.add_projection(MappingProjection(reward_input_layer, em.input_ports[REWARD_INPUT_LAYER_NAME]))

    # Inputs to Context ---------------------------------------------------------------------------
    # actual state -> attention_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer, attention_layer.input_ports[ACTUAL_STATE_INPUT]))
    # retrieved state -> attention_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{STATE_INPUT_LAYER_NAME}'],
                                              attention_layer.input_ports[RETRIEVED_STATE_INPUT]))
    # attention_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attention_layer, context_layer,
                                              matrix=np.eye(STATE_SIZE) * state_weight))
    # retrieved context -> context_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{CONTEXT_LAYER_NAME}'],
                                              context_layer,
                                              matrix=np.eye(STATE_SIZE) * context_weight))

    # Rest of EM retrieval ---------------------------------------------------------------------------
    # retreieved reward -> retrieved reward
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{REWARD_INPUT_LAYER_NAME}'],
                                              retrieved_reward_layer))

    # Validate construction
    assert context_layer.input_port.path_afferents[0].sender.owner == context_layer
    assert context_layer.input_port.path_afferents[0].parameters.matrix.get()[0][0] == 1-context_integration_rate
    assert context_layer.input_port.path_afferents[1].sender.owner == attention_layer
    assert context_layer.input_port.path_afferents[1].parameters.matrix.get()[0][0] == state_weight
    assert context_layer.input_port.path_afferents[2].sender.owner == em
    assert context_layer.input_port.path_afferents[2].parameters.matrix.get()[0][0] == context_weight

    print(f'{model_name} constructed')
    return EGO_comp
#endregion

#region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

model = None

if CONSTRUCT_MODEL:
    model = construct_model()

if DISPLAY_MODEL is not None:
    if model:
        model.show_graph(**DISPLAY_MODEL)
    else:
        print("Model not yet constructed")

if RUN_MODEL:
    inputs = build_inputs(state_size=STATE_SIZE,
                          time_drift_rate=TIME_DRIFT_RATE,
                          num_baseline_seqs=NUM_BASELINE_SEQS,
                          num_revaluation_seqs=NUM_REVALUATION_SEQS,
                          num_predict_trials=NUM_PREDICT_TRIALS)
    input_layers = [TIME_INPUT_LAYER_NAME,
                    TASK_INPUT_LAYER_NAME,
                    STATE_INPUT_LAYER_NAME,
                    REWARD_INPUT_LAYER_NAME]
    inputs_dicts = {k: v for k, v in zip(input_layers, inputs)}

    model.run(inputs=inputs_dicts,
              report_output=REPORT_OUTPUT,
              report_progress=REPORT_PROGRESS)

    if PRINT_RESULTS:
        print(model.results)
#endregion