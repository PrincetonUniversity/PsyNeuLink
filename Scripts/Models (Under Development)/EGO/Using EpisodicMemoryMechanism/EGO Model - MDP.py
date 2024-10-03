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
from enum import IntEnum

from psyneulink import *
from psyneulink._typing import Union, Literal
from psyneulink.core.scheduling.condition import Any, And, AllHaveRun, AtRunStart

# Settings for running script:

NUM_EXP_SEQS = 5               # Number of sequences to run in EXPERIENCE Phase (includes baseline + revaluation)
NUM_PRED_TRIALS = 10           # Number of trials (ROLL OUTS) to run in PREDICTION Phase

CONSTRUCT_MODEL = True                 # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = (                      # Only one of the following can be uncommented:
    # None                             # suppress display of model
    {}                               # show simple visual display of model
    # {'show_node_structure': True}    # show detailed view of node structures and projections
)
# RUN_MODEL = True                       # True => run the model
RUN_MODEL = False                      # False => don't run the model
ANALYZE_RESULTS = False                # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.FULL     # Sets console output during run [ReportOutput.ON, .TERSE OR .FULL]
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
MODEL_NAME = "EGO Model"
TASK_INPUT_LAYER_NAME = "TASK"
STATE_INPUT_LAYER_NAME = "STATE"
TIME_INPUT_LAYER_NAME = "TIME"
ATTEND_EXTERNAL_LAYER_NAME = "ATTEND\nEXTERNAL\nSTATE"
ATTEND_MEMORY_LAYER_NAME = "ATTEND\nMEMORY\nOF STATE"
CONTROL_LAYER_NAME = "CONTROL"
CONTEXT_LAYER_NAME = 'CONTEXT'
REWARD_INPUT_LAYER_NAME = "REWARD"
# RETRIEVED_TIME_NAME = "RETRIEVED\nTIME"
RETRIEVED_TIME_NAME = "RETRIEVED_TIME"  # NAME OF EM OUTPUT PORT FOR TIME
# RETRIEVED_REWARD_NAME = "RETRIEVED\nREWARD"
RETRIEVED_REWARD_NAME = "RETRIEVED REWARD"
EM_NAME = "EM"
DECISION_LAYER_NAME = "DECISION"
RESPONSE_LAYER_NAME = "RESPONSE"


Task = IntEnum('Task', ['EXPERIENCE', 'PREDICT'],
               start=0)

EMFieldsIndex = IntEnum('EMFields',
                        ['STATE',
                         'TIME',
                         'CONTEXT',
                         'REWARD'],
                        start=0)

StateFeatureIndex = IntEnum('StateFeatureIndex',
                            ['TASK',
                             'REWARD'],
                            start=0)

ControlSignalIndex = IntEnum('ControlSignalIndex',
                             ['ATTEND_EXTERNAL',
                              'ATTEND_MEMORY',
                              'EM_FIELD_WEIGHTS',
                              'STORAGE_PROB',
                              'DECISION_GATE'],
                             start=0)

# CONSTRUCTION PARAMETERS

# Layer sizes:
TASK_SIZE = 1                         # length of task vector
STATE_SIZE = model_params['state_d']  # length of state vector
TIME_SIZE = model_params['time_d']    # length of time vector
REWARD_SIZE = 1                       # length of reward vector
DECISION_SIZE = 1                     # length of decision vector

# Context processing:
STATE_WEIGHT = model_params['input_weight']  # rate at which external vs. memory state are integrated in context_layer
CONTEXT_INTEGRATION_RATE = model_params['retrieved_context_weight']  # rate at which retrieved context (from EM)
                                                                     # is integrated into context_layer
assert (model_params['retrieved_context_weight'] + STATE_WEIGHT + CONTEXT_INTEGRATION_RATE) == 1,\
    (f"Sum of STATE_WEIGHT ({STATE_WEIGHT}), CONTEXT_INTEGRATION_RATE ({CONTEXT_INTEGRATION_RATE}), "
     f"and RETRIEVED_CONTEXT_WEIGHT ({model_params['retrieved_context_weight']}) must equal 1")

# EM retrieval
STATE_RETRIEVAL_WEIGHT = model_params['state_weight']     # weight of state field in retrieval from EM
TIME_RETRIEVAL_WEIGHT = model_params['time_weight']       # weight of time field in retrieval from EM
CONTEXT_RETRIEVAL_WEIGHT = model_params['context_weight'] # weight of context field in retrieval from EM
REWARD_RETRIEVAL_WEIGHT = 0                               # weight of reward field in retrieval from EM
RETRIEVAL_SOFTMAX_GAIN = 1/model_params['temperature']    # gain on softmax retrieval function
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
NUM_STIM_PER_SEQ = model_params['n_steps'] # number of stimuli in a sequence
NUM_BASELINE_SEQS = NUM_EXP_SEQS     # num trials for Task.EXPERIENCE (passive encoding into EM) BEFORE revaluation
NUM_REVALUATION_SEQS = NUM_EXP_SEQS  # num trials for Task.EXPERIENCE (passive encoding into EM) AFTER revaluation
NUM_EXPERIENCE_SEQS = NUM_BASELINE_SEQS + NUM_REVALUATION_SEQS  # total number of trials for Task.EXPERIENCE
TOTAL_NUM_EXPERIENCE_STIMS = (NUM_BASELINE_SEQS * NUM_STIM_PER_SEQ) + (NUM_REVALUATION_SEQS * (NUM_STIM_PER_SEQ-1))
NUM_ROLL_OUTS = NUM_PRED_TRIALS    # number of times to roll out each sequence in Task.PREDICT

STIM_SEQS = [list(range(1,NUM_STIM_PER_SEQ*2,2)),
            list(range(2,NUM_STIM_PER_SEQ*2+1,2))]
REWARD_VALS = [10,1]         # reward values for seq_0 and seq_1, respectively
RATIO = 1                    # ratio of seq_0 to seq_1 in EXPERIENCE phase (ratio / 1 + ratio)
SAMPLING_TYPE = 'random'     # 'random' or 'alternating' in EXPERIENCE phase
PREDICT_SEQ_TYPE = 'blocked' # 'blocked' or 'alternating' in PREDICT phase

def get_states(state_size=STATE_SIZE):
    state_reps = np.eye(state_size)
    return state_reps

def build_experience_inputs(state_size:int=STATE_SIZE,
                            time_drift_rate:float=TIME_DRIFT_RATE,
                            num_baseline_seqs:int=NUM_BASELINE_SEQS,
                            num_revaluation_seqs:int=NUM_REVALUATION_SEQS,
                            reward_vals:list=REWARD_VALS,
                            sampling_type:Literal[Union['random','alternating']]=SAMPLING_TYPE,
                            ratio:int=RATIO,
                            stim_seqs:list=STIM_SEQS)->tuple:
    """Build inputs for full sequence of trials (with one stim per trial) for EGO MDP model
    Return tuple in which each item is list of all trials for a layer of the model: (time, task, state, reward)
    """
    assert isinstance(ratio,int),f"ratio ({ratio}) must be an integer"
    assert sampling_type in ['random','alternating'], f"Sampling type must be 'random' or 'alternating'"

    def gen_baseline_states_and_rewards(state_size:int=state_size,
                                        stim_seqs:list=stim_seqs,
                                        reward_vals:list=reward_vals,
                                        num_seqs:int=num_baseline_seqs,
                                        sampling_type:Literal[Union['random','alternating']]=sampling_type,
                                        ratio:int=ratio,
                                        )->tuple:
        """Generate states and rewards for reward revaluation phase of Experiment 1
        Return tuple with one-hot representations of (states, rewards, length of a single sequence)
        """
        # Generate one-hots
        state_reps = get_states(state_size)

        # Generate sequence of states
        visited_states, rewards = [], []
        seq_len = len(stim_seqs[0])
        for i in range(num_seqs):
            seq_0 = np.random.random() < (ratio / (ratio + 1)) if sampling_type == 'random' else i % (ratio + 1)
            if seq_0:
                visited_states.extend(stim_seqs[0])
                rewards.extend([0] * (seq_len - 1) + [reward_vals[0]])
            else:
                visited_states.extend(stim_seqs[1])
                rewards.extend([0] * (seq_len - 1) + [reward_vals[1]])

        # Pick one-hots corresponding to each state
        visited_states = state_reps[visited_states]
        rewards = np.array(rewards)

        return visited_states, rewards, seq_len

    def gen_reward_revaluation_states_and_reward(state_size:int=STATE_SIZE,
                                                 stim_seqs:list=stim_seqs,
                                                 reward_vals:list=reward_vals,
                                                 num_seqs:int=num_revaluation_seqs,
                                                 sampling_type:Literal[Union['random','alternating']]=sampling_type,
                                                 ratio:int=ratio,
                                                 )->tuple:
        """Generate states and rewards for reward revaluation phase of Experiment 1
        Return tuple with one-hot representations of (states, rewards, length of a single sequence)
        """

        # Generate one-hots
        state_reps = get_states(state_size)

        # Generate sequence of states
        visited_states, rewards = [], []
        seq_len = len(stim_seqs[0][1:])
        for trial_idx in range(num_seqs):
            seq_0 = np.random.random() < (ratio / (ratio + 1)) if sampling_type == 'random' else trial_idx % (ratio + 1)
            if seq_0:
                visited_states.extend(stim_seqs[0][1:])
                rewards.extend([0] * (seq_len - 1) + [reward_vals[0]])
            else:
                visited_states.extend(stim_seqs[1][1:])
                rewards.extend([0] * (seq_len - 1) + [reward_vals[1]])

        # Pick one-hots corresponding to each state
        visited_states = state_reps[visited_states]
        rewards = np.array(rewards)

        return visited_states, rewards, seq_len

    # Get sequences of states and rewards for baseline trials
    baseline_states, baseline_rewards, num_stim_per_baseline_seq = \
        (gen_baseline_states_and_rewards(state_size=state_size,
                                         stim_seqs=stim_seqs,
                                         reward_vals=reward_vals,
                                         num_seqs=num_baseline_seqs,
                                         sampling_type=sampling_type,
                                         ratio=ratio))

    # Get sequences of states and rewards for reward revaluation trials
    reward_revaluation_states, reward_revaluation_rewards, num_stim_per_revaluation_seq = \
        (gen_reward_revaluation_states_and_reward(state_size=state_size,
                                                  stim_seqs=stim_seqs,
                                                  reward_vals=reward_vals,
                                                  num_seqs=num_revaluation_seqs,
                                                  sampling_type=sampling_type,
                                                  ratio=ratio))

    states = np.concatenate((baseline_states, reward_revaluation_states))
    rewards = np.concatenate((baseline_rewards, reward_revaluation_rewards))

    # Get sequences of task and time inputs
    num_trials = (num_baseline_seqs * num_stim_per_baseline_seq
                             + num_revaluation_seqs * num_stim_per_revaluation_seq)
    tasks = np.full(num_trials, Task.EXPERIENCE.value)
    times = np.array([time_fct(time_drift_rate) for i in range(num_trials)])

    assert len(times) == num_trials
    assert len(tasks) == num_trials
    assert len(states) == num_trials
    assert len(rewards) == num_trials

    return times, tasks, states, rewards

def build_prediction_inputs(state_size:int=STATE_SIZE,
                            time_drift_rate:float=TIME_DRIFT_RATE,
                            num_roll_outs_per_stim:int=int(NUM_ROLL_OUTS / 2),
                            stim_seqs:list=STIM_SEQS,
                            reward_vals:list=REWARD_VALS,
                            seq_type:Literal[Union['blocked','alternate']]=PREDICT_SEQ_TYPE)->tuple:

    # Get stimulus sequences
    num_trials = int(num_roll_outs_per_stim * 2)
    state_reps = get_states(state_size)
    test_stims = [stim_seqs[0][0],stim_seqs[1][0]]
    stim_seq = []
    rewards = []
    for i in range(num_trials):
        seq = int(i * 2 / num_trials) if seq_type == 'blocked' else i % 2
        stim_seq.extend([test_stims[seq]])
        rewards.extend([reward_vals[seq]])

    # Generate time, task and state inputs
    times = np.array([time_fct(time_drift_rate) for i in range(num_trials)])
    tasks = np.full(num_trials, Task.PREDICT.value)
    states =  state_reps[stim_seq]

    return times, tasks, states, np.array(rewards)


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
                    # attention_layer_name=ATTENTION_LAYER_NAME,
                    attend_external_layer_name=ATTEND_EXTERNAL_LAYER_NAME,
                    attend_memory_layer_name=ATTEND_MEMORY_LAYER_NAME,
                    attentional_control_name=CONTROL_LAYER_NAME,
                    context_name:str=CONTEXT_LAYER_NAME,
                    state_weight:Union[float,int]=STATE_WEIGHT,
                    context_integration_rate:Union[float,int]=CONTEXT_INTEGRATION_RATE,

                    # EM:
                    em_name:str=EM_NAME,
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    time_retrieval_weight:Union[float,int]=TIME_RETRIEVAL_WEIGHT,
                    context_retrieval_weight:Union[float,int]=CONTEXT_RETRIEVAL_WEIGHT,
                    reward_retrieval_weight:Union[float,int]=REWARD_RETRIEVAL_WEIGHT,
                    retrieved_reward_name:str=RETRIEVED_REWARD_NAME,

                    # Output / decision processing:
                    decision_layer_name:str=DECISION_LAYER_NAME,

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

    task_input_layer = ProcessingMechanism(name=task_input_name, input_shapes=task_size)
    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    time_input_layer = ProcessingMechanism(name=time_input_name, input_shapes=time_size)
    reward_input_layer = ProcessingMechanism(name=reward_input_name, input_shapes=reward_size)
    attend_external_layer = ProcessingMechanism(name=attend_external_layer_name, input_shapes=state_size)
    attend_memory_layer = ProcessingMechanism(name=attend_memory_layer_name, input_shapes=state_size)
    retrieved_reward_layer = TransferMechanism(name=retrieved_reward_name, input_shapes=reward_size)
    context_layer = RecurrentTransferMechanism(name=context_name,
                                               input_shapes=state_size,
                                               auto=1-context_integration_rate,
                                               hetero=0.0)
    em = EpisodicMemoryMechanism(name=em_name,
                                 default_variable=[[0] * state_size,   # state
                                                   [0] * time_size,    # time
                                                   [0] * state_size,   # context
                                                   [0] * reward_size], # reward
                                 input_ports=[{NAME:state_input_name, INPUT_SHAPES:state_size},
                                              {NAME:time_input_name, INPUT_SHAPES:time_size},
                                              {NAME:context_name, INPUT_SHAPES:state_size},
                                              {NAME:reward_input_name, INPUT_SHAPES:reward_size}],
                                 function=ContentAddressableMemory(
                                     # selection_function=SoftMax(gain=retrieval_softmax_gain),
                                     distance_field_weights=[state_retrieval_weight,
                                                             time_retrieval_weight,
                                                             context_retrieval_weight,
                                                             reward_retrieval_weight]))
    # em.output_ports[RETRIEVED_TIME_NAME].parameters.require_projection_in_composition.set(False, override=True)

    decision_layer = DDM(function=DriftDiffusionIntegrator(noise=0.0, rate=1.0),
                         execute_until_finished=False,
                         reset_stateful_function_when=AtRunStart(),
                         name=decision_layer_name)


    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Control  ----------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    tot = NUM_ROLL_OUTS * NUM_STIM_PER_SEQ
    #                +-------------------------------------------------------------------+
    #                |    KEYS    |                       VALUES                         |
    #                |   state    |                 control_allocation                   |
    #                |  features  |                                                      |
    #                +-------------------------------------------------------------------+
    #                 TASK REWARD | ACTUAL EM | TIME STATE CONTEXT REWARD | STORE | DDM
    control_policy = [[[0], [-1],    [1], [0],  [1,    1,     1,      0],    [1],   [0]], # EXPERIENCE
                      [[1],  [0],    [0], [0],  [1,    0,     1,      0],    [0],   [1]], # PREDICT - ROLLOUT
                      [[1],  [1],    [1], [0],  [1,    1,     1,      0],    [0],   [1]]] # PREDICT - REWARD/ENCODE NEXT
    control_em = ContentAddressableMemory(initializer=control_policy,
                                          distance_function=Distance(metric=EUCLIDEAN),
                                          distance_field_weights=[1, 1, 0, 0, 0, 0, 0],
                                          storage_prob=0.0)
    control_em.name = 'CONTROL EM'
    num_keys = len(StateFeatureIndex)
    num_vals = len(ControlSignalIndex)
    num_fields = num_keys + num_vals

    def control_function(variable, context=None, **kwargs):
        """Use by control_layer to govern EM storage & retrieval, access to context, and gating decision

        - Use control_layer variable ("state_features" per ocm) to get control_allocation from control_em:
            - for Task EXPERIENCE, set distance_field_weights to [1] + [0] * 9
            - for Task PREDICT, set distance_field_weights to [1] * 4 + [0] * 6
        - Use retrieved values from control_em to set control_signals for control_layer

        .. table::
           :align: left
           :alt: Player Piano (aka Production System)
           +--------+------------------------------------------------------------------------------------+-----------+
           |        |                                    **CONTROL POLICY**                              |           |
           |        +-----------------------+------------------------------------------------------------+           |
           |        |   "STATE FEATURES"    |                   CONTROL ALLOCATION                       |           |
           |        | (monitor_for_control) +--------------+---------------------------------------------+           |
           |        |                       |  STATE ATTN  |                  EM CONTROL                 |           |
           |        |                       |              +-------------------------------+-------+-----+           |
           |        |                       | ON CURR PASS |      MATCH ON CURR PASS       | STORE | DDM |           |
           |  NUM   |                       | (W/IN TRIAL) |       (field_weights)         |       |     |           |
           | TRIALS |     TASK     REWARD   |  ACTUAL  EM  |  TIME  STATE  CONTEXT REWARD  |       |     |           |
           +--------+-----------------------+--------------+-------------------------------+-------+-----+-----------+
           |   80   |      EXP      ANY     |    1     0   |   1      1       1       0    |   1   |  0  | TRIAL END |
           +--------+-----------------------+--------------+-------------------------------+-------+-----+-----------+
           |        |      PRED      0      |    0     0   |   1      0       1       0    |   0   |  1  |           |
           | #R/O's |      PRED      1      |    1     0   |   1      1       1       0    |   0   |  1  | TRIAL END |
           +--------+-----------------------+--------------+-------------------------------+-------+-----+-----------+
           NOTES:
           - Requires that EXPERIENCE TRIALS be run first, so that
           - STATE ATTN on first PREDICT trial is [1 0] and RETRIEVED_REWARD is [1]
             (since those are carried over from last trial of EXPERIENCE phase)
           - Control signal in response to retrieved_reward has its effect on next trial
                 (since Projection from retrieved_reward -> control is specified as feedback,
                  so that control will execute immediately after input and before EM (in order to control it),
                  but therefore also before retrieve_reward is updated on the current trial)
           - DDM integrates continuously;  could end run on DDM.when_finished instead of fixed number of trials
           - #R/O's = NUM_ROLL_OUTS
        """

        task = variable[StateFeatureIndex.TASK]
        reward = [1] if variable[StateFeatureIndex.REWARD] else [0]
        keys = [task, reward]
        # set values to 0 since they don't matter (should only be retrieving based on keys)
        query = np.array(keys + [[0],[0],[0,0,0,0],[0],[0]], dtype=object)

        # FIX: IF PARAMETERS ARE USED AND CONTEXT IS ADDED TO CALLS BELOW, FAILS COMPLAINING ABOUT NEED FOR SEED
        if task == Task.EXPERIENCE:
            # Set distance_field_weights for EXPERIENCE
            # control_em.parameters.distance_field_weights.set([1] + [0] * (num_fields - 1))
            control_em.distance_field_weights = [1] + [0] * (num_fields - 1)
            # Get control_signals for EXPERIENCE
            control_signals = control_em.execute(query)[num_keys:]

        elif task == Task.PREDICT:
            # Set distance_field_weights for PREDICT
            # control_em.parameters.distance_field_weights.set([1] * num_keys + [0] * num_vals)
            control_em.distance_field_weights = [1] * num_keys + [0] * num_vals
            # Set control_signals for PREDICT
            control_signals = control_em.execute(query)[num_keys:]

        return control_signals

    # Monitored for control
    state_features = [None] * len(StateFeatureIndex)
    state_features[StateFeatureIndex.TASK] = task_input_layer
    state_features[StateFeatureIndex.REWARD] = (retrieved_reward_layer, FEEDBACK)

    # Control signals
    control_signals = [None] * len(ControlSignalIndex)
    control_signals[ControlSignalIndex.ATTEND_EXTERNAL] = (SLOPE, attend_external_layer)
    control_signals[ControlSignalIndex.ATTEND_MEMORY] = (SLOPE, attend_memory_layer)
    control_signals[ControlSignalIndex.EM_FIELD_WEIGHTS] = ('distance_field_weights', em)
    control_signals[ControlSignalIndex.STORAGE_PROB] = (STORAGE_PROB, em)
    control_signals[ControlSignalIndex.DECISION_GATE] = decision_layer.input_port

    control_layer = ControlMechanism(name=attentional_control_name,
                                     monitor_for_control=state_features,
                                     function = control_function,
                                     control=control_signals)
    
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------
    
    EGO_comp = Composition(name=model_name,
                           # # Terminate a Task.PREDICT trial after decision_layer executes if a reward is retrieved
                           # termination_processing={
                           #     # TimeScale.TRIAL: And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                      Condition(lambda: retrieved_reward_layer.value),
                           #     #                      JustRan(decision_layer))}
                           #     # CRASHES:
                           #     # TimeScale.TRIAL: Any(And(Condition(lambda: task_input_layer.value == Task.EXPERIENCE),
                           #     #                          JustRan(em)),
                           #     #                      And(Condition(lambda: task_input_layer.value == Task.PREDICT),
                           #     #                          Condition(lambda: retrieved_reward_layer.value),
                           #     #                          JustRan(decision_layer)))}
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
                        decision_layer
                        ])
    EGO_comp.exclude_node_roles(task_input_layer, NodeRole.OUTPUT)

    # Projections:

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
    # actual state -> attend_external_layer
    EGO_comp.add_projection(MappingProjection(state_input_layer, attend_external_layer))
    # retrieved state -> attend_memory_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{STATE_INPUT_LAYER_NAME}'],
                                              attend_memory_layer))
    # attend_external_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attend_external_layer, context_layer,
                                              matrix=np.eye(STATE_SIZE) * state_weight))
    # attend_memory_layer -> context_layer
    EGO_comp.add_projection(MappingProjection(attend_memory_layer, context_layer,
                                              matrix=np.eye(STATE_SIZE) * state_weight))
    # retrieved context -> context_layer
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{CONTEXT_LAYER_NAME}'], context_layer,
                                              matrix=np.eye(STATE_SIZE) * context_weight))

    # Decision pathway ---------------------------------------------------------------------------
    # retrieved reward -> retrieved reward
    EGO_comp.add_projection(MappingProjection(em.output_ports[f'RETRIEVED_{REWARD_INPUT_LAYER_NAME}'],
                                              retrieved_reward_layer))
    # retrieved reward -> decision layer
    EGO_comp.add_projection(MappingProjection(retrieved_reward_layer, decision_layer))

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
    assert context_layer.input_port.path_afferents[3].sender.owner == em  # memory of context
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
                                                    sampling_type=SAMPLING_TYPE,
                                                    ratio=RATIO,
                                                    stim_seqs=STIM_SEQS)
        input_layers = [TIME_INPUT_LAYER_NAME,
                        TASK_INPUT_LAYER_NAME,
                        STATE_INPUT_LAYER_NAME,
                        REWARD_INPUT_LAYER_NAME]

        # Experience Phase
        print(f"Presenting {model.name} with {TOTAL_NUM_EXPERIENCE_STIMS} EXPERIENCE stimuli")
        model.run(inputs={k: v for k, v in zip(input_layers, experience_inputs)},
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
                                 # JustRan(model.nodes[DECISION_LAYER_NAME])
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