
"""

**Overview**
------------

This implements a model of the `nback task <https://en.wikipedia.org/wiki/N-back#Neurobiology_of_n-back_task>`_
described in `Beukers et al. (2022) <https://psyarxiv.com/jtw5p>`_.  The model uses a simple implementation of episodic
memory (EM, as a form of content-retrieval memory) to store previous stimuli along with the temporal context in which
they occurred, and a feedforward neural network (FFN) to evaluate whether the current stimulus is a match to the n'th
preceding stimulus (n-back level) retrieved from EM.

The model is an example of proposed interactions between working memory (subserved by neocortical structures) and
episodic memory (subserved by hippocampus, and possibly cerebellum) in the performance of tasks demanding of sequential
processing and control, along the lines of models emerging from machine learning that augment the use of recurrent
neural networks (e.g., long short-term memory mechanisms; LSTMs) for active memory and control, with an external memory
capable of rapid storage and content-based retrieval, such as the
Neural Turing Machine (NTN; `Graves et al., 2016 <https://arxiv.org/abs/1410.5401>`_),
Episodic Planning Networks (EPN; `Ritter et al., 2020 <https://arxiv.org/abs/2006.03662>`_), and
Emergent Symbols through Binding Networks (ESBN; `Webb et al., 2021 <https://arxiv.org/abs/2012.14601>`_).

The script contains methods to construct, train, and run the model, and analyze the results of its execution:

* `construct_model <nback.construct_model>`:
  takes as arguments parameters used to construct the model;  for convenience, defaults are defined below,
  (under "Construction parameters")

* `train_network <nback.train_network>`:
  takes as arguments the feedforward neural network Composition (FFN_COMPOSITION) and number of epochs to train.
  Note: learning_rate is set at construction (can specify using LEARNING_RATE under "Training parameters" below).

* `run_model <nback.run_model>`:
  takes as arguments the drift rate in the temporal context vector to be applied on each trial,
  and the number of trials to execute, as well as reporting and animation specifications
  (see "Execution parameters").

* `analyze_results <nback.analyze_results>`:
  takes as arguments the results of executing the model, and optionally a number of trials and nback_level to analyze;
  returns d-prime statistics and plots results for different conditions at each nback_level executed.


**The Model**
-------------

The model is comprised of two `Compositions <Composition>`: an outer one that contains the full model (`nback_model
<nback_model_composition>`), and an `AutodiffComposition`, nested within nback_model, that implements the feedforward
neural network (`ffn <nback_ffn_composition>`) (see red box in the figure below).  Both of these are constructed in
the `construct_model <nback.construct_model>` function (see `below <nback_methods_reference>`).

.. _nback_Fig:

.. figure:: _static/N-Back_Model_movie.gif
   :align: left
   :alt: N-Back Model Animation

.. _nback_model_composition:

*nback_model Composition*
~~~~~~~~~~~~~~~~~~~~~~~~~

This is comprised of three input Mechanisms, and the nested `ffn <nback_ffn_composition>` `Composition`.

.. _nback_ffn_composition:

*FFN Composition*
~~~~~~~~~~~~~~~~~

The temporal context is provided by a randomly drifting high dimensional vector that maintains a constant norm (i.e.,
drifts on a sphere).  The FFN is trained, given an n-back level of *n*, to identify when the current stimulus matches
one stored in EM with a temporal context vector that differs by an amount corresponding to *n* time steps of drift.
During n-back performance, the model encodes the current stimulus and temporal context, retrieves an item from EM
that matches the current stimulus, weighted by the similarity of its temporal context vector (i.e., most recent), and
then uses the FFN to evaluate whether it is an n-back match.  The model responds "match" if the FFN detects a match;
otherwise, it either responds "non-match" or, with a fixed probability (hazard rate), it uses the current stimulus
and temporal context to retrieve another sample from EM and repeat the evaluation.

The ffn Composition is trained using the train_network() method


**Construction and Execution**
------------------------------

.. _nback_settings:

*Settings*
~~~~~~~~~~

The default parameters are ones that have been fit to empirical data concerning human performance
(taken from `Kane et al., 2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_).

See "Settings for running the script" to specify whether the model is trained and/or executed when the script is run,
and whether a graphic display of the network is generated when it is constructed.

.. _nback_stimuli:

*Stimuli*
~~~~~~~~~

Sequences of stimuli are constructed either using `SweetPea <https://sites.google.com/view/sweetpea-ai?pli=1>`_
(using the script in stim/SweetPea) or replicate those used in the study by `Kane et al.,
2007 <https://psycnet.apa.org/record/2007-06096-010?doi=1>`_ (from stimulus files in stim/Kane_et_al).

    .. note::
       Use of SweetPea for stimulus generation requires it be installed::
       >> pip install sweetpea

.. _nback_training:

*Training*
~~~~~~~~~~

MORE HERE

.. _nback_execution:

*Execution*
~~~~~~~~~~~

MORE HERE

.. _nback_methods_reference:

**Methods Reference**
---------------------

COMMENT:
TODO:
    - from Andre
             - network architecture;  in particular, size of hidden layer and projection patterns to and from it
                - the stim+context input vector (length 90) projects to a hidden layer (length 80);
                - the task input vector (length 2) projects to a different hidden layer (length 80);
                - those two hidden layers project (over fixed, nonlearnable, one-one-projections?) to a third hidden layer (length 80) that simply sums them;
                - the third hidden layer projects to the length 2 output layer;
                - a softmax is taken over the output layer to determine the response.
                - fix: were biases trained?
          - training:
              - learning rate: 0.001; epoch: 1 trial per epoch of training
              - fix: state_dict with weights (still needed)
          - get empirical stimulus sequences (still needed)
          - put nback script (with pointer to latest version on PNL) in nback-paper repo
    - train_network() and run_model(): refactor to take inputs and trial_types, and training_set, respectively
    - fix: get rid of objective_mechanism (see "VERSION *WITHOUT* ObjectiveMechanism" under control(...)
    - fix: warnings on run
    - fix: remove num_nback_levels from contruct_model
    - complete documentation in BeukersNbackModel.rst
    - validate against nback-paper results
    - after validation:
        - try with STIM_SIZE = NUM_STIMS rather than 20 (as in nback-paper)
        - refactor generate_stim_sequence() to use actual empirical stimulus sequences
        - replace get_input_sequence and _get_training_inputs with generators passed to nback_model.run() and ffn.learn
        - build version that *can* maintain in WM, and uses EVC to decide which would be easier:
           maintenance in WM vs. storage/retrieval from EM (and the fit to Jarrod's data)
COMMENT

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
CONSTRUCT_MODEL = True # THIS MUST BE SET TO True to run the script
DISPLAY_MODEL = False # True = show visual graphic of model
TRAIN_FFN = True  # True => train the FFN (WM)
TEST_FFN = True  # True => test the FFN on training stimuli (WM)
RUN_MODEL = True  # True => test the model on sample stimulus sequences
ANALYZE_RESULTS = True # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.OFF       # Sets console output during run
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

#region ========================================= PARAMETERS ===========================================================

# Fixed parameters:
MAX_NBACK_LEVELS = 3
NUM_STIM = 8 # number of different stimuli in stimulus set -  QUESTION: WHY ISN"T THIS EQUAL TO STIM_SIZE OR VICE VERSA?
FFN_TRANSFER_FUNCTION = ReLU

# Constructor parameters:  (values are from nback-paper)


STIM_SIZE = 20 # length of stimulus vector
CONTEXT_SIZE = 25 # length of temporal context vector
TASK_SIZE = 10 # length of task specification vector
FFN_INPUT_SIZE = (STIM_SIZE + CONTEXT_SIZE) * 2 # length of full input vector
H1_SIZE = FFN_INPUT_SIZE # dimension of stimulus hidden units in ff
H2_SIZE = 80
NBACK_LEVELS = [2,3] # Currently restricted to these
NUM_NBACK_LEVELS = len(NBACK_LEVELS)
CONTEXT_DRIFT_NOISE = 0.0  # noise used by DriftOnASphereIntegrator (function of Context mech)
RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections
DROPOUT_PROB = 0.05
RETRIEVAL_SOFTMAX_TEMP = 1 / 8 # express as gain # precision of retrieval process
RETRIEVAL_HAZARD_RATE = 0.04 # rate of re=sampling of em following non-match determination in a pass through ffn
RETRIEVAL_STIM_WEIGHT = .05 # weighting of stimulus field in retrieval from em
RETRIEVAL_CONTEXT_WEIGHT = 1 - RETRIEVAL_STIM_WEIGHT # weighting of context field in retrieval from em
# DECISION_SOFTMAX_TEMP=1

# Training parameters:
NUM_TRAINING_SETS_PER_EPOCH = 1
MINIBATCH_SIZE=None
NUM_EPOCHS = 6250 # 12500 # 20000  # nback-paper: 400,000 @ one trial per epoch = 6,250 @ 64 trials per epoch
FOILS_ALLOWED_BEFORE = False
LEARNING_RATE=0.001  # nback-paper: .001

# Execution parameters:
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
NUM_TRIALS = 48 # number of stimuli presented in a trial sequence

# Names of Compositions and Mechanisms:
NBACK_MODEL = "nback Model"
FFN_COMPOSITION = "WORKING MEMORY (fnn)"
FFN_INPUT = "CURRENT INPUT LAYER"
FFN_TASK = "CURRENT TASK LAYER"
FFN_TASK_EMBED = "TASK EMBEDDING LAYER"
FFN_H1 = "H1 LAYER"
FFN_ADD_LAYER = "ADD LAYER"
FFN_DROPOUT = "DROPOUT LAYER"
FFN_H2 = "H2 LAYER"
FFN_OUTPUT = "OUTPUT LAYER"
MODEL_STIMULUS_INPUT ='STIM'
MODEL_CONTEXT_INPUT = 'CONTEXT'
MODEL_TASK_INPUT = "TASK"
CONCATENATE_FFN_INPUT = "CONCATENATE INPUT"
EM = "EPISODIC MEMORY (dict)"
DECISION = "DECISION"
CONTROLLER = "READ/WRITE CONTROLLER"


class TrialTypes(Enum):
    """Trial types explicitly assigned and counter-balanced in _get_run_inputs()
    In notation below, "A" is always current stimulus.
    Foils are only explicitly assigned to items immediately following nback item,
        or before if **foils_allowed_before** is specified in _get_training_inputs()
    Subseq designated below as "not explicitly assigned" may still appear in the overall stimulus seq,
        either within the subseq through random assignment,
        and/or through cross-subseq relationships that are not controlled in this design.
    """
    MATCH_NO_FOIL = 'match'             # ABA (2-back) or ABCA (3-back); not explicitly assigned: ABBA
    MATCH_WITH_FOIL = 'stim_lure'       # AAA (2-back) or AABA (3-back); not explicitly assigned: ABAA or AAAA
    NO_MATCH_NO_FOIL = 'non_lure'       # BBA (2-back) or BCDA (3-back); not explicitly assigned: BBCA, BCCA or BBBA
    NO_MATCH_WITH_FOIL = 'context_lure' # BAA (2-back) or BACA (3-back); not explicitly assigned: BCAA or BAAA


num_trial_types = len(TrialTypes)


class Stimuli(Enum):
    SWEETPEA = 'sweetpea'
    KANE_STIMULI = 'kane'

#endregion

#region ===================================== MODEL CONSTRUCTION =======================================================

def construct_model(stim_size:int = STIM_SIZE,
                    context_size:int = CONTEXT_SIZE,
                    ffn_input_size:int = FFN_INPUT_SIZE,
                    task_size = TASK_SIZE,
                    h1_size:int = H1_SIZE,
                    h2_size:int = H2_SIZE,
                    num_nback_levels:int = NUM_NBACK_LEVELS,
                    context_drift_noise:float = CONTEXT_DRIFT_NOISE,
                    retrievel_softmax_temp:float = RETRIEVAL_SOFTMAX_TEMP,
                    retrieval_hazard_rate:float = RETRIEVAL_HAZARD_RATE,
                    retrieval_stimulus_weight:float = RETRIEVAL_STIM_WEIGHT,
                    retrieval_context_weight:float = RETRIEVAL_CONTEXT_WEIGHT,
                    )->Composition:
    """**Construct nback_model**

    Arguments
    ---------
    stim_size: int
      dimensionality of stimulus vector
    context_size: int
      dimensionality of context vector
    ffn_input_size: int
      dimensionality of input to ffn (current stimulus and context + retrieved stimulus and context)
    task_size: int
      dimensionality of task embedding layer
    h1_size: int
      dimensionality of  input embedding layer
    h2_size: int
      dimensionality of hidden layer
    num_nback_levels: int
      number of nback_levels to implement
    context_drift_noise: float
      rate of temporal context drift
    retrievel_softmax_temp: float
      temperature of softmax on retrieval from episodic memory
    retrieval_hazard_rate: float
      rate at which episodic memory is sampled if a match is not found
    retrieval_stimulus_weight: float
      weighting on stimulus component for retrieval of vectors stored in `episodic memory <EpisodicMemoryMechanism>`
    retrieval_context_weight: float
      weighting on context component for retrieval of vectors stored in `episodic memory <EpisodicMemoryMechanism>`

    Returns
    -------
    Composition implementing nback model
    """

    print(f"constructing '{FFN_COMPOSITION}'...")

    # FEED FORWARD NETWORK -----------------------------------------
    #     inputs: encoding of current stimulus and context, retrieved stimulus and retrieved context,
    #     task input:  one hot
    #     output: match [1,0] or non-match [0,1]
    # Must be trained to detect match for specified task (1-back, 2-back, etc.)
    stim_context_input = TransferMechanism(name=FFN_INPUT,
                                           size=ffn_input_size)
    task_input = ProcessingMechanism(name=FFN_TASK,
                                     size=task_size)
    task_embedding = ProcessingMechanism(name=FFN_TASK,
                                         size=h1_size)
    h1 = ProcessingMechanism(name=FFN_H1,
                             size=h1_size,
                             function=FFN_TRANSFER_FUNCTION)
    add_layer = ProcessingMechanism(name=FFN_ADD_LAYER,
                                    size=h1_size)
    dropout = ProcessingMechanism(name=FFN_DROPOUT,
                                  size=h1_size,
                                  function=Dropout(p=DROPOUT_PROB))
    h2 = ProcessingMechanism(name=FFN_H2,
                             size=h2_size,
                             function=FFN_TRANSFER_FUNCTION)
    output = ProcessingMechanism(name=FFN_OUTPUT,
                                 size=2,
                                 function = Linear
                                 # function=ReLU
                                 )
    PASS_THROUGH = MappingProjection(matrix = IDENTITY_MATRIX, exclude_in_autodiff=True),
    input_pway = Pathway([stim_context_input, RANDOM_WEIGHTS_INITIALIZATION, h1, IDENTITY_MATRIX, add_layer])
    task_pway = Pathway([task_input, RANDOM_WEIGHTS_INITIALIZATION, task_embedding, IDENTITY_MATRIX, add_layer])
    output_pway = Pathway([add_layer, IDENTITY_MATRIX, dropout,
                           RANDOM_WEIGHTS_INITIALIZATION, h2, RANDOM_WEIGHTS_INITIALIZATION, output])

    ffn = AutodiffComposition(pathways = [input_pway, task_pway, output_pway],
                              name=FFN_COMPOSITION,
                              learning_rate=LEARNING_RATE,
                              optimizer_type='adam',
                              # optimizer_type='sgd',
                              loss_spec=Loss.CROSS_ENTROPY
                              # loss_spec=Loss.MSE
                              )

    # FULL MODEL (Outer Composition, including input, EM and control Mechanisms) ------------------------

    print(f"constructing '{NBACK_MODEL}'...")

    # Stimulus Encoding: takes stim_size vector as input
    stim = TransferMechanism(name=MODEL_STIMULUS_INPUT, size=stim_size)

    # Context Encoding: takes scalar as drift step for current trial
    context = ProcessingMechanism(name=MODEL_CONTEXT_INPUT,
                                  function=DriftOnASphereIntegrator(
                                      initializer=np.random.random(context_size - 1),
                                      noise=context_drift_noise,
                                      dimension=context_size))

    # Task: task one-hot indicating n-back (1, 2, 3 etc.) - must correspond to what ffn has been trained to do
    task = ProcessingMechanism(name=MODEL_TASK_INPUT,
                               size=task_size)

    # Episodic Memory:
    #    - entries: stimulus (field[0]) and context (field[1]); randomly initialized
    #    - uses Softmax to retrieve best matching input, subject to weighting of stimulus and context by STIM_WEIGHT
    em = EpisodicMemoryMechanism(name=EM,
                                 input_ports=[{NAME:"STIMULUS_FIELD",
                                               SIZE:stim_size},
                                              {NAME:"CONTEXT_FIELD",
                                               SIZE:context_size}],
                                 function=ContentAddressableMemory(
                                     initializer=[[[0] * stim_size, [0] * context_size]],
                                     distance_field_weights=[retrieval_stimulus_weight,
                                                             retrieval_context_weight],
                                     # equidistant_entries_select=NEWEST,
                                     selection_function=SoftMax(output=MAX_INDICATOR,
                                                                gain=retrievel_softmax_temp)))

    # Input to FFN
    concat_input = ProcessingMechanism(name=CONCATENATE_FFN_INPUT,
                                       input_ports=[stim, context,
                                                    em.output_ports["RETRIEVED_STIMULUS_FIELD"],
                                                    em.output_ports["RETRIEVED_CONTEXT_FIELD"]],
                                       function=Concatenate)

    decision = TransferMechanism(name=DECISION,
                                 size=2,
                                 function=SoftMax(output=MAX_INDICATOR))

    # Control Mechanism
    #  Ensures current stimulus and context are only encoded in EM once (at beginning of trial)
    #    by controlling the storage_prob parameter of em:
    #      - if outcome of decision signifies a match or hazard rate is realized:
    #        - set  EM[store_prob]=1 (as prep encoding stimulus in EM on next trial)
    #        - this also serves to terminate trial (see nback_model.termination_processing condition)
    #      - if outcome of decision signifies a non-match and hazard rate is not realized:
    #        - set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
    #        - continue trial
    control = ControlMechanism(name=CONTROLLER,
                               default_variable=[[1]],  # Ensure EM[store_prob]=1 at beginning of first trial
                               # ---------
                               # VERSION *WITH* ObjectiveMechanism:
                               objective_mechanism=ObjectiveMechanism(name="OBJECTIVE MECHANISM",
                                                                      monitor=decision,
                                                                      # Outcome=1 if match, else 0
                                                                      function=lambda x: int(x[0][0]>x[0][1])),
                               # Set ControlSignal for EM[store_prob]
                               #   to 1 if match or hazard rate is realized (i.e., store stimulus and end trial)
                               #   else 0 (i.e., don't store stimulus and continue retrieving)
                               function=lambda outcome: int(bool(outcome)
                                                            or (np.random.random() < retrieval_hazard_rate)),
                               # ---------
                               # # VERSION *WITHOUT* ObjectiveMechanism:
                               # monitor_for_control=decision,
                               # # Set Evaluate outcome and set ControlSignal for EM[store_prob]
                               # #   - outcome is received from decision as one hot in the form: [[match, no-match]]
                               # function=lambda outcome: int(int(outcome[0][1]>outcome[0][0])
                               #                              or (np.random.random() > retrieval_hazard_rate)),
                               # ---------
                               control=(STORAGE_PROB, em))

    nback_model = Composition(name=NBACK_MODEL,
                              # nodes=[stim, context, task, ffn, em, logit, decision, control],
                              nodes=[stim, context, task, ffn, em, concat_input, decision, control],
                              # Terminate trial if value of control is still 1 after first pass through execution
                              termination_processing={TimeScale.TRIAL: And(Condition(lambda: control.value),
                                                                           AfterPass(0, TimeScale.TRIAL))},
                              )
    nback_model.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
    nback_model.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])
    nback_model.add_projection(MappingProjection(), task, task_input)
    nback_model.add_projection(MappingProjection(), output, decision, IDENTITY_MATRIX)
    nback_model.add_projection(MappingProjection(), concat_input, stim_context_input)

    if DISPLAY_MODEL:
        nback_model.show_graph(
            # show_cim=True,
            # show_node_structure=ALL,
            # show_dimensions=True
        )

    print(f'full model constructed')
    return nback_model
#endregion

#region =====================================STIMULUS GENERATION =======================================================

def _get_stim_set(stim_size=STIM_SIZE, num_stim=NUM_STIM):
    """Construct an array of unique stimuli for use in an experiment, used by train_network() and run_model()"""
    # For now, use one-hots
    return np.eye(stim_size)[0:num_stim]

def _get_task_input(nback_level):
    """Construct input to task Mechanism for a given nback_level, used by train_network() and run_model()"""
    task_input = [0] * TASK_SIZE
    task_input[nback_level - NBACK_LEVELS[0]] = 1
    return task_input

def _get_training_inputs(network:AutodiffComposition,
                         num_training_sets_per_epoch:int=NUM_TRAINING_SETS_PER_EPOCH,
                         num_epochs:int=1,
                         nback_levels:int=NBACK_LEVELS,
                         foils_allowed_before:bool=True,
                         return_generator:bool=True
                         )->(dict, list, int):
    """Construct set of training stimuli used by ffn.learn() in train_network()
    Construct one example of each condition for each stimulus and each nback_level:
        MATCH_NO_FOIL (match):  stim_current = stim_retrieved  and context_current = context_retrieved
        MATCH_WITH_FOIL (stim_lure):  stim_current = stim_retrieved  and context_current != context_retrieved
        NO_MATCH_WITH_FOIL (context_lure):  stim_current != stim_retrieved  and context_current == context_retrieved
        NO_MATCH_NO_FOIL (non_lure):  stim_current != stim_retrieved  and context_current != context_retrieved
    Arguments
    ---------
    network: AutodiffComposition
        network to be trained.
    num_training_sets_per_epoch: int : default 1
        number of complete sets of training stimuli to be trained in a single epoch (i.e., for a single weight update);
        used to determine batch_size (= num_training_sets_per_epoch * number of trials in a training set)
    num_epochs: int : default 1
        number of epochs of training to be executed (passed to `learn <AutodiffComposition.learn>`)
    nback_levels: list[int] : default NBACK_LEVELS
        list of n-back levels for which to generate training sets;
        nback_levels themselves must be specified in the global NBACK_LEVELS
    foils_allowed_before: bool : default True
        only allows foils to occur after the target (e.g., for 2-back: XBAA and not ABXA)
    return_generator: bool : True
        return generator rather than explicit list of training stimuli
    Return
    ------
    (training_set: dict,  conditions: list[TrialType], batch_size: int)
    """

    assert is_iterable(nback_levels) and all([0<i<=MAX_NBACK_LEVELS for i in nback_levels])
    stimuli = _get_stim_set()
    num_stim = len(stimuli)
    if num_stim != NUM_STIM:
        warnings.warn(f"Number of stimuli used for training set ({num_stim}) is not same "
                      f"as specified in NUM_STIM ({NUM_STIM}); unexpexpected results may occur.")
    context_fct = DriftOnASphereIntegrator(initializer=np.random.random(CONTEXT_SIZE - 1),
                                           noise=CONTEXT_DRIFT_NOISE,
                                           dimension=CONTEXT_SIZE)

    def trial_gen():
        for ep in range(num_epochs):
            stim_array = []
            target = []
            current_task = []
            for n in range(num_training_sets_per_epoch):
                for nback_level in nback_levels:
                    # Construct one hot encoding for nback level
                    task_input = _get_task_input(nback_level)
                    # Construct an example of each trial type for each stimulus
                    for i in range(num_stim):
                        contexts = []
                        # Get current stimulus and a randomly selected distractor
                        stims = list(stimuli.copy())
                        # Get stim, and remove from stims so distractor can be picked randomly from remaining ones
                        current_stim = stims.pop(i)
                        # Pick distractor randomly from stimuli remaining in set
                        distractor_stim = stims[np.random.randint(0,len(stims))]

                        # IF foils_allowed_before IS True:
                        #    total_contexts = nback+2 contexts:  [0]=potential foil; [1]=nback, [2+nback]=current
                        # IF foils_allowed_before IS False:
                        #    total_contexts = nback+1 contexts:  [0]=nback, [1+nback]=current
                        total_contexts = nback_level + 1 + int(foils_allowed_before)
                        for j in range(total_contexts):
                            contexts.append(context_fct(CONTEXT_DRIFT_RATE))
                        # Get current context as last in list
                        current_context = contexts.pop(-1)
                        # Get nback context as either first or second in list, based on foils_allowed_before
                        nback_context = contexts.pop(int(foils_allowed_before))
                        # Choose distractor foil randomly from remaining contexts
                        # (note:  if foils_allowed_before = False, only those after target remain in list)
                        distractor_context = contexts[np.random.randint(0,len(contexts))]

                        # Assign retrieved stimulus and context accordingly to trial_type
                        for trial_type in TrialTypes:
                            # Assign current stimulus as retrieved stimulus for MATCH_ trials
                            if trial_type in {TrialTypes.MATCH_NO_FOIL, TrialTypes.MATCH_WITH_FOIL}:
                                stim_retrieved = current_stim
                            # Assign distractor stimulus as retrieved stimulus for NON_MATCH_ trials
                            else:
                                stim_retrieved = distractor_stim
                            # Assign nback context as retrieved context for _NO_FOIL trials
                            if trial_type in {TrialTypes.MATCH_NO_FOIL,TrialTypes.NO_MATCH_NO_FOIL}:
                                context_retrieved = nback_context
                            # Assign distractor context as retrieved context for _WITH_FOIL trials
                            else:
                                context_retrieved = distractor_context
                            # Assign target
                            if trial_type in {TrialTypes.MATCH_NO_FOIL, TrialTypes.MATCH_WITH_FOIL}:
                                target.append([1,0])
                            else:
                                target.append([0,1])
                            current_task.append([task_input])
                            stim_array.append([np.array(current_stim.tolist()
                                                        + current_context.tolist()
                                                        + stim_retrieved.tolist()
                                                        + context_retrieved.tolist())])
            inputs = {network.nodes[FFN_INPUT]: stim_array}
            targets = {network.nodes[FFN_OUTPUT]: target}

            training_set = {INPUTS: inputs,
                            TARGETS: targets,
                            # EPOCHS: num_epochs}
                            EPOCHS: 1}

            yield training_set

    # Get training_set
    if return_generator:
        training_set = trial_gen()
    else:
        training_set = next(trial_gen())

    # Get sequence of conditions
    conditions = []
    for n in range(num_training_sets_per_epoch):
        for l in range(len(nback_levels)):
            for i in range(num_stim):
                for trial_type in TrialTypes:
                    conditions.append(trial_type.name)

    batch_size = num_stim * len(TrialTypes) * len(nback_levels) * num_training_sets_per_epoch

    return training_set, conditions, batch_size,

def _get_run_inputs(model,
                    nback_level:int,
                    context_drift_rate:float=CONTEXT_DRIFT_RATE,
                    num_trials:int=NUM_TRIALS,
                    mini_blocks:bool=True,
                    inputs_source:Stimuli=None)->(dict,list):
    """Construct set of stimulus inputs for run_model(), balancing across four conditions.
    Trial_type assignments:
      - trial_types are assigned to subseqs of nback_level+1 stimuli that are concatenated to form the full trial seq
      - trial_type subseqs are constructed in get_stim_subseq_for_trial_type(), by randomly picking a target stimulus,
          and then assigning the preceding stimuli in the subseq to conform the trial_type
      - the balancing of trial_types is enforced *only* for the last stimulus in each set;
          the others are inferred and may not be fully balanced across conditions
          (depending on number of stimuli, this may be impossible).
    Mini_blocks:
      - if True (default) trials are sequenced in mini-blocks each of which contains one set of trials
          for each trial_type; order of trial_type subseq within each mini_block is randomized across them;
          number of trials in a mini-block = nback_level+1 * num_trial_types; trials not assigned to
          mini_blocks (i.e., modulus of num_trial % (num_mini_blocks * mini_block_size) are assigned random
          stimuli and trial_type is inferred posthoc).
      - if False, sampling of trial_types is balanced,
          but order of presentation is randomized over the entire sequence
    Returns
    -------
    dict with inputs to each input node of model for each trial and array with corresponding trial_type_assignments
    """

    def generate_stim_sequence(nback_level, num_trials):
        assert nback_level in {2,3} # At present, only 2- and 3-back levels are supported

        stim_set = _get_stim_set()

        def get_stim_subseq_for_trial_type(trial_type):
            """Return stimulus seq (as indices into stim_set) for the specified trial_type."""
            subseq_size = nback_level + 1
            subseq = [None] * subseq_size
            curr_stim = subseq[nback_level] = random.choice(np.arange(len(stim_set)))
            other_stims = np.setdiff1d(np.arange(len(stim_set)),curr_stim).tolist()
            trial_type = list(TrialTypes)[trial_type]

            if trial_type == TrialTypes.MATCH_NO_FOIL:           # ABA (2-back) or ABCA (3-back)
                subseq[0] = curr_stim  # Assign nback stim to match
                # Assign remaining items in sequence to any stimuli other than curr_stim
                subseq[1:nback_level] = random.sample(other_stims, nback_level - 1)
            elif trial_type == TrialTypes.MATCH_WITH_FOIL:        # AAA (2-back) or AABA (3-back)
                subseq[0] = curr_stim  # Assign nback stim to match current stim
                subseq[1] = curr_stim  # Assign curr_stim to stim next to nback as foil
                # Assign any remaining items in sequence to any stimuli other than curr_stim
                subseq[2:nback_level] = random.sample(other_stims, nback_level - 2)
            elif trial_type == TrialTypes.NO_MATCH_NO_FOIL:       # ABB (2-back) or BCDA (3-back)
                # Assign remaining items in sequence to any stimuli than curr_stim
                subseq[0:nback_level] = random.sample(other_stims, nback_level)
            elif trial_type == TrialTypes.NO_MATCH_WITH_FOIL:     # BAA (2-back) or BACA (3-back)
                # Assign remaining items in sequence to any stimuli than curr_stim
                subseq[1] = curr_stim  # Assign curr_stim to stim next to nback as foil
                subseq[0:1] = random.sample(other_stims, 1)
                subseq[2:nback_level] = random.sample(other_stims, nback_level - 2)
            assert None not in subseq, "Failed to assign all stims for subseq in get_stim_subseq_for_trial_type."
            return subseq

        def get_trial_type_for_stim(subseq):
            if subseq[-1] == subseq[0] and not subseq[-1] in subseq[1:-1]:
                return TrialTypes.MATCH_NO_FOIL.value
            elif subseq[-1] == subseq[0] and subseq[-1] in subseq[0:-1]:
                return TrialTypes.MATCH_WITH_FOIL.value
            elif subseq[-1] not in subseq[0:-1]:
                return TrialTypes.NO_MATCH_NO_FOIL.value
            elif subseq[-1] != subseq[0] and subseq[-1] in subseq[0:-1]:
                # Note: for 3back, this includes: BAXA, BXAA, and BAAA
                return TrialTypes.NO_MATCH_WITH_FOIL.value

        subseq_size = nback_level + 1
        num_sub_seqs = int(num_trials / num_trial_types)
        extra_trials = num_trials % num_trial_types

        # Construct seq of mini-blocks (subseqs) each containing one sample of each trial_type in a random order
        #    note: this is done over number of mini-blocks that fit into num_trials;
        #          remaining trials are randomly assigned trial_types below

        num_mini_blocks = int(num_trials / (num_trial_types * (nback_level + 1)))
        mini_block_size = subseq_size * num_trial_types # Number of trials in a mini_block
        seq_of_trial_type_subseqs = []
        # Generate randomly ordered trial_type assignments for subseqs in each mini_block
        for i in range(num_mini_blocks):
            seq_of_trial_type_subseqs.extend(random.sample(range(num_trial_types), num_trial_types))
        if not mini_blocks:
            # Randomize the order of trial types across the entire sequence:
            random.shuffle(seq_of_trial_type_subseqs)

        if extra_trials:  # Warn if conditions can't be fully balanced
            warnings.warn(f"Number of trials ({num_trials}) precludes fully balancing across all five trial types")

        stim_seq = [None] * num_trials
        trial_type_seq = [None] * num_trials
        # Construct actual stimulus sequence by getting stimuli for each subseq, up to num_sub_seqs
        #   note: the trial type only applies (and a trial_type is only assigned) to the last trial of each subsequence;
        #         trial_type of preceding ones set below on the full sequence of stimuli is assigned
        # stim_seq.append(get_stim_seq_for_trial_type(i) for i in seq_of_trial_type_subseqs) # <- CONDENSED VERSION
        for i, trial_type in enumerate(seq_of_trial_type_subseqs):  # <- LOOP VERSION
            idx = i * subseq_size
            # Get seq of stimuli for subseq of specified trial_type
            stim_seq[idx:idx + nback_level + 1] = get_stim_subseq_for_trial_type(trial_type)
            # Assign trial_type to last stim in subseq (since it was constructed specifically for that trial_type)
            trial_type_seq[idx + nback_level] = list(TrialTypes)[trial_type].value
        # Pad remainder to get to num_trials with randomly selected stimuli
        stim_seq.extend(random.sample(range(num_trial_types),extra_trials))
        # Infer trial_types for all remaining stimuli (which should currently be marked as None)
        for i in range(subseq_size,num_trials,subseq_size):
            for j in range(i,i + nback_level):
                assert trial_type_seq[j] is None, f"trial_type should still be None for trial {j}."
                trial_type_seq[j] = get_trial_type_for_stim(stim_seq[i - subseq_size:i])
                assert True

        trial_type_counts = [None] * num_trial_types
        for i in range(num_trial_types):
            trial_type_counts[i] = trial_type_seq.count(i)

        return(stim_seq, trial_type_seq)

    def get_input_sequence(nback_level, num_trials=NUM_TRIALS, inputs_source=None):
        """Construct sequence of stimulus and trial_type indices"""
        # Use SweetPea if specified
        if inputs_source == Stimuli.SWEETPEA:
            if nback_level == 2:
                from stim.SweetPea.sweetpea_script import create_two_back
                Kane_stimuli = {stim:idx for idx, stim in enumerate(['B', 'F', 'H', 'K', 'M', 'Q', 'R', 'X'])}
                Kane_trial_types = {'1/1/0': TrialTypes.MATCH_NO_FOIL,
                                    '1/2/0': TrialTypes.MATCH_WITH_FOIL,
                                    '2/1/0': TrialTypes.NO_MATCH_NO_FOIL,
                                    '2/2/0': TrialTypes.NO_MATCH_WITH_FOIL}
                stim_dict = create_two_back()
                assert True
                stim_seq = [Kane_stimuli[i.upper()] for i in stim_dict[0]['letter']]
                trial_type_seq = [Kane_trial_types[i] if i else None for i in stim_dict[0]['condi']]
                assert True
            else:
                raise Exception(f"Use of SweetPea currently restricted to nback_level = 2")
        elif inputs_source == Stimuli.KANE_STIMULI:
            assert False, "KANE STIMULI NOT YET SUPPORTED AS INPUTS"
        # Else, use local algorithm
        else:
            stim_seq, trial_type_seq = generate_stim_sequence(nback_level, num_trials)
            # Return list of corresponding stimulus input vectors

        input_set = [_get_stim_set()[i] for i in stim_seq]
        return input_set, trial_type_seq

    input_set, trial_type_seq = get_input_sequence(nback_level, num_trials, inputs_source=inputs_source)
    return {model.nodes[MODEL_STIMULUS_INPUT]: input_set,
            model.nodes[MODEL_CONTEXT_INPUT]: [[context_drift_rate]] * num_trials,
            model.nodes[MODEL_TASK_INPUT]: [_get_task_input(nback_level)] * num_trials}, \
           trial_type_seq
#endregion

#region ================================== MODEL EXECUTION METHODS =====================================================

def train_network(network:AutodiffComposition,
                  training_set:dict=None,
                  minibatch_size:int=MINIBATCH_SIZE,
                  learning_rate:float=LEARNING_RATE,
                  num_epochs:int=NUM_EPOCHS,
                  save_weights_to:Union[Path,str,None]=None
                  )->Path:
    """**Train feedforward network** on example stimulus sequences for each condition.

    Arguments
    ---------
    network: AutodiffComposition
        network to be trained;  this must be an `AutodiffComposition`.
    training_set: dict : default _get_training_inputs()
        inputs (see `Composition_Input_Dictionary`), including targets (`Composition_Target_Inputs`)
        to use for training;  these are constructed in a call to _get_training_inputs() if not specified here.
    minibatch_size: int : default MINIBATCH_SIZE
        number of inputs that will be presented within a single training epoch
        (i.e. over which weight changes are aggregated and applied);  if it is not specified and MINIBATCH_SIZE=None,
        it is determined by the batch_size for an epoch returned in the call to _get_training_inputs().
    learning_rate: float : default LEARNING_RATE
        learning_rate to use for training;  this overrides the value of `learning_rate
        <AutodiffComposition.learning_rate>` specified in construction of the network.  If None is specified
         here, either the value specified at construction, or the default for `AutodiffComposition
         <AutodiffComposition.learning_rate>` is used.
    num_epochs: int : default NUM_EPOCHS
        number of training epochs (i.e., sets of minibatchs) to execute during training.
    save_weights_to: Path : 'results' subdirectory of current working directory
        location to store weights at end of training.

    Returns
    -------
    Path containing saved weights for matrices of feedforward Projections in network.
    """
    print(f"\nconstructing training set for '{network.name}'...")
    if training_set is None:
        training_set, conditions, batch_size = \
            _get_training_inputs(network=network,
                                 num_training_sets_per_epoch=NUM_TRAINING_SETS_PER_EPOCH,
                                 foils_allowed_before=FOILS_ALLOWED_BEFORE,
                                 num_epochs=num_epochs,
                                 nback_levels=NBACK_LEVELS)
    else:
        batch_size = len(training_set[MODEL_STIMULUS_INPUT])

    minibatch_size = minibatch_size or batch_size
    print(f'num training stimuli per training set: {minibatch_size//NUM_TRAINING_SETS_PER_EPOCH}')
    print(f'num training sets per epoch: {NUM_TRAINING_SETS_PER_EPOCH}')
    print(f'total num training stimuli per epoch: {minibatch_size}')
    print(f'num epochs (weight updates): {num_epochs}')
    print(f'total num trials: {num_epochs*minibatch_size}')
    print(f"\ntraining '{network.name}' (started at {time.localtime()[3]%12}:{'{:02}'.format(time.localtime()[4])})...")
    start_time = timeit.default_timer()
    network.learn(inputs=training_set,
                  minibatch_size=minibatch_size,
                  num_trials=minibatch_size,
                  # report_output=REPORT_OUTPUT,
                  # report_progress=REPORT_PROGRESS,
                  # report_progress=ReportProgress.ON,
                  learning_rate=learning_rate,
                  # execution_mode=ExecutionMode.LLVMRun
                  # execution_mode=ExecutionMode.Python
                  execution_mode=ExecutionMode.PyTorch
                  )
    stop_time = timeit.default_timer()
    print(f"training of '{network.name}' done")
    training_time = stop_time - start_time
    if training_time <= 60:
        training_time_str = f'{int(training_time)} seconds'
    else:
        training_time_str = f'{int(training_time/60)} minutes {int(training_time%60)} seconds'
    print(f'training time: {training_time_str} for {num_epochs} epochs')
    path = network.save(filename=save_weights_to, directory="results_original")
    print(f'saved weights to: {save_weights_to}')
    return path
    # print(f'saved weights sample: {network.nodes[FFN_H1].path_afferents[0].matrix.base[0][:3]}...')
    # network.load(path)
    # print(f'loaded weights sample: {network.nodes[FFN_H1].path_afferents[0].matrix.base[0][:3]}...')

def network_test(network:AutodiffComposition,
                 load_weights_from:Union[Path,str,None]=None,
                 nback_levels=NBACK_LEVELS,
                 )->(dict,list,list,list,list,list):

    print(f"constructing test set for '{network.name}'...")
    test_set, conditions, set_size = _get_training_inputs(network=network,
                                                          num_epochs=1,
                                                          nback_levels=NBACK_LEVELS,
                                                          foils_allowed_before=FOILS_ALLOWED_BEFORE,
                                                          return_generator=False)
    print(f'total num trials: {set_size}')

    inputs = [test_set[INPUTS][network.nodes['CURRENT INPUT LAYER']][i] for i in range(set_size)]
    # cxt_distances = [Distance(metric=COSINE)([inputs[i][2],inputs[i][3]]) for i in range(set_size)]
    current_stim_idx = slice(0,STIM_SIZE)
    current_context_idx = slice(current_stim_idx.stop, current_stim_idx.stop + CONTEXT_SIZE)
    retrieved_stim_idx = slice(current_context_idx.stop, current_context_idx.stop + STIM_SIZE)
    retrieved_context_idx = slice(retrieved_stim_idx.stop, retrieved_stim_idx.stop + CONTEXT_SIZE)
    stim_distances = [Distance(metric=COSINE)([inputs[i][0][current_stim_idx],
                                              inputs[i][0][retrieved_stim_idx]]) for i in range(set_size)]
    cxt_distances = [Distance(metric=COSINE)([inputs[i][0][current_context_idx],
                                              inputs[i][0][retrieved_context_idx]]) for i in range(set_size)]
    targets = list(test_set[TARGETS].values())[0]
    trial_type_stats = []

    num_items_per_nback_level = int(set_size / NUM_NBACK_LEVELS)
    for i in range(NUM_NBACK_LEVELS):
        start = i * num_items_per_nback_level
        stop = start + num_items_per_nback_level
        stimulus_distances_for_level = np.array(stim_distances[start:stop])
        context_distances_for_level = np.array(cxt_distances[start:stop])
        conditions_for_level = np.array(conditions[start:stop])
        for trial_type in TrialTypes:
            trial_type_stats.append(
                (f'{NBACK_LEVELS[i]}-back', trial_type.name, trial_type.value,
                 context_distances_for_level[np.where(conditions_for_level==trial_type.name)].mean(),
                 context_distances_for_level[np.where(conditions_for_level==trial_type.name)].std()))

    # FIX: COMMENTED OUT TO TEST TRAINING LOSS
    if load_weights_from:
        print(f"nback_model loading '{FFN_COMPOSITION}' weights from {load_weights_from}...")
        network.load(filename=load_weights_from)

    network.run(inputs=test_set[INPUTS], report_progress=ReportProgress.ON)

    if ANALYZE_RESULTS:
        coded_responses, stats = analyze_results([network.results,conditions], test=True)
        import torch
        cross_entropy_loss = \
            [network.loss(torch.Tensor(output[0]),torch.Tensor(np.array(target))).detach().numpy().tolist()
             for output, target in zip(network.results, targets)]
    coded_responses_flat = []
    for nback_level in nback_levels:
        coded_responses_flat.extend(coded_responses[nback_level])
    return inputs, cxt_distances, targets, conditions, network.results, coded_responses_flat, cross_entropy_loss, \
           trial_type_stats, stats

def run_model(model,
              load_weights_from:Union[Path,str,None]=None,
              context_drift_rate:float=CONTEXT_DRIFT_RATE,
              num_trials:int=NUM_TRIALS,
              inputs_source:Stimuli=None,
              report_output:ReportOutput=REPORT_OUTPUT,
              report_progress:ReportProgress=REPORT_PROGRESS,
              animate:Union[dict,bool]=ANIMATE,
              save_results_to:Union[Path,str,None]=None
              )->list:
    """**Run model** for all nback levels with a specified context drift rate and number of trials

    Arguments
    --------
    load_weights_from:  Path
        specifies file from which to load pre-trained weights for matrices of FFN_COMPOSITION.
    context_drift_rate: float
        specifies drift rate as input to CONTEXT_INPUT, used by DriftOnASphere function of FFN_CONTEXT_INPUT.
    num_trials: int
        number of trials (stimuli) to run.
    report_output: ReportOutput
        specifies whether to report results during execution of run (see `Report_Output` for additional details).
    report_progress: ReportProgress
        specifies whether to report progress of execution during run (see `Report_Progress` for additional details).
    animate: dict or bool
        specifies whether to generate animation of execution (see `ShowGraph_Animation` for additional details).
    save_results_to: Path
        specifies location to save results of the run along with trial_type_sequences for each nback level;
        if None, those are returned by call but not saved.
    """
    ffn = model.nodes[FFN_COMPOSITION]
    em = model.nodes[EM]
    if load_weights_from:
        print(f"nback_model loading '{FFN_COMPOSITION}' weights from {load_weights_from}...")
        ffn.load(filename=load_weights_from)
    print(f"'{model.name}' executing...")
    trial_type_seqs = [None] * NUM_NBACK_LEVELS
    start_time = timeit.default_timer()
    for i, nback_level in enumerate(NBACK_LEVELS):
        # Reset episodic memory for new task using first entry (original initializer)
        em.function.reset(em.memory[0])
        inputs, trial_type_seqs[i] = _get_run_inputs(model=model,
                                                     nback_level=nback_level,
                                                     context_drift_rate=context_drift_rate,
                                                     num_trials=num_trials,
                                                     inputs_source=inputs_source)
        model.run(inputs=inputs,
                  report_output=report_output,
                  report_progress=report_progress,
                  animate=animate
                  )
    # print("Number of entries in EM: ", len(model.nodes[EM].memory))
    stop_time = timeit.default_timer()
    assert len(model.nodes[EM].memory) == NUM_TRIALS + 1 # extra one is for initializer
    if REPORT_PROGRESS == ReportProgress.ON:
        print('\n')
    print(f"'{model.name}' done: {len(model.results)} trials executed")
    execution_time = stop_time - start_time
    if execution_time <= 60:
        execution_time_str = f'{int(execution_time)} seconds'
    else:
        execution_time_str = f'{int(execution_time/60)} minutes {int(execution_time%60)} seconds'
    print(f'execution time: {execution_time_str}')
    results = np.array([model.results, trial_type_seqs], dtype=object)
    if save_results_to:
        np.save(save_results_to, results)
    # print(f'results: \n{model.results}')
    if ANALYZE_RESULTS:
        coded_responses, stats = analyze_results(results, test=False)
    return results, coded_responses, stats
#endregion

#region ================================= MODEL PERFORMANCE ANALYSIS ===================================================

def analyze_results(results:list,
                    nback_levels:list=NBACK_LEVELS,
                    test:bool=False
                    )->(dict,dict):
    """**Analyze and plot results** of executed model.

    Arguments
    --------
    results: ndarray
        results returned from `run_model <nback.run_model>`.
    nback_levels: list : default NBACK_LEVELS
        list of nback levels executed in run() or test()
    test: bool : default False
        if True, analyze results for running ffn on set of stimuli used for training
        else, anaylze results of running full model using experimental sequence of stimuli
    """
    responses_and_trial_types = [None] * len(nback_levels)
    stats = np.zeros((len(nback_levels),num_trial_types))
    MATCH = 'match'
    NON_MATCH = 'non-match'
    num_trials = int(len(results[0]) / len(nback_levels))

    # FOR TEST
    if test:
        print(f"\n\nTest results (of ffn on training set):")
        for i, nback_level in enumerate(nback_levels):
            # conditions = results[1][i]
            conditions = results[1]
            # Code responses for given nback_level as 1 (match) or 0 (non-match)
            responses_for_nback_level = [r[0] for r in results[0][i * num_trials:i * num_trials + num_trials]]
            responses_for_nback_level = [MATCH if r[0] > r[1] else NON_MATCH for r in responses_for_nback_level]
            responses_and_trial_types[i] = list(zip(responses_for_nback_level, conditions))
            for j, trial_type in enumerate(TrialTypes):
                relevant_data = [[response, condition] for response, condition in zip(responses_for_nback_level, conditions)
                                 if condition == trial_type.name]
                # Report % matches in each condition (should be 1.0 for MATCH trials and 0.0 for NON-MATCH trials
                stats[i][j] = [d[0] for d in relevant_data
                                        if d[0] is not None].count(MATCH) / (len(relevant_data))
            print(f"nback level {nback_level}:")
            for j, performance in enumerate(stats[i]):
                print(f"\t{list(TrialTypes)[j].name}: {performance:.1f}")

    # FOR RUN
    else:
        print(f"\n\nTest results (running full model on experimental sequence):")
        for i, nback_level in enumerate(nback_levels):
            conditions = results[1][i]
            # Code responses for given nback_level as 1 (match) or 0 (non-match)
            responses_for_nback_level = [r[0] for r in results[0][i * num_trials:i * num_trials + num_trials]]
            responses_for_nback_level = [MATCH if r[0] > r[1] else NON_MATCH for r in responses_for_nback_level]
            responses_and_trial_types[i] = list(zip(responses_for_nback_level, conditions))
            for j, trial_type in enumerate(TrialTypes):
                relevant_data = [[response, condition] for response, condition in zip(responses_for_nback_level,
                                                                                      conditions)
                                 if condition == trial_type.value]
                # Report % matches in each condition (should be 1.0 for MATCH trials and 0.0 for NON-MATCH trials
                stats[i][j] = [d[0] for d in relevant_data
                                        if d[0] is not None].count(MATCH) / (len(relevant_data))
            print(f"nback level {nback_level}:")
            for j, performance in enumerate(stats[i]):
                print(f"\t{list(TrialTypes)[j].name}: {performance:.1f}")

    data_dict = {k:v for k,v in zip(nback_levels, responses_and_trial_types)}
    stats_dict = {}
    for i, nback_level in enumerate(nback_levels):
        stats_dict.update({nback_level: {trial_type.name:stat for trial_type,stat in zip(TrialTypes, stats[i])}})

    return data_dict, stats_dict

def _compute_dprime(hit_rate, fa_rate):
    """returns dprime and sensitivity
    """
    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)
    # hit_rate = clamp(hit_rate, 0.01, 0.99)
    # fa_rate = clamp(fa_rate, 0.01, 0.99)

    dl = np.log(hit_rate * (1 - fa_rate) / ((1 - hit_rate) * fa_rate))
    c = 0.5 * np.log((1 - hit_rate) * (1 - fa_rate) / (hit_rate * fa_rate))
    return dl, c

def _plot_results(response_and_trial_types, stats):

    # SCORE IN NBACK-PAPER IS RETURNED BY THIS METHOD:
    # def run_EMexp(neps,tsteps,argsD):
    #   score = np.zeros([2,4,tsteps,neps])
    #   for ep in range(neps):
    #     for nback in nbackL:
    #       for seq_int,tstep in itertools.product(range(4),np.arange(5,tsteps)):
    #         print(ep,tstep)
    #         stim,ctxt,ytarget = generate_trial(nback,tstep,stype=seq_int)
    #         yhat = run_model_trial(stim,ctxt,nback-2,argsD)
    #         [nback-2,seq_int,tstep,ep] = int(yhat==ytarget)
    #   print(score.shape) # nback,seqtype,tsteps,epoch
    #   acc = score.mean((2,3))
    #   return acc,score

    import matplotlib as plt
    hits_stderr = np.concatenate((score.mean(2).std(-1) / np.sqrt(neps))[:,(0,1)])
    correj_stderr = np.concatenate((score.mean(2).std(-1) / np.sqrt(neps))[:,(2,3)])
    d,s = _compute_dprime(
      np.concatenate(score.mean(2)[:,(0,1)]),
      np.concatenate(score.mean(2)[:,(2,3)])
    )
    print(d.shape,s.shape)
    dprime_stderr = d.std(-1) / np.sqrt(neps)
    bias_stderr = s.std(-1) / np.sqrt(neps)
    #%%
    # 2back-target, 2back-lure, 3back-target, 3back-lure
    hits = np.concatenate(acc[:,(0,1)])
    correj = np.concatenate(acc[:,(2,3)])
    dprime = np.zeros(4)
    bias = np.zeros(4)
    for i in range(4):
        d,s = _compute_dprime(hits[i], 1 - correj[i])
        dprime[i]=d
        bias[i]=s

    #%%
    f,axar = plt.subplots(2,2,figsize=(15,8))
    axar=axar.reshape(-1)
    cL = ['blue','darkblue','lightgreen','forestgreen']
    labL = ['2b,ctrl','2b,lure','3b,ctrl','3b,lure']

    # correct reject
    ax = axar[0]
    ax.set_title('correct rejection')
    ax.bar(range(4),correj,color=cL,yerr=correj_stderr)

    # hits
    ax = axar[1]
    ax.set_title('hits')
    ax.bar(range(4),hits,color=cL,yerr=hits_stderr)

    #
    ax = axar[2]
    ax.set_title('dprime')
    ax.bar(range(4),dprime,color=cL,yerr=dprime_stderr)

    #
    ax = axar[3]
    ax.set_title('bias')
    ax.bar(range(4),bias,color=cL,yerr=bias_stderr)

    ##
    for ax in axar[:2]:
        ax.set_xticks(np.arange(4))
        ax.set_xticklabels(labL)
        ax.set_ylim(0,1)

    plt.savefig('figures/EMmetrics-%s-t%i.jpg' % (mtag,tstamp))
    plt.savefig('figures/EMmetrics_yerr-%s-t%i.svg' % (mtag,tstamp))
#endregion


#region ===================================== SCRIPT EXECUTION =========================================================
# Construct, train and/or run model based on settings at top of script

# Only execute if called from command line (i.e., not on import)
if __name__ == '__main__':
    if CONSTRUCT_MODEL:
        nback_model = construct_model()

    if TRAIN_FFN:
        weights_filename = f'results/ffn.wts_nep_{NUM_EPOCHS}_lr_{str(LEARNING_RATE).split(".")[1]}.pnl'
        weights_path = Path('/'.join([os.getcwd(), weights_filename]))
        saved_weights = train_network(nback_model.nodes[FFN_COMPOSITION],
                                      save_weights_to=weights_path
                                      )

    if TEST_FFN:
        try:
            weights_path
        except:
            weights_filename = f'results/ffn.wts_nep_{NUM_EPOCHS}_lr_{str(LEARNING_RATE).split(".")[1]}.pnl'
            weights_path = Path('/'.join([os.getcwd(), weights_filename]))

        inputs, cxt_distances, targets, conditions, results, coded_responses, ce_loss, \
        trial_type_stats, stats = \
            network_test(nback_model.nodes[FFN_COMPOSITION],
                         load_weights_from = weights_path
                         )
        headings = ['condition', 'inputs', 'target', 'context distance', 'results', 'coded response', 'ce loss']
        results = (headings,
                   list(zip(conditions, inputs, targets, cxt_distances, results, coded_responses, ce_loss)),
                   trial_type_stats,
                   stats)

        # SAVE RESULTS in CSV FORMAT
        import csv
        threshold = .005

        high_loss = [list(x) for x in [results[1][i] for i in range(64)] if x[6] > threshold]
        for i in range(len(high_loss)):
            high_loss[i][6] = '{:.4f}'.format(high_loss[i][6])
        high_loss.insert(0,headings)
        file = open('high_loss.csv', 'w+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows(high_loss)
        file.close()

        low_loss = [list(x) for x in [results[1][i] for i in range(64)] if x[6] <= threshold]
        for i in range(len(low_loss)):
            low_loss[i][6] = '{:.4f}'.format(low_loss[i][6])
        low_loss.insert(0,headings)
        file = open('low_loss.csv', 'w+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows(low_loss)
        file.close()

        full_results = [list(x) for x in [results[1][i] for i in range(64)]]
        for i in range(len(full_results)):
            full_results[i][6] = '{:.4f}'.format(full_results[i][6])
        full_results.insert(0,headings)
        file = open('full_results.csv', 'w+', newline ='')
        with file:
            write = csv.writer(file)
            write.writerows(full_results)
        file.close()

    if RUN_MODEL:
        results_path = Path('/'.join([os.getcwd(), f'results/nback.results_nep_{NUM_EPOCHS}_lr'
                                                   f'_{str(LEARNING_RATE).split(".")[1]}.pnl']))
        try:
            weights_path
        except:
            weights_filename = f'results/ffn.wts_nep_{NUM_EPOCHS}_lr_{str(LEARNING_RATE).split(".")[1]}.pnl'
            weights_path = Path('/'.join([os.getcwd(), weights_filename]))
        results = run_model(nback_model,
                            load_weights_from = weights_path,
                            save_results_to= results_path
                            # inputs_source=Stimuli.SWEETPEA,
                            )
    # if ANALYZE_RESULTS:
    #     coded_responses, stats = analyze_results(results,
    #                                              nback_levels=NBACK_LEVELS)
#endregion
