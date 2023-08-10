
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

The script conatins methods to construct, train, and run the model, and analyze the results of its execution:

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
otherwise, it either uses the current stimulus and temporal context to retrieve another sample from EM and repeat the
evaluation or, with a fixed probability (hazard rate), it responds "non-match".

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
DISPLAY_MODEL = True # True = show visual graphic of model
TRAIN_FFN = False  # True => train the FFN (WM)
TEST_FFN = False  # True => test the FFN on training stimuli (WM)
RUN_MODEL = False  # True => test the model on sample stimulus sequences
ANALYZE_RESULTS = False # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.OFF       # Sets console output during run
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

#region ========================================= PARAMETERS ===========================================================

# Fixed (structural) parameters:

# Layer Names:
CONTEXT_LAYER = 'CONTEXT_LAYER'

# Constructor parameters:  (values are from nback-paper)
STIM_SIZE = 8 # length of stimulus vector
CONTEXT_SIZE = 25 # length of context vector
HIDDEN_SIZE = STIM_SIZE * 4 # dimension of hidden units in ff
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
NUM_EPOCHS= 500 # 6250 # 12500 # 20000  # nback-paper: 400,000 @ one trial per epoch = 6,250 @ 64 trials per epoch
LEARNING_RATE=0.001  # nback-paper: .001

# Execution parameters:
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
NUM_TRIALS = 48 # number of stimuli presented in a trial sequence

# Names of Compositions and Mechanisms:
EM = "EPISODIC MEMORY (dict)"
DECISION = "DECISION"
CONTROLLER = "READ/WRITE CONTROLLER"

context_fct = DriftOnASphereIntegrator(initializer=np.random.random(CONTEXT_SIZE - 1),
                                       noise=CONTEXT_DRIFT_NOISE,
                                       dimension=CONTEXT_SIZE)

context_layer = TransferMechanism(name=CONTEXT_LAYER,
                                 size=CONTEXT_SIZE)

time_layer = TransferMechanism(name=TIME_LAYER,
                               function=context_fct,
                               TIME_SIZE)

retrieved_reward = TransferMechanism(name=RETRIEVED_REWARD,
                                    size=REWARD_SIZE)

retrieved_state = TransferMechanism(name=RETRIEVED_STATE,
                                    size=STATE_SIZE)


em = EpisodicMemoryMechanism(name=EM,
                             input_ports=[{NAME:"CONTEXT_FIELD",
                                           SIZE:CONTEXT_SIZE},
                                          {NAME:"TIME_FIELD",
                                           SIZE:TIME_SIZE}],
                             function=ContentAddressableMemory(
                                 initializer=[[[0] * CONTEXT_SIZE, [0] * TIME_SIZE]],
                                 distance_field_weights=[CONTEXT_RETRIEVAL_WEIGHT,
                                                         TIME_RETRIEVAL_WEIGHT],
                                 # equidistant_entries_select=NEWEST,
                                 selection_function=SoftMax(gain=RETRIEVAL_SOFTMAX_TEMP)))


EGO_comp = Composition(name=MODEL_NAME,
                       pathways=[])

