"""
QUESTIONS:
- WHERE DOES INPUT TO REWARD FIELD OF EM COME FROM?

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
    {}                               # show summary visual display of model
    # {'show_node_structure': ALL}       # show detailed view of node structures and projections
)
# TRAIN_MODEL = False                  # True => train the model
RUN_MODEL = False                      # True => run the model
ANALYZE_RESULTS = False                # True => output analysis of results of run
REPORT_OUTPUT = ReportOutput.OFF       # Sets console output during run
REPORT_PROGRESS = ReportProgress.OFF   # Sets console progress bar during run
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

#region ========================================= PARAMETERS ===========================================================

# Fixed (structural) parameters:

# Names:
MODEL_NAME = "EGO Model"
STATE_INPUT_LAYER_NAME = "STATE"
TIME_INPUT_LAYER_NAME = "TIME"
REWARD_INPUT_LAYER_NAME = "REWARD"
CONTEXT_LAYER_NAME = 'CONTEXT'
RETRIEVED_TIME_NAME = "RETRIEVED TIME"
RETRIEVED_REWARD_NAME = "RETRIEVED REWARD"
EM_NAME = "EPISODIC MEMORY"
DECISION_LAYER_NAME = "DECISION"
CONTROLLER_NAME = "READ/WRITE CONTROLLER"

# Constructor parameters:
STATE_SIZE = 8                 # length of state vector
TIME_SIZE = 25                 # length of time vector
CONTEXT_SIZE = 10              # length of context vector
REWARD_SIZE = 1                # length of reward vector
DECISION_SIZE = 2              # length of decision vector
CONTEXT_INTEGRATION_RATE = .1  # rate of integration of context vector
TIME_DRIFT_NOISE = 0.0         # noise used by DriftOnASphereIntegrator (function of Context mech)
STATE_RETRIEVAL_WEIGHT = 1     # weight of state field in retrieval from em
TIME_RETRIEVAL_WEIGHT = 1      # weight of time field in retrieval from em
CONTEXT_RETRIEVAL_WEIGHT = 1   # weight of context field in retrieval from em
REWARD_RETRIEVAL_WEIGHT = 0    # weight of reward field in retrieval from em
RETRIEVAL_SOFTMAX_GAIN = 10    # gain on softmax retrieval function
RETRIEVAL_HAZARD_RATE = 0.04   # rate of re=sampling of em following non-match determination in a pass through ffn
RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections

# Training parameters:
NUM_TRAINING_SETS_PER_EPOCH = 1
MINIBATCH_SIZE=None
NUM_EPOCHS= 500 # 6250 # 12500 # 20000  # EGO-paper: 400,000 @ one trial per epoch = 6,250 @ 64 trials per epoch
LEARNING_RATE=0.001  # EGO-paper: .001

# Execution parameters:
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
NUM_TRIALS = 48 # number of stimuli presented in a trial sequence

time_fct = DriftOnASphereIntegrator(initializer=np.random.random(TIME_SIZE - 1),
                                    noise=TIME_DRIFT_NOISE,
                                    dimension=TIME_SIZE)


def construct_model(model_name:str=MODEL_NAME,

                    state_input_name:str=STATE_INPUT_LAYER_NAME,
                    state_size:int=STATE_SIZE,

                    time_input_name:str=TIME_INPUT_LAYER_NAME,
                    time_size:int=TIME_SIZE,


                    context_name:str=CONTEXT_LAYER_NAME,
                    context_size:int=CONTEXT_SIZE,
                    context_integration_rate:float=CONTEXT_INTEGRATION_RATE,


                    reward_input_name = REWARD_INPUT_LAYER_NAME,
                    reward_size:int=REWARD_SIZE,

                    state_retrieval_weight:Union[float,int]=STATE_RETRIEVAL_WEIGHT,
                    context_retrieval_weight:Union[float,int]=CONTEXT_RETRIEVAL_WEIGHT,

                    retrieved_time_name:str=RETRIEVED_TIME_NAME,
                    time_retrieval_weight:Union[float,int]=TIME_RETRIEVAL_WEIGHT,

                    retrieved_reward_name:str=RETRIEVED_REWARD_NAME,
                    reward_retrieval_weight:Union[float,int]=REWARD_RETRIEVAL_WEIGHT,

                    em_name:str=EM_NAME,
                    retrieval_softmax_gain=RETRIEVAL_SOFTMAX_GAIN,
                    retrieval_hazard_rate=RETRIEVAL_HAZARD_RATE,
                    
                    controller_name=CONTROLLER_NAME,
                    
                    decision_layer_name:str=DECISION_LAYER_NAME,
                    decision_size:int=DECISION_SIZE,

                    )->Composition:

    state_input_layer = ProcessingMechanism(name=state_input_name,
                                              size=state_size)

    time_input_layer = ProcessingMechanism(name=time_input_name,
                                           size=time_size)

    context_layer = TransferMechanism(name=context_name,
                                      size=context_size,
                                      integrator_mode=True,
                                      integration_rate=context_integration_rate)

    reward_input_layer = ProcessingMechanism(name=reward_input_name,
                                              size=reward_size)

    retrieved_time_layer = TransferMechanism(name=retrieved_time_name,
                                       size=time_size)

    retrieved_reward_layer = TransferMechanism(name=retrieved_reward_name,
                                         size=reward_size)

    em = EpisodicMemoryMechanism(name=em_name,
                                 input_ports=[{NAME:"STATE_FIELD", SIZE:state_size},
                                              {NAME:"TIME_FIELD", SIZE:time_size},
                                              {NAME:"CONTEXT_FIELD", SIZE:context_size},
                                              {NAME:"REWARD_FIELD", SIZE:reward_size}
                                              ],
                                 function=ContentAddressableMemory(
                                     initializer=[[0] * state_size,
                                                  [0] * time_size,
                                                  [0] * context_size,
                                                  [0] * reward_size],
                                     distance_field_weights=[state_retrieval_weight,
                                                             time_retrieval_weight,
                                                             context_retrieval_weight,
                                                             reward_retrieval_weight],
                                     # equidistant_entries_select=NEWEST,
                                     selection_function=SoftMax(gain=retrieval_softmax_gain)))

    decision_layer = TransferMechanism(name=decision_layer_name,
                                       size=decision_size,
                                       function=SoftMax(output=PROB))

    def trial_number(variable,context):
        if context and context.composition:
            return [context.composition.get_current_execution_time(context)[TimeScale.TRIAL]]
        else:
            return variable

    # Control Mechanism
    #  Ensures current stimulus and context are only encoded in EM once (at beginning of trial)
    #    by controlling the storage_prob parameter of em:
    #      - if outcome of decision signifies a match or hazard rate is realized:
    #        - set  EM[store_prob]=1 (as prep encoding stimulus in EM on next trial)
    #        - this also serves to terminate trial (see nback_model.termination_processing condition)
    #      - if outcome of decision signifies a non-match
    #        - set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
    #        - continue trial
    retrieval_control_layer = ControlMechanism(name=controller_name,
                               default_variable=[[1]],  # Ensure EM[store_prob]=1 at beginning of first trial
                               monitor_for_control=decision_layer,
                               # Set Evaluate outcome and set ControlSignal for EM[store_prob]
                               #   - outcome is received from decision as one hot in the form: [[match, no-match]]
                               function=lambda outcome: int(int(outcome[0][1]>outcome[0][0])
                                                            or (np.random.random() < retrieval_hazard_rate)),
                               control=(STORAGE_PROB, em))


    EGO_comp = Composition(name=model_name,
                           pathways=[[state_input_layer, context_layer],  # Input
                                     [em],
                                     [retrieved_reward_layer, decision_layer, retrieval_control_layer]], # Decision
                           # Terminate trial if value of control is still 1 after first pass through execution
                           termination_processing={
                               TimeScale.TRIAL: And(Condition(lambda: retrieval_control_layer.value),
                                                    AfterPass(0, TimeScale.TRIAL))})
    # # Terminate trial if value of control is still 1 after first pass through execution
    # EGO_comp.show_graph()
    # EGO_comp.add_projection(MappingProjection(context_input_layer, context_layer))
    EGO_comp.add_nodes([time_input_layer, reward_input_layer, retrieved_time_layer])
    EGO_comp.add_projection(MappingProjection(state_input_layer, em.input_ports["STATE_FIELD"]))
    EGO_comp.add_projection(MappingProjection(time_input_layer, em.input_ports["TIME_FIELD"]))
    EGO_comp.add_projection(MappingProjection(context_layer, em.input_ports["CONTEXT_FIELD"]))
    EGO_comp.add_projection(MappingProjection(reward_input_layer, em.input_ports["REWARD_FIELD"]))
    EGO_comp.add_projection(MappingProjection(em.output_ports[0], context_layer))
    EGO_comp.add_projection(MappingProjection(em.output_ports[1], retrieved_time_layer))
    EGO_comp.add_projection(MappingProjection(em.output_ports[2], context_layer))
    EGO_comp.add_projection(MappingProjection(em.output_ports[3], retrieved_reward_layer))
    # EGO_comp.add_projection(MappingProjection(retrieved_reward_layer, decision_layer))

    print(f'{model_name} constructed')
    return EGO_comp

model = None
if CONSTRUCT_MODEL:
    model = construct_model()

if DISPLAY_MODEL is not None:
    if model:
        model.show_graph(**DISPLAY_MODEL)
    else:
        print("Model not yet constructed")
