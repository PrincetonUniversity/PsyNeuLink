"""
This implements a model of the `N-back task <https://en.wikipedia.org/wiki/N-back#Neurobiology_of_n-back_task>`_
described in `Beukers et al. (2022) <https://psyarxiv.com/jtw5p>`_.  The model uses a simple implementation of episodic
(content-addressable) memory to store previous stimuli and the temporal context in which they occured,
and a feedforward neural network to evaluate whether the current stimulus is a match to the n'th preceding stimulus
(n-back level).  This model is an example of proposed interactions between working memory (e.g., in neocortex) and
episodic memory e.g., in hippocampus and/or cerebellum) in the performance of tasks demanding of sequential processing
and control, and along the lines of models emerging machine learning that augment the use of recurrent neural networks
(e.g., long short-term memory mechanisms; LSTMs) for active memory and control with an external memory capable of
rapid storage and content-based retrieval, such as the Neural Turing Machine (NTN; `Graves et al., 2016
<https://arxiv.org/abs/1410.5401>`_), Episodic Planning Networks (EPN; `Ritter et al., 2020
<https://arxiv.org/abs/2006.03662>`_), and Emergent Symbols through Binding Networks (ESBN; `Webb et al., 2021
<https://arxiv.org/abs/2012.14601>`_).

There are three primary methods in the script:

* construct_model(args):
  takes as arguments parameters used to construct the model;  for convenience, defaults are defined below,
  (under "Construction parameters")

* train_network(args)
  takes as arguments the feedforward neural network Composition (FFN_COMPOSITION) and number of epochs to train.
  Note: learning_rate is set at construction (can specify using LEARNING_RATE under "Training parameters" below).

* run_model()
  takes the context drift rate to be applied on each trial and the number of trials to execute as args, as well as
  reporting and animation specifications (see "Execution parameters" below).

See "Settings for running the script" to specify whether the model is trained and/or executed when the script is run,
and whether a graphic display of the network is generated when it is constructed.

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
          - put N-back script (with pointer to latest version on PNL) in nback-paper repo
    - fix: get rid of objective_mechanism (see "VERSION *WITHOUT* ObjectiveMechanism" under control(...)
    - fix: warnings on run
    - complete documentation in BeukersNbackModel.rst
    - validate against nback-paper results
    - after validation:
        - try with STIM_SIZE = NUM_STIMS rather than 20 (as in nback-paper)
        - refactor generate_stim_sequence() to use actual empirical stimulus sequences
        - replace get_input_sequence and get_training_inputs with generators passed to nback_model.run() and ffn.learn

"""
import enum
import numpy as np

from graph_scheduler import *

from psyneulink import *

# Settings for running script:
TRAIN = False
RUN = True
DISPLAY_MODEL = False # show visual graphic of model

# PARAMETERS -------------------------------------------------------------------------------------------------------

# Fixed (structural) parameters:
MAX_NBACK_LEVELS = 3
NUM_STIM = 8 # number of different stimuli in stimulus set -  QUESTION: WHY ISN"T THIS EQUAL TO STIM_SIZE OR VICE VERSA?
FFN_TRANSFER_FUNCTION = ReLU

# Constructor parameters:  (values are from nback-paper)
STIM_SIZE=8 # length of stimulus vector
CONTEXT_SIZE=25 # length of context vector
HIDDEN_SIZE=STIM_SIZE*4 # dimension of hidden units in ff
NBACK_LEVELS = [2,3] # Currently restricted to these
NUM_NBACK_LEVELS = len(NBACK_LEVELS)
CONTEXT_DRIFT_NOISE=0.0  # noise used by DriftOnASphereIntegrator (function of Context mech)
RANDOM_WEIGHTS_INITIALIZATION=RandomMatrix(center=0.0, range=0.1)  # Matrix spec used to initialize all Projections
RETRIEVAL_SOFTMAX_TEMP=1/8 # express as gain # precision of retrieval process
RETRIEVAL_HAZARD_RATE=0.04 # rate of re=sampling of em following non-match determination in a pass through ffn
RETRIEVAL_STIM_WEIGHT=.05 # weighting of stimulus field in retrieval from em
RETRIEVAL_CONTEXT_WEIGHT = 1-RETRIEVAL_STIM_WEIGHT # weighting of context field in retrieval from em
DECISION_SOFTMAX_TEMP=1

# Training parameters:
NUM_EPOCHS= 6250    # nback-paper: 400,000 @ one trial per epoch = 6,250 @ 64 trials per epoch
LEARNING_RATE=0.01  # nback-paper: .001

# Execution parameters:
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
NUM_TRIALS = 48 # number of stimuli presented in a trial sequence
REPORT_OUTPUT = ReportOutput.OFF   # Sets console output during run
REPORT_PROGRESS = ReportProgress.OFF  # Sets console progress bar during run
REPORT_LEARNING = ReportLearning.OFF  # Sets console progress bar during training
ANIMATE = False # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

# Names of Compositions and Mechanisms:
NBACK_MODEL = "N-back Model"
FFN_COMPOSITION = "WORKING MEMORY (fnn)"
FFN_STIMULUS_INPUT = "CURRENT STIMULUS"
FFN_CONTEXT_INPUT = "CURRENT CONTEXT"
FFN_STIMULUS_RETRIEVED = "RETRIEVED STIMULUS"
FFN_CONTEXT_RETRIEVED = "RETRIEVED CONTEXT"
FFN_TASK = "CURRENT TASK"
FFN_HIDDEN = "HIDDEN LAYER"
FFN_OUTPUT = "DECISION LAYER"
MODEL_STIMULUS_INPUT ='STIM'
MODEL_CONTEXT_INPUT = 'CONTEXT'
MODEL_TASK_INPUT = "TASK"
EM = "EPISODIC MEMORY (dict)"
CONTROLLER = "READ/WRITE CONTROLLER"

# ======================================== MODEL CONSTRUCTION =========================================================

def construct_model(stim_size = STIM_SIZE,
                    context_size = CONTEXT_SIZE,
                    hidden_size = HIDDEN_SIZE,
                    num_nback_levels = NUM_NBACK_LEVELS,
                    context_drift_noise = CONTEXT_DRIFT_NOISE,
                    retrievel_softmax_temp = RETRIEVAL_SOFTMAX_TEMP,
                    retrieval_hazard_rate = RETRIEVAL_HAZARD_RATE,
                    retrieval_stimulus_weight = RETRIEVAL_STIM_WEIGHT,
                    retrieval_context_weight = RETRIEVAL_CONTEXT_WEIGHT,
                    decision_softmax_temp = DECISION_SOFTMAX_TEMP):
    """Construct nback_model"""

    print(f"constructing '{FFN_COMPOSITION}'...")

    # FEED FORWARD NETWORK -----------------------------------------

    #     inputs: encoding of current stimulus and context, retrieved stimulus and retrieved context,
    #     output: decision: match [1,0] or non-match [0,1]
    # Must be trained to detect match for specified task (1-back, 2-back, etc.)
    input_current_stim = TransferMechanism(name=FFN_STIMULUS_INPUT,
                                           size=stim_size,
                                           function=FFN_TRANSFER_FUNCTION)
    input_current_context = TransferMechanism(name=FFN_CONTEXT_INPUT,
                                              size=context_size,
                                              function=FFN_TRANSFER_FUNCTION)
    input_retrieved_stim = TransferMechanism(name=FFN_STIMULUS_RETRIEVED,
                                             size=stim_size,
                                             function=FFN_TRANSFER_FUNCTION)
    input_retrieved_context = TransferMechanism(name=FFN_CONTEXT_RETRIEVED,
                                                size=context_size,
                                                function=FFN_TRANSFER_FUNCTION)
    input_task = TransferMechanism(name=FFN_TASK,
                                   size=num_nback_levels,
                                   function=FFN_TRANSFER_FUNCTION)
    hidden = TransferMechanism(name=FFN_HIDDEN,
                               size=hidden_size,
                               function=FFN_TRANSFER_FUNCTION)
    decision = ProcessingMechanism(name=FFN_OUTPUT,
                                   size=2, function=SoftMax(output=MAX_INDICATOR,
                                                            gain=decision_softmax_temp))
    ffn = AutodiffComposition(([{input_current_stim,
                                input_current_context,
                                input_retrieved_stim,
                                input_retrieved_context,
                                input_task},
                               hidden, decision],
                              RANDOM_WEIGHTS_INITIALIZATION,
                               ),
                              name=FFN_COMPOSITION,
                              learning_rate=LEARNING_RATE
                              )

    # FULL MODEL (Outer Composition, including input, EM and control Mechanisms) ------------------------

    print(f"'constructing {NBACK_MODEL}'...")

    # Stimulus Encoding: takes STIM_SIZE vector as input
    stim = TransferMechanism(name=MODEL_STIMULUS_INPUT, size=stim_size)

    # Context Encoding: takes scalar as drift step for current trial
    context = ProcessingMechanism(name=MODEL_CONTEXT_INPUT,
                                  function=DriftOnASphereIntegrator(
                                      initializer=np.random.random(context_size-1),
                                      noise=context_drift_noise,
                                      dimension=context_size))

    # Task: task one-hot indicating n-back (1, 2, 3 etc.) - must correspond to what ffn has been trained to do
    task = ProcessingMechanism(name=MODEL_TASK_INPUT,
                               size=num_nback_levels)

    # Episodic Memory:
    #    - entries: stimulus (field[0]) and context (field[1]); randomly initialized
    #    - uses Softmax to retrieve best matching input, subject to weighting of stimulus and context by STIM_WEIGHT
    em = EpisodicMemoryMechanism(name=EM,
                                 input_ports=[{NAME:"STIMULUS_FIELD",
                                               SIZE:stim_size},
                                              {NAME:"CONTEXT_FIELD",
                                               SIZE:context_size}],
                                 function=ContentAddressableMemory(
                                     initializer=[[[0]*stim_size, [0]*context_size]],
                                     distance_field_weights=[retrieval_stimulus_weight,
                                                             retrieval_context_weight],
                                     # equidistant_entries_select=NEWEST,
                                     selection_function=SoftMax(output=MAX_INDICATOR,
                                                                gain=retrievel_softmax_temp)),
                                 )

    # Control Mechanism
    #  Ensures current stimulus and context are only encoded in EM once (at beginning of trial)
    #    by controlling the storage_prob parameter of em:
    #      - if outcome of decision signifies a match or hazard rate is realized:
    #        - set  EM[store_prob]=1 (as prep encoding stimulus in EM on next trial)
    #        - this also serves to terminate trial (see nback_model.termination_processing condition)
    #      - if outcome of decision signifies a non-match
    #        - set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
    #        - continue trial
    control = ControlMechanism(name=CONTROLLER,
                               default_variable=[[1]],  # Ensure EM[store_prob]=1 at beginning of first trial
                               # ---------
                               # VERSION *WITH* ObjectiveMechanism:
                               objective_mechanism=ObjectiveMechanism(name="OBJECTIVE MECHANISM",
                                                                      monitor=decision,
                                                                      # Outcome=1 if match, else 0
                                                                      function=lambda x: int(x[0][1]>x[0][0])),
                               # Set ControlSignal for EM[store_prob]
                               function=lambda outcome: int(bool(outcome)
                                                            or (np.random.random() > retrieval_hazard_rate)),
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
                              nodes=[stim, context, task, ffn, em, control],
                              # Terminate trial if value of control is still 1 after first pass through execution
                              termination_processing={TimeScale.TRIAL: And(Condition(lambda: control.value),
                                                                           AfterPass(0, TimeScale.TRIAL))},
                              )
    # # Terminate trial if value of control is still 1 after first pass through execution
    nback_model.add_projection(MappingProjection(), stim, input_current_stim)
    nback_model.add_projection(MappingProjection(), context, input_current_context)
    nback_model.add_projection(MappingProjection(), task, input_task)
    nback_model.add_projection(MappingProjection(), em.output_ports["RETRIEVED_STIMULUS_FIELD"], input_retrieved_stim)
    nback_model.add_projection(MappingProjection(), em.output_ports["RETRIEVED_CONTEXT_FIELD"], input_retrieved_context)
    nback_model.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
    nback_model.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])

    if DISPLAY_MODEL:
        nback_model.show_graph(
            # show_cim=True,
            # show_node_structure=ALL,
            # show_dimensions=True
        )

    print(f'full model constructed')
    return nback_model

# ==========================================STIMULUS GENERATION =======================================================



def get_stim_set(num_stim=STIM_SIZE):
    """Construct an array of unique stimuli for use in an experiment, used by train_network() and run_model()"""
    # For now, use one-hots
    return np.eye(num_stim)

def get_task_input(nback_level):
    """Construct input to task Mechanism for a given nback_level, used by train_network() and run_model()"""
    task_input = list(np.zeros_like(NBACK_LEVELS))
    task_input[nback_level-NBACK_LEVELS[0]] = 1
    return task_input

def get_training_inputs(network, num_epochs, nback_levels):
    """Construct set of training stimuli used by ffn.learn() in train_network()
    Construct one example of each condition:
     match:  stim_current = stim_retrieved  and context_current = context_retrieved
     stim_lure:  stim_current = stim_retrieved  and context_current != context_retrieved
     context_lure:  stim_current != stim_retrieved  and context_current == context_retrieved
     non_lure:  stim_current != stim_retrieved  and context_current != context_retrieved
    """
    assert is_iterable(nback_levels) and all([0<i<=MAX_NBACK_LEVELS for i in nback_levels])
    stimuli = get_stim_set()
    context_fct =  DriftOnASphereIntegrator(initializer=np.random.random(CONTEXT_SIZE-1),
                                            noise=CONTEXT_DRIFT_NOISE,
                                            dimension=CONTEXT_SIZE)
    contexts = []
    trial_types = ['match', 'stim_lure', 'context_lure', 'non_lure']

    stim_current = []
    context_current = []
    stim_retrieved = []
    context_retrieved = []
    target = []
    num_nback_levels = len(nback_levels)
    current_task = []

    # for i in range(num_epochs):
    for nback_level in nback_levels:
        # Construct one hot encoding for nback level
        # task_input = list(np.zeros(num_nback_levels))
        # task_input[nback_level-nback_levels[0]] = 1
        task_input = get_task_input(nback_level)
        for i in range(len(stimuli)):
            # Get current stimulus and distractor
            stims = list(stimuli.copy())
            # Get stim, and remove from stims so distractor can be picked randomly from remaining ones
            current_stim = stims.pop(i)
            # Pick distractor randomly from stimuli remaining in set
            distractor_stim = stims[np.random.randint(0,len(stims))]

            # Get current context, nback context, and distractor
            # Get nback+1 contexts (to bracket correct one)
            for i in range(num_nback_levels+1):
                contexts.append(context_fct(CONTEXT_DRIFT_RATE))
            # Get current context as one that is next to last from list (leaving last one as potential lure)
            current_context = contexts.pop(num_nback_levels-1)
            #
            nback_context = contexts.pop(0)
            distractor_context = contexts[np.random.randint(0,len(contexts))]

            # Assign retrieved stimulus and context accordingly to trial_type
            for trial_type in trial_types:
                stim_current.append(current_stim)
                context_current.append(current_context)
                # Assign retrieved stimulus
                if trial_type in {'match','stim_lure'}:
                    stim_retrieved.append(current_stim)
                else: # context_lure or non_lure
                    stim_retrieved.append(distractor_stim)
                # Assign retrieved context
                if trial_type in {'match','context_lure'}:
                    context_retrieved.append(nback_context)
                else: # stimulus_lure or non_lure
                    context_retrieved.append(distractor_context)
                # Assign target
                if trial_type == 'match':
                    target.append([1,0])
                else:
                    target.append([0,1])
                current_task.append([task_input])

    batch_size = len(target)
    training_set = {INPUTS: {network.nodes[FFN_STIMULUS_INPUT]: stim_current,
                             network.nodes[FFN_CONTEXT_INPUT]: context_current,
                             network.nodes[FFN_STIMULUS_RETRIEVED]: stim_retrieved,
                             network.nodes[FFN_CONTEXT_RETRIEVED]: context_retrieved,
                             network.nodes[FFN_TASK]: current_task},
                    TARGETS: {network.nodes[FFN_OUTPUT]:  target},
                    EPOCHS: num_epochs*batch_size}

    return training_set, batch_size




# # BASED ON nback-paper
# def get_run_inputs(model, nback_level, context_drift_rate, num_trials):
#     """Construct set of stimulus inputs for run_model()
#     Pad sequence of trials with random stimuli up to the last nback_level+1 trials.
#     Assign condition-specific sequence to the last nback_level_1 trials.
#     NOTE: THIS IS CARRYOVER FROM THE nback-paper IMPLEMENTATION, AND SHOULD BE REPLACED WITH
#           A SEQUENCE THAT BALANCES THE FOUR CONDITIONS THROUGH THE FULL LENGTH OF THE RUN.
#     """
#
#     def generate_stim_sequence(nback_level, trial_num, trial_type=0, num_stim=NUM_STIM, num_trials=NUM_TRIALS):
#         assert nback_level in {2,3} # At present, only 2- and 3-back levels are supported
#
#         def gen_subseq_stim():
#             A = np.random.randint(0,num_stim)
#             B = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A])
#                 )
#             C = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A,B])
#                 )
#             X = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A,B])
#                 )
#             return A,B,C,X
#
#         def generate_match_no_foils_sequence(nback_level,trial_num):
#             # AXA (2-back) or ABXA (3-back)
#             # Pad sequence of trials up to the last nback_level+1 trials with random stimuli
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             # Generate nback_level+1 stimuli for this condition
#             if nback_level==2:
#                 subseq = [A,X,A]
#             elif nback_level==3:
#                 subseq = [A,B,X,A]
#             # Assign sequence for condition to last nback_level+1 trials
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_non_match_no_foils_sequence(nback_level,trial_num):
#             # AXB (2-back) or ABXC (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [A,X,B]
#             elif nback_level==3:
#                 subseq = [A,B,X,C]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_match_with_foil_sequence(nback_level,trial_num):
#             # AAA (2-back) or AAXA (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [A,A,A]
#             elif nback_level==3:
#                 subseq = [A,A,X,A]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_non_match_with_foil_sequence(nback_level,trial_num):
#             # XAA (2-back) or ABXB (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [X,A,A]
#             elif nback_level==3:
#                 subseq = [A,B,X,B]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         trial_types = [generate_match_no_foils_sequence,
#                        generate_match_with_foil_sequence,
#                        generate_non_match_no_foils_sequence,
#                        generate_non_match_with_foil_sequence]
#         stim_seq = trial_types[trial_type](nback_level,trial_num)
#         # ytarget = [1,1,0,0][trial_type]
#         # ctxt = spherical_drift(trial_num)
#         # return stim,ctxt,ytarget
#         return stim_seq
#
#     # def stim_set_generation(nback_level, num_trials):
#     #     stim_sequence = []
#     #     # for seq_int, trial in itertools.product(range(4),np.arange(5,trials)): # This generates all length sequences
#     #     for trial_type, trial_num in itertools.product(range(4),[num_trials]):  # This generates only longest seq (
#     #         # num_trials)
#     #         return stim_sequence.append(generate_stim_sequence(nback_level, trial_num, trial_type=trial_type, trials=num_trials))
#
#     def get_input_sequence(nback_level, num_trials=NUM_TRIALS):
#         """Get sequence of inputs for a run"""
#         input_set = get_stim_set()
#         # Construct sequence of stimulus indices
#         trial_seq = generate_stim_sequence(nback_level, num_trials)
#         # Return list of corresponding stimulus input vectors
#         return [input_set[trial_seq[i]] for i in range(num_trials)]
#
#     return {model.nodes[MODEL_STIMULUS_INPUT]: get_input_sequence(nback_level, num_trials),
#             model.nodes[MODEL_CONTEXT_INPUT]: [[context_drift_rate]]*num_trials,
#             model.nodes[MODEL_TASK_INPUT]: [get_task_input(nback_level)]*num_trials}





# BALANCED / SEQUENTIAL
def get_run_inputs(model, nback_level,
                   context_drift_rate=CONTEXT_DRIFT_RATE,
                   num_stim=NUM_STIM,
                   num_trials=NUM_TRIALS,
                   mini_blocks=True):
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
    """

    import random
    from enum import IntEnum
    assert nback_level in {2,3} # At present, only 2- and 3-back levels are supported

    stim_set = get_stim_set()
    class trial_types(IntEnum):
        """Trial types explicitly assigned and counter-balanced
        In notation below, "A" is always current stimulus.
        Foils are only explicitly assigned to items immediately following nback item.
        Subseq designated below as "not explicitly assigned" may still appear in the overall stimulus seq,
            either within the subseq through random assignment,
            and/or through cross-subseq relationships that are not controlled in this design
        """
        MATCH_NO_FOIL = 0       # ABA (2-back) or ABCA (3-back); not explicitly assigned: ABBA
        MATCH_WITH_FOIL = 1     # AAA (2-back) or AABA (3-back); not explicitly assigned: ABAA or AAAA
        NO_MATCH_NO_FOIL = 2    # ABB (2-back) or BCDA (3-back); not explicitly assigned: BBCA, BCCA or BBBA
        NO_MATCH_WITH_FOIL = 3  # BAA (2-back) or BACA (3-back); not explicitly assigned: BCAA or BAAA
    num_trial_types = len(trial_types)

    def get_stim_subseq_for_trial_type(trial_type):
        """Return stimulus seq (as indices into stim_set) for the specified trial_type."""
        subseq_size = nback_level+1
        subseq = [None] * subseq_size
        curr_stim = subseq[nback_level] = random.choice(np.arange(len(stim_set)))
        other_stims = np.setdiff1d(np.arange(len(stim_set)),curr_stim).tolist()

        if trial_type == trial_types.MATCH_NO_FOIL:           # ABA (2-back) or ABCA (3-back)
            subseq[0] = curr_stim  # Assign nback stim to match
            # Assign remaining items in sequence to anything stimuli than curr_stim
            subseq[1:nback_level] = random.sample(other_stims, nback_level-1)
        elif trial_type == trial_types.MATCH_WITH_FOIL:        # AAA (2-back) or AABA (3-back)
            subseq[0] = curr_stim  # Assign nback stim to match current stim
            subseq[1] = curr_stim  # Assign curr_stim to stim next to nback as foil
            # Assign any remaining items in sequence to any stimuli other than curr_stim
            subseq[2:nback_level] = random.sample(other_stims, nback_level-2)
        elif trial_type == trial_types.NO_MATCH_NO_FOIL:       # ABB (2-back) or BCDA (3-back)
            # Assign remaining items in sequence to any stimuli than curr_stim
            subseq[0:nback_level] = random.sample(other_stims, nback_level)
        elif trial_type == trial_types.NO_MATCH_WITH_FOIL:     # BAA (2-back) or BACA (3-back)
            # Assign remaining items in sequence to any stimuli than curr_stim
            subseq[1] = curr_stim  # Assign curr_stim to stim next to nback as foil
            subseq[0:1] = random.sample(other_stims, 1)
            subseq[2:nback_level] = random.sample(other_stims, nback_level-2)
        assert not None in subseq, "Failed to assign all stims for subseq in get_stim_subseq_for_trial_type."
        return subseq

    def get_trial_type_for_stim(subseq):
        # assert len(subseq) == nback_level+1, \
        #     f"Bad subseq len ({len(subseq)}) for nback_level ({nback_level})."
        # assert all(trial_type==None for trial_type in trial_type_seq[trial_num:trial_num + nback_level]), \
        #     f"trial_type should still be None for trials {trial_num - nback_level} to {trial_num - 1}."
        if subseq[-1] == subseq[0] and not subseq[-1] in subseq[1:-1]:
            return trial_types.MATCH_NO_FOIL.value
        elif subseq[-1] == subseq[0] and subseq[-1] in subseq[0:-1]:
            return trial_types.MATCH_WITH_FOIL.value
        elif subseq[-1] not in subseq[0:-1]:
            return trial_types.NO_MATCH_NO_FOIL.value
        elif subseq[-1] != subseq[0] and subseq[-1] in subseq[0:-1]:
            # Note: for 3back, this includes: BAXA, BXAA, and BAAA
            return trial_types.NO_MATCH_WITH_FOIL.value

    subseq_size = nback_level+1
    num_sub_seqs = int(num_trials / num_trial_types)
    extra_trials = num_trials % num_trial_types

    # Construct seq of mini-blocks (subseqs) each containing one sample of each trial_type in a random order
    #    note: this is done over number of mini-blocks that fit into num_trials;
    #          remaining trials are randomly assigned trial_types below

    num_mini_blocks = int(num_trials / (num_trial_types * (nback_level+1)))
    mini_block_size = subseq_size * num_trial_types # Number of trials in a mini_block
    # seq_of_trial_type_subseqs = [None] * num_mini_blocks * num_trial_types
    seq_of_trial_type_subseqs = []
    # Generate randomly ordered trial_type assignments for subseqs in each mini_block
    for i in range(num_mini_blocks):
        # seq_of_trial_type_subseqs[i*num_trial_types:i+num_trial_types] = \
        #     random.sample(range(num_trial_types), num_trial_types)
        seq_of_trial_type_subseqs.extend(random.sample(range(num_trial_types), num_trial_types))
    # seq_of_trial_type_subseqs = random.sample(range(num_trial_types), num_trial_types) * mini_block_size
    if not mini_blocks:
        # Randomize the order of trial types across the entire sequence:
        random.shuffle(seq_of_trial_type_subseqs)

    stim_seq = [None] * num_trials
    trial_type_seq = [None] * num_trials
    # Construct actual stimulus sequence by getting stimuli for each subseq, up to num_sub_seqs
    #   note: the trial type only applies (and a trial_type is only assigned) to the last trial of each subsequence;
    #         trial_type of preceding ones set below on the full sequence of stimuli is assigned
    # stim_seq.append(get_stim_seq_for_trial_type(i) for i in seq_of_trial_type_subseqs) # <- CONDENSED VERSION
    for i, trial_type in enumerate(seq_of_trial_type_subseqs):  # <- LOOP VERSION
        idx = i * subseq_size
        # Get seq of stimuli for subseq of specified trial_type
        stim_seq[idx:idx+nback_level+1] = get_stim_subseq_for_trial_type(trial_type)
        # Assign trial_type to last stim in subseq (since it was constructed specifically for that trial_type)
        trial_type_seq[idx+nback_level] = trial_type
    # Pad remainder to get to num_trials with randomly selected stimuli
    stim_seq.extend(random.sample(range(num_trial_types),extra_trials))
    # Infer trial_types for all remaining stimuli (which should currently be marked as None)
    for i in range(subseq_size,num_trials,subseq_size):
        for j in range(i,i+nback_level):
            assert trial_type_seq[j]==None, f"trial_type should still be None for trial {j}."
            trial_type_seq[j] = get_trial_type_for_stim(stim_seq[i-subseq_size:i])
            assert True

    # FIX: GET ACTUAL STIMULI FROM stim_set HERE

    # -----------------------------------------------
    # Construct stimulus sequence from trial_type_seq:
    # - first, pad initial nback_level-1 trials
    stim_seq = [None] * num_trials
    for i, trial_type in enumerate(trial_type_seq):

        # Pad first nback_level-1 trials with randomly selected stimuli
        if trial_type is None:
            assert i < nback_level, f"Error in construction of initial padding stimuli"
            stim_seq[i] = stim_set[random.randint(0,num_stim)]
            continue

        # # Get stimulus for trial_type
        # #  Example for preceding_stim_seq
        # #     nback_level: 2
        # #     i = 4
        # #                     nb=3 nb=2 nb=1
        # #     stim_seq: [i-4, i-3, i-2, i-1, <i unassigned>]
        # #     preceding_stim_seq:  i-2:i-1
        # stim_seq[i] = get_stim_for_trial_type(trial_type=trial_type,
        #                                       nback_preceding_stims=stim_seq[i-nback_level:i-1])
        # Get stim_subseq_for_trial_type
        stim_seq[i] = get_stim_for_trial_type(trial_type=trial_type,
                                              nback_preceding_stims=stim_seq[i-nback_level:i-1])


    # ---------------

    if (num_trials - nback_level + 1) % num_trial_types:  # Warn if conditions can't be fully balanced
        warnings.warn(f"Number of trials ({num_trials}) precludes fully balancing across all five trial types")

    # Get set of stimuli from which to draw
    input_set = get_stim_set()
    # Pad initial items in sequence with random stimuli up to nback_level-1
    stimulus_sequence = np.random.randint(0,nback_level-1)

    # Construct trials in sets of num_trial_types and assign one trial to each type (to balance conditions)
    for trial in range(nback_level-1, int(num_trials/num_trial_types)):
        trial_type_seq = random.sample(trial_types, num_trial_types)
        for trial_type in trial_type_seq:
            stimulus_sequence.append(get_trial_type(trial_type))
    # FIX: FILL OUT REMAINDER OF TRIALS RANDOMLY IF NOT DIVISIBLE BY num_trial_types







    # -----------------------------------------------------------------
    def generate_balanced_stim_sequence(nback_level, num_stim=NUM_STIM, num_trials=NUM_TRIALS):
        assert nback_level in {2,3} # At present, only 2- and 3-back levels are supported

        def gen_subseq_stim():
            A = np.random.randint(0,num_stim)
            B = np.random.choice(
                 np.setdiff1d(np.arange(num_stim),[A])
                )
            C = np.random.choice(
                 np.setdiff1d(np.arange(num_stim),[A,B])
                )
            X = np.random.choice(
                 np.setdiff1d(np.arange(num_stim),[A,B,C])
                )
            return A,B,C,X



        def generate_match_no_foils_sequence(nback_level,trial_num):
            # AXA (2-back) or ABXA (3-back)
            # Pad sequence of trials up to the last nback_level+1 trials with random stimuli
            seq = np.random.randint(0,num_stim,num_trials)
            A,B,C,X = gen_subseq_stim()
            # Generate nback_level+1 stimuli for this condition
            if nback_level==2:
                subseq = [A,X,A]
            elif nback_level==3:
                subseq = [A,B,X,A]
            # Assign sequence for condition to last nback_level+1 trials
            seq[trial_num-(nback_level+1):trial_num] = subseq
            return seq[:trial_num]

        def generate_non_match_no_foils_sequence(nback_level,trial_num):
            # AXB (2-back) or ABXC (3-back)
            seq = np.random.randint(0,num_stim,num_trials)
            A,B,C,X = gen_subseq_stim()
            #
            if nback_level==2:
                subseq = [A,X,B]
            elif nback_level==3:
                subseq = [A,B,X,C]
            seq[trial_num-(nback_level+1):trial_num] = subseq
            return seq[:trial_num]

        def generate_match_with_foil_sequence(nback_level,trial_num):
            # AAA (2-back) or AAXA (3-back)
            seq = np.random.randint(0,num_stim,num_trials)
            A,B,C,X = gen_subseq_stim()
            #
            if nback_level==2:
                subseq = [A,A,A]
            elif nback_level==3:
                subseq = [A,A,X,A]
            seq[trial_num-(nback_level+1):trial_num] = subseq
            return seq[:trial_num]

        def generate_non_match_with_foil_sequence(nback_level,trial_num):
            # XAA (2-back) or ABXB (3-back)
            seq = np.random.randint(0,num_stim,num_trials)
            A,B,C,X = gen_subseq_stim()
            #
            if nback_level==2:
                subseq = [X,A,A]
            elif nback_level==3:
                subseq = [A,B,X,B]
            seq[trial_num-(nback_level+1):trial_num] = subseq
            return seq[:trial_num]

        trial_types = [generate_match_no_foils_sequence,
                       generate_match_with_foil_sequence,
                       generate_non_match_no_foils_sequence,
                       generate_non_match_with_foil_sequence]
        stim_seq = trial_types[trial_type](nback_level,trial_num)
        # ytarget = [1,1,0,0][trial_type]
        # ctxt = spherical_drift(trial_num)
        # return stim,ctxt,ytarget
        return stim_seq

    # def stim_set_generation(nback_level, num_trials):
    #     stim_sequence = []
    #     # for seq_int, trial in itertools.product(range(4),np.arange(5,trials)): # This generates all length sequences
    #     for trial_type, trial_num in itertools.product(range(4),[num_trials]):  # This generates only longest seq (
    #         # num_trials)
    #         return stim_sequence.append(generate_stim_sequence(nback_level, trial_num, trial_type=trial_type, trials=num_trials))

    def get_input_sequence(nback_level, num_trials=NUM_TRIALS):
        """Get sequence of inputs for a run"""
        input_set = get_stim_set()
        # Construct sequence of stimulus indices
        trial_seq = generate_stim_sequence(nback_level, num_trials)
        # Return list of corresponding stimulus input vectors
        return [input_set[trial_seq[i]] for i in range(num_trials)]

    return {model.nodes[MODEL_STIMULUS_INPUT]: get_input_sequence(nback_level, num_trials),
            model.nodes[MODEL_CONTEXT_INPUT]: [[context_drift_rate]]*num_trials,
            model.nodes[MODEL_TASK_INPUT]: [get_task_input(nback_level)]*num_trials}




# BASED BLOCKED

# def FULL_get_run_inputs(model, nback_level, context_drift_rate, num_trials):
#     """Construct set of stimulus inputs for run_model()
#     Pad sequence of trials with random stimuli up to the last nback_level+1 trials.
#     Assign condition-specific sequence to the last nback_level_1 trials.
#     NOTE: THIS IS CARRYOVER FROM THE nback-paper IMPLEMENTATION, AND SHOULD BE REPLACED WITH
#           A SEQUENCE THAT BALANCES THE FOUR CONDITIONS THROUGH THE FULL LENGTH OF THE RUN.
#     """
#
#     def generate_stim_sequence(nback_level, trial_num, trial_type=0, num_stim=NUM_STIM, num_trials=NUM_TRIALS):
#         assert nback_level in {2,3} # At present, only 2- and 3-back levels are supported
#
#         def gen_subseq_stim():
#             A = np.random.randint(0,num_stim)
#             B = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A])
#                 )
#             C = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A,B])
#                 )
#             X = np.random.choice(
#                  np.setdiff1d(np.arange(num_stim),[A,B])
#                 )
#             return A,B,C,X
#
#         def generate_match_no_foils_sequence(nback_level,trial_num):
#             # AXA (2-back) or ABXA (3-back)
#             # Pad sequence of trials up to the last nback_level+1 trials with random stimuli
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             # Generate nback_level+1 stimuli for this condition
#             if nback_level==2:
#                 subseq = [A,X,A]
#             elif nback_level==3:
#                 subseq = [A,B,X,A]
#             # Assign sequence for condition to last nback_level+1 trials
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_non_match_no_foils_sequence(nback_level,trial_num):
#             # AXB (2-back) or ABXC (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [A,X,B]
#             elif nback_level==3:
#                 subseq = [A,B,X,C]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_match_with_foil_sequence(nback_level,trial_num):
#             # AAA (2-back) or AAXA (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [A,A,A]
#             elif nback_level==3:
#                 subseq = [A,A,X,A]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         def generate_non_match_with_foil_sequence(nback_level,trial_num):
#             # XAA (2-back) or ABXB (3-back)
#             seq = np.random.randint(0,num_stim,num_trials)
#             A,B,C,X = gen_subseq_stim()
#             #
#             if nback_level==2:
#                 subseq = [X,A,A]
#             elif nback_level==3:
#                 subseq = [A,B,X,B]
#             seq[trial_num-(nback_level+1):trial_num] = subseq
#             return seq[:trial_num]
#
#         trial_types = [generate_match_no_foils_sequence,
#                        generate_match_with_foil_sequence,
#                        generate_non_match_no_foils_sequence,
#                        generate_non_match_with_foil_sequence]
#
#         # FIX: LOOP THROUGH GETTING SETS OF RANDOMLY ORDERED ONE OF EACH TRIAL_TYPE FOR NUM_TRIALS/4
#
#
#         stim_seq = trial_types[trial_type](nback_level,trial_num)
#         # ytarget = [1,1,0,0][trial_type]
#         # ctxt = spherical_drift(trial_num)
#         # return stim,ctxt,ytarget
#         return stim_seq
#
#     # def stim_set_generation(nback_level, num_trials):
#     #     stim_sequence = []
#     #     # for seq_int, trial in itertools.product(range(4),np.arange(5,trials)): # This generates all length sequences
#     #     for trial_type, trial_num in itertools.product(range(4),[num_trials]):  # This generates only longest seq (
#     #         # num_trials)
#     #         return stim_sequence.append(generate_stim_sequence(nback_level, trial_num, trial_type=trial_type, trials=num_trials))
#
#     def get_input_sequence(nback_level, num_trials=NUM_TRIALS):
#         """Get sequence of inputs for a run"""
#         input_set = get_stim_set()
#         # Construct sequence of stimulus indices
#         trial_seq = generate_stim_sequence(nback_level, num_trials)
#         # Return list of corresponding stimulus input vectors
#         return [input_set[trial_seq[i]] for i in range(num_trials)]
#
#     return {model.nodes[MODEL_STIMULUS_INPUT]: get_input_sequence(nback_level, num_trials),
#             model.nodes[MODEL_CONTEXT_INPUT]: [[context_drift_rate]]*num_trials,
#             model.nodes[MODEL_TASK_INPUT]: [get_task_input(nback_level)]*num_trials}












# ======================================== MODEL EXECUTION ============================================================


def train_network(network,
                  learning_rate=LEARNING_RATE,
                  num_epochs=NUM_EPOCHS,
                  save_weights_to=None):
    print(f"constructing training set for '{network.name}'...")
    training_set, batch_size = get_training_inputs(network=network,
                                                   num_epochs=num_epochs,
                                                   nback_levels=NBACK_LEVELS)
    print(f'num training stimuli per training set (batch size): {batch_size}')
    print(f'num training sets (num_epochs): {num_epochs}')
    print(f'total num trials: {num_epochs*batch_size}')
    print(f"\ntraining '{network.name}'...")
    import timeit
    start_time = timeit.default_timer()
    network.learn(inputs=training_set,
                  minibatch_size=batch_size,
                  report_progress=REPORT_PROGRESS,
                  # report_learning=REPORT_LEARNING,
                  learning_rate=learning_rate,
                  execution_mode=ExecutionMode.LLVMRun)
    stop_time = timeit.default_timer()
    print(f"'{network.name}' trained")
    training_time = stop_time-start_time
    if training_time <= 60:
        training_time_str = f'{int(training_time)} seconds'
    else:
        training_time_str = f'{int(training_time/60)} minutes'
    print(f'training time: {training_time_str} for {num_epochs} epochs')
    path = network.save(filename=save_weights_to)
    print(f'saved weights to: {save_weights_to}')
    return path
    # print(f'saved weights sample: {network.nodes[FFN_HIDDEN].path_afferents[0].matrix.base[0][:3]}...')
    # network.load(path)
    # print(f'loaded weights sample: {network.nodes[FFN_HIDDEN].path_afferents[0].matrix.base[0][:3]}...')

def run_model(model,
              load_weights_from=None,
              context_drift_rate=CONTEXT_DRIFT_RATE,
              num_trials=NUM_TRIALS,
              report_output=REPORT_OUTPUT,
              report_progress=REPORT_PROGRESS,
              animate=ANIMATE,
              save_results_to=None
              ):
    ffn = nback_model.nodes[FFN_COMPOSITION]
    em = model.nodes[EM]
    if load_weights_from:
        print(f"nback_model loading '{FFN_COMPOSITION}' weights from {load_weights_from}...")
        ffn.load(load_weights_from)
    print('nback_model executing...')
    for nback_level in NBACK_LEVELS:
        em.function.reset(em.memory[0]) # Reset episodic memory for new task using first entry (original initializer)
        model.run(inputs=get_run_inputs(model, nback_level, context_drift_rate, num_trials),
                  report_output=report_output,
                  report_progress=report_progress,
                  animate=animate
                  )
    # print("Number of entries in EM: ", len(model.nodes[EM].memory))
    assert len(model.nodes[EM].memory) == NUM_TRIALS + 1 # extra one is for initializer
    if REPORT_PROGRESS == ReportProgress.ON:
        print('\n')
    print(f'nback_model done: {len(nback_model.results)} trials executed')
    saved_results = None
    if save_results_to:
        saved_results = np.save(save_results_to, model.results)
    print(f'results: \n{model.results}')
    return saved_results

nback_model = construct_model()
if TRAIN:
    weights_filename = f'ffn.wts_nep_{NUM_EPOCHS}_lr_{str(LEARNING_RATE).split(".")[1]}.pnl'
    saved_weights = train_network(nback_model.nodes[FFN_COMPOSITION],
                                  save_weights_to=weights_filename)
if RUN:
    results_filename = f'nback.results_nep_{NUM_EPOCHS}_lr_{str(LEARNING_RATE).split(".")[1]}.pnl'
    saved_results = run_model(nback_model,
                              # load_weights_from='ffn.wts.pnl'
                              # load_weights_from=INITIALIZER
                              save_results_to= results_filename
                              )
