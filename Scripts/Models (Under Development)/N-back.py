"""
This implements a model of the `N-back task <https://en.wikipedia.org/wiki/N-back#Neurobiology_of_n-back_task>`_
described in `Beukers et al. (2022) <https://psyarxiv.com/jtw5p>`_.  The model uses a simple implementation of episodic
(content-addressable) memory to store previous stimuli and the temporal context in which they occured,
and a feedforward neural network to evaluate whether the current stimulus is a match to the n'th preceding stimulus (
n-back level).

There are three primary methods:

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
    - get rid of objective_mechanism (see "VERSION *WITHOUT* ObjectiveMechanism" under control(...)
    - from nback-paper:
      - why SDIM=20 if it is a one-hot encoding (np.eye), and NSTIM=8? (i.e., SHOULDN'T NUM_STIM == STIM_SIZE)?
      - do input layers use logistic (as suggested in figure)?
    - validate against nback-paper results
    - replace get_input_sequence and get_training_inputs with generators passed to nback_model.run() and ffn.learn
    - make termination processing part of the Composition definition (fix bug)
    - fix warnings on run

"""

from graph_scheduler import *
from psyneulink import *
import numpy as np
import itertools

# Settings for running script:
TRAIN = True
RUN = False
DISPLAY = False # show visual graphic of model

# PARAMETERS -------------------------------------------------------------------------------------------------------

MAX_NBACK_LEVELS = 3

# Construction parameters:  (values are from nback-paper)
STIM_SIZE=20 # length of stimulus vector
CONTEXT_SIZE=25 # length of context vector
HIDDEN_SIZE=STIM_SIZE*4 # dimension of hidden units in ff
NBACK_LEVELS = [2,3]
NUM_NBACK_LEVELS = len(NBACK_LEVELS)
# NUM_TASKS=2 # number of different variants of n-back tasks (set sizes)
NUM_STIM = 8 # number of different stimuli in stimulus set -  QUESTION: WHY ISN"T THIS EQUAL TO STIM_SIZE OR VICE VERSA?
CONTEXT_DRIFT_NOISE=0.0  # noise used by DriftOnASphereIntegrator (function of Context mech)
SOFT_MAX_TEMP=1/8 # express as gain # precision of retrieval process
HAZARD_RATE=0.04 # rate of re=sampling of em following non-match determination in a pass through ffn
STIM_WEIGHT=.05 # weighting of stimulus field in retrieval from em
CONTEXT_WEIGHT = 1-STIM_WEIGHT # weighting of context field in retrieval from em

# Training parameters:
LEARNING_RATE=0.1
NUM_EPOCHS=1000

# Execution parameters:
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
NUM_TRIALS = 48 # number of stimuli presented in a trial sequence
REPORT_OUTPUT = ReportOutput.OFF   # Sets console output during run
REPORT_PROGRESS = ReportProgress.ON  # Sets console progress bar during run
ANIMATE = None # {UNIT:EXECUTION_SET} # Specifies whether to generate animation of execution

# Names of Compositions and Mechanisms:
NBACK_MODEL = "N-Back Model"
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
                    retrievel_softmax_temp = SOFT_MAX_TEMP,
                    retrieval_hazard_rate = HAZARD_RATE,
                    retrieval_stimulus_weight = STIM_WEIGHT,
                    context_stimulus_weight = CONTEXT_WEIGHT):
    """Construct nback_model"""

    # FEED FORWARD NETWORK -----------------------------------------

    #     inputs: encoding of current stimulus and context, retrieved stimulus and retrieved context,
    #     output: decIsion: match [1,0] or non-match [0,1]
    # Must be trained to detect match for specified task (1-back, 2-back, etc.)
    input_current_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name=FFN_STIMULUS_INPUT) # function=Logistic)
    input_current_context = TransferMechanism(size=CONTEXT_SIZE, function=Linear, name=FFN_CONTEXT_INPUT) # function=Logistic)
    input_retrieved_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name=FFN_STIMULUS_RETRIEVED) # function=Logistic)
    input_retrieved_context = TransferMechanism(size=CONTEXT_SIZE, function=Linear, name=FFN_CONTEXT_RETRIEVED)  # function=Logistic)
    input_task = TransferMechanism(size=NUM_NBACK_LEVELS, function=Linear, name=FFN_TASK) # function=Logistic)
    hidden = TransferMechanism(size=HIDDEN_SIZE, function=Logistic, name=FFN_HIDDEN)
    decision = ProcessingMechanism(size=2, name=FFN_OUTPUT)
    # TODO: THIS NEEDS TO BE REPLACED BY (OR AT LEAST TRAINED AS) AutodiffComposition
    # ffn = Composition([{input_current_stim,
    ffn = AutodiffComposition([{input_current_stim,
                                input_current_context,
                                input_retrieved_stim,
                                input_retrieved_context,
                                input_task}, RandomMatrix(center=0.0, range=0.1),
                               hidden, RandomMatrix(center=0.0, range=0.1), decision],
                              name=FFN_COMPOSITION,
                              learning_rate=LEARNING_RATE
                              )

    # FULL MODEL (Outer Composition, including input, EM and control Mechanisms) ------------------------

    # Stimulus Encoding: takes STIM_SIZE vector as input
    stim = TransferMechanism(name=MODEL_STIMULUS_INPUT, size=STIM_SIZE)

    # Context Encoding: takes scalar as drift step for current trial
    context = ProcessingMechanism(name=MODEL_CONTEXT_INPUT,
                                  function=DriftOnASphereIntegrator(
                                      initializer=np.random.random(CONTEXT_SIZE-1),
                                      noise=CONTEXT_DRIFT_NOISE,
                                      dimension=CONTEXT_SIZE))

    # Task: task one-hot indicating n-back (1, 2, 3 etc.) - must correspond to what ffn has been trained to do
    task = ProcessingMechanism(name=MODEL_TASK_INPUT, size=NUM_NBACK_LEVELS)

    # Episodic Memory:
    #    - entries: stimulus (field[0]) and context (field[1]); randomly initialized
    #    - uses Softmax to retrieve best matching input, subject to weighting of stimulus and context by STIM_WEIGHT
    em = EpisodicMemoryMechanism(name=EM,
                                 input_ports=[{NAME:"STIMULUS_FIELD",
                                               SIZE:STIM_SIZE},
                                              {NAME:"CONTEXT_FIELD",
                                               SIZE:CONTEXT_SIZE}],
                                 function=ContentAddressableMemory(
                                     initializer=[[[0]*STIM_SIZE, [0]*CONTEXT_SIZE]],
                                     distance_field_weights=[STIM_WEIGHT, CONTEXT_WEIGHT],
                                     # equidistant_entries_select=NEWEST,
                                     selection_function=SoftMax(output=MAX_INDICATOR,
                                                                gain=SOFT_MAX_TEMP)),
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
                               # # VERSION *WITH* ObjectiveMechanism:
                               objective_mechanism=ObjectiveMechanism(name="OBJECTIVE MECHANISM",
                                                                      monitor=decision,
                                                                      # Outcome=1 if match, else 0
                                                                      function=lambda x: int(x[0][1]>x[0][0])),
                               # Set ControlSignal for EM[store_prob]
                               function=lambda outcome: int(bool(outcome) or (np.random.random() > HAZARD_RATE)),
                               # # VERSION *WITHOUT* ObjectiveMechanism:
                               # monitor_for_control=decision,
                               # # Set Evaluate outcome and set ControlSignal for EM[store_prob]
                               # #   - outcome is received from decision as one hot in the form: [[match, no-match]]
                               # function=lambda outcome: int(int(outcome[0][1]>outcome[0][0])
                               #                              or (np.random.random() > HAZARD_RATE)),
                               control=(STORAGE_PROB, em))

    nback_model = Composition(nodes=[stim, context, task, em, ffn, control],
                              # # # Terminate trial if value of control is still 1 after first pass through execution
                              # # FIX: STOPS AFTER ~ NUMBER OF TRIALS (?90+); SHOULD BE: NUM_TRIALS*NUM_NBACK_LEVELS + 1
                              # termination_processing={TimeScale.TRIAL: And(Condition(lambda: control.value),
                              #                                              AfterPass(0, TimeScale.TRIAL))},
                              name=NBACK_MODEL)
    # # Terminate trial if value of control is still 1 after first pass through execution
    # # FIX: ALL OF THE FOLLOWING STOP AFTER ~ NUMBER OF TRIALS (?90+); SHOULD BE: NUM_TRIALS*NUM_NBACK_LEVELS + 1
    # nback_model.scheduler.add_condition(nback_model, And(Condition(lambda: control.value), AfterPass(0, TimeScale.TRIAL)))
    # nback_model.scheduler.termination_conds = ({TimeScale.TRIAL: And(Condition(lambda: control.value),
    #                                                                      AfterPass(0, TimeScale.TRIAL))})
    # nback_model.scheduler.termination_conds.update({TimeScale.TRIAL: And(Condition(lambda: control.value),
    #                                                                      AfterPass(0, TimeScale.TRIAL))})
    nback_model.add_projection(MappingProjection(), stim, input_current_stim)
    nback_model.add_projection(MappingProjection(), context, input_current_context)
    nback_model.add_projection(MappingProjection(), task, input_task)
    nback_model.add_projection(MappingProjection(), em.output_ports["RETRIEVED_STIMULUS_FIELD"], input_retrieved_stim)
    nback_model.add_projection(MappingProjection(), em.output_ports["RETRIEVED_CONTEXT_FIELD"], input_retrieved_context)
    nback_model.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
    nback_model.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])

    if DISPLAY:
        nback_model.show_graph(
            # show_cim=True,
            # show_node_structure=ALL,
            # show_dimensions=True)
        )

    return nback_model

# ==========================================STIMULUS GENERATION =======================================================
# Based on nback-paper

def get_stim_set(num_stim=STIM_SIZE):
    """Construct an array of stimuli for use an experiment"""
    # For now, use one-hots
    return np.eye(num_stim)

def get_task_iAnput(nback_level):
    """Construct input to task Mechanism for a given nback_level, used by run_model() and train_network()"""
    task_input = list(np.zeros_like(NBACK_LEVELS))
    task_input[nback_level-NBACK_LEVELS[0]] = 1
    return task_input

def get_run_inputs(model, nback_level, context_drift_rate, num_trials):
    """Construct set of stimulus inputs for run_model()"""

    def generate_stim_sequence(nback_level, trial_num, trial_type=0, num_stim=NUM_STIM, num_trials=NUM_TRIALS):
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
                 np.setdiff1d(np.arange(num_stim),[A,B])
                )
            return A,B,C,X

        def generate_match_no_foils_sequence(nback_level,trial_num):
            # AXA (2-back) or ABXA (3-back)
            seq = np.random.randint(0,num_stim,num_trials)
            A,B,C,X = gen_subseq_stim()
            #
            if nback_level==2:
                subseq = [A,X,A]
            elif nback_level==3:
                subseq = [A,B,X,A]
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

def get_training_inputs(network, num_epochs, nback_levels):
    """Construct set of training stimuli used by ffn.learn() in train_network()
    Construct one example of each condition:
     match:  stim_current = stim_retrieved  and context_current = context_retrieved
     stim_lure:  stim_current = stim_retrieved  and context_current != context_retrieved
     context_lure:  stim_current != stim_retrieved  and context_current == context_retrieved
     non_lure:  stim_current != stim_retrieved  and context_current != context_retrieved
    """
    assert is_iterable(nback_levels) and all([0<i<MAX_NBACK_LEVELS for i in nback_levels])
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

    for i in range(num_epochs):
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

    training_set = {INPUTS: {network.nodes[FFN_STIMULUS_INPUT]: stim_current,
                             network.nodes[FFN_CONTEXT_INPUT]: context_current,
                             network.nodes[FFN_STIMULUS_RETRIEVED]: stim_retrieved,
                             network.nodes[FFN_CONTEXT_RETRIEVED]: context_retrieved,
                             network.nodes[FFN_TASK]: current_task},
                    TARGETS: {network.nodes[FFN_OUTPUT]:  target},
                    EPOCHS: num_epochs}

    return training_set

# ======================================== MODEL EXECUTION ============================================================

def train_network(network, num_epochs=NUM_EPOCHS):
    training_set = get_training_inputs(network=network, num_epochs=num_epochs, nback_levels=NBACK_LEVELS)
    network.learn(inputs=training_set,
                  minibatch_size=NUM_TRIALS,
                  execution_mode=ExecutionMode.LLVMRun)

def run_model(model,
              context_drift_rate=CONTEXT_DRIFT_RATE,
              num_trials=NUM_TRIALS,
              report_output=REPORT_OUTPUT,
              report_progress=REPORT_PROGRESS,
              animate=ANIMATE):
    for nback_level in NBACK_LEVELS:
        model.run(inputs=get_run_inputs(model, nback_level, context_drift_rate, num_trials),
                  # FIX: MOVE THIS TO MODEL CONSTRUCTION ONCE THAT WORKS
                  # Terminate trial if value of control is still 1 after first pass through execution
                  termination_processing={TimeScale.TRIAL: And(Condition(lambda: model.nodes[CONTROLLER].value),
                                                               AfterPass(0, TimeScale.TRIAL))}, # function arg
                  report_output=report_output,
                  report_progress=report_progress,
                  animate=animate
                  )
        # FIX: RESET MEMORY HERE?
    # print("Number of entries in EM: ", len(model.nodes[EM].memory))
    assert len(model.nodes[EM].memory) == NUM_TRIALS*NUM_NBACK_LEVELS + 1


nback_model = construct_model()
print('nback_model constructed')
if TRAIN:
    print('nback_model training...')
    train_network(nback_model.nodes[FFN_COMPOSITION])
    print('nback_model trained')
if RUN:
    print('nback_model executing...')
    run_model(nback_model)
    if REPORT_PROGRESS == ReportProgress.ON:
        print('\n')
print(f'nback_model done: {len(nback_model.results)} trials executed')

# ===========================================================================

# TEST OF SPHERICAL DRIFT:
# stims = np.array([x[0] for x in em.memory])
# contexts = np.array([x[1] for x in em.memory])
# cos = Distance(metric=COSINE)
# dist = Distance(metric=EUCLIDEAN)
# diffs = [np.sum([contexts[i+1] - contexts[1]]) for i in range(NUM_TRIALS)]
# diffs_1 = [np.sum([contexts[i+1] - contexts[i]]) for i in range(NUM_TRIALS)]
# diffs_2 = [np.sum([contexts[i+2] - contexts[i]]) for i in range(NUM_TRIALS-1)]
# dots = [[contexts[i+1] @ contexts[1]] for i in range(NUM_TRIALS)]
# dot_diffs_1 = [[contexts[i+1] @ contexts[i]] for i in range(NUM_TRIALS)]
# dot_diffs_2 = [[contexts[i+2] @ contexts[i]] for i in range(NUM_TRIALS-1)]
# angle = [cos([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
# angle_1 = [cos([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
# angle_2 = [cos([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
# euclidean = [dist([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
# euclidean_1 = [dist([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
# euclidean_2 = [dist([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
# print("STIMS:", stims, "\n")
# print("DIFFS:", diffs, "\n")
# print("DIFFS 1:", diffs_1, "\n")
# print("DIFFS 2:", diffs_2, "\n")
# print("DOT PRODUCTS:", dots, "\n")
# print("DOT DIFFS 1:", dot_diffs_1, "\n")
# print("DOT DIFFS 2:", dot_diffs_2, "\n")
# print("ANGLE: ", angle, "\n")
# print("ANGLE_1: ", angle_1, "\n")
# print("ANGLE_2: ", angle_2, "\n")
# print("EUCILDEAN: ", euclidean, "\n")
# print("EUCILDEAN 1: ", euclidean_1, "\n")
# print("EUCILDEAN 2: ", euclidean_2, "\n")

# n_back_model()
