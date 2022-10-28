"""
TODO:
    - get rid of objective_mechanism (see "VERSION *WITHOUT* ObjectiveMechanism" under control(...)
    - from nback-paper:
      - get ffn weights?
      - why SDIM=20 if it is a one-hot encoding (np.eye), and NSTIM=8? (i.e., SHOULDN'T NUM_STIM == STIM_SIZE)?
      - do input layers use logistic (as suggested in figure)?
    - construct training set and train in ffn using Autodiff
    - validate against nback-paper results

"""

from graph_scheduler import *
from psyneulink import *
import numpy as np
import itertools

DISPLAY = False # show visual of model
REPORTING_OPTIONS = ReportOutput.ON # Console output during run


# ======================================== MODEL CONSTRUCTION =========================================================

# PARAMETERS -------------------------------------------------------------------------------------------------------

# FROM nback-paper:
# SDIM = 20
# CDIM = 25
# indim = 2 * (CDIM + SDIM)
# hiddim = SDIM * 4
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
# 'stim_weight':0.05,
# 'smtemp':8,
# HAZARD_RATE=0.04

# TEST:
NUM_TASKS=2 # number of different variants of n-back tasks (set sizes)
NUM_STIM = 8 # number of different stimuli in stimulus set -  QUESTION: WHY ISN"T THIS EQUAL TO STIM_SIZE OR VICE VERSA?
NUM_TRIALS = 48 # number of stimuli presented in a sequence
STIM_SIZE=20 # length of stimulus vector
CONTEXT_SIZE=25 # length of context vector
HIDDEN_SIZE=STIM_SIZE*4 # dimension of hidden units in ff
CONTEXT_DRIFT_RATE=.1 # drift rate used for DriftOnASphereIntegrator (function of Context mech) on each trial
CONTEXT_DRIFT_NOISE=0.0  # noise used by DriftOnASphereIntegrator (function of Context mech)
STIM_WEIGHT=.05 # weighting of stimulus field in retrieval from em
CONTEXT_WEIGHT = 1-STIM_WEIGHT # weighting of context field in retrieval from em
SOFT_MAX_TEMP=1/8 # express as gain # precision of retrieval process
HAZARD_RATE=0.04 # rate of re=sampling of em following non-match determination in a pass through ffn

# FEEDFORWARD NEURAL NETWORK COMPOSITION (WM)  --------------------------------------------------------------------
#     - inputs:
#        encoding of current stimulus and context, retrieved stimulus and retrieved context,
#     - output:
#        decIsion: match [1,0] or non-match [0,1]
#     - must be trained to detect match for specified task (1-back, 2-back, etc.)
input_current_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT STIMULUS") # function=Logistic)
input_current_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT CONTEXT") # function=Logistic)
input_retrieved_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED STIMULUS") # function=Logistic)
input_retrieved_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED CONTEXT")  # function=Logistic)
input_task = TransferMechanism(size=NUM_TASKS, function=Linear, name="CURRENT TASK") # function=Logistic)
hidden = TransferMechanism(size=HIDDEN_SIZE, function=Logistic, name="HIDDEN LAYER")
decision = ProcessingMechanism(size=2, name="DECISION LAYER")
# TODO: THIS NEEDS TO BE REPLACED BY (OR AT LEAST TRAINED AS) AutodiffComposition
#       TRAINING:
#       - 50% matches and 50% non-matches
#       - all possible stimuli
#       - 2back and 3back
#       - contexts of various distances
ffn = Composition([{input_current_stim,
                    input_current_context,
                    input_retrieved_stim,
                    input_retrieved_context,
                    input_task},
                   hidden, decision],
                  name="WORKING MEMORY (fnn)")
# ffn.learn()

# FULL MODEL (Outer Composition, including input, EM and control Mechanisms) -----------------------------------------

# Stimulus Encoding: takes STIM_SIZE vector as input
stim = TransferMechanism(name='STIM', size=STIM_SIZE)

# Context Encoding: takes scalar as drift step for current trial
context = ProcessingMechanism(name='CONTEXT',
                              function=DriftOnASphereIntegrator(
                                  initializer=np.random.random(CONTEXT_SIZE-1),
                                  noise=CONTEXT_DRIFT_NOISE,
                                  dimension=CONTEXT_SIZE))

# Task: task one-hot indicating n-back (1, 2, 3 etc.) - must correspond to what ffn has been trained to do
task = ProcessingMechanism(name="TASK", size=NUM_TASKS)

# Episodic Memory:
#    - entries: stimulus (field[0]) and context (field[1]); randomly initialized
#    - uses Softmax to retrieve best matching input, subject to weighting of stimulus and context by STIM_WEIGHT
em = EpisodicMemoryMechanism(name='EPISODIC MEMORY (dict)',
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
#        - this also serves to terminate trial (see comp.run(termination_processing condition)
#      - if outcome of decision signifies a non-match
#        - set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
#        - continue trial
control = ControlMechanism(name="READ/WRITE CONTROLLER",
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

comp = Composition(nodes=[stim, context, task, em, ffn, control], name="N-Back Model")
comp.add_projection(MappingProjection(), stim, input_current_stim)
comp.add_projection(MappingProjection(), context, input_current_context)
comp.add_projection(MappingProjection(), task, input_task)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_STIMULUS_FIELD"], input_retrieved_stim)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_CONTEXT_FIELD"], input_retrieved_context)
comp.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
comp.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])

if DISPLAY:
    comp.show_graph(
        # show_cim=True,
        # show_node_structure=ALL,
        # show_dimensions=True)
)

# ==========================================STIMULUS GENERATION =======================================================
# Based on nback-paper

def generate_stim_sequence(nback,trial, stype=0, num_stim=NUM_STIM, trials=NUM_TRIALS):

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

    def genseqCT(nback,trial):
        # ABXA / AXA
        seq = np.random.randint(0,num_stim,trials)
        A,B,C,X = gen_subseq_stim()
        #
        if nback==3:
            subseq = [A,B,X,A]
        elif nback==2:
            subseq = [A,X,A]
        seq[trial-(nback+1):trial] = subseq
        return seq[:trial]

    def genseqCF(nback,trial):
        # ABXC
        seq = np.random.randint(0,num_stim,trials)
        A,B,C,X = gen_subseq_stim()
        #
        if nback==3:
            subseq = [A,B,X,C]
        elif nback==2:
            subseq = [A,X,B]
        seq[trial-(nback+1):trial] = subseq
        return seq[:trial]

    def genseqLT(nback,trial):
        # AAXA
        seq = np.random.randint(0,num_stim,trials)
        A,B,C,X = gen_subseq_stim()
        #
        if nback==3:
            subseq = [A,A,X,A]
        elif nback==2:
            subseq = [A,A,A]
        seq[trial-(nback+1):trial] = subseq
        return seq[:trial]

    def genseqLF(nback,trial):
        # ABXB
        seq = np.random.randint(0,num_stim,trials)
        A,B,C,X = gen_subseq_stim()
        #
        if nback==3:
            subseq = [A,B,X,B]
        elif nback==2:
            subseq = [X,A,A]
        seq[trial-(nback+1):trial] = subseq
        return seq[:trial]

    genseqL = [genseqCT,genseqLT,genseqCF,genseqLF]
    stim = genseqL[stype](nback,trial)
    # ytarget = [1,1,0,0][stype]
    # ctxt = spherical_drift(trial)
    # return stim,ctxt,ytarget
    return stim

def stim_set_generation(nback,trials):
    # for seq_int,trial in itertools.product(range(4),np.arange(5,trials)): # This generates all length sequences
    stim_sequence = []
    for seq_int,trial in itertools.product(range(4),[trials]): # This generates only longest seq (45)
        return stim_sequence.append(generate_stim_sequence(nback,trial,stype=seq_int,trials=trials))

def get_input_sequence(trials):
    """Get sequence of inputs for a run"""
    # Construct array of one hot input vectors as inputs
    input_set = np.eye(20)
    # Construct sequence of stimulus indices
    trial_seq = generate_stim_sequence(2,trials)
    # Return list of corresponding stimulus input vectors
    return [input_set[trial_seq[i]] for i in range(trials)]


# ==============================================EXECUTION ===========================================================

input_dict = {# stim:[[0]*STIM_SIZE]*NUM_TRIALS,
              stim:get_input_sequence(NUM_TRIALS),
              context:[[CONTEXT_DRIFT_RATE]]*NUM_TRIALS,
              task: np.array([[0]*NUM_TASKS]*NUM_TRIALS)}

comp.run(inputs=input_dict,
         # Terminate trial if value of control is still 1 after first pass through execution
         termination_processing={TimeScale.TRIAL: And(Condition(lambda: control.value),
                                                      AfterPass(0, TimeScale.TRIAL))}, # function arg
         report_output=REPORTING_OPTIONS
         )
print(len(em.memory))


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
