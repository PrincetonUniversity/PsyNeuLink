# from psyneulink.core.scheduling.condition import When
from graph_scheduler import *
from psyneulink import *
import numpy as np

# TODO:
#     - from nback-paper:
#       - get ffn weights
#       - import stimulus generation code
#       - do input layers use logistic (as suggested in figure)?
#     - retrain on full set of 1,2,3,4,5 back
#     - validate against nback-paper results
#     - DriftOnASphereIntegrator:  fix for noise=0
#     - write test that compares DriftOnASphereIntegrator with spherical_drift code in nback-paper
#     - Why is ReportOutput being written only if it executes more than once per Trial?


# CONSTRUCTION ---------------------------------------------------------------------------------------------------

# Parameters:

# FROM nback-paper:
# SDIM = 20
# indim = 2 * (CDIM + SDIM)
# hiddim = SDIM * 4
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
# 'stim_weight':0.05,
# 'smtemp':8,

# TEST:
NUM_TASKS=3
STIM_SIZE=1
CONTEXT_SIZE=25
HIDDEN_SIZE=STIM_SIZE*4
CONTEXT_DRIFT_RATE=.1
CONTEXT_DRIFT_NOISE=.00000000001
STIM_WEIGHT=.05
SOFT_MAX_TEMP=1/8 # express as gain

DISPLAY = True

# MECHANISMS:

# Outer composition (including input, EM and control Mechanisms):

# Stimulus: takes STIM_SIZE vector as input
stim = TransferMechanism(name='STIM', size=STIM_SIZE)

# Context: takes scalar as drift step for current trial
context = ProcessingMechanism(name='CONTEXT',
                              function=DriftOnASphereIntegrator(
                                  initializer=np.random.random(CONTEXT_SIZE-1),
                                  noise=CONTEXT_DRIFT_NOISE,
                                  dimension=CONTEXT_SIZE))

# Task: task one-hot indicating n-back (1, 2, 3 etc.) - must correspond to what ffn has been trained to do
task = ProcessingMechanism(name="TASK", size=NUM_TASKS)

# Episodic memory:
#    - entries: stimulus (field[0]) and context (field[1]); randomly initialized
#    - uses Softmax to retrieve best matching input, subject to weighting of stimulus and context by STIM_WEIGHT
em = EpisodicMemoryMechanism(name='EPISODIC MEMORY (dict)',
                             input_ports=[{NAME:"STIMULUS_FIELD",
                                           SIZE:STIM_SIZE},
                                          {NAME:"CONTEXT_FIELD",
                                           SIZE:CONTEXT_SIZE}],
                             function=ContentAddressableMemory(
                                 initializer=[[[0]*STIM_SIZE, [0]*CONTEXT_SIZE]],
                                 distance_field_weights=[STIM_WEIGHT, 1-STIM_WEIGHT],
                                 # equidistant_entries_select=NEWEST,
                                 selection_function=SoftMax(output=MAX_INDICATOR,
                                                            gain=SOFT_MAX_TEMP)),
                             )

# Feedforward Neural Network (WM) Composition:
#     - inputs:
#        encoding of current stimulus and context, retrieved stimulus and retrieved context,
#     - output:
#        decsion: match [1,0] or non-match [0,1]
#     - must be trained to detect match for specified task (1-back, 2-back, etc.)
input_current_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT STIMULUS") # function=Logistic)
input_current_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT CONTEXT") # function=Logistic)
input_retrieved_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED STIMULUS") # function=Logistic)
input_retrieved_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED CONTEXT")  # function=Logistic)
input_task = TransferMechanism(size=NUM_TASKS, function=Linear, name="CURRENT TASK") # function=Logistic)
hidden = TransferMechanism(size=HIDDEN_SIZE, function=Logistic, name="HIDDEN LAYER")
decision = ProcessingMechanism(size=2, name="DECISION LAYER")
ffn = Composition([{input_current_stim,
                    input_current_context,
                    input_retrieved_stim,
                    input_retrieved_context,
                    input_task},
                   hidden, decision],
                  name="WORKING MEMORY (fnn)")

# Full model (outer) Composition:

# def control_function(outcome):
#     """Evaluate response and set ControlSignal for EM[store_prob] accordingly.
#
#     outcome[0] = ffn output
#     If ffn_output signifies a MATCH:
#        set  EM[store_prob]=1 (as prep encoding stimulus in EM on next trial)
#        terminate trial
#     If ffn_output signifies a NON-MATCH:
#        set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
#        continue trial
#
#     Notes:
#     - outcome is passed as 2d array with a single 1d length 2 entry, such that output[0] = ffn output
#     - ffn output: [1,0]=MATCH, [0,1]=NON-MATCH
#     - return value is used by:
#         - control Mechanism to set ControlSignal for EM[store_prob] (per above)
#         - terminate_trial(), which is used by Condition specified as termination_processing for comp.run(),
#             to determine whether to end or continue trial
#
#     """
#     ffn_output = outcome[0]
#     if ffn_output[1] > ffn_output[0]:
#         return 1
#     else:                   # NON-MATCH:
#         return 0
#     return None


HAZARD_RATE=0.8

def control_function(outcome):
    """Evaluate response and set ControlSignal for EM[store_prob] accordingly.

    outcome[0] = ffn output
    If ffn_output signifies a MATCH:
       set  EM[store_prob]=1 (as prep encoding stimulus in EM on next trial)
       terminate trial
    If ffn_output signifies a NON-MATCH:
       set  EM[store_prob]=0 (as prep for another retrieval from EM without storage)
       continue trial

    Notes:
    - outcome is passed as 2d array with a single 1d length 2 entry, such that output[0] = ffn output
    - ffn output: [1,0]=MATCH, [0,1]=NON-MATCH
    - return value is used by:
        - control Mechanism to set ControlSignal for EM[store_prob] (per above)
        - terminate_trial(), which is used by Condition specified as termination_processing for comp.run(),
            to determine whether to end or continue trial
    """
    return int(bool(outcome) or (np.random.random() > HAZARD_RATE))

# Control Mechanism
#     - determines whether or not to end trial,
#     - ensures current stimulus and context are only encoded in EM once (at beginning of trial)
control = ControlMechanism(name="READ/WRITE CONTROLLER",
                           default_variable=[[0]],
                           objective_mechanism=ObjectiveMechanism(name="OBJECTIVE MECHANISM",
                                                                  monitor=decision,
                                                                  function=lambda x: int(x[0][1]>x[0][0])),
                           function=control_function,
                           control=(STORAGE_PROB, em))

comp = Composition(nodes=[stim, context, task, em, ffn, control],
                   name="N-Back Model")
comp.add_projection(MappingProjection(), stim, input_current_stim)
comp.add_projection(MappingProjection(), context, input_current_context)
comp.add_projection(MappingProjection(), task, input_task)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_STIMULUS_FIELD"], input_retrieved_stim)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_CONTEXT_FIELD"], input_retrieved_context)
comp.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
comp.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])
# comp.add_projection(MappingProjection(), decision, control)

if DISPLAY:
    comp.show_graph(
        # show_cim=True,
        # show_node_structure=ALL,
        # show_dimensions=True)
)

# EXECUTION ---------------------------------------------------------------------------------------------------

# # nback-paper:
# HAZARD_RATE=0.04
# NUM_TRIALS = 45

# Test:
NUM_TRIALS=20

input_dict = {stim: np.array(list(range(NUM_TRIALS))).reshape(NUM_TRIALS,1)+1,
              context:[[CONTEXT_DRIFT_RATE]]*NUM_TRIALS,
              task: np.array([[0,0,1]]*NUM_TRIALS)}

comp.run(inputs=input_dict,
         # termination_processing={TimeScale.TRIAL: Condition(lambda: terminate_trial)}, # function arg
         termination_processing={TimeScale.TRIAL: And(Condition(lambda: control.value),
                                                      AfterPass(0, TimeScale.TRIAL))}, # function arg
         report_output=ReportOutput.ON,
         )
print(len(em.memory))
# ---------------------------------------------------------------------------------------------

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
