import numpy as np
from psyneulink import *
# from psyneulink.core.scheduling.condition import When
from graph_scheduler import *

# TODO:
#     - from nback-paper:
#       - get ffn weights
#       - import stimulus generation code
#     - retrain on full set of 1,2,3,4,5 back
#     - validate against nback-paper results
#     - DriftOnASphereIntegrator:  fix for noise=0
#     - write test that compares DriftOnASphereIntegrator with spherical_drift code in nback-paper

# FROM nback-paper:
# 'smtemp':8,
# 'stim_weight':0.05,
# 'hrate':0.04
# SDIM = 20
# indim = 2 * (CDIM + SDIM)
# hiddim = SDIM * 4

# TEST:
# Structural parameters:
NUM_TASKS=3
# Test:
STIM_SIZE=1
# Replicate model:
# STIM_SIZE=20
# ----------
CONTEXT_SIZE=25
HIDDEN_SIZE=STIM_SIZE*4

# Execution parameters
# Test:
CONTEXT_DRIFT_RATE=.1
CONTEXT_DRIFT_NOISE=.00000000001
# Replicate model:
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
# ----
NUM_TRIALS=20
NBACK=2
TOLERANCE=.5
STIM_WEIGHT=.05
HAZARD_RATE=0.04
SOFT_MAX_TEMP=1/8

# # MODEL:
# STIM_SIZE=25
# CONTEXT_SIZE=20
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
# NUM_TRIALS = 25

def control_function(outcome):
    # Assumes, for outcome:  [1,0]=MATCH, [0,1]=NON-MATCH
    # Return value is used:
    # - as value of Mechanism used for termination condition
    # - to set ControlSignal for EM[store_prob]=
    if all(outcome[0]==[0,1]): # MATCH
        return 1
    else:                   # NON-MATCH:
        return 0
    return None

# def n_back_model():

# Mechanisms:
stim = TransferMechanism(name='STIM', size=STIM_SIZE)
context = ProcessingMechanism(name='CONTEXT',
                              function=DriftOnASphereIntegrator(
                                  initializer=np.random.random(CONTEXT_SIZE-1),
                                  noise=CONTEXT_DRIFT_NOISE,
                                  dimension=CONTEXT_SIZE))
task = ProcessingMechanism(name="TASK", size=NUM_TASKS)
em = EpisodicMemoryMechanism(name='EPISODIC MEMORY (dict)',
                             # default_variable=[[0]*STIM_SIZE, [0]*CONTEXT_SIZE],
                             input_ports=[{NAME:"STIMULUS_FIELD",
                                           SIZE:STIM_SIZE},
                                          {NAME:"CONTEXT_FIELD",
                                           SIZE:CONTEXT_SIZE}],
                             function=ContentAddressableMemory(
                                 initializer=[[[0]*STIM_SIZE, [0]*CONTEXT_SIZE]],
                                 distance_field_weights=[STIM_WEIGHT, 1-STIM_WEIGHT],
                                 equidistant_entries_select=NEWEST,
                                 selection_function=SoftMax(output=MAX_INDICATOR,
                                                            gain=SOFT_MAX_TEMP)),
                             )
stim_comparator = ComparatorMechanism(name='STIM COMPARATOR',
                                      # sample=STIM_SIZE, target=STIM_SIZE
                                      input_ports=[{NAME:"CURRENT_STIMULUS", SIZE:STIM_SIZE},
                                                   {NAME:"RETRIEVED_STIMULUS", SIZE:STIM_SIZE}],
                                      )
context_comparator = ComparatorMechanism(name='CONTEXT COMPARATOR',
                                         # sample=np.zeros(STIM_SIZE),
                                         # target=np.zeros(CONTEXT_SIZE)
                                         input_ports=[{NAME:"CURRENT_CONTEXT", SIZE:CONTEXT_SIZE},
                                                      {NAME:"RETRIEVED_CONTEXT", SIZE:CONTEXT_SIZE}],
                                         function=Distance(metric=COSINE))

# QUESTION: GET INFO ABOUT INPUT FUNCTIONS FROM ANDRE:
input_current_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT STIMULUS") # function=Logistic)
input_current_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="CURRENT CONTEXT") # function=Logistic)
input_retrieved_stim = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED STIMULUS") # function=Logistic)
input_retrieved_context = TransferMechanism(size=STIM_SIZE, function=Linear, name="RETRIEVED CONTEXT")  # function=Logistic)
input_task = TransferMechanism(size=NUM_TASKS, function=Linear, name="CURRENT TASK") # function=Logistic)
hidden = TransferMechanism(size=HIDDEN_SIZE, function=Logistic, name="HIDDEN LAYER")
decision = ProcessingMechanism(size=2, name="DECISION LAYER")

control = ControlMechanism(name="READ/WRITE CONTROLLER",
                       function=control_function,
                       control=(STORAGE_PROB, em),)

# Compositions:
ffn = Composition([{input_current_stim,
                    input_current_context,
                    input_retrieved_stim,
                    input_retrieved_context,
                    input_task},
                   hidden, decision],
                  name="WORKING MEMORY (fnn)")
comp = Composition(nodes=[stim, context, task, em, ffn, control],
                   name="N-Back Model")
comp.add_projection(MappingProjection(), stim, input_current_stim)
comp.add_projection(MappingProjection(), context, input_current_context)
comp.add_projection(MappingProjection(), task, input_task)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_STIMULUS_FIELD"], input_retrieved_stim)
comp.add_projection(MappingProjection(), em.output_ports["RETRIEVED_CONTEXT_FIELD"], input_retrieved_context)
comp.add_projection(MappingProjection(), stim, em.input_ports["STIMULUS_FIELD"])
comp.add_projection(MappingProjection(), context, em.input_ports["CONTEXT_FIELD"])
comp.add_projection(MappingProjection(), decision, control)

comp.show_graph()
# comp.show_graph(show_cim=True,
#                 show_node_structure=ALL,
#                 show_dimensions=True)

# Execution:

# Define a function that detects when the a Mechanism's value has converged, such that the change in all of the
# elements of its value attribute from the last execution (given by its delta attribute) falls below ``epsilon``
#
# def converge(mech, thresh):
#     return all(abs(v) <= thresh for v in mech.delta)
#
# # Add Conditions to the ``color_hidden`` and ``word_hidden`` Mechanisms that depend on the converge function:
# epsilon = 0.01
# Stroop_model.scheduler.add_condition(color_hidden, When(converge, task, epsilon)))
# Stroop_model.scheduler.add_condition(word_hidden, When(converge, task, epsilon)))

input_dict = {#stim:[[1]*STIM_SIZE]*NUM_TRIALS,
              stim: np.array(list(range(NUM_TRIALS))).reshape(NUM_TRIALS,1)+1,
              context:[[CONTEXT_DRIFT_RATE]]*NUM_TRIALS,
              task: np.array([[0,0,1]]*NUM_TRIALS)}

def termination_trial():
    if control.value==1 or np.random.random() > HAZARD_RATE:
        return 1
    else:
        return 0

comp.run(inputs=input_dict,
         # termination_processing={TimeScale.TRIAL: terminate(control.value)},
         # termination_processing={TimeScale.TRIAL: (NWhen, terminate, control)},
         # termination_processing={TimeScale.TRIAL: termination_condition},
         termination_processing={TimeScale.TRIAL: Condition(termination_trial)},
         report_output=ReportOutput.ON
         )
assert True

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
