import numpy as np
from psyneulink import *

# TODO:
#   Nback::
#     - separate out stim/context external inputs from those from EM into FFN
#     - figure out how to specify feedback from DDM to EM:
#     - figure out how to execute EM/ffn/control:
#       > At beginning of trial:
#         - cue with stimulus as key, and retrieve value
#             (with prob storage = 0;  prob of retrieval = 1)
#         - use ffn to compare retrieved value with current context, giving yes or no answer
#         - two alternative evaluation methods:
#           - repeat per hazard rate, integrating yes/no, report answer (no=<.5; yes=>.5)
#           - repeat, sending answer as +1/-1 to DDM, continue until DDM crosses threshold
#       > At end of trial:
#         - encode current stimulus & context
#             (with prob storage = 1;  prob of retrieval = 0)
#         scheduler.add_condition(A, pnl.AfterNCalls(CM, 1))
#         scheduler.add_condition(CM, pnl.Always())
#         composition.run(...termination_conds={pnl.TimeScale.TRIAL: pnl.And(pnl.AfterNCalls(CM, 2), pnl.JustRan(CM))})
#     - Match mechanism(s):
#       > ComparatorMechanisms, one for current stim vs. retrieved stim, and other for current context vs. retrieved
#       context
#       > should compare each sample EM:  context against context, stim against stim, and yield an answer
#     - ffn:
#       > should be trained on 1,2,3,4,5 back, with cue as to what the right answer is
#     - ADD PNL FEATURE:  should be able to use InputPort as spec for a pathway (if there is nothing after it);
#             same for OutputPort (if there is nothing before it)

# TEST:
STIM_SIZE=1
CONTEXT_SIZE=25
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
CONTEXT_DRIFT_RATE=.1
CONTEXT_DRIFT_NOISE=.00000000001
NUM_TRIALS=20
NBACK = 2
TOLERANCE = .5

# # MODEL:
# STIM_SIZE=25
# CONTEXT_SIZE=20
# CONTEXT_DRIFT_RATE=.25
# CONTEXT_DRIFT_NOISE=.075
# NUM_TRIALS = 25

def context_nback_fct(outcome):
    if abs(outcome - NBACK) > TOLERANCE:
        return 1
    else:
        return 0

# def n_back_model():

# Mechanisms:
stim = TransferMechanism(name='STIM', size=STIM_SIZE)
context = ProcessingMechanism(name='CONTEXT',
                              function=DriftOnASphereIntegrator(
                                  initializer=np.random.random(CONTEXT_SIZE-1),
                                  noise=CONTEXT_DRIFT_NOISE,
                                  dimension=CONTEXT_SIZE))
em = EpisodicMemoryMechanism(name='EPISODIC MEMORY (dict)',
                             default_variable=[[0]*STIM_SIZE, [0]*CONTEXT_SIZE],
                             function=DictionaryMemory(
                                 initializer=[[[0]*STIM_SIZE, [0]*CONTEXT_SIZE]],
                                 equidistant_keys_select=NEWEST))
stim_comparator = ComparatorMechanism(name='STIM COMPARATOR', sample=STIM_SIZE, target=STIM_SIZE)
context_comparator = ComparatorMechanism(name='CONTEXT COMPARATOR',
                                         # sample=np.zeros(STIM_SIZE),
                                         # target=np.zeros(CONTEXT_SIZE)
                                         input_ports=[{NAME:"CURRENT_CONTEXT", SIZE:CONTEXT_SIZE},
                                                      {NAME:"RETRIEVED_CONTEXT", SIZE:CONTEXT_SIZE}],
                                         function=Distance(metric=COSINE))
ctl = ControlMechanism(name="READ/WRITE CONTROLLER",
                       function=context_nback_fct,
                       control=(STORAGE_PROB, em),)
# decision = DDM(name='DECISION')
decision = ProcessingMechanism(name='DECISION')

# Compositions:
ffn = Composition(stim_comparator, context_comparator, name="WORKING MEMORY (fnn)")
comp = Composition(nodes=[stim, context, ffn, em, (decision, NodeRole.OUTPUT), ctl])
comp.add_projection(MappingProjection(), stim, stim_comparator.input_ports[TARGET])
comp.add_projection(MappingProjection(), context, context_comparator.input_ports["CURRENT_CONTEXT"])
comp.add_projection(MappingProjection(), stim, em.input_ports[KEY_INPUT])
comp.add_projection(MappingProjection(), context, em.input_ports[VALUE_INPUT])
comp.add_projection(MappingProjection(), em.output_ports[KEY_OUTPUT], stim_comparator.input_ports[SAMPLE])
comp.add_projection(MappingProjection(),
                    em.output_ports[VALUE_OUTPUT],
                    context_comparator.input_ports["RETRIEVED_CONTEXT"])
comp.add_projection(MappingProjection(), context_comparator, decision)
comp.add_projection(MappingProjection(), context_comparator, ctl)
comp.show_graph()
# comp.show_graph(show_cim=True,
#                 show_node_structure=ALL,
#                 show_dimensions=True)

# Execution:
input_dict = {#stim:[[1]*STIM_SIZE]*NUM_TRIALS,
              stim: np.array(list(range(NUM_TRIALS))).reshape(NUM_TRIALS,1)+1,
              context:[[CONTEXT_DRIFT_RATE]]*NUM_TRIALS}
comp.run(inputs=input_dict,
         report_output=ReportOutput.ON
         )
assert True

# TEST OF SPHERICAL DRIFT:
stims = np.array([x[0] for x in em.memory])
contexts = np.array([x[1] for x in em.memory])
cos = Distance(metric=COSINE)
dist = Distance(metric=EUCLIDEAN)
diffs = [np.sum([contexts[i+1] - contexts[1]]) for i in range(NUM_TRIALS)]
diffs_1 = [np.sum([contexts[i+1] - contexts[i]]) for i in range(NUM_TRIALS)]
diffs_2 = [np.sum([contexts[i+2] - contexts[i]]) for i in range(NUM_TRIALS-1)]
dots = [[contexts[i+1] @ contexts[1]] for i in range(NUM_TRIALS)]
dot_diffs_1 = [[contexts[i+1] @ contexts[i]] for i in range(NUM_TRIALS)]
dot_diffs_2 = [[contexts[i+2] @ contexts[i]] for i in range(NUM_TRIALS-1)]
angle = [cos([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
angle_1 = [cos([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
angle_2 = [cos([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
euclidean = [dist([contexts[i+1], contexts[1]]) for i in range(NUM_TRIALS)]
euclidean_1 = [dist([contexts[i+1], contexts[i]]) for i in range(NUM_TRIALS)]
euclidean_2 = [dist([contexts[i+2], contexts[i]]) for i in range(NUM_TRIALS-1)]
print("STIMS:", stims, "\n")
print("DIFFS:", diffs, "\n")
print("DIFFS 1:", diffs_1, "\n")
print("DIFFS 2:", diffs_2, "\n")
print("DOT PRODUCTS:", dots, "\n")
print("DOT DIFFS 1:", dot_diffs_1, "\n")
print("DOT DIFFS 2:", dot_diffs_2, "\n")
print("ANGLE: ", angle, "\n")
print("ANGLE_1: ", angle_1, "\n")
print("ANGLE_2: ", angle_2, "\n")
print("EUCILDEAN: ", euclidean, "\n")
print("EUCILDEAN 1: ", euclidean_1, "\n")
print("EUCILDEAN 2: ", euclidean_2, "\n")

# n_back_model()
