import numpy as np
from psyneulink import *

# TODO:
#   Nback::
#     - figure out how to specify feedback from DDM to EM
#     - figure out how to execute EM twice:
#       > first, at beginning of trial, to retrieve item based on current stimulus & context
#             (with prob retrieval = 1, prob storage = 0)
#       > second time, at end of trial (under influence of ControlMechanism) to encode current stimulus & context
#             (with prob storage = 1;  prob of retrieval = 0)
#     - implement circular drift as function for an input mechanism
#     - BUG:  should be able to use InputPort as spec for a pathway (if there is nothing after it);  same for OutputPort (if there is nothing before it)


#region N-BACK MODEL
def n_back_model():
    stim = TransferMechanism(name='STIM')
    context = TransferMechanism(name='CONTEXT')
    em = EpisodicMemoryMechanism(name='EM', content_size=3, assoc_size=3)
    ctl = ControlMechanism()

    input_layer = TransferMechanism(name='INPUT LAYER', size=10)
    output_layer = TransferMechanism(name='OUTPUT LAYER', size=1)
    ffn = AutodiffComposition(name='FFN', pathways=[input_layer,output_layer])
    decision = DDM(name='DECISION')

    # stimulus_encoding = [stim, em.input_ports[CONTENT_INPUT]]
    # context_encoding = [context, em.input_ports[ASSOC_INPUT]]
    stimulus_encoding = [stim, em]
    context_encoding = [context, em]
    processing = [em, ffn, (decision, NodeRole.OUTPUT)]
    storage = [(decision, NodeRole.OUTPUT), (ctl, NodeRole.FEEDBACK_SENDER), em]

    comp = Composition(pathways=[context_encoding,
                                 stimulus_encoding,
                                 processing,
                                 storage])
    # comp.show_graph()
    comp.run(inputs={stim:[1,2,3], context:[4,5,6]})
    # comp.run(inputs={a:2.5}, report_output=ReportOutput.FULL)
#endregion
n_back_model()
