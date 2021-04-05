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

    # Input Mechs
    stim = TransferMechanism(name='STIM', size=5)
    context = TransferMechanism(name='CONTEXT', size=5)

    # Feedforward Network:
    stim_input_layer = TransferMechanism(name='STIM INPUT LAYER', size=5)
    context_input_layer = TransferMechanism(name='CONTEXT INPUT LAYER', size=5)
    match_output_layer = TransferMechanism(name='MATCH LAYER', size=1)
    # ffn = AutodiffComposition(name='FFN', pathways=[[stim_input,match_output], [context_input, match_output]])
    ffn = Composition(name='FFN', pathways=[[stim_input_layer, match_output_layer],
                                            [context_input_layer, match_output_layer]])

    # Episodic Memory, Decision and Control
    # em = EpisodicMemoryMechanism(name='EM', content_size=5, assoc_size=5)
    em = EpisodicMemoryMechanism(name='EM', content_size=5, assoc_size=5,
                                 # function=ContentAddressableMemory(initializer=[[[0,0,0,0,0],[0,0,0,0,0]]])
                                 )
    ctl = ControlMechanism(control=(STORAGE_PROB, em))
    decision = DDM(name='DECISION')

    resp_decision = Pathway([match_output_layer, (decision, NodeRole.OUTPUT)])
    # FIX: ENHANCE add_linear_processing_pathway TO SUPPORT InputPort at end, or OutputPort at beginning:
    # stimulus_encoding = [stim, em.input_ports[CONTENT_INPUT]]
    # context_encoding = [context, em.input_ports[ASSOC_INPUT]]

    # MappingProjection(sender=stim, receiver=stim_input_layer)
    # MappingProjection(sender=stim, receiver=em.input_ports[CONTENT_INPUT])
    # MappingProjection(sender=context, receiver=context_input_layer)
    # MappingProjection(sender=context, receiver=em.input_ports[ASSOC_INPUT])
    # MappingProjection(sender=em.output_ports[CONTENT_OUTPUT], receiver=stim_input_layer)
    # MappingProjection(sender=em.output_ports[ASSOC_OUTPUT], receiver=context_input_layer)
    # stim_processing = Pathway([stim, ffn])
    # context_processing = Pathway([context, ffn])
    # stim_encoding = Pathway([stim, em])
    # context_encoding = Pathway([context, em])
    # stim_retrieval = Pathway([em, stim_input_layer])
    # context_retrieval = Pathway([em, context_input_layer])
    # storage = Pathway([(decision, NodeRole.OUTPUT), (ctl, NodeRole.FEEDBACK_SENDER), em])
    # # FIX: show_graph NOT RECOGNIZING STIM->STIM_INPUT_LAYER AND CONTEXT->CONTEXT_INPUT_LAYER
    # comp = Composition(pathways=[stim_processing,
    #                              context_processing,
    #                              ffn,
    #                              context_encoding,
    #                              stim_encoding,
    #                              resp_decision,
    #                              stim_retrieval,
    #                              context_retrieval,
    #                              storage])
    # FIX: show_graph NOT RECOGNIZING STIM->STIM_INPUT_LAYER AND CONTEXT->CONTEXT_INPUT_LAYER
    # comp = Composition(pathways=[[stim, ffn],
    #                              [stim,em],
    #                              [context,ffn],
    #                              [context,em],
    #                              [em,ffn],
    #                              [ffn, em],
    #                              [ffn, decision, ctl, em]])

    # comp = Composition(pathways=[ffn,
    #                              [stim, stim_input_layer],
    #                              [stim, MappingProjection(stim, em.input_ports[CONTENT_INPUT]), em],
    #                              [context, context_input_layer],
    #                              [context, MappingProjection(context, em.input_ports[ASSOC_INPUT]), em],
    #                              [em,stim_input_layer],
    #                              [em,context_input_layer],
    #                              [ffn, decision, ctl, em]])

    comp = Composition()
    comp.add_nodes([stim, context, ffn, em, (decision, NodeRole.OUTPUT), ctl])
    comp.add_projection(MappingProjection(), stim, stim_input_layer)
    comp.add_projection(MappingProjection(), context, context_input_layer)
    comp.add_projection(MappingProjection(), stim, em.input_ports[CONTENT_INPUT])
    comp.add_projection(MappingProjection(), context, em.input_ports[ASSOC_INPUT])
    comp.add_projection(MappingProjection(), em.output_ports[CONTENT_OUTPUT], stim_input_layer)
    comp.add_projection(MappingProjection(), em.output_ports[ASSOC_OUTPUT], context_input_layer)
    comp.add_projection(MappingProjection(), match_output_layer, decision)
    comp.add_projection(MappingProjection(), decision, ctl)
    # comp.add_projection(MappingProjection(), decision, stim_input_layer)

    # comp._analyze_graph()
    # comp.show_graph()
    # comp.show_graph(show_cim=True,
    #                 show_node_structure=ALL,
    #                 show_projection_labels=True,
    #                 show_dimensions=True)
    # comp.show_graph(show_cim=True,
    #                 show_node_structure=ALL,
    #                 show_projection_labels=True,
    #                 show_dimensions=True)
    comp.run(inputs={stim:[1,2,3,4,5],
                     context:[6,7,8,9,10]},
             report_output=ReportOutput.ON)
    # comp.run(inputs={a:2.5}, report_output=ReportOutput.FULL)
#endregion
n_back_model()
