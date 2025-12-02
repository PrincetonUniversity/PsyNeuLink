

def target_projection_test():
    #              [   / hidden_2 ] -> output_1 -> output_2
    # input_1 ---> | <            |       '    /
    #          \   [   \ hidden_1 ]      '    /
    #           \                       v    /
    #            \-> hidden_3 -> hidden_4---/
    print("Running direct_and_1_ordinary_output_from_nested_to_another_and_parallel_pathway...")
    input_node_1 = ProcessingMechanism(name='input_1')
    hidden_node_1 = ProcessingMechanism(name='hidden_1')
    hidden_node_2 = ProcessingMechanism(name='hidden_2')
    hidden_node_3 = ProcessingMechanism(name='hidden_3')
    hidden_node_4 = ProcessingMechanism(name='hidden_4')
    output_node_1 = ProcessingMechanism(name='output_1')
    output_node_2 = ProcessingMechanism(name='output_2')
    input_nodes = [input_node_1]
    nested = AutodiffComposition(nodes = [hidden_node_1, hidden_node_2], name='nested')
    pathway_a = [input_node_1,
                 MappingProjection(input_node_1, hidden_node_1, learnable=False),
                 nested]
    pathway_b = [input_node_1,
                 MappingProjection(input_node_1, hidden_node_2, learnable=False),
                 nested,
                 MappingProjection(hidden_node_2, learnable=False),
                 output_node_1,
                 MappingProjection(output_node_1, output_node_2, learnable=False),
                 output_node_2]
    pathway_c = Pathway([input_node_1,
                         hidden_node_3,
                         hidden_node_4,
                         output_node_2],
                        learning_rate=2
                        )
    comp = AutodiffComposition(pathways=[pathway_a, pathway_b, pathway_c],)
    hidden_4_teacher_proj = TargetProjection(sender=output_node_1,
                                             receiver=hidden_4)
    comp.add_projection(hidden_4_teacher_proj)
    comp.learn(inputs={inputs},
               targets={output_2: [[1.0]]},
               execution_mode=pnl.ExecutionMode.PyTorch)
target_projection_test()