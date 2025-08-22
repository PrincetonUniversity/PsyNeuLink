from psyneulink import *


def construct_model(
        config,
        memory_capacity,

) -> Composition:
    model_name: str = config['name']

    # Input layer:
    state_input_name: str = config['state_input_layer_name']
    state_size: int = config['state_d']

    # Previous state
    previous_state_name: str = config['previous_state_layer_name']

    # Context representation (learned):
    context_name = config['context_layer_name']
    context_size = config['context_d']
    integration_rate = config['integration_rate']
    # EM:
    em_name = config['em_name']
    retrieval_softmax_threshold = config['softmax_threshold']
    state_retrieval_weight = config['state_weight']
    previous_state_retrieval_weight = config['previous_state_weight']
    context_retrieval_weight = config['context_weight']
    normalize_field_weights = config['normalize_field_weights']
    concatenate_queries = config['concatenate_queries']
    enable_learning = config['enable_learning']

    memory_init = config['memory_init']

    # Output:
    prediction_layer_name = config['prediction_layer_name']

    # Learning
    loss_spec = config['loss_spec']
    learning_rate = config['learning_rate']
    device = config['device']

    if is_numeric_scalar(config['softmax_temperature']):  # translate to gain of softmax retrieval function
        retrieval_softmax_gain = 1 / config['softmax_temperature']
    else:
        retrieval_softmax_gain = config['softmax_temperature']

    assert 0 <= integration_rate <= 1, \
        f"integrator_retrieval_weight must be a number from 0 to 1"

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_name, input_shapes=state_size)
    context_layer = TransferMechanism(name=context_name,
                                      input_shapes=context_size,
                                      function=Tanh(gain=1),
                                      integrator_mode=True,
                                      integration_rate=integration_rate)

    em = EMComposition(
        name=em_name,
        memory_template=[[0] * state_size,  # state
                         [0] * state_size,  # previous state
                         [0] * state_size],  # context
        memory_fill=memory_init,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=retrieval_softmax_threshold,
        fields={state_input_name: {FIELD_WEIGHT: state_retrieval_weight,
                                   LEARN_FIELD_WEIGHT: False,
                                   TARGET_FIELD: True},
                previous_state_name: {FIELD_WEIGHT: previous_state_retrieval_weight,
                                      LEARN_FIELD_WEIGHT: False,
                                      TARGET_FIELD: False},
                context_name: {FIELD_WEIGHT: context_retrieval_weight,
                               LEARN_FIELD_WEIGHT: False,
                               TARGET_FIELD: False}},
        normalize_field_weights=normalize_field_weights,
        normalize_memories=config['normalize_memories'],
        concatenate_queries=concatenate_queries,
        enable_learning=enable_learning,
        learning_rate=learning_rate,
        device=device,
        store_on_optimization='last')

    prediction_layer = ProcessingMechanism(name=prediction_layer_name, input_shapes=state_size)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
    state_to_previous_state_pathway = [state_input_layer,
                                       MappingProjection(matrix=IDENTITY_MATRIX,
                                                         learnable=False),
                                       previous_state_layer]
    state_to_context_pathway = [state_input_layer,
                                MappingProjection(matrix=IDENTITY_MATRIX,
                                                  learnable=False),
                                context_layer]
    state_to_em_pathway = [state_input_layer,
                           MappingProjection(sender=state_input_layer,
                                             receiver=em.nodes[state_input_name + VALUE],
                                             matrix=IDENTITY_MATRIX,
                                             learnable=False),
                           em]
    previous_state_to_em_pathway = [previous_state_layer,
                                    MappingProjection(sender=previous_state_layer,
                                                      receiver=em.nodes[previous_state_name + QUERY],
                                                      matrix=IDENTITY_MATRIX,
                                                      learnable=False),
                                    em]
    context_learning_pathway = [context_layer,
                                MappingProjection(sender=context_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  receiver=em.nodes[context_name + QUERY],
                                                  learnable=True),
                                em,
                                MappingProjection(sender=em.nodes[state_input_name + RETRIEVED],
                                                  receiver=prediction_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  learnable=False),
                                prediction_layer]

    # Composition
    EGO_comp = AutodiffComposition([state_to_previous_state_pathway,
                                    state_to_context_pathway,
                                    state_to_em_pathway,
                                    previous_state_to_em_pathway,
                                    context_learning_pathway],
                                   learning_rate=learning_rate,
                                   loss_spec=loss_spec,
                                   execute_in_additional_optimizations={context_layer: LAST,
                                                                        previous_state_layer: LAST},
                                   # BREADCRUMB: REQUIRED HERE UNTIL IMPLEMENTED FOR learn()
                                   optimizations_per_minibatch=config['num_optimization_steps'],
                                   name=model_name,
                                   device=device)

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    EGO_comp.add_projection(MappingProjection(sender=state_input_layer,
                                              receiver=learning_components[0],
                                              learnable=False))

    EGO_comp.scheduler.add_condition(em, BeforeNodes(previous_state_layer, context_layer))
    EGO_comp.scheduler.add_condition(prediction_layer, BeforeNodes(previous_state_layer, context_layer))

    return EGO_comp, context_layer, state_input_layer, em


def run_model(model,
              # context_layer,
              # state_input_layer,
              # em,
              trials,
              config,
              # learning=True,
              ):
    model.learn(inputs={config['state_input_layer_name']: trials},
                # learning_rate=config['learning_rate'],
                execution_mode=config['execution_mode'],
                optimizations_per_minibatch=config['num_optimization_steps'],
                minibatch_size=1,
                synch_projection_matrices_with_torch=RUN,
                synch_node_values_with_torch=RUN,
                synch_results_with_torch=RUN,
                )
    # model.learn(inputs={params_ego['state_input_layer_name']: trials},
    #             learning_rate=params_ego['learning_rate'],
    #             execution_mode=params_ego['execution_mode'],
    #             synch_projection_matrices_with_torch=params_ego['synch_weights'],
    #             synch_node_values_with_torch=params_ego['synch_values'],
    #             synch_results_with_torch=params_ego['synch_results'],
    #             minibatch_size=1,
    #             )
    # memory = em.memory
    print(model.results)
    # return model.results[config['num_optimization_steps'] - 1::config['num_optimization_steps']][:, 2]
    return model.results[::config['num_optimization_steps']][:, 2]

if __name__ == '__main__':
    trials = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]]
    model, _, _, _ = construct_model(memory_capacity=5, state_size=5, context_size=5)
    results = run_model(model, trials)
    print(results)
