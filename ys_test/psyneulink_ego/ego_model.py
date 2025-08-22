from psyneulink import *
from psyneulink._typing import Union

from ys_test.params import params_ego

if is_numeric_scalar(params_ego['softmax_temperature']):  # translate to gain of softmax retrieval function
    retrieval_softmax_gain = 1 / params_ego['softmax_temperature']
else:  # pass along ADAPTIVE or CONTROL spec
    retrieval_softmax_gain = params_ego['softmax_temperature']


def construct_model(
        memory_capacity,
        model_name: str = params_ego['name'],

        # Input layer:
        state_input_name: str = params_ego['state_input_layer_name'],
        state_size: int = params_ego['state_d'],

        # Previous state
        previous_state_name: str = params_ego['previous_state_layer_name'],

        # Context representation (learned):
        context_name: str = params_ego['context_layer_name'],
        context_size: Union[float, int] = params_ego['context_d'],
        integration_rate: float = params_ego['integration_rate'],

        # EM:
        em_name: str = params_ego['em_name'],
        retrieval_softmax_gain=retrieval_softmax_gain,
        retrieval_softmax_threshold=params_ego['softmax_threshold'],
        state_retrieval_weight: Union[float, int] = params_ego['state_weight'],
        previous_state_retrieval_weight: Union[float, int] = params_ego['previous_state_weight'],
        context_retrieval_weight: Union[float, int] = params_ego['context_weight'],
        normalize_field_weights=params_ego['normalize_field_weights'],
        concatenate_queries=params_ego['concatenate_queries'],
        enable_learning=params_ego['enable_learning'],

        memory_init=params_ego['memory_init'],

        # Output:
        prediction_layer_name: str = params_ego['prediction_layer_name'],

        # Learning
        loss_spec=params_ego['loss_spec'],
        learning_rate=params_ego['learning_rate'],
        device=params_ego['device']

) -> Composition:
    assert 0 <= integration_rate <= 1, \
        f"integrator_retrieval_weight must be a number from 0 to 1"

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_name, input_shapes=state_size)
    context_layer = TransferMechanism(name=context_name,
                                      input_shapes=context_size,
                                      function=Tanh,
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
        normalize_memories=False,
        concatenate_queries=concatenate_queries,
        enable_learning=enable_learning,
        learning_rate=learning_rate,
        device=device)

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
                                   name=model_name,
                                   device=device)

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    EGO_comp.add_projection(MappingProjection(sender=state_input_layer,
                                              receiver=learning_components[0],
                                              learnable=False))

    return EGO_comp, context_layer, state_input_layer, em


def run_model(model, inputs):
    for t in trials:
        model.learn(inputs={params_ego['state_input_layer_name']: [t]},
                    learning_rate=params_ego['learning_rate'],
                    execution_mode=params_ego['execution_mode'],
                    optimizations_per_minibatch=params_ego['num_optimization_steps'],
                    minibatch_size=1,
                    )
    return model.results[:, 2]


if __name__ == '__main__':
    inputs = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]]
    model, _, _, _ = construct_model(memory_capacity=5, state_size=5, context_size=5)
    results = run_model(model, inputs)
    print(results)
