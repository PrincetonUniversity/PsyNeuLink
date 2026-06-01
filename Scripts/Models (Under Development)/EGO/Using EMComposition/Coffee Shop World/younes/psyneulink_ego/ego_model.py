import numpy as np

from psyneulink import *

# NAMES
model_name: str = 'EGO'
state_input_name: str = 'STATE'
previous_state_name: str = 'PREVIOUS STATE'
context_name: str = 'CONTEXT'
prediction_layer_name = 'PREDICTION'
em_name: str = 'EM'


def construct_model(
        state_size,
        integration_rate,
        state_retrieval_weight,
        previous_state_retrieval_weight,
        context_retrieval_weight,
        learning_rate,
        memory_fill,
        softmax_temperature,
        softmax_threshold,
        loss_spec_name,
        num_optimization_steps,
        memory_capacity,
        **kwargs):
    retrieval_softmax_gain = 1 / softmax_temperature

    enable_learning = learning_rate > 0
    loss_spec = Loss(loss_spec_name)  # allow ValueError to raise if invalid
    if loss_spec is not Loss.BINARY_CROSS_ENTROPY:
        raise ValueError(loss_spec_name)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_name, input_shapes=state_size)
    context_layer = TransferMechanism(name=context_name,
                                      input_shapes=state_size,
                                      function=Tanh,
                                      integrator_mode=True,
                                      integration_rate=integration_rate)
    context_bias = TransferMechanism(name=context_name + '[bias]',
                                     input_shapes=1,
                                     default_variable=[1.],
                                     )

    context_normalized = TransferMechanism(name=CONTEXT + '[normalized]',
                                           input_shapes=state_size,
                                           function=Normalize(),
                                           )

    em = Emcomposition_Proj(
        name=em_name,
        memory_template=[[0] * state_size,  # state
                         [0] * state_size,  # previous state
                         [0] * state_size],  # context
        memory_fill=memory_fill,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=softmax_threshold,
        fields={state_input_layer.name: {FIELD_WEIGHT: state_retrieval_weight,
                                   LEARN_FIELD_WEIGHT: False,
                                   TARGET_FIELD: False},
                previous_state_layer.name: {FIELD_WEIGHT: previous_state_retrieval_weight,
                                      LEARN_FIELD_WEIGHT: False,
                                      TARGET_FIELD: False},
                context_layer.name: {FIELD_WEIGHT: context_retrieval_weight,
                               LEARN_FIELD_WEIGHT: False,
                               TARGET_FIELD: False}},
        normalize_field_weights=False,
        normalize_memories=False,
        concatenate_queries=False,
        enable_learning=False,
        device=CPU,
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
                                             receiver=em.nodes[state_input_layer.name + VALUE],
                                             matrix=IDENTITY_MATRIX,
                                             learnable=False),
                           em]
    previous_state_to_em_pathway = [previous_state_layer,
                                    MappingProjection(sender=previous_state_layer,
                                                      receiver=em.nodes[previous_state_layer.name + QUERY],
                                                      matrix=IDENTITY_MATRIX,
                                                      learnable=False),
                                    em]
    context_learning_pathway = [context_layer,
                                MappingProjection(sender=context_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  receiver=context_normalized,
                                                  learnable=True),
                                context_normalized,
                                MappingProjection(sender=context_normalized,
                                                  matrix=IDENTITY_MATRIX,
                                                  receiver=em.nodes[context_layer.name + QUERY],
                                                  learnable=False),
                                em,
                                MappingProjection(sender=em.nodes[state_input_layer.name + RETRIEVED],
                                                  receiver=prediction_layer,
                                                  matrix=IDENTITY_MATRIX,
                                                  learnable=False),
                                prediction_layer]

    # Composition
    EGO_comp = AutodiffComposition(
        [state_to_previous_state_pathway,
         state_to_context_pathway,
         state_to_em_pathway,
         previous_state_to_em_pathway,
         context_learning_pathway],
        learning_rate=learning_rate,
        loss_spec=loss_spec,
        execute_in_additional_optimizations={
            context_layer: LAST,
            previous_state_layer: LAST
        },
        optimizations_per_minibatch=num_optimization_steps,
        name=model_name,
        device=CPU)

    EGO_comp.add_node(context_bias)
    EGO_comp.add_projection(
        sender=context_bias,
        receiver=context_normalized,
        projection=MappingProjection(
            matrix=np.zeros((1, state_size)),
            learnable=True
        )
    )

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    EGO_comp.add_projection(MappingProjection(sender=state_input_layer,
                                              receiver=learning_components[0],
                                              learnable=False))

    EGO_comp.scheduler.add_condition(em, BeforeNodes(previous_state_layer, context_layer))
    EGO_comp.scheduler.add_condition(context_normalized, BeforeNodes(em))
    EGO_comp.scheduler.add_condition(prediction_layer, BeforeNodes(previous_state_layer, context_layer))

    return EGO_comp, state_input_layer


def run_model(model, input_layer, states, num_optimization_steps, **kwargs):
    model.learn(
        inputs={input_layer: states},
        execution_mode=ExecutionMode.PyTorch,
        optimizations_per_minibatch=num_optimization_steps,
        minibatch_size=1,
        synch_projection_matrices_with_torch=RUN,
        synch_node_values_with_torch=RUN,
        synch_results_with_torch=RUN,

    )

    return model.results[::num_optimization_steps][:, 2]

if __name__ == '__main__':
    trials = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]]
    # model, _, _, _ = construct_model(memory_capacity=5, state_size=5, context_size=5)
    config = {'name' : 'EGO_Model',
              'em_name': 'EM',
              'concatenate_queries': False,
              'enable_learning': True,
              'softmax_temperature': 0.1,
              'softmax_threshold': 1e-5,
              'normalize_memories': False,
              'num_optimization_steps': 5,
              'learning_rate': 0.01,
              'state_input_layer_name': 'STATE',
              'previous_state_layer_name': 'PREVIOUS_STATE',
              'context_layer_name': 'CONTEXT',
              'prediction_layer_name': 'PREDICTION',
              'state_size': 5,
              'context_d': 5,
              'memory_fill': 0.001,
              'state_retrieval_weight': 1.,
              'previous_state_retrieval_weight': 1.,
              'context_retrieval_weight': 1.,
              'normalize_field_weights': False,
              'device': 'cpu',
              'loss_spec_name': Loss.BINARY_CROSS_ENTROPY,
              'execution_mode': ExecutionMode.PyTorch,
              'integration_rate': 0.5,
              }
    model, _, _, _ = construct_model(memory_capacity=5, **config)
    results = run_model(model, trials)
    print(results)
