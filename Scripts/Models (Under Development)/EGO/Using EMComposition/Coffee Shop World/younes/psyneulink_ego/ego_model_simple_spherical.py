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
        context_retrieval_weight,
        memory_fill,
        softmax_temperature,
        softmax_threshold,
        memory_capacity,
        **kwargs):
    retrieval_softmax_gain = 1 / softmax_temperature


    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    state_input_layer = ProcessingMechanism(name=state_input_name, input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name=previous_state_name, input_shapes=state_size)
    context_layer = TransferMechanism(name=context_name,
                                      input_shapes=state_size,
                                      integrator_mode=True,
                                      integrator_function=DriftOnASphereIntegrator(
                                          dimension=state_size,
                                          rate=integration_rate,
                                          noise=[0.01]*(state_size-1),
                                          input_space='target',

                                          
                                      )
                                     )


    em = EMComposition(
        name=em_name,
        memory_template=[[0] * state_size,  # state
                         [0] * state_size],  # context
        memory_fill=memory_fill,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=softmax_threshold,
        fields={state_input_layer.name: {FIELD_WEIGHT: None,
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
    )

    prediction_layer = ProcessingMechanism(name=prediction_layer_name, input_shapes=state_size)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
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
    context_to_em_pathway = [context_layer,
                                MappingProjection(sender=context_layer,
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
        [
         state_to_context_pathway,
         state_to_em_pathway,
         context_to_em_pathway],
        name=model_name,
        device=CPU)

    EGO_comp.scheduler.add_condition(em, BeforeNodes(context_layer))
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
