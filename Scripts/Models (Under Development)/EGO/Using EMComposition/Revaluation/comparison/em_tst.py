import numpy as np

import psyneulink as pnl

import params as params
import data as data

# Script Control
DISPLAY_MODEL = False

# Constants
MODEL_NAME = 'EGO RETRIEVAL'

STATE_INPUT_NAME = 'STATE'
CONTEXT_LAYER_NAME = 'CONTEXT'

EM_NAME = 'EM'


# region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model(capacity):
    # input layers
    state_input_layer = pnl.ProcessingMechanism(
        name=STATE_INPUT_NAME,
        input_shapes=params.STATE_SIZE)
    context_layer = pnl.TransferMechanism(
        name=CONTEXT_LAYER_NAME,
        input_shapes=params.CONTEXT_SIZE,
        function=pnl.Tanh(gain=1),
        integrator_mode=True,
        integration_rate=params.STATE_INTEGRATION_RATE  # How much of the input to integrate
    )

    # Cache the names (if we construct multiple models, these will be enumerated)
    _state_input_name = state_input_layer.name
    _context_input_name = context_layer.name

    # Retrieval rates (these are controlled in the reward estimation)
    _state_retrieval_rate = 0.45
    _context_retrieval_rate = .45

    em = pnl.EMComposition(name=EM_NAME,
                           memory_template=[[0] * params.STATE_SIZE,  # state
                                            [0] * params.CONTEXT_SIZE],  # c
                           memory_fill=params.MEMORY_INIT,
                           memory_capacity=capacity,
                           memory_decay_rate=0,
                           softmax_gain=1.0 / params.TEMPERATURE,
                           softmax_threshold=params.SOFTMAX_THRESHOLD,
                           fields={
                               _state_input_name: {
                                   pnl.FIELD_WEIGHT: _state_retrieval_rate,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },

                               _context_input_name: {
                                   pnl.FIELD_WEIGHT: _context_retrieval_rate,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                           },
                           # normalize_field_weights=False,
                           normalize_memories=False,
                           # concatenate_queries=False,
                           )

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    state_to_context = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        context_layer
    ]
    state_to_em = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              sender=state_input_layer,
                              receiver=em.nodes[STATE_INPUT_NAME + QUERY],
                              learnable=False),
        em
    ]
    context_to_em = [
        context_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              sender=context_layer,
                              receiver=em.nodes[CONTEXT_LAYER_NAME + QUERY],
                              learnable=False),
        em
    ]

    ego_comp = pnl.Composition(name=MODEL_NAME,
                                       pathways=[
                                           state_to_context,
                                           state_to_em,
                                           context_to_em,

                                       ]
                                       )
    ego_comp.scheduler.add_condition(em, pnl.BeforeNode(context_layer))

    # Constants to access em:
    #   QUERY -> keys to retrieve memory
    #   VALUE -> values to store (without query)
    #   RETRIEVED -> retrieved values (we don't need this for now)

    # EM encoding --------------------------------------------------------------------------------
    # state -> em

    # Inputs to Context ---------------------------------------------------------------------------
    # state -> context_layer (integration rate is taken care of through the integration rate

    return ego_comp, state_input_layer


# region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

if __name__ == '__main__':
    model, state_input_layer = construct_model(30)

    if DISPLAY_MODEL:
        model.show_graph()

    states, rewards = data.get_baseline_trials(num_seqs=10)

    times = data.get_time_sequence(num_trials=len(states))

    inputs = {
        state_input_layer: states
    }

    model.run(inputs=inputs)

    # print(_memory)
    print(model.results)

    # print(model.results)
