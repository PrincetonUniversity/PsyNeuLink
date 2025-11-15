import numpy as np

import psyneulink as pnl

import params as params
import data as data

# Script Control
DISPLAY_MODEL = False

# Constants
MODEL_NAME = 'EGO RETRIEVAL'

STATE_INPUT_NAME = 'STATE'
TIME_INPUT_NAME = 'TIME'
REWARD_INPUT_NAME = 'REWARD'
CONTEXT_LAYER_NAME = 'CONTEXT'

EM_NAME = 'EM'


# region   MODEL
# ======================================================================================================================
#                                                      MODEL
# ======================================================================================================================

def construct_model():
    # input layers
    state_input_layer = pnl.ProcessingMechanism(
        name=STATE_INPUT_NAME,
        input_shapes=params.STATE_SIZE)
    time_input_layer = pnl.ProcessingMechanism(
        name=TIME_INPUT_NAME,
        input_shapes=params.TIME_SIZE)
    reward_input_layer = pnl.ProcessingMechanism(
        name=REWARD_INPUT_NAME,
        input_shapes=params.REWARD_SIZE)

    # context layer (with integration)
    context_layer = pnl.TransferMechanism(
        name=CONTEXT_LAYER_NAME,
        input_shapes=params.CONTEXT_SIZE,
        integrator_mode=True,
        integration_rate=params.STATE_INTEGRATION_RATE  # How much of the input to integrate
    )

    # Cache the names (if we construct multiple models, these will be enumerated)
    _state_input_name = state_input_layer.name
    _time_input_name = time_input_layer.name
    _reward_input_name = reward_input_layer.name
    _context_input_name = context_layer.name

    # Retrieval rates (these are controlled in the reward estimation)
    _state_retrieval_rate = .45 #0.
    _context_retrieval_rate = .45 #1. - params.TIME_RETRIEVAL_WEIGHT

    em = pnl.EMComposition(name=EM_NAME,
                           memory_template=[[0] * params.STATE_SIZE,  # state
                                            [0] * params.TIME_SIZE,  # time
                                            [0] * params.CONTEXT_SIZE,  # context
                                            [0] * params.REWARD_SIZE],  # reward
                           memory_fill=params.MEMORY_INIT,
                           memory_capacity=params.N_EXPERIENCE_SEQS,
                           memory_decay_rate=0,
                           softmax_gain=1.0 / params.TEMPERATURE,
                           softmax_threshold=params.SOFTMAX_THRESHOLD,
                           fields={
                               _state_input_name: {
                                   pnl.FIELD_WEIGHT: _state_retrieval_rate,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               _time_input_name: {
                                   pnl.FIELD_WEIGHT: params.TIME_RETRIEVAL_WEIGHT,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               _context_input_name: {
                                   pnl.FIELD_WEIGHT: _context_retrieval_rate,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               _reward_input_name: {
                                   pnl.FIELD_WEIGHT: None,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                           })

    ego_comp = pnl.Composition(name=MODEL_NAME)

    # Nodes not included in (decision output) Pathway specified above
    ego_comp.add_nodes(
        [state_input_layer, time_input_layer, context_layer, reward_input_layer, em]
    )

    # Constants to access em:
    #   QUERY -> keys to retrieve memory
    #   VALUE -> values to store (without query)
    #   RETRIEVED -> retrieved values (we don't need this for now)
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding --------------------------------------------------------------------------------
    # state -> em
    ego_comp.add_projection(
        pnl.MappingProjection(state_input_layer, em.nodes[_state_input_name + QUERY]))
    # time -> em
    ego_comp.add_projection(
        pnl.MappingProjection(time_input_layer, em.nodes[_time_input_name + QUERY]))
    # context -> em
    ego_comp.add_projection(
        pnl.MappingProjection(context_layer, em.nodes[_context_input_name + QUERY]))
    # reward -> em
    ego_comp.add_projection(
        pnl.MappingProjection(reward_input_layer, em.nodes[_reward_input_name + VALUE]))

    # Inputs to Context ---------------------------------------------------------------------------
    # state -> context_layer (integration rate is taken care of through the integration rate
    ego_comp.add_projection(
        pnl.MappingProjection(state_input_layer, context_layer))

    ego_comp.scheduler.add_condition(context_layer, pnl.BeforeNodes(state_input_layer))

    return ego_comp, state_input_layer, time_input_layer, reward_input_layer


# region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================

if __name__ == '__main__':
    model, state_input_layer, time_input_layer, reward_input_layer = construct_model()

    if DISPLAY_MODEL:
        model.show_graph()

    states, rewards = data.get_baseline_trials(num_seqs=10)

    times = data.get_time_sequence(num_trials=len(states))

    inputs = {
        state_input_layer: states,
        time_input_layer: times,
        reward_input_layer: rewards,
    }

    model.run(inputs=inputs)

    _memory = model.nodes[EM_NAME].parameters.memory.get(MODEL_NAME)

    #print(_memory)
    print(model.results)

    #print(model.results)
