"""
Here, we simulate reward estimation task from
`Giallanza et al. (2024)<https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00143/12108>`_.

Task
----
In an initial learning phase, the model sees two different sequences of three stimuli, each followed by a
different reward value. In a second, revaluation learning phase, the model sees sequences of two stimuli,
each beginning with the second stimulus in one of the sequences seen in the initial learning phase,
followed by one of two changes. In the reward *revaluation condition*, the sequence continued as in the
learning phase, but the reward associated with the two trajectories was swapped. In the
*transition revaluation* condition, the third stimulus was swapped between the two sequences, which was
then followed by the same reward that originally followed each of the third stimuli in the learning phase.
Finally, in a decision phase that followed each revaluation learning phase, the model estimates indicated its
preference between the starting states of the two trajectories by estimated the expected reward for
both ot them.

Model
-----
The model consists of an episodic memory module, a context (working memory) module, and a control unit
that controls if it is in "observation mode" or "estimation mode":

Observation Mode
~~~~~~~~~~~~~~~~
In observation mode, the model stores the current state, context, reward and a time value (implemented as
random drift on a sphere) to the episodic memory and updates its context by integrating the state.

Estimation Mode
~~~~~~~~~~~~~~~
In estimation model, the model uses a "simulated context" to retrieve states. It successively uses these
retrieved states to update its simulated context.

In addition, the model can also use states to retrieve past contexts (instead of simulating them). In this
mode the model behaves like a model based RL model:
    - _Without_ context retrieval, the model successfully estimates rewards in the *reward revaluation* condition
     but not in the *transition revaluation* condition.
    - _With_ context retrieval, the model successfully estimates rewards in both conditions (similar to model based
    RL learning)

Control
~~~~~~~
Control is implemented to switch between observation mode and estimation mode by controlling which inputs are
used to query the episodic memory and weather state or the retrieved state should update the (simulated) context.
It is also used to halt storing in estimation mode and to freeze the context in estimation mode.
"""

# Imports
import psyneulink as pnl

import params as params
import data as data

# Script Control
DISPLAY_MODEL = False

# Constants - Names
MODEL_NAME = 'EGO RETRIEVAL'
EM_NAME = 'EM'

STATE_NAME = 'STATE'
TIME_NAME = 'TIME'
REWARD_NAME = 'REWARD'
CONTEXT_NAME = 'CONTEXT'
TASK_NAME = 'TASK'


def construct_model(
        capacity=params.N_EXPERIENCE_SEQS,
        context_retrieval_in_sim=params.CONTEXT_RETRIEVE_IN_SIM,
        time_retrieval_weight=params.TIME_RETRIEVAL_WEIGHT,
):
    # Input Layers
    state_input = pnl.ProcessingMechanism(
        name=STATE_NAME,
        input_shapes=params.STATE_SIZE)
    time_input = pnl.ProcessingMechanism(
        name=TIME_NAME,
        input_shapes=params.TIME_SIZE)
    reward_input = pnl.ProcessingMechanism(
        name=REWARD_NAME,
        input_shapes=params.REWARD_SIZE)

    # Task: Observation or Prediction
    task_input = pnl.ProcessingMechanism(
        name=TASK_NAME,
        input_shapes=params.TASK_SIZE
    )

    # Context layer (working memory)
    context = pnl.TransferMechanism(
        name=CONTEXT_NAME,
        input_shapes=params.CONTEXT_SIZE,
        integrator_mode=True,
        integration_rate=params.STATE_INTEGRATION_RATE  # How much of the input to integrate
    )

    # Cache the names (if we construct multiple models, the constants above will be enumerated)
    _state_name = state_input.name
    _time_name = time_input.name
    _reward_name = reward_input.name
    _task_name = task_input.name

    _context_name = context.name

    # EM composition
    #   Note: We set retrieval weights to 1. since they will be modulated in the control signal (all but
    #       reward)
    em = pnl.EMComposition(name=EM_NAME,
                           memory_template=[[0] * params.STATE_SIZE,  # state
                                            [0] * params.TIME_SIZE,  # time
                                            [0] * params.CONTEXT_SIZE,  # context
                                            [0] * params.REWARD_SIZE],  # reward
                           memory_fill=params.MEMORY_INIT,
                           memory_capacity=capacity,
                           memory_decay_rate=0,
                           softmax_gain=1.0 / params.TEMPERATURE,
                           softmax_threshold=params.SOFTMAX_THRESHOLD,
                           normalize_memories=False,
                           normalize_field_weights=False,
                           concatenate_queries=False,
                           fields={
                               _state_name: {
                                   pnl.FIELD_WEIGHT: 1.,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               _time_name: {
                                   pnl.FIELD_WEIGHT: time_retrieval_weight,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               _context_name: {
                                   pnl.FIELD_WEIGHT: 1.,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                               _reward_name: {
                                   pnl.FIELD_WEIGHT: None,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                           })

    # create the model
    ego_comp = pnl.Composition(name=MODEL_NAME)

    ego_comp.add_nodes(
        [state_input, time_input, context, reward_input, em]
    )

    # Constants to access em:
    #   QUERY -> keys to retrieve memory
    #   VALUE -> values to store (without query)
    #   RETRIEVED -> retrieved values (we don't need this for now)
    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # EM encoding --------------------------------------------------------------------------------

    # Time and reward are not controlled with fixed weights.
    # time -> em
    ego_comp.add_projection(
        pnl.MappingProjection(time_input, em.nodes[_time_name + QUERY]))
    # reward -> em
    ego_comp.add_projection(
        pnl.MappingProjection(reward_input, em.nodes[_reward_name + VALUE]))

    # Inputs to Context ---------------------------------------------------------------------------
    # state -> context (integration rate is taken care of through the integration rate)
    ego_comp.add_projection(
        pnl.MappingProjection(state_input, context)
    )

    # === REWARD ESTIMATION MODE === #

    # -- Context Projection -- #
    # This is a projection of the context by simulating the same integration as
    # is happening above with state or retrieved state
    context_projected = pnl.TransferMechanism(
        name=_context_name + ' PROJECTED',
        input_shapes=params.CONTEXT_SIZE,
        integrator_mode=True,
        integration_rate=1.
    )

    # In the rollout, we alternate between two steps:
    #   - (1) retrieving reward from state_retrieved
    #   - (2) retrieving state_retrieved from context_projected
    # More specifically, here is the update schema for the context
    # (0) Very first step: $context_projected = (1-sr)*context + sr*state$ (initializing)
    # (1) $context_projected = (1-sr)*context_projected + sr*state_retrieved$ (projecting)
    # (2) Between step: $context_projected = context_projected$
    #       (freeze since we haven't retrieved state yet)
    # The following attention layers are used to control the inputs to the `context_projected`
    # We add the suffix _cp to indicate this attention is for context projection
    # (instead of for retrieval, see bellow)

    ego_comp.add_node(context_projected)
    # In the first step of the simulation (0), the current context is integrated into the
    # projected context as is the state:
    # Attend context to update context_projected
    attend_context_cp = pnl.ProcessingMechanism(
        name='ATTEND ' + _context_name + ' CP',
        input_shapes=params.CONTEXT_SIZE,
    )
    ego_comp.add_node(attend_context_cp)
    ego_comp.add_projection(
        pnl.MappingProjection(context, attend_context_cp)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_context_cp, context_projected)
    )

    # Attend state to update context_projected
    attend_state_cp = pnl.ProcessingMechanism(
        name='ATTEND ' + _state_name + ' CP',
        input_shapes=params.STATE_SIZE,
    )
    ego_comp.add_node(attend_state_cp)
    ego_comp.add_projection(
        pnl.MappingProjection(state_input, attend_state_cp)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_state_cp, context_projected)
    )

    # After the initial phase (1), context_projected is not updated with
    # the state anymore but with the retrieved state: the next state is
    # predicted from memory and used to update the simulated contex
    # Attend retrieved state to update context_projected
    attend_state_retrieved_cp = pnl.ProcessingMechanism(
        name='ATTEND ' + _state_name + RETRIEVED + ' CP',
        input_shapes=params.STATE_SIZE,
    )
    ego_comp.add_node(attend_state_retrieved_cp)
    ego_comp.add_projection(
        pnl.MappingProjection(em.nodes[_state_name + RETRIEVED], attend_state_retrieved_cp)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_state_retrieved_cp, context_projected)
    )

    # In the "model based" version of the model the context is not
    # updated from start, but (as the state) retrieved from episodic memory.
    # Attend retrieved context to update context_projected (instead of previous context_projected)
    attend_context_retrieved_cp = pnl.ProcessingMechanism(
        name='ATTEND ' + _context_name + RETRIEVED + ' CP',
        input_shapes=params.CONTEXT_SIZE,
    )
    ego_comp.add_node(attend_context_retrieved_cp)
    ego_comp.add_projection(
        pnl.MappingProjection(em.nodes[_context_name + RETRIEVED], attend_context_retrieved_cp)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_context_retrieved_cp, context_projected)
    )

    # -- Retrieval -- #
    # The queries that are used for retrieving from episodic memory, are
    # different for observation and prediction mode:
    # # (1) In observation mode: use "real" context + state
    # # (2) In prediction mode: use predicted context + retrieved weight
    #
    # The following attention layers are used to control the queries for em
    # We add the suffix _r to indicate this attention is for retrieval

    # Attend state (in observation mode)
    attend_state_r = pnl.ProcessingMechanism(
        name='ATTEND ' + _state_name + ' R',
        input_shapes=params.STATE_SIZE,
    )
    ego_comp.add_node(attend_state_r)
    ego_comp.add_projection(
        pnl.MappingProjection(state_input, attend_state_r)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_state_r, em.nodes[_state_name + QUERY])
    )

    # Attend retrieved state (in the estimation mode)
    attend_state_retrieved_r = pnl.ProcessingMechanism(
        name='ATTEND ' + _state_name + RETRIEVED + ' R',
        input_shapes=params.STATE_SIZE,
    )
    ego_comp.add_node(attend_state_retrieved_r)
    ego_comp.add_projection(
        pnl.MappingProjection(em.nodes[_state_name + RETRIEVED], attend_state_retrieved_r)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_state_retrieved_r, em.nodes[_state_name + QUERY])
    )

    # Attend context (in observation mode)
    attend_context_r = pnl.ProcessingMechanism(
        name='ATTEND ' + _context_name + ' R',
        input_shapes=params.CONTEXT_SIZE,
    )
    ego_comp.add_node(attend_context_r)
    ego_comp.add_projection(
        pnl.MappingProjection(context, attend_context_r)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_context_r, em.nodes[_context_name + QUERY])
    )

    # Attend projected context (in estimation mode)
    attend_context_projected_r = pnl.ProcessingMechanism(
        name='ATTEND ' + _context_name + ' PROJECTED' + ' R',
        input_shapes=params.CONTEXT_SIZE,
    )
    ego_comp.add_node(attend_context_projected_r)
    ego_comp.add_projection(
        pnl.MappingProjection(context_projected, attend_context_projected_r)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_context_projected_r, em.nodes[_context_name + QUERY])
    )

    # -- Control -- #

    # Task input to set control signals
    state_features = [task_input]

    # Control Schema
    control_signals = [
        ## -- Projected Context Integration -- ##
        # From Context: (integration rate is used for "recursion")
        (pnl.SLOPE, attend_context_cp),
        (pnl.INTEGRATION_RATE, context_projected),
        (pnl.SLOPE, attend_context_retrieved_cp),
        # From State:
        (pnl.SLOPE, attend_state_cp),
        (pnl.SLOPE, attend_state_retrieved_cp),

        ## -- Retrieval -- ##
        # For Context:
        (pnl.SLOPE, attend_context_r),
        (pnl.SLOPE, attend_context_projected_r),
        # For State:
        (pnl.SLOPE, attend_state_r),
        (pnl.SLOPE, attend_state_retrieved_r),

        ## -- Retrieval Weights -- #
        # Alternating between retrieving reward from state and state from context
        (pnl.SLOPE, em.nodes['STATE [WEIGHT]']), (pnl.SLOPE, em.nodes['CONTEXT [WEIGHT]']),

        ## -- Storage -- ##
        (pnl.STORAGE_PROB, em.nodes['STORE']),
        ## -- Context -- ## (This is used to 'freeze' the context during estimation
        (pnl.INTEGRATION_RATE, context),
    ]

    # how to decide control function
    # Observation mode:
    _control_signal_observation = [
        # Projected Context integration from context (no integration)
        0, 0, 0,  # context, context_projected (integration rate), context_retrieved
        0, 0,  # state, state_retrieved
        # Retrieval
        1, 0,  # context, projected context
        1, 0,  # state, retrieved state
        # Retrieval weights
        0, 1 - time_retrieval_weight,  # state, context [query]
        # storage
        1,  # storage probability
        # context integration
        1,  # attention this modulates the one set in context (doesn't set it)
    ]

    # Here, we set the context_projected = context * (1 - sr) + state * sr
    # Note, we have set the integration rate of the context_project layer to 1 but
    # set the attend_state_cp to sr

    # how much the context vs context_retrieved (model free vs model based)
    context_rate = (1 - context_retrieval_in_sim) * (1 - params.STATE_INTEGRATION_RATE)
    context_retrieved_rate = context_retrieval_in_sim * (1 - params.STATE_INTEGRATION_RATE)
    _control_signal_estimation_init = [
        # Projected Context integration from context
        context_rate, 1., context_retrieved_rate,  # context, context_projected (integration rate), retrieved
        # Projected Context integration from state
        params.STATE_INTEGRATION_RATE, 0,  # state, retrieved state
        # Retrieval weights
        0, 1,  # context, projected context
        1, 0,  # state, retrieved state
        # Retrieval weights
        1 - time_retrieval_weight, 0,  # state, context [query]
        # Storage
        0,
        # Context integration (0 means freeze)
        0
    ]

    # Here, we control how much of the old vs retrieved context is stored, by
    # manipulating
    # (1) it's integration rate between sr and 1 (1 indicating full replacement via retrieved context)
    # (2) The weight of retrieved context between 0 and 1-sr
    # (3) The weight of state between 1 and sr
    context_projected_rate = (
            params.STATE_INTEGRATION_RATE + (1 - params.STATE_INTEGRATION_RATE) * context_retrieval_in_sim)
    context_retrieved_rate = (1 - params.STATE_INTEGRATION_RATE) * context_retrieval_in_sim
    state_rate = 1 - (1 - params.STATE_INTEGRATION_RATE) * context_retrieval_in_sim

    _control_signal_estimation_1 = [
        # Projected Context integration from context
        0, context_projected_rate, context_retrieved_rate,  # context, context projected, context retrieved
        0, state_rate,  # state, state retrieved
        # Retrieval weights
        0, 1,  # context, projected context
        0, 1,  # state, retrieved state
        # Retrieval weights
        1 - time_retrieval_weight, 0,  # state, context [query]
        # Storage
        0,
        # Context integration (0 means freeze)
        0
    ]

    # Here, we set the "freeze" context_projected and retrieve with context the next state
    # Note, we have set the integration rate of the context_project layer to 1 but
    # set the attend_state_cp to sr

    _control_signal_estimation_2 = [
        # Projected Context integration from context
        0, 0, 0,  # context, context projected, context retrieved
        0, 0,  # state, state retrieved
        # Retrieval weights
        0, 1,  # context, projected context
        0, 1,  # state, retrieved state
        # Retrieval weights
        0, 1 - time_retrieval_weight,  # state, context [query]
        # Storage
        0,
        # Context integration (0 means freeze)
        0
    ]

    def control_function(_state_features):
        """
        For now, all the control signals are just read, but we will later add them
        to an emMechanism instead
        """
        if _state_features[0] == 0:
            _control_signals = _control_signal_observation
        # state_features[0] == 1 -> init mode
        if _state_features[0] == 1:
            _control_signals = _control_signal_estimation_init
        if _state_features[0] == 2:
            _control_signals = _control_signal_estimation_1
        if _state_features[0] == 3:
            _control_signals = _control_signal_estimation_2
        _control_signals = [[c] for c in _control_signals]

        return _control_signals

    # control node
    control = pnl.ControlMechanism(name='CONTROL',
                                   monitor_for_control=state_features,
                                   function=control_function,
                                   control=control_signals)

    ego_comp.add_nodes([task_input, control])

    # Scheduling make sure context runs after em to update after retrieal
    ego_comp.scheduler.add_condition(em, pnl.BeforeNodes(context))
    ego_comp.scheduler.add_condition(em, pnl.BeforeNodes(context_projected))
    ego_comp.scheduler.add_condition(attend_state_retrieved_r, pnl.BeforeNodes(em))
    ego_comp.scheduler.add_condition(context, pnl.BeforeNodes(context_projected))
    ego_comp.scheduler.add_condition(context_projected, pnl.BeforeNodes(attend_context_projected_r))

    return ego_comp, state_input, time_input, reward_input, task_input


# region SCRIPT EXECUTION
# ======================================================================================================================
#                                                   SCRIPT EXECUTION
# ======================================================================================================================


if __name__ == '__main__':
    seq_baseline = 20

    memory_capacity = seq_baseline * 3
    model, _state_input, _time_input, _reward_input, _task_input = construct_model(memory_capacity)

    if DISPLAY_MODEL:
        model.show_graph()

    states, rewards = data.get_baseline_trials(num_seqs=seq_baseline)

    times = data.get_time_sequence(num_trials=len(states))

    inputs = {
        _state_input: states,
        _time_input: times,
        _reward_input: rewards,
        _task_input: [0] * len(states),
    }


    def _cb():
        print('*' * 10)


    def _ca():
        print('*' * 10)


    model.run(inputs=inputs,
              # call_before_trial=_cb,
              # call_after_trial=_ca
              )
    _memory = model.nodes[EM_NAME].parameters.memory.get(MODEL_NAME)

    print('*' * 10)
    print()
    print(model.results)
    #
    prediction_task_baseline = {
        _task_input: [1, 3, 2, 3, 2],
        _state_input: [[0, 1, 0, 0, 0, 0, 0]] * 5,
        _time_input: [times[-1]] * 5,
        _reward_input: [0] * 5,
    }

    model.run(
        inputs=prediction_task_baseline,
        call_before_trial=_cb,
        call_after_trial=_ca
    )
    print('*' * 10)
    print('Estimated Reward State 1 (~10)')
    print(model.results[-1])
    print('*' * 10)

# print(model.results)
