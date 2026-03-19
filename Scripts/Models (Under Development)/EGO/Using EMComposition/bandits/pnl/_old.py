"""
Here, we simulate reward estimation task from
`Giallanza et al. (2024)<https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00143/12108>`_ (Study 1).

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


Control
~~~~~~~
Control is implemented to switch between observation mode and estimation mode by controlling which inputs are
used to query the episodic memory and weather state or the retrieved state should update the (simulated) context.
It is also used to halt storing in estimation mode and to freeze the context in estimation mode.
"""

# Imports
import psyneulink as pnl

import defaults

# import data as data

# Script Control
DISPLAY_MODEL = False

# Constants - Names
MODEL_NAME = 'EGO RETRIEVAL'
EM_NAME = 'EM'

STATE_NAME = 'STATE'
TIME_NAME = 'TIME'
REWARD_NAME = 'REWARD'
CONTEXT_NAME = 'CONTEXT'
TASK_NAME = 'TASK CONTROL'

# Constants to access em:
#   QUERY -> keys to retrieve memory
#   VALUE -> values to store
#   RETRIEVED -> retrieved values
QUERY = ' [QUERY]'
VALUE = ' [VALUE]'
RETRIEVED = ' [RETRIEVED]'


def construct_model(
        # EM Parameters
        capacity: int = 100,
        memory_fill: float = .01,
        em_softmax_temperature: float = .1,
        em_softmax_threshold: float = .01,

        state_size: int = 9,
        time_size: int = 25,
        reward_size: int = 1,
        task_size: int = 1,

        # Context Parameters
        state_integration_rate: float = .5,

        # Weights
        time_retrieval_weight: float = .5
):
    """

    :param capacity: The number of slots in episodic memory (usually number of stimuli presented)
    :param memory_fill: The values with which the episodic memory is initialized
    :param em_softmax_temperature: The softmax temperature when retrieving memories from em
    :param em_softmax_threshold: A threshold for the similarity of retrieved memories (if the dot product
        between a memory and the query is below this value this memory is discarded)
    :param state_size: States are encoded one-hot. This is usually the number of different states.
    :param time_size: The length of the time vector.
    :param reward_size: The length of the reward vector (1 if reward is encoded as a scalar)
    :param task_size: The length of the task vector. This governs the control signals in the model (for
        example, observation vs prediction mode)
    :param state_integration_rate: State integration rate. The integration rate of the state vector into the context.
        1 means the context is a perfect copy of the state (em will store context as previous state and state)
        0 means the context is not updated at all
    :param time_retrieval_weight: Weight of time retrieval. This governs how much time biases the retrieval of
        memories

    """
    assert em_softmax_temperature > 0, 'Softmax temperature must be above 0'
    em_softmax_gain = 1 / em_softmax_temperature

    # Input Layers
    state_input = pnl.ProcessingMechanism(
        name=STATE_NAME,
        input_shapes=state_size)
    # TODO: make this a drift on a sphere integrator (no input)
    time_input = pnl.ProcessingMechanism(
        name=TIME_NAME,
        input_shapes=time_size)
    reward_input = pnl.ProcessingMechanism(
        name=REWARD_NAME,
        input_shapes=reward_size)

    # Task control input (controls task and phase of the model)
    # For example:
    #   - In observation mode: integrate context, store to memory
    #   - In prediction mode: Switch to predicted context, don't store memories ...
    task_input = pnl.ProcessingMechanism(
        name=TASK_NAME,
        input_shapes=task_size
    )

    # Context layer (simple integrator as working memory)
    context = pnl.TransferMechanism(
        name=CONTEXT_NAME,
        input_shapes=state_size,
        integrator_mode=True,
        integration_rate=state_integration_rate  # How much of the input to integrate
    )

    # Cache the names (if we construct multiple models, the constants above will be enumerated)
    _state_name = state_input.name
    _time_name = time_input.name
    _reward_name = reward_input.name
    _task_name = task_input.name
    _context_name = context.name

    # EM composition
    em = pnl.EMComposition(name=EM_NAME,
                           memory_template=[
                               [0] * state_size,  # state
                               [0] * time_size,  # time
                               [0] * state_size,  # context
                               [0] * reward_size  # reward
                           ],
                           memory_fill=memory_fill,
                           memory_capacity=capacity,
                           memory_decay_rate=0.,
                           softmax_gain=em_softmax_gain,
                           softmax_threshold=em_softmax_threshold,
                           normalize_memories=False,
                           normalize_field_weights=False,
                           concatenate_queries=False,
                           fields={
                               _state_name: {
                                   pnl.FIELD_WEIGHT: None,
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               _time_name: {
                                   pnl.FIELD_WEIGHT: time_retrieval_weight,  # <- fixed time retrieval weight
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               _context_name: {
                                   pnl.FIELD_WEIGHT: 1-time_retrieval_weight,  # <- fixed context retrieval weight
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False
                               },
                               _reward_name: {
                                   pnl.FIELD_WEIGHT: None,  # <- never use reward to retrieve from memory
                                   pnl.LEARN_FIELD_WEIGHT: False,
                                   pnl.TARGET_FIELD: False},
                           })

    # create the model
    ego_comp = pnl.Composition(name=MODEL_NAME)

    ego_comp.add_nodes(
        [state_input, time_input, context, reward_input, em]
    )

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
        input_shapes=state_size,
        integrator_mode=True,
        integration_rate=1.
    )

    # In the rollout, we alternate between two steps:
    #   - (1) retrieving reward from state_retrieved
    #   - (2) retrieving state_retrieved from context_projected
    # More specifically, here is the update schema for the context
    # (0) Very first step: $context_projected = (1-sir)*context + sir*state$ (initializing)
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
        input_shapes=state_size,
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
        input_shapes=defaults.STATE_SIZE,
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
        input_shapes=defaults.STATE_SIZE,
    )
    ego_comp.add_node(attend_state_retrieved_cp)
    ego_comp.add_projection(
        pnl.MappingProjection(em.nodes[_state_name + RETRIEVED], attend_state_retrieved_cp)
    )
    ego_comp.add_projection(
        pnl.MappingProjection(attend_state_retrieved_cp, context_projected)
    )

    # -- Retrieval -- #
    # The queries that are used for retrieving from episodic memory, are
    # different for observation and prediction mode:
    # # (1) In observation mode: use "real" context
    # # (2) In prediction mode: use predicted context
    #
    # The following attention layers are used to control the queries for em
    # We add the suffix _r to indicate this attention is for retrieval

    # Attend context (in observation mode)
    attend_context_r = pnl.ProcessingMechanism(
        name='ATTEND ' + _context_name + ' R',
        input_shapes=defaults.CONTEXT_SIZE,
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
        input_shapes=defaults.CONTEXT_SIZE,
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
        (pnl.SLOPE, attend_context_cp), (pnl.INTEGRATION_RATE, context_projected),

        # From State:
        (pnl.SLOPE, attend_state_cp), (pnl.SLOPE, attend_state_retrieved_cp),

        ## -- Retrieval -- ##
        # For Context:
        (pnl.SLOPE, attend_context_r), (pnl.SLOPE, attend_context_projected_r),

        ## -- Storage -- ##
        (pnl.STORAGE_PROB, em.storage_node),
        ## -- Context -- ## (This is used to 'freeze' the context and time during estimation
        (pnl.INTEGRATION_RATE, context),
        (pnl.NOISE, time_input)
    ]

    def get_control_signal(
            ### PROJECTED CONTEXT
            ## Inputs for the `projected context`
            attend_context_for_context_projected,
            integration_rate_for_context_projected,
            attend_state_for_context_projected,
            attend_state_retrieved_for_context_projected,
            ### RETRIEVAL
            ## Inputs for em to retrieve memories
            attend_context_for_retrieval,
            attend_context_projected_for_retrieval,
            ### Storage
            storage_probability,
            ### Integration
            context_integration_rate,
            time_noise,


    ):
        """
        :param attend_context_for_context_projected: this is mainly for initialization of the `projected context`
        :param integration_rate_for_context_projected: how the `projected context` is updated
        :param attend_state_for_context_projected: this is mainly for initialization of the `projected context`
        :param attend_state_retrieved_for_context_projected: in ''estimation mode`` the projected context is
            updated by `retrieved state` (not actual state)
        :param attend_context_for_retrieval: in ''observation mode``, the "real" `context` is used for retrieval
            (we don't "use" the retrieved memory, but by retrieving, we also store)
        :param attend_context_projected_for_retrieval: in ''estimation mode``, the `projected context` is used for
            retrieval
        """
        return [
            attend_context_for_context_projected,
            integration_rate_for_context_projected,
            attend_state_for_context_projected,
            attend_state_retrieved_for_context_projected,
            attend_context_for_retrieval,
            attend_context_projected_for_retrieval
            storage_probability,
            context_integration_rate,
            time_noise
        ]

    # how to decide control function
    # Observation mode:
    _control_signal_observation = get_control_signal(
        attend_context_for_context_projected=0,  # the projected context (cp) is not updated
        integration_rate_for_context_projected=0,  # ir == 0 -> "storage" of internal, value (cp not updated)
        attend_state_for_context_projected=0,  # cp is not updated
        attend_state_retrieved_for_context_projected=0,  # cp is not updated
        attend_context_for_retrieval=1., # the "real" context is used to retrieve (and store!)
        attend_context_projected_for_retrieval=0. # the projected context is not used to retrieve

    )
    _control_signal_observation = [
        # Projected Context integration from context (no integration)
        0, 0,  # context, context_projected (integration rate), context_retrieved
        0, 0,  # state, state_retrieved

        # Retrieval
        1, 0,  # context, projected context
        1, 0,  # state, retrieved state
        # Retrieval weights
        1 - time_retrieval_weight, 0,  # state, context [query]

        # storage
        1,  # storage probability
        # context integration
        1,  # attention this modulates the one set in context (doesn't set it)
    ]

    # Here, we set the context_projected = context * (1 - sr) + state * sr
    # Note, we have set the integration rate of the context_project layer to 1 but
    # set the attend_state_cp to sr

    # how much the context vs context_retrieved (model free vs model based)
    context_rate = (1 - defaults.STATE_INTEGRATION_RATE)
    _control_signal_estimation_init = [
        # Projected Context integration from context
        context_rate, 1.,  # context, context_projected (integration rate)
        # Projected Context integration from state
        defaults.STATE_INTEGRATION_RATE, 0,  # state, retrieved state
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
    # context_projected_rate = (
    #         defaults.STATE_INTEGRATION_RATE + (1 - defaults.STATE_INTEGRATION_RATE) * context_retrieval_in_sim)
    # context_retrieved_rate = (1 - defaults.STATE_INTEGRATION_RATE) * context_retrieval_in_sim
    # state_rate = 1 - (1 - defaults.STATE_INTEGRATION_RATE) * context_retrieval_in_sim
    context_projected_rate = context_retrieved_rate = state_rate = 1.

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
        0, 0,  # context, context projected, context retrieved
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

    # Scheduling make sure context runs after em to update after retrieval
    ego_comp.scheduler.add_condition(em, pnl.BeforeNodes(context))
    ego_comp.scheduler.add_condition(em, pnl.BeforeNodes(context_projected))
    ego_comp.scheduler.add_condition(attend_state_retrieved_r, pnl.BeforeNodes(em))
    ego_comp.scheduler.add_condition(context, pnl.BeforeNodes(context_projected))
    ego_comp.scheduler.add_condition(context_projected, pnl.BeforeNodes(attend_context_projected_r))

    return ego_comp, state_input, time_input, reward_input, task_input


if __name__ == '__main__':
    model, _, _, _, _ = construct_model()
    model.show_graph()
