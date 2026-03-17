import psyneulink
import psyneulink as pnl

# Names
EM_NAME = 'EM'
STATE_NAME = 'STATE'
CONTEXT_NAME = 'CONTEXT'
TIME_NAME = 'TIME'
REWARD_NAME = 'REWARD'

# Sizes
STATE_SIZE = CONTEXT_SIZE = 7
TIME_SIZE = 25
REWARD_SIZE = 1  # <- I think this should be one-hot as well


def create_model(
        memory_capacity: int,
        memory_fill: float = .01,
        softmax_temperature: float = 1.,
        softmax_threshold: float = .01,
        context_integration_rate: float = .5,
        noise_on_a_sphere: float = .05
):
    assert softmax_temperature > 0, "Softmax temperature must be > 0"
    softmax_gain = 1. / softmax_temperature

    state_input_layer = pnl.ProcessingMechanism(
        name=STATE_NAME,
        input_shapes=STATE_SIZE
    )

    reward_input_layer = pnl.ProcessingMechanism(
        name=REWARD_NAME,
        input_shapes=STATE_SIZE
    )

    context_layer = pnl.TransferMechanism(
        name=CONTEXT_NAME,
        input_shapes=CONTEXT_SIZE,
        integrator_mode=True,
        integration_rate=context_integration_rate

    )

    time_layer = pnl.TransferMechanism(
        name=TIME_NAME,
        input_shapes=TIME_SIZE,
        integrator_mode=True,
        integration_rate=pnl.DriftOnASphereIntegrator(
            dimension=TIME_SIZE,
            rate=0.,
            noise=[noise_on_a_sphere] * TIME_SIZE - 1,

        )
    )

    #
    em = pnl.EMComposition(
        name=EM_NAME,
        memory_template=[
            [0] * STATE_SIZE,
            [0] * CONTEXT_SIZE,
            [0] * TIME_SIZE,
            [0] * REWARD_SIZE
        ],
        memory_fill=memory_fill,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=softmax_gain,
        softmax_threshold=softmax_threshold,
        fields={
            state_input_layer.name: {
                pnl.FIELD_WEIGHT: 1.,  # <- depends on the "phase"
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False},
            context_layer.name: {
                pnl.FIELD_WEIGHT: 1.,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False},
            time_layer.name: {
                pnl.FIELD_WEIGHT: 1.,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False}},
        normalize_field_weights=False,
        normalize_memories=False,
        concatenate_queries=False,
        enable_learning=False,
    )

    # Mappings
    # State -> Context
    state_context_pathway = [
        state_input_layer,
        pnl.MappingProjection(
            sender=state_input_layer,
            receiver=context_layer,
            matrix=pnl.IDENTITY_MATRIX,
            learnable=False
        ),
        context_layer]

    # TO EM
    state_em = [
        state_input_layer,
        pnl.MappingProjection(

        )
    ]
