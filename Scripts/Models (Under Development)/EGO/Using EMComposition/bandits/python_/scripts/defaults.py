PARAMS = dict(
    n_base_trials=200,  # number of trials
    common_prob=0.7,  # probability of common transition (vs rare transition)

    sigma=0.025,  # bandit drift rate
    lo=0.25,  # bandit reward probability lower bound
    hi=0.75,  # bandit reward probability upper bound

    # value to choice parameters (only for the two-step bandit task)
    choice_temperature=5.,  # inverse temperature for choice stochasticity
    choice_bias=0.,  # .5,  # bias towards staying with the same choice as the previous trial

    # ego parameters
    time_drift_noise=.035,  # drift of the time representation in memory
    ego_temperature=.02,  # temperature of softmax for sampling memories
    ego_threshold=.01,  # threshold for discarding retrieved memories

    memory_capacity=None
)

STATE_INTEGRATION_RATES = [
    0.,
    .1,
    .2,
    .3,
    .4,
    .5,
    .6,
    .7,
    .8,
    .9,
    1.
]

TIME_RETRIEVAL_WEIGHTS = [
    0.,
    .1,
    .2,
    .3,
    .4,
    .5,
    .6,
    .7,
    .8,
    .9,
    1.
]


def get_folder(**kwargs):
    base = (f"time_drift_noise_{kwargs['time_drift_noise']}_"
            f"choice_temp_{kwargs['choice_temperature']}_"
            f"choice_bias_{kwargs['choice_bias']}_"
            f"sigma_{kwargs['sigma']}"
            f"memory_capacity_{kwargs['memory_capacity']}")
    if 'state_integration_rate' not in kwargs or 'time_retrieval_weight' not in kwargs:
        return (
            base,
            None
        )
    return (
        base,
        f"state_integration_rate_{kwargs['state_integration_rate']}_time_retrieval_weight_{kwargs['time_retrieval_weight']}"
    )
