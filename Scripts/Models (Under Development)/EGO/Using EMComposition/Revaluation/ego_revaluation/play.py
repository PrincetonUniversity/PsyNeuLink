import psyneulink as pnl
import numpy as np


def safe_softmax(t, threshold=0.01, **kwargs):
    """
    NumPy version of the Torch safe_softmax.
    EXACT behavior preserved:
    - mask values below threshold → -inf
    - subtract max
    - exponentiate
    - normalize
    - if all masked → return zeros
    """

    v = t.astype(float)

    # Apply mask: only include values greater than threshold
    if threshold is not None:
        mask = np.abs(t) > threshold
        v = np.where(mask, v, -np.inf)

    # Shift by global max (only if not all -inf)
    if np.any(v != -np.inf):
        v = v - np.max(v)

    # Exponential
    v = np.exp(v)

    # Normalize (to sum to 1)
    if not np.any(v):  # equivalent to "if not v.any()"
        return v  # all zeros → return zeros
    else:
        return v / np.sum(v)


def normalized(x: np.ndarray) -> np.ndarray:
    """
    Row-wise normalize vectors along the last axis.

    - Zero vectors remain zero.
    """
    x = np.asarray(x, dtype=float)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)

    # avoid division by zero row-wise
    safe_norm = np.where(norm == 0, 1, norm)

    return x / safe_norm


def get_em(normalize,
           state_weight,
           context_weight,
           softmax_temperature,
           mem_init,
           softmax_threshold,
           dim_states,
           dim_context,
           capacity
           ):
    state_in = pnl.ProcessingMechanism(name='STATE', input_shapes=dim_states)
    context_in = pnl.ProcessingMechanism(name='CONTEXT', input_shapes=dim_context)
    em = pnl.EMComposition(
        name='em',
        memory_template=[
            [0] * dim_states,
            [0] * dim_context,
        ],
        memory_fill=mem_init,
        softmax_threshold=softmax_threshold,
        memory_capacity=capacity,
        memory_decay_rate=0,
        fields={
            'STATE': {
                pnl.FIELD_WEIGHT: state_weight,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            },
            'CONTEXT': {
                pnl.FIELD_WEIGHT: context_weight,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            }
        },
        normalize_field_weights=False,
        normalize_memories=normalize,
        softmax_gain=1 / softmax_temperature,
        enable_learning=False
    )

    state_to_em = [
        state_in,
        pnl.MappingProjection(
            sender=state_in,
            receiver=em.nodes['STATE [QUERY]'],
            matrix=pnl.IDENTITY_MATRIX),
        em
    ]
    context_to_em = [
        context_in,
        pnl.MappingProjection(
            sender=context_in,
            receiver=em.nodes['CONTEXT [QUERY]'],
            matrix=pnl.IDENTITY_MATRIX),
        em
    ]

    comp = pnl.Composition(
        name='ego',
        pathways=[state_to_em, context_to_em]
    )
    return comp, state_in, context_in


def get_results_pnl(normalize,
                    state_vec,
                    context_vec,
                    state_weight,
                    context_weight,
                    softmax_temperature,
                    mem_init,
                    softmax_threshold,
                    dim_states,
                    dim_context,
                    capacity
                    ):
    _comp, _state_in, _context_in = get_em(
        normalize=normalize,
        state_weight=state_weight,
        context_weight=context_weight,
        softmax_temperature=softmax_temperature,
        mem_init=mem_init,
        softmax_threshold=softmax_threshold,
        dim_states=dim_states,
        dim_context=dim_context,
        capacity=capacity
    )

    inputs = {
        _state_in: state_vec,
        _context_in: context_vec
    }

    _comp.run(
        inputs=inputs,
    )

    state_out = _comp.results[-1][0]
    context_out = _comp.results[-1][1]

    return state_out, context_out


def get_results_manual(
        normalize,
        state_vec,
        context_vec,
        state_weight,
        context_weight,
        softmax_temperature,
        mem_init,
        softmax_threshold,
        verbose=False,

):
    mem_states = state_vec[:-1] + [[mem_init] * len(state_vec[0])]
    mem_context = context_vec[:-1] + [[mem_init] * len(context_vec[0])]

    if verbose:
        print('Memory')
        print(mem_states)
        print(mem_context)

    if normalize:
        mem_states_tmp = normalized(np.array(mem_states))
        mem_context_tmp = normalized(np.array(mem_context))
    else:
        mem_states_tmp = np.array(mem_states)
        mem_context_tmp = np.array(mem_context)

    if verbose:
        print('Memory (normalized)')
        print(mem_states_tmp)
        print(mem_context_tmp)

    query_state = state_vec[-1]
    query_context = context_vec[-1]

    if verbose:
        print()
        print('Query')
        print(query_state)
        print(query_context)

    if normalize:
        query_state_tmp = normalized(np.array(query_state))
        query_context_tmp = normalized(np.array(query_context))
    else:
        query_state_tmp = np.array(query_state)
        query_context_tmp = np.array(query_context)

    if verbose:
        print('Query (normalized)')
        print(query_state_tmp)
        print(query_context_tmp)

    weights = (state_weight * np.sum(mem_states_tmp * query_state_tmp, axis=-1) +
               context_weight * np.sum(mem_context_tmp * query_context_tmp, axis=-1)) / softmax_temperature

    if verbose:
        print()
        print('Weights')
        print(weights)

    weights_sm = safe_softmax(weights, threshold=softmax_threshold)
    if verbose:
        print('Weights (softmax)')
        print(weights_sm)

    w = weights_sm[:, None]

    state_ret = np.sum(w * mem_states, axis=0)
    context_ret = np.sum(w * mem_context, axis=0)

    return state_ret, context_ret


def main():
    import random

    normalize = random.choice([True, False])

    mem_capacity = random.randint(3, 100)

    mem_init = random.random() * .1 + 1e-12

    softmax_temperature = random.random() + 1e-12

    softmax_threshold = random.random() + 1e-12

    dim_states = random.randint(2, 10)
    dim_context = random.randint(2, 10)

    state_vec = []
    context_vec = []

    for _ in range(mem_capacity):
        _state_vec = []
        for _ in range(dim_states):
            _state_vec.append(random.random())
        _context_vec = []
        for _ in range(dim_context):
            _context_vec.append(random.random())
        state_vec.append(_state_vec)
        context_vec.append(_context_vec)

    state_weight =random.random()
    context_weight =random.random()

    assert len(state_vec) == len(context_vec)

    state_out_pnl, context_out_pnl = get_results_pnl(
        normalize=normalize,
        state_vec=state_vec,
        context_vec=context_vec,
        state_weight=state_weight,
        context_weight=context_weight,
        softmax_temperature=softmax_temperature,
        mem_init=mem_init,
        softmax_threshold=softmax_threshold,
        dim_states=dim_states,
        dim_context=dim_context,
        capacity=mem_capacity
    )

    state_out_man, context_out_man = get_results_manual(
        normalize=normalize,
        state_vec=state_vec,
        context_vec=context_vec,
        state_weight=state_weight,
        context_weight=context_weight,
        softmax_temperature=softmax_temperature,
        mem_init=mem_init,
        softmax_threshold=softmax_threshold,
        verbose=False,
    )
    print('STATES')
    print(state_out_man)
    print(state_out_pnl)
    print('CONTEXTS')
    print(context_out_man)
    print(context_out_pnl)
    assert np.allclose(state_out_pnl, state_out_man, atol=1e-24)
    assert np.allclose(context_out_pnl, context_out_man, atol=1e-24)


if __name__ == '__main__':
    main()
