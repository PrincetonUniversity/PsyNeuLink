import numpy as np
import psyneulink as pnl

# ---------- helper ----------
def normalize_rows(x):
    x = np.array(x, dtype=float)
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return x / norms


def py_retrieval(mem_s, mem_c, mem_t,
                 q_s, q_c, q_t,
                 ws, wc, wt,
                 normalize=True):

    mem_s = np.array(mem_s)
    mem_c = np.array(mem_c)
    mem_t = np.array(mem_t)

    q_s = np.array(q_s)
    q_c = np.array(q_c)
    q_t = np.array(q_t)

    if normalize:
        mem_s_n = normalize_rows(mem_s)
        mem_c_n = normalize_rows(mem_c)
        mem_t_n = normalize_rows(mem_t)

        q_s_n = q_s / np.linalg.norm(q_s) if np.linalg.norm(q_s) else q_s
        q_c_n = q_c / np.linalg.norm(q_c) if np.linalg.norm(q_c) else q_c
        q_t_n = q_t / np.linalg.norm(q_t) if np.linalg.norm(q_t) else q_t
    else:
        mem_s_n = mem_s
        mem_c_n = mem_c
        mem_t_n = mem_t

        q_s_n = q_s
        q_c_n = q_c
        q_t_n = q_t

    # compute cosine scores
    scores = (
        ws * mem_s_n.dot(q_s_n) +
        wc * mem_c_n.dot(q_c_n) +
        wt * mem_t_n.dot(q_t_n)
    )

    # safe softmax
    probs = np.exp(scores - np.max(scores))
    probs /= probs.sum()

    # weighted sums
    return (
        probs @ mem_s,
        probs @ mem_c,
        probs @ mem_t
    )


# =============  MINIMAL FAILING EXAMPLE ============= #

mem_state = [
    [1., 0., 0.],
    [0., 1., 0.]
]

mem_context = [
    [1., 2., 0.],
    [0., 1., 3.]
]

mem_time = [
    [0., 0., 1.],
    [1., 0., 0.]
]

query_state = [0., 0., 1.]
query_context = [1., 0., 1.]
query_time = [0., 1., 0.]

# weights
ws = 1.
wc = 1.
wt = 1.

# ------------ Python version (correct) ----------------------

py_s, py_c, py_t = py_retrieval(
    mem_state, mem_context, mem_time,
    query_state, query_context, query_time,
    1, 0, 1,
    normalize=True
)

print("PYTHON STATE:", py_s)
print("PYTHON CONTEXT:", py_c)
print("PYTHON TIME:", py_t)

# ------------ PNL version (BUG when 3rd field added) -----------

state_in = pnl.ProcessingMechanism(input_shapes=3)
context_in = pnl.ProcessingMechanism(input_shapes=3)
time_in = pnl.ProcessingMechanism(input_shapes=3)

em = pnl.EMComposition(
    name='EM',
    memory_template=[
        [0, 0, 0],  # STATE
        [0, 0, 0],  # CONTEXT
        [0, 0, 0],  # TIME
    ],
    memory_fill=0.01,
    memory_capacity=2,
    #softmax_threshold=1e-24,
    softmax_threshold=None,
    fields={
        'STATE':   {pnl.FIELD_WEIGHT: ws, pnl.LEARN_FIELD_WEIGHT: False, pnl.TARGET_FIELD: False},
        'CONTEXT': {pnl.FIELD_WEIGHT: wc, pnl.LEARN_FIELD_WEIGHT: False, pnl.TARGET_FIELD: False},
        'TIME':    {pnl.FIELD_WEIGHT: wt, pnl.LEARN_FIELD_WEIGHT: False, pnl.TARGET_FIELD: False},
    },
    normalize_memories=True,   # <-- Bug activates here
    normalize_field_weights=False,
    memory_decay_rate=0,
    enable_learning=False
)

# Control: override weights
control_signals = [
    (pnl.SLOPE, em.nodes['STATE [WEIGHT]']),
    (pnl.SLOPE, em.nodes['CONTEXT [WEIGHT]']),
    (pnl.SLOPE, em.nodes['TIME [WEIGHT]']),
]

_control_signal = [[1], [0], [0]]

def control_function(_):
    return _control_signal

control = pnl.ControlMechanism(
    monitor_for_control=[state_in],
    function=control_function,
    control=control_signals
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

time_to_em = [
        time_in,
        pnl.MappingProjection(
            sender=time_in,
            receiver=em.nodes['TIME [QUERY]'],
            matrix=pnl.IDENTITY_MATRIX),
        em
    ]
# pathways
comp = pnl.Composition(pathways=[
    state_to_em,
    context_to_em,
    time_to_em
])

comp.add_nodes([control])

inputs = {
    state_in: mem_state + [query_state],
    context_in: mem_context + [query_context],
    time_in: mem_time + [query_time]
}

comp.run(inputs)

pnl_s = comp.results[-1][0]
pnl_c = comp.results[-1][1]
pnl_t = comp.results[-1][2]

print("\nPNL STATE:", pnl_s)
print("PNL CONTEXT:", pnl_c)
print("PNL TIME:", pnl_t)
#
# print("\nSTATE match?", np.allclose(py_s, pnl_s))
# print("CONTEXT match?", np.allclose(py_c, pnl_c))
# print("TIME match?", np.allclose(py_t, pnl_t))
