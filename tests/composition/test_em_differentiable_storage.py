"""Tests for the `differentiable_storage <EMComposition.differentiable_storage>` option of ``EMComposition``.

The minimal model here is the core of the Emergent Symbol Binding Network (ESBN; Webb et
al., 2021), stripped of the perceptual encoder and recurrent controller: two *learnable*
projections produce a query and a key from the input; the (query, key) pair is **stored**
in episodic memory at every step, and the current query retrieves a weighted sum of the
**stored** keys as the prediction:

    q_t     = x_t @ W_q
    k_t     = x_t @ W_k
    scores  = softmax(gain * M_q @ q_t)   (current query against *stored* queries)
    pred_t  = scores @ M_k                (weighted sum of *stored* keys)
    store (q_t, k_t)                      (read-before-write)

The two weight matrices have crucially different gradient paths:

  * ``W_q`` gets gradient through the *current* query ``q_t`` (the retrieval path) --
    this is the EGO-style path that works with ordinary, non-differentiable storage;
  * ``W_k`` gets gradient **only** through the stored keys: the key it produces does
    nothing except get stored and later retrieved. This is the ESBN-style path that
    exists only with ``differentiable_storage=True``.

The tests assert:

  * with ``differentiable_storage=True``, predictions and *both* learned weight matrices
    match a pure-PyTorch ESBN-style reference exactly (gradient flows through storage);
  * with the default ``differentiable_storage=False``, predictions are identical and
    ``W_q`` still matches the (detached-storage) reference, but ``W_k`` receives no
    gradient at all and never changes.

Both tests run in ``full_sequence_mode`` (whole sequence = one trial, one backward pass),
the only regime in which gradient flow through stored entries is possible: entries must
be stored and retrieved within the same forward pass.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from psyneulink import (  # noqa: E402
    AutodiffComposition,
    CPU,
    EMComposition,
    ExecutionMode,
    FIELD_WEIGHT,
    IDENTITY_MATRIX,
    LEARN_FIELD_WEIGHT,
    Loss,
    MappingProjection,
    ProcessingMechanism,
    RUN,
    TARGET_FIELD,
)

# =============================================================================
# Shared configuration
# =============================================================================
STATE_SIZE = 5
SOFTMAX_GAIN = 5.0
SOFTMAX_THRESHOLD = 0.001
MEMORY_FILL = 0.01
MEMORY_CAPACITY = 10
LEARNING_RATE = 0.1
TOLERANCE = 1e-6

# A short, deterministic one-hot sequence.
SEQUENCE = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
]

# Initial weights, bounded away from 0 so stored entries always out-norm the memory fill
# (keeps the weakest-slot selection sequential, matching the reference).
QUERY_WEIGHTS_INIT = np.random.RandomState(0).uniform(0.5, 1.5, (STATE_SIZE, STATE_SIZE))
KEY_WEIGHTS_INIT = np.random.RandomState(1).uniform(0.5, 1.5, (STATE_SIZE, STATE_SIZE))


# =============================================================================
# Pure-PyTorch ESBN-style reference
# =============================================================================

def _safe_softmax(t, threshold=SOFTMAX_THRESHOLD):
    """Thresholded, numerically-stable softmax (matches ``EMComposition``)."""
    v = t
    if threshold is not None:
        v = torch.where(abs(t) > threshold, v, torch.tensor(-torch.inf, dtype=t.dtype))
    if torch.any(v != -torch.inf):
        v = v - torch.max(v)
    v = torch.exp(v)
    if not v.any():
        return v
    return v / torch.sum(v)


def run_torch_reference(sequence, query_weights_init, key_weights_init, differentiable_storage,
                        learning_rate=LEARNING_RATE):
    """Run the minimal ESBN-style reference; return ``(predictions, learned_W_q, learned_W_k)``.

    One forward pass over the whole sequence, loss on the final element, one backward pass
    and one SGD step (matching ``full_sequence_mode`` with ``epochs=1``). If
    ``differentiable_storage``, the stored (query, key) entries keep their graph; otherwise
    they are detached on store (in which case ``W_k`` gets no gradient at all, and ``W_q``
    gets gradient only through the current-step query).
    """
    seq = [torch.tensor(s, dtype=torch.float64) for s in sequence]
    W_q = torch.tensor(np.asarray(query_weights_init), dtype=torch.float64, requires_grad=True)
    W_k = torch.tensor(np.asarray(key_weights_init), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.SGD([W_q, W_k], lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    mem_q = torch.full((MEMORY_CAPACITY, STATE_SIZE), MEMORY_FILL, dtype=torch.float64)
    mem_k = torch.full((MEMORY_CAPACITY, STATE_SIZE), MEMORY_FILL, dtype=torch.float64)

    optimizer.zero_grad()
    preds, losses = [], []
    for t, x in enumerate(seq):
        q = x @ W_q
        k = x @ W_k
        # Retrieve (read a clone, as EMComposition does, so stores don't invalidate the graph)
        scores = _safe_softmax(SOFTMAX_GAIN * (mem_q.clone() @ q))
        pred = scores @ mem_k.clone()
        preds.append(pred.detach().numpy().copy())
        losses.append(loss_fn(pred, x))
        # Store (read-before-write); sequential slots (fill entries are always the weakest).
        # Out-of-place (clone-and-write), as EMComposition does with differentiable_storage.
        mem_q = mem_q.clone()
        mem_k = mem_k.clone()
        if differentiable_storage:
            mem_q[t] = q
            mem_k[t] = k
        else:
            mem_q[t] = q.detach()
            mem_k[t] = k.detach()

    losses[-1].backward()  # loss on the final element, backpropagated through the sequence
    optimizer.step()
    return np.stack(preds), W_q.detach().numpy().copy(), W_k.detach().numpy().copy()


# =============================================================================
# PsyNeuLink model
# =============================================================================

def build_pnl_model(differentiable_storage):
    """Build the minimal ESBN-style ``AutodiffComposition`` in ``full_sequence_mode``."""
    state_input = ProcessingMechanism(name="STATE", input_shapes=STATE_SIZE)
    em = EMComposition(
        name="EM",
        memory_template=[[0] * STATE_SIZE, [0] * STATE_SIZE],
        memory_fill=MEMORY_FILL, memory_capacity=MEMORY_CAPACITY, memory_decay_rate=0,
        softmax_gain=SOFTMAX_GAIN, softmax_threshold=SOFTMAX_THRESHOLD,
        fields={
            "QUERY": {FIELD_WEIGHT: 1.0, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            "KEY": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        },
        normalize_field_weights=False, normalize_memories=False, concatenate_queries=False,
        enable_learning=False, device=CPU, store_on_optimization="last",
        differentiable_storage=differentiable_storage,
    )
    prediction = ProcessingMechanism(name="PREDICTION", input_shapes=STATE_SIZE)

    query_projection = MappingProjection(sender=state_input, receiver=em.nodes["QUERY [QUERY]"],
                                         matrix=QUERY_WEIGHTS_INIT.copy(), learnable=True)
    key_projection = MappingProjection(sender=state_input, receiver=em.nodes["KEY [VALUE]"],
                                       matrix=KEY_WEIGHTS_INIT.copy(), learnable=True)
    comp = AutodiffComposition(
        [
            [state_input, key_projection, em],
            [state_input, query_projection, em,
             MappingProjection(sender=em.nodes["KEY [RETRIEVED]"], receiver=prediction,
                               matrix=IDENTITY_MATRIX, learnable=False), prediction],
        ],
        full_sequence_mode=True, learning_rate=LEARNING_RATE, loss_spec=Loss.MSE,
        targets=(prediction, state_input),
        name="ESBN_MINIMAL", device=CPU,
    )
    comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    return comp, state_input, query_projection, key_projection


def run_pnl_model(comp, state_input, capture_predictions=True):
    """Train one epoch on the sequence (one trial in ``full_sequence_mode``); return per-step predictions."""
    captured = []
    if capture_predictions:
        import psyneulink.library.compositions.pytorchwrappers as _pw
        original_execute = _pw.PytorchMechanismWrapper.execute

        def _hooked_execute(self, *args, **kwargs):
            out = original_execute(self, *args, **kwargs)
            if getattr(self.mechanism, "name", "") == "PREDICTION":
                captured.append(np.asarray(self.output.detach().cpu().numpy()).reshape(-1).copy())
            return out

        _pw.PytorchMechanismWrapper.execute = _hooked_execute
    try:
        comp.learn(
            inputs=[{state_input: SEQUENCE}], epochs=1, execution_mode=ExecutionMode.PyTorch,
            synch_projection_matrices_with_torch=RUN, synch_node_values_with_torch=RUN,
            synch_results_with_torch=RUN, minibatch_size=1, optimizations_per_minibatch=1,
        )
    finally:
        if capture_predictions:
            _pw.PytorchMechanismWrapper.execute = original_execute
    return np.stack(captured) if capture_predictions else None


def _pnl_learned_matrix(comp, projection):
    """Learned projection matrix, read from the live torch parameter (source of truth in PyTorch mode).

    Projections into the nested EMComposition are rerouted through its input CIM when the model is
    flattened, so ``projections_map`` may key on a rerouted Projection; look the wrapper up by sender
    and receiver names instead.
    """
    sender_name = projection.sender.owner.name
    receiver_name = projection.receiver.owner.name
    for proj, wrapper in comp.pytorch_representation.projections_map.items():
        if (wrapper.sender_wrapper.mechanism.name == sender_name
                and wrapper.receiver_wrapper.mechanism.name == receiver_name):
            return wrapper.matrix.detach().cpu().numpy().copy()
    raise AssertionError(f"no projection wrapper found for {sender_name} -> {receiver_name}")


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.pytorch
@pytest.mark.composition
def test_differentiable_storage_matches_esbn_reference():
    """With ``differentiable_storage=True``, gradients flow through the stored entries: the learned
    key weights (whose *only* gradient path is store-then-retrieve) match the pure-PyTorch
    ESBN-style reference exactly, as do the query weights and per-step predictions.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, query_projection, key_projection = build_pnl_model(differentiable_storage=True)
        pnl_preds = run_pnl_model(comp, state_input)
        pnl_W_q = _pnl_learned_matrix(comp, query_projection)
        pnl_W_k = _pnl_learned_matrix(comp, key_projection)
        ref_preds, ref_W_q, ref_W_k = run_torch_reference(
            SEQUENCE, QUERY_WEIGHTS_INIT, KEY_WEIGHTS_INIT, differentiable_storage=True)

        assert pnl_preds.shape == ref_preds.shape == (len(SEQUENCE), STATE_SIZE)
        assert np.allclose(pnl_preds, ref_preds, atol=TOLERANCE), (
            f"predictions differ; max abs diff = {np.max(np.abs(pnl_preds - ref_preds))}"
        )
        assert np.max(np.abs(pnl_W_k - KEY_WEIGHTS_INIT)) > 1e-6, (
            "key weights did not change; gradient is not flowing through the stored entries"
        )
        assert np.allclose(pnl_W_k, ref_W_k, atol=TOLERANCE), (
            f"learned key weights differ; max abs diff = {np.max(np.abs(pnl_W_k - ref_W_k))}"
        )
        assert np.allclose(pnl_W_q, ref_W_q, atol=TOLERANCE), (
            f"learned query weights differ; max abs diff = {np.max(np.abs(pnl_W_q - ref_W_q))}"
        )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.pytorch
@pytest.mark.composition
def test_default_storage_blocks_gradient_through_stored_entries():
    """With the default ``differentiable_storage=False``, the same model produces identical
    *predictions*, and the query weights still learn (their gradient flows through the current-step
    retrieval query, EGO-style, matching the detached-storage reference) -- but the key weights
    receive no gradient at all: their only path is through the stored entries, which are detached.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, query_projection, key_projection = build_pnl_model(differentiable_storage=False)
        pnl_preds = run_pnl_model(comp, state_input)
        pnl_W_q = _pnl_learned_matrix(comp, query_projection)
        pnl_W_k = _pnl_learned_matrix(comp, key_projection)
        ref_preds, ref_W_q, _ = run_torch_reference(
            SEQUENCE, QUERY_WEIGHTS_INIT, KEY_WEIGHTS_INIT, differentiable_storage=False)

        assert np.allclose(pnl_preds, ref_preds, atol=TOLERANCE), (
            f"predictions differ; max abs diff = {np.max(np.abs(pnl_preds - ref_preds))}"
        )
        assert np.allclose(pnl_W_q, ref_W_q, atol=TOLERANCE), (
            f"learned query weights differ; max abs diff = {np.max(np.abs(pnl_W_q - ref_W_q))}"
        )
        assert np.allclose(pnl_W_k, KEY_WEIGHTS_INIT, atol=1e-12), (
            "key weights changed with non-differentiable storage; "
            f"max abs change = {np.max(np.abs(pnl_W_k - KEY_WEIGHTS_INIT))}"
        )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.pytorch
@pytest.mark.composition
def test_differentiable_storage_warns_without_full_sequence_mode():
    """``differentiable_storage=True`` outside of ``full_sequence_mode`` has no possible effect (each trial is
    its own forward/backward pass, so a stored entry's graph is severed before it can be retrieved); a warning
    should be issued.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        state_input = ProcessingMechanism(name="STATE", input_shapes=STATE_SIZE)
        em = EMComposition(
            name="EM",
            memory_template=[[0] * STATE_SIZE, [0] * STATE_SIZE],
            memory_fill=MEMORY_FILL, memory_capacity=MEMORY_CAPACITY, memory_decay_rate=0,
            softmax_gain=SOFTMAX_GAIN, softmax_threshold=SOFTMAX_THRESHOLD,
            fields={
                "QUERY": {FIELD_WEIGHT: 1.0, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
                "KEY": {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            },
            normalize_field_weights=False, normalize_memories=False, concatenate_queries=False,
            enable_learning=False, device=CPU, store_on_optimization="last",
            differentiable_storage=True,
        )
        prediction = ProcessingMechanism(name="PREDICTION", input_shapes=STATE_SIZE)
        comp = AutodiffComposition(
            [
                [state_input, MappingProjection(sender=state_input, receiver=em.nodes["KEY [VALUE]"],
                                                matrix=KEY_WEIGHTS_INIT.copy(), learnable=True), em],
                [state_input, MappingProjection(sender=state_input, receiver=em.nodes["QUERY [QUERY]"],
                                                matrix=QUERY_WEIGHTS_INIT.copy(), learnable=True), em,
                 MappingProjection(sender=em.nodes["KEY [RETRIEVED]"], receiver=prediction,
                                   matrix=IDENTITY_MATRIX, learnable=False), prediction],
            ],
            full_sequence_mode=False, learning_rate=LEARNING_RATE, loss_spec=Loss.MSE,
            targets=(prediction, state_input),
            name="ESBN_MINIMAL", device=CPU,
        )
        comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)

        with pytest.warns(UserWarning, match="differentiable_storage.*full_sequence_mode"):
            comp.learn(
                inputs={state_input: SEQUENCE}, epochs=1, execution_mode=ExecutionMode.PyTorch,
                minibatch_size=1, optimizations_per_minibatch=1,
            )
    finally:
        torch.set_default_dtype(old_dtype)
