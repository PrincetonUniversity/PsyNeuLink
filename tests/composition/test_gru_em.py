"""Tests for an isolated ``GRUComposition`` + ``EMComposition`` pairing trained with
PyTorch, in both ``full_sequence_mode`` (against a faithful pure-PyTorch reference)
and ordinary trial-by-trial mode.

What this guards
----------------
This is the minimal "GRU + episodic memory" core of the EGO/ESBN family (cf.
``tests/models/test_giallanza_ego.py``),
isolated to a single recurrent ``GRUComposition`` feeding an ``EMComposition``:

  * At every step *t* of an input sequence the GRU advances its hidden state and
    its (normalized) output is used as one of two episodic-memory query keys (the
    other being the previous input). EM retrieves a softmax-weighted value (the
    predicted next state). EM then stores the step's ``(value, prev, context)``
    triplet -- read-before-write.

The two learning regimes tested:

  * ``full_sequence_mode=True``: the whole sequence is one trial, processed one
    step at a time, and learning backpropagates through the *entire* sequence
    (BPTT) before a single optimizer step -- exactly like a plain ``torch.nn.GRU``
    run over the sequence with full backprop-through-time.
  * ``full_sequence_mode=False`` (ordinary trial-by-trial mode): each sequence
    element is its own trial with its own backward pass and optimizer step
    (1-step truncated BPTT). Note the scheduling difference: here EM must read
    the *current* trial's GRU output (no lagged schedule), because a lagged
    schedule would make each trial's loss depend on the previous trial's graph,
    whose weights were already modified by the previous optimizer step.

Because ``GRUComposition`` is implemented with ``torch.nn.GRU`` internally, a
faithful reference is built from ``nn.GRU`` directly, and the two are seeded with
identical initial GRU weights (via
``PytorchGRUCompositionWrapper.get_parameters_from_torch_gru`` /
``GRUComposition.set_weights``) so their trajectories can be compared exactly.

The tests assert that the PsyNeuLink model and the PyTorch reference produce the
same per-step predictions, and -- crucially -- the same *learned* GRU weights
after backprop. The learnable-GRU cases are the regression guard for gradient
flow through a GRU paired with EM: the GRU must (a) carry its hidden state across
the sequence and (b) receive correct gradients in both modes.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import psyneulink as pnl  # noqa: E402
from psyneulink import (  # noqa: E402
    AutodiffComposition,
    BeforeNodes,
    CONTEXT,
    CPU,
    EMComposition,
    ExecutionMode,
    FIELD_WEIGHT,
    GRUComposition,
    IDENTITY_MATRIX,
    LAST,
    LEARN_FIELD_WEIGHT,
    Loss,
    MappingProjection,
    Normalize,
    ProcessingMechanism,
    RUN,
    TARGET_FIELD,
)

# =============================================================================
# Shared configuration
# =============================================================================
STATE_SIZE = 5
HIDDEN_SIZE = 5  # == STATE_SIZE so the GRU output identity-maps into the EM query
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


# =============================================================================
# Pure-PyTorch reference
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


def _normalize(x, eps=1e-12):
    """L2-normalize the last dim (matches PsyNeuLink ``Normalize``)."""
    norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


class _EMModule(nn.Module):
    """Key-value episodic memory; torch analogue of the EMComposition used here.

    Queried with ``(previous_state, context)``; returns a softmax-weighted sum of
    stored values (the next states). One row is written per step.
    """

    def __init__(self, state_dim, gain, threshold, fill, capacity):
        super().__init__()
        self.gain = gain
        self.threshold = threshold
        self.prev_keys = torch.full((capacity, state_dim), fill, dtype=torch.float64)
        self.ctx_keys = torch.full((capacity, state_dim), fill, dtype=torch.float64)
        self.values = torch.full((capacity, state_dim), fill, dtype=torch.float64)
        self.index = 0

    def retrieve(self, prev, ctx):
        # Read a snapshot (clone) of memory so a subsequent in-place store does not invalidate the autograd
        # graph for this retrieval (this mirrors EMComposition, which clones memory on read). The stored keys
        # are detached, so gradient flows only through the query (prev, ctx) -- not through stored memory.
        prev_keys, ctx_keys, values = self.prev_keys.clone(), self.ctx_keys.clone(), self.values.clone()
        match = self.gain * (torch.mv(prev_keys, prev) + torch.mv(ctx_keys, ctx))
        weights = _safe_softmax(match, self.threshold)
        return torch.clamp(torch.mv(values.t(), weights), min=0.0, max=1.0)

    def store(self, value, prev, ctx):
        self.prev_keys[self.index] = prev.detach()
        self.ctx_keys[self.index] = ctx.detach()
        self.values[self.index] = value.detach()
        self.index += 1


def run_torch_reference(sequence, init_weights, learn_gru, learning_rate=LEARNING_RATE, epochs=1,
                        loss_mode="last"):
    """Run the full-sequence (BPTT) GRU+EM reference; return ``(predictions, losses, learned_weights)``.

    ``predictions`` has shape ``(epochs * T, STATE_SIZE)``; ``losses`` is the list
    of per-step losses; ``learned_weights`` is the GRU weight tuple in PNL format.

    The GRU processes the whole sequence in one ``nn.GRU`` call (so the hidden
    state is carried across steps and gradients flow through the full sequence).
    The context used at step *t* is ``normalize(h_{t-1})`` (lagged one step,
    because in the PNL schedule the GRU updates at the end of each step), and the
    previous-state query is ``x_{t-1}`` -- read-before-write next-state prediction.
    """
    seq = [torch.tensor(s, dtype=torch.float64) for s in sequence]
    T = len(seq)

    gru = nn.GRU(input_size=STATE_SIZE, hidden_size=HIDDEN_SIZE, bias=False, batch_first=True).double()
    _load_gru_weights(gru, init_weights)
    optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate) if learn_gru else None
    loss_fn = nn.BCELoss()

    all_preds, all_losses = [], []
    for _epoch in range(epochs):
        if optimizer is not None:
            optimizer.zero_grad()
        em = _EMModule(STATE_SIZE, SOFTMAX_GAIN, SOFTMAX_THRESHOLD, MEMORY_FILL, MEMORY_CAPACITY)

        x_seq = torch.stack(seq).unsqueeze(0)  # [1, T, S]
        h_all, _ = gru(x_seq)                  # [1, T, H]; h_all[0, k] = GRU state after input k

        prev = torch.zeros(STATE_SIZE, dtype=torch.float64)
        h_prev = torch.zeros(HIDDEN_SIZE, dtype=torch.float64)  # lagged GRU state
        losses = []
        for t in range(T):
            ctx = _normalize(h_prev)
            pred = em.retrieve(prev, ctx)             # read
            all_preds.append(pred.detach().numpy().copy())
            losses.append(loss_fn(torch.clamp(pred, 1e-7, 1 - 1e-7), seq[t]))
            em.store(seq[t], prev, ctx)               # write (read-before-write)
            prev = seq[t]
            h_prev = h_all[0, t]                      # advance carried GRU state

        total = losses[-1] if loss_mode == "last" else torch.stack(losses).sum()
        if optimizer is not None:
            total.backward()
            optimizer.step()
        all_losses.extend(float(l.detach()) for l in losses)

    return np.stack(all_preds), np.array(all_losses), _gru_weights_as_pnl(gru)


def run_torch_reference_trialwise(sequence, init_weights, learn_gru, learning_rate=LEARNING_RATE,
                                  epochs=1):
    """Run the trial-by-trial (non-``full_sequence_mode``) GRU+EM reference.

    Each sequence element is its own trial: the GRU advances one step, EM reads using the *current*
    GRU output as context, and the trial's loss is backpropagated and applied immediately (1-step
    truncated BPTT).

    Returns ``(predictions, learned_weights)``.
    """
    seq = [torch.tensor(s, dtype=torch.float64) for s in sequence]

    gru = nn.GRU(input_size=STATE_SIZE, hidden_size=HIDDEN_SIZE, bias=False, batch_first=True).double()
    _load_gru_weights(gru, init_weights)
    optimizer = torch.optim.SGD(gru.parameters(), lr=learning_rate) if learn_gru else None
    loss_fn = nn.BCELoss()

    all_preds = []
    for _epoch in range(epochs):
        em = _EMModule(STATE_SIZE, SOFTMAX_GAIN, SOFTMAX_THRESHOLD, MEMORY_FILL, MEMORY_CAPACITY)
        prev = torch.zeros(STATE_SIZE, dtype=torch.float64)
        h = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float64)
        for x in seq:
            if optimizer is not None:
                optimizer.zero_grad()
            _, h_new = gru(x.view(1, 1, STATE_SIZE), h)
            ctx = _normalize(h_new.view(-1))
            pred = em.retrieve(prev, ctx)
            all_preds.append(pred.detach().numpy().copy())
            loss = loss_fn(torch.clamp(pred, 1e-7, 1 - 1e-7), x)
            em.store(x, prev, ctx)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            prev = x
            # PNL re-seeds the hidden state each trial from the cached HIDDEN_LAYER node value, which is
            # not updated within a run -- so the GRU state effectively resets to zero every trial.
            h = torch.zeros(1, 1, HIDDEN_SIZE, dtype=torch.float64)
    return np.stack(all_preds), _gru_weights_as_pnl(gru)


def _load_gru_weights(torch_gru, init_weights):
    """Set ``torch_gru`` parameters from a PNL-style ``(weights, biases)`` tuple."""
    weights, _biases = init_weights
    wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn = (
        torch.tensor(np.asarray(w), dtype=torch.float64) for w in weights
    )
    with torch.no_grad():
        torch_gru.weight_ih_l0.copy_(torch.cat([wts_ir.T, wts_iu.T, wts_in.T], dim=0))
        torch_gru.weight_hh_l0.copy_(torch.cat([wts_hr.T, wts_hu.T, wts_hn.T], dim=0))


def _gru_weights_as_pnl(torch_gru):
    """Return GRU weights as the PNL ``(wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn)`` tuple."""
    return pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_gru)[0]


# =============================================================================
# PsyNeuLink model
# =============================================================================

def build_pnl_model(enable_gru_learning, full_sequence_mode=True):
    """Build the GRU+EM ``AutodiffComposition``.

    The EM read schedule differs between the two modes:

    * ``full_sequence_mode=True``: EM reads the *lagged* GRU output (the GRU and
      ``PREVIOUS STATE`` update after EM), matching the BPTT reference where the
      context at step *t* is ``normalize(h_{t-1})``. The whole sequence is one
      trial and one backward pass.
    * ``full_sequence_mode=False``: each element is its own trial with its own
      backward pass and optimizer step, so EM must read the *current* trial's GRU
      output. (A lagged schedule would make each trial's loss depend on the
      previous trial's graph, whose weights the previous optimizer step already
      modified in place -- an autograd error.)
    """
    state_input = ProcessingMechanism(name="STATE", input_shapes=STATE_SIZE)
    previous_state = ProcessingMechanism(name="PREVIOUS STATE", input_shapes=STATE_SIZE)
    gru = GRUComposition(name="CONTEXT_GRU", input_size=STATE_SIZE, hidden_size=HIDDEN_SIZE,
                         bias=False, enable_learning=enable_gru_learning, learning_rate=LEARNING_RATE)
    context_norm = ProcessingMechanism(name=CONTEXT + "[normalized]", input_shapes=HIDDEN_SIZE,
                                        function=Normalize())
    em = EMComposition(
        name="EM",
        memory_template=[[0] * STATE_SIZE, [0] * STATE_SIZE, [0] * STATE_SIZE],
        memory_fill=MEMORY_FILL, memory_capacity=MEMORY_CAPACITY, memory_decay_rate=0,
        softmax_gain=SOFTMAX_GAIN, softmax_threshold=SOFTMAX_THRESHOLD,
        fields={
            state_input.name: {FIELD_WEIGHT: None, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            previous_state.name: {FIELD_WEIGHT: 1.0, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            gru.name: {FIELD_WEIGHT: 1.0, LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        },
        normalize_field_weights=False, normalize_memories=False, concatenate_queries=False,
        enable_learning=False, device=CPU, store_on_optimization="last",
    )
    prediction = ProcessingMechanism(name="PREDICTION", input_shapes=STATE_SIZE)
    q, v, r = " [QUERY]", " [VALUE]", " [RETRIEVED]"

    comp = AutodiffComposition(
        [
            [state_input, MappingProjection(matrix=IDENTITY_MATRIX, learnable=False), previous_state],
            [state_input, gru, MappingProjection(matrix=IDENTITY_MATRIX, learnable=False), context_norm],
            [state_input, MappingProjection(sender=state_input, receiver=em.nodes[state_input.name + v],
                                            matrix=IDENTITY_MATRIX, learnable=False), em],
            [previous_state, MappingProjection(sender=previous_state, receiver=em.nodes[previous_state.name + q],
                                               matrix=IDENTITY_MATRIX, learnable=False), em],
            [context_norm, MappingProjection(sender=context_norm, receiver=em.nodes[gru.name + q],
                                             matrix=IDENTITY_MATRIX, learnable=False), em,
             MappingProjection(sender=em.nodes[state_input.name + r], receiver=prediction,
                               matrix=IDENTITY_MATRIX, learnable=False), prediction],
        ],
        full_sequence_mode=full_sequence_mode, learning_rate=LEARNING_RATE,
        loss_spec=Loss.BINARY_CROSS_ENTROPY,
        targets=(prediction, state_input),
        execute_in_additional_optimizations={previous_state: LAST},
        name="GRU_EM", device=CPU,
    )
    comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)
    if full_sequence_mode:
        comp.scheduler.add_condition(em, BeforeNodes(previous_state, gru))
        comp.scheduler.add_condition(context_norm, BeforeNodes(em))
        comp.scheduler.add_condition(prediction, BeforeNodes(previous_state, gru))
    else:
        comp.scheduler.add_condition(em, BeforeNodes(previous_state))
        comp.scheduler.add_condition(prediction, BeforeNodes(previous_state))
    return comp, state_input, gru


def _seed_and_build(enable_gru_learning, seed=0, full_sequence_mode=True):
    """Seed a torch GRU, build a PNL model initialized with the same GRU weights.

    Returns ``(comp, state_input, gru, init_weights)``.
    """
    torch.manual_seed(seed)
    seed_gru = nn.GRU(STATE_SIZE, HIDDEN_SIZE, bias=False, batch_first=True).double()
    init_weights = pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(seed_gru)
    comp, state_input, gru = build_pnl_model(enable_gru_learning, full_sequence_mode)
    gru.set_weights(*init_weights)
    return comp, state_input, gru, init_weights


def run_pnl_model(comp, state_input, epochs=1, capture_predictions=False, full_sequence_mode=True):
    """Train the PNL GRU+EM model on the sequence; return ``(per_step_predictions, losses)``.

    In ``full_sequence_mode`` the whole sequence must be a *single* trial -- a list with one per-trial
    dict whose value is the multi-step sequence -- so that ``full_sequence_mode`` steps through its
    elements as time steps. In ordinary (non-sequence) mode, ``{state_input: SEQUENCE}`` is passed so
    each element is its own trial.

    In ``full_sequence_mode`` the sequence is one trial, so ``comp.results`` retains only one row; to
    compare the model's per-element predictions we capture the PREDICTION node's output on each of its
    executions (one per sequence element per epoch). In non-sequence mode the per-trial PREDICTION
    outputs are read from ``comp.results`` directly.
    """
    captured = []
    if capture_predictions and full_sequence_mode:
        import psyneulink.library.compositions.pytorchwrappers as _pw
        original_execute = _pw.PytorchMechanismWrapper.execute

        def _hooked_execute(self, *args, **kwargs):
            out = original_execute(self, *args, **kwargs)
            if getattr(self.mechanism, "name", "") == "PREDICTION":
                captured.append(np.asarray(self.output.detach().cpu().numpy()).reshape(-1).copy())
            return out

        _pw.PytorchMechanismWrapper.execute = _hooked_execute
    try:
        inputs = [{state_input: SEQUENCE}] if full_sequence_mode else {state_input: SEQUENCE}
        comp.learn(
            inputs=inputs, epochs=epochs, execution_mode=ExecutionMode.PyTorch,
            synch_projection_matrices_with_torch=RUN, synch_node_values_with_torch=RUN,
            synch_results_with_torch=RUN, minibatch_size=1, optimizations_per_minibatch=1,
        )
    finally:
        if capture_predictions and full_sequence_mode:
            _pw.PytorchMechanismWrapper.execute = original_execute
    losses = np.array([float(np.asarray(l).squeeze()) for l in comp.torch_losses])
    if not capture_predictions:
        predictions = None
    elif full_sequence_mode:
        predictions = np.stack(captured)
    else:
        predictions = _predictions_from_results(comp.results)
    return predictions, losses


def _predictions_from_results(results):
    """Extract the per-trial PREDICTION outputs (column 2) from ``comp.results``."""
    predictions = []
    for row in results:
        cell = row[2]
        if isinstance(cell, torch.Tensor):
            cell = cell.detach().cpu().numpy()
        predictions.append(np.asarray(cell, dtype=float).reshape(-1))
    return np.stack(predictions)


def _pnl_learned_gru_weights(gru):
    """Learned GRU weights read from the *live* torch module, in PNL
    ``(wts_ir, wts_iu, wts_in, wts_hr, wts_hu, wts_hn)`` format.

    Note: ``GRUComposition.get_weights()`` reads the PNL Projection matrices, which are not synchronized
    back from the torch module after learning, so it reports the *initial* weights; the torch module is
    the source of truth in PyTorch mode.
    """
    torch_gru = gru.pytorch_representation.torch_gru
    return pnl.PytorchGRUCompositionWrapper.get_parameters_from_torch_gru(torch_gru)[0]


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.pytorch
@pytest.mark.composition
def test_gru_em_forward_matches_torch():
    """Forward pass: PNL GRU+EM per-step predictions == torch reference.

    Exercises, in ``full_sequence_mode``, both (a) the GRU carrying its hidden state across the
    sequence elements and (b) the per-step EM read-before-write retrieval (each element retrieves
    against the entries stored by preceding elements).
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, _gru, init_w = _seed_and_build(enable_gru_learning=False)
        pnl_preds, _ = run_pnl_model(comp, state_input, epochs=1, capture_predictions=True)
        ref_preds, _, _ = run_torch_reference(SEQUENCE, init_w, learn_gru=False, epochs=1)

        assert pnl_preds.shape == ref_preds.shape == (len(SEQUENCE), STATE_SIZE)
        assert np.allclose(pnl_preds, ref_preds, atol=TOLERANCE), (
            f"forward per-step predictions differ; max abs diff = {np.max(np.abs(pnl_preds - ref_preds))}"
        )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.pytorch
@pytest.mark.composition
def test_gru_em_bptt_weights_match_torch():
    """Backprop-through-time: the GRU weights learned via one full-sequence backward pass match the
    torch reference exactly.

    The whole sequence is processed as one trial; the loss is taken at the final element and
    backpropagated through the *entire* sequence (full BPTT through the GRU and through the EM query
    path). This is the core guard that gradients flow correctly through a GRU paired with EM in
    ``full_sequence_mode``. Weights are read from the live torch module (the source of truth in
    PyTorch mode), not ``GRUComposition.get_weights()``.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, gru, init_w = _seed_and_build(enable_gru_learning=True)
        run_pnl_model(comp, state_input, epochs=1)
        pnl_weights = _pnl_learned_gru_weights(gru)
        _, _, ref_weights = run_torch_reference(SEQUENCE, init_w, learn_gru=True, epochs=1, loss_mode="last")

        for name, pnl_w, ref_w in zip(
            ("wts_ir", "wts_iu", "wts_in", "wts_hr", "wts_hu", "wts_hn"), pnl_weights, ref_weights
        ):
            assert np.allclose(np.asarray(pnl_w), np.asarray(ref_w), atol=TOLERANCE), (
                f"learned GRU weight '{name}' differs; "
                f"max abs diff = {np.max(np.abs(np.asarray(pnl_w) - np.asarray(ref_w)))}"
            )
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.pytorch
@pytest.mark.composition
def test_gru_em_gru_learns_in_full_sequence_mode():
    """Regression guard for the gradient-flow bug: with learning, the GRU receives gradient (its
    weights move) and the loss decreases.

    (Previously the GRU received no gradient in ``full_sequence_mode`` -- its hidden state was
    re-seeded from the stale cached node value on every sequence element, severing the recurrence --
    so its weights never updated and all outputs were identical.)
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, gru, init_w = _seed_and_build(enable_gru_learning=True)
        _, losses = run_pnl_model(comp, state_input, epochs=4)
        learned = _pnl_learned_gru_weights(gru)

        max_change = max(float(np.max(np.abs(np.asarray(lw) - np.asarray(iw))))
                         for lw, iw in zip(learned, init_w[0]))
        assert max_change > 1e-6, (
            f"GRU weights did not change with learning (max change {max_change}); "
            "gradients are not flowing through the GRU in full_sequence_mode"
        )
        assert losses[-1] < losses[0], f"loss did not decrease over learning: {losses}"
    finally:
        torch.set_default_dtype(old_dtype)


@pytest.mark.pytorch
@pytest.mark.composition
def test_gru_em_trialwise_matches_torch():
    """Trial-by-trial mode (``full_sequence_mode=False``): per-trial predictions and the learned GRU
    weights match the stepwise torch reference exactly.

    Each sequence element is its own trial with its own backward pass and optimizer step (1-step
    truncated BPTT), so this guards gradient flow through the GRU paired with EM in ordinary
    (non-sequence) PyTorch learning mode.
    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        comp, state_input, gru, init_w = _seed_and_build(enable_gru_learning=True, full_sequence_mode=False)
        pnl_preds, _ = run_pnl_model(comp, state_input, epochs=1, capture_predictions=True,
                                     full_sequence_mode=False)
        pnl_weights = _pnl_learned_gru_weights(gru)
        ref_preds, ref_weights = run_torch_reference_trialwise(SEQUENCE, init_w, learn_gru=True, epochs=1)

        assert pnl_preds.shape == ref_preds.shape == (len(SEQUENCE), STATE_SIZE)
        assert np.allclose(pnl_preds, ref_preds, atol=TOLERANCE), (
            f"trialwise per-trial predictions differ; max abs diff = {np.max(np.abs(pnl_preds - ref_preds))}"
        )
        for name, pnl_w, ref_w in zip(
            ("wts_ir", "wts_iu", "wts_in", "wts_hr", "wts_hu", "wts_hn"), pnl_weights, ref_weights
        ):
            assert np.allclose(np.asarray(pnl_w), np.asarray(ref_w), atol=TOLERANCE), (
                f"trialwise learned GRU weight '{name}' differs; "
                f"max abs diff = {np.max(np.abs(np.asarray(pnl_w) - np.asarray(ref_w)))}"
            )
        max_change = max(float(np.max(np.abs(np.asarray(lw) - np.asarray(iw))))
                         for lw, iw in zip(pnl_weights, init_w[0]))
        assert max_change > 1e-6, (
            f"GRU weights did not change with learning (max change {max_change}); "
            "gradients are not flowing through the GRU in trial-by-trial mode"
        )
    finally:
        torch.set_default_dtype(old_dtype)


# =============================================================================
# Self-validation harness (run directly, not under pytest)
# =============================================================================
if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    comp, state_input, gru, init_w = _seed_and_build(enable_gru_learning=False)
    pnl_preds, _ = run_pnl_model(comp, state_input, epochs=1, capture_predictions=True)
    ref_preds, _, _ = run_torch_reference(SEQUENCE, init_w, learn_gru=False, epochs=1)
    print("PNL predictions:\n", np.round(pnl_preds, 5))
    print("REF predictions:\n", np.round(ref_preds, 5))
    print("max |pred diff| =", np.max(np.abs(pnl_preds - ref_preds)))
