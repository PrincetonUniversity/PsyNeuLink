"""Equivalence test for the EGO model of `Giallanza et al. (2024)
<https://pubmed.ncbi.nlm.nih.gov/38828434/>`_ on the "Coffee Shop World" /
Context-Sensitive World (CSW) task.

What this test guarantees
-------------------------
The EGO model can be expressed two ways:

1. **PsyNeuLink** -- an :class:`AutodiffComposition` that wraps an
   :class:`EMComposition` (episodic memory) together with a recurrent context
   layer and a *learnable* context-mapping projection. Learning is enabled and
   the composition is executed in ``ExecutionMode.PyTorch``.

2. **Pure PyTorch** -- a hand-written reference implementation of the exact same
   architecture (``EMModule`` + ``RecurrentContextModule`` + ``ContextMapping``)
   trained with ``nn.BCELoss`` and plain SGD.

Both implementations are driven with the *same* sequence of CSW states and the
*same* hyper-parameters, with learning active. The test asserts that the two
produce **numerically identical** trial-by-trial episodic-memory predictions
(``np.allclose`` at ``atol=1e-10``).

This is a regression guard: any change to ``EMComposition`` /
``AutodiffComposition`` learning (gradient flow, the extra
per-minibatch optimization steps, projection synchronization with torch, the
scheduling of the EM read/write relative to the context update, etc.) that
silently alters the model's numerical behavior will break this test.

The reference and PsyNeuLink builders below are self-contained ports of the
scratch model under
``Scripts/Models (Under Development)/EGO/Using EMComposition/Coffee Shop World``
so the test has no dependency on that scratch directory.

Note on the architecture
-------------------------
The single learnable component is the ``CONTEXT -> CONTEXT[normalized]``
projection (``ContextMapping`` in torch). It is initialized to the identity
(plus a zero bias) and updated by ``num_optimization_steps`` SGD steps per
trial. Crucially, those optimization steps re-run the EM retrieval with the
updated context weights, so the *learned* projection lives effectively in the
read path of the EM module.
"""

import random

import numpy as np
import pytest

# The whole module is meaningless without torch (both the reference model and
# PsyNeuLink's PyTorch execution mode require it), so skip collection if absent.
torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402  (imported after importorskip on purpose)

from psyneulink import (  # noqa: E402
    AutodiffComposition,
    BeforeNodes,
    CONTEXT,
    CPU,
    EMComposition,
    ExecutionMode,
    FIELD_WEIGHT,
    IDENTITY_MATRIX,
    LAST,
    LEARN_FIELD_WEIGHT,
    Loss,
    MappingProjection,
    Normalize,
    ProcessingMechanism,
    RUN,
    TARGET_FIELD,
    Tanh,
    TransferMechanism,
)
from psyneulink.library.compositions.emcomposition.emcomposition_proj import (  # noqa: E402
    EMComposition_Proj,
)


# =============================================================================
# Hyper-parameters
# =============================================================================
# A single configuration shared by both implementations. ``state_size`` is 11
# because CSW states are one-hot encoded over 11 classes (states 1-10, with
# index 0 unused). ``nr_trials`` is deliberately small to keep the test fast;
# the equivalence holds for any length.
CONFIG = dict(
    paradigm="blocked",
    tolerance=1e-10,
    nr_trials=10,
    seed=42,
    state_size=11,
    integration_rate=0.7,
    state_retrieval_weight=None,
    previous_state_retrieval_weight=1.0,
    context_retrieval_weight=1.0,
    learning_rate=2.0,
    memory_fill=0.01,
    softmax_temperature=0.2,
    softmax_threshold=0.001,
    loss_spec_name="BINARY_CROSS_ENTROPY",
    num_optimization_steps=10,
)


def set_random_seed(seed, **_):
    """Seed every RNG that can influence either model."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # no-op if CUDA is unavailable
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.manual_seed(seed)
        except AttributeError:
            pass
    np.random.seed(seed)
    random.seed(seed)


# =============================================================================
# Task environment: Coffee Shop World / Context-Sensitive World (CSW)
# =============================================================================
# There are two latent contexts, each cued by a distinct start state (9 or 10)
# and each defining a different deterministic transition rule between states.
# The model is never told the context; it must infer it from the sequence.

def _one_hot_encode(labels, num_classes):
    """One-hot encode integer labels as a float64 tensor."""
    return torch.tensor(
        (np.arange(num_classes) == labels[..., None]).astype(float),
        dtype=torch.float64,
    )


def _gen_context1():
    """Context 1 (cued by state 9): transition rule "+2"."""
    states = [9, random.choice([1, 2])]
    for _ in range(3):
        states.append(states[-1] + 2)
    return states


def _gen_context2():
    """Context 2 (cued by state 10): "+1" if even else "+3"."""
    states = [10, random.choice([1, 2])]
    for _ in range(3):
        if states[-1] % 2 == 0:
            states.append(states[-1] + 1)
        else:
            states.append(states[-1] + 3)
    return states


def _gen_run(n_samples_per_context, contexts_to_load):
    all_trials = []
    for i, context in enumerate(contexts_to_load):
        for _ in range(n_samples_per_context[i]):
            all_trials.extend(_gen_context1() if context == 0 else _gen_context2())
    xs = _one_hot_encode(np.array(all_trials), 11)
    return xs.reshape((-1, 11))


def get_state_sequence(paradigm, **_):
    """Generate the full CSW state sequence for ``paradigm``.

    ``blocked``: the two contexts are presented in long blocks.
    ``interleaved``: the two contexts alternate trial-by-trial.
    """
    if paradigm == "blocked":
        contexts_to_load = [0, 1, 0, 1] + [random.randint(0, 1) for _ in range(40)]
        n_samples_per_context = [40, 40, 40, 40] + [1] * 40
    elif paradigm == "interleaved":
        contexts_to_load = [0, 1] * 80 + [random.randint(0, 1) for _ in range(40)]
        n_samples_per_context = [1] * 160 + [1] * 40
    else:
        raise ValueError(f"Unknown paradigm: {paradigm!r}")
    return _gen_run(n_samples_per_context, contexts_to_load)


# =============================================================================
# Pure-PyTorch reference implementation
# =============================================================================

def _safe_softmax(t, threshold=0.01):
    """Softmax that masks values below ``threshold`` and is numerically stable.

    Mirrors ``EMComposition``'s thresholded softmax so the two retrievals match.
    """
    v = t
    if threshold is not None:
        v = torch.where(abs(t) > threshold, v, torch.tensor(-torch.inf, device=t.device))
    if torch.any(v != -torch.inf):
        v = v - torch.max(v)
    v = torch.exp(v)
    if not v.any():
        return v
    return v / torch.sum(v)


def _normalize(x, eps=1e-12):
    """L2-normalize the last dimension (matches PsyNeuLink ``Normalize``)."""
    norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    return x / torch.clamp(norm, min=eps)


class EMModule(nn.Module):
    """Key-value episodic memory.

    Queried with a (previous-state, context) pair, it returns a softmax-weighted
    sum of stored values (next states). Keys/values are written one row per
    trial. This is the torch analogue of ``EMComposition``.
    """

    def __init__(self, state_dim, temperature, softmax_threshold,
                 memory_fill=0.001, memory_fill_n=None):
        super().__init__()
        self.temperature = temperature
        self.softmax_threshold = softmax_threshold
        self.index = 0
        self.state_keys = self.context_keys = self.values = None
        self._initialize_memories(memory_fill_n, state_dim, memory_fill)

    def _initialize_memories(self, n, length, fill=0.001):
        self.state_keys = torch.full((n, length), fill)
        self.context_keys = torch.full((n, length), fill)
        self.values = torch.full((n, length), fill)
        self.index = 0

    def get_match_weights(self, state, context):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        state_match = torch.einsum("b a, c a -> c b", self.state_keys, state) / self.temperature
        if len(context.shape) == 1:
            context = context.unsqueeze(0)
        context_match = torch.einsum("b a, c a -> c b", self.context_keys, context) / self.temperature
        return state_match + context_match

    def forward(self, state, context):
        match_weights = self.get_match_weights(state, context)
        matches = torch.einsum(
            "a b, c a -> c b", self.values,
            _safe_softmax(match_weights, self.softmax_threshold),
        )
        return torch.clamp(matches, min=0, max=1)

    def write(self, state_key, context_key, value):
        self.state_keys[self.index] = state_key
        self.context_keys[self.index] = context_key
        self.values[self.index] = value
        self.index += 1


class RecurrentContextModule(nn.Module):
    """Leaky integrator over states (working memory), followed by ``tanh``."""

    def __init__(self, state_dim, integration_rate=0.5):
        super().__init__()
        self.integration_rate = integration_rate
        self.hidden_state = torch.zeros((state_dim,))

    def forward(self, x):
        h_new = self.integration_rate * x + (1 - self.integration_rate) * self.hidden_state
        self.hidden_state = h_new.detach().clone()
        return torch.tanh(h_new)


class ContextMapping(nn.Module):
    """The single *learnable* component: a Linear init to identity + zero bias,
    with its output L2-normalized. Analogous to the learnable
    ``CONTEXT -> CONTEXT[normalized]`` projection in PsyNeuLink.
    """

    def __init__(self, state_dim):
        super().__init__()
        self.in_to_out = nn.Linear(state_dim, state_dim)
        with torch.no_grad():
            self.in_to_out.weight.copy_(torch.eye(state_dim, dtype=torch.float))
            self.in_to_out.bias.zero_()
        self.in_to_out.weight.requires_grad = True
        self.in_to_out.bias.requires_grad = True

    def forward(self, x):
        return _normalize(self.in_to_out(x))


def _gen_torch_model(state_size, integration_rate, softmax_temperature,
                     softmax_threshold, memory_fill, len_memory=2, **_):
    context_module = RecurrentContextModule(state_size, integration_rate=integration_rate)
    context_mapping = ContextMapping(state_dim=state_size)
    em_module = EMModule(
        state_dim=state_size,
        temperature=softmax_temperature,
        softmax_threshold=softmax_threshold,
        memory_fill=memory_fill,
        memory_fill_n=len_memory,
    )
    return context_module, context_mapping, em_module


def run_torch_reference(states, len_memory, **kwargs):
    """Train the pure-torch EGO model and return its per-trial EM predictions.

    For each trial we run ``num_optimization_steps`` of SGD on the context
    mapping (re-querying EM each step), record the *pre-learning* EM prediction,
    then write the trial's experience into EM and advance the context.
    """
    context_module, context_mapping, em_module = _gen_torch_model(**kwargs, len_memory=len_memory)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(lr=kwargs["learning_rate"], params=context_mapping.parameters())

    predictions = []
    context = torch.zeros(11)
    prev_state = torch.zeros_like(context)
    learned_context_representation = torch.zeros_like(context)
    pred_init = torch.zeros_like(context)

    for state in states:
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        for i in range(kwargs["num_optimization_steps"]):
            learned_context_representation = context_mapping(context)
            pred_em = em_module(prev_state, learned_context_representation)
            if i == 0:
                # Snapshot the prediction *before* this trial's learning.
                pred_init = pred_em.detach().cpu().numpy().copy()
            optimizer.zero_grad()
            loss = loss_fn(pred_em, state)
            loss.backward()
            optimizer.step()

        em_module.write(prev_state, learned_context_representation.detach().cpu(), state)
        context = context_module(state)
        prev_state = state.detach().cpu()
        predictions.append(pred_init)

    return np.stack(predictions).squeeze()


# =============================================================================
# PsyNeuLink implementation
# =============================================================================

def construct_pnl_model(state_size, integration_rate, state_retrieval_weight,
                        previous_state_retrieval_weight, context_retrieval_weight,
                        learning_rate, memory_fill, softmax_temperature,
                        softmax_threshold, loss_spec_name, num_optimization_steps,
                        memory_capacity, em_class=EMComposition, **_):
    """Build the EGO model as an ``AutodiffComposition`` wrapping ``EMComposition``."""
    retrieval_softmax_gain = 1 / softmax_temperature
    loss_spec = loss_spec_name if isinstance(loss_spec_name, Loss) else Loss[loss_spec_name]
    if loss_spec is not Loss.BINARY_CROSS_ENTROPY:
        raise ValueError(loss_spec_name)

    # ----- Nodes -----
    state_input_layer = ProcessingMechanism(name="STATE", input_shapes=state_size)
    previous_state_layer = ProcessingMechanism(name="PREVIOUS STATE", input_shapes=state_size)
    context_layer = TransferMechanism(
        name="CONTEXT", input_shapes=state_size, function=Tanh,
        integrator_mode=True, integration_rate=integration_rate,
    )
    context_bias = TransferMechanism(
        name="CONTEXT[bias]", input_shapes=1, default_variable=[1.0],
    )
    context_normalized = TransferMechanism(
        name=CONTEXT + "[normalized]", input_shapes=state_size, function=Normalize(),
    )

    em = em_class(
        name="EM",
        memory_template=[[0] * state_size,   # state
                         [0] * state_size,   # previous state
                         [0] * state_size],  # context
        memory_fill=memory_fill,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=softmax_threshold,
        fields={
            state_input_layer.name: {FIELD_WEIGHT: state_retrieval_weight,
                                     LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            previous_state_layer.name: {FIELD_WEIGHT: previous_state_retrieval_weight,
                                        LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
            context_layer.name: {FIELD_WEIGHT: context_retrieval_weight,
                                 LEARN_FIELD_WEIGHT: False, TARGET_FIELD: False},
        },
        normalize_field_weights=False,
        normalize_memories=False,
        concatenate_queries=False,
        enable_learning=False,
        device=CPU,
        store_on_optimization="last",
    )

    prediction_layer = ProcessingMechanism(name="PREDICTION", input_shapes=state_size)

    # ----- Pathways -----
    QUERY, VALUE, RETRIEVED = " [QUERY]", " [VALUE]", " [RETRIEVED]"

    state_to_previous_state_pathway = [
        state_input_layer,
        MappingProjection(matrix=IDENTITY_MATRIX, learnable=False),
        previous_state_layer,
    ]
    state_to_context_pathway = [
        state_input_layer,
        MappingProjection(matrix=IDENTITY_MATRIX, learnable=False),
        context_layer,
    ]
    state_to_em_pathway = [
        state_input_layer,
        MappingProjection(sender=state_input_layer,
                          receiver=em.nodes[state_input_layer.name + VALUE],
                          matrix=IDENTITY_MATRIX, learnable=False),
        em,
    ]
    previous_state_to_em_pathway = [
        previous_state_layer,
        MappingProjection(sender=previous_state_layer,
                          receiver=em.nodes[previous_state_layer.name + QUERY],
                          matrix=IDENTITY_MATRIX, learnable=False),
        em,
    ]
    # The only learnable projection: CONTEXT -> CONTEXT[normalized].
    context_learning_pathway = [
        context_layer,
        MappingProjection(sender=context_layer, matrix=IDENTITY_MATRIX,
                          receiver=context_normalized, learnable=True),
        context_normalized,
        MappingProjection(sender=context_normalized,
                          matrix=IDENTITY_MATRIX,
                          receiver=em.nodes[context_layer.name + QUERY], learnable=False),
        em,
        MappingProjection(sender=em.nodes[state_input_layer.name + RETRIEVED],
                          receiver=prediction_layer,
                          matrix=IDENTITY_MATRIX, learnable=False),
        prediction_layer,
    ]

    ego_comp = AutodiffComposition(
        [state_to_previous_state_pathway,
         state_to_context_pathway,
         state_to_em_pathway,
         previous_state_to_em_pathway,
         context_learning_pathway],
        learning_rate=learning_rate,
        loss_spec=loss_spec,
        targets=(prediction_layer, state_input_layer),
        execute_in_additional_optimizations={context_layer: LAST,
                                             previous_state_layer: LAST},
        optimizations_per_minibatch=num_optimization_steps,
        name="EGO",
        device=CPU,
    )

    # A learnable bias into the normalized-context layer (torch: ContextMapping bias).
    ego_comp.add_node(context_bias)
    ego_comp.add_projection(
        sender=context_bias,
        receiver=context_normalized,
        projection=MappingProjection(matrix=np.zeros((1, state_size)), learnable=True),
    )

    ego_comp.infer_backpropagation_learning_pathways(ExecutionMode.PyTorch)

    # Ordering: read EM before the context/previous-state update, normalize the
    # context before EM reads it, and emit the prediction before the update.
    ego_comp.scheduler.add_condition(em, BeforeNodes(previous_state_layer, context_layer))
    ego_comp.scheduler.add_condition(context_normalized, BeforeNodes(em))
    ego_comp.scheduler.add_condition(prediction_layer, BeforeNodes(previous_state_layer, context_layer))

    return ego_comp, state_input_layer


def run_pnl_model(model, input_layer, states, num_optimization_steps, **_):
    """Train the PsyNeuLink model and return its per-trial EM predictions.

    ``results[::num_optimization_steps]`` keeps one row per trial (the first of
    each minibatch's optimization steps, i.e. the pre-learning prediction);
    column 2 is the PREDICTION output, matching ``run_torch_reference``.
    """
    model.learn(
        inputs={input_layer: states},
        execution_mode=ExecutionMode.PyTorch,
        optimizations_per_minibatch=num_optimization_steps,
        minibatch_size=1,
        synch_projection_matrices_with_torch=RUN,
        synch_node_values_with_torch=RUN,
        synch_results_with_torch=RUN,
    )
    # Extract the PREDICTION output (column 2) for the first optimization step of
    # each trial. Coerce each cell explicitly: depending on torch global state
    # left by earlier tests in a full-suite run, ``model.results`` cells may come
    # back as torch tensors or as a ragged object array rather than a clean float
    # ndarray. Going cell-by-cell (instead of ``np.asarray(model.results)`` over
    # the whole structure) keeps this robust to that and avoids numpy trying to
    # convert a multi-element tensor to a scalar.
    results = model.results
    predictions = []
    for i in range(0, len(results), num_optimization_steps):
        cell = results[i][2]
        if isinstance(cell, torch.Tensor):
            cell = cell.detach().cpu().numpy()
        predictions.append(np.asarray(cell, dtype=float).reshape(-1))
    return np.stack(predictions)


# =============================================================================
# The test
# =============================================================================

# This formerly failed for the new ``EMComposition``: with
# ``execute_in_additional_optimizations`` in use (as the EGO model requires for
# the ``context``/``previous_state`` layers), the per-field ``field_memory_nodes``
# defer their STORE into ``_nodes_to_execute_after_gradient_calc`` but were then
# skipped by the post-gradient gate in ``compositionrunner.py`` (the gate treated a
# non-empty ``_execute_in_additional_optimizations`` as a whitelist, excluding any
# node not listed in it). No field ever stored, so memory stayed at ``memory_fill``
# and every retrieval was flat. The old ``EMComposition_Proj`` (single
# ``EMStorageMechanism``) stored correctly. See the accompanying regression report.
#
# Fixed in ``compositionrunner.py``: the post-gradient gate now skips only nodes
# actually listed in ``_execute_in_additional_optimizations`` (matching the
# filtering in ``PytorchCompositionWrapper.forward``), so deferred STORE runs.
_EM_IMPLEMENTATIONS = [
    pytest.param(EMComposition_Proj, id="old_EMComposition_Proj"),
    pytest.param(EMComposition, id="new_EMComposition"),
]


@pytest.mark.model
@pytest.mark.pytorch
@pytest.mark.parametrize("em_class", _EM_IMPLEMENTATIONS)
def test_giallanza_ego_matches_torch_reference(em_class):
    """PsyNeuLink EGO model == pure-torch reference, trial-by-trial, with learning.

    Runs the EGO model (an ``AutodiffComposition`` wrapping ``em_class``, learning
    enabled, ``ExecutionMode.PyTorch``) and a hand-written pure-torch reference on
    the same CSW state sequence, and asserts their per-trial episodic-memory
    predictions are numerically identical (``atol=1e-10``).
    """
    config = dict(CONFIG)
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # exact match requires double precision
    try:
        # Seed, then build the (shared) state sequence and truncate for speed.
        set_random_seed(**config)
        states = get_state_sequence(**config)[: config["nr_trials"]]

        # Re-seed so both models start from an identical RNG state. (Both builds
        # below are deterministic given ``states``; this just removes any doubt.)
        set_random_seed(**config)
        pnl_model, input_layer = construct_pnl_model(
            **config, memory_capacity=len(states), em_class=em_class
        )
        # PsyNeuLink's input parser expects NumPy-compatible values.  In the
        # older torch version used by the Python 3.8 jobs, NumPy cannot convert
        # a list containing a multi-element tensor directly.
        pnl_states = states.detach().cpu().numpy()
        pnl_results = np.asarray(run_pnl_model(pnl_model, input_layer, pnl_states, **config))

        torch_results = np.asarray(run_torch_reference(states, len(states), **config))

        assert pnl_results.shape == torch_results.shape, (
            f"shape mismatch: PNL {pnl_results.shape} vs torch {torch_results.shape}"
        )
        assert np.allclose(pnl_results, torch_results, atol=config["tolerance"]), (
            "PsyNeuLink and torch EGO predictions differ beyond tolerance "
            f"{config['tolerance']}; max abs diff="
            f"{np.max(np.abs(pnl_results - torch_results))}"
        )
    finally:
        torch.set_default_dtype(old_dtype)
