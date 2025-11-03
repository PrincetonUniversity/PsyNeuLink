"""
This implements Figure 8 from `Giallanza et al. (2024)<https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00143/121081/Toward-the-Emergence-of-Intelligent-Control>`_.


Differences from the original implementation
--------------------------------------------

We replaced the original single-gate (UGRNN/MGU-style) update with a fixed-rate leaky Elman-style update.

The original implementation in PyTorch (gated with an adaptive integration rate). Here we show the forward pass of
the RecurrentContextModule class that implements the context layer:
```python
def forward(self, x: torch.tensor) -> torch.tensor:
    h_prev = self.hidden_state
    h_update = torch.tanh(self.state_to_hidden(x) + self.hidden_to_hidden(h_prev))
    h_weight = torch.sigmoid(self.state_to_hidden_wt(x) + self.hidden_to_hidden_wt(h_prev))
    h_new = h_weight * h_prev + (1 - h_weight) * h_update
    if self.update_hidden_state:
        self.hidden_state = h_new.detach().clone()
    return self.hidden_to_context(h_new)
```

The modified implementation (how it would look like in PyTorch):
```python
def forward(self, x: torch.tensor) -> torch.tensor:
    h_prev = self.hidden_state
    h_new = self.integration_rate * self.state_to_hidden(x) + (1 - self.integration_rate) * self.hidden_to_hidden(h_prev)
    if self.update_hidden_state:
        self.hidden_state = h_new.detach().clone()
    return self.hidden_to_context(torch.tanh(h_new))
```
"""
# --- Imports --- #
from __future__ import annotations

import matplotlib.pyplot as plt
import psyneulink as pnl
import numpy as np
import random
from typing import List, Tuple, Optional

# --- Script Control Parameters --- #
STORE_FIG = True  # Weather to store the generated figures
SHOW_FIG = True  # Weather to show the generated figures

# --- Configuration Parameters --- #
STATE_INPUT_NAME = "STATE"
PREVIOUS_STATE_NAME = "PREVIOUS_STATE"
CONTEXT_NAME = "CONTEXT"

EM_NAME = "EM"
PREDICTION_NAME = "PREDICTION"

NUM_STATES: int = 11  # Total number of unique states in the task environment.
STATE_SIZE: int = NUM_STATES  # Size of the state that can be represented (one-hot). This is the number of unique state in the task environment.
CONTEXT_SIZE: int = STATE_SIZE  # Size of the context representation. Here, we set it to be the same as the state size.

INTEGRATION_RATE: float = .69  # The integration rate fot the context representation. In the original model, this was an adaptive gate, see note above.
SOFTMAX_THRESHOLD: float = 1e-3  # The softmax threshold for memory retrieval. All retrievals with a probability below this threshold are ignored.
STATE_RETRIEVAL_WEIGHT = None  # Weight of the state during memory retrieval. Since the state is the target of prediction, this is set to None.
PREVIOUS_STATE_RETRIEVAL_WEIGHT = 1.  # Weight of the previous state during memory retrieval.
CONTEXT_RETRIEVAL_WEIGHT = 1.  # Weight of the context during memory retrieval.
NORMALIZE_FIELD_WEIGHT = False  # Whether to normalize the field weights during memory retrieval.
NORMALIZE_MEMORIES = False  # Whether to normalize the memories during memory retrieval.
CONCATENATE_QUERIES = False  # Weather to concatenate the queries before memory retrieval.
ENABLE_LEARNING = True  # Weather to enable learning for the context layer
MEMORY_INIT = 0.01  # The initial values for the memory entries.
LOSS_SPEC = pnl.Loss.BINARY_CROSS_ENTROPY  # The loss function used for learning.
LEARNING_RATE = 1  # The learning rate for the context layer.
DEVICE = pnl.CPU  # The device to use for computation.
SOFTMAX_TEMPERATURE = .1  # The softmax temperature for memory retrieval. Lower values make the retrieval more argmax-like.
EXECUTION_MODE = pnl.ExecutionMode.PyTorch  # The execution mode for the model.
NUM_OPTIMIZATION_STEPS = 10  # The number of optimization steps to learn the context representation per trial.



def construct_model(
        memory_capacity,
) -> (pnl.Composition, pnl.ProcessingMechanism):
    if pnl.is_numeric_scalar(SOFTMAX_TEMPERATURE):  # translate to gain of softmax retrieval function
        retrieval_softmax_gain = 1 / SOFTMAX_TEMPERATURE
    else:
        retrieval_softmax_gain = SOFTMAX_TEMPERATURE

    assert 0 <= INTEGRATION_RATE <= 1, \
        f"integrator_retrieval_weight must be a number from 0 to 1"

    # --- Nodes --- #
    # Input Layers
    state_input_layer = pnl.ProcessingMechanism(
        name=STATE_INPUT_NAME, input_shapes=STATE_SIZE
    )
    previous_state_layer = pnl.ProcessingMechanism(
        name=PREVIOUS_STATE_NAME, input_shapes=STATE_SIZE
    )
    context_layer = pnl.TransferMechanism(name=CONTEXT_NAME,
                                          input_shapes=STATE_SIZE,
                                          function=pnl.Tanh,
                                          integrator_mode=True,
                                          integration_rate=INTEGRATION_RATE)

    em = pnl.EMComposition(
        name=EM_NAME,
        memory_template=[[0] * STATE_SIZE,  # state
                         [0] * STATE_SIZE,  # previous state
                         [0] * CONTEXT_SIZE],  # context
        memory_fill=MEMORY_INIT,
        memory_capacity=memory_capacity,
        normalize_memories=NORMALIZE_MEMORIES,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=SOFTMAX_THRESHOLD,
        fields={state_input_layer.name: {pnl.FIELD_WEIGHT: STATE_RETRIEVAL_WEIGHT,
                                         pnl.LEARN_FIELD_WEIGHT: False,
                                         pnl.TARGET_FIELD: True},
                previous_state_layer.name: {pnl.FIELD_WEIGHT: PREVIOUS_STATE_RETRIEVAL_WEIGHT,
                                            pnl.LEARN_FIELD_WEIGHT: False,
                                            pnl.TARGET_FIELD: False},
                context_layer.name: {pnl.FIELD_WEIGHT: CONTEXT_RETRIEVAL_WEIGHT,
                                     pnl.LEARN_FIELD_WEIGHT: False,
                                     pnl.TARGET_FIELD: False}},
        normalize_field_weights=NORMALIZE_FIELD_WEIGHT,

        concatenate_queries=CONCATENATE_QUERIES,
        enable_learning=ENABLE_LEARNING,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        store_on_optimization='last')

    prediction_layer = pnl.ProcessingMechanism(name=PREDICTION_NAME, input_shapes=STATE_SIZE)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
    state_to_previous_state_pathway = [state_input_layer,
                                       pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                                                             learnable=False),
                                       previous_state_layer]
    state_to_context_pathway = [state_input_layer,
                                pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                                                      learnable=False),
                                context_layer]
    state_to_em_pathway = [state_input_layer,
                           pnl.MappingProjection(sender=state_input_layer,
                                                 receiver=em.nodes[state_input_layer.name + VALUE],
                                                 matrix=pnl.IDENTITY_MATRIX,
                                                 learnable=False),
                           em]
    previous_state_to_em_pathway = [previous_state_layer,
                                    pnl.MappingProjection(sender=previous_state_layer,
                                                          receiver=em.nodes[previous_state_layer.name + QUERY],
                                                          matrix=pnl.IDENTITY_MATRIX,
                                                          learnable=False),
                                    em]
    context_learning_pathway = [context_layer,
                                pnl.MappingProjection(sender=context_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      receiver=em.nodes[context_layer.name + QUERY],
                                                      learnable=True,
                                                      ),
                                em,
                                pnl.MappingProjection(sender=em.nodes[state_input_layer.name + RETRIEVED],
                                                      receiver=prediction_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      learnable=False),
                                prediction_layer]

    # Composition
    EGO_comp = pnl.AutodiffComposition(
        name="EGO Model",
        pathways=[state_to_previous_state_pathway,
                  state_to_context_pathway,
                  state_to_em_pathway,
                  previous_state_to_em_pathway,
                  context_learning_pathway],
        learning_rate=LEARNING_RATE,
        loss_spec=LOSS_SPEC,
        execute_in_additional_optimizations={
            context_layer: pnl.LAST,
            previous_state_layer: pnl.LAST},
        optimizations_per_minibatch=NUM_OPTIMIZATION_STEPS,

        device=DEVICE)

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
    EGO_comp.add_projection(pnl.MappingProjection(sender=state_input_layer,
                                                  receiver=learning_components[0],
                                                  learnable=False))

    EGO_comp.scheduler.add_condition(em, pnl.BeforeNodes(previous_state_layer, context_layer))
    EGO_comp.scheduler.add_condition(prediction_layer, pnl.BeforeNodes(previous_state_layer, context_layer))

    return EGO_comp, state_input_layer


def state_seq_ctx1(rng: random.Random) -> List[int]:
    """
    Generate a state sequence for Context 1:
        The context indicator (first state of the sequence) is always 10, then
        the second state is randomly chosen between 1 and 2.
        In context 1, the update rule is state = prev_state + 2:

        Context 1: [9, 1|2, then always +2 each step].
    """
    states = [9, rng.choice([1, 2])]  # initialize first two states
    while True:
        state = states[-1] + 2  # update rule
        if state >= NUM_STATES - 2:  # if the next state exceeds limit, return
            return states
        states.append(state)  # otherwise, append and continue


def state_seq_ctx2(rng: random.Random) -> List[int]:
    """
    Generate a state sequence for Context 2:
        The context indicator (first state of the sequence) is always 10, then
        the second state is randomly chosen between 1 and 2.
        In context 2, the update rule is
            state = prev_state + 1 if even else prev_state + 3

        Context 1: [10, 1|2, then if even +1 else +3].
    """
    states = [10, rng.choice([1, 2])]
    while True:
        state = states[-1] + (1 if states[-1] % 2 == 0 else 3)
        if state >= NUM_STATES - 2:
            return states
        states.append(state)


def build(paradigm: str, n: int, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Concatenate states and a parallel context array (one context per timestep).
    Returns (states, contexts). In addition to the trainig phase, a final test phase with random context is added
    """
    rng = random.Random(seed)
    p = paradigm.lower()
    if p == "blocked":
        ctx_seq = [0] * n + [1] * n + [0] * n + [1] * n
    elif p == "interleaved":
        ctx_seq = [0, 1] * (2 * n)  # explicit repetition
    else:
        raise ValueError(f"Unknown paradigm {paradigm}")

    states: List[int] = []
    contexts: List[int] = []

    for ctx in ctx_seq:
        seq = state_seq_ctx1(rng) if ctx == 0 else state_seq_ctx2(rng)
        states.extend(seq)
        contexts.extend([ctx] * len(seq))

    # add final test phase with random contexts
    for _ in range(n):
        ctx = rng.choice([0, 1])
        seq = state_seq_ctx1(rng) if ctx == 0 else state_seq_ctx2(rng)
        states.extend(seq)
        contexts.extend([2] * len(seq))  # mark test context as '2'

    return (
        np.asarray(states, dtype=int),
        np.asarray(contexts, dtype=int),
    )


def one_hot_encoded(states: np.ndarray) -> np.ndarray:
    """
    Convert state indexes to one-hot encoded vectors.
    """
    nr_states = len(states)
    one_hot = np.zeros((nr_states, NUM_STATES), dtype=float)
    one_hot[np.arange(nr_states), states] = 1.
    return one_hot


def plot_with_context(ax: plt.Axes,
                      states: np.ndarray,
                      contexts: np.ndarray,
                      title: str) -> None:
    nr_trials = len(states)
    x = np.arange(nr_trials)

    edges = [0]
    for i in range(1, nr_trials):
        if contexts[i] != contexts[i - 1]:
            edges.append(i)
    edges.append(nr_trials)

    # context-shaded background
    for s in range(len(edges) - 1):
        start, end = edges[s], edges[s + 1]
        ctx = int(contexts[start])
        col = 'blue' if ctx == 0 else ('orange' if ctx == 1 else "yellow")
        ax.axvspan(start, end, facecolor=col, alpha=0.3, linewidth=0)

    # plot the state indexes
    ax.scatter(x, states, s=1, alpha=0.3)

    # plot vertical lines that indicate a new state sequence
    for start in range(nr_trials):
        if states[start] in (9, 10):
            ax.axvline(start, linestyle=':', color='k', alpha=0.45, linewidth=0.7, zorder=1)

    ax.set_title(title)

    ax.set_xlim(0, nr_trials)  # no white space
    ax.margins(x=0)  # no extra space on x-axis
    ax.set_xlabel("Trial Index")
    ax.set_ylabel("State index")


def calc_prob(predictions, targets, seq_length=NUM_STATES // 2, exclude_first_n=2):
    """
    predictions: The predictions from the model (probabilities over states)
    targets: The true one-hot encoded target states
    seq_len: The length of each state sequence
    exclude_first_n: Number of initial states to exclude from evaluation since they are not predictable (first
        state of each sequence is the context indicator, second state is randomly chosen between two options)

    Returns:
      per_seq_mean: [N_seq] mean true-class prob over the kept positions in each sequence (this will be
      nr_total_state/seq_length trials)
    """
    pred = np.asarray(predictions, dtype=float)
    y = np.asarray(targets, dtype=float)
    T, K = pred.shape

    # number of trials is not a multiple of seq_length
    if T % seq_length:
        raise ValueError("Prediction length is not a multiple of sequence length")

    # reshape to [N, seq_length, K] so each sequence is one "trial"
    N = pred.shape[0] // seq_length
    _preds = pred.reshape(N, seq_length, K)
    _y = y.reshape(N, seq_length, K)

    # exclude first n positions in each sequence since they are not predictable
    sl = slice(exclude_first_n, seq_length)

    # true-class probability per state
    p_true = (_preds * _y).sum(-1)  # [N, seq_len]
    per_state_kept = p_true[:, sl]  # [N, kept]
    per_seq_mean = per_state_kept.mean(-1)  # [N]

    return per_seq_mean


# def main():



if __name__ == "__main__":
    # First, we build the two datasets

    # n is the number of sequences per block (in blocked). The number of
    # total trials in the training phase is 4 * n * len(sequence) (which is 5 in this case) Total: 800 trials
    # the number of trails in the test phase is n * len(sequence) Total: 200 trials
    n = 40
    seed = None

    # Build the dataset
    states_blocked, ctx_b = build("blocked", n=n, seed=seed)
    states_interleaved, ctx_i = build("interleaved", n=n, seed=seed)

    # Here, we plot the two datasets. The grey dots indicate the state index at each trial
    # The background color indicates the context in the training phase and is yellow for the test phase
    # Dotted vertical lines indicate the beginning of a new state sequence
    if STORE_FIG or SHOW_FIG:
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
        plot_with_context(axes[0], states_blocked, ctx_b, "Blocked")
        plot_with_context(axes[1], states_interleaved, ctx_i, "Interleaved")
        fig.tight_layout()
        if STORE_FIG:
            fig.savefig("training_schedules.png", dpi=200, bbox_inches="tight")
        if SHOW_FIG:
            plt.show()

    # The model expects one-hot encoded inputs, so we convert the state labels to one-hot vectors
    states_blocked_one_hot = one_hot_encoded(states_blocked)
    states_interleaved_one_hot = one_hot_encoded(states_interleaved)

    # Now, we construct the model.
    model_blocked, input_blocked = construct_model(memory_capacity=len(states_blocked_one_hot))


    # and run it on the blocked dataset
    def print_stuff(**kwargs):
        print(model_blocked.nodes['EM'].parameters.memory.get(model_blocked.name))

    # states_blocked_one_hot = states_blocked_one_hot[:3]
    # for s in states_blocked_one_hot:
    model_blocked.learn(
            inputs={input_blocked: states_blocked_one_hot},
            execution_mode=EXECUTION_MODE,
            optimizations_per_minibatch=NUM_OPTIMIZATION_STEPS,
            minibatch_size=1,
            synch_projection_matrices_with_torch=pnl.RUN,
            synch_node_values_with_torch=pnl.RUN,
            synch_results_with_torch=pnl.RUN,
            learning_rate=LEARNING_RATE,
    )
    results_blocked = model_blocked.results[::NUM_OPTIMIZATION_STEPS][:, 2]

        # print(model_blocked.nodes['EM'].parameters.ba.get(model_blocked.name))
        # print(model_blocked.projections['Context to EM Query'].parameters.matrix.get(model_blocked.name))

    #
    # # We reset the model and run it on the interleaved dataset
    # model_interleaved, input_interleaved = construct_model(memory_capacity=len(states_interleaved_one_hot))
    # model_interleaved.learn(
    #     inputs={input_interleaved: states_interleaved_one_hot},
    #     execution_mode=EXECUTION_MODE,
    #     optimizations_per_minibatch=NUM_OPTIMIZATION_STEPS,
    #     minibatch_size=1,
    #     synch_projection_matrices_with_torch=pnl.RUN,
    #     synch_node_values_with_torch=pnl.RUN,
    #     synch_results_with_torch=pnl.RUN,
    #     learning_rate=LEARNING_RATE,
    # )
    # results_interleaved = model_interleaved.results[::NUM_OPTIMIZATION_STEPS][:, 2]
    if STORE_FIG or SHOW_FIG:
        prob_blocked = calc_prob(results_blocked, states_blocked_one_hot)
        # prb_interleaved = calc_prob(results_interleaved, states_interleaved_one_hot)
        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

        ax[0].plot(prob_blocked, label="True-class probability")  # "soft accuracy"
        ax[0].set_title("Blocked")
        ax[0].set_xlabel("Trial Index (Sequences of 5 excluding the first 2 positions)")
        ax[0].set_ylabel("True-class Probability")
        ax[0].legend()
    #
        # ax[1].plot(prb_interleaved, label="True-class probability")
        # ax[1].set_title("Interleaved")
        # ax[1].set_xlabel("Trial Index (Sequences of 5 excluding the first 2 positions)")
        # ax[1].set_ylabel("True-class Probability")
        # ax[1].legend()
        # fig.tight_layout()
        if STORE_FIG:
            fig.savefig("model_predictions_blocked.png", dpi=200, bbox_inches="tight")
        if SHOW_FIG:
            plt.show()
