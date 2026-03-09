"""
Implements the model of `Giallanza et al. (2024)<https://pubmed.ncbi.nlm.nih.gov/38828434/>`_ for Study 2.
"""
import numpy as np
import matplotlib.pyplot as plt

import psyneulink as pnl

# MODEL CONFIGURATION
EGO_NAME = "EGO"
EM_NAME = "EM"

STATE_NAME = "STATE"
PREVIOUS_STATE_NAME = "PREVIOUS STATE"
CONTEXT_NAME = "CONTEXT"
PREDICTION_NAME = "PREDICTION"

STATE_SIZE: int = 10

CONTEXT_INTEGRATION_RATE: float = 0.69
assert 0 <= CONTEXT_INTEGRATION_RATE <= 1

STATE_RETRIEVAL_WEIGHT: float | None = None
CONTEXT_RETRIEVAL_WEIGHT: float = 1.
PREVIOUS_STATE_RETRIEVAL_WEIGHT: float = 1.

MEMORY_INIT = .0001

SOFTMAX_TEMPERATURE: float = .2
SOFTMAX_THRESHOLD: float = .001

LEARNING_RATE: float = 2.
NUM_OPTIM_STEPS: int = 10
LOSS_SPEC = pnl.Loss.BINARY_CROSS_ENTROPY


def construct_model(
        memory_capacity,
) -> pnl.AutodiffComposition:
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # Input
    state_input_layer = pnl.ProcessingMechanism(name=STATE_NAME, input_shapes=STATE_SIZE)
    previous_state_layer = pnl.ProcessingMechanism(name=PREVIOUS_STATE_NAME, input_shapes=STATE_SIZE)

    # Context layer
    # Simple integrator that projects to a normalize version of itself.
    # The projection from the integrator to the normalized version is the only learnable
    # component of the EGO model. It includes a bias term that is also learnable
    context_layer = pnl.TransferMechanism(
        name=CONTEXT_NAME,
        input_shapes=STATE_SIZE,
        integrator_mode=True,
        integration_rate=CONTEXT_INTEGRATION_RATE)

    context_bias = pnl.TransferMechanism(
        name=CONTEXT_NAME + '[bias]',
        input_shapes=1,
        default_variable=[1.],
    )

    context_normalized = pnl.TransferMechanism(
        name=CONTEXT_NAME + '[normalized]',
        input_shapes=STATE_SIZE,
        function=pnl.Normalize(),
    )

    em = pnl.EMComposition(
        name=EM_NAME,
        memory_template=[[0] * STATE_SIZE,  # state
                         [0] * STATE_SIZE,  # previous state
                         [0] * STATE_SIZE],  # context
        memory_fill=MEMORY_INIT,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=1 / SOFTMAX_TEMPERATURE,
        softmax_threshold=SOFTMAX_THRESHOLD,
        fields={
            STATE_NAME: {
                pnl.FIELD_WEIGHT: STATE_RETRIEVAL_WEIGHT,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: True},
            PREVIOUS_STATE_NAME: {
                pnl.FIELD_WEIGHT: PREVIOUS_STATE_RETRIEVAL_WEIGHT,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False},
            CONTEXT_NAME: {
                pnl.FIELD_WEIGHT: CONTEXT_RETRIEVAL_WEIGHT,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False}},
        normalize_field_weights=False,
        normalize_memories=False,
        concatenate_queries=False,
        enable_learning=True,
        learning_rate=LEARNING_RATE,
        device=pnl.CPU,
        store_on_optimization='last')

    prediction_layer = pnl.ProcessingMechanism(name=PREDICTION_NAME, input_shapes=STATE_SIZE)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
    state_to_previous_state_pathway = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        previous_state_layer
    ]
    state_to_context_pathway = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        context_layer
    ]
    state_to_em_pathway = [
        state_input_layer,
        pnl.MappingProjection(sender=state_input_layer,
                              receiver=em.nodes[STATE_NAME + VALUE],
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        em
    ]
    previous_state_to_em_pathway = [
        previous_state_layer,
        pnl.MappingProjection(sender=previous_state_layer,
                              receiver=em.nodes[PREVIOUS_STATE_NAME + QUERY],
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        em
    ]
    context_learning_pathway = [
        context_layer,
        pnl.MappingProjection(sender=context_layer,
                              matrix=pnl.IDENTITY_MATRIX,
                              receiver=context_normalized,
                              learnable=True),
        context_normalized,
        pnl.MappingProjection(sender=context_normalized,
                              matrix=pnl.IDENTITY_MATRIX,
                              receiver=em.nodes[CONTEXT_NAME + QUERY],
                              learnable=False),
        em,
        pnl.MappingProjection(sender=em.nodes[STATE_NAME + RETRIEVED],
                              receiver=prediction_layer,
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        prediction_layer]

    # Composition
    EGO_comp = pnl.AutodiffComposition(
        name=EGO_NAME,
        pathways=[
            state_to_previous_state_pathway,
            state_to_context_pathway,
            state_to_em_pathway,
            previous_state_to_em_pathway,
            context_learning_pathway
        ],
        learning_rate=LEARNING_RATE,
        loss_spec=LOSS_SPEC,
        execute_in_additional_optimizations={
            context_layer: pnl.LAST,
            previous_state_layer: pnl.LAST
        },
        optimizations_per_minibatch=NUM_OPTIM_STEPS,

        device=pnl.CPU)

    EGO_comp.add_node(context_bias)
    EGO_comp.add_projection(
        sender=context_bias,
        receiver=context_normalized,
        projection=pnl.MappingProjection(
            matrix=np.zeros((1, STATE_SIZE)),
            learnable=True
        )
    )

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
    EGO_comp.add_projection(pnl.MappingProjection(sender=state_input_layer,
                                                  receiver=learning_components[0],
                                                  learnable=False))

    EGO_comp.scheduler.add_condition(em, pnl.BeforeNodes(previous_state_layer, context_layer))
    EGO_comp.scheduler.add_condition(context_normalized, pnl.BeforeNodes(em))
    EGO_comp.scheduler.add_condition(prediction_layer, pnl.BeforeNodes(previous_state_layer, context_layer))

    return EGO_comp


def to_one_hot(state, num_states=10):
    """
    Convert a state to one-hot encoding.
    """
    # create an identity matrix
    one_hot = np.eye(num_states)
    # access the row corresponding to the state (subtract 1 since states are 1-indexed)
    return list(one_hot[state - 1])


def get_trials(
        paradigm: str,
        n_contexts: int = None,
        n_blocks: int = None) -> list[list[int]] | None:
    """
    Generate trials for the model.

    There are two different contexts, each with a different set of transitions between states:
    The cue for the contexts are state 9 and state 10, respectively.
    The transitions between states depend on the context:

    Context 1: (context clue is 9), the transition rule is "+2" resulting in the following possible transitions:
    9 (cue) -> 1 -> 3 -> 5 -> 7
    9 (cue) -> 2 -> 4 -> 6 -> 8

    Context 2: (context clue is 10), the transition rule is "+1" if the current state is even and "+3" if
    the current state is odd, resulting in the following possible transitions:
    10 (cue) -> 1 -> 4 -> 5 -> 8
    10 (cue) -> 2 -> 3 -> 6 -> 7

    Blocked paradigm: contexts are blocked together
    Interleaved paradigm: contexts are alternating

    Arguments:
        paradigm: The name of the paradigm (allowed values are "interleaved" and "blocked").
        n_contexts: The number of contexts to generate (required for both paradigms). The number of states
            generated will be 5 times the number of contexts since each context has 5 states.
        n_blocks: The number of blocks to generate (only for the blocked paradigm).

    Returns:
        A list of lists, where each inner list is a one-hot encoded state representing a state.
    """

    def gen_context_1():
        # context cue + random "first" state (1 or 2)
        states = [9, np.random.choice([1, 2])]
        for _ in range(3):
            states.append(states[-1] + 2)
        return states

    def gen_context_2():
        # context cue + random "first" state (1 or 2)
        states = [10, np.random.choice([1, 2])]
        for _ in range(3):
            if states[-1] % 2 == 0:
                states.append(states[-1] + 1)
            else:
                states.append(states[-1] + 3)
        return states

    def gen_contexts_interleaved():
        """
        Generate interleaved contexts for the model.
        """
        contexts = []
        for i in range(n_contexts):
            if i % 2 == 0:
                contexts += gen_context_1()
            else:
                contexts += gen_context_2()
        return contexts

    def gen_contexts_blocked():
        """
        Generate blocked contexts for the model.
        """
        if n_blocks is None:
            raise ValueError('Number of blocks must be specified for blocked paradigm.')
        if n_blocks % 2:
            raise ValueError('Number of blocks must be even for blocked paradigm.')
        if n_contexts % n_blocks:
            raise ValueError('Number of contexts must be divisible by number of blocks for blocked paradigm.')
        contexts = []
        for i in range(n_blocks):
            if i % 2 == 0:
                for j in range(n_contexts // n_blocks):
                    contexts += gen_context_1()
            else:
                for j in range(n_contexts // n_blocks):
                    contexts += gen_context_2()

        return contexts

    def gen_contexts_test():
        contexts = []
        for i in range(n_contexts // 4):
            if np.random.random() < 0.5:
                contexts += gen_context_1()
            else:
                contexts += gen_context_2()
        return contexts

    if paradigm not in ['interleaved', 'blocked']:
        raise ValueError('Paradigm must be either `interleaved` or `blocked`.')

    if paradigm == 'interleaved':
        return [to_one_hot(c) for c in gen_contexts_interleaved() + gen_contexts_test()]
    if paradigm == 'blocked':
        return [to_one_hot(c) for c in gen_contexts_blocked() + gen_contexts_test()]


def run_model(model, trials):
    model.learn(
        inputs={STATE_NAME: trials},
        execution_mode=pnl.ExecutionMode.PyTorch,
        optimizations_per_minibatch=NUM_OPTIM_STEPS,
        minibatch_size=1,
        synch_projection_matrices_with_torch=pnl.RUN,
        synch_node_values_with_torch=pnl.RUN,
        synch_results_with_torch=pnl.RUN,
    )

    R = np.asarray(model.results)

    # prediction output
    preds = R[:, 2]

    # one logical trial contributes NUM_OPTIM_STEPS entries
    predictions = preds[2::NUM_OPTIM_STEPS]

    return np.asarray(predictions)


def compute_accuracy(predictions, targets, trial_len=None, ignore_first_n=0):
    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    if preds.shape != targs.shape:
        raise ValueError(
            f"Shape mismatch: predictions.shape={preds.shape}, targets.shape={targs.shape}"
        )

    # accuracy for one-hot vectors:
    # identical -> 1
    # completely wrong one-hot -> 0
    accuracy = 1 - np.abs(preds - targs).sum(axis=-1) / 2

    if trial_len is not None:
        if len(accuracy) % trial_len != 0:
            raise ValueError(
                f"Number of samples ({len(accuracy)}) is not divisible by trial_len ({trial_len})"
            )
        accuracy = accuracy.reshape(-1, trial_len)

        if ignore_first_n > 0:
            accuracy = accuracy[:, ignore_first_n:]

        accuracy = accuracy.reshape(-1)

    return accuracy


def plot_results(*series, labels=None, ylabel="Accuracy"):
    fig, ax = plt.subplots(figsize=(8, 4))

    for i, values in enumerate(series):
        label = labels[i] if labels is not None else f"series {i + 1}"
        ax.plot(values, label=label, alpha=0.9)

    ax.set_xlabel("Stimuli")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    if len(series) > 1:
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    trials = get_trials("blocked", 200, n_blocks=4)

    model = construct_model(memory_capacity=len(trials))
    predictions = run_model(model, trials)

    acc = compute_accuracy(
        predictions,
        trials,
        trial_len=5,  # each logical trial has 5 stimuli
        ignore_first_n=2  # ignore first 2, keep only the predictable last 3
    )

    plot_results(acc, labels=["blocked"])
