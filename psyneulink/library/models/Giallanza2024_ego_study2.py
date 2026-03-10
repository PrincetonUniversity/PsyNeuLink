"""
Implements the model of `Giallanza et al. (2024) <https://pubmed.ncbi.nlm.nih.gov/38828434/>`_
for Study 2.

Note:
The word "context" is overloaded in this script. In the model, "context" refers to the internal representation
of the context that is mainly an integrated representation of the past states.

In the task environment, "context" refers to the two different sets of state transitions that the model has to learn.
The context is cued by the first stimulus in each trial (state 9 or state 10). However, the model is not
explicitly given a context signal, it is given a sequence of states that it has to learn to predict.

Differences from the original paper
-----------------------------------

- In the original implementation, the tanh activation function is applied to the
  state before it is passed to the context layer. In this implementation, the
  tanh activation is applied to the output of the context layer instead.

- The Episodic Memory (EM) module in the original paper is implemented as a
  dynamically growing list of memories initialized as an empty list. Here, the
  EM module is implemented as a fixed-size memory buffer initialized with a
  small non-zero value.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import psyneulink as pnl

# MODEL NAMES
EGO: str = 'EGO'
STATE: str = 'STATE'
PREVIOUS_STATE: str = 'PREVIOUS STATE'
CONTEXT: str = 'CONTEXT'
PREDICTION = 'PREDICTION'
EM: str = 'EM'

# MODEL CONFIGURATION
STATE_SIZE: int = 10

CONTEXT_INTEGRATION_RATE: float = 0.7
assert 0 <= CONTEXT_INTEGRATION_RATE <= 1

STATE_RETRIEVAL_WEIGHT: float | None = None
CONTEXT_RETRIEVAL_WEIGHT: float = 1.
PREVIOUS_STATE_RETRIEVAL_WEIGHT: float = 1.

MEMORY_FILL = .001

SOFTMAX_TEMPERATURE: float = .2
SOFTMAX_THRESHOLD: float = .01

LEARNING_RATE: float = 2.
NUM_OPTIM_STEPS: int = 5
LOSS_SPEC = pnl.Loss.BINARY_CROSS_ENTROPY


def construct_model(
        memory_capacity: int = 2,
        state_size: int = STATE_SIZE,
        integration_rate: float = CONTEXT_INTEGRATION_RATE,
        state_retrieval_weight: float = STATE_RETRIEVAL_WEIGHT,
        previous_state_retrieval_weight: float = PREVIOUS_STATE_RETRIEVAL_WEIGHT,
        context_retrieval_weight: float = CONTEXT_RETRIEVAL_WEIGHT,
        learning_rate: float = LEARNING_RATE,
        memory_fill: float = MEMORY_FILL,
        softmax_temperature: float = SOFTMAX_TEMPERATURE,
        softmax_threshold: float = SOFTMAX_THRESHOLD,
        num_optimization_steps: int = NUM_OPTIM_STEPS,
        loss_spec: pnl.Loss = LOSS_SPEC,
):
    """
    construct the model
    """
    retrieval_softmax_gain = 1 / softmax_temperature

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # *** State *** #
    # state input layer. This is the only input to the model
    state_input_layer = pnl.ProcessingMechanism(name=STATE, input_shapes=state_size)
    # previous state layer. This is used to store the previous state.
    previous_state_layer = pnl.ProcessingMechanism(name=PREVIOUS_STATE, input_shapes=state_size)

    # *** Context *** #
    # context_layer is the main context represenation that is integrated over time
    context_layer = pnl.TransferMechanism(name=CONTEXT,
                                          input_shapes=state_size,
                                          function=pnl.Tanh(),
                                          integrator_mode=True,
                                          integration_rate=integration_rate)

    # context_bias is a bias term that can be used to learn a constant bias to the context representation
    context_bias = pnl.TransferMechanism(name=CONTEXT + ' [BIAS]',
                                         input_shapes=1,
                                         default_variable=[1.],
                                         )

    # the learned context representation is normalized before being passed to the EM module
    context_normalized = pnl.TransferMechanism(name=CONTEXT + ' [NORMALIZED]',
                                               input_shapes=state_size,
                                               function=pnl.Normalize(),
                                               )

    # Episodic memory (content-addressable memory).
    # Stores (state, previous_state, context) triplets and retrieves the successor
    # state based on similarity across these fields. Field weights control how much
    # each field contributes to retrieval.
    em = pnl.EMComposition(
        name=EM,
        memory_template=[[0] * state_size,  # state
                         [0] * state_size,  # previous state
                         [0] * state_size],  # context
        memory_fill=memory_fill,
        memory_capacity=memory_capacity,
        memory_decay_rate=0,
        softmax_gain=retrieval_softmax_gain,
        softmax_threshold=softmax_threshold,
        fields={state_input_layer.name: {
            pnl.FIELD_WEIGHT: state_retrieval_weight,  # <- None, since state is not used for retrieval
            pnl.LEARN_FIELD_WEIGHT: False,
            pnl.TARGET_FIELD: False
        },
            previous_state_layer.name: {
                pnl.FIELD_WEIGHT: previous_state_retrieval_weight,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            },
            context_layer.name: {
                pnl.FIELD_WEIGHT: context_retrieval_weight,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            }
        },
        normalize_field_weights=False,
        normalize_memories=False,
        concatenate_queries=False,
        enable_learning=False,
        # <- the EM composition itself does not learn, the only learnable part is the context representation
        device=pnl.CPU,
        store_on_optimization=pnl.LAST)

    # we use a prediction layer to read out
    prediction_layer = pnl.ProcessingMechanism(name=PREDICTION, input_shapes=state_size)

    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  EGO Composition  --------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    QUERY = ' [QUERY]'
    VALUE = ' [VALUE]'
    RETRIEVED = ' [RETRIEVED]'

    # Pathways
    # the state is passed to the previous state (see, scheduling condition below)
    state_to_previous_state_pathway = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        previous_state_layer
    ]
    # the state is passed to the context layer to be integrated
    state_to_context_pathway = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        context_layer
    ]
    # the state is passed to EM where it's value is stored (it is not used for retrieval since its field weight is set
    # to None
    state_to_em_pathway = [
        state_input_layer,
        pnl.MappingProjection(sender=state_input_layer,
                              receiver=em.nodes[state_input_layer.name + VALUE],
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        em
    ]
    # the previous state is used for retrieval
    previous_state_to_em_pathway = [
        previous_state_layer,
        pnl.MappingProjection(sender=previous_state_layer,
                              receiver=em.nodes[previous_state_layer.name + QUERY],
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        em
    ]
    # the context is passed to the context_normalized layer which contains a learned mapping
    # which then is used to query the EM.
    context_learning_pathway = [
        context_layer,
        pnl.MappingProjection(sender=context_layer,
                              matrix=pnl.IDENTITY_MATRIX,
                              receiver=context_normalized,
                              learnable=True),  # <- this is the only learnable mapping in the model
        # (apart from the als learnable bias).
        context_normalized,
        pnl.MappingProjection(sender=context_normalized,
                              matrix=pnl.IDENTITY_MATRIX,
                              receiver=em.nodes[context_layer.name + QUERY],
                              learnable=False),
        em,
        pnl.MappingProjection(sender=em.nodes[state_input_layer.name + RETRIEVED],
                              receiver=prediction_layer,
                              matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        prediction_layer
    ]

    # Composition
    EGO_comp = pnl.AutodiffComposition(
        name=EGO,
        pathways=[state_to_previous_state_pathway,
                  state_to_context_pathway,
                  state_to_em_pathway,
                  previous_state_to_em_pathway,
                  context_learning_pathway],
        learning_rate=learning_rate,
        loss_spec=loss_spec,
        execute_in_additional_optimizations={
            context_layer: pnl.LAST,
            previous_state_layer: pnl.LAST
        },
        optimizations_per_minibatch=num_optimization_steps,  # <- we perform multiple optimization steps
        device=pnl.CPU)

    # here we add the bias node and make it learnable
    EGO_comp.add_node(context_bias, required_roles=[pnl.NodeRole.BIAS])  # <- mark as bias
    EGO_comp.add_projection(
        sender=context_bias,
        receiver=context_normalized,
        projection=pnl.MappingProjection(
            matrix=np.zeros((1, state_size)),  # the bias itself is stored as scalar,
            # we expand it to match the dimensionality of the context representation
            # therefore allowing each individual dimension of the context
            # representation to learn a different bias
            learnable=True  # <- this makes the bias learnable
        )
    )

    # add a learning projection
    learning_components = EGO_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
    EGO_comp.add_projection(pnl.MappingProjection(sender=state_input_layer,
                                                  receiver=learning_components[0],
                                                  learnable=False))

    # here we add the scheduling:
    # Each trial, we follow this sequence of operations:
    # (1) EM retrieval with `previous state` and `context`
    # (2) compare prediction made by EM with current state and learn the context representqtion based on error
    # (3) update previous state and context with current state for next trial
    EGO_comp.scheduler.add_condition(em, pnl.BeforeNodes(previous_state_layer, context_layer))
    EGO_comp.scheduler.add_condition(context_normalized, pnl.BeforeNodes(em))
    EGO_comp.scheduler.add_condition(prediction_layer, pnl.BeforeNodes(previous_state_layer, context_layer))

    return EGO_comp, state_input_layer


def run_model(model, input_layer, states, num_optimization_steps=NUM_OPTIM_STEPS):
    """
    run the model given a sequence of states.
    """
    model.learn(
        inputs={input_layer: states},
        execution_mode=pnl.ExecutionMode.PyTorch,
        optimizations_per_minibatch=num_optimization_steps,
        minibatch_size=1,
        synch_projection_matrices_with_torch=pnl.RUN,
        synch_node_values_with_torch=pnl.RUN,
        synch_results_with_torch=pnl.RUN,
    )
    return model.results[::num_optimization_steps][:, 2]


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


def calc_prob(em_preds, test_ys):
    """
    Calculate the probability of the EM model predicting the correct next state through EM retrieval.
    """
    # Only consider the terminal three states (they are the only predictable transitions).
    em_preds_new, test_ys_new = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds_new * test_ys_new).sum(-1).mean(-1)
    trial_probs = (em_preds * test_ys)
    return em_probability, trial_probs


def get_df(results, states, participant_nr, paradigm):
    performance_data = {
        'participant_nr': [],
        'paradigm': [],
        'trial': [],
        'probability': [],
    }
    em_preds = np.vstack([results]).reshape(-1, 5, STATE_SIZE)

    test_ys = np.vstack([states]).reshape(-1, 5, STATE_SIZE)
    correct_prob_first, _ = calc_prob(em_preds, test_ys)

    performance_data['probability'].extend(correct_prob_first)
    performance_data['participant_nr'].extend([participant_nr] * len(correct_prob_first))
    performance_data['paradigm'].extend([paradigm] * len(correct_prob_first))
    performance_data['trial'].extend(list(range(len(correct_prob_first))))
    return pd.DataFrame(performance_data)


def run_experiment(nr_participants, n_contexts, n_blocks):
    performance_data = []

    for participant in range(nr_participants):
        print(f'Running participant {participant}')
        for paradigm in ['interleaved', 'blocked']:
            states = get_trials(paradigm=paradigm, n_contexts=n_contexts, n_blocks=n_blocks)
            pnl_model, input_layer = construct_model(memory_capacity=len(states))
            results = run_model(pnl_model, input_layer, states)
            participant_df = get_df(results=results, states=states, paradigm=paradigm, participant_nr=participant)
            performance_data.append(participant_df)
    exp_df = pd.concat(performance_data).reset_index(drop=True)
    return exp_df


def plot_results(df, plot_title, n_contexts, n_blocks):
    """
    Plot the results of the experiment in the style of the paper
    shaded areas are blocked contexts in training phase.
    green area is the test phase with random contexts
    black line is the model prediction accuracy in the blocked paradigm
    gray line is the model prediction accuracy in the interleaved paradigm
    """
    palette = {
        "interleaved": "grey",
        "blocked": "black",
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for paradigm, color in palette.items():
        d = df[df["paradigm"] == paradigm]

        mean = d.groupby("trial")["probability"].mean()
        se = d.groupby("trial")["probability"].sem()

        ax.plot(mean.index, mean.values, color=color, label=paradigm)
        ax.fill_between(mean.index, mean - se, mean + se, color=color, alpha=0.2)

        # training phase
    if n_blocks is not None:
        block_size = n_contexts // n_blocks
        for block in range(n_blocks):
            start = block * block_size
            end = (block + 1) * block_size
            color = "blue" if block % 2 == 0 else "orange"
            ax.axvspan(start, end, color=color, alpha=0.08)

        # test phase: get_trials() appends n_contexts // 4 random test sequences
    test_start = n_contexts
    test_end = n_contexts + (n_contexts // 4)
    ax.axvspan(test_start, test_end, color="green", alpha=0.10)

    ax.set_xlim(0, n_contexts + (n_contexts // 4))
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(correct)")
    ax.set_title(plot_title)

    ax.legend()

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    SHOW_GRAPH = False
    N_CONTEXTS = 200
    N_BLOCKS = 4
    N_PARTICIPANTS = 10

    if SHOW_GRAPH:
        ego_model, _ = construct_model()
        ego_model.show_graph()

    data = run_experiment(nr_participants=N_PARTICIPANTS, n_contexts=N_CONTEXTS, n_blocks=N_BLOCKS)
    fig = plot_results(data, '', n_contexts=N_CONTEXTS, n_blocks=N_BLOCKS)
    fig.show()
