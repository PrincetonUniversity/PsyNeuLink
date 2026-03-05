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
CONTEXT_RETRIEVAL_WEIGHT: float = .5
PREVIOUS_STATE_RETRIEVAL_WEIGHT: float = .5

MEMORY_INIT = .01

SOFTMAX_TEMPERATURE: float = .1
SOFTMAX_THRESHOLD: float = .001

LEARNING_RATE: float = 0.5
NUM_OPTIM_STEPS: int = 10


def construct_model(
        memory_capacity,
) -> pnl.AutodiffComposition:
    # ----------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  Nodes  ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------

    # Input layers (present state and previous state)
    state_input_layer = pnl.ProcessingMechanism(name=STATE_NAME, input_shapes=STATE_SIZE)
    previous_state_layer = pnl.ProcessingMechanism(name=PREVIOUS_STATE_NAME, input_shapes=STATE_SIZE)

    # Context layer (learned representation of the context, which is integrated over time)
    context_layer = pnl.TransferMechanism(
        name=CONTEXT_NAME, input_shapes=STATE_SIZE,
        function=pnl.Tanh(gain=1),
        integrator_mode=True,
        integration_rate=CONTEXT_INTEGRATION_RATE)

    em = pnl.EMComposition(
        name=EM_NAME,
        memory_template=[
            [0] * STATE_SIZE,  # state
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
                pnl.TARGET_FIELD: True
            },
            PREVIOUS_STATE_NAME: {
                pnl.FIELD_WEIGHT: PREVIOUS_STATE_RETRIEVAL_WEIGHT,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            },
            CONTEXT_NAME: {
                pnl.FIELD_WEIGHT: CONTEXT_RETRIEVAL_WEIGHT,
                pnl.LEARN_FIELD_WEIGHT: False,
                pnl.TARGET_FIELD: False
            }
        },
        normalize_field_weights=True,
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
        previous_state_layer]
    state_to_context_pathway = [
        state_input_layer,
        pnl.MappingProjection(matrix=pnl.IDENTITY_MATRIX,
                              learnable=False),
        context_layer]
    state_to_em_pathway = [state_input_layer,
                           pnl.MappingProjection(sender=state_input_layer,
                                                 receiver=em.nodes[STATE_NAME + VALUE],
                                                 matrix=pnl.IDENTITY_MATRIX,
                                                 learnable=False),
                           em]
    previous_state_to_em_pathway = [previous_state_layer,
                                    pnl.MappingProjection(sender=previous_state_layer,
                                                          receiver=em.nodes[PREVIOUS_STATE_NAME + QUERY],
                                                          matrix=pnl.IDENTITY_MATRIX,
                                                          learnable=False),
                                    em]
    context_learning_pathway = [context_layer,
                                pnl.MappingProjection(sender=context_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      receiver=em.nodes[CONTEXT_NAME + QUERY],
                                                      learnable=True),
                                em,
                                pnl.MappingProjection(sender=em.nodes[STATE_NAME + RETRIEVED],
                                                      receiver=prediction_layer,
                                                      matrix=pnl.IDENTITY_MATRIX,
                                                      learnable=False),
                                prediction_layer]

    # Composition
    EGO_comp = pnl.AutodiffComposition(
        name=EGO_NAME,
        pathways=[state_to_previous_state_pathway,
                  state_to_context_pathway,
                  state_to_em_pathway,
                  previous_state_to_em_pathway,
                  context_learning_pathway],
        learning_rate=LEARNING_RATE,
        loss_spec=pnl.Loss.BINARY_CROSS_ENTROPY,
        execute_in_additional_optimizations={context_layer: pnl.LAST,
                                             previous_state_layer: pnl.LAST},
        # BREADCRUMB: REQUIRED HERE UNTIL IMPLEMENTED FOR learn()
        optimizations_per_minibatch=NUM_OPTIM_STEPS,

        device=pnl.CPU)

    learning_components = EGO_comp.infer_backpropagation_learning_pathways(pnl.ExecutionMode.PyTorch)
    EGO_comp.add_projection(pnl.MappingProjection(sender=state_input_layer,
                                                  receiver=learning_components[0],
                                                  learnable=False))

    EGO_comp.scheduler.add_condition(em, pnl.BeforeNodes(previous_state_layer, context_layer))
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
        for _ in range(2):
            states.append(states[-1] + 2)
        return states

    def gen_context_2():
        # context cue + random "first" state (1 or 2)
        states = [10, np.random.choice([1, 2])]
        for _ in range(2):
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
        for _ in range(n_contexts):
            if np.random.rand() < 0.5:
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

    if paradigm not in ['interleaved', 'blocked']:
        raise ValueError('Paradigm must be either `interleaved` or `blocked`.')

    if paradigm == 'interleaved':
        return [to_one_hot(c) for c in gen_contexts_interleaved()]
    if paradigm == 'blocked':
        return [to_one_hot(c) for c in gen_contexts_blocked()]



def run_model(model,
              trials,
              ):
    model.learn(inputs={STATE_NAME: trials},
                execution_mode=pnl.ExecutionMode.PyTorch,
                optimizations_per_minibatch=NUM_OPTIM_STEPS,
                minibatch_size=1,
                synch_projection_matrices_with_torch=pnl.RUN,
                synch_node_values_with_torch=pnl.RUN,
                synch_results_with_torch=pnl.RUN,
                )

    print(model.results)

    return model.results[::NUM_OPTIM_STEPS][:, 2]

def plot_results(predictions, targets):
    fig, ax = plt.subplots()
    # print(len(predictions))
    # print(target)
    accuracy = 1 - (np.abs(predictions - targets)).sum(-1) / 2
    ax.plot(accuracy)
    ax.set_xlabel('Stimuli')
    ax.set_ylabel('loss')
    plt.show()


if __name__ == '__main__':
    trials = get_trials('interleaved', 200, n_blocks=4)
    targets = trials
    print(len(trials))
    model = construct_model(memory_capacity=len(trials))
    results = run_model(model, trials)


    plot_results(results, targets)
    # model, _, _, _ = construct_model(memory_capacity=5)
    # results = run_model(model, trials)
    print(results)
