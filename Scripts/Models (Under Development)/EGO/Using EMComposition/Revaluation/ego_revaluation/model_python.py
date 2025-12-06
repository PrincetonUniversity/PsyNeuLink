"""
Change from original:

- Fix bugs for time code and estimation (see changeSet.md)
- Use softmax with threshold + memory with small value initialized
"""
from typing import Optional

import numpy as np

from .config import defaults
from .src import utils


def match(key: np.ndarray,
          memories: np.ndarray,
          metric=defaults.METRIC,
          ) -> np.ndarray:
    """
    Cosine similarity or dot product between a key vector and an array of memory vectors.
    """
    if metric == 'cosine_similarity':
        _key = utils.normalized(key)
        _mem = utils.normalized(memories)
    elif metric == 'dot_product':
        _key = key
        _mem = memories
    else:
        raise ValueError('Unknown metric {}'.format(metric))

    return np.sum(_mem * _key, axis=-1)


def project_next_context(context,
                         state,
                         integration_rate):
    """
    Projected next context as weighted combination of old context, new state, and new context.
    """
    return context * (1 - integration_rate) + integration_rate * state


def gen_memories(states: np.ndarray,
                 rewards: np.ndarray,
                 times: np.ndarray,
                 state_integration_rate: float = defaults.STATE_INTEGRATION_RATE,
                 memory_init: float = defaults.MEMORY_INIT,
                 memory_capacity: Optional[int] = None,
                 ):
    """
    Generate episodic memories from a sequence of visited states, rewards, and time codes.
    """

    if memory_capacity is None:
        memory_capacity = len(states)

    # Calculate dimensions
    state_d = states.shape[1]
    context_d = state_d
    time_d = times.shape[1]
    reward_d = 1

    # Initialize empty memories
    state_memories = np.ones((memory_capacity, state_d)) * memory_init
    context_memories = np.ones((memory_capacity, context_d)) * memory_init
    time_memories = np.ones((memory_capacity, time_d)) * memory_init
    reward_memories = np.ones((memory_capacity, reward_d)) * memory_init

    # Initialize starting context representation
    context_rep = np.zeros(context_d, dtype=float)

    for t in range(len(states)):
        state_cur = states[t]
        reward_cur = rewards[t]
        time_cur = times[t]

        # store (not integrated yet) context with current state
        state_memories[t] = state_cur
        context_memories[t] = context_rep
        time_memories[t] = time_cur
        reward_memories[t] = reward_cur

        # Integration of context
        context_rep = context_rep * (1 - state_integration_rate) + \
                      state_cur * state_integration_rate

    return state_memories, context_memories, time_memories, reward_memories


def sample_memory(memories,
                  query,
                  state_retrieval_weight=0,
                  context_retrieval_weight=1 - defaults.TIME_RETRIEVAL_WEIGHT,
                  time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
                  temperature=defaults.TEMPERATURE,
                  softmax_threshold=defaults.SOFTMAX_THRESHOLD,
                  metric=defaults.METRIC,
                  mode='sample',
                  retrieval_strategy=defaults.RETRIEVAL_STRATEGY,
                  ):
    """
    Retrieve from memory based on a query and retrieval weights.

    Modes:
        'sample': sample a single memory using the match scores as probabilities
        'argmax': retrieve the memory with the highest match score
        'softmax': return the weighted sum of all memories using the match scores as weights
    """
    # Unpack memories and query
    state_memories, context_memories, time_memories, reward_memories = memories
    state, context, time, _ = query

    # Compute the match scores for each key and memory
    state_match = match(state, state_memories, metric=metric)
    context_match = match(context, context_memories, metric=metric)
    time_match = match(time, time_memories, metric=metric)

    if retrieval_strategy == 'multiplicative':
        total_match = ((state_retrieval_weight * state_match + 1.) *
                       (context_retrieval_weight * context_match + 1.) *
                       (time_retrieval_weight * time_match + 1.) - 1.) / temperature
    else:
        total_match = (state_retrieval_weight * state_match +
                       context_retrieval_weight * context_match +
                       time_retrieval_weight * time_match) / temperature

    total_match = utils.safe_softmax(total_match, softmax_threshold)

    if mode == 'sample':
        index = np.random.choice(len(total_match), p=total_match)
        return (
            state_memories[index],
            context_memories[index],
            time_memories[index],
            reward_memories[index],
            index
        )

    if mode == 'argmax':
        index = int(np.argmax(total_match))
        return (
            state_memories[index],
            context_memories[index],
            time_memories[index],
            reward_memories[index],
            index
        )

    if mode == 'softmax':
        # add axis for broadcasting: (N,) -> (N,1)
        w = total_match[:, None]

        state_ret = np.sum(w * state_memories, axis=0)
        context_ret = np.sum(w * context_memories, axis=0)
        time_ret = np.sum(w * time_memories, axis=0)
        reward_ret = np.sum(total_match * reward_memories.squeeze(), axis=0)

        return state_ret, context_ret, time_ret, reward_ret, 0

    raise NotImplementedError(f'Mode {mode} not implemented. Try one of [\"sample\", \"argmax\", \"softmax\"].')


def sample_memory_sequential(memories,
                             starting_query,
                             n_simulations=1,  # number of simulation trajectories
                             n_steps=3,  # number of steps per simulation trajectory
                             time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
                             state_integration_rate=defaults.STATE_INTEGRATION_RATE,
                             model_based_ness=defaults.MODEL_BASED_NESS,
                             context_d=defaults.STATE_SIZE,
                             state_d=defaults.STATE_SIZE,
                             time_d=defaults.TIME_SIZE,
                             metric=defaults.METRIC,
                             mode=defaults.SAMPLE_MODE,
                             retrieval_strategy=defaults.RETRIEVAL_STRATEGY,
                             ):
    # Unpack memories and query
    state_memories, context_memories, time_memories, reward_memories = memories
    starting_state, starting_context, starting_time, _ = starting_query

    # Initialize arrays to hold retrieved values (for all simulations and steps)
    retrieved_states = np.zeros((n_simulations, n_steps, state_d))
    retrieved_contexts = np.zeros((n_simulations, n_steps, context_d))
    retrieved_times = np.zeros((n_simulations, n_steps, time_d))
    retrieved_rewards = np.zeros((n_simulations, n_steps))

    retrieved_memory_idxs = np.zeros((n_simulations, n_steps), dtype=int)

    # simulate n_simulation trajectories
    for sim_idx in range(n_simulations):
        # initialize state, context, and time for this simulation
        state_sim = starting_state
        context_sim = starting_context
        time_sim = starting_time

        for step_idx in range(n_steps):
            # retrieve from memory
            memories = (state_memories, context_memories, time_memories, reward_memories)  # tuple of memories

            context_retrieval_weight_sim = 0.
            if retrieval_strategy == 'multiplicative':
                state_retrieval_weight_sim = 1. / time_retrieval_weight if time_retrieval_weight > 0 else 1.
            else:
                state_retrieval_weight_sim = 1 - time_retrieval_weight

            queries = (state_sim, context_sim, time_sim, 0)

            # retrieve reward based on current state (state_retrieval_weight_sim == 0)
            retrieved_state, retrieved_context, _, retrieved_reward, _ = \
                sample_memory(memories,
                              queries,
                              state_retrieval_weight_sim,
                              context_retrieval_weight_sim,
                              time_retrieval_weight,
                              metric=metric,
                              mode=mode)

            # project the next context based on context and state

            context_sim = context_sim * (1 - model_based_ness) + model_based_ness * retrieved_context

            context_sim = project_next_context(
                context_sim,
                retrieved_state,
                state_integration_rate,
            )

            state_retrieval_weight_sim = 0.
            if retrieval_strategy == 'multiplicative':
                context_retrieval_weight_sim = 1. / time_retrieval_weight if time_retrieval_weight > 0 else 1.
            else:
                context_retrieval_weight_sim = 1 - time_retrieval_weight

            queries = (retrieved_state, context_sim, time_sim, 0)

            # retrieve state based on projected context (state_retrieval_weight_sim == 0)

            retrieved_state, retrieved_context, retrieved_time, _, retrieved_memory_idx = \
                sample_memory(memories,
                              queries,
                              state_retrieval_weight_sim,
                              context_retrieval_weight_sim,
                              time_retrieval_weight,
                              metric=metric,
                              mode=mode)

            state_sim = retrieved_state

            # store the retrieved values
            retrieved_states[sim_idx, step_idx] = retrieved_state
            retrieved_contexts[sim_idx, step_idx] = retrieved_context
            retrieved_times[sim_idx, step_idx] = retrieved_time
            retrieved_rewards[sim_idx, step_idx] = retrieved_reward.item()
            retrieved_memory_idxs[sim_idx, step_idx] = retrieved_memory_idx

    return retrieved_states, retrieved_contexts, retrieved_times, retrieved_rewards, retrieved_memory_idxs


def estimate_reward_from_starting_state(memories,
                                        starting_state,
                                        time,
                                        n_simulations=1,  # number of simulation trajectories
                                        n_steps=3,  # number of steps per simulation trajectory
                                        time_retrieval_weight=defaults.TIME_RETRIEVAL_WEIGHT,
                                        state_integration_rate=defaults.STATE_INTEGRATION_RATE,
                                        model_based_ness=defaults.MODEL_BASED_NESS,
                                        context_d=defaults.STATE_SIZE,
                                        state_d=defaults.STATE_SIZE,
                                        time_d=defaults.TIME_SIZE,
                                        mode='softmax',
                                        metric='dot_product',
                                        return_trajectories=False):
    starting_context = memories[1][-1]
    starting_time = time
    starting_query = (starting_state, starting_context, starting_time, None)
    sampled_trajectories = sample_memory_sequential(
        memories=memories,
        starting_query=starting_query,
        n_simulations=n_simulations,
        n_steps=n_steps,
        time_retrieval_weight=time_retrieval_weight,
        state_integration_rate=state_integration_rate,
        model_based_ness=model_based_ness,
        context_d=context_d,
        state_d=state_d,
        time_d=time_d,
        metric=metric,
        mode=mode,
    )
    estimated_reward = sampled_trajectories[3].sum(axis=-1).mean()  # Sum over steps in each sim and avg over sims
    if return_trajectories:
        return estimated_reward, sampled_trajectories
    else:
        return estimated_reward
