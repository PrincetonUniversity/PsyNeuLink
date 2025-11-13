"""
Change from original:

- Fix bugs for time code and estimation (see changeSet.md)
- Use softmax with threshold + memory with small value initialized
"""

import torch
import numpy as np

import params as params
import utils as utils


def match(key, memories):
    """
    Match between key (query) and list of vectors (memories) as cosine similarity.
    Examples:
        >>> _key = torch.tensor([1.0, 0.0])
        >>> _memories = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7071, 0.7071]])
        >>> match(key, memories)
        tensor([1.0000, 0.0000, 0.7071])
    """
    return (utils.normalized(memories) * utils.normalized(key)).sum(-1)


def torch_append(tensor, value):
    """
    Append a new value to a tensor along the first dimension. Used to build memories.
    """
    # make sure value and tensor are both torch tensors
    if tensor is None or isinstance(tensor, torch.Tensor):
        pass
    else:
        tensor = torch.tensor(tensor, dtype=torch.float)
    if value is None or isinstance(value, torch.Tensor):
        pass
    else:
        value = torch.tensor(value, dtype=torch.float)
    if tensor is None:
        tensor = value.detach().clone().unsqueeze(0)
    else:
        tensor = torch.cat([tensor, value.detach().clone().unsqueeze(0)], axis=0)
    return tensor


def project_next_context(old_context,
                         state,
                         new_context,
                         old_context_integration_rate,
                         state_integration_rate,
                         new_context_integration_rate):
    """
    Projected next context as weighted combination of old context, new state, and new context.
    """
    return old_context_integration_rate * old_context + \
        state_integration_rate * state + \
        new_context_integration_rate * new_context


def gen_memories(visited_states,
                 rewards,
                 time_sequence,
                 old_context_integration_rate,
                 state_integration_rate,
                 retrieved_context_integration_rate,
                 state_retrieval_weight,
                 context_retrieval_weight,
                 time_retrieval_weight,
                 context_d=params.CONTEXT_SIZE):
    """
    Generate episodic memories from a sequence of visited states, rewards, and time codes.

    Integration rates determine how much old context, state and retrieved context contribute to the
    updated context representation.

    Retrieval weights determine the importance of state, context, and time during retrieval from memory (to
    get the retrieved context).
    """

    # Initialize empty memories
    memory_capacity = len(visited_states)
    if isinstance(visited_states, torch.Tensor):
        state_d = visited_states.size()[1]
    elif isinstance(visited_states, np.ndarray):
        state_d = visited_states.shape[1]
    else:
        raise TypeError("Expected torch.Tensor or numpy.ndarray")
    time_d = len(time_sequence[0])
    reward_d = 1

    state_memories, context_memories, time_memories, reward_memories = (
        torch.ones(memory_capacity, state_d) * params.MEMORY_INIT,
        torch.ones(memory_capacity, context_d) * params.MEMORY_INIT,
        torch.ones(memory_capacity, time_d) * params.MEMORY_INIT,
        torch.ones(memory_capacity, reward_d) * params.MEMORY_INIT
    )

    # Initialize the context representation
    context_rep = torch.zeros((context_d,), dtype=torch.float) + .01

    for t in range(len(visited_states)):
        state_cur = visited_states[t]
        time_code = time_sequence[t]
        reward_cur = rewards[t]

        # retrieve context
        memories = (state_memories, context_memories, time_memories, reward_memories)  # tuple of memories
        query = (state_cur, context_rep, time_code, reward_cur)  # tuple of keys

        _, retrieved_context, _, _, _ = sample_memory(memories,
                                                      query,
                                                      state_retrieval_weight,
                                                      context_retrieval_weight,
                                                      time_retrieval_weight,
                                                      mode='softmax')

        # store (not integrated yet) context with current state
        if isinstance(visited_states[t], torch.Tensor):
            state_memories[t] = visited_states[t].detach().clone()
        else:
            state_memories[t] = torch.tensor(visited_states[t])
        context_memories[t] = context_rep
        if isinstance(time_sequence[t], torch.Tensor):
            time_memories[t] = time_sequence[t].detach().clone()
        else:
            time_memories[t] = torch.tensor(time_sequence[t])
        reward_memories[t] = reward_cur

        # Integration of context
        context_rep = context_rep * old_context_integration_rate \
                      + retrieved_context * retrieved_context_integration_rate \
                      + state_cur * state_integration_rate

    return state_memories, context_memories, time_memories, reward_memories


def sample_memory(memories,
                  query,
                  state_retrieval_weight,
                  context_retrieval_weight,
                  time_retrieval_weight,
                  temperature=params.TEMPERATURE,
                  mode='sample'):
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
    state_match = match(state, state_memories)
    context_match = match(context, context_memories)
    time_match = match(time, time_memories)

    total_match = (state_retrieval_weight * state_match +
                   context_retrieval_weight * context_match +
                   time_retrieval_weight * time_match) / temperature

    total_match = utils.safe_softmax(total_match, params.SOFTMAX_THRESHOLD)

    if mode == 'sample':
        index = torch.multinomial(total_match, 1).item()
        return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
    if mode == 'argmax':
        index = total_match.argmax().item()
        return state_memories[index], context_memories[index], time_memories[index], reward_memories[index], index
    if mode == 'softmax':
        return (total_match.unsqueeze(-1) * state_memories).sum(0), (total_match.unsqueeze(-1) * context_memories).sum(
            0), (total_match.unsqueeze(-1) * time_memories).sum(0), (total_match * reward_memories).sum(0), 0
    raise NotImplementedError(f'Mode {mode} not implemented. Try one of ["sample", "argmax", "softmax"].')


def _sample_memory_sequential(memories,
                             starting_query,
                             n_simulations,  # number of simulation trajectories
                             n_steps,  # number of steps per simulation trajectory

                             state_retrieval_weight,
                             context_retrieval_weight,
                             time_retrieval_weight,

                             old_context_integration_rate,
                             state_integration_rate,
                             new_context_integration_rate,

                             context_d=params.STATE_SIZE,
                             state_d=params.STATE_SIZE,
                             time_d=params.TIME_SIZE,
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

        # initialize retrieval weights for this simulation
        state_retrieval_weight_sim = state_retrieval_weight
        context_retrieval_weight_sim = context_retrieval_weight
        time_retrieval_weight_sim = time_retrieval_weight

        for step_idx in range(n_steps):
            # retrieve from memory
            memories = (state_memories, context_memories, time_memories, reward_memories)  # tuple of memories
            queries = (state_sim, context_sim, time_sim, 0)

            # retrieve memory based on current query and current retrieval weights
            retrieved_state, retrieved_context, retrieved_time, retrieved_reward, retrieved_memory_idx = \
                sample_memory(memories,
                              queries,
                              state_retrieval_weight_sim,
                              context_retrieval_weight_sim,
                              time_retrieval_weight_sim,
                              mode='sample')

            # project the next context based on retrieved context and current state
            context_sim = project_next_context(
                context_sim,
                retrieved_state,
                retrieved_context,
                old_context_integration_rate,
                state_integration_rate,
                new_context_integration_rate
            )

            # store the retrieved values
            retrieved_states[sim_idx, step_idx] = retrieved_state.detach().clone().numpy()
            retrieved_contexts[sim_idx, step_idx] = retrieved_context.detach().clone().numpy()
            retrieved_times[sim_idx, step_idx] = retrieved_time.detach().clone().numpy()
            retrieved_rewards[sim_idx, step_idx] = retrieved_reward.item()
            retrieved_memory_idxs[sim_idx, step_idx] = retrieved_memory_idx

            # update the retrieval weights for the neext step (don't use state after first step)
            state_retrieval_weight_sim = 0.  # set state weight to 0 after first step

            # since state weight is 0, we re-normalize the other two weights to sum to 1
            _len = context_retrieval_weight_sim + time_retrieval_weight_sim
            if _len > 0:
                context_retrieval_weight_sim = context_retrieval_weight_sim / _len
                time_retrieval_weight_sim = time_retrieval_weight_sim / _len
            else:
                context_retrieval_weight_sim = 0
                time_retrieval_weight_sim = 0

    return retrieved_states, retrieved_contexts, retrieved_times, retrieved_rewards, retrieved_memory_idxs

def sample_memory_sequential(memories,
                             starting_query,
                             n_simulations,  # number of simulation trajectories
                             n_steps,  # number of steps per simulation trajectory

                             state_retrieval_weight,
                             context_retrieval_weight,
                             time_retrieval_weight,

                             old_context_integration_rate,
                             state_integration_rate,
                             new_context_integration_rate,

                             context_d=params.STATE_SIZE,
                             state_d=params.STATE_SIZE,
                             time_d=params.TIME_SIZE,
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

        # initialize retrieval weights for this simulation
        state_retrieval_weight_sim = state_retrieval_weight
        context_retrieval_weight_sim = context_retrieval_weight
        time_retrieval_weight_sim = time_retrieval_weight

        for step_idx in range(n_steps):
            # retrieve from memory
            memories = (state_memories, context_memories, time_memories, reward_memories)  # tuple of memories


            context_retrieval_weight_sim = 0.
            state_retrieval_weight_sim = .9
            time_retrieval_weight_sim = .1

            queries = (state_sim, context_sim, time_sim, 0)

            # retrieve memory based on current query and current retrieval weights
            retrieved_state, retrieved_context, retrieved_time, retrieved_reward, retrieved_memory_idx = \
                sample_memory(memories,
                              queries,
                              state_retrieval_weight_sim,
                              context_retrieval_weight_sim,
                              time_retrieval_weight_sim,
                              mode='sample')

            # project the next context based on retrieved context and current state
            context_sim = project_next_context(
                context_sim,
                # retrieved_state,
                state_sim,
                retrieved_context,
                old_context_integration_rate,
                state_integration_rate,
                new_context_integration_rate
            )

            context_retrieval_weight_sim = .9
            state_retrieval_weight_sim = 0.
            time_retrieval_weight_sim = .1


            queries = (state_sim, context_sim, time_sim, 0)

            # retrieve memory based on current query and current retrieval weights
            retrieved_state, retrieved_context, retrieved_time, _, retrieved_memory_idx = \
                sample_memory(memories,
                              queries,
                              state_retrieval_weight_sim,
                              context_retrieval_weight_sim,
                              time_retrieval_weight_sim,
                              mode='sample')

            state_sim = retrieved_state



            # store the retrieved values
            retrieved_states[sim_idx, step_idx] = retrieved_state.detach().clone().numpy()
            retrieved_contexts[sim_idx, step_idx] = retrieved_context.detach().clone().numpy()
            retrieved_times[sim_idx, step_idx] = retrieved_time.detach().clone().numpy()
            retrieved_rewards[sim_idx, step_idx] = retrieved_reward.item()
            retrieved_memory_idxs[sim_idx, step_idx] = retrieved_memory_idx

            # # update the retrieval weights for the neext step (don't use state after first step)
            # state_retrieval_weight_sim = 0.  # set state weight to 0 after first step
            #
            # # since state weight is 0, we re-normalize the other two weights to sum to 1
            # _len = context_retrieval_weight_sim + time_retrieval_weight_sim
            # if _len > 0:
            #     context_retrieval_weight_sim = context_retrieval_weight_sim / _len
            #     time_retrieval_weight_sim = time_retrieval_weight_sim / _len
            # else:
            #     context_retrieval_weight_sim = 0
            #     time_retrieval_weight_sim = 0

    return retrieved_states, retrieved_contexts, retrieved_times, retrieved_rewards, retrieved_memory_idxs


def estimate_reward_from_starting_state(memories,
                                        starting_state,
                                        n_simulations,  # number of simulation trajectories
                                        n_steps,  # number of steps per simulation trajectory
                                        state_retrieval_weight,
                                        context_retrieval_weight,
                                        time_retrieval_weight,

                                        old_context_integration_rate,
                                        state_integration_rate,
                                        new_context_integration_rate,

                                        context_d=params.STATE_SIZE,
                                        state_d=params.STATE_SIZE,
                                        time_d=params.TIME_SIZE,
                                        return_trajectories=False):
    starting_context = memories[1][-1]
    starting_time = memories[2][-1]
    starting_query = (starting_state, starting_context, starting_time, None)
    sampled_trajectories = sample_memory_sequential(memories,
                                                    starting_query,
                                                    n_simulations,  # number of simulation trajectories
                                                    n_steps,  # number of steps per simulation trajectory
                                                    state_retrieval_weight,
                                                    context_retrieval_weight,
                                                    time_retrieval_weight,
                                                    old_context_integration_rate,
                                                    state_integration_rate,
                                                    new_context_integration_rate,
                                                    context_d,
                                                    state_d,
                                                    time_d,
                                                    )
    estimated_reward = sampled_trajectories[3].sum(axis=-1).mean()  # Sum over steps in each sim and avg over sims
    if return_trajectories:
        return estimated_reward, sampled_trajectories
    else:
        return estimated_reward
