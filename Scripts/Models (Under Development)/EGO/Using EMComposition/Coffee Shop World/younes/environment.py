# This file contains the environment for the coffee shop world,
# which is used to generate state sequences for the EGO model.

from random import randint

import random

import numpy as np
import torch


def one_hot_encode(labels, num_classes):
    """
    One hot encode labels and convert to tensor.
    """
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float), dtype=torch.float64)


def gen_context1():
    """
    Generate samples for context 1 based on the given probabilities.

    Args:
        probs (list): A list of probabilities for generating the samples.

    Returns:
        list: A list of states representing the generated samples.
    """
    states = [9, random.choice([1, 2])]
    for _ in range(3):
        states.append(states[-1] + 2)
    return states


def gen_context2():
    """
    Generate samples for context 2 based on the given probabilities.

    Args:
        probs (list): A list of probabilities for generating the samples.

    Returns:
        list: A list of states representing the generated samples.
    """
    states = [10, random.choice([1, 2])]
    for _ in range(3):

        if states[-1] % 2 == 0:
            states.append(states[-1] + 1)
        else:
            states.append(states[-1] + 3)

    return states


def gen_run(n_samples_per_context, contexts_to_load):
    all_trials = []
    for i, context in enumerate(contexts_to_load):
        for _ in range(n_samples_per_context[i]):
            if context == 0:
                all_trials.extend(gen_context1())
            else:
                all_trials.extend(gen_context2())

    xs = one_hot_encode(np.array(all_trials), 11)
    xs = xs.reshape((-1, 11))
    return xs


def get_state_sequence(paradigm, **kwargs):
    if paradigm == 'blocked':
        contexts_to_load = [0, 1, 0, 1] + [randint(0, 1) for _ in range(40)]
        n_samples_per_context = [40, 40, 40, 40] + [1] * 40
        states = gen_run(n_samples_per_context, contexts_to_load)
    elif paradigm == 'interleaved':
        contexts_to_load = [0, 1] * 80 + [randint(0, 1) for _ in range(40)]
        n_samples_per_context = [1] * 160 + [1] * 40
        states = gen_run(n_samples_per_context, contexts_to_load)
    else:
        raise ValueError('Unknown paradigm.')
    return states
