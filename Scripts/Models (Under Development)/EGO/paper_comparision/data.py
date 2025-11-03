import random

import numpy as np
from torch.utils.data import dataset
import torch

import utils


class CSWDataset(dataset.Dataset):
    """
    A custom dataset class for generating samples based on different contexts.

    Args:
        n_samples_per_context (list): A list of integers representing the number of samples to generate for each context.
        contexts_to_load (list): A list of integers representing the contexts to load.
    """

    def __init__(self, n_samples_per_context, contexts_to_load) -> None:
        super().__init__()

        self.n_samples_per_context = n_samples_per_context
        self.all_trials = []

        for i, context in enumerate(contexts_to_load):
            for _ in range(n_samples_per_context[i]):
                if context == 0:
                    self.all_trials.extend(self.gen_context1())
                else:
                    self.all_trials.extend(self.gen_context2())

        self.ys = utils.one_hot_encode(np.array(self.all_trials), 11)
        self.ys = self.ys.reshape((-1, 11))
        self.xs = torch.cat([torch.zeros((1, 11)), self.ys[:-1]], dim=0)

        self.contexts = self.xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]

    def gen_context1(self):
        """
        Generate samples for context 1 based on the given probabilities.

        Returns:
            list: A list of states representing the generated samples.
        """
        states = [9, random.choice([1, 2])]
        for _ in range(3):
            states.append(states[-1] + 2)
        return states

    def gen_context2(self):
        """
        Generate samples for context 2 based on the given probabilities.

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


def gen_data_loader(paradigm):
    if paradigm == 'blocked':
        contexts_to_load = [0, 1, 0, 1] + [random.randint(0, 1) for _ in range(40)]
        n_samples_per_context = [40, 40, 40, 40] + [1] * 40
        ds = CSWDataset(n_samples_per_context, contexts_to_load)
    elif paradigm == 'interleaved':
        contexts_to_load = [0, 1] * 80 + [random.randint(0, 1) for _ in range(40)]
        n_samples_per_context = [1] * 160 + [1] * 40
        ds = CSWDataset(n_samples_per_context, contexts_to_load)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
