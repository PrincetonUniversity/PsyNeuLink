import random

from torch.utils.data import dataset
import numpy as np

from random import randint
import torch


class CSWDataset(dataset.Dataset):
    """
    A custom dataset class for generating samples based on different contexts.

    Args:
        n_samples_per_context (list): A list of integers representing the number of samples to generate for each context.
        contexts_to_load (list): A list of integers representing the contexts to load.
        probs (list, optional): A list of probabilities for generating the samples. Defaults to [1, 1, 1].
    """

    def __init__(self, n_samples_per_context, contexts_to_load, probs=[1, 1, 1]) -> None:
        super().__init__()

        self.n_samples_per_context = n_samples_per_context
        self.all_trials = []

        for i, context in enumerate(contexts_to_load):
            for _ in range(n_samples_per_context[i]):
                if context == 0:
                    self.all_trials.extend(self.gen_context1(probs))
                else:
                    self.all_trials.extend(self.gen_context2(probs))

        self.xs = one_hot_encode(np.array(self.all_trials), 11)
        self.xs = self.xs.reshape((-1, 11))
        self.ys = torch.cat([self.xs[1:], one_hot_encode(np.array([0]), 11)], dim=0)

        # Remove the last transition since there's no next state available
        self.xs = self.xs[:-1]
        self.ys = self.ys[:-1]
        self.contexts = self.xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]

    def gen_context1(self, probs):
        """
        Generate samples for context 1 based on the given probabilities.

        Args:
            probs (list): A list of probabilities for generating the samples.

        Returns:
            list: A list of states representing the generated samples.
        """
        states = [9, random.choice([1, 2])]
        for p in probs:
            if random.random() <= p:
                states.append(states[-1] + 2)
            else:
                if states[-1] % 2 == 0:
                    states.append(states[-1] + 1)
                else:
                    states.append(states[-1] + 3)
        return states

    def gen_context2(self, probs):
        """
        Generate samples for context 2 based on the given probabilities.

        Args:
            probs (list): A list of probabilities for generating the samples.

        Returns:
            list: A list of states representing the generated samples.
        """
        states = [10, random.choice([1, 2])]
        for p in probs:
            if random.random() <= p:
                if states[-1] % 2 == 0:
                    states.append(states[-1] + 1)
                else:
                    states.append(states[-1] + 3)
            else:
                states.append(states[-1] + 2)
        return states


def gen_data_loader(paradigm, probs=[1., 1., 1.], n=1):
    if paradigm == 'tst':
        contexts_to_load = [0]
        n_samples_per_context = [n]
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=[1.])

    if paradigm == 'blocked':
        contexts_to_load = [0, 1, 0, 1] + [randint(0, 2) for _ in range(n)]
        n_samples_per_context = [n, n, n, n] + [1] * n
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    elif paradigm == 'interleaved':
        contexts_to_load = [0, 1] * (2 * n) + [randint(0, 2) for _ in range(n)]
        n_samples_per_context = [1] * (4 * n) + [1] * n
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


def one_hot_encode(labels, num_classes):
    """
    One hot encode labels and convert to tensor.
    """
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float), dtype=torch.float32)
