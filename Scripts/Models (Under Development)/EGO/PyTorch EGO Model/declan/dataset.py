import random

import numpy as np
from torch.utils.data import dataset
import torch

import declan.utils as utils


class FusiDataset(dataset.Dataset):
    '''
    A custom dataset class for generating samples from Fusi's hippocampal transitive inference task.
    '''

    def __init__(self, n_samples_per_context, contexts_to_load):
        super().__init__()
        self.context1_xs = np.arange(4)
        self.context1_ys = np.array([0, 0, 1, 1])
        self.context2_xs = np.arange(4)
        self.context2_ys = np.array([1, 1, 0, 0])
        self.n_samples_per_context = n_samples_per_context
        self.all_xs = []
        self.all_ys = []
        for i, context in enumerate(contexts_to_load):
            for _ in range(n_samples_per_context[i]):
                if context == 0:
                    xs, ys = self.gen_context1()
                    self.all_xs.extend(xs)
                    self.all_ys.extend(ys)
                else:
                    xs, ys = self.gen_context2()
                    self.all_xs.extend(xs)
                    self.all_ys.extend(ys)
        self.xs = utils.one_hot_encode(np.array(self.all_xs), max(self.all_xs) + 1)
        self.ys = utils.one_hot_encode(np.array(self.all_ys), max(self.all_ys) + 1)
        self.xs = self.xs.reshape((-1, max(self.all_xs) + 1))
        self.ys = self.ys.reshape((-1, max(self.all_ys) + 1))
        self.contexts = self.xs  # won't end up being used in the experiment

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]

    def gen_context1(self):
        n_states = len(self.context1_ys)
        shuffled_idx = np.random.choice(n_states, size=n_states, replace=False)
        return self.context1_xs[shuffled_idx], self.context1_ys[shuffled_idx]

    def gen_context2(self):
        n_states = len(self.context1_ys)
        shuffled_idx = np.random.choice(n_states, size=n_states, replace=False)
        return self.context2_xs[shuffled_idx], self.context2_ys[shuffled_idx]


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

        self.xs = utils.one_hot_encode(np.array(self.all_trials), 11)
        self.xs = self.xs.reshape((-1, 11))
        self.ys = torch.cat([self.xs[1:], utils.one_hot_encode(np.array([0]), 11)], dim=0)

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


class CompositionalWords(dataset.Dataset):
    """
    A dataset class for generating compositional words.

    Args:
        n_samples_per_context (list): A list of integers representing the number of samples per context.
        contexts_to_load (list): A list of integers representing the contexts to load.
        probs (list, optional): A list of probabilities for generating the compositional words. Defaults to [0.95, 0.6, 0.95].

    Attributes:
        n_samples_per_context (list): A list of integers representing the number of samples per context.
        all_trials (list): A list to store all generated trials.
        xs (numpy.ndarray): An array of one-hot encoded input states.
        ys (torch.Tensor): A tensor of one-hot encoded output states.
        contexts (numpy.ndarray): An array of one-hot encoded context states.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Returns the input, context, and output states at the given index.
        gen_context1(probs): Generates compositional words for context 1.
        gen_context2(probs): Generates compositional words for context 2.
        gen_context3(): Generates compositional words for context 3.
    """

    def __init__(self, n_samples_per_context, contexts_to_load, probs=[0.95, 0.6, 0.95]) -> None:
        super().__init__()

        self.n_samples_per_context = n_samples_per_context
        self.all_trials = []

        for i, context in enumerate(contexts_to_load):
            for _ in range(n_samples_per_context[i]):
                if context == 0:
                    self.all_trials.extend(self.gen_context1(probs))
                elif context == 1:
                    self.all_trials.extend(self.gen_context2(probs))
                else:
                    self.all_trials.extend(self.gen_context3())

        self.xs = utils.one_hot_encode(np.array(self.all_trials), 11)
        self.xs = self.xs.reshape((-1, 11))
        self.ys = torch.cat([self.xs[1:], utils.one_hot_encode(np.array([0]), 11)], dim=0)

        # Remove the last transition since there's no next state available
        self.xs = self.xs[:-1]
        self.ys = self.ys[:-1]
        self.contexts = self.xs

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]

    def gen_context1(self, probs):
        states = [9, random.choice([1, 2])]
        for p in probs:
            if random.random() < p:
                states.append(states[-1] + 2)
            else:
                if states[-1] % 2 == 0:
                    states.append(states[-1] + 1)
                else:
                    states.append(states[-1] + 3)
        return states

    def gen_context2(self, probs):
        states = [10, random.choice([1, 2])]
        for p in probs:
            if random.random() < p:
                if states[-1] % 2 == 0:
                    states.append(states[-1] + 1)
                else:
                    states.append(states[-1] + 3)
            else:
                states.append(states[-1] + 2)
        return states

    def gen_context3(self):
        states = [9, random.choice([1, 2])]
        if states[-1] == 1:
            states.extend([3, 6, 8])
        else:
            states.extend([4, 5, 7])
        return states

