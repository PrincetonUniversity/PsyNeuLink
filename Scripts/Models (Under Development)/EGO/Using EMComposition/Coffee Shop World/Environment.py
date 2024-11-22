import numpy as np
import torch
from torch.utils.data import dataset
from random import randint

def one_hot_encode(labels, num_classes):
    """
    One hot encode labels and convert to tensor.
    """
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float),dtype=torch.float32)

class DeterministicCSWDataset(dataset.Dataset):
    def __init__(self, n_samples_per_context, contexts_to_load) -> None:
        super().__init__()
        raw_xs = np.array([
            [[9,1,3,5,7],[9,2,4,6,8]],
            [[10,1,4,5,8],[10,2,3,6,7]]
        ])
        
        item_indices = np.random.choice(raw_xs.shape[1],sum(n_samples_per_context),replace=True)
        task_names = [0,1] # Flexible so these can be renamed later
        task_indices = [task_names.index(name) for name in contexts_to_load]
        
        context_indices = np.repeat(np.array(task_indices),n_samples_per_context)
        self.xs = one_hot_encode(raw_xs[context_indices,item_indices],11)

        self.xs = self.xs.reshape((-1,11))
        self.ys = torch.cat([self.xs[1:],one_hot_encode(np.array([0]),11)],dim=0)
        context_indices = np.repeat(np.array(task_indices),[x*5 for x in n_samples_per_context])
        self.contexts = one_hot_encode(context_indices, len(task_names))

        # Remove the last transition since there's no next state available
        self.xs = self.xs[:-1]
        self.ys = self.ys[:-1]
        self.contexts = self.contexts[:-1]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.contexts[idx], self.ys[idx]

def generate_dataset(condition='Blocked'):
    # Generate the dataset for either the blocked or interleaved condition
    if condition=='Blocked':
        contexts_to_load = [0,1,0,1] + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [40,40,40,40] + [1]*40
    elif condition == 'Interleaved':
        contexts_to_load = [0,1]*80 + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [1]*160 + [1]*40
    else:
        raise ValueError(f'Unknown dataset condition: {condition}')

    return DeterministicCSWDataset(n_samples_per_context, contexts_to_load)
