import random

import torch
import numpy as np
import matplotlib.pyplot as plt

def safe_softmax(t, threshold=0.01, **kwargs):
    """
    Softmax function that always sums to 1 or less. Handles occasional numerical errors in torch's softmax.
    Nullifies values below the given threshold.
    """
    v = t

    # Apply mask: only include values greater than mask_threshold
    if threshold is not None:
        v = torch.where(abs(t) > threshold, v, torch.tensor(-torch.inf, device=t.device))

    # Shift by the global max to avoid extreme values
    if torch.any(v != -torch.inf):
        v = v - torch.max(v)

    # Exponential
    v = torch.exp(v)

    # Normalize (to sum to 1)
    if not v.any():
        return v
    else:
        return v / torch.sum(v)


def normalized(vector):
    """
    Normalize the provided vector to unit length
    """
    return vector / vector.norm(dim=-1, keepdim=True)


def one_hot_encode(labels, num_classes):
    """
    One hot encode labels and convert to tensor.
    """
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float), dtype=torch.float32)


def plot_trials(trial_states, trial_rewards):
    """
    Simple visualization: each trial is one column.
    Inputs:
      - trial_states: list/array of one-hot vectors (shape: num_trials x state_dim)
      - trial_rewards: list/array of scalars (length num_trials)
    Result: image of shape (state_dim + 1, num_trials) where last row is reward.
    """
    ts = np.asarray(trial_states)
    tr = np.asarray(trial_rewards).reshape(-1) / 10

    num_trials = ts.shape[0]
    state_dim = ts.shape[1]

    print(num_trials, state_dim)

    # Build grid: state rows then reward row
    res = []
    for i in range(num_trials):
        _t = np.concatenate([ts[i], [tr[i]]])
        res.append(_t)

    grid = np.vstack(res)  # shape: (state_dim + 1, num_trials)

    y_labels = [f'T{j}' for j in range(num_trials)]

    fig, ax = plt.subplots(figsize=(max(6, state_dim * 0.5), max(3, num_trials * 0.4)))
    ax.imshow(grid, aspect='auto', interpolation='nearest', cmap='viridis')

    ax.set_xticks(np.arange(grid.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(grid.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.tick_params(which='minor', length=0)
    ax.axvline(x=grid.shape[1] - 1.5, color='white', linewidth=1.0, linestyle='-', alpha=0.7, zorder=5)

    ax.set_xlabel("State + Reward (last column)")
    ax.set_xticks(np.arange(state_dim + 1))
    ax.set_xticklabels(list(range(state_dim)) + ['R'])
    ax.set_yticks(np.arange(num_trials))
    ax.set_yticklabels(y_labels)

    plt.tight_layout()
    plt.show()

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)