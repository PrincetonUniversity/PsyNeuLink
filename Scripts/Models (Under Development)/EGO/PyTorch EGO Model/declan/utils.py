import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns



def plot_results(df, plot_title):
    # Plot the results
    fig, ax = plt.subplots(figsize=(8, 4))
    palette={'interleaved': 'grey', 'blocked': 'black', 'compositional_words': 'gray', 'compositional_words_foil': 'black', 'early': 'blue', 'middle': 'orange', 'late': 'green'}
    sns.lineplot(data=df, x='trial', y='probability', hue='paradigm', errorbar='se', palette=palette)

    # color the experiment phases
    plt.axvspan(0, 40, color='blue', alpha=0.1)
    plt.axvspan(40, 80, color='orange', alpha=0.1)
    plt.axvspan(80, 120, color='blue', alpha=0.1)
    plt.axvspan(120, 160, color='orange', alpha=0.1)
    plt.axvspan(160, 200, color='green', alpha=0.1)

    # add legend with colored background to distinguish the phases 
    handles, labels = ax.get_legend_handles_labels()
    block1_patch = Patch(facecolor='orange', label='Block1', alpha=0.4)
    block2_patch = Patch(facecolor='blue', label='Block2', alpha=0.4)
    block3_patch = Patch(facecolor='green', label='Block3', alpha=0.4)
    handles.extend([block1_patch, block2_patch, block3_patch])
    labels.extend(['Context 1', 'Context 2', 'Test phase'])
    ax.legend(handles=handles, labels=labels)
    
    # clean up the plot
    plt.ylim(0, 1.1)
    plt.xlim(0, 200)
    plt.title(plot_title)
    plt.xlabel('Trial')
    plt.ylabel('Probability of correct response')
    return fig

class Map(dict):
    '''
    Class that extends dictionry to allow for dot access of run parameters.

    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])
    m.first_name # Eduardo
    m['first_name'] # Eduardo
    '''
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(Map, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(Map, self).__delitem__(key)
        del self.__dict__[key]


def safe_softmax(t, eps=1e-6, **kwargs):
    '''
    Softmax function that always sums to 1 or less. Handles occasional numerical errors in torch's softmax.
    '''
    return torch.softmax(t, **kwargs) #-eps


def set_random_seed(params):
    try:
        seed = params.seed
    except AttributeError:
        seed = params
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def one_hot_encode(labels, num_classes):
    '''
    One hot encode labels and convert to tensor.
    '''
    return torch.tensor((np.arange(num_classes) == labels[..., None]).astype(float),dtype=torch.float32)