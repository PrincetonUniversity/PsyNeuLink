# reproduce figure 8 in "Toward the Emergence of Intelligent Control:
# Episodic Generalization and Optimization"

from tqdm import tqdm
import random

from environment import get_state_sequence
from pytorch_ego.run import run_participant
from psyneulink_ego.ego_model_simple_spherical import construct_model, run_model

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RUN_TYPE = 'PNL'  # 'PyTorch' or 'PNL'

CONFIG = dict(
    tolerance=1e-10,
    nr_trials=10,
    seed=0,
    state_size=11,
    integration_rate=.8,
    state_retrieval_weight=None,
    previous_state_retrieval_weight=1.,
    context_retrieval_weight=1.,
    learning_rate=2.,
    memory_fill=.01,
    softmax_temperature=.05,
    softmax_threshold=.001,
    loss_spec_name='BinaryCrossEntropy',
    num_optimization_steps=1,
    nr_participants=1,
)


def set_random_seed(seed, **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def calc_prob(em_preds, test_ys):
    """
    Calculate the probability of the EM model predicting the correct next state through EM retrieval.
    """
    # Only consider the terminal three states (they are the only predictable transitions).
    em_preds_new, test_ys_new = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds_new * test_ys_new).sum(-1).mean(-1)
    trial_probs = (em_preds * test_ys)
    return em_probability, trial_probs



def get_df(results, states, seed, paradigm, **kwargs):
    performance_data = {
        'seed': [],
        'paradigm': [],
        'trial': [],
        'probability': [],
    }
    em_preds = np.vstack([results]).reshape(-1, 5, 11)

    test_ys = np.vstack([states]).reshape(-1, 5, 11)
    correct_prob_first, _ = calc_prob(em_preds, test_ys)

    performance_data['probability'].extend(correct_prob_first)
    performance_data['seed'].extend([seed] * len(correct_prob_first))
    performance_data['paradigm'].extend([paradigm] * len(correct_prob_first))
    performance_data['trial'].extend(list(range(len(correct_prob_first))))
    return pd.DataFrame(performance_data)


def run_experiment():
    performance_data = []

    torch.set_default_dtype(torch.float64)
    for _ in tqdm(range(CONFIG['nr_participants'])):
        for paradigm in ['interleaved', 'blocked']:
            CONFIG['seed']+=1
            set_random_seed(**CONFIG)
            states = get_state_sequence(**CONFIG, paradigm=paradigm)
            if RUN_TYPE == 'PNL':
                pnl_model, input_layer = construct_model(**CONFIG, memory_capacity=len(states))
                results = run_model(pnl_model, input_layer, states, **CONFIG)
            elif RUN_TYPE == 'PyTorch':
                results = run_participant(states, len(states), **CONFIG)
            else:
                raise NotImplementedError

            participant_df = get_df(results=results, states=states, paradigm=paradigm, **CONFIG)
            performance_data.append(participant_df)
    exp_df = pd.concat(performance_data).reset_index(drop=True)
    return exp_df

def plot_results(df, plot_title):
    palette = {
        "interleaved": "grey",
        "blocked": "black",
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for paradigm, color in palette.items():
        d = df[df["paradigm"] == paradigm]

        mean = d.groupby("trial")["probability"].mean()
        se = d.groupby("trial")["probability"].sem()

        ax.plot(mean.index, mean.values, color=color, label=paradigm)
        ax.fill_between(mean.index, mean - se, mean + se, color=color, alpha=0.2)

    # experiment phases
    ax.axvspan(0, 40, color="blue", alpha=0.1)
    ax.axvspan(40, 80, color="orange", alpha=0.1)
    ax.axvspan(80, 120, color="blue", alpha=0.1)
    ax.axvspan(120, 160, color="orange", alpha=0.1)
    ax.axvspan(160, 200, color="green", alpha=0.1)

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 1.1)
    ax.set_xlabel("Trial")
    ax.set_ylabel("P(correct)")
    ax.set_title(plot_title)

    ax.legend()

    plt.tight_layout()
    return fig

def main():
    df = run_experiment()
    file_name = f"IntegrationR({CONFIG['integration_rate']}) LearningR({CONFIG['learning_rate']}) OPTIMIZATION STEPS({CONFIG['num_optimization_steps']})"
    fig = plot_results(df, '')
    fig.show()
    fig.savefig(f"{file_name}.svg", format='svg')

if __name__ == '__main__':
    main()