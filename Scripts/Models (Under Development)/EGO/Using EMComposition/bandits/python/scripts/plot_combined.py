import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from defaults import PARAMS, get_folder


def minmax_normalize(df):
    return (df - df.min().min()) / (df.max().max() - df.min().min())


def plot_heatmap(kwargs):
    _root, _ = get_folder(**kwargs)
    root_reval = f"../results/revaluation/{_root}"
    df_averaged_reval = pd.read_csv(os.path.join(root_reval, 'averaged_totals.csv'))

    pivot_table_reval = df_averaged_reval.pivot(
        index='state_integration_rate',
        columns='time_retrieval_weight',
        values='reval_score')

    root_two_step = f"../results/two_step/{_root}"
    df_averaged_reward = pd.read_csv(os.path.join(root_two_step, 'averaged_results.csv'))
    pivot_table_two_step = df_averaged_reward.pivot(
        index='state_integration_rate',
        columns='time_retrieval_weight',
        values='accumulated_reward'
    )

    pivot_reval_norm = minmax_normalize(pivot_table_reval)
    pivot_two_step_norm = minmax_normalize(pivot_table_two_step)

    alpha = 0.5
    beta = 0.5

    combined = alpha * pivot_reval_norm + beta * pivot_two_step_norm

    plt.figure(figsize=(10, 8))
    sns.heatmap(combined, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.gca().invert_yaxis()
    plt.title('Combined Score')
    plt.xlabel('Time Retrieval Weight')
    plt.ylabel('State Integration Rate')

    _root, _ = get_folder(**kwargs)
    root_combined = f"../results/combined/{_root}"
    if not os.path.exists(root_combined):
        os.makedirs(root_combined)

    plt.savefig(f'{root_combined}/combined_heatmap.png')
    plt.savefig(f'{root_combined}/combined_heatmap.pdf')


if __name__ == "__main__":
    plot_heatmap(PARAMS)
