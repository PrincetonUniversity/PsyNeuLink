import os
import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from defaults import PARAMS, get_folder


# HEATMAP
def accumulate_reval_scores(df_):
    df = df_.loc[[0], ['state_integration_rate', 'time_retrieval_weight']].copy()
    df['reval_scores_reward'] = df_['reval_scores_reward']
    df['reval_scores_transition'] = df_['reval_scores_transition']
    df['reval_scores_total'] = df_['reval_scores_reward'] + df_['reval_scores_transition']
    return df


def create_average_scores(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/revaluation/data/{_root}"

    dirs = os.listdir(root)

    dirs = [d for d in dirs if not d.startswith('.') and not os.path.isfile(os.path.join(root, d))]

    df = pd.DataFrame(columns=['state_integration_rate', 'time_retrieval_weight', 'accumulated_reward'])
    for d in dirs:
        path = os.path.join(root, d)
        files = [f for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f)) and f.endswith('.csv') and not f.startswith('.')]
        for file in files:
            file_path = os.path.join(path, file)
            df_ = pd.read_csv(file_path)
            df_accumulated = accumulate_reval_scores(df_)
            df = pd.concat([df, df_accumulated], ignore_index=True)

    df_averaged = df.groupby(['state_integration_rate', 'time_retrieval_weight']).mean().reset_index()
    df_avg_sem = df.groupby(['state_integration_rate', 'time_retrieval_weight']).sem().reset_index()
    for el in ['reval_scores_reward', 'reval_scores_transition', 'reval_scores_total']:
        df_averaged[f'{el}_sem'] = df_avg_sem[el]
    averaged_path = f"../results/revaluation/{_root}"
    root = Path(averaged_path)
    root.mkdir(parents=True, exist_ok=True)
    df_averaged.to_csv(os.path.join(averaged_path, 'averaged_scores.csv'), index=False)


def plot_heatmap(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/revaluation/{_root}"
    df_averaged = pd.read_csv(os.path.join(root, 'averaged_scores.csv'))
    pivot_table = df_averaged.pivot(
        index='state_integration_rate',
        columns='time_retrieval_weight',
        values='reval_scores_total'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.gca().invert_yaxis()
    plt.title('Average Total Revaluation Score')
    plt.xlabel('Time Retrieval Weight')
    plt.ylabel('State Integration Rate')
    plt.savefig(f'{root}/revaluation_heatmap.png')
    plt.savefig(f'{root}/revaluation_heatmap.pdf')


def plot_scores(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/revaluation/{_root}"

    df = pd.read_csv(os.path.join(root, "averaged_scores.csv"))

    ir_values = sorted(df["state_integration_rate"].unique(), reverse=True)
    tw_values = sorted(df["time_retrieval_weight"].unique())

    BLUE = "#0399C6"
    YELLOW = "#FF8F27"

    x_ = np.array([0, 1])

    fig, axes = plt.subplots(
        len(ir_values),
        len(tw_values),
        figsize=(4.5 * len(tw_values), 3.5 * len(ir_values)),
        sharey=True,
        squeeze=False,
    )

    r_max = 18. #max(df["reval_scores_reward"].max(), df['reval_scores_transition'].max())

    for i, ir in enumerate(ir_values):
        for j, tw in enumerate(tw_values):
            ax = axes[i, j]
            sub = df[
                (df["state_integration_rate"] == ir) &
                (df["time_retrieval_weight"] == tw)
                ]

            if sub.empty:
                ax.axis("off")
                continue

            rr = sub['reval_scores_reward'].iloc[0] / r_max
            rt = sub['reval_scores_transition'].iloc[0] / r_max

            rr_sem = sub['reval_scores_reward_sem'].iloc[0] / r_max
            rt_sem = sub['reval_scores_transition_sem'].iloc[0] / r_max

            ax.bar(x_[0], rr, color=BLUE)
            ax.bar(x_[1], rt, color=YELLOW)
            ax.axhline(y=.5, color="black", linestyle="--")


            ax.errorbar(x_[0], rr, yerr=rr_sem, fmt="none", capsize=3)
            ax.errorbar(x_[1], rt, yerr=rt_sem, fmt="none", capsize=3)

            ax.set_xlim(-0.8, 1.8)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Reward\nRevaluation", "Transition\nRevaluation"], fontsize=8)

            ax.set_title(f"SIR={ir:g}, TRW={tw:g}", fontsize=9)

            if j == 0:
                ax.set_ylabel("Revaluation Scores")

    fig.tight_layout()

    plt.savefig(f"{root}/revaluation_panel.png")
    plt.savefig(f"{root}/revaluation_panel.pdf")


if __name__ == "__main__":
    create_average_scores(PARAMS)
    plot_heatmap(PARAMS)
    plot_scores(PARAMS)
