import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

import seaborn as sns
import matplotlib.pyplot as plt
from defaults import PARAMS, get_folder


# HEATMAP
def accumulate_reward(df_):
    df = df_.loc[[0], ['state_integration_rate', 'time_retrieval_weight']].copy()
    df['accumulated_reward'] = df_['reward'].sum()
    return df


def create_average_rewards(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/two_step/data/{_root}"

    dirs = os.listdir(root)

    dirs = [d for d in dirs if not d.startswith('.') and not os.path.isfile(os.path.join(root, d))]

    df = pd.DataFrame(columns=['state_integration_rate', 'time_retrieval_weight', 'accumulated_reward'])
    for d in tqdm(dirs):
        path = os.path.join(root, d)
        files = [f for f in os.listdir(path) if
                 os.path.isfile(os.path.join(path, f)) and f.endswith('.csv') and not f.startswith('.')]
        for file in tqdm(files, leave=False):
            file_path = os.path.join(path, file)
            df_ = pd.read_csv(file_path)
            df_accumulated = accumulate_reward(df_)
            df = pd.concat([df, df_accumulated], ignore_index=True)

    df_averaged = df.groupby(['state_integration_rate', 'time_retrieval_weight']).mean().reset_index()
    df_avg_sem = df.groupby(['state_integration_rate', 'time_retrieval_weight']).sem().reset_index()
    df_averaged['accumulated_reward_sem'] = df_avg_sem['accumulated_reward']
    averaged_path = f"../results/two_step/{_root}"
    root = Path(averaged_path)
    root.mkdir(parents=True, exist_ok=True)
    df_averaged.to_csv(os.path.join(averaged_path, 'averaged_results.csv'), index=False)


def plot_heatmap(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/two_step/{_root}"
    df_averaged = pd.read_csv(os.path.join(root, 'averaged_results.csv'))
    pivot_table = df_averaged.pivot(
        index='state_integration_rate',
        columns='time_retrieval_weight',
        values='accumulated_reward'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.gca().invert_yaxis()
    plt.title('Average Accumulated Reward')
    plt.xlabel('Time Retrieval Weight')
    plt.ylabel('State Integration Rate')
    plt.savefig(f'{root}/two_step_heatmap.png')
    plt.savefig(f'{root}/two_step_heatmap.pdf')


# STAY PROBABILITY
def compute_stay_stats_from_df(df):
    """
    Compute Daw 2011 stay probabilities from one participant dataframe.
    Uses conditioning on previous trial.
    Returns dict with RC, RR, UC, UR.
    """
    rc, rr, uc, ur = [], [], [], []

    for _, row in df.iterrows():
        stay = row.get("stay", None)
        if pd.isna(stay):
            continue

        reward = row.get("prev_reward")
        transition = row.get("prev_transition")

        if reward == 1 and transition == "common":
            rc.append(stay)
        elif reward == 1 and transition == "rare":
            rr.append(stay)
        elif reward == 0 and transition == "common":
            uc.append(stay)
        elif reward == 0 and transition == "rare":
            ur.append(stay)

    return {
        "RC": np.mean(rc) if rc else np.nan,
        "RR": np.mean(rr) if rr else np.nan,
        "UC": np.mean(uc) if uc else np.nan,
        "UR": np.mean(ur) if ur else np.nan,
    }


def create_average_stay_data(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/two_step/data/{_root}"

    dirs = os.listdir(root)
    dirs = [d for d in dirs if not d.startswith('.') and not os.path.isfile(os.path.join(root, d))]

    rows = []

    for d in dirs:
        path = os.path.join(root, d)
        files = [
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv') and not f.startswith('.')
        ]

        for file in files:
            file_path = os.path.join(path, file)
            df_ = pd.read_csv(file_path)

            stay_stats = compute_stay_stats_from_df(df_)

            row = {
                "state_integration_rate": df_.loc[0, "state_integration_rate"],
                "time_retrieval_weight": df_.loc[0, "time_retrieval_weight"],
                **stay_stats
            }

            rows.append(row)

    df_all = pd.DataFrame(rows)

    grouped = df_all.groupby(
        ["state_integration_rate", "time_retrieval_weight"]
    )

    df_mean = grouped.mean().reset_index()
    df_sem = grouped.sem().reset_index()

    # attach SEM columns
    for col in ["RC", "RR", "UC", "UR"]:
        df_mean[f"{col}_sem"] = df_sem[col]

    averaged_path = f"../results/two_step/{_root}"
    df_mean.to_csv(os.path.join(averaged_path, "averaged_stay_results.csv"), index=False)


def plot_stay_probs(kwargs):
    _root, _ = get_folder(**kwargs)
    root = f"../results/two_step/{_root}"

    df = pd.read_csv(os.path.join(root, "averaged_stay_results.csv"))

    ir_values = sorted(df["state_integration_rate"].unique(), reverse=True)
    tw_values = sorted(df["time_retrieval_weight"].unique())

    BLUE = "#2563eb"
    RED = "#dc2626"

    x_rewarded = np.array([0, 1])
    x_unrewarded = np.array([3, 4])

    fig, axes = plt.subplots(
        len(ir_values),
        len(tw_values),
        figsize=(4.5 * len(tw_values), 3.5 * len(ir_values)),
        sharey=True,
        squeeze=False,
    )

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

            r = sub.iloc[0]

            ax.bar(x_rewarded[0], r["RC"], color=BLUE)
            ax.bar(x_rewarded[1], r["RR"], color=RED)
            ax.bar(x_unrewarded[0], r["UC"], color=BLUE)
            ax.bar(x_unrewarded[1], r["UR"], color=RED)
            ax.axhline(y=.5, color="black", linestyle="--")

            ax.errorbar(x_rewarded[0], r["RC"], yerr=r["RC_sem"], fmt="none", capsize=3)
            ax.errorbar(x_rewarded[1], r["RR"], yerr=r["RR_sem"], fmt="none", capsize=3)
            ax.errorbar(x_unrewarded[0], r["UC"], yerr=r["UC_sem"], fmt="none", capsize=3)
            ax.errorbar(x_unrewarded[1], r["UR"], yerr=r["UR_sem"], fmt="none", capsize=3)

            ax.set_xlim(-0.8, 4.8)
            ax.set_ylim(0, 1)
            ax.set_xticks([0, 1, 3, 4])
            ax.set_xticklabels(["Common", "Rare", "Common", "Rare"], fontsize=8)

            ax.set_title(f"SIR={ir:g}, TRW={tw:g}", fontsize=9)

            if j == 0:
                ax.set_ylabel("Stay probability")

    fig.suptitle("Stay Probability")
    fig.tight_layout()

    plt.savefig(f"{root}/two_step_stay_panel.png")
    plt.savefig(f"{root}/two_step_stay_panel.pdf")


if __name__ == "__main__":
    create_average_rewards(PARAMS)
    create_average_stay_data(PARAMS)
    plot_heatmap(PARAMS)
    plot_stay_probs(PARAMS)
