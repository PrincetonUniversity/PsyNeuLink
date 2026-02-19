import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv('./data/twostep.csv')
    ids = pd.unique(data['sub'])
    ids = ids

    rewarded = []
    transition = []
    stay = []

    for _id in ids:
        subj_data = data[data['sub'] == _id]

        rewards = subj_data['r'].tolist()
        rockets_choice = subj_data['c1'].tolist()
        landed_planet = subj_data['s'].tolist()
        alien_choice = subj_data['c2'].tolist()

        # plt.plot(alien_choice)
        # plt.show()
        # return
        n = len(rockets_choice)

        for trial in range(n):
            transition.append('common' if rockets_choice[trial - 1] == landed_planet[trial - 1] else 'rare')
            rewarded.append(rewards[trial - 1] == 1)
            stay.append(rockets_choice[trial - 1] == rockets_choice[trial])

    df = pd.DataFrame({
            'rewarded': rewarded,
            'transition': transition,
            'stay': stay
        })

        # --- Aggregate to 4 conditions (means = stay probability, counts = trials) ---
    summary = (
        df.groupby(['rewarded', 'transition'])['stay']
        .agg(stay_prob='mean', n='size')
        .reindex(pd.MultiIndex.from_product([[0, 1], ['common', 'rare']],
                                            names=['rewarded', 'transition']))
        .reset_index()
    )
    print(summary)

    labels = ["U-C", "U-R", "R-C", "R-R"]  # Unrewarded/Common, Unrewarded/Rare, Rewarded/Common, Rewarded/Rare
    stay_prob = [
        summary.loc[(summary.rewarded == 0) & (summary.transition == 'common'), 'stay_prob'].iloc[0],
        summary.loc[(summary.rewarded == 0) & (summary.transition == 'rare'), 'stay_prob'].iloc[0],
        summary.loc[(summary.rewarded == 1) & (summary.transition == 'common'), 'stay_prob'].iloc[0],
        summary.loc[(summary.rewarded == 1) & (summary.transition == 'rare'), 'stay_prob'].iloc[0],
    ]
    counts = [
        summary.loc[(summary.rewarded == 0) & (summary.transition == 'common'), 'n'].iloc[0],
        summary.loc[(summary.rewarded == 0) & (summary.transition == 'rare'), 'n'].iloc[0],
        summary.loc[(summary.rewarded == 1) & (summary.transition == 'common'), 'n'].iloc[0],
        summary.loc[(summary.rewarded == 1) & (summary.transition == 'rare'), 'n'].iloc[0],
    ]

    x = np.arange(4)

    # --- Plot: bars are stay probability; "hist" is the counts on a twin axis ---
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.bar(x, stay_prob)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("P(stay)")
    ax.set_title(f"Stay probability by reward × transition (sub={_id})")

    ax2 = ax.twinx()
    ax2.step(np.r_[x, x[-1] + 1], np.r_[counts, counts[-1]], where='post', linewidth=2)
    ax2.set_ylabel("Trial count (hist-style)")
    ax2.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 1)

    # Optional: annotate counts on bars
    for i, (p, n_) in enumerate(zip(stay_prob, counts)):
        if not np.isnan(p):
            ax.text(i, min(p + 0.03, 0.98), f"n={n_}", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()

    # plot
import pandas as pd
import numpy as np


def main_data():
    data = pd.read_csv('./data/twostep.csv')
    ids = pd.unique(data['sub'])

    per_subject_rows = []

    for _id in ids:
        subj_data = data[data['sub'] == _id]

        rewards = subj_data['r'].to_numpy()
        rockets_choice = subj_data['c1'].to_numpy()
        landed_planet = subj_data['s'].to_numpy()

        rewarded = []
        transition = []
        stay = []

        n = len(rockets_choice)

        for trial in range(1, n):
            transition.append('common' if rockets_choice[trial - 1] == landed_planet[trial - 1] else 'rare')
            rewarded.append(int(rewards[trial - 1] == 1))
            stay.append(int(rockets_choice[trial - 1] == rockets_choice[trial]))

        df = pd.DataFrame({
            'rewarded': rewarded,
            'transition': transition,
            'stay': stay
        })

        summary = (
            df.groupby(['rewarded', 'transition'])['stay']
            .mean()
            .reindex(
                pd.MultiIndex.from_product(
                    [[0, 1], ['common', 'rare']],
                    names=['rewarded', 'transition']
                )
            )
        )

        per_subject_rows.append({
            "Rewarded Common": summary.loc[(1, 'common')],
            "Rewarded Rare": summary.loc[(1, 'rare')],
            "Rewarded UC": summary.loc[(0, 'common')],
            "Rewarded UR": summary.loc[(0, 'rare')],
        })

    subj_df = pd.DataFrame(per_subject_rows)

    means = subj_df.mean()
    sems = subj_df.sem()

    out = pd.DataFrame([{
        "Rewarded Common": means["Rewarded Common"],
        "Rewarded Common (SEM)": sems["Rewarded Common"],
        "Rewarded Rare": means["Rewarded Rare"],
        "Rewarded Rare (SEM)": sems["Rewarded Rare"],
        "Rewarded UC": means["Rewarded UC"],
        "Rewarded UC (SEM)": sems["Rewarded UC"],
        "Rewarded UR": means["Rewarded UR"],
        "Rewarded UR (SEM)": sems["Rewarded UR"],
    }])

    out.to_csv("human_data_two_step.csv", index=False)

    print("\nHuman two-step summary (model-format):")
    print(out.round(3))


main_data()

