

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ego_revaluation import run_original
from ego_revaluation.config import defaults

DOT_PRODUCT = 'dot_product'
COSINE_SIMILARITY = 'cosine_similarity'

metric = COSINE_SIMILARITY
model_based_ness = 0.

state_integration_rates = [.2, .5, 1.]
time_weights = [0, .1, .2, .35, .4, 3.]


reward_reval_mean = []
reward_reval_sem = []

transition_reval_mean = []
transition_reval_sem = []

conditions = []


for s in state_integration_rates:
    for t in time_weights:

            data = run_original.run(200, metric=metric, model_based_ness=model_based_ness,
                        state_integration_rate=s,
time_retrieval_weight=t)

            condition = f'IR={s}_TIME={t}'

            PATH = f"reward_vs_transition_reval_{condition}.pdf"

# Model data
            reval_scores_reward = np.array(data['reval_scores_reward'])
            reval_scores_transition = np.array(data['reval_scores_transition'])

# get scaling factor: scale reward_reval to .5199
            scaling_factor = 1  # / reval_scores_reward.mean()

# scale data
            rs_reward_scaled = reval_scores_reward * scaling_factor
            rs_transition_scaled = reval_scores_transition * scaling_factor

# Means
            means = [
                rs_reward_scaled.mean(),
                rs_transition_scaled.mean()
            ]

# SEMs
            sems = [
                rs_reward_scaled.std(ddof=1) / np.sqrt(len(rs_reward_scaled)),
                rs_transition_scaled.std(ddof=1) / np.sqrt(len(rs_transition_scaled))
            ]

            # Estimated Rewards
            er_baseline_1 = np.array(data['estimated_reward_state_1_baseline']).mean()
            er_baseline_2 = np.array(data['estimated_reward_state_2_baseline']).mean()
            er_reward_reval_1 = np.array(data['estimated_reward_state_1_reward_reval']).mean()
            er_reward_reval_2 = np.array(data['estimated_reward_state_2_reward_reval']).mean()
            er_transition_reval_1 = np.array(data['estimated_reward_state_1_transition_reval']).mean()
            er_transition_reval_2 = np.array(data['estimated_reward_state_2_transition_reval']).mean()

            print("*** Reward Estimation ***")
            print("** Baseline **")
            print(f"    State 1: {er_baseline_1}    (Real: {defaults.REWARD_BASELINE_1})")
            print(f"    State 2: {er_baseline_2}    (Real: {defaults.REWARD_BASELINE_2})")
            print('** Reward Revaluation **')
            print(f"    State 1: {er_reward_reval_1}    (Real: {defaults.REWARD_BASELINE_2})")
            print(f"    State 2: {er_reward_reval_2}    (Real: {defaults.REWARD_BASELINE_1})")
            print('** Transition Revaluation **')
            print(f"    State 1: {er_transition_reval_1}    (Real: {defaults.REWARD_BASELINE_2})")
            print(f"    State 2: {er_transition_reval_2}    (Real: {defaults.REWARD_BASELINE_1})")

            print("*** Reward Revaluation (scaled) ***")
            print(f"  Mean = {means[0]:.4f}")
            print(f"  SEM  = {sems[0]:.4f}")

            print("\nTransition revaluation:")
            print(f"  Mean = {means[1]:.4f}")
            print(f"  SEM  = {sems[1]:.4f}")

            reward_reval_mean.append(means[0])
            reward_reval_sem.append(sems[0])

            transition_reval_mean.append(means[1])
            transition_reval_sem.append(sems[1])

            conditions.append(condition)

            # Plot
            plt.figure(figsize=(5, 4))
            x = np.arange(2)
            labels = ["Reward reval", "Transition reval"]


            plt.bar(x, means, yerr=sems, capsize=5,
                    color=["tab:blue", "tab:orange"])

            plt.xticks(x, labels)
            plt.ylabel("Scaled score")
            plt.ylim(0, 20)

            plt.title("Revaluation scores (mean ± SEM)")
            plt.tight_layout()
            plt.savefig(PATH)
            # plt.show()
    df = pd.DataFrame({
        'reward_reval_score': reward_reval_mean,
        'transition_reval_score': transition_reval_mean,
    'conditions': conditions,
    'reward_reval_sem': reward_reval_sem,
    'transition_reval_sem': transition_reval_sem})
    df.to_csv('reward_vs_transition_reval.csv', index=False)

