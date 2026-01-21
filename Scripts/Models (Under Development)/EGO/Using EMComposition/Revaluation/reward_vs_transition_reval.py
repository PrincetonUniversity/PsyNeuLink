"""
Replicating
"""

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

from ego_revaluation import run_original
from ego_revaluation.config import defaults

DOT_PRODUCT = 'dot_product'
COSINE_SIMILARITY = 'cosine_similarity'

metric = COSINE_SIMILARITY
model_based_ness = 0.

TIME_DRIFT_NOISE = .2 # "original": .05
STATE_INTEGRATION_RATE = .4 # 1.  # "original": .6
TIME_RETRIEVAL_WEIGHT = .3 #.2  # "original": .2

data = run_original.run(
    500,
    metric=metric,
    model_based_ness=model_based_ness,
    time_noise=TIME_DRIFT_NOISE,
    state_integration_rate=STATE_INTEGRATION_RATE,
    time_retrieval_weight=TIME_RETRIEVAL_WEIGHT,
)

condition = f'IR={STATE_INTEGRATION_RATE}_TIME_RW={TIME_RETRIEVAL_WEIGHT}'

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
plt.show()
