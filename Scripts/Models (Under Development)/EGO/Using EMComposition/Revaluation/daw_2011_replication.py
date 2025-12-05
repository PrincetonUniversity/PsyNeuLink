"replicating Daw, 2011"

import numpy as np
import matplotlib.pyplot as plt
import os

from tqdm import tqdm

# your model
from ego_revaluation.run_original_probabilistic import run


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ============================================================
# Daw 2011 Stay Probability Computation
# ============================================================

def compute_stay_stats(trial_log):
    """Return stay probabilities for the four Daw 2011 conditions."""
    rc, rr, uc, ur = [], [], [], []

    for tr in trial_log:
        stay = tr["stay"]
        if stay is None:
            continue

        reward = tr["reward"]
        transition = tr["transition"]

        if reward == 1 and transition == "common":
            rc.append(stay)
        elif reward == 1 and transition == "rare":
            rr.append(stay)
        elif reward == 0 and transition == "common":
            uc.append(stay)
        elif reward == 0 and transition == "rare":
            ur.append(stay)

    return np.array([
        np.mean(rc) if rc else np.nan,
        np.mean(rr) if rr else np.nan,
        np.mean(uc) if uc else np.nan,
        np.mean(ur) if ur else np.nan,
    ])


# ============================================================
# Averaging across participants
# ============================================================

def simulate_participants(
    n_participants,
    n_base_trials,
    state_integration_rate,
    time_retrieval_weight,
    model_based_ness,
    metric="cosine_similarity",
    common_prob=0.7
):
    """Run M simulated participants and return Nx4 matrix of stay stats."""
    all_stats = []

    for _ in range(n_participants):
        trial_log = run(
            state_integration_rate=state_integration_rate,
            time_retrieval_weight=time_retrieval_weight,
            model_based_ness=model_based_ness,
            metric=metric,
            n_base_trials=n_base_trials,
            common_prob=common_prob,
        )
        all_stats.append(compute_stay_stats(trial_log))

    return np.array(all_stats)   # shape = (participants, 4)


# ============================================================
# Plotting
# ============================================================

def plot_daw_style(mean_stats, sem_stats, title):
    """
    mean_stats: [RC, RR, UC, UR]
    sem_stats:  same shape
    """
    labels = ["Rewarded", "Unrewarded"]

    common = [mean_stats[0], mean_stats[2]]
    rare   = [mean_stats[1], mean_stats[3]]

    common_sem = [sem_stats[0], sem_stats[2]]
    rare_sem   = [sem_stats[1], sem_stats[3]]

    x = np.arange(2)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - width/2, common, width, yerr=common_sem, label="Common", color="royalblue", alpha=0.8)
    ax.bar(x + width/2, rare, width, yerr=rare_sem, label="Rare", color="darkorange", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Stay probability")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    return fig


# ============================================================
# Parameter sweep figure (integration × time × MB)
# ============================================================
def parameter_sweep(
    save_dir,
    integration_rates=[0.6, 1.0],
    time_weights=[0.0, 0.2],
    model_based_ness_list=[0.0, 1.0],
    n_base_trials=200,
    n_participants=50
):
    ensure_dir(save_dir)

    # We want: rows = integration rates
    #          columns = time weights × mb levels
    n_rows = len(integration_rates)
    n_cols = len(time_weights) * len(model_based_ness_list)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharey=True
    )

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(n_rows, 1)

    total_jobs = len(integration_rates) * len(time_weights) * len(model_based_ness_list)

    with tqdm(total=total_jobs, desc="Parameter sweep", ncols=100) as pbar:
        for i, ir in enumerate(integration_rates):

            col_index = 0  # track which subplot column we're filling

            for tw in time_weights:
                for mb in model_based_ness_list:

                    # --- Simulate participants ---
                    stats = simulate_participants(
                        n_participants=n_participants,
                        n_base_trials=n_base_trials,
                        state_integration_rate=ir,
                        time_retrieval_weight=tw,
                        model_based_ness=mb,
                    )

                    mean_stats = np.nanmean(stats, axis=0)
                    sem_stats = np.nanstd(stats, axis=0) / np.sqrt(stats.shape[0])

                    print(f"\n=== Condition: int={ir}, time={tw}, mb={mb} ===")
                    print(f"RC={mean_stats[0]:.3f} ± {sem_stats[0]:.3f}")
                    print(f"RR={mean_stats[1]:.3f} ± {sem_stats[1]:.3f}")
                    print(f"UC={mean_stats[2]:.3f} ± {sem_stats[2]:.3f}")
                    print(f"UR={mean_stats[3]:.3f} ± {sem_stats[3]:.3f}")
                    print("----------------------------------------------------")

                    # -----------------------------------------------------
                    # 🔥 MACHINE-READABLE BLOCK FOR PASTING BACK TO CHATGPT
                    # -----------------------------------------------------
                    print(
                        f"### RESULT int={ir}, time={tw}, mb={mb}\n"
                        f"RC={mean_stats[0]:.6f}, RR={mean_stats[1]:.6f}, "
                        f"UC={mean_stats[2]:.6f}, UR={mean_stats[3]:.6f}\n"
                        f"SEM_RC={sem_stats[0]:.6f}, SEM_RR={sem_stats[1]:.6f}, "
                        f"SEM_UC={sem_stats[2]:.6f}, SEM_UR={sem_stats[3]:.6f}\n"
                    )

                    ax = axes[i, col_index]

                    # Prepare Daw-style bars
                    common = [mean_stats[0], mean_stats[2]]
                    rare   = [mean_stats[1], mean_stats[3]]

                    common_sem = [sem_stats[0], sem_stats[2]]
                    rare_sem   = [sem_stats[1], sem_stats[3]]

                    x = np.arange(2)
                    width = 0.35

                    ax.bar(x - width/2, common, width, yerr=common_sem,
                           label="Common", color="royalblue", alpha=0.85)
                    ax.bar(x + width/2, rare, width, yerr=rare_sem,
                           label="Rare", color="darkorange", alpha=0.85)

                    ax.set_xticks(x)
                    ax.set_xticklabels(["Rewarded", "Unrewarded"], fontsize=8)
                    ax.set_ylim(0, 1)
                    ax.set_title(f"int={ir}, time={tw}, mb={mb}")

                    col_index += 1
                    pbar.update(1)

    fig.suptitle("Daw 2011 Replication — Parameter Sweep", fontsize=18)
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "parameter_sweep.pdf")
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)

    print(f"Saved: {fig_path}")




# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    save_dir = "./figures_daw2011"
    ensure_dir(save_dir)

    parameter_sweep(
        save_dir=save_dir,
        integration_rates=[.6],
        time_weights=[0, 0.1, 1.6],
        model_based_ness_list=[0.0],
        n_base_trials=200,
        n_participants=50
    )
