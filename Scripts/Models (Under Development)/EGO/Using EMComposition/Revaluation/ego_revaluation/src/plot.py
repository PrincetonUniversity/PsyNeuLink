import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_trials(states, rewards, times=None,
                title=None,
                path=None,
                figsize=(4, 6),
                show_time_vector=False,
                show_trial_labels=False,
                reward_cmap="Reds",
                cell_aspect=0.25):  # 0.25 = much thinner rows
    """
    Final stable version:
    - pcolormesh = perfect alignment
    - labels above heatmap
    - thinner cells using cell_aspect
    - reward colored differently
    """

    num_trials, num_states = states.shape

    # ------------------------------------------------------
    # Build matrix
    # ------------------------------------------------------
    blocks = [states, rewards.reshape(-1, 1)]
    if show_time_vector and times is not None:
        blocks.append(times)

    X = np.hstack(blocks)
    n_rows, n_cols = X.shape

    labels = [f"S{i}" for i in range(num_states)] + ["Reward"]
    if show_time_vector and times is not None:
        labels += [f"T{i}" for i in range(times.shape[1])]

    # ------------------------------------------------------
    # Normalize (same logic you used)
    # ------------------------------------------------------
    Xn = np.zeros_like(X, float)

    r = rewards.astype(float)
    if r.max() == r.min():
        r_norm = np.zeros_like(r)
    else:
        r_norm = (r - r.min()) / (r.max() - r.min())
        r_norm = np.sqrt(r_norm)

    for col in range(n_cols):
        if col == num_states:
            Xn[:, col] = r_norm
        else:
            c = X[:, col]
            if c.max() == c.min():
                Xn[:, col] = c
            else:
                Xn[:, col] = (c - c.min()) / (c.max() - c.min())

    # ------------------------------------------------------
    # Geometry — thin cells!
    # ------------------------------------------------------
    # normal x-edges
    xedges = np.arange(n_cols + 1)

    # compressed y-edges → thinner cells
    yedges = np.arange(n_rows + 1) * cell_aspect

    # EXTRA space above heatmap for labels
    extra_label_space = cell_aspect * 1.2

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------
    # Draw heatmap column by column for custom cmap
    # ------------------------------------------------------
    for i in range(n_cols):
        cmap = reward_cmap if i == num_states else "gray_r"
        ax.pcolormesh(
            xedges[i:i+2],
            yedges,
            Xn[:, i:i+1],
            shading="flat",
            cmap=cmap
        )

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, yedges[-1])  # ← extra space for labels
    ax.invert_yaxis()

    # ------------------------------------------------------
    # Horizontal dashed lines (reward boundaries)
    # ------------------------------------------------------
    for j in range(num_trials - 1):
        if rewards[j] != 0:
            y = (j + 1) * cell_aspect
            ax.hlines(y, 0, n_cols, color="#777", linestyle="--", linewidth=0.6)

    # Cell borders
    ax.vlines(np.arange(n_cols + 1), -extra_label_space, yedges[-1], color="black", linewidth=0.3)

    # ------------------------------------------------------
    # Column labels ABOVE heatmap
    # ------------------------------------------------------
    for i, label in enumerate(labels):
        ax.text(
            i + 0.5,
            -.5 * cell_aspect,       # just above the heatmap
            label,
            ha="center",
            va="bottom",
            fontsize=12,

        )

    # ------------------------------------------------------
    # Ticks
    # ------------------------------------------------------
    ax.set_xticks([])
    if show_trial_labels:
        ax.set_yticks([(j + 0.5) * cell_aspect for j in range(n_rows)])
        ax.set_yticklabels(np.arange(n_rows))
    else:
        ax.set_yticks([])

    if title:
        ax.set_title(title, pad=25, fontweight="bold", fontsize=16)

    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=400)
    else:
        plt.show()
