import numpy as np


def generate_environment_to_graph(a, n_features, samples_per_task, sd_scale, 
    same_stimuli_across_tasks):
    n_pathways = np.max(np.size(a))
    # Extract relevant tasks
    task_m = np.transpose(np.reshape(np.arrange(n_pathways^2), (n_pathways, n_pathways)))
    mask = np.vstack(
        np.concatenate(a, np.zeros((np.size(a, 0), n_pathways-np.size(a, 1)))),
        np.zeros((n_pathways-np.size(a, 0), n_pathways))
    )
    relevant_tasks = task_m[mask == 1]
    create_task_patterns(n_pathways, n_features, samples_per_task, sd_scale, 
        same_stimuli_across_tasks, relevant_tasks)