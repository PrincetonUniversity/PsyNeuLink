import numpy as np


def create_task_patterns(n_pathways, n_features, samples, sd_scale, 
    same_stimuli_across_tasks, relevant_tasks):
    curr_features_dims = np.arrange(n_pathways)
    # Pre-allocate matrices for input, task, and training signal
    samples_per_task = samples
    # Get the total number of training samples
    total_samples = np.size(relevant_tasks) * samples_per_task
    tasks_sglt = np.zeros(total_samples, n_pathways*n_pathways)
    input_sglt = np.zeros(total_samples, n_pathways*n_features)
    train_sglt = np.zeros(total_samples, n_pathways*n_features)

    # Generate the task layer inputs. These are just one-hot encoded vector
    # of the task index, repeated samples_per_task time for each.
    r_tasks = np.tile(np.transpose(relevant_tasks), (samples_per_task, 1))
    tasks_sglt[np.ravel_multi_index(np.size(tasks_sglt), 
        (np.arrange(np.size(tasks_sglt, 0)), np.transpose(r_tasks)))] = 1
    # Generate a matrix that specifies the task index for each sample
    tasks_idx_sglt = np.tile(np.transpose(relevant_tasks), (samples_per_task, 1))
    # Generate a matrix that specifies the sample\stim index within each task's group.
    stim_idx_sglt = np.tile(np.transpose(np.arrange(samples_per_task)), np.size(relevant_tasks))

    # Convert linear tasks indices to input\output row\column subscripts.
    relevant_outputs, relevant_inputs = np.unravel_index([n_pathways, n_pathways], relevant_tasks)

    # Fill in with multi-variate gaussian samples
    input_sgl_mask = input_sglt
    tasks_sgl_mask = tasks_sglt
    train_sgl_mask = train_sglt
    tasks_idx_sgl_mask = tasks_idx_sglt
    stim_idx_sgl_mask = stim_idx_sglt
    input_sgl = np.zeros(np.size(input_sgl_mask))
    tasks_sgl = tasks_sgl_mask
    train_sgl = train_sgl_mask
    tasks_idx_sgl = tasks_idx_sgl_mask
    stim_idx_sgl = stim_idx_sgl_mask

    sd = np.random.rand(n_features, n_features) * sd_scale
    input_sgl_temp = np.zeros(np.size(input_sgl_mask))

    for curr_t_idx in range(len(relevant_tasks)):
        relevant_output = relevant_outputs[curr_t_idx]
        relevant_input = relevant_inputs[curr_t_idx]
        if curr_t_idx == 1 or not same_stimuli_across_tasks:
            for input_idx in range(samples_per_task):
                row_idx = (curr_t_idx - 1) * samples_per_task + input_idx
                # Generate stimulus
                stim_combs = np.random.choice(n_features, n_pathways, True)
                input_sgl_mask[row_idx, (curr_features_dims-1)*n_features+stim_combs] = 1
                # Compute correct training pattern
                train_sgl[row_idx, ((relevant_output-1)*n_features+1):((relevant_output-1)*n_features+n_features)] \
                    = input_sgl_mask[row_idx, ((relevant_input-1)*n_features+1):((relevant_input-1)*n_features+n_features)]
                for dimension_idx in range(n_pathways):
                    col_idx = np.arrange(n_features*(dimension_idx-1)+1, n_features*dimension_idx)
                    mu = input_sgl_mask[row_idx, col_idx]
                    x = np.random.multivariate_normal(mu, sd[mu==1], 1)
                    input_sgl_temp[row_idx, col_idx] = x
    input_sgl = input_sgl_temp
    return input_sgl, tasks_sgl, train_sgl, tasks_idx_sgl, stim_idx_sgl