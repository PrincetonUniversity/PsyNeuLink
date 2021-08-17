import math
import numpy as np


def validate_mis(task_net, r_hidden, r_output, tasks_to_perform):
    n_pathways = math.sqrt(task_net.n_task)
    pathway_capacities, max_carrying_capacity, bk_mis, a_bipartite, a_tasks_idx, a_dual = \
        get_max_carrying_capacity(r_hidden, r_output, corr_threshold)
    a_tasks_idx[a_tasks_idx > 0] = tasks_to_perform[a_tasks_idx[a_tasks_idx > 0]]
    a_tasks_order = a_tasks_idx[:]
    a_tasks_order[a_tasks_order == 0] = []
    pathway_capacities = np.concatenate(a_tasks_order, pathway_capacities)
    good_multitask_conditions = np.zeros(np.size(bk_mis, 1), n_pathways*n_pathways)
    good_maximum_capacity = np.sum(bk_mis)
    for multicase in range(np.size(bk_mis, 2)):
        multitask_idx = pathway_capacities[bk_mis[:, multicase] == 1, 1]
        good_multitask_conditions[multicase, multitask_idx] = 1

    all_capacities = np.unique(good_maximum_capacity)
    all_capacities[all_capacities == 1] = []
    multi_performance_mean = np.zeros(3, np.size(all_capacities))
    multi_performance_sem = np.zeros(np.size(multi_performance_mean))
    multi_performance_mean[0, :] = all_capacities
    multi_performance_sem[0, :] = all_capacities
    # mse_data = []
    # for cap_idx in range(np.size(all_capacities)):
    #     cap = all_capacities[cap_idx]
    #     input_multi_cap = multicap[cap]