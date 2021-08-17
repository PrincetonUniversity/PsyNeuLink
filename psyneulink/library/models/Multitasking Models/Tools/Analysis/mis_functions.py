import numpy as np
from get_dependency_graph import *


def find_full_set(a, row, thresh, curr_set):
    set = []
    for i in range(np.size(a, 1)):
        if a[row, i] > thresh and i not in curr_set:
            set.append(i)
            set += find_full_set(a, i, thresh, curr_set + set)
            curr_set = curr_set + set
    return set


def find_mis(ad, priority):
    n = np.size(ad, 0)
    m = np.size(ad, 1)
    x = -np.ones((n, 1))
    n_id = range(n)
    # Finds the vertices with the minimum degree
    degree = np.sum(ad, 0)
    md = np.amin(degree)
    min_deg = degree[degree == md]

    if np.size(min_deg) > 1:
        support = np.zeros(np.size(min_deg))
        for i in np.nonzero(min_deg):
            support[i] = np.sum(degree[ad[i:]])
        ms = np.amax(support)
        if ms > 0:
            mindeg_maxsup = np.nonzero(support[support == ms])
        else:
            mindeg_maxsup = np.nonzero(min_deg)
    else:
        mindeg_maxsup = np.nonzero(min_deg)
    if np.size(mindeg_maxsup) > 1:
        j = np.amin(priority[mindeg_maxsup])
        nod_sel = mindeg_maxsup[j]
    else:
        nod_sel = mindeg_maxsup
    x[nod_sel] = 1
    x[ad[nod_sel, :]] = 0
    assigned = x[x > -1]
    ad[assigned, :] = []
    ad[:, assigned] = []
    n_id[assigned] = []
    priority[assigned] = []
    if np.size(ad) > 0:
        x[n_id] = find_mis(ad, priority)
    return x


def bk_max_is(int_matrix):
    no_vertices = np.size(int_matrix, 1)
    m = []
    s = []
    t = range(no_vertices)
    return None


def get_max_carrying_capacity(r_hidden, r_output, corr_threshold):
    # Hidden layer
    hidden_sets = []
    curr_set = []
    for i in range(np.size(r_hidden, 0)):
        hidden_sets.append(find_full_set(r_hidden, i, corr_threshold, curr_set))
        curr_set += hidden_sets[i]
    hidden_sets = hidden_sets
    # Output layer
    output_sets = []
    curr_set = []
    for i in range(np.size(r_output, 0)):
        output_sets.append(find_full_set(r_output, i, corr_threshold, curr_set))
        curr_set += output_sets[i]
    output_sets = output_sets
    # Remove empty fields
    hidden_sets = [set for set in hidden_sets if set != []]
    output_sets = [set for set in output_sets if set != []]

    hidden_components = len(hidden_sets)
    output_components = len(output_sets)
    a_bipartite = np.zeros((hidden_components, output_components))
    a_tasks_idx = np.zeros((hidden_components, output_components))
    for row in range(hidden_components):
        for hidden_task_rep in range(len(hidden_sets[row])):
            task_idx = hidden_sets[row][hidden_task_rep]
            for col in range(output_components):
                if task_idx in output_sets[col]:
                    a_tasks_idx[row, col] = task_idx
                    break

    # Generate interference adjacency matrix
    a_dual = get_dependency_graph(a)
    multitask = find_mis(a_dual, range(m))
    bk_mis = bk_max_is(a_dual)
    pathway_capacities = np.concatenate(x, y, multitask)
    max_carrying_capacity = np.sum(multitask)
    np.savetxt("a_dual.csv", a_dual, delimiter=",")
    return (pathway_capacities, max_carrying_capacity, bk_mis, a_bipartite, a_tasks_idx, a_dual)