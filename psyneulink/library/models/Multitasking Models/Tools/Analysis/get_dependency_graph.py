import numpy as np


def get_dependency_graph(a):
    m = np.count_nonzero(a)
    x = np.ones(np.size(a, 0))
    y = np.nonzero(a)
    a_dual = np.zeros(m)
    for i in range(m):
        for j in range(i, m):
            first_in = x[i]
            first_out = y[i]
            second_in = x[j]
            second_out = y[j]
            if first_in == second_in:
                a_dual[i, j] = 1
            elif first_out == second_out:
                a_dual[i, j] = 1
            elif not (np.size(y[x[x == first_in] == second_out]) == 0 
                and np.size(y[x[x == second_in] == first_out]) == 0):
                a_dual[i, j] = 1
    return a_dual + np.transpose(a_dual)