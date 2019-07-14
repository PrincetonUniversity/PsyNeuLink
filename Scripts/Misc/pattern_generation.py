import numpy as np
import itertools


def generate_inputs(num_dimensions, num_features):
    base = np.zeros((1, num_features))
    base[0, 0] = 1.0

    base_perm = base.copy()  # make a copy as to not modify the base vector

    # roll the vector to get all possible nodes active in a single input dimension
    for i in range(1, num_features):
        base_perm = np.vstack((base_perm, np.roll(base, i, axis=1)))

    # get all possible combinations of input dimensions
    return np.array([np.array(p) for p in itertools.product(base_perm, repeat=num_dimensions)])


def generate_tasks(num_dimensions, out_dimensions=None):
    if out_dimensions is None:
        out_dimensions = num_dimensions

    units = num_dimensions * out_dimensions

    # generate base vector
    base = np.zeros((1, units))
    base[0, 0] = 1.0

    tasks = base.copy()

    # roll vector to get all possible nodes active
    for i in range(1, units):
        tasks = np.vstack((tasks, np.roll(base, i, axis=1)))

    # Generate a map of dimensions mapped by each task
    task_numbers = np.arange(units)
    task_map = np.reshape(task_numbers, (num_dimensions, out_dimensions))

    return tasks, task_map


def generate_training_patterns(num_dimensions, num_features, inputs=None, tasks=None, task_map=None):
    if inputs is None:
        inputs = generate_inputs(num_dimensions, num_features)

    if tasks is None or task_map is None:
        tasks, task_map = generate_tasks(num_dimensions)

    num_cases = len(inputs) * len(tasks)

    # changed to a tensor to support the sort of indexing I need
    target_patterns = np.zeros((num_cases, num_dimensions, num_features))
    input_patterns = np.zeros((num_cases, num_dimensions, num_features))
    task_patterns = np.zeros((num_cases, np.size(tasks, 1)))
    in_out_map = np.zeros((num_cases, task_map.ndim))  # logs dim map per trial

    # generate input-task combination and target pattern per trial
    trial = 0

    for task_p in tasks:  # loops along task patterns
        for input_p in inputs:  # loops along input patterns
            input_patterns[trial] = input_p
            task_patterns[trial] = task_p

            task_index = np.argmax(task_p) # np.where(task_p == np.max(task_p))[0][0]

            mapping = np.array(np.where(task_map == task_index))[:,0]
            in_out_map[trial] = mapping
            in_dim = mapping[0]
            out_dim = mapping[1]

            # Perform the dimension mapping from input to output
            target_patterns[trial, out_dim, :] = input_patterns[trial, in_dim, :]

            trial += 1

    return input_patterns, task_patterns, in_out_map, target_patterns


if __name__ == '__main__':
    # ins = generate_inputs(3, 4)
    # print(ins[0])
    # print(ins[0][0])
    # print(ins[0].shape)

    # input_patterns, task_patterns, in_out_map, target_patterns = generate_training_patterns(3, 4)
    # x = 333
    # print(input_patterns[x])
    # print(task_patterns[x])
    # print(in_out_map[x])
    # print(target_patterns[x])
    # print(np.argmax(task_patterns, 1))
    pass



