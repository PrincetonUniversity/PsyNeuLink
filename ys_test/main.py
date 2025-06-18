from declan.run import run_participant
from data.dataset import gen_data_loader

from psyneulink_ego.ego_model import construct_model, run_model

import declan.utils as utils
import numpy as np

import matplotlib.pyplot as plt
import torch

from params import params_declan

TRAINING_PARADIGM = 'tst'
PROBS = [1., 1., 1.]

RUN_PSY_EGO = True
RUN_DECLAN = True


def main():
    torch.set_default_dtype(torch.float64)
    utils.set_random_seed_and_capture_state(0)
    data_loader = gen_data_loader(TRAINING_PARADIGM, PROBS, 2)
    fig, axes = plt.subplots(2, 1, figsize=(5, 12))

    ego_inputs = data_loader.dataset.xs.numpy().copy()
    ego_targets = data_loader.dataset.ys.numpy().copy()
    state_before_ego = None
    state_before_declan = None

    state_after_ego = None
    state_after_declan = None

    if RUN_PSY_EGO:
        print('running psyneulink ego')
        state_before_ego = utils.set_random_seed_and_capture_state(0)
        model, context, state, em = construct_model(memory_capacity=len(ego_inputs))
        ego_results = run_model(model, context, state, em, ego_inputs)
        state_after_ego = utils.set_random_seed_and_capture_state(0)
        # print(ego_results)
        plot_results(ego_results, ego_targets, axes[0], 1)
        print('done running psyneulink ego')
        print('****')
        print()
    if RUN_DECLAN:
        print()
        state_before_declan = utils.set_random_seed_and_capture_state(0)
        declan_results = run_participant(params_declan, data_loader, len(ego_inputs))
        state_after_declan = utils.set_random_seed_and_capture_state(0)
        plot_results(declan_results, ego_targets, axes[1], 1)

        # print(declan_results)
        print('done running declan')
        print('****')
    # Comp
    # random seed comparision:
    utils.compare_rng_states(state_before_ego, state_before_declan, 'Ego vs Declan (Before)')

    utils.compare_rng_states(state_after_ego, state_after_declan, 'Ego vs Declan (After)')

    print('Results Comparison:')
    print('EGO Results:')
    print(ego_results)
    print('Declan Results:')
    print(declan_results)
    print('Difference between EGO and Declan results:')
    print(np.abs(ego_results - declan_results).sum())
    print(
        np.allclose(ego_results, declan_results, atol=1e-3) # Compare the two results
    )

    plt.show()


def plot_results(predictions, target, ax, stride=1):
    ax.plot((1 - np.abs(predictions[stride:len(target)] - target[:len(target) - stride])).sum(-1))
    ax.set_xlabel('Stimuli')
    ax.set_ylabel('loss')


if __name__ == '__main__':
    main()
