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
PLOT = False


def main():
    torch.set_default_dtype(torch.float64)
    utils.set_random_seed_and_capture_state(0)
    data_loader = gen_data_loader(TRAINING_PARADIGM, PROBS, 2)
    fig, axes = plt.subplots(2, 1, figsize=(5, 12))

    ego_inputs = data_loader.dataset.xs.numpy().copy()
    ego_targets = data_loader.dataset.ys.numpy().copy()


    if RUN_PSY_EGO:
        utils.set_random_seed_and_capture_state(0)
        model, context, state, em = construct_model(memory_capacity=len(ego_inputs))
        ego_results = run_model(model, context, state, em, ego_inputs)
        plot_results(ego_results, ego_targets, axes[0])
    if RUN_DECLAN:
        utils.set_random_seed_and_capture_state(0)
        declan_results = run_participant(params_declan, data_loader, len(ego_inputs))
        plot_results(declan_results, ego_targets, axes[1])


    if RUN_PSY_EGO:
        print('Results Comparison:')
        print('EGO Results:')
        print(ego_results)

    if RUN_DECLAN:
        print('Declan Results:')
        print(declan_results)

    if RUN_PSY_EGO and RUN_DECLAN:
        print('EGO and Declan results comparison:')
        print('Difference between EGO and Declan results:')
        print(np.abs(ego_results - declan_results).sum())
        print(
            np.allclose(ego_results, declan_results, atol=1e-8) # Compare the two results
        )

    plt.show()


def plot_results(predictions, target, ax):
    if not PLOT:
        return
    ax.plot((np.abs(predictions[1:] - target[:-1])).sum(-1))
    ax.set_xlabel('Stimuli')
    ax.set_ylabel('loss')


if __name__ == '__main__':
    main()
