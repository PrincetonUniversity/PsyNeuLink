# Implements a torch version where the learnable projection is in the EM module rather than the context module.
# This means that in the additional optimization steps, we are actually calling the learnable projection and
# using the updated weights between the num optimization steps

from pytorch_ego.run import run_participant
from data.dataset import gen_data_loader_with_first

from psyneulink_ego.ego_model import construct_model, run_model

import pytorch_ego.utils as utils
import numpy as np
import torch

import matplotlib.pyplot as plt

from params import params_torch, params_ego

TRAINING_PARADIGM = 'blocked'
PROBS = [1., 1., 1.]

RUN_PSY_EGO = True
RUN_TORCH = True

IS_PLOT = True
IS_TEST = True
IS_VERBOSE = True

TOLERANCE = 1e-10
NR_TRIALS_TO_CHECK = 20


def main():
    utils.set_random_seed(0)
    data_loader = gen_data_loader_with_first(TRAINING_PARADIGM, PROBS, 1)
    fig, axes = plt.subplots(2, 1, figsize=(5, 12))

    pnl_inputs = data_loader.dataset.xs.numpy().copy()[1:]  # Exclude first input for PNL
    pnl_targets = data_loader.dataset.ys.numpy().copy()[1:]  # Exclude first target for PNL
    if RUN_PSY_EGO:
        if IS_VERBOSE:
            print('running psyneulink ego')
        utils.set_random_seed(0)
        model, context, state, em = construct_model(config=params_ego,
                                                    memory_capacity=len(pnl_inputs))
        # from psyneulink import ALL
        # em.show_graph(show_node_structure=ALL)
        pnl_results = run_model(model, pnl_inputs, config=params_ego)
        if IS_PLOT:
            print('plotting psyneulink ego')
            plot_results(pnl_results, pnl_targets, axes[0], 1)
        if IS_VERBOSE:
            print('done running psyneulink ego', end='\n\n')
    if RUN_TORCH:
        torch.set_default_dtype(torch.float64)
        if IS_VERBOSE:
            print('running pytorch ego')
        utils.set_random_seed(0)
        torch_results = run_participant(params_torch, data_loader, len(pnl_inputs))
        if IS_PLOT:
            print('plotting pytorch ego')
            plot_results(torch_results, pnl_targets, axes[1], 1)
        if IS_VERBOSE:
            print('done running declan', end='\n\n')
    if IS_PLOT:
        plt.show()

    if IS_TEST:
        if not RUN_PSY_EGO or not RUN_TORCH:
            raise ValueError('Both models must be run for testing.')

        # First, check they are NumPy arrays (or convertible)
        pnl_results = np.asarray(pnl_results)
        torch_results = np.asarray(torch_results)

        # Check shape matches
        if pnl_results.shape != torch_results.shape:
            raise ValueError(
                f"Shape mismatch: results {pnl_results.shape} vs targets {torch_results.shape}"
            )

        assert NR_TRIALS_TO_CHECK < pnl_results.shape[0], "Not enough trials to check"

        # Now check values
        if not np.allclose(
                pnl_results[:NR_TRIALS_TO_CHECK],
                torch_results[:NR_TRIALS_TO_CHECK],
                atol=TOLERANCE, rtol=TOLERANCE):
            raise AssertionError("PNL and torch results differ beyond allowed tolerance")
        print(f"PNL and torch produce identical results w/in tolerance of {TOLERANCE} for {NR_TRIALS_TO_CHECK} trials")


def plot_results(predictions, target, ax, stride=1):
    print(predictions)
    # print(len(predictions))
    # print(target)
    ax.plot((1 - np.abs(predictions - target)).sum(-1))
    ax.set_xlabel('Stimuli')
    ax.set_ylabel('loss')


if __name__ == '__main__':
    main()