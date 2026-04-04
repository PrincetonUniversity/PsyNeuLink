# Implements a torch version where the learnable projection is in the EM module rather than the context module.
# This means that in the additional optimization steps, we are actually calling the learnable projection and
# using the updated weights between the num optimization steps

import random

from environment import get_state_sequence

from pytorch_ego.run import run_participant
from psyneulink_ego.ego_model import construct_model, run_model

import numpy as np
import torch




def set_random_seed(seed, **kwargs):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.mps.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


CONFIG = dict(
    paradigm='blocked',
    tolerance=1e-10,
    nr_trials=10,
    seed=42,
    state_size=11,
    integration_rate=.7,
    state_retrieval_weight=None,
    previous_state_retrieval_weight=1.,
    context_retrieval_weight=1.,
    learning_rate=2.,
    memory_fill=.01,
    softmax_temperature=.2,
    softmax_threshold=.001,
    loss_spec_name='BinaryCrossEntropy',
    num_optimization_steps=10,
)


def main():
    torch.set_default_dtype(torch.float64)
    set_random_seed(**CONFIG)
    states = get_state_sequence(**CONFIG)

    # For testing, we reduce the number of trials to speed up the process.
    states = states[:CONFIG['nr_trials']]

    set_random_seed(**CONFIG)
    pnl_model, input_layer = construct_model(**CONFIG, memory_capacity=len(states))
    pnl_results = run_model(pnl_model,input_layer, states, **CONFIG)

    torch_results = run_participant(states, len(states), **CONFIG)
    pnl_results = np.asarray(pnl_results)
    torch_results = np.asarray(torch_results)


    assert np.allclose(pnl_results,torch_results, atol=CONFIG['tolerance']), \
        f"PNL and torch results differ beyond allowed tolerance of {CONFIG['tolerance']}"


if __name__ == '__main__':
    main()
