from .models import gen_model
import torch
import numpy as np
from torch import nn


def run_participant(states, len_memory=2, **kwargs):
    # Context module = Integrator that integrates states (working memory)
    # Context mapping = maps the current context to the representation used in the EM module (learned)
    # EM module = Episodic memory module that stores prev_state, context representation, and next state
    context_module, context_mapping, em_module = gen_model(**kwargs, len_memory=len_memory)

    # Learning setup
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(lr=kwargs['learning_rate'], params=context_mapping.parameters())

    # Container to store predictions
    predictions = []

    # Initialization
    context = torch.zeros(11)
    learned_context_representation = torch.zeros_like(context)
    prev_state = torch.zeros_like(context)
    pred_init = torch.zeros_like(context)

    # Loop over each state of the CSW task.
    for trial_idx, state in enumerate(states):
        # Ensure that state is correct shape (1, state_dim). If it is (state_dim,), unsqueeze
        if len(state.shape) == 1:
            state = state.unsqueeze(0)

        # For each state, we perform several optimization steps to update the context representations
        for i in range(kwargs['num_optimization_steps']):
            # get current context representation from current context
            learned_context_representation = context_mapping(context)
            # retrieve the next state prediction from the EM module using the previous state and current context.
            pred_em = em_module(prev_state, learned_context_representation)

            # Capture the initial prediction before any learning for analysis
            if i == 0:
                pred_init = pred_em.detach().cpu().numpy().copy()

                # Backpropagation step to update the context mapping
            optimizer.zero_grad()

            loss = loss_fn(pred_em, state)  # calculate the loss between the predicted and actual next state.
            loss.backward()  # compute the gradients of the context module.
            optimizer.step()  # backprop to update context module weights.

        # After the trial, we write the new experience into EM and update the context for the next trial.
        context_to_store = learned_context_representation.detach().cpu()
        em_module.write(prev_state, context_to_store, state)
        context = context_module(state)
        prev_state = state.detach().cpu()

        # Store the initial EM prediction for this trial for analysis.
        predictions.append(pred_init)

    return np.stack(predictions).squeeze()
