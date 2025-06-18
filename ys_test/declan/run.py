from declan.models import gen_model
import torch
import numpy as np
from torch import nn


def run_participant(params, data_loader, len_memory=2):
    loss_fn = nn.BCELoss()
    context_module, em_module = gen_model(params, len_memory=len_memory)
    optimizer = torch.optim.SGD(lr=params['learning_rate'], params=context_module.parameters())
    em_preds = []
    context = None
    x_last = None
    # Loop over each state of the CSW task.
    for trial, (x, _, y) in enumerate(data_loader):

        # For each state, optimize the context representation for predicting the next state over n_optimization_steps.
        for _ in range(params['n_optimization_steps']):
              # retrieve the context representation from the integrator.
             # Initialize context if not already done.
            # Skip first state bc which sequence within the context is randomly assigned.
            # i.e., we have not yet observed a full state transition.
            if trial > 0:
                context = context_module(x_last)
                 # retrieve the context representation from the integrator.
                optimizer.zero_grad()  # Zero the gradients before each optimization step.
                pred_em = em_module(x_last, context)  # retrieve the next state prediction from the EM module.
                loss = loss_fn(pred_em, x)  # calculate the loss between the predicted and actual next state.
                loss.backward()  # compute the gradients of the context module.
                optimizer.step()  # backprop to update context module weights.
            else:
                x_last = torch.zeros([1, params["state_d"]]).float()
                context = torch.zeros([1, params["context_d"]]).float()
                pred_em = torch.zeros([1, params["output_d"]]).float()

        with torch.no_grad():
            em_module.write(x_last, context, x)
            em_preds.append(pred_em.cpu().detach().numpy())
        x_last = x.cpu()

    # Collect some metrics from the training run for analysis.
    em_preds = np.stack(em_preds).squeeze()
    return em_preds
