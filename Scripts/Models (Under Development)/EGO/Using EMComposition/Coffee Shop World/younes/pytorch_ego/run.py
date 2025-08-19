from .models import gen_model
import torch
import numpy as np
from torch import nn

def run_participant(params, data_loader, len_memory=2):
    loss_fn = nn.BCELoss()
    context_module, em_module = gen_model(params, len_memory=len_memory)
    optimizer = torch.optim.SGD(lr=params['learning_rate'], params=context_module.parameters())
    em_preds = []

    # BREADCRUMB PRINT
    print(f"TORCH LOSSES: ")
    # Loop over each state of the CSW task.
    for trial, (x, _, y) in enumerate(data_loader):
        if trial == 0:
            with torch.no_grad():
                em_module.write(
                    torch.zeros_like(x),
                    torch.zeros_like(x),
                    x,
                )
            em_preds.append(torch.zeros_like(x))
        # For each state, optimize the context representation for predicting the next state over n_optimization_steps.
        if trial >= 0 and trial < 23:
            context = context_module(x)  # retrieve the context representation from the integrator.
            for i in range(params['n_optimization_steps']):
                # Skip first state bc which sequence within the context is randomly assigned.
                # i.e., we have not yet observed a full state transition.

                optimizer.zero_grad()  # Zero the gradients before each optimization step.
                pred_em = em_module(x, context)  # retrieve the next state prediction from the EM module.
                loss = loss_fn(pred_em, y)  # calculate the loss between the predicted and actual next state.
                loss.backward(retain_graph=True)  # compute the gradients of the context module.
                optimizer.step()  # backprop to update context module weights.
                print(f"STIM {trial} optimization step {i}: {loss:{5}f}")
            em_module.write(x, context, y)
            em_preds.append(pred_em.detach().cpu().numpy())


    # Collect some metrics from the training run for analysis.
    em_preds = np.stack(em_preds).squeeze()
    return em_preds
