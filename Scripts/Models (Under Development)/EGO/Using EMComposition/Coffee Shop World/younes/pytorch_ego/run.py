from .models import gen_model
import torch
import numpy as np
from torch import nn


def run_participant(params, data_loader, len_memory=2):
    loss_fn = nn.BCELoss()
    context_module, em_module = gen_model(params, len_memory=len_memory)
    optimizer = torch.optim.SGD(lr=params['learning_rate'], params=em_module.parameters())
    em_preds = []
    context = torch.zeros(11)
    prev_state = torch.zeros_like(context)

    # Loop over each state of the CSW task.
    for trial, (x, _, y) in enumerate(data_loader):
        if trial < 1:
            continue



        for i in range(params['n_optimization_steps']):
            # Skip first state bc which sequence within the context is randomly assigned.
            # i.e., we have not yet observed a full state transition.

            pred_em = em_module(prev_state, context)
            if i == 0:
                pred_init = pred_em.detach().cpu().numpy().copy()


            optimizer.zero_grad()  # Zero the gradients before each optimization step.
              # retrieve the next state prediction from the EM module.
            # The initial prediction is our first guess before learning

            loss = loss_fn(pred_em, x)  # calculate the loss between the predicted and actual next state.
            loss.backward()  # compute the gradients of the context module.
            optimizer.step()  # backprop to update context module weights.
            print(f"STIM {trial} optimization step {i}: {float(loss):{5}f}")


        # with torch.no_grad():
        #     context_to_store = em_module.context_in(context)

        # with torch.no_grad():# After optimization, write the current state to the EM module and update the context.
        context_to_store = em_module.context_query.detach().cpu()
        em_module.write(prev_state, context_to_store, x)
        context = context_module(x)
        prev_state = x.detach().cpu()

        em_preds.append(pred_init)

    # Collect some metrics from the training run for analysis.
    em_preds = np.stack(em_preds).squeeze()
    return em_preds
