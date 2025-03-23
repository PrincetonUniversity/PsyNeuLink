from declan.models import EMModule, RecurrentContextModule, prep_EM
import torch
import numpy as np
from torch import nn
import declan.utils as utils
from declan.dataset import CSWDataset, CompositionalWords
from random import randint

FILL = 0

def gen_data_loader(paradigm, probs=[1., 1., 1.], n=1):
    ds = None
    if paradigm == 'tst':
        contexts_to_load =[0]
        n_samples_per_context = [n]
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=[1.])

    if paradigm == 'blocked':
        contexts_to_load = [0, 1, 0, 1] + [randint(0, 2) for _ in range(n)]
        n_samples_per_context = [n, n, n, n] + [1] * n
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    elif paradigm == 'interleaved':
        contexts_to_load = [0, 1] * (2 * n) + [randint(0, 2) for _ in range(n)]
        n_samples_per_context = [1] * (4 * n) + [1] * n
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)


def prep_recurrent_networkLeg(rnet, state_d, persistance=-0.6):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()
        rnet.hidden_to_hidden.weight.zero_()
        rnet.hidden_to_hidden.bias.zero_()
        rnet.state_to_hidden_wt.weight.zero_()
        rnet.state_to_hidden_wt.bias.copy_(
            torch.ones((len(rnet.state_to_hidden_wt.bias),), dtype=torch.float) * persistance)
        rnet.hidden_to_hidden_wt.weight.zero_()
        rnet.hidden_to_hidden_wt.bias.zero_()
        # Set hidden to context weights as an identity matrix.
        rnet.hidden_to_context.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_context.bias.zero_()

    # Set requires_grad to True for hidden_to_context.weight before freezing other parameters
    rnet.hidden_to_context.weight.requires_grad = True
    rnet.hidden_to_context.bias.requires_grad = True

    # Freeze recurrent weights to stabilize training
    for name, p in rnet.named_parameters():
        if 'hidden_to_context' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return rnet


def prep_recurrent_network(rnet, state_d):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()

        rnet.hidden_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_hidden.bias.zero_()
        rnet.hidden_to_context.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.hidden_to_context.bias.zero_()

    # Set requires_grad to True for hidden_to_context.weight before freezing other parameters
    rnet.hidden_to_context.weight.requires_grad = True
    rnet.hidden_to_context.bias.requires_grad = True

    # Freeze recurrent weights to stabilize training
    for name, p in rnet.named_parameters():
        if 'hidden_to_context' not in name:
            p.requires_grad = False
        else:
            p.requires_grad = True
    return rnet


def gen_model(params, len_memory=2):
    context_module = RecurrentContextModule(params.state_d, params.state_d, params.context_d)
    em_module = EMModule(params.temperature)
    context_module = prep_recurrent_network(context_module, params.state_d)
    em_module = prep_EM(em_module, len_memory, params.state_d, FILL)
    return context_module, em_module


def run_participant(params, data_loader, len_memory=2):
    loss_fn = nn.BCELoss()
    context_module, em_module = gen_model(params, len_memory=len_memory)
    optimizer = torch.optim.SGD(lr=params.episodic_lr, params=context_module.parameters())
    em_preds = []

    # Loop over each state of the CSW task.
    for trial, (x, _, y) in enumerate(data_loader):

        # For each state, optimize the context representation for predicting the next state over n_optimization_steps.
        for _ in range(params['n_optimization_steps']):
            context = context_module(x)  # retrieve the context representation from the integrator.

            # Skip first state bc which sequence within the context is randomly assigned.
            # i.e., we have not yet observed a full state transition.
            if trial >= 0:
                optimizer.zero_grad()  # Zero the gradients before each optimization step.
                pred_em = em_module(x, context)  # retrieve the next state prediction from the EM module.
                loss = loss_fn(pred_em, y)  # calculate the loss between the predicted and actual next state.
                loss.backward()  # compute the gradients of the context module.
                optimizer.step()  # backprop to update context module weights.
            else:
                pred_em = torch.zeros([1, params.output_d]).float()
        with torch.no_grad():
            em_module.write(x, context, y)
            em_preds.append(pred_em.cpu().detach().numpy())

    # Collect some metrics from the training run for analysis.
    em_preds = np.stack(em_preds).squeeze()
    # print('***')
    # print('print state')
    # print(em_module.values)
    # print('prev state')
    # print(em_module.state_keys)
    # print('context')
    # print(em_module.context_keys)
    # print('***')
    return em_preds