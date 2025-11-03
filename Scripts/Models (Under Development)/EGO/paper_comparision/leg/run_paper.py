from random import randint

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import torch
import torch.nn as nn
from torch.utils.data import dataset

from models_paper import EMModule, RecurrentContextModule
import utils
from data import *
    

def gen_data_loader(paradigm, probs=[0.65, 0.65, 0.65]):
    if paradigm=='blocked':
        contexts_to_load = [0,1,0,1] + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [40,40,40,40] + [1]*40
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    elif paradigm == 'interleaved':
        contexts_to_load = [0,1]*80 + [randint(0,1) for _ in range(40)]
        n_samples_per_context = [1]*160 + [1]*40
        ds = CSWDataset(n_samples_per_context, contexts_to_load, probs=probs)
    return torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    

""" Model preparation functions """
def prep_recurrent_network(rnet, state_d, persistance=-0.6):
    with torch.no_grad():
        rnet.state_to_hidden.weight.copy_(torch.eye(state_d, dtype=torch.float))
        rnet.state_to_hidden.bias.zero_()
        rnet.hidden_to_hidden.weight.zero_()
        rnet.hidden_to_hidden.bias.zero_()
        rnet.state_to_hidden_wt.weight.zero_()
        rnet.state_to_hidden_wt.bias.copy_(torch.ones((len(rnet.state_to_hidden_wt.bias),), dtype=torch.float) * persistance)
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


def gen_model(params):
    context_module = RecurrentContextModule(params.state_d, params.state_d, params.context_d)
    em_module = EMModule(params.temperature)
    context_module = prep_recurrent_network(context_module, params.state_d, params.persistance)
    return context_module, em_module

def calc_accuracy(pred, true):
    return ((pred.sum()>2.5)==(true.sum()>2.5)).float().item()

def filter_run(run_em, thresh=0.8):
    '''Filter runs that learn uniform context representations. 
    This usually only happens in a smaller number of seeds, but helps clean up the results.
    '''
    foil = np.zeros([4,4])
    foil[::2, ::2] = 1
    foil[1::2, 1::2] = 1
    run_em = run_em.reshape(200, 5, 11).mean(axis=1)
    mat = cosine_similarity(run_em, run_em)
    vec = mat[:160, :160].reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()
    return cosine_similarity(foil.reshape(1, -1), vec.reshape(1, -1))[0][0]

def calc_prob(em_preds, test_ys):
    # only consider the terminal three states (they are the only predictable transitions).
    em_preds_new, test_ys_new = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds_new*test_ys_new).sum(-1).mean(-1)
    trial_probs = (em_preds*test_ys) 
    return em_probability, trial_probs

def run_participant(params, training_paradigm):
    performance_data = {'seed':[], 'paradigm':[], 'trial':[], 'probability':[]}
    loss_fn = nn.BCELoss()
    data_loader = gen_data_loader(training_paradigm, params['probs'])
    context_module, em_module = gen_model(params)
    optimizer = torch.optim.SGD([{'params': context_module.parameters(), 'lr': params.episodic_lr}])
    em_accuracy = []
    em_preds = []
    em_contexts = []
    em_probs = []
    for trial, (x,_,y) in enumerate(data_loader):
        for _ in range(params['n_optimization_steps']):
            # context_module.update_hidden_state = _ == params['n_optimization_steps'] - 1
            context = context_module(x)
            if trial > 0:
                optimizer.zero_grad()
                pred_em = em_module(x,context)
                loss = loss_fn(pred_em,y)
                loss.backward()
                optimizer.step()
            else:
                pred_em = torch.zeros([1,params.output_d]).float()
        with torch.no_grad():
            em_module.write(x,context,y)
            em_accuracy.append(calc_accuracy(pred_em,y))
            em_preds.append(pred_em.cpu().detach().numpy())
            em_contexts.append(context.cpu().detach().numpy())

    # Collect some training data for analysis.
    em_contexts.append(np.zeros([1,params.context_d]))
    em_preds = np.stack(em_preds).squeeze()
    em_preds = np.vstack([em_preds, np.zeros([1,11])]).reshape(-1,5,11)
    test_ys = np.vstack([data_loader.dataset.ys.cpu().numpy(), np.zeros([1,11])]).reshape(-1,5,11)
    correct_prob, trial_probs = calc_prob(em_preds, test_ys)
    em_probs.append(trial_probs)
    performance_data['probability'].extend(correct_prob)
    performance_data['seed'].extend([params.seed]*len(correct_prob))
    performance_data['paradigm'].extend([training_paradigm]*len(correct_prob))
    performance_data['trial'].extend(list(range(len(correct_prob))))
    run_sim_score = filter_run(em_preds)
    return pd.DataFrame(performance_data), em_probs, em_contexts, run_sim_score


def run_experiment(params):
    performance_data = []
    correct_probs = []
    context_reps = []
    sim_scores = []
    for i in range(params.n_participants):
        utils.set_random_seed(i)
        for training_paradigm in params['paradigms']:
            if 'blocked' in training_paradigm.lower():
                thresh = params['sim_thresh']
            else:
                thresh = 0.7
            run_sim_score = 0
            while run_sim_score < thresh:
                utils.set_random_seed(random.randint(0, 10000))
                participant_df, em_probs, em_contexts, run_sim_score = run_participant(params, training_paradigm)
            performance_data.append(participant_df)
            correct_probs.append(em_probs)
            context_reps.append(em_contexts)
            sim_scores.append(run_sim_score)
    exp_df = pd.concat(performance_data).reset_index(drop=True)
    correct_probs = np.stack(correct_probs)
    context_reps = np.stack(context_reps)
    sim_scores = np.array(sim_scores)
    return exp_df, correct_probs, context_reps, sim_scores