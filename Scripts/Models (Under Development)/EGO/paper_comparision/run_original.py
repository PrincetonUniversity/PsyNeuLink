import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from torch_model_original import EMModule, RecurrentContextModule, gen_model

from data import *


def calc_accuracy(pred, true):
    return ((pred.sum() > 2.5) == (true.sum() > 2.5)).float().item()


def filter_run(run_em, thresh=0.8):
    '''Filter runs that learn uniform context representations. 
    This usually only happens in a smaller number of seeds, but helps clean up the results.
    '''
    foil = np.zeros([4, 4])
    foil[::2, ::2] = 1
    foil[1::2, 1::2] = 1
    run_em = run_em.reshape(200, 5, 11).mean(axis=1)
    mat = cosine_similarity(run_em, run_em)
    vec = mat[:160, :160].reshape(4, 40, 4, 40).mean(axis=(1, 3)).ravel()
    return cosine_similarity(foil.reshape(1, -1), vec.reshape(1, -1))[0][0]


def calc_prob(em_preds, test_ys):
    # only consider the terminal three states (they are the only predictable transitions).
    em_preds_new, test_ys_new = em_preds[:, 2:-1, :], test_ys[:, 2:-1, :]
    em_probability = (em_preds_new * test_ys_new).sum(-1).mean(-1)
    trial_probs = (em_preds * test_ys)
    return em_probability, trial_probs


def run_participant(params, training_paradigm):
    performance_data = {'seed': [], 'paradigm': [], 'trial': [], 'probability': []}
    loss_fn = nn.BCELoss()
    data_loader = gen_data_loader(training_paradigm)
    context_module, em_module = gen_model(params)

    optimizer = torch.optim.SGD([{'params': context_module.parameters(), 'lr': params.episodic_lr}])
    em_accuracy = []
    em_preds = []
    em_contexts = []
    em_probs = []
    for trial, (x, _, y) in enumerate(data_loader):
        if trial < 1:
            continue
        for _ in range(params['n_optimization_steps']):
            context = context_module(x)
            optimizer.zero_grad()
            pred_em = em_module(x, context)
            loss = loss_fn(pred_em, y)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            em_module.write(x, context, y)
            em_accuracy.append(calc_accuracy(pred_em, y))
            em_preds.append(pred_em.cpu().detach().numpy())
            em_contexts.append(context.cpu().detach().numpy())

    # Collect some training data for analysis.
    em_contexts.append(np.zeros([1, params.context_d]))
    em_preds = np.stack(em_preds).squeeze()
    em_preds = np.vstack([em_preds, np.zeros([1, 11])]).reshape(-1, 5, 11)
    test_ys = np.vstack([data_loader.dataset.ys.cpu().numpy()[1:], np.zeros([1, 11])]).reshape(-1, 5, 11)
    correct_prob, trial_probs = calc_prob(em_preds, test_ys)
    em_probs.append(trial_probs)
    performance_data['probability'].extend(correct_prob)
    performance_data['seed'].extend([params.seed] * len(correct_prob))
    performance_data['paradigm'].extend([training_paradigm] * len(correct_prob))
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
