## pepared for publication
import os
import itertools
import numpy as np
import torch as tr
from sklearn.metrics.pairwise import cosine_similarity
from torch.distributions import Categorical
from matplotlib import pyplot as plt
from utilsWM import *

## trial generation params
PMATCH, PSLURE, PCLURE, pr_nomatch = 0.4, 0.2, 0.2, 0.2

# exp params (imported)
nbackL = [2, 3]
EXPLEN = 48
NSTIM = 8
# context params
CDIM = 25
CVAR = 0.075
CMEAN = 0.25
CSTEPS = EXPLEN
# layers
SDIM = 20
indim = 2 * (CDIM + SDIM)
hiddim = SDIM * 4

## Context drift


def spherical_drift(n_steps=CSTEPS, dim=CDIM, var=CVAR, mean=CMEAN):
    """
    # model of CR as spherical coordinates updated by a noisy drift
    """

    def convert_spherical_to_angular(dim, ros):
        ct = np.zeros(dim)
        ct[0] = np.cos(ros[0])
        prod = np.prod([np.sin(ros[k]) for k in range(1, dim - 1)])
        n_prod = prod
        for j in range(dim - 2):
            n_prod /= np.sin(ros[j + 1])
            amt = n_prod * np.cos(ros[j + 1])
            ct[j + 1] = amt
        ct[dim - 1] = prod
        return ct

    # initialize the spherical coordinates to ensure each context run begins in a new random location on the unit sphere
    ros = np.random.random(dim - 1)
    slen = n_steps
    ctxt = np.zeros((slen, dim))
    for i in range(slen):
        noise = np.random.normal(
            mean, var, size=(dim - 1)
        )  # add a separately-drawn Gaussian to each spherical coord
        ros += noise
        ctxt[i] = convert_spherical_to_angular(dim, ros)
    return ctxt


# ctxt_fn = lambda n_steps: spherical_drift(n_steps=n_steps)


## plot and eval funs


def plot_train_accuracy(score_tr, ttype, ssizeL, figure_path=""):
    task_labels = ["setsize " + str(ssize) for ssize in ssizeL]
    colors = [
        ["#edc174", "#d4982f", "#c27e08", "#a1690a"],
        ["#8dcbf7", "#61b3ed", "#2a86c7", "#0762a3"],
    ]
    labels = ["match", "slure", "clure", "nomatch"]
    n_intervals = 1000
    for i in range(len(score_tr)):
        task_score = score_tr[i]
        task_color = colors[i]
        task_trialtypes = ttype[i]
        for tt in range(4):
            filt_inds = task_trialtypes == tt
            ep_ttype = np.extract(filt_inds, task_score)
            ep_ttype = ep_ttype[: -(len(ep_ttype) % n_intervals)]
            ac = ep_ttype.reshape(-1, n_intervals).mean(1)
            lt = task_labels[i] + " " + labels[tt]
            plt.plot(ac, color=task_color[tt], label=lt)
    plt.legend(loc="best")
    plt.ylim(0, 1)
    plt.ylabel("Train accuracy")
    plt.savefig(figure_path + "/train-accuracy")
    plt.close("all")


def eval_by_ttype(net, sample_fn, taskintL, ssizeL, neps):
    """
    eval on given task for separate trial types
    returns evac on (match,nomatch,slure,clure)
    """

    taskL_ev = []
    # generate a list of tasks which are trials all of one kind so we can see accuracy by trial type
    for task_int in taskintL:
        taskL_ev.append([task_int, sample_fn(1, 0, 0), ssizeL[task_int]])
        taskL_ev.append([task_int, sample_fn(0, 0, 0), ssizeL[task_int]])
        taskL_ev.append([task_int, sample_fn(0, 1, 0), ssizeL[task_int]])
        taskL_ev.append([task_int, sample_fn(0, 0, 1), ssizeL[task_int]])

    evsc, ttype = run_model_for_epochs(
        net, taskL_ev, training=False, neps_per_task=neps, verb=False
    )

    evac = evsc.mean(1)
    print(evac)
    # regroup the list of scores into a list of lists grouped by the setsize
    scores_by_ss = [[]] * len(taskintL)
    for task_int in taskintL:
        scores_by_ss[task_int] = evac[task_int * 4 : (task_int + 1) * 4]
    return scores_by_ss


def plot_accuracy_by_trial_type(evac, taskintL, ssizeL, figure_path=""):
    for tidx, (task_int, ssize) in enumerate(zip(taskintL, ssizeL)):
        plt.title("Accuracy by trial type")
        plt.bar(
            np.arange(4) + (0.45 * tidx),
            evac[tidx],
            width=0.45,
            label="setsize:" + str(ssize),
        )
    plt.legend()
    plt.xticks(range(4), ["match", "nomatch", "slure", "clure"])
    if not False:
        plt.savefig(figure_path + "/trial-type-accuracy")
        plt.close("all")
