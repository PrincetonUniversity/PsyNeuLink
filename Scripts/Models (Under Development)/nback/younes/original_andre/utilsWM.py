## pepared for publication
from utilsCommon import *


## feed-forward WM
class FFWM(tr.nn.Module):
    """Model used for Sternberg and N-back"""

    def __init__(self, indim, hiddim, outdim=2, bias=False):
        super().__init__()
        self.indim = indim
        self.hiddim = hiddim
        self.hid1_layer = tr.nn.Linear(indim, indim, bias=bias)
        self.hid2_layer = tr.nn.Linear(indim, hiddim, bias=bias)
        self.out_layer = tr.nn.Linear(hiddim, outdim, bias=bias)
        self.drop2 = tr.nn.Dropout(p=0.05, inplace=False)
        bias_dim = indim
        max_num_bias_modes = 10
        self.embed_bias = tr.nn.Embedding(max_num_bias_modes, bias_dim)
        return None

    def forward(self, inputL, control_bias_int=0):
        """inputL is list of tensors"""
        hid1_in = tr.cat(inputL, -1)
        hid1_act = self.hid1_layer(hid1_in).relu()
        control_bias = self.embed_bias(tr.tensor(control_bias_int))
        hid2_in = hid1_act + control_bias
        hid2_in = self.drop2(hid2_in)
        hid2_act = self.hid2_layer(hid2_in).relu()
        yhat_t = self.out_layer(hid2_act)
        return yhat_t


## WM training


def run_model_for_epochs(net, taskL, training, neps_per_task, verb=True):

    maxsoftmax = lambda x: tr.argmax(tr.softmax(x, -1), -1).squeeze()
    if training:
        net.train()
        print("Training WM...")
    else:
        net.eval()
        print("Evaluating WM...")
    lossop = tr.nn.CrossEntropyLoss()
    optiop = tr.optim.Adam(net.parameters(), lr=0.001)
    score = -np.ones([len(taskL), neps_per_task])
    ttype = -np.ones([len(taskL), neps_per_task])
    for ep in range(neps_per_task):
        if verb and ep % (neps_per_task / 5) == 0:
            print(ep / neps_per_task)
        # resample stim and context on each ep
        stimset = tr.Tensor(np.eye(20))
        cdrift = spherical_drift()
        cdrift = tr.Tensor(cdrift)
        # interleave train on every task per epoch
        for task_idx, (control_int, sample_trial_fn, setsize) in enumerate(taskL):
            # use the input function to generate a trial sample
            out = sample_trial_fn(stimset, cdrift, setsize)
            stim_t, stim_m, context_t, context_m, ytarget, ttype_idx = out
            # forward prop
            inputL = [stim_t, stim_m, context_t, context_m]
            yhat = net(inputL, control_bias_int=control_int)
            # eval
            score[task_idx, ep] = maxsoftmax(yhat) == ytarget
            ttype[task_idx, ep] = ttype_idx
            # backprop
            if training:
                eploss = lossop(yhat.unsqueeze(0), ytarget)
                optiop.zero_grad()
                eploss.backward(retain_graph=True)
                optiop.step()
    return score, ttype


## experiment generation funs

# generate single trial
def single_nback_comparison(
    stimset,
    cdrift,
    setsize,
    pr_match=PMATCH,
    pr_stim_lure=PSLURE,
    pr_context_lure=PCLURE,
):
    """
    given
      @ stimset: array with stimulus vectors
      @ cdrift: draw from the context drift
      @ setsize: nback target step
    returns
      @ stim_t: current stimulus
      @ stim_m: stimulus from memory
      @ context_t: current context
      @ context_m: context from memory
      @ ytarget: indicates whether nback match or no
      @ ttype_code: encodes the trial type (used for error reporting).
    """
    ntokens, sdim = stimset.shape
    min_context_t = setsize

    # set current stim and context
    stim_t_idx = np.random.randint(0, ntokens)
    context_t_idx = np.random.randint(min_context_t, ntokens)
    stim_t = stimset[stim_t_idx]
    context_t = cdrift[context_t_idx]

    ttype_randn = np.random.random()  # randomly-selected trial type
    ttype_code = -1  # code used to record trial type for analysis

    if ttype_randn < pr_match:
        stim_m, context_m = nback_both_match(
            stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens
        )
        ytarget = tr.LongTensor([1])
        ttype_code = 0

    elif ttype_randn < (pr_match + pr_context_lure):
        stim_m, context_m = nback_ctxt_lure(
            stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens
        )
        ytarget = tr.LongTensor([0])
        ttype_code = 1

    elif ttype_randn < (pr_match + pr_context_lure + pr_stim_lure):
        stim_m, context_m = nback_stim_lure(
            stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens
        )
        ytarget = tr.LongTensor([0])
        ttype_code = 2

    else:
        stim_m, context_m = nback_neither_match(
            stimset, cdrift, stim_t_idx, context_t_idx, setsize, ntokens
        )
        ytarget = tr.LongTensor([0])
        ttype_code = 3

    return stim_t, stim_m, context_t, context_m, ytarget, ttype_code


# trial types
def nback_both_match(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    stim_m = stim_set[stim_t_idx]
    context_m = cdrift[context_t_idx - setsize]
    return (stim_m, context_m)


def nback_stim_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    stim_m = stim_set[stim_t_idx]
    context_m = get_lure_context(cdrift, context_t_idx, ntokens, setsize)
    return (stim_m, context_m)


def nback_ctxt_lure(stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens):
    idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
    stim_m = stim_set[idx_stim_m]
    context_m = cdrift[context_t_idx - setsize]
    return (stim_m, context_m)


def nback_neither_match(
    stim_set,
    cdrift,
    stim_t_idx,
    context_t_idx,
    setsize,
    ntokens,
    pr_prewindow_match=0.0,
):
    """
    return a neither match trace -- the stimuli don't match, and the context is not the n-back context.
    optionally, for the EM simulations, this can probabilistically return a matching stimulus and a long-past context
    (s.t. the trace isn't a proper lure, but simulates the repeating-stimuli dynamics of the task).
    """
    if np.random.uniform() > pr_prewindow_match or ntokens - context_t_idx < 6:
        idx_stim_m = np.random.choice(np.setdiff1d(range(ntokens), stim_t_idx))
        stim_m = stim_set[idx_stim_m]
        context_m = get_lure_context(cdrift, context_t_idx, ntokens, setsize)
        return stim_m, context_m
    else:
        return nback_distant_slure(
            stim_set, cdrift, stim_t_idx, context_t_idx, setsize, ntokens
        )


# helper
def get_lure_context(cdrift, context_t_idx, ntokens, setsize):
    try:
        nback_context_idx = context_t_idx - setsize
        rlo = max(0, nback_context_idx - 2)
        rhi = min(nback_context_idx + setsize, ntokens)
        idx_context_m = np.random.choice(
            np.setdiff1d(range(rlo, rhi), nback_context_idx)
        )
        context_m = cdrift[idx_context_m]
        return context_m
    except:
        print("Error in generating lure context")
        return
