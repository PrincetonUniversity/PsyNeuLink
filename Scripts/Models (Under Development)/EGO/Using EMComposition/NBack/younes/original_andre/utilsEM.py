## pepared for publication
from utilsCommon import *


## main


def generate_trial(nback, tstep, stype=0):
    def gen_subseq_stim():
        A = np.random.randint(0, NSTIM)
        B = np.random.choice(np.setdiff1d(np.arange(NSTIM), [A]))
        C = np.random.choice(np.setdiff1d(np.arange(NSTIM), [A, B]))
        X = np.random.choice(np.setdiff1d(np.arange(NSTIM), [A, B]))
        return A, B, C, X

    def genseqCT(nback, tstep):
        # ABXA / AXA
        seq = np.random.randint(0, NSTIM, EXPLEN)
        A, B, C, X = gen_subseq_stim()
        #
        if nback == 3:
            subseq = [A, B, X, A]
        elif nback == 2:
            subseq = [A, X, A]
        seq[tstep - (nback + 1) : tstep] = subseq
        return seq[:tstep]

    def genseqCF(nback, tstep):
        # ABXC
        seq = np.random.randint(0, NSTIM, EXPLEN)
        A, B, C, X = gen_subseq_stim()
        #
        if nback == 3:
            subseq = [A, B, X, C]
        elif nback == 2:
            subseq = [A, X, B]
        seq[tstep - (nback + 1) : tstep] = subseq
        return seq[:tstep]

    def genseqLT(nback, tstep):
        # AAXA
        seq = np.random.randint(0, NSTIM, EXPLEN)
        A, B, C, X = gen_subseq_stim()
        #
        if nback == 3:
            subseq = [A, A, X, A]
        elif nback == 2:
            subseq = [A, A, A]
        seq[tstep - (nback + 1) : tstep] = subseq
        return seq[:tstep]

    def genseqLF(nback, tstep):
        # ABXB
        seq = np.random.randint(0, NSTIM, EXPLEN)
        A, B, C, X = gen_subseq_stim()
        #
        if nback == 3:
            subseq = [A, B, X, B]
        elif nback == 2:
            subseq = [X, A, A]
        seq[tstep - (nback + 1) : tstep] = subseq
        return seq[:tstep]

    genseqL = [genseqCT, genseqLT, genseqCF, genseqLF]
    stim = genseqL[stype](nback, tstep)
    ytarget = [1, 1, 0, 0][stype]
    ctxt = spherical_drift(tstep)
    return stim, ctxt, ytarget


def run_model_trial(stim, ctxt, ctrl, argsD, ffwm=False):
    """
    stim: list of ints, embed inside
    ctxt: array [time,cdim]
    ctrl: [0,1] indicates [2-back,4-back]

    To simulate a single nback trial:
        calculate memory similarity
        if most similar below thresh,
            return no match
        sample from memory, pass to ffwm
            if match, terminate
            if nomatch, continue
        w.p. hazard_rate
            return nomatch, terminate
    """
    if not ffwm:
        # load pretrained WM
        from utilsWM import FFWM

        ffwm = FFWM(indim, hiddim)
        netpath = "trained-net.pt"
        ffwm.load_state_dict(tr.load(netpath))
        ffwm.eval()

    softmax = lambda sim, tau: tr.exp(sim * tau) / tr.sum(tr.exp(sim * tau))

    # unpack arguments
    smtemp = argsD["smtemp"]
    retrieval_sweight = argsD["stim_weight"]
    h_rate = argsD["hrate"]
    # prepare inputs
    stim = tr.eye(SDIM)[stim]
    ctxt = tr.Tensor(ctxt)
    stim_t = stim[-1:]
    ctxt_t = ctxt[-1:]
    stim_M = stim[:-1]
    ctxt_M = ctxt[:-1]
    # loop over trials
    done = False
    IORL = []
    while not done:
        # calculate retrieval probability
        ctxt_sim = cosine_similarity(ctxt_t, ctxt_M)
        stim_sim = cosine_similarity(stim_t, stim_M)
        sim = retrieval_sweight * stim_sim + (1 - retrieval_sweight) * ctxt_sim
        retrieval_pr = softmax(tr.Tensor(sim), smtemp)
        # sample EM
        midx = Categorical(retrieval_pr).sample()
        stim_m = stim_M[midx]
        ctxt_m = ctxt_M[midx]
        ## inhibition of return
        ctxt_M = tr.cat([ctxt_M[:midx, :], ctxt_M[-midx + 1 :, :]])
        stim_M = tr.cat([stim_M[:midx, :], stim_M[-midx + 1 :, :]])
        # forward pass ffwm
        inputL = [stim_t, stim_m, ctxt_t, ctxt_m]
        yhat_act = ffwm(inputL, ctrl)
        yhat = maxsoftmax(yhat_act).item()
        ## terminate conditions
        if yhat == 1:
            done = True
            return yhat
        # hazard rate
        elif np.random.binomial(1, h_rate):
            yhat = 0
            done = True
        # IOR
        elif len(ctxt_M) == 0:
            yhat = 0
            done = True
    return yhat


def run_EMexp(neps, tsteps, argsD):
    score = np.zeros([2, 4, tsteps, neps])
    for ep in range(neps):
        for nback in nbackL:
            for seq_int, tstep in itertools.product(range(4), np.arange(5, tsteps)):
                print(ep, tstep)
                stim, ctxt, ytarget = generate_trial(nback, tstep, stype=seq_int)
                yhat = run_model_trial(stim, ctxt, nback - 2, argsD)
                score[nback - 2, seq_int, tstep, ep] = int(yhat == ytarget)
    print(score.shape)  # nback,seqtype,tsteps,epoch
    acc = score.mean((2, 3))
    return acc, score


# calculate d' based on hit rate and false alarm rate as in Kane et al
def paper_dprime(hit_rate, fa_rate):
    """returns dprime and sensitivity"""

    def clamp(n, minn, maxn):
        return max(min(maxn, n), minn)

    # hit_rate = clamp(hit_rate, 0.01, 0.99)
    # fa_rate = clamp(fa_rate, 0.01, 0.99)

    dl = np.log(hit_rate * (1 - fa_rate) / ((1 - hit_rate) * fa_rate))
    c = 0.5 * np.log((1 - hit_rate) * (1 - fa_rate) / (hit_rate * fa_rate))
    return dl, c


# def get_lure_context(cdrift, context_t_idx, ntokens, setsize):
#     try:
#         nback_context_idx = context_t_idx - setsize
#         rlo = max(0, nback_context_idx - 2)
#         rhi = min(nback_context_idx + setsize, ntokens)
#         idx_context_m = np.random.choice(np.setdiff1d(range(rlo, rhi), nback_context_idx))
#         context_m = cdrift[idx_context_m]
#         return context_m
#     except:
#         print("Error in generating lure context")
#         return


maxsoftmax = lambda x: tr.argmax(tr.softmax(x, -1), -1).squeeze()
