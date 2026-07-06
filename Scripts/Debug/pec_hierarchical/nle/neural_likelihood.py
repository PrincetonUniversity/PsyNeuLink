"""Neural (MNLE-style) likelihood for the two-boundary DDM used in the hierarchical PEC study.

A mixture density network learns p(choice, rt | drift, threshold) from simulated trials, giving a
fast differentiable-free surrogate for the KDE simulation likelihood. Matches the PsyNeuLink
process exactly: X0 = 0, symmetric bounds +/-threshold, drift = rate * stimulus, unit noise,
Euler-Maruyama with dt = 0.01, rt = non_decision_time + first-passage time, decision 1 = upper.

Also provides the exact Navarro-Fuss (2009) analytical density as ground truth for validating
both the neural and the KDE likelihoods.
"""

import numpy as np
import torch
import torch.nn as nn

DT = 0.01
NOISE = 1.0
NDT = 0.15
LOG_ZERO = np.log(1e-10)

DRIFT_RANGE = (-3.2, 3.2)      # covers rate in [-1.5, 1.5] x stimulus in {+/-0.5, 1, 2}
THRESH_RANGE = (0.28, 1.55)


# --- simulator ------------------------------------------------------------------------------

def simulate_ddm(drift, threshold, rng, dt=DT, noise=NOISE, max_t=20.0):
    """Vectorized Euler-Maruyama first-passage simulation.

    Returns (choice, fpt, valid): choice 1 = upper bound, fpt in seconds; valid False for the
    (rare) paths not absorbed by max_t.
    """
    drift = np.asarray(drift, float)
    threshold = np.asarray(threshold, float)
    n = drift.size
    x = np.zeros(n)
    fpt = np.full(n, np.nan)
    choice = np.zeros(n, dtype=np.int64)
    active = np.arange(n)
    sqdt = np.sqrt(dt) * noise
    n_steps = int(max_t / dt)
    for step in range(1, n_steps + 1):
        x[active] += drift[active] * dt + sqdt * rng.standard_normal(active.size)
        crossed = np.abs(x[active]) >= threshold[active]
        if crossed.any():
            idx = active[crossed]
            fpt[idx] = step * dt
            choice[idx] = (x[idx] > 0).astype(np.int64)
            active = active[~crossed]
            if active.size == 0:
                break
    valid = ~np.isnan(fpt)
    return choice, fpt, valid


def make_training_set(n_configs, trials_per_config, rng):
    """Simulate trials at parameters spanning the fit box; mirror-augment to enforce symmetry."""
    drift = rng.uniform(*DRIFT_RANGE, size=n_configs).repeat(trials_per_config)
    thresh = rng.uniform(*THRESH_RANGE, size=n_configs).repeat(trials_per_config)
    choice, fpt, valid = simulate_ddm(drift, thresh, rng)
    drift, thresh, choice, fpt = drift[valid], thresh[valid], choice[valid], fpt[valid]
    # p(c, t | d, a) == p(1 - c, t | -d, a)
    drift = np.concatenate([drift, -drift])
    thresh = np.concatenate([thresh, thresh])
    choice = np.concatenate([choice, 1 - choice])
    fpt = np.concatenate([fpt, fpt])
    return drift, thresh, choice, fpt


# --- mixture density network ----------------------------------------------------------------

class DDMMixtureNet(nn.Module):
    """conditioning vector -> Bernoulli(choice) x Gaussian mixture over standardized log time.

    For the plain DDM the conditioning is (drift, threshold); for richer compositions it is the
    fitted parameters plus per-trial input features.
    """

    def __init__(self, n_comp=6, hidden=128, n_inputs=2):
        super().__init__()
        self.n_comp = n_comp
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.head_choice = nn.Linear(hidden, 1)
        self.head_mix = nn.Linear(hidden, 2 * n_comp * 3)  # per choice: logits, means, log-scales
        # normalization constants, filled by fit()
        self.register_buffer("cond_mean", torch.zeros(n_inputs))
        self.register_buffer("cond_std", torch.ones(n_inputs))
        self.register_buffer("y_mean", torch.zeros(1))
        self.register_buffer("y_std", torch.ones(1))

    def _params(self, cond):
        h = self.trunk((cond - self.cond_mean) / self.cond_std)
        logit = self.head_choice(h).squeeze(-1)
        mix = self.head_mix(h).view(-1, 2, 3, self.n_comp)
        return logit, mix

    def log_prob(self, cond, choice, y_std):
        """log p(choice) + log p(y_std | choice) for standardized y = (log fpt - m) / s."""
        logit, mix = self._params(cond)
        lp_choice = -nn.functional.softplus(torch.where(choice == 1, -logit, logit))
        m = mix[torch.arange(cond.shape[0]), choice]        # (B, 3, n_comp)
        logw = torch.log_softmax(m[:, 0], dim=-1)
        mean, log_scale = m[:, 1], m[:, 2].clamp(-6.0, 3.0)
        z = (y_std.unsqueeze(-1) - mean) * torch.exp(-log_scale)
        lp_comp = -0.5 * z ** 2 - log_scale - 0.5 * np.log(2 * np.pi)
        return lp_choice + torch.logsumexp(logw + lp_comp, dim=-1)

    @torch.no_grad()
    def cond_log_density(self, cond, choice, rt, ndt=0.0):
        """log p(choice, rt) in natural rt units for an arbitrary conditioning matrix."""
        fpt = np.asarray(rt, float) - ndt
        ok = fpt > 0
        out = np.full(fpt.shape, LOG_ZERO)
        if ok.any():
            cond_t = torch.tensor(np.asarray(cond, float)[ok], dtype=torch.float32)
            y = (torch.tensor(np.log(fpt[ok]), dtype=torch.float32) - self.y_mean) / self.y_std
            c = torch.tensor(np.asarray(choice)[ok], dtype=torch.long)
            lp = self.log_prob(cond_t, c, y)
            # change of variables: y_std -> rt
            out[ok] = lp.numpy() - np.log(self.y_std.item()) - np.log(fpt[ok])
        return out

    def trial_log_density(self, drift, thresh, choice, rt, ndt=NDT):
        """DDM convenience wrapper: conditioning is (drift, threshold)."""
        cond = np.stack([np.asarray(drift, float), np.asarray(thresh, float)], axis=1)
        return self.cond_log_density(cond, choice, rt, ndt=ndt)


def fit(net, cond, choice, fpt, epochs=25, batch=4096, lr=1e-3, seed=0, log=print):
    """Train by maximum likelihood on simulated (conditioning, outcome) pairs."""
    g = torch.Generator().manual_seed(seed)
    cond = torch.tensor(np.asarray(cond, float), dtype=torch.float32)
    y = torch.tensor(np.log(fpt), dtype=torch.float32)
    c = torch.tensor(choice, dtype=torch.long)
    net.cond_mean.copy_(cond.mean(0))
    net.cond_std.copy_(cond.std(0))
    net.y_mean.copy_(y.mean().unsqueeze(0))
    net.y_std.copy_(y.std().unsqueeze(0))
    y_std = (y - net.y_mean) / net.y_std

    n = cond.shape[0]
    n_val = min(50_000, n // 10)
    perm = torch.randperm(n, generator=g)
    val, trn = perm[:n_val], perm[n_val:]
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        order = trn[torch.randperm(trn.numel(), generator=g)]
        for i in range(0, order.numel(), batch):
            idx = order[i:i + batch]
            loss = -net.log_prob(cond[idx], c[idx], y_std[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_nll = -net.log_prob(cond[val], c[val], y_std[val]).mean().item()
        log(f"epoch {epoch + 1:3d}  val NLL/trial = {val_nll:.4f}")
    return net


# --- likelihood interfaces ------------------------------------------------------------------

class NeuralGroupLikelihood:
    """log_likelihood_s(theta, s) over per-subject DataFrames, backed by the trained net."""

    def __init__(self, net, subject_frames, stimuli):
        self.net = net
        self.frames = subject_frames    # list of DataFrames with decision, response_time
        self.stimuli = stimuli          # (num_trials,) shared stimulus sequence

    def log_likelihood_s(self, theta, s):
        rate, thresh = float(theta[0]), float(theta[1])
        df = self.frames[s]
        n = len(df)
        lp = self.net.trial_log_density(
            rate * self.stimuli[:n], np.full(n, thresh),
            df["decision"].to_numpy(float).astype(int), df["response_time"].to_numpy(float),
        )
        return float(lp.sum())


# --- exact analytical reference (Navarro & Fuss 2009) ----------------------------------------

def _ftt01w(tt, w, err=1e-10):
    """Standardized first-passage density at the lower bound of [0, 1], drift 0, start w."""
    tt = np.asarray(tt, float)
    p = np.zeros_like(tt)
    small = tt < 0.7
    if small.any():
        ts = tt[small]
        k = np.arange(-6, 7)[None, :]
        terms = (w + 2 * k) * np.exp(-((w + 2 * k) ** 2) / (2 * ts[:, None]))
        p[small] = terms.sum(axis=1) / np.sqrt(2 * np.pi * ts ** 3)
    if (~small).any():
        tl = tt[~small]
        k = np.arange(1, 12)[None, :]
        terms = k * np.exp(-(k ** 2) * np.pi ** 2 * tl[:, None] / 2) * np.sin(k * np.pi * w)
        p[~small] = np.pi * terms.sum(axis=1)
    return np.maximum(p, 0.0)


def wfpt_logpdf(rt, choice, drift, threshold, ndt=NDT):
    """Exact log density of (choice, rt) for the symmetric-bound DDM (unit noise)."""
    rt = np.asarray(rt, float)
    choice = np.asarray(choice, int)
    drift = np.broadcast_to(np.asarray(drift, float), rt.shape).copy()
    threshold = np.broadcast_to(np.asarray(threshold, float), rt.shape)
    t = rt - ndt
    out = np.full(rt.shape, LOG_ZERO)
    ok = t > 0
    if not ok.any():
        return out
    # upper-bound hits are lower-bound hits of the sign-flipped process; w = 0.5 is symmetric
    v = np.where(choice == 1, -drift, drift)[ok]
    a = 2.0 * threshold[ok]
    tt = t[ok] / a ** 2
    f = _ftt01w(tt, 0.5) / a ** 2
    logp = np.where(f > 0, np.log(np.maximum(f, 1e-300)) - v * a * 0.5 - (v ** 2) * t[ok] / 2, LOG_ZERO)
    out[ok] = np.maximum(logp, LOG_ZERO)
    return out
