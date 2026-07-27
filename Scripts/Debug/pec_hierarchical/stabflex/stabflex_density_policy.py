"""Standalone density-scoring policy for stab-flex simulation likelihoods.

Scores an already-simulated cube ``sim`` of shape (num_trials, num_estimates, 2) with
outcomes (decision, response_time) against observed (decision, rt).  Two axes, mirroring
the DDM study but built here so nothing in the shared/frozen PEC core is touched:

* estimator: ``"fastkde"`` or ``"gaussian"`` (1-D KDE over rt within each choice category)
* scope:     ``local`` (per-trial density) vs ``pooled`` (one density per repeated condition)

Condition-pooling groups trials by their trial features.  Discrete features group exactly;
continuous features (e.g. a varying cue-stimulus interval) are binned first via ``bin_edges``
so pooling still applies when the condition is not exactly repeated -- the honest continuous
analogue of exact condition pooling.

This module only *reads* the frozen core by import; it never edits it.
"""

import numpy as np

FLOOR = 1e-10


def condition_groups(features, bin_edges=None):
    """Group trial indices by (optionally binned) condition.

    features: (num_trials, n_features). bin_edges: optional {feature_dim: edges} to digitize
    continuous dims before grouping. Returns a list of index arrays, one per unique condition.
    """
    feats = np.asarray(features, dtype=float).copy()
    if bin_edges:
        for dim, edges in bin_edges.items():
            feats[:, dim] = np.digitize(feats[:, dim], np.asarray(edges, dtype=float))
    _uniq, inv = np.unique(feats, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    return [np.flatnonzero(inv == g) for g in range(inv.max() + 1)]


def local_groups(num_trials):
    """Each trial is its own group: density from that trial's own draws only."""
    return [np.array([t], dtype=int) for t in range(num_trials)]


def _kde_1d(samples, estimator):
    """Return a vectorized pdf callable for 1-D samples, or None if under-determined."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[np.isfinite(samples)]
    if samples.size < 5 or np.ptp(samples) < 1e-9:
        return None
    if estimator == "gaussian":
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(samples)  # Scott's-rule bandwidth
        return lambda x: np.asarray(kde(np.atleast_1d(x)), dtype=float)
    if estimator == "fastkde":
        from fastkde import fastKDE
        from scipy.interpolate import interpn
        fk = fastKDE.fastKDE(samples, do_save_marginals=False)  # canonical core API
        axes, pdf = fk.axes, fk.pdf

        def _f(x):
            x = np.atleast_1d(np.asarray(x, dtype=float)).reshape(-1, 1)
            vals = interpn(axes, pdf, x, method="linear", bounds_error=False, fill_value=0.0)
            return np.clip(np.asarray(vals, dtype=float), 0.0, None)

        return _f
    raise ValueError(f"unknown estimator {estimator!r}")


def score_cube(sim, obs_dec, obs_rt, groups, estimator, floor=FLOOR):
    """Sum of per-trial log densities of observed (decision, rt) under a group-wise KDE.

    sim: (num_trials, num_estimates, 2) = (decision, rt). For each group, pool that group's
    simulated draws, split by choice, and score every observed trial in the group against the
    pooled density.  ``local`` passes singleton groups; ``pooled`` passes condition groups.
    """
    sim = np.asarray(sim)
    sim_dec = sim[:, :, 0]
    sim_rt = sim[:, :, 1]
    obs_dec = np.asarray(obs_dec).astype(int)
    obs_rt = np.asarray(obs_rt, dtype=float)
    logp = np.full(len(obs_dec), np.log(floor), dtype=float)

    for grp in groups:
        g_dec = sim_dec[grp].reshape(-1)
        g_rt = sim_rt[grp].reshape(-1)
        total = g_dec.size
        if total == 0:
            continue
        for cv in (0, 1):
            trials_cv = grp[obs_dec[grp] == cv]
            if trials_cv.size == 0:
                continue
            mask = g_dec == cv
            p_choice = mask.sum() / total
            pdf = _kde_1d(g_rt[mask], estimator) if mask.sum() >= 5 else None
            if pdf is None or p_choice <= 0.0:
                continue  # leave at log(floor)
            dens = np.maximum(p_choice * pdf(obs_rt[trials_cv]), floor)
            logp[trials_cv] = np.log(dens)
    return float(logp.sum())


def make_groups(config_scope, features, num_trials, bin_edges=None):
    """scope 'local' -> per-trial groups; 'pooled' -> condition groups (binned if continuous)."""
    if config_scope == "local":
        return local_groups(num_trials)
    if config_scope == "pooled":
        return condition_groups(features, bin_edges=bin_edges)
    raise ValueError(f"unknown scope {config_scope!r}")


# The four matrix configs plus a canonical parse of "estimator_scope".
CONFIGS = ("fastkde_local", "fastkde_pooled", "gaussian_local", "gaussian_pooled")


def parse_config(name):
    estimator, scope = name.split("_")
    assert estimator in ("fastkde", "gaussian") and scope in ("local", "pooled")
    return estimator, scope
