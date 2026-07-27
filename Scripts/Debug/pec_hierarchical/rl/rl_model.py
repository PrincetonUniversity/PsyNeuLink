"""Decay-Q bandit + DDM: the sequential-structure test model.

Two-armed bandit with forgetting Q-learning and a DDM read-out:

    Q_t = (1 - alpha) * Q_{t-1} + alpha * (onehot(choice_t) * reward_t)     (both arms decay)
    drift_t = beta * (Q_t-1[1] - Q_t-1[0])
    (choice_t, rt_t) ~ DDM(drift_t, threshold, ndt),  choice 1 = upper = arm 1

The decay update is exactly PsyNeuLink's AdaptiveIntegrator semantics, so the PNL composition,
this numpy generator, and the exact likelihood implement the same model. Fitting clamps the
observed choice/reward history as inputs, making Q_t deterministic given (theta, history) and
the exact per-trial likelihood available through the Navarro-Fuss density.

theta = (alpha, beta, threshold, ndt).
"""

import os
import sys

import numpy as np

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _dir in ("core", "nle"):
    _path = os.path.join(_PARENT, _dir)
    if _path not in sys.path:
        sys.path.append(_path)
from neural_likelihood import simulate_ddm, wfpt_logpdf, LOG_ZERO  # noqa: E402
from transforms import BoundedTransform  # noqa: E402

FIT_PARAMS = ("alpha", "beta", "threshold", "ndt")
FIT_RANGES = {"alpha": (0.02, 0.8), "beta": (0.5, 8.0), "threshold": (0.3, 1.5), "ndt": (0.1, 0.4)}


def fit_bounds():
    lo = np.array([FIT_RANGES[p][0] for p in FIT_PARAMS])
    hi = np.array([FIT_RANGES[p][1] for p in FIT_PARAMS])
    return lo, hi


def transform():
    lo, hi = fit_bounds()
    return BoundedTransform(lo, hi)


def reward_schedule(num_trials, rng, block=40, p_good=0.8, p_bad=0.2):
    """Bernoulli reward probabilities per arm, with the good arm reversing every ``block`` trials."""
    p = np.empty((num_trials, 2))
    good_is_1 = True
    for start in range(0, num_trials, block):
        end = min(start + block, num_trials)
        p[start:end] = [p_bad, p_good] if good_is_1 else [p_good, p_bad]
        good_is_1 = not good_is_1
    return p


def q_path(theta, choices, rewards):
    """Deterministic Q trajectory and per-trial drift given the observed history.

    Returns drift_t computed from Q BEFORE trial t's update (the value driving trial t's DDM).
    """
    alpha, beta = float(theta[0]), float(theta[1])
    n = len(choices)
    q = np.zeros(2)
    drift = np.empty(n)
    for t in range(n):
        drift[t] = beta * (q[1] - q[0])
        signal = np.zeros(2)
        signal[int(choices[t])] = rewards[t]
        q = (1.0 - alpha) * q + alpha * signal
    return drift


def simulate_subject(theta, schedule, rng):
    """Generative run: the model's own stochastic choices drive learning."""
    alpha, beta, thresh, ndt = (float(x) for x in theta)
    n = len(schedule)
    q = np.zeros(2)
    choices = np.empty(n, dtype=int)
    rewards = np.empty(n)
    rts = np.empty(n)
    for t in range(n):
        drift = beta * (q[1] - q[0])
        c, fpt, valid = simulate_ddm(np.array([drift]), np.array([thresh]), rng)
        if not valid[0]:
            c, fpt = np.array([rng.integers(0, 2)]), np.array([20.0])
        choices[t] = int(c[0])
        rts[t] = fpt[0] + ndt
        rewards[t] = float(rng.random() < schedule[t, choices[t]])
        signal = np.zeros(2)
        signal[choices[t]] = rewards[t]
        q = (1.0 - alpha) * q + alpha * signal
    return choices, rts, rewards


def exact_log_likelihood(theta, choices, rts, rewards):
    """Exact conditional log-likelihood: deterministic Q -> per-trial Navarro-Fuss density."""
    thresh, ndt = float(theta[2]), float(theta[3])
    drift = q_path(theta, choices, rewards)
    return float(wfpt_logpdf(rts, choices, drift, thresh, ndt=ndt).sum())


def generate_group_data(n_subjects, beta_z, sigma_z, num_trials, rng, block=40):
    """Draw subject parameters from the group and simulate each subject's bandit run."""
    tf = transform()
    beta_z = np.asarray(beta_z, float)
    sigma_z = np.asarray(sigma_z, float)
    z_true = rng.normal(beta_z, np.sqrt(sigma_z), size=(n_subjects, len(FIT_PARAMS)))
    theta_true = np.array([tf.to_natural(z_true[s]) for s in range(n_subjects)])
    schedule = reward_schedule(num_trials, rng, block=block)
    subjects = []
    for s in range(n_subjects):
        choices, rts, rewards = simulate_subject(theta_true[s], schedule, rng)
        subjects.append({"choices": choices, "rts": rts, "rewards": rewards})
    return {"subjects": subjects, "schedule": schedule,
            "z_true": z_true, "theta_true": theta_true}
