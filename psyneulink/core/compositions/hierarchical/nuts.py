# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  NUTS  ********************************************************************

"""The No-U-Turn sampler, and the hierarchical posterior it is run on.

`fit_laplace_em <laplaceem>` reports each participant's single best parameter values and summarizes
the uncertainty around them with a Gaussian placed at that peak.  The summary is only as good as
that assumption: a posterior that is skewed, or bounded, or has a ridge running through it is not a
Gaussian, and no amount of fitting makes the reported interval right.

Sampling makes no such assumption.  Intervals come out of the draws themselves, which costs many
more evaluations of the likelihood than EM -- tens of thousands rather than hundreds -- and needs
each one to be differentiable.  Both are why this path requires a trained estimator (see
`neurallikelihoodfunctions <NeuralLikelihood>`): an evaluation is one batched network call rather
than a fresh batch of simulations, and the network hands back a gradient.

The sampler is the multinomial No-U-Turn sampler: Hamiltonian Monte Carlo that chooses its own
trajectory length by doubling until the path starts to double back on itself, with dual-averaging
step-size adaptation and a diagonal mass matrix estimated during warmup.

The model is written in the non-centred form ``z_s = X_s beta + L w_s`` with ``w_s ~ N(0, I)``,
rather than ``z_s ~ N(X_s beta, L L')`` directly.  The two describe the same distribution, but the
second couples each participant's parameters to the group scale, which narrows the posterior into a
funnel that a sampler has to take very small steps to get into; the first does not.
"""

from dataclasses import dataclass, field
from enum import auto

import numpy as np

from psyneulink.core.compositions.hierarchical.laplaceem import Covariance
from psyneulink.core.globals.utilities import PNLStrEnum

try:
    import torch
except ImportError:
    torch = None

__all__ = [
    "HierarchicalNeuralPosterior",
    "NUTSConfig",
    "NUTSDiagnostics",
    "NUTSError",
    "Sampler",
    "SamplingWarning",
    "SubjectTerms",
    "effective_sample_size",
    "potential_scale_reduction",
    "run_nuts",
]


class Sampler(PNLStrEnum):
    """How a hierarchical fit explores the posterior, where it is not fitted by EM.

    Attributes
    ----------

    NUTS
        The multinomial No-U-Turn sampler; see `run_nuts`.
    """

    NUTS = auto()


#: Energy error above which a trajectory is treated as having gone wrong rather than as merely
#: being rejected.  Stan's value; the scale is arbitrary but any real error is far smaller.
MAX_ENERGY_ERROR = 1000.0

#: Dual-averaging constants, from Hoffman and Gelman (2014).
DUAL_AVERAGING_GAMMA = 0.05
DUAL_AVERAGING_T0 = 10.0
DUAL_AVERAGING_KAPPA = 0.75


class NUTSError(Exception):
    """Raised when a sampling run cannot be set up or cannot proceed."""


class SamplingWarning(UserWarning):
    """Raised when a sampling run finished but something about it warrants attention."""


def _require_torch():
    if torch is None:
        raise NUTSError(
            "sampling needs PyTorch, which is not installed. Install it with "
            'pip install "psyneulink[nle]".'
        )


@dataclass(frozen=True)
class NUTSConfig:
    """Settings for a sampling run.

    Attributes
    ----------

    draws, warmup : int
        Draws to keep per chain, and iterations to spend adapting before keeping any.  Warmup
        draws are discarded: they were produced while the step size and mass matrix were still
        changing, so they do not come from the target distribution.

    chains : int
        Independent chains, started from different points.  More than one is what makes
        `potential_scale_reduction` meaningful: a single chain cannot reveal that it has missed
        part of the posterior.

    target_accept : float
        Acceptance probability the step size is tuned towards.  Higher means smaller steps: more
        evaluations per draw, but fewer divergences on a difficult posterior.

    max_tree_depth : int
        Cap on trajectory doubling, so one draw costs at most ``2 ** max_tree_depth`` gradient
        evaluations.  Reaching it repeatedly means the sampler is being stopped before it would
        have turned, which is reported rather than silently accepted.

    seed : int
        Seeds both the starting points and the sampler's own randomness.

    beta_prior_sd, scale_prior_sd : float
        Standard deviations of the priors on the group means and on the log of the group scales,
        in unconstrained units.  The defaults are weak: the parameter space is already bounded by
        the model's search range, which the transform folds in.
    """

    draws: int = 1000
    warmup: int = 1000
    chains: int = 4
    target_accept: float = 0.8
    max_tree_depth: int = 10
    seed: int = 0
    beta_prior_sd: float = 5.0
    scale_prior_sd: float = 1.0

    def __post_init__(self):
        if self.draws < 1 or self.warmup < 1:
            raise ValueError("draws and warmup must both be at least 1")
        if self.chains < 1:
            raise ValueError("chains must be at least 1")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must lie strictly between 0 and 1")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be at least 1")
        if self.beta_prior_sd <= 0 or self.scale_prior_sd <= 0:
            raise ValueError("prior standard deviations must be positive")


@dataclass
class NUTSDiagnostics:
    """What a sampling run did, as distinct from what it found.

    A sampler that ran without complaint has not thereby been shown to be right, but one that
    complains has been shown to be wrong, so these are reported alongside every result.

    Attributes
    ----------

    divergences : numpy.ndarray
        Per chain, how many draws ended in a trajectory whose energy blew up.  Any at all means
        the sampler could not follow the posterior somewhere, and the draws are biased in a
        direction it cannot report.  Raising `NUTSConfig.target_accept` is the usual remedy.

    max_depth_hits : numpy.ndarray
        Per chain, how many draws were stopped by `NUTSConfig.max_tree_depth` rather than by the
        trajectory turning back.  This costs efficiency rather than correctness.

    accept_rate, step_size : numpy.ndarray
        Per chain, the mean acceptance statistic over the kept draws and the step size adaptation
        settled on.

    tree_depth : numpy.ndarray
        Per chain, the mean depth reached.
    """

    divergences: np.ndarray
    max_depth_hits: np.ndarray
    accept_rate: np.ndarray
    step_size: np.ndarray
    tree_depth: np.ndarray
    warnings: tuple = field(default_factory=tuple)

    def __repr__(self):
        note = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return (
            f"<NUTSDiagnostics: {int(self.divergences.sum())} divergence(s), "
            f"{int(self.max_depth_hits.sum())} at max depth, mean acceptance "
            f"{float(np.mean(self.accept_rate)):.2f}{note}>"
        )


# =====================================================================================
# The sampler
#
# Written against a callable returning the log density and its gradient, so it can be
# run on a distribution with a known answer as readily as on a hierarchical model.
# =====================================================================================
@dataclass
class _Point:
    """One end of a trajectory: position, momentum, log density and its gradient there."""

    q: "torch.Tensor"
    p: "torch.Tensor"
    log_prob: float
    grad: "torch.Tensor"

    @property
    def energy(self):
        return -self.log_prob + self._kinetic

    _kinetic: float = 0.0


def _leapfrog(log_prob_grad, point, step, inverse_mass):
    """One leapfrog step of size `step` from `point`.

    The momentum is advanced half a step, the position a whole step, and the momentum the
    remaining half, which makes the map volume-preserving and reversible -- the two properties
    the Metropolis correction below relies on.
    """
    p_half = point.p + 0.5 * step * point.grad
    q_new = point.q + step * inverse_mass * p_half
    log_prob, grad = log_prob_grad(q_new)
    p_new = p_half + 0.5 * step * grad
    kinetic = 0.5 * float(torch.sum(p_new * p_new * inverse_mass))
    return _Point(q_new, p_new, log_prob, grad, _kinetic=kinetic)


def _is_turning(q_minus, p_minus, q_plus, p_plus, inverse_mass):
    """Whether a trajectory spanning these two ends has begun to double back on itself.

    The span is measured in position; a trajectory is still making progress while it points the
    same way as the momentum at both ends.  Continuing past that point revisits ground already
    covered, which costs evaluations without improving the draw.
    """
    span = q_plus - q_minus
    forward = float(torch.sum(span * inverse_mass * p_plus))
    backward = float(torch.sum(span * inverse_mass * p_minus))
    return forward < 0.0 or backward < 0.0


@dataclass
class _Tree:
    """A trajectory segment, and the one draw chosen to represent it."""

    minus: _Point            # earliest end
    plus: _Point             # latest end
    sample: _Point           # the state this segment contributes as its draw
    log_weight: float        # log of the total Boltzmann weight in the segment
    accept_sum: float        # summed Metropolis ratio, for step-size adaptation
    n_leapfrog: int
    diverging: bool
    turning: bool


def _build_tree(log_prob_grad, point, direction, depth, step, inverse_mass, energy0, generator):
    """Double the trajectory `depth` more times in `direction`, returning the segment built.

    Multinomial rather than slice sampling: every state in the trajectory is a candidate, weighted
    by ``exp(-energy)``, which uses the whole trajectory rather than the part of it that clears a
    threshold.
    """
    if depth == 0:
        leaf = _leapfrog(log_prob_grad, point, direction * step, inverse_mass)
        energy = leaf.energy
        error = energy - energy0
        diverging = not np.isfinite(energy) or error > MAX_ENERGY_ERROR
        return _Tree(
            minus=leaf, plus=leaf, sample=leaf,
            log_weight=-np.inf if diverging else -float(energy),
            accept_sum=float(min(1.0, np.exp(-error))) if np.isfinite(error) else 0.0,
            n_leapfrog=1, diverging=diverging, turning=False,
        )

    near = _build_tree(log_prob_grad, point, direction, depth - 1, step,
                       inverse_mass, energy0, generator)
    if near.diverging or near.turning:
        return near

    far_start = near.plus if direction > 0 else near.minus
    far = _build_tree(log_prob_grad, far_start, direction, depth - 1, step,
                      inverse_mass, energy0, generator)

    minus, plus = (near.minus, far.plus) if direction > 0 else (far.minus, near.plus)
    combined = _log_add(near.log_weight, far.log_weight)
    # Progressive sampling: the further half is accepted with its share of the combined weight,
    # so the segment's draw is uniform over the whole of it without holding all of it in memory.
    sample = near.sample
    if far.log_weight > -np.inf and _bernoulli(np.exp(far.log_weight - combined), generator):
        sample = far.sample

    return _Tree(
        minus=minus, plus=plus, sample=sample, log_weight=combined,
        accept_sum=near.accept_sum + far.accept_sum,
        n_leapfrog=near.n_leapfrog + far.n_leapfrog,
        diverging=far.diverging,
        turning=far.turning or _is_turning(minus.q, minus.p, plus.q, plus.p, inverse_mass),
    )


def _log_add(a, b):
    """``log(exp(a) + exp(b))``, without overflowing on either."""
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + float(np.log1p(np.exp(lo - hi)))


def _bernoulli(probability, generator):
    if not np.isfinite(probability) or probability <= 0.0:
        return False
    return bool(torch.rand((), generator=generator).item() < probability)


def _nuts_transition(log_prob_grad, q, log_prob, grad, step, inverse_mass,
                     max_tree_depth, generator):
    """One draw: grow a trajectory around `q` until it turns, and pick a state from it."""
    momentum = torch.randn(q.shape, generator=generator, dtype=q.dtype) * inverse_mass.rsqrt()
    kinetic = 0.5 * float(torch.sum(momentum * momentum * inverse_mass))
    start = _Point(q, momentum, log_prob, grad, _kinetic=kinetic)
    energy0 = start.energy

    minus = plus = start
    sample = start
    log_weight = -float(energy0)
    accept_sum, n_leapfrog, depth = 0.0, 0, 0
    diverging = at_max_depth = False

    while True:
        if depth >= max_tree_depth:
            at_max_depth = True
            break
        direction = 1 if _bernoulli(0.5, generator) else -1
        segment = _build_tree(
            log_prob_grad, plus if direction > 0 else minus, direction, depth,
            step, inverse_mass, energy0, generator,
        )
        accept_sum += segment.accept_sum
        n_leapfrog += segment.n_leapfrog
        if segment.diverging:
            diverging = True
            break
        if direction > 0:
            plus = segment.plus
        else:
            minus = segment.minus
        if segment.turning:
            break

        # Biased progressive sampling: the new half is favoured over the old, which moves the
        # draw further from where the trajectory started and decorrelates it faster.
        if _bernoulli(np.exp(min(0.0, segment.log_weight - log_weight)), generator):
            sample = segment.sample
        log_weight = _log_add(log_weight, segment.log_weight)

        if _is_turning(minus.q, minus.p, plus.q, plus.p, inverse_mass):
            break
        depth += 1

    accept_stat = accept_sum / n_leapfrog if n_leapfrog else 0.0
    return sample, accept_stat, depth, diverging, at_max_depth


class _StepSizeAdapter:
    """Dual averaging on the log step size, targeting a given acceptance rate.

    Averaging rather than following the latest estimate is what makes the step size settle:
    the acceptance statistic of a single draw is noisy, and chasing it never converges.
    """

    def __init__(self, step, target_accept):
        self.restart(step)
        self._target = target_accept

    def restart(self, step):
        self._mu = float(np.log(10.0 * step))
        self._log_step_bar = 0.0
        self._error_bar = 0.0
        self._count = 0
        self.step = float(step)

    def update(self, accept_stat):
        self._count += 1
        weight = 1.0 / (self._count + DUAL_AVERAGING_T0)
        self._error_bar = ((1.0 - weight) * self._error_bar
                           + weight * (self._target - accept_stat))
        log_step = self._mu - np.sqrt(self._count) / DUAL_AVERAGING_GAMMA * self._error_bar
        decay = self._count ** -DUAL_AVERAGING_KAPPA
        self._log_step_bar = decay * log_step + (1.0 - decay) * self._log_step_bar
        self.step = float(np.exp(log_step))

    @property
    def settled(self):
        """The averaged step size, which is what the kept draws should be taken with."""
        return float(np.exp(self._log_step_bar))


def _initial_step_size(log_prob_grad, q, log_prob, grad, inverse_mass, generator):
    """Find a step size whose first leapfrog is neither wildly rejected nor absurdly small.

    Dual averaging converges from anywhere, but starting several orders of magnitude out wastes
    much of warmup getting back; doubling or halving once from a crude probe avoids that.
    """
    step = 1.0
    momentum = torch.randn(q.shape, generator=generator, dtype=q.dtype) * inverse_mass.rsqrt()
    kinetic = 0.5 * float(torch.sum(momentum * momentum * inverse_mass))
    start = _Point(q, momentum, log_prob, grad, _kinetic=kinetic)

    def error(step):
        moved = _leapfrog(log_prob_grad, start, step, inverse_mass)
        delta = start.energy - moved.energy
        return delta if np.isfinite(delta) else -np.inf

    direction = 1.0 if error(step) > np.log(0.8) else -1.0
    for _ in range(50):
        step *= 2.0 ** direction
        current = error(step)
        if direction > 0 and current <= np.log(0.8):
            break
        if direction < 0 and current >= np.log(0.8):
            break
    return step


def _run_chain(log_prob_grad, q0, config, generator):
    """One chain: adapt through warmup, then keep `config.draws` draws."""
    q = q0.clone()
    log_prob, grad = log_prob_grad(q)
    if not np.isfinite(log_prob):
        raise NUTSError(
            "the starting point has no posterior density, so the chain cannot begin; this "
            "usually means a participant's data are impossible under their model at the "
            "parameters the fit started from"
        )
    inverse_mass = torch.ones_like(q)

    adapter = _StepSizeAdapter(
        _initial_step_size(log_prob_grad, q, log_prob, grad, inverse_mass, generator),
        config.target_accept,
    )

    # Warmup in three parts, after Stan: settle the step size, then use the middle stretch to
    # estimate how far the posterior runs in each direction, then settle the step size again for
    # the mass matrix that came out of it.
    first = max(1, config.warmup // 4)
    last = max(1, config.warmup // 4)
    middle_start, middle_end = first, config.warmup - last
    collected = []

    draws = np.empty((config.draws, q.numel()))
    accept = np.empty(config.draws)
    depths = np.empty(config.draws, dtype=int)
    divergences = max_depth_hits = 0

    for iteration in range(config.warmup + config.draws):
        warming = iteration < config.warmup
        sample, accept_stat, depth, diverging, at_max_depth = _nuts_transition(
            log_prob_grad, q, log_prob, grad, adapter.step if warming else adapter.settled,
            inverse_mass, config.max_tree_depth, generator,
        )
        q, log_prob, grad = sample.q, sample.log_prob, sample.grad

        if warming:
            adapter.update(accept_stat)
            if middle_start <= iteration < middle_end:
                collected.append(q.clone())
            if iteration == middle_end - 1 and len(collected) > 1:
                stacked = torch.stack(collected)
                # A diagonal mass matrix set to the posterior variance makes each direction take
                # steps of comparable size, which is what lets one step size serve all of them.
                inverse_mass = stacked.var(dim=0).clamp(min=1e-10)
                # The step size that suited the old metric need not suit the new one, so it is
                # probed again rather than carried over, and dual averaging starts from there.
                adapter.restart(_initial_step_size(
                    log_prob_grad, q, log_prob, grad, inverse_mass, generator
                ))
        else:
            kept = iteration - config.warmup
            draws[kept] = q.detach().numpy()
            accept[kept] = accept_stat
            depths[kept] = depth
            divergences += int(diverging)
            max_depth_hits += int(at_max_depth)

    return draws, accept, depths, divergences, max_depth_hits, adapter.settled


def run_nuts(log_prob_grad, initial_points, config=None):
    """Sample a differentiable log density.

    Arguments
    ---------

    log_prob_grad : callable
        ``log_prob_grad(q) -> (float, tensor)``, the log density at `q` and its gradient there.
        `q` is a one-dimensional `torch.Tensor`; the gradient must have the same shape.  Any
        constraint on the parameters has to be folded into `q` beforehand, since the sampler moves
        `q` freely.

    initial_points : sequence of array-like
        One starting point per chain.  They should differ: chains started together cannot show
        that they would have agreed from anywhere else.

    config : NUTSConfig : default None
        Settings; a default-constructed `NUTSConfig` if omitted.

    Returns
    -------

    ``(draws, diagnostics)``, where `draws` is ``(chains, draws, n_params)`` and `diagnostics` is
    a `NUTSDiagnostics`.
    """
    _require_torch()
    config = config if config is not None else NUTSConfig()
    starts = [torch.as_tensor(np.asarray(p, dtype=float), dtype=torch.float64)
              for p in initial_points]
    if len(starts) != config.chains:
        raise NUTSError(
            f"config asks for {config.chains} chain(s) but {len(starts)} starting point(s) "
            f"were given"
        )

    chains, accepts, depths, steps = [], [], [], []
    divergences, max_depth_hits = [], []
    for index, start in enumerate(starts):
        generator = torch.Generator().manual_seed(config.seed + index)
        drawn, accept, depth, diverged, at_max, step = _run_chain(
            log_prob_grad, start, config, generator
        )
        chains.append(drawn)
        accepts.append(float(np.mean(accept)))
        depths.append(float(np.mean(depth)))
        steps.append(step)
        divergences.append(diverged)
        max_depth_hits.append(at_max)

    draws = np.stack(chains)
    messages = []
    if sum(divergences):
        messages.append(
            f"{sum(divergences)} of {config.chains * config.draws} draws ended in a divergent "
            f"trajectory. The sampler could not follow the posterior somewhere, so the draws "
            f"are biased. Raising target_accept above {config.target_accept} usually helps."
        )
    if sum(max_depth_hits):
        messages.append(
            f"{sum(max_depth_hits)} draws reached max_tree_depth={config.max_tree_depth} "
            f"without the trajectory turning, which costs efficiency rather than correctness."
        )

    diagnostics = NUTSDiagnostics(
        divergences=np.array(divergences),
        max_depth_hits=np.array(max_depth_hits),
        accept_rate=np.array(accepts),
        step_size=np.array(steps),
        tree_depth=np.array(depths),
        warnings=tuple(messages),
    )
    return draws, diagnostics


# =====================================================================================
# Convergence diagnostics
#
# Both are computed on split chains: each chain is halved and the halves treated as
# separate, so that a chain which drifted through its own run is caught even when the
# chains agree with each other.
# =====================================================================================
def _split(draws):
    """``(chains, draws, params)`` into ``(2 * chains, draws // 2, params)``."""
    n_chains, n_draws, n_params = draws.shape
    half = n_draws // 2
    if half < 2:
        raise ValueError("splitting needs at least four draws per chain")
    return np.concatenate([draws[:, :half], draws[:, half:2 * half]], axis=0)


def potential_scale_reduction(draws):
    """Split R-hat: how much the chains still disagree, per parameter.

    The ratio of the spread of all the draws pooled together to the spread within each chain.
    Chains exploring the same distribution give a ratio near 1; a larger one means at least one
    chain is somewhere the others are not, and the draws do not yet describe the posterior.
    Values above about 1.01 are usually taken as not converged.

    Arguments
    ---------

    draws : array-like
        ``(chains, draws, n_params)``.

    Returns
    -------

    ``(n_params,)`` of R-hat values.
    """
    split = _split(np.asarray(draws, dtype=float))
    n_chains, n_draws, _ = split.shape
    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)

    within = chain_vars.mean(axis=0)
    between = n_draws * chain_means.var(axis=0, ddof=1) if n_chains > 1 else 0.0
    pooled = ((n_draws - 1) / n_draws) * within + between / n_draws
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(within > 0, np.sqrt(pooled / within), 1.0)


def effective_sample_size(draws):
    """How many independent draws the chains are worth, per parameter.

    Successive draws from a sampler are correlated, so a run of 1000 carries less information
    than 1000 independent ones.  This is the equivalent independent count, computed from the
    autocorrelation summed until consecutive pairs stop being positive (Geyer's initial positive
    sequence).  It is what a reported interval's precision actually rests on.

    Arguments
    ---------

    draws : array-like
        ``(chains, draws, n_params)``.

    Returns
    -------

    ``(n_params,)`` of effective sample sizes.
    """
    split = _split(np.asarray(draws, dtype=float))
    n_chains, n_draws, n_params = split.shape
    total = n_chains * n_draws

    chain_vars = split.var(axis=1, ddof=1)
    within = chain_vars.mean(axis=0)
    chain_means = split.mean(axis=1)
    between = n_draws * chain_means.var(axis=0, ddof=1) if n_chains > 1 else 0.0
    pooled = ((n_draws - 1) / n_draws) * within + between / n_draws

    result = np.full(n_params, float(total))
    for k in range(n_params):
        if not np.isfinite(pooled[k]) or pooled[k] <= 0:
            continue
        # Autocorrelation of each chain by FFT, averaged over chains.
        centred = split[:, :, k] - chain_means[:, k][:, None]
        size = int(2 ** np.ceil(np.log2(2 * n_draws)))
        spectrum = np.fft.rfft(centred, n=size, axis=1)
        acov = np.fft.irfft(spectrum * np.conjugate(spectrum), n=size, axis=1)[:, :n_draws]
        acov /= n_draws
        rho = 1.0 - (within[k] - acov.mean(axis=0)) / pooled[k]

        # Geyer: sum consecutive pairs while they stay positive, which keeps the estimate from
        # accumulating the noise in the long-lag autocorrelations.
        total_rho = 0.0
        for lag in range(1, n_draws - 2, 2):
            pair = rho[lag] + rho[lag + 1]
            if pair < 0:
                break
            total_rho += pair
        tau = max(1.0 + 2.0 * total_rho, 1.0)
        result[k] = total / tau
    return result


# =====================================================================================
# The hierarchical posterior
# =====================================================================================
def _cholesky_from_unconstrained(log_scale, off_diagonal, n_params):
    """Lower-triangular factor of the group covariance, from unconstrained parameters.

    The factor is sampled rather than the covariance itself, which keeps every point the sampler
    can reach a valid covariance: ``L L'`` is positive definite for any `L` with a positive
    diagonal, and the diagonal is positive because it is an exponential.
    """
    factor = torch.diag(torch.exp(log_scale))
    if off_diagonal is not None and off_diagonal.numel():
        rows, cols = torch.tril_indices(n_params, n_params, offset=-1)
        factor = factor.clone()
        factor[rows, cols] = off_diagonal
    return factor


@dataclass(frozen=True)
class SubjectTerms:
    """One participant's trained estimator and the trials it scores.

    Attributes
    ----------

    likelihood : NeuralLikelihood
        The estimator this participant's model is scored by.

    outcomes : torch.Tensor
        Their observed trials, already in the estimator's layout.

    trial_features : torch.Tensor or None
        Per-trial conditioning, when the estimator was trained with any.
    """

    likelihood: object
    outcomes: "torch.Tensor"
    trial_features: object = None

    @property
    def n_trials(self):
        return int(self.outcomes.shape[0])


class HierarchicalNeuralPosterior:
    """The joint posterior over the group and every participant at once.

    Unlike EM, which alternates between the two, a sampler moves in all of them together: the
    group estimate and every participant's parameters are one point, and the uncertainty in each
    is reported having accounted for the uncertainty in the rest.

    The parameters the sampler moves are, in order: `beta`, the group means, one row per
    predictor; the logs of the group scales; the below-diagonal entries of the covariance factor,
    when the covariance is full; and one standard normal vector per participant.  Every one of
    them is unbounded, which is what the sampler requires.

    Arguments
    ---------

    terms : sequence of SubjectTerms
        One per participant, in participant order.

    lower, upper : array-like
        Search range per parameter, the same one the group model's transform is built from.

    design_matrix : array-like : default None
        ``(n_subjects, n_predictors)``; an intercept when omitted.

    covariance : "diagonal" or "full"
        Whether the group covariance may express correlations between parameters.

    config : NUTSConfig : default None
        Read for the prior widths.
    """

    def __init__(self, terms, lower, upper, design_matrix=None, covariance="diagonal",
                 config=None):
        _require_torch()
        if covariance not in Covariance:
            raise ValueError(
                f"covariance must be one of {[c.value for c in Covariance]}; got "
                f"{covariance!r}"
            )
        self.terms = list(terms)
        if not self.terms:
            raise NUTSError("sampling needs at least one participant")
        self.config = config if config is not None else NUTSConfig()
        self.covariance = covariance

        self.lower = torch.as_tensor(np.asarray(lower, dtype=float), dtype=torch.float64)
        self.width = torch.as_tensor(np.asarray(upper, dtype=float), dtype=torch.float64)
        self.width = self.width - self.lower
        self.n_params = int(self.lower.numel())
        self.n_subjects = len(self.terms)

        design = (np.ones((self.n_subjects, 1)) if design_matrix is None
                  else np.asarray(design_matrix, dtype=float))
        if design.shape[0] != self.n_subjects:
            raise ValueError(
                f"design_matrix must have one row per participant; got {design.shape[0]} rows "
                f"for {self.n_subjects} participants"
            )
        self.design = torch.as_tensor(design, dtype=torch.float64)
        self.n_predictors = int(self.design.shape[1])

        self.n_off_diagonal = (
            self.n_params * (self.n_params - 1) // 2 if covariance == "full" else 0
        )
        self._beta_end = self.n_predictors * self.n_params
        self._scale_end = self._beta_end + self.n_params
        self._off_end = self._scale_end + self.n_off_diagonal
        self.size = self._off_end + self.n_subjects * self.n_params

        # Participants whose models share one estimator object are scored together, which is one
        # network call instead of one per participant. A factory that loads the artifact once and
        # hands the same object to every participant gets this; one that loads it again for each
        # does not, and is scored participant by participant instead.
        self._batches = self._group_by_estimator()

    def _group_by_estimator(self):
        batches = {}
        for index, term in enumerate(self.terms):
            batches.setdefault(id(term.likelihood), []).append(index)
        return list(batches.values())

    def unpack(self, q):
        """Split one sampler position into the quantities it stands for."""
        beta = q[:self._beta_end].reshape(self.n_predictors, self.n_params)
        log_scale = q[self._beta_end:self._scale_end]
        off_diagonal = q[self._scale_end:self._off_end] if self.n_off_diagonal else None
        raw = q[self._off_end:].reshape(self.n_subjects, self.n_params)
        return beta, log_scale, off_diagonal, raw

    def subject_z(self, q):
        """Every participant's parameters in the unconstrained space, from one position."""
        beta, log_scale, off_diagonal, raw = self.unpack(q)
        factor = _cholesky_from_unconstrained(log_scale, off_diagonal, self.n_params)
        # z_s = X_s beta + L w_s, the non-centred form; see the module docstring.
        return self.design @ beta + raw @ factor.T

    def group_covariance(self, q):
        """The group covariance implied by one position."""
        _, log_scale, off_diagonal, _ = self.unpack(q)
        factor = _cholesky_from_unconstrained(log_scale, off_diagonal, self.n_params)
        return factor @ factor.T

    def to_natural(self, z):
        """Unconstrained parameters into the model's own units."""
        return self.lower + self.width * torch.sigmoid(z)

    def log_prob(self, q):
        """Log posterior density at `q`, differentiable with respect to it."""
        beta, log_scale, off_diagonal, raw = self.unpack(q)
        theta = self.to_natural(self.subject_z(q))

        total = torch.zeros((), dtype=torch.float64)
        for batch in self._batches:
            likelihood = self.terms[batch[0]].likelihood
            rows = torch.cat([
                theta[index].expand(self.terms[index].n_trials, self.n_params)
                for index in batch
            ])
            outcomes = torch.cat([self.terms[index].outcomes for index in batch])
            features = None
            if self.terms[batch[0]].trial_features is not None:
                features = torch.cat([self.terms[index].trial_features for index in batch])
            scored = likelihood.trial_log_prob(rows, outcomes, features, encoded=True)
            total = total + scored.to(torch.float64).sum()

        # The prior on each participant is standard normal by construction: the spread the group
        # model gives them is carried by the factor multiplying it, not by this term.
        total = total - 0.5 * (raw ** 2).sum()
        total = total - 0.5 * (beta / self.config.beta_prior_sd).pow(2).sum()
        total = total - 0.5 * (log_scale / self.config.scale_prior_sd).pow(2).sum()
        if off_diagonal is not None:
            total = total - 0.5 * (off_diagonal ** 2).sum()
        return total

    def log_prob_grad(self, q):
        """`log_prob` and its gradient, in the form `run_nuts` expects."""
        q = q.detach().requires_grad_(True)
        value = self.log_prob(q)
        gradient, = torch.autograd.grad(value, q)
        return float(value.detach()), gradient.detach()

    def initial_points(self, chains, seed=0):
        """Starting positions, spread out so the chains can be seen to agree.

        Group means start at the middle of the search range, which the transform puts at zero,
        and participants start at the group.  The jitter is what makes the chains independent.
        """
        generator = np.random.default_rng(seed)
        points = []
        for _ in range(chains):
            start = np.zeros(self.size)
            start[self._beta_end:self._scale_end] = np.log(0.5)
            points.append(start + generator.normal(scale=0.25, size=self.size))
        return points
