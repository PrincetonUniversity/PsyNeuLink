# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ******************************************  NUTS  ********************************************************************

"""Sampling the hierarchical posterior with the No-U-Turn sampler.

`fit_laplace_em <laplaceem>` reports each participant's single best parameter values and summarizes
the uncertainty around them with a Gaussian placed at that peak.  The summary is only as good as
that assumption: a posterior that is skewed, or bounded, or has a ridge running through it is not a
Gaussian, and no amount of fitting makes the reported interval right.

Sampling makes no such assumption.  Intervals come out of the draws themselves, which costs many
more evaluations of the likelihood than EM -- tens of thousands rather than hundreds -- and needs
each one to be differentiable.  Both are why this path requires a trained estimator (see
`neurallikelihoodfunctions <NeuralLikelihood>`): an evaluation is one batched network call rather
than a fresh batch of simulations, and the network hands back a gradient.

The sampler is `Pyro <https://pyro.ai>`_'s No-U-Turn sampler, run on the posterior's log density.

The model is written in the non-centred form ``z_s = X_s beta + sigma * w_s`` with
``w_s ~ N(0, I)``, rather than ``z_s ~ N(X_s beta, diag(sigma^2))`` directly.  The two describe the
same distribution, but the second couples each participant's parameters to the group scale, which
narrows the posterior into a funnel that a sampler has to take very small steps to get into; the
first does not.
"""

from dataclasses import dataclass, field
from enum import auto

import numpy as np

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
    "run_nuts",
]


class Sampler(PNLStrEnum):
    """How a hierarchical fit explores the posterior, where it is not fitted by EM.

    Attributes
    ----------

    NUTS
        The No-U-Turn sampler; see `run_nuts`.
    """

    NUTS = auto()


#: Standard deviations of the priors on the group means and on the logs of the group scales, in
#: unconstrained units.  They are weak: the model's search range already bounds every parameter,
#: and the transform folds it in.
BETA_PRIOR_SD = 5.0
LOG_SCALE_PRIOR_SD = 1.0


class NUTSError(Exception):
    """Raised when a sampling run cannot be set up or cannot proceed."""


class SamplingWarning(UserWarning):
    """Raised when a sampling run finished but something about it warrants attention."""


def _require_pyro():
    try:
        import pyro  # noqa: F401
    except ImportError as e:
        raise NUTSError(
            "sampling needs Pyro, which is not installed. Install it with "
            'pip install "psyneulink[nle]".'
        ) from e


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
        Independent chains, started from different points.  More than one is what makes the
        ``r_hat`` of a result meaningful: a single chain cannot reveal that it has missed part of
        the posterior.

    target_accept : float
        Acceptance probability the step size is tuned towards.  Higher means smaller steps: more
        evaluations per draw, but fewer divergences on a difficult posterior.

    max_tree_depth : int
        Cap on trajectory doubling, so one draw costs at most ``2 ** max_tree_depth`` gradient
        evaluations.

    seed : int
        Seeds both the starting points and the sampler's own randomness.
    """

    draws: int = 1000
    warmup: int = 1000
    chains: int = 4
    target_accept: float = 0.8
    max_tree_depth: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.draws < 1 or self.warmup < 1:
            raise ValueError("draws and warmup must both be at least 1")
        if self.chains < 1:
            raise ValueError("chains must be at least 1")
        if not 0.0 < self.target_accept < 1.0:
            raise ValueError("target_accept must lie strictly between 0 and 1")
        if self.max_tree_depth < 1:
            raise ValueError("max_tree_depth must be at least 1")


@dataclass
class NUTSDiagnostics:
    """What a sampling run did, as distinct from what it found.

    A sampler that ran without complaint has not thereby been shown to be right, but one that
    complains has been shown to be wrong, so these are reported alongside every result.

    Attributes
    ----------

    divergences : numpy.ndarray
        Per chain, how many kept draws ended in a trajectory whose energy blew up.  Any at all
        means the sampler could not follow the posterior somewhere, and the draws are biased in a
        direction it cannot report.  Raising `NUTSConfig.target_accept` is the usual remedy.

    accept_rate, step_size : numpy.ndarray
        Per chain, the mean acceptance probability over the kept draws, and the step size warmup
        settled on.

    warnings : tuple
        What the sampler reported, also issued as a `SamplingWarning`.
    """

    divergences: np.ndarray
    accept_rate: np.ndarray
    step_size: np.ndarray
    warnings: tuple = field(default_factory=tuple)

    def __repr__(self):
        note = f", {len(self.warnings)} warning(s)" if self.warnings else ""
        return (
            f"<NUTSDiagnostics: {int(self.divergences.sum())} divergence(s), mean acceptance "
            f"{float(np.mean(self.accept_rate)):.2f}{note}>"
        )


def run_nuts(log_prob, initial_points, config=None):
    """Sample a differentiable log density with Pyro's No-U-Turn sampler.

    Warmup adapts the step size and a diagonal mass matrix.  Chains are run one after another, in
    this process.

    Arguments
    ---------

    log_prob : callable
        ``log_prob(q) -> torch scalar``, the log density at `q`, differentiable with respect to
        it.  `q` is a one-dimensional `torch.Tensor`.  Any constraint on the parameters has to be
        folded into `q` beforehand, since the sampler moves `q` freely.

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
    _require_pyro()
    from pyro.infer import MCMC, NUTS

    config = config if config is not None else NUTSConfig()
    starts = [torch.as_tensor(np.asarray(p, dtype=float), dtype=torch.float64)
              for p in initial_points]
    if len(starts) != config.chains:
        raise NUTSError(
            f"config asks for {config.chains} chain(s) but {len(starts)} starting point(s) "
            f"were given"
        )

    chains, divergences, accepts, steps = [], [], [], []
    for index, start in enumerate(starts):
        kernel = NUTS(potential_fn=lambda params: -log_prob(params["q"]),
                      target_accept_prob=config.target_accept,
                      max_tree_depth=config.max_tree_depth)
        mcmc = MCMC(kernel, num_samples=config.draws, warmup_steps=config.warmup,
                    initial_params={"q": start}, disable_progbar=True)
        # Each chain seeded from config.seed, without changing the caller's random state.
        with torch.random.fork_rng():
            torch.manual_seed(config.seed + index)
            mcmc.run()
        chains.append(mcmc.get_samples()["q"].detach().numpy())
        reported = mcmc.diagnostics()
        divergences.append(len(reported["divergences"]["chain 0"]))
        accepts.append(float(reported["acceptance rate"]["chain 0"]))
        steps.append(float(kernel.step_size))

    messages = []
    if sum(divergences):
        messages.append(
            f"{sum(divergences)} of {config.chains * config.draws} draws ended in a divergent "
            f"trajectory. The sampler could not follow the posterior somewhere, so the draws "
            f"are biased. Raising target_accept above {config.target_accept} usually helps."
        )
    diagnostics = NUTSDiagnostics(
        divergences=np.array(divergences),
        accept_rate=np.array(accepts),
        step_size=np.array(steps),
        warnings=tuple(messages),
    )
    return np.stack(chains), diagnostics


@dataclass(frozen=True)
class SubjectTerms:
    """One participant's trained estimator and the trials it scores.

    Attributes
    ----------

    likelihood : NeuralLikelihood
        The estimator this participant's model is scored by.

    outcomes : torch.Tensor
        Their observed trials, already in the estimator's layout.

    parameter_index : torch.Tensor
        ``parameter_index[t, k]`` is the position, among the values fitted to this participant, of
        the value the model's k-th parameter takes on trial t.

    trial_features : torch.Tensor or None
        The trials' features, when the estimator was trained with any.
    """

    likelihood: object
    outcomes: "torch.Tensor"
    parameter_index: "torch.Tensor"
    trial_features: object = None


class HierarchicalNeuralPosterior:
    """The joint posterior over the group and every participant at once.

    Unlike EM, which alternates between the two, a sampler moves in all of them together: the
    group estimate and every participant's parameters are one point, and the uncertainty in each
    is reported having accounted for the uncertainty in the rest.

    The parameters the sampler moves are, in order: `beta`, the group means, one row per
    predictor; the logs of the group scales; and one standard normal vector per participant.
    Every one of them is unbounded, which is what the sampler requires.

    Arguments
    ---------

    terms : sequence of SubjectTerms
        One per participant, in participant order.

    lower, upper : array-like
        Search range per parameter, the same one the group model's transform is built from.

    design_matrix : array-like : default None
        ``(n_subjects, n_predictors)``; an intercept when omitted.
    """

    def __init__(self, terms, lower, upper, design_matrix=None):
        if torch is None:
            raise NUTSError(
                "sampling needs PyTorch, which is not installed. Install it with "
                'pip install "psyneulink[nle]".'
            )
        self.terms = list(terms)
        if not self.terms:
            raise NUTSError("sampling needs at least one participant")

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

        self._beta_end = self.n_predictors * self.n_params
        self._scale_end = self._beta_end + self.n_params
        self.size = self._scale_end + self.n_subjects * self.n_params

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
        raw = q[self._scale_end:].reshape(self.n_subjects, self.n_params)
        return beta, log_scale, raw

    def subject_z(self, q):
        """Every participant's parameters in the unconstrained space, from one position."""
        beta, log_scale, raw = self.unpack(q)
        # z_s = X_s beta + sigma * w_s, the non-centred form; see the module docstring.
        return self.design @ beta + raw * torch.exp(log_scale)

    def group_scale(self, q):
        """The group's standard deviation of each parameter, unconstrained, at one position."""
        return torch.exp(self.unpack(q)[1])

    def to_natural(self, z):
        """Unconstrained parameters into the model's own units."""
        return self.lower + self.width * torch.sigmoid(z)

    def log_prob(self, q):
        """Log posterior density at `q`, differentiable with respect to it."""
        beta, log_scale, raw = self.unpack(q)
        theta = self.to_natural(self.subject_z(q))

        total = torch.zeros((), dtype=torch.float64)
        for batch in self._batches:
            likelihood = self.terms[batch[0]].likelihood
            rows = torch.cat([theta[index][self.terms[index].parameter_index] for index in batch])
            outcomes = torch.cat([self.terms[index].outcomes for index in batch])
            features = None
            if self.terms[batch[0]].trial_features is not None:
                features = torch.cat([self.terms[index].trial_features for index in batch])
            scored = likelihood.trial_log_prob(rows, outcomes, features, encoded=True)
            total = total + scored.to(torch.float64).sum()

        # The prior on each participant is standard normal by construction: the spread the group
        # model gives them is carried by the scale multiplying it, not by this term.
        total = total - 0.5 * (raw ** 2).sum()
        total = total - 0.5 * (beta / BETA_PRIOR_SD).pow(2).sum()
        total = total - 0.5 * (log_scale / LOG_SCALE_PRIOR_SD).pow(2).sum()
        return total

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
