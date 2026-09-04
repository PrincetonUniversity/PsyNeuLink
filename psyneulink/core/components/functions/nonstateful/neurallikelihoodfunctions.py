"""Neural likelihood estimation for `ParameterEstimationComposition`.

`ParameterEstimationComposition` normally scores a candidate parameter setting by
simulating the composition and turning the simulated outcomes into a density by kernel
density estimation.  Every evaluation therefore costs ``num_estimates`` simulations, and
the surface it produces carries the Monte Carlo noise of those simulations.

A neural likelihood replaces that pipeline with a conditional density model
``p(outcomes | parameters)``, trained once on simulated data.  Fitting afterwards costs a
network forward pass instead of a simulation batch, and the surface is smooth and
differentiable rather than noisy.

The trade is a training step, and a network that is only valid over the parameter region
it was trained on.  A `NeuralLikelihood` therefore records what it was trained for and
refuses to score a model it does not match.

.. _NeuralLikelihood_Conditioning:

The conditioning vector is ``[parameters, trial features]``.  Trial features are the
values entering the composition's input nodes on that trial, so a model whose trials
differ -- congruent against incongruent, switch against repeat -- is conditioned on which
trial it is scoring rather than on the parameters alone.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, asdict

import numpy as np

# Optional, as elsewhere in PsyNeuLink: the package is importable without it.
try:
    import torch
except ImportError:
    torch = None

__all__ = [
    "NeuralLikelihood",
    "NeuralLikelihoodError",
    "NeuralLikelihoodWarning",
    "train_neural_likelihood",
]


class NeuralLikelihoodError(Exception):
    """Raised when a neural likelihood is misconfigured or does not match its model."""


class NeuralLikelihoodWarning(UserWarning):
    """Warns that a trained estimator did not pass one of its validation gates."""


def _require_sbi():
    """Import sbi, or explain how to install it."""
    try:
        import sbi  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "A neural likelihood requires the sbi package, which is not installed. "
            "Install it with `pip install \"psyneulink[nle]\"`."
        ) from e
    return sbi


@dataclass(frozen=True)
class NeuralLikelihoodProvenance:
    """What a trained estimator is valid for.

    Loading an estimator trained for one model against a different one produces
    plausible-looking numbers rather than an error, so every field here is checked
    before an estimator is used.
    """

    fit_param_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    outcome_names: tuple[str, ...]
    categorical: tuple[bool, ...]
    categories: tuple[tuple[float, ...], ...]
    log_transform: bool
    n_trial_features: int
    n_parameter_samples: int
    n_trials_per_sample: int
    epochs: int
    val_nll: float
    seed: int
    simulator_hash: str
    psyneulink_version: str
    sbi_version: str

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> NeuralLikelihoodProvenance:
        raw = json.loads(text)
        return cls(
            fit_param_names=tuple(raw["fit_param_names"]),
            lower=tuple(raw["lower"]),
            upper=tuple(raw["upper"]),
            outcome_names=tuple(raw["outcome_names"]),
            categorical=tuple(raw["categorical"]),
            categories=tuple(tuple(c) for c in raw["categories"]),
            log_transform=raw["log_transform"],
            n_trial_features=raw["n_trial_features"],
            n_parameter_samples=raw["n_parameter_samples"],
            n_trials_per_sample=raw["n_trials_per_sample"],
            epochs=raw["epochs"],
            val_nll=raw["val_nll"],
            seed=raw["seed"],
            simulator_hash=raw["simulator_hash"],
            psyneulink_version=raw["psyneulink_version"],
            sbi_version=raw["sbi_version"],
        )

    def check_matches(self, names, lower, upper, outcome_names, categorical):
        """Raise unless this estimator was trained for the model described.

        Bounds must be contained by the trained box: evaluating outside it is
        extrapolation, whose error is unbounded and silent.
        """
        if tuple(names) != self.fit_param_names:
            raise NeuralLikelihoodError(
                f"This neural likelihood was trained for parameters "
                f"{list(self.fit_param_names)}, but is being used to fit {list(names)}. "
                f"Order matters: the conditioning vector is positional."
            )
        if tuple(outcome_names) != self.outcome_names:
            raise NeuralLikelihoodError(
                f"This neural likelihood was trained for outcome variables "
                f"{list(self.outcome_names)}, but is being used with {list(outcome_names)}."
            )
        if tuple(bool(c) for c in categorical) != self.categorical:
            raise NeuralLikelihoodError(
                f"This neural likelihood was trained with categorical outcomes "
                f"{list(self.categorical)}, but is being used with "
                f"{[bool(c) for c in categorical]}."
            )
        for name, lo, hi, tlo, thi in zip(names, lower, upper, self.lower, self.upper):
            if lo < tlo - 1e-9 or hi > thi + 1e-9:
                raise NeuralLikelihoodError(
                    f"Parameter {name!r} is being fit over [{lo}, {hi}], which reaches "
                    f"outside the range this neural likelihood was trained on "
                    f"[{tlo}, {thi}]. Retrain over the wider range, or narrow the fit."
                )


class NeuralLikelihood:
    """A trained conditional density ``p(outcomes | parameters, trial features)``.

    Built by `train_neural_likelihood`, saved with `save`, and reloaded with `load`.
    Scoring is differentiable, so the same object serves gradient-free optimization, an
    exact-curvature E-step, and gradient-based sampling.
    """

    def __init__(self, estimator, provenance: NeuralLikelihoodProvenance, shape_probe):
        self._estimator = estimator
        self.provenance = provenance
        # A small batch of (outcomes, conditioning) rows, kept so `load` can rebuild the
        # architecture before restoring weights: sbi infers shapes from example data.
        self._shape_probe = shape_probe

    @property
    def fit_param_names(self) -> tuple[str, ...]:
        return self.provenance.fit_param_names

    # -- column order ---------------------------------------------------------------
    #
    # sbi's mixed estimator requires continuous columns first and categorical columns
    # last.  PEC's `outcome_variables` order is the user's, and commonly puts the
    # categorical column first, so outcomes are permuted on the way in.

    def _encode_outcomes(self, outcomes: np.ndarray) -> torch.Tensor:
        """Reorder to sbi's layout and map categorical values onto codes 0..K-1."""
        return _encode_outcomes(
            outcomes,
            self.provenance.categorical,
            self.provenance.categories,
            self.provenance.outcome_names,
        )

    def _conditioning(self, theta: torch.Tensor, trial_features, n_trials) -> torch.Tensor:
        """Broadcast one parameter vector across trials and append per-trial features."""
        cond = theta.reshape(1, -1).expand(n_trials, -1)
        if self.provenance.n_trial_features:
            if trial_features is None:
                raise NeuralLikelihoodError(
                    f"This neural likelihood was trained with "
                    f"{self.provenance.n_trial_features} per-trial feature(s), so "
                    f"scoring requires trial_features."
                )
            feats = torch.as_tensor(np.asarray(trial_features, dtype=float), dtype=torch.float32)
            if feats.shape != (n_trials, self.provenance.n_trial_features):
                raise NeuralLikelihoodError(
                    f"Expected trial_features of shape "
                    f"({n_trials}, {self.provenance.n_trial_features}), got "
                    f"{tuple(feats.shape)}."
                )
            cond = torch.cat([cond, feats], dim=-1)
        return cond

    def trial_log_prob(self, theta, outcomes, trial_features=None) -> torch.Tensor:
        """Per-trial log densities, differentiable with respect to ``theta``."""
        theta_t = (
            theta
            if isinstance(theta, torch.Tensor)
            else torch.as_tensor(np.asarray(theta, dtype=float), dtype=torch.float32)
        )
        x = self._encode_outcomes(outcomes)
        cond = self._conditioning(theta_t.to(torch.float32), trial_features, x.shape[0])
        return self._estimator.log_prob(x, condition=cond).reshape(-1)

    def log_likelihood(self, theta, outcomes, trial_features=None) -> float:
        """Total log-likelihood of ``outcomes`` under ``theta``."""
        with torch.no_grad():
            return float(self.trial_log_prob(theta, outcomes, trial_features).sum())

    # -- persistence ----------------------------------------------------------------

    def save(self, path):
        """Write weights and provenance to ``path``."""
        torch.save(
            {
                "state_dict": self._estimator.state_dict(),
                "provenance": self.provenance.to_json(),
                "probe_x": self._shape_probe[0],
                "probe_cond": self._shape_probe[1],
            },
            path,
        )

    @classmethod
    def load(cls, path) -> NeuralLikelihood:
        """Read back an estimator written by `save`."""
        _require_sbi()
        blob = torch.load(path, weights_only=True)
        provenance = NeuralLikelihoodProvenance.from_json(blob["provenance"])
        estimator = _build_estimator(
            blob["probe_x"],
            blob["probe_cond"],
            provenance.categorical,
            provenance.categories,
            provenance.log_transform,
        )
        estimator.load_state_dict(blob["state_dict"])
        estimator.eval()
        return cls(estimator, provenance, (blob["probe_x"], blob["probe_cond"]))


def _build_estimator(x, cond, categorical, categories, log_transform):
    """Construct an untrained sbi estimator sized from example data.

    A mixed estimator is used when any outcome is categorical, and a plain flow
    otherwise.  ``x`` must already be in sbi's layout: continuous columns first.
    """
    _require_sbi()
    n_cat = int(sum(bool(c) for c in categorical))
    if n_cat:
        from sbi.neural_nets.net_builders.mixed_nets import build_mnle

        counts = [len(c) for c, is_cat in zip(categories, categorical) if is_cat]
        with warnings.catch_warnings():
            # sbi warns that categorical columns must come last; they do, by construction.
            warnings.simplefilter("ignore")
            return build_mnle(
                batch_x=x,
                batch_y=cond,
                log_transform_x=log_transform,
                num_categories_per_variable=torch.tensor(counts),
            )
    from sbi.neural_nets import likelihood_nn

    # sbi's plain flows take no log-transform option, and applying one here would need
    # its Jacobian carried through scoring; all-continuous outcomes are modelled in
    # their natural units instead. `train_neural_likelihood` sets log_transform
    # accordingly, so this branch never receives it set.
    return likelihood_nn(model="nsf")(batch_x=x, batch_y=cond)


def _infer_categorical(outcomes: np.ndarray) -> tuple[bool, ...]:
    """Mark integer-valued columns taking few distinct values as categorical."""
    flags = []
    for j in range(outcomes.shape[1]):
        column = outcomes[:, j]
        finite = column[np.isfinite(column)]
        integral = finite.size and np.allclose(finite, np.round(finite))
        flags.append(bool(integral and len(np.unique(finite)) <= 20))
    return tuple(flags)


def _trial_features(inputs, n_trials: int) -> np.ndarray | None:
    """Per-trial conditioning drawn from the composition's inputs.

    The values entering the input nodes on a trial are what distinguishes one trial from
    another, so they condition the density alongside the parameters.  Columns that do not
    vary across trials carry no information and are dropped.
    """
    if not inputs:
        return None
    columns = []
    for value in inputs.values():
        array = np.asarray(value, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        array = array.reshape(array.shape[0], -1)
        if array.shape[0] != n_trials:
            continue
        columns.append(array)
    if not columns:
        return None
    features = np.concatenate(columns, axis=1)
    varying = features.std(axis=0) > 0
    return features[:, varying] if varying.any() else None


def _simulator_hash(pec, n_trial_features: int) -> str:
    """Identify the simulator a network was trained on.

    Changing the model's structure while keeping the parameter names invalidates a
    trained estimator without changing anything else recorded here.
    """
    parts = [type(pec).__name__, str(n_trial_features)]
    try:
        composition = pec.nodes[0]
        parts += sorted(node.name for node in composition.nodes)
    except Exception:
        pass
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _simulate_chunk(pec_factory, thetas, n_trials, n_outcomes):
    """Simulate every parameter draw in ``thetas``; returns (conditioning, outcomes).

    The PEC is built once for the whole chunk: constructing and compiling a composition
    is expensive relative to a simulation, so chunks amortize it.
    """
    import pandas as pd

    dummy = pd.DataFrame(np.zeros((n_trials, n_outcomes)))
    pec, inputs = pec_factory(dummy)
    features = _trial_features(inputs, n_trials)

    # Training needs the simulated outcomes, not a score for them. Scoring here would pay
    # the per-evaluation density cost a neural likelihood exists to remove, so the
    # objective is stubbed out and only the simulations are kept.
    pec.controller.function.set_pec_objective_function(lambda sim_data: 0.0)

    cond_rows, x_rows = [], []
    for theta in thetas:
        _, sim = pec.log_likelihood(*theta, inputs=inputs, return_sim_data=True)
        sim = np.asarray(sim, dtype=float)
        n_estimates = sim.shape[1]
        x_rows.append(sim.reshape(-1, sim.shape[-1]))
        block = np.repeat(np.asarray(theta, dtype=float).reshape(1, -1),
                          n_trials * n_estimates, axis=0)
        if features is not None:
            block = np.concatenate([block, np.repeat(features, n_estimates, axis=0)], axis=1)
        cond_rows.append(block)
    return np.concatenate(cond_rows), np.concatenate(x_rows)


def _fit_estimator(x, cond, categorical, categories, log_transform, *, epochs,
                   batch_size, learning_rate, validation_fraction, seed):
    """Train a density estimator by maximum likelihood; returns it and its held-out NLL."""
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    n_val = max(1, int(validation_fraction * n))
    order = torch.randperm(n, generator=generator)
    val_idx, train_idx = order[:n_val], order[n_val:]

    estimator = _build_estimator(x[train_idx], cond[train_idx], categorical, categories,
                                 log_transform)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=learning_rate)
    best = float("inf")
    for _ in range(epochs):
        shuffled = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        for start in range(0, shuffled.numel(), batch_size):
            batch = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            estimator.loss(x[batch], condition=cond[batch]).mean().backward()
            optimizer.step()
        with torch.no_grad():
            best = min(best, float(estimator.loss(x[val_idx], condition=cond[val_idx]).mean()))
    estimator.eval()
    return estimator, best


def _encode_outcomes(outcomes, categorical, categories, outcome_names) -> torch.Tensor:
    """Put outcomes in sbi's layout: continuous columns first, category codes last."""
    outcomes = np.asarray(outcomes, dtype=float)
    if outcomes.ndim != 2 or outcomes.shape[1] != len(outcome_names):
        raise NeuralLikelihoodError(
            f"Expected outcomes with {len(outcome_names)} columns "
            f"{list(outcome_names)}, got shape {outcomes.shape}."
        )
    encoded = outcomes.copy()
    for j, (is_cat, cats) in enumerate(zip(categorical, categories)):
        if not is_cat:
            continue
        codes = np.full(outcomes.shape[0], -1.0)
        for code, value in enumerate(cats):
            codes[np.isclose(outcomes[:, j], value)] = float(code)
        if (codes < 0).any():
            unseen = sorted(set(outcomes[codes < 0, j].tolist()))
            raise NeuralLikelihoodError(
                f"Outcome {outcome_names[j]!r} contains values {unseen} that were never "
                f"simulated during training (trained categories: {list(cats)})."
            )
        encoded[:, j] = codes
    flags = np.asarray(categorical, dtype=bool)
    order = np.concatenate([np.flatnonzero(~flags), np.flatnonzero(flags)])
    return torch.as_tensor(encoded[:, order], dtype=torch.float32)


def train_neural_likelihood(
    pec_factory: Callable,
    bounds: Mapping[str, tuple[float, float]],
    outcome_names: Sequence[str],
    *,
    n_parameter_samples: int = 20000,
    n_trials_per_sample: int = 100,
    categorical: Sequence[bool] | None = None,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    seed: int = 0,
    distributed_options: Mapping | None = None,
    n_chunks: int | None = None,
    strict: bool = True,
) -> NeuralLikelihood:
    """Train a `NeuralLikelihood` on data simulated from a composition.

    Arguments
    ---------

    pec_factory : callable
        ``pec_factory(data) -> (pec, inputs)``, the same contract distributed and
        hierarchical fitting use.  It is called with a placeholder table of
        **n_trials_per_sample** rows, since training simulates rather than fits.

    bounds : Mapping
        Parameter name to ``(lower, upper)``.  Iteration order fixes the order of the
        conditioning vector, and the trained estimator is valid only inside this box.

    outcome_names : Sequence[str]
        Names of the outcome variables, in the order the composition reports them.

    categorical : Sequence[bool] : default None
        Which outcomes are categorical.  Inferred from the simulated data when omitted,
        and recorded either way, so a disagreement with the data being fit is caught
        before the estimator is used.

    n_parameter_samples, n_trials_per_sample : int
        Parameter draws across the box, and trials simulated per draw.

    distributed_options : Mapping : default None
        Resolved exactly as for distributed fitting.  Generation is embarrassingly
        parallel; omit for a single process.

    strict : bool : default True
        Raise rather than warn when a validation gate fails.

    Returns
    -------
    A trained `NeuralLikelihood`.
    """
    from scipy.stats import qmc

    names = tuple(bounds)
    if not names:
        raise NeuralLikelihoodError("bounds must name at least one parameter.")
    lower = np.array([float(bounds[n][0]) for n in names])
    upper = np.array([float(bounds[n][1]) for n in names])
    if not np.all(upper > lower):
        bad = [n for n, lo, hi in zip(names, lower, upper) if hi <= lo]
        raise NeuralLikelihoodError(f"bounds must satisfy lower < upper; got {bad} reversed.")
    if n_parameter_samples < 2 or n_trials_per_sample < 1:
        raise NeuralLikelihoodError(
            "n_parameter_samples must be at least 2 and n_trials_per_sample at least 1."
        )

    # Sobol draws cover the box more evenly than independent uniforms at the same count.
    engine = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    thetas = qmc.scale(engine.random(n_parameter_samples), lower, upper)

    chunks = np.array_split(thetas, n_chunks or max(1, min(len(thetas), 64)))
    n_outcomes = len(outcome_names)
    if distributed_options is None:
        results = [_simulate_chunk(pec_factory, c, n_trials_per_sample, n_outcomes)
                   for c in chunks]
    else:
        from psyneulink.core.components.functions.nonstateful import fitfunctions

        client, close_fn = fitfunctions._dask_client(distributed_options)
        try:
            futures = [client.submit(_simulate_chunk, pec_factory, c,
                                     n_trials_per_sample, n_outcomes, pure=False)
                       for c in chunks]
            results = client.gather(futures)
        finally:
            if close_fn is not None:
                close_fn()

    cond = torch.as_tensor(np.concatenate([r[0] for r in results]), dtype=torch.float32)
    raw = np.concatenate([r[1] for r in results])
    if raw.shape[1] != n_outcomes:
        raise NeuralLikelihoodError(
            f"The composition reported {raw.shape[1]} outcome columns but "
            f"{n_outcomes} outcome_names were given: {list(outcome_names)}."
        )

    flags = tuple(bool(c) for c in categorical) if categorical is not None \
        else _infer_categorical(raw)
    if len(flags) != n_outcomes:
        raise NeuralLikelihoodError(
            f"categorical has {len(flags)} entries but there are {n_outcomes} outcomes."
        )
    categories = tuple(
        tuple(float(v) for v in np.unique(raw[:, j])) if is_cat else ()
        for j, is_cat in enumerate(flags)
    )
    continuous = raw[:, ~np.asarray(flags, dtype=bool)]
    log_transform = bool(any(flags)) and continuous.size > 0 and bool((continuous > 0).all())

    x = _encode_outcomes(raw, flags, categories, outcome_names)
    estimator, val_nll = _fit_estimator(
        x, cond, flags, categories, log_transform,
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        validation_fraction=validation_fraction, seed=seed,
    )

    from psyneulink import __version__ as pnl_version

    provenance = NeuralLikelihoodProvenance(
        fit_param_names=names,
        lower=tuple(lower.tolist()),
        upper=tuple(upper.tolist()),
        outcome_names=tuple(outcome_names),
        categorical=flags,
        categories=categories,
        log_transform=log_transform,
        n_trial_features=int(cond.shape[1] - len(names)),
        n_parameter_samples=int(n_parameter_samples),
        n_trials_per_sample=int(n_trials_per_sample),
        epochs=int(epochs),
        val_nll=float(val_nll),
        seed=int(seed),
        simulator_hash=_simulator_hash(pec_factory, int(cond.shape[1] - len(names))),
        psyneulink_version=str(pnl_version),
        sbi_version=str(_require_sbi().__version__),
    )
    probe = (x[:256].clone(), cond[:256].clone())
    likelihood = NeuralLikelihood(estimator, provenance, probe)
    _check_gates(likelihood, x, cond, val_nll, strict)
    return likelihood


def _check_gates(likelihood, x, cond, val_nll, strict):
    """Refuse an estimator that did not train, or that cannot score its own training data."""
    failures = []
    if not np.isfinite(val_nll):
        failures.append(f"held-out negative log-likelihood is {val_nll}")
    with torch.no_grad():
        sample = slice(0, min(4096, x.shape[0]))
        scored = likelihood._estimator.log_prob(x[sample], condition=cond[sample])
    finite = float(torch.isfinite(scored).float().mean())
    if finite < 0.999:
        failures.append(
            f"only {100 * finite:.2f}% of held-out rows received a finite log-density"
        )
    if not failures:
        return
    message = ("This neural likelihood did not pass its validation gates: "
               + "; ".join(failures) + ".")
    if strict:
        raise NeuralLikelihoodError(message + " Pass strict=False to return it anyway.")
    warnings.warn(message, NeuralLikelihoodWarning, stacklevel=3)
