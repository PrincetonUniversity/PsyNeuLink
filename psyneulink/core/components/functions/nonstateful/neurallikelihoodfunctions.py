# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ***************************************  Neural Likelihoods  *********************************************************

"""Density estimators trained on simulated data, used by `ParameterEstimationComposition` to compute
the likelihood of its data.  See :ref:`Neural Likelihoods <NeuralLikelihood>`.
"""

from __future__ import annotations

import copy
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
    """What an estimator was trained for, and how.

    The parameters, their ranges and the outcomes are checked against a model before the
    estimator is used with it (`check_matches`); the rest is a record of the training.
    """

    fit_param_names: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    outcome_names: tuple[str, ...]
    categorical: tuple[bool, ...]
    categories: tuple[tuple[float, ...], ...]
    log_transform: bool
    n_input_columns: int
    trial_feature_columns: tuple[int, ...]
    n_parameter_samples: int
    n_trials_per_sample: int
    epochs: int
    val_nll: float
    seed: int
    psyneulink_version: str
    sbi_version: str

    @property
    def n_trial_features(self) -> int:
        return len(self.trial_feature_columns)

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
            n_input_columns=raw["n_input_columns"],
            trial_feature_columns=tuple(raw["trial_feature_columns"]),
            n_parameter_samples=raw["n_parameter_samples"],
            n_trials_per_sample=raw["n_trials_per_sample"],
            epochs=raw["epochs"],
            val_nll=raw["val_nll"],
            seed=raw["seed"],
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
    `trial_log_prob` is differentiable with respect to the parameters.
    """

    def __init__(self, estimator, provenance: NeuralLikelihoodProvenance, shape_probe):
        self._estimator = estimator
        self.provenance = provenance
        # Example rows, from which `load` rebuilds the network before restoring its weights.
        self._shape_probe = shape_probe

    @property
    def fit_param_names(self) -> tuple[str, ...]:
        return self.provenance.fit_param_names

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

    # Continuous outcomes alone are modelled in their own units: sbi's plain flows take no
    # log transform, and `train_neural_likelihood` requests none for them.
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


def _input_columns(inputs, n_trials: int) -> np.ndarray:
    """The values entering the composition's input nodes, one row per trial."""
    columns = [np.zeros((n_trials, 0))]
    for value in (inputs or {}).values():
        array = np.asarray(value, dtype=float)
        array = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1, 1)
        if array.shape[0] == n_trials:
            columns.append(array)
    return np.concatenate(columns, axis=1)


def _split(thetas, n):
    """Split parameter draws into at most ``n`` non-empty chunks."""
    return np.array_split(thetas, max(1, min(int(n), len(thetas))))


def _check_parameters(pec, names):
    """Raise unless `pec` fits exactly `names`, in that order: draws are passed to it by position."""
    from psyneulink.core.compositions.hierarchical.subjectlikelihood import _reported_names

    declared = _reported_names(pec.controller.function.fit_param_names)
    if tuple(declared) != tuple(names):
        raise NeuralLikelihoodError(
            f"training draws parameters in the order {list(names)}, but the model fits "
            f"{list(declared)}. They are matched by position, so these have to agree; "
            f"name the bounds in the order the model declares them."
        )


def _simulate(pec, inputs, thetas, names, n_outcomes):
    """Simulate every draw through ``pec``.

    Returns the conditioning rows, the simulated outcomes, the number of trials per draw (set
    by ``inputs``, not by the model's data), and the layout of the inputs: how many columns
    they have, and which of them were used.
    """
    _check_parameters(pec, names)
    n_trials = None
    features = None
    layout = None

    # Only the simulated outcomes are needed, so scoring is switched off; and each draw needs
    # noise of its own, which a model built for fitting may be set to share. Both are restored
    # after, for a caller still using the model.
    function = pec.controller.function
    scoring = function._pec_objective_function
    function.set_pec_objective_function(lambda sim_data: 0.0)
    shared_noise = pec.controller.parameters.same_seed_for_all_allocations
    sharing = dict(shared_noise.values)
    for execution_id in sharing:
        shared_noise.set(False, execution_id)

    cond_rows, x_rows = [], []
    try:
        for theta in thetas:
            _, sim = pec.log_likelihood(*theta, inputs=inputs, return_sim_data=True)
            sim = np.asarray(sim, dtype=float)
            n_estimates = sim.shape[1]
            if n_trials is None:
                n_trials = sim.shape[0]
                # Inputs that vary from trial to trial are what tell trials apart; the
                # rest say nothing, and are left out.
                columns = _input_columns(inputs, n_trials)
                used = tuple(int(j) for j in np.flatnonzero(columns.std(axis=0) > 0))
                layout = (columns.shape[1], used)
                features = columns[:, list(used)] if used else None
            x_rows.append(sim.reshape(-1, sim.shape[-1]))
            block = np.repeat(np.asarray(theta, dtype=float).reshape(1, -1),
                              n_trials * n_estimates, axis=0)
            if features is not None:
                block = np.concatenate(
                    [block, np.repeat(features, n_estimates, axis=0)], axis=1
                )
            cond_rows.append(block)
    finally:
        function.set_pec_objective_function(scoring)
        for execution_id, value in sharing.items():
            shared_noise.set(value, execution_id)
    return np.concatenate(cond_rows), np.concatenate(x_rows), n_trials, layout


def _simulate_chunk(pec_factory, thetas, n_trials, names, n_outcomes):
    """Build a model and simulate ``thetas`` through it.

    This is what a worker is sent, since a composition cannot be sent to another process.
    """
    import pandas as pd

    pec, inputs = pec_factory(pd.DataFrame(np.zeros((n_trials, n_outcomes))))
    return _simulate(pec, inputs, thetas, names, n_outcomes)


def _fit_estimator(x, cond, categorical, categories, log_transform, *, epochs,
                   batch_size, learning_rate, validation_fraction, seed):
    """Train a density estimator by maximum likelihood; returns it and its held-out NLL.

    The weights returned are those of the epoch with the lowest held-out NLL, which is the one
    reported.
    """
    generator = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    n_val = max(1, int(validation_fraction * n))
    order = torch.randperm(n, generator=generator)
    val_idx, train_idx = order[:n_val], order[n_val:]

    estimator = _build_estimator(x[train_idx], cond[train_idx], categorical, categories,
                                 log_transform)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=learning_rate)
    best, best_state = float("inf"), None
    for _ in range(epochs):
        shuffled = train_idx[torch.randperm(train_idx.numel(), generator=generator)]
        for start in range(0, shuffled.numel(), batch_size):
            batch = shuffled[start:start + batch_size]
            optimizer.zero_grad()
            estimator.loss(x[batch], condition=cond[batch]).mean().backward()
            optimizer.step()
        with torch.no_grad():
            held_out = float(estimator.loss(x[val_idx], condition=cond[val_idx]).mean())
        if held_out < best:
            best, best_state = held_out, copy.deepcopy(estimator.state_dict())
    if best_state is not None:
        estimator.load_state_dict(best_state)
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
    # Categorical values become codes 0..K-1.
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
    # sbi's mixed estimator takes continuous columns first and categorical ones last.
    flags = np.asarray(categorical, dtype=bool)
    order = np.concatenate([np.flatnonzero(~flags), np.flatnonzero(flags)])
    return torch.as_tensor(encoded[:, order], dtype=torch.float32)


def train_neural_likelihood(
    bounds: Mapping[str, tuple[float, float]],
    outcome_names: Sequence[str],
    *,
    pec=None,
    inputs: Mapping | None = None,
    pec_factory: Callable | None = None,
    n_parameter_samples: int = 20000,
    n_trials_per_sample: int | None = None,
    categorical: Sequence[bool] | None = None,
    epochs: int = 30,
    batch_size: int = 512,
    learning_rate: float = 5e-4,
    validation_fraction: float = 0.1,
    seed: int = 0,
    distributed_options: Mapping | None = None,
    strict: bool = True,
) -> NeuralLikelihood:
    """Train a :class:`NeuralLikelihood` on data simulated from a composition.

    See :ref:`Neural Likelihoods <NeuralLikelihood>`.

    Arguments
    ---------

    bounds : Mapping
        specifies the range ``(lower, upper)`` of each fitted parameter, by name, in the order the model
        lists them.  The estimator is valid only within these ranges.

    outcome_names : Sequence[str]
        specifies the names of the outcome variables, which must be the column names of the **data** the
        estimator will be used to fit, in the order of the model's ``outcome_variables``.

    pec : ParameterEstimationComposition : default None
        specifies a model to simulate in this process.  Requires **inputs**.

    inputs : Mapping : default None
        specifies the inputs with which **pec** is run; the number of trials simulated for each parameter
        draw is the number of trials in **inputs**.

    pec_factory : callable : default None
        specifies a function ``pec_factory(data) -> (pec, inputs)`` that builds the model, as used for
        :ref:`distributed fitting <DistributedFitting>`; it is called with a table of **n_trials_per_sample**
        rows.  Required by **distributed_options**.  Exactly one of **pec** and **pec_factory** must be
        specified.

    n_parameter_samples : int : default 20000
        specifies the number of parameter values drawn from within **bounds** and simulated.

    n_trials_per_sample : int : default 100
        specifies the number of trials simulated for each parameter draw, when **pec_factory** is used.

    categorical : Sequence[bool] : default None
        specifies which outcome variables are categorical; if not specified, this is inferred from the
        simulated data.

    epochs : int : default 30
        specifies the number of passes over the simulated data made in training.

    batch_size : int : default 512
        specifies the number of rows in each training step.

    learning_rate : float : default 5e-4
        specifies the learning rate of the Adam optimizer used for training.

    validation_fraction : float : default 0.1
        specifies the fraction of the simulated data held out to evaluate the estimator.

    seed : int : default 0
        specifies the seed for the parameter draws and for training.

    distributed_options : Mapping : default None
        specifies a Dask cluster over which to distribute the simulations, as for :ref:`distributed fitting
        <DistributedFitting>`.  Requires **pec_factory**.  Each worker builds its own model before simulating,
        so this is worthwhile only for more than a few hundred parameter draws.

    strict : bool : default True
        specifies whether an estimator that fails validation raises a `NeuralLikelihoodError` (True) or
        issues a `NeuralLikelihoodWarning` (False).

    Returns
    -------
    A trained :class:`NeuralLikelihood`.
    """
    from scipy.stats import qmc

    if (pec is None) == (pec_factory is None):
        raise NeuralLikelihoodError(
            "Supply exactly one of pec, a model to simulate in this process, or "
            "pec_factory, a callable that builds one."
        )
    if pec is not None and distributed_options is not None:
        raise NeuralLikelihoodError(
            "Distributing generation requires pec_factory: a composition cannot be sent "
            "to another process, so each worker has to build its own."
        )
    if pec is not None and inputs is None:
        raise NeuralLikelihoodError(
            "pec requires inputs: they set how many trials each draw simulates, and "
            "what distinguishes one trial from another."
        )
    if pec is not None and n_trials_per_sample is not None:
        raise NeuralLikelihoodError(
            "n_trials_per_sample applies to pec_factory only; with pec the number of "
            "trials simulated per draw is the length of inputs."
        )

    names = tuple(bounds)
    if not names:
        raise NeuralLikelihoodError("bounds must name at least one parameter.")
    lower = np.array([float(bounds[n][0]) for n in names])
    upper = np.array([float(bounds[n][1]) for n in names])
    if not np.all(upper > lower):
        bad = [n for n, lo, hi in zip(names, lower, upper) if hi <= lo]
        raise NeuralLikelihoodError(f"bounds must satisfy lower < upper; got {bad} reversed.")
    if n_parameter_samples < 2:
        raise NeuralLikelihoodError("n_parameter_samples must be at least 2.")
    if n_trials_per_sample is not None and n_trials_per_sample < 1:
        raise NeuralLikelihoodError("n_trials_per_sample must be at least 1.")
    # Checked before simulating, which is most of the cost.
    _require_sbi()

    # Sobol draws cover the box more evenly than independent uniforms at the same count.
    engine = qmc.Sobol(d=len(names), scramble=True, seed=seed)
    thetas = qmc.scale(engine.random(n_parameter_samples), lower, upper)

    n_outcomes = len(outcome_names)
    n_trials = n_trials_per_sample or 100

    # One share of the draws per worker: building a model costs far more than simulating it.
    if pec is not None:
        results = [_simulate(pec, inputs, thetas, names, n_outcomes)]
    elif distributed_options is None:
        results = [_simulate_chunk(pec_factory, thetas, n_trials, names, n_outcomes)]
    else:
        from psyneulink.core.components.functions.nonstateful import fitfunctions

        client, close_fn = fitfunctions._dask_client(distributed_options)
        try:
            # nthreads() lists every worker; scheduler_info() lists only the first few.
            workers = len(client.nthreads()) or 1
            futures = [client.submit(_simulate_chunk, pec_factory, c,
                                     n_trials, names, n_outcomes, pure=False)
                       for c in _split(thetas, workers)]
            results = client.gather(futures)
        finally:
            if close_fn is not None:
                close_fn()

    n_trials, layout = results[0][2], results[0][3]
    if any(r[3] != layout for r in results):
        raise NeuralLikelihoodError(
            "pec_factory returned inputs laid out differently on different workers; each "
            "call has to return the same inputs for the same number of trials."
        )
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
        n_input_columns=int(layout[0]),
        trial_feature_columns=layout[1],
        n_parameter_samples=int(n_parameter_samples),
        n_trials_per_sample=int(n_trials),
        epochs=int(epochs),
        val_nll=float(val_nll),
        seed=int(seed),
        psyneulink_version=str(pnl_version),
        sbi_version=str(_require_sbi().__version__),
    )
    probe = (x[:256].clone(), cond[:256].clone())
    likelihood = NeuralLikelihood(estimator, provenance, probe)
    _check_gates(likelihood, x, cond, val_nll, strict)
    return likelihood


def _check_gates(likelihood, x, cond, val_nll, strict):
    """Refuse an estimator whose held-out loss is not finite, or that cannot score the data it was trained on."""
    failures = []
    if not np.isfinite(val_nll):
        failures.append(f"held-out negative log-likelihood is {val_nll}")
    with torch.no_grad():
        sample = slice(0, min(4096, x.shape[0]))
        scored = likelihood._estimator.log_prob(x[sample], condition=cond[sample])
    finite = float(torch.isfinite(scored).float().mean())
    if finite < 0.999:
        failures.append(
            f"only {100 * finite:.2f}% of the simulated rows received a finite log-density"
        )
    if not failures:
        return
    message = ("This neural likelihood did not pass its validation gates: "
               + "; ".join(failures) + ".")
    if strict:
        raise NeuralLikelihoodError(message + " Pass strict=False to return it anyway.")
    warnings.warn(message, NeuralLikelihoodWarning, stacklevel=3)
