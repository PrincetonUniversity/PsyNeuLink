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
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from psyneulink.core.globals import SampleIterator

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


@dataclass(frozen=True)
class NeuralLikelihoodMetadata:
    """What an estimator was trained for, saved with it so that it can be rebuilt and checked.

    The parameters, their ranges and the outcomes are checked against a model before the
    estimator is used with it (`check_matches`), and its inputs and observed outcomes against
    the trials it scores (`check_inputs`, `check_outcomes`).  ``val_nll`` is the estimator's
    negative log-likelihood per trial on the data held out from training.
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
    constant_inputs: tuple[float, ...]
    val_nll: float

    @property
    def n_trial_features(self) -> int:
        return len(self.trial_feature_columns)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> NeuralLikelihoodMetadata:
        raw = json.loads(text)
        raw["categories"] = [tuple(c) for c in raw["categories"]]
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in raw.items()})

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

    def check_inputs(self, columns):
        """Raise unless the model's inputs, as trial-by-trial ``columns``, are those of training.

        Inputs that were the same on every trial in training told the estimator nothing about the
        trial it was scoring, so it can score only trials in which they hold the same values.
        """
        if columns.shape[1] != self.n_input_columns:
            raise NeuralLikelihoodError(
                f"This neural likelihood was trained on inputs with {self.n_input_columns} "
                f"value(s) per trial; these inputs have {columns.shape[1]}. Pass inputs laid out "
                f"as they were for training."
            )
        held = [j for j in range(self.n_input_columns) if j not in self.trial_feature_columns]
        if not np.allclose(columns[:, held], self.constant_inputs):
            raise NeuralLikelihoodError(
                f"This neural likelihood was trained with inputs held at "
                f"{list(self.constant_inputs)} on every trial, and cannot score trials in which "
                f"they differ."
            )

    def check_outcomes(self, outcomes):
        """Raise unless every observed outcome is one the estimator can score."""
        outcomes = np.asarray(outcomes, dtype=float)
        if not np.isfinite(outcomes).all():
            raise NeuralLikelihoodError(
                "The data contain outcomes that are not finite, which a neural likelihood cannot score."
            )
        continuous = outcomes[:, ~np.asarray(self.categorical, dtype=bool)]
        if self.log_transform and (continuous <= 0).any():
            raise NeuralLikelihoodError(
                "The data contain continuous outcomes that are not positive, which this neural "
                "likelihood cannot score: it models their logarithm."
            )


class NeuralLikelihood:
    """A trained conditional density ``p(outcomes | parameters, trial features)``.

    Built by `train_neural_likelihood`, saved with `save`, and reloaded with `load`.
    `trial_log_prob` is differentiable with respect to the parameters.

    Attributes
    ----------

    metadata : NeuralLikelihoodMetadata
        what the estimator was trained for, which is checked against a model before it is used to fit one
        (see `Neural_Likelihood_Matching`), and its negative log-likelihood per trial on the data held out
        from training (``val_nll``).
    """

    def __init__(self, estimator, metadata: NeuralLikelihoodMetadata, shape_probe):
        self._estimator = estimator
        self.metadata = metadata
        # Example rows, from which `load` rebuilds the network before restoring its weights.
        self._shape_probe = shape_probe

    def _encode_outcomes(self, outcomes: np.ndarray) -> torch.Tensor:
        """Reorder to sbi's layout and map categorical values onto codes 0..K-1."""
        return _encode_outcomes(
            outcomes,
            self.metadata.categorical,
            self.metadata.categories,
            self.metadata.outcome_names,
        )

    def _conditioning(self, theta: torch.Tensor, trial_features, n_trials) -> torch.Tensor:
        """Give each trial its parameters, one vector for all or a row each, then its features."""
        cond = theta.reshape(-1, theta.shape[-1]).expand(n_trials, -1)
        if self.metadata.n_trial_features:
            if trial_features is None:
                raise NeuralLikelihoodError(
                    f"This neural likelihood was trained with "
                    f"{self.metadata.n_trial_features} per-trial feature(s), so "
                    f"scoring requires trial_features."
                )
            feats = torch.as_tensor(np.asarray(trial_features, dtype=float), dtype=torch.float32)
            if feats.shape != (n_trials, self.metadata.n_trial_features):
                raise NeuralLikelihoodError(
                    f"Expected trial_features of shape "
                    f"({n_trials}, {self.metadata.n_trial_features}), got "
                    f"{tuple(feats.shape)}."
                )
            cond = torch.cat([cond, feats], dim=-1)
        return cond

    def trial_log_prob(self, theta, outcomes, trial_features=None) -> torch.Tensor:
        """Per-trial log densities, differentiable with respect to ``theta``.

        ``theta`` is one vector of parameters for every trial, or one row of them per trial.
        """
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
        """Write weights and metadata to ``path``."""
        torch.save(
            {
                "state_dict": self._estimator.state_dict(),
                "metadata": self.metadata.to_json(),
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
        metadata = NeuralLikelihoodMetadata.from_json(blob["metadata"])
        estimator = _build_estimator(
            blob["probe_x"],
            blob["probe_cond"],
            metadata.categorical,
            metadata.categories,
            metadata.log_transform,
        )
        estimator.load_state_dict(blob["state_dict"])
        estimator.eval()
        return cls(estimator, metadata, (blob["probe_x"], blob["probe_cond"]))


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


def _input_columns(inputs, n_trials: int, model) -> np.ndarray:
    """The values entering the model's input nodes, one row per trial.

    They are taken in the order ``model`` lists its nodes, so the same inputs give the same
    columns however ``inputs`` lists them.
    """
    position = {node: i for i, node in enumerate(model.nodes)}
    columns = [np.zeros((n_trials, 0))]
    for _, value in sorted((inputs or {}).items(), key=lambda item: position.get(item[0], -1)):
        array = np.asarray(value, dtype=float)
        array = array.reshape(array.shape[0], -1) if array.ndim > 1 else array.reshape(-1, 1)
        if array.shape[0] == n_trials:
            columns.append(array)
    return np.concatenate(columns, axis=1)


def _check_model(pec, names):
    """Raise unless `pec` can be simulated for training and fits exactly `names`, in that order.

    Draws are passed to it by position.
    """
    from psyneulink.core.compositions.hierarchical.subjectlikelihood import _reported_names

    if not pec.scores_by_simulation:
        raise NeuralLikelihoodError(
            "Train on a model scored by simulating it: this one is scored by a trained estimator "
            '(likelihood_estimator="neural"), and so is never simulated.'
        )
    if pec.depends_on:
        raise NeuralLikelihoodError(
            "Train on a model without depends_on: the estimator is trained over each parameter's "
            "range, and scores every condition's value of it when fitting."
        )
    declared = _reported_names(pec.controller.function.fit_param_names)
    if tuple(declared) != tuple(names):
        raise NeuralLikelihoodError(
            f"training draws parameters in the order {list(names)}, but the model fits "
            f"{list(declared)}. They are matched by position, so these have to agree; "
            f"name the bounds in the order the model declares them."
        )


def _simulate(pec, inputs, thetas, names, seed=0, first_draw=0):
    """Simulate every draw through ``pec``; ``first_draw`` is the place of the first among all draws.

    Returns the conditioning rows, the simulated outcomes, and the layout of the inputs: how many
    columns they have, which of them were used, and the values of the rest.  Each draw simulates
    as many trials as ``inputs`` has, whatever the model's data.
    """
    _check_model(pec, names)
    n_trials = None
    features = None
    layout = None

    # Only the simulated outcomes are needed, so scoring is switched off; it is restored after, for
    # a caller still using the model, as are the seeds the model was about to use.
    function = pec.controller.function
    scoring = function._pec_objective_function
    function.set_pec_objective_function(lambda sim_data: 0.0)
    seed_dimension = function.parameters.randomization_dimension.get()
    model_seeds = function.search_space[seed_dimension]
    n_estimates = int(np.squeeze(pec.controller.parameters.num_estimates.get()))

    cond_rows, x_rows = [], []
    try:
        for draw, theta in enumerate(thetas, start=first_draw):
            # Seeded by the draw's place among all draws, so each draw has noise of its own, and the
            # same noise however the draws are divided among workers.
            seeds = np.random.SeedSequence([seed, draw]).generate_state(n_estimates) % (2**31 - 1)
            function.search_space[seed_dimension] = SampleIterator(seeds.tolist())
            _, sim = pec.log_likelihood(*theta, inputs=inputs, return_sim_data=True)
            sim = np.asarray(sim, dtype=float)
            if n_trials is None:
                n_trials = sim.shape[0]
                # Inputs that vary from trial to trial are what tell trials apart; the rest say
                # nothing, and are recorded by value.
                columns = _input_columns(inputs, n_trials, pec.model)
                used = tuple(int(j) for j in np.flatnonzero(columns.std(axis=0) > 0))
                held = [j for j in range(columns.shape[1]) if j not in used]
                layout = (columns.shape[1], used, tuple(float(v) for v in columns[0, held]))
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
        function.search_space[seed_dimension] = model_seeds
    return np.concatenate(cond_rows), np.concatenate(x_rows), layout


def _simulate_chunk(pec_factory, data, thetas, first_draw, names, seed, worker_cores, training_id):
    """Simulate ``thetas`` on a Dask worker, through the model it builds with ``pec_factory``.

    As for a distributed fit, the model is built once per worker, and the lock keeps two models
    from being compiled or run at once in one process.
    """
    from psyneulink.core.components.functions.nonstateful import fitfunctions

    with fitfunctions._PEC_EVALUATION_LOCK:
        pec, inputs = fitfunctions._worker_pec(pec_factory, data, worker_cores, training_id)
        return _simulate(pec, inputs, thetas, names, seed, first_draw)


def _held_out_draws(cond, n_params, validation_fraction, generator):
    """Split rows into training and held-out ones by parameter draw, so that the held-out rows
    come from parameter values not trained on.
    """
    _, draw = torch.unique(cond[:, :n_params], dim=0, return_inverse=True)
    n_draws = int(draw.max()) + 1
    n_val = min(max(1, int(validation_fraction * n_draws)), n_draws - 1)
    held_out = torch.isin(draw, torch.randperm(n_draws, generator=generator)[:n_val])
    return torch.nonzero(~held_out).flatten(), torch.nonzero(held_out).flatten()


def _fit_estimator(x, cond, categorical, categories, log_transform, *, n_params, epochs,
                   batch_size, learning_rate, validation_fraction, seed):
    """Train a density estimator by maximum likelihood; returns it and its held-out NLL.

    The weights returned are those of the epoch with the lowest held-out NLL, which is the one
    reported.
    """
    generator = torch.Generator().manual_seed(seed)
    train_idx, val_idx = _held_out_draws(cond, n_params, validation_fraction, generator)

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
    n_parameter_samples: int = 16384,
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

    n_parameter_samples : int : default 16384
        specifies the number of parameter values drawn from within **bounds** and simulated; they cover
        **bounds** most evenly when this is a power of 2.

    n_trials_per_sample : int : default None
        specifies the number of trials simulated for each parameter draw when **pec_factory** is used; 100
        if not specified.

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
        specifies the fraction of the parameter draws whose simulated data are held out from training, to
        evaluate the estimator.

    seed : int : default 0
        specifies the seed for the parameter draws, the noise simulated at each of them, and training.

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
            "Distributing the simulations requires pec_factory: a composition cannot be sent "
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
    # What a factory is called with: one row per trial to simulate, in the data's columns.
    placeholder = pd.DataFrame(np.zeros((n_trials_per_sample or 100, n_outcomes)),
                               columns=list(outcome_names))

    if pec is not None:
        results = [_simulate(pec, inputs, thetas, names, seed)]
    elif distributed_options is None:
        results = [_simulate(*pec_factory(placeholder), thetas, names, seed)]
    else:
        from psyneulink.core.components.functions.nonstateful import fitfunctions

        client, close_fn = fitfunctions._dask_client(distributed_options)
        try:
            # One share of the draws per worker: building a model costs far more than simulating
            # it. nthreads() lists every worker; scheduler_info() lists only the first few.
            workers = len(client.nthreads()) or 1
            worker_cores = fitfunctions._resolve_worker_cores(distributed_options)
            training_id = uuid.uuid4().hex
            shares = np.array_split(thetas, min(workers, len(thetas)))
            starts = np.cumsum([0] + [len(share) for share in shares[:-1]])
            futures = [client.submit(_simulate_chunk, pec_factory, placeholder, share, int(start),
                                     names, seed, worker_cores, training_id, pure=False)
                       for share, start in zip(shares, starts)]
            results = client.gather(futures)
        finally:
            if close_fn is not None:
                close_fn()

    layout = results[0][2]
    if any(r[2] != layout for r in results):
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
    if all(flags):
        raise NeuralLikelihoodError(
            "Every outcome is categorical, and a neural likelihood needs at least one continuous "
            "outcome, such as a response time."
        )
    categories = tuple(
        tuple(float(v) for v in np.unique(raw[:, j])) if is_cat else ()
        for j, is_cat in enumerate(flags)
    )
    continuous = raw[:, ~np.asarray(flags, dtype=bool)]
    log_transform = bool(any(flags)) and bool((continuous > 0).all())

    x = _encode_outcomes(raw, flags, categories, outcome_names)
    estimator, val_nll = _fit_estimator(
        x, cond, flags, categories, log_transform, n_params=len(names),
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        validation_fraction=validation_fraction, seed=seed,
    )

    metadata = NeuralLikelihoodMetadata(
        fit_param_names=names,
        lower=tuple(lower.tolist()),
        upper=tuple(upper.tolist()),
        outcome_names=tuple(outcome_names),
        categorical=flags,
        categories=categories,
        log_transform=log_transform,
        n_input_columns=int(layout[0]),
        trial_feature_columns=layout[1],
        constant_inputs=layout[2],
        val_nll=float(val_nll),
    )
    probe = (x[:256].clone(), cond[:256].clone())
    likelihood = NeuralLikelihood(estimator, metadata, probe)
    _check_gates(likelihood, x, cond, val_nll, strict)
    return likelihood


def _check_gates(likelihood, x, cond, val_nll, strict):
    """Refuse an estimator whose held-out loss is not finite, or that cannot score the data it was trained on."""
    failures = []
    if not np.isfinite(val_nll):
        failures.append(f"held-out negative log-likelihood is {val_nll}")
    # Rows are ordered by parameter draw, so an even spread of them covers every draw.
    rows = np.unique(np.linspace(0, x.shape[0] - 1, min(4096, x.shape[0])).astype(int))
    with torch.no_grad():
        scored = likelihood._estimator.log_prob(x[rows], condition=cond[rows])
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
