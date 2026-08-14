# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# **************************************  Subject Likelihood  **********************************************************

"""Per-participant likelihoods for hierarchical fitting.

The EM driver needs one thing from the model: the log-likelihood of a given participant's data at
given parameters.  It is deliberately kept behind `SubjectLikelihoodProvider` so that the driver
never touches a `Composition` or a `ParameterEstimationComposition` -- it can be fitted against a
closed-form test model and against a real simulation with no change.

`PECFactorySubjectLikelihood` is the implementation that uses real models.  A participant's data is
one slice of the stacked table the user supplied, and the user's factory turns that slice into a
`ParameterEstimationComposition` for that participant alone.

A factory is required rather than optional because PsyNeuLink has no way to copy a `Composition`:
there is no ``Composition.copy()`` and no ``__deepcopy__``, so a participant's model cannot be
cloned from a template and must be built.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from psyneulink._typing import Tuple
from psyneulink.core.compositions.hierarchical.transforms import BoundedTransform

__all__ = [
    "PECFactorySubjectLikelihood",
    "ParameterSchema",
    "SubjectLikelihoodProvider",
    "SubjectSplit",
    "split_stacked_data",
]

#: Participant count above which building every model in one process is likely to exhaust memory.
#: Each participant's model is compiled separately, so they accumulate.
IN_PROCESS_SUBJECT_WARN_THRESHOLD = 32


def _comparable_names(qualified):
    """Parameter names with the mechanism instance stripped, for comparing one model to another.

    A model reports its fitted parameters as ``<mechanism>.<parameter>``, and the mechanism part
    carries a number assigned in construction order.  Since every model here is built separately,
    the same parameter appears as ``DDM-6.rate`` in one and ``DDM-7.rate`` in the next; only the
    parameter itself identifies it across models.
    """
    return tuple(name.rsplit(".", 1)[-1] for name in qualified)


def _reported_names(qualified):
    """Names to report results under.

    Normally the mechanism instance is dropped, for the reason `_comparable_names` gives.  A model
    that fits the same parameter on two mechanisms -- ``left.rate`` and ``right.rate`` -- would be
    left with one name for two parameters, so in that case the full names are kept instead.
    """
    stripped = _comparable_names(qualified)
    return stripped if len(set(stripped)) == len(stripped) else tuple(qualified)


@dataclass(frozen=True)
class ParameterSchema:
    """The parameters a model fits and the ranges it searches them over.

    Every model in a hierarchical fit must agree on this.  The group prior is defined in a space
    derived from the search ranges, so ranges that differed between participants would silently mean
    different things for different people, and parameters in a different order would be assigned each
    other's values.
    """

    names: Tuple[str, ...]      # with the mechanism instance stripped; what comparisons use
    reported: Tuple[str, ...]   # what results are labelled with
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]
    source: str                 # where this came from, for error messages

    @classmethod
    def from_pec(cls, pec, source):
        """Read the schema off a built `ParameterEstimationComposition`."""
        function = pec.controller.function
        qualified = tuple(function.fit_param_names)
        bound_map = function.fit_param_bounds
        return cls(
            names=_comparable_names(qualified),
            reported=_reported_names(qualified),
            lower=tuple(float(bound_map[n][0]) for n in qualified),
            upper=tuple(float(bound_map[n][1]) for n in qualified),
            source=source,
        )

    def check_matches(self, other):
        """Raise unless `other` fits the same parameters, in the same order, over the same ranges."""
        if other.names != self.names:
            raise ValueError(
                f"{other.source} fits {list(other.names)}, but {self.source} fits "
                f"{list(self.names)}; every model in a hierarchical fit must fit the same "
                f"parameters in the same order"
            )
        if not (np.allclose(other.lower, self.lower) and np.allclose(other.upper, self.upper)):
            raise ValueError(
                f"{other.source} searches ranges {list(zip(other.lower, other.upper))}, but "
                f"{self.source} searches {list(zip(self.lower, self.upper))}; the group prior is "
                f"defined in terms of these ranges, so they must agree"
            )


class SubjectLikelihoodProvider:
    """What the EM driver requires of a model, and nothing more.

    Implementations expose the number of participants and parameters, the parameter names and their
    bounds, the transform between the model's units and the unconstrained space the group model uses,
    and a log-likelihood for one participant at one parameter vector.
    """

    n_subjects: int
    n_params: int
    fit_param_names: Tuple[str, ...]

    def log_likelihood(self, theta, subject_index):
        """Log-likelihood of participant `subject_index`'s data at parameters `theta`."""
        raise NotImplementedError

    def close(self):
        """Release anything held open.  Optional."""


@dataclass
class SubjectSplit:
    """A stacked trial table divided into one frame per participant."""

    labels: tuple            # participant identifiers, in first-appearance order
    masks: list              # boolean array per participant, over rows of the original table
    frames: list             # DataFrame per participant, participant column removed

    @property
    def n_subjects(self):
        return len(self.labels)


def split_stacked_data(data, subject_id):
    """Divide a stacked trial table into one frame per participant.

    Participants are ordered by first appearance rather than sorted, so that the order of results
    follows the data as given.  Sorting would silently reorder results if a user relabelled their
    participants.

    Arguments
    ---------

    data : pandas.DataFrame
        One row per trial, all participants stacked, including a column identifying who produced
        each row.

    subject_id : str
        Name of that column.

    Returns
    -------

    A `SubjectSplit`.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(
            f"hierarchical fitting requires data as a pandas DataFrame so that the "
            f"'{subject_id}' column can be identified; got {type(data).__name__}"
        )
    if subject_id not in data.columns:
        raise ValueError(
            f"subject_id '{subject_id}' is not a column of data; "
            f"available columns are {list(data.columns)}"
        )

    # A missing identifier cannot be matched to itself, so such a trial would belong to no
    # participant and be dropped from the fit without saying so.
    n_missing = int(data[subject_id].isna().sum())
    if n_missing:
        raise ValueError(
            f"column '{subject_id}' has no participant identifier on {n_missing} of "
            f"{len(data)} rows; every trial must say who produced it"
        )

    column = data[subject_id].to_numpy()
    labels = tuple(pd.unique(data[subject_id]))
    if len(labels) < 2:
        raise ValueError(
            f"hierarchical fitting needs at least two participants; column '{subject_id}' "
            f"holds {len(labels)}. With one participant the group variance is not identified."
        )

    masks, frames = [], []
    for label in labels:
        mask = column == label
        masks.append(mask)
        frames.append(data.loc[mask].drop(columns=[subject_id]).reset_index(drop=True))
    return SubjectSplit(labels=labels, masks=masks, frames=frames)


class PECFactorySubjectLikelihood(SubjectLikelihoodProvider):
    """Per-participant likelihoods backed by one `ParameterEstimationComposition` each.

    Arguments
    ---------

    pec_factory : callable
        ``pec_factory(data, subject_index=None) -> (pec, inputs)``.  Given one participant's trials,
        returns a freshly built `ParameterEstimationComposition` for that participant and the inputs
        to run it with.  Must be defined at module level so that it can be sent to a worker process.

        `subject_index` lets the factory vary per participant, which matters for the random seed:
        if every participant's model shared one seed they would share one stream of simulation
        noise, and that common component would be absorbed into the group variance rather than
        averaging out.

    data_slices : sequence of pandas.DataFrame
        One participant's trials each, in participant order.  Normally `SubjectSplit.frames`.

    schema : ParameterSchema : default None
        What every participant's model is required to fit, normally read off the composition the
        user configured the fit on.  When omitted the first participant's model defines it.

    Notes
    -----

    Every participant's model must fit the same parameters, in the same order, over the same ranges.
    The group prior is defined in a space derived from those ranges, so ranges that differ between
    participants would silently mean different things for different people.  This is checked, not
    assumed; see `ParameterSchema`.
    """

    def __init__(self, pec_factory, data_slices, schema=None):
        self._factory = pec_factory
        self._data_slices = list(data_slices)
        self._cache = {}
        self._schema = schema

        if self.n_subjects > IN_PROCESS_SUBJECT_WARN_THRESHOLD:
            warnings.warn(
                f"building {self.n_subjects} participant models in one process; each is compiled "
                f"separately and they accumulate. Consider distributing the fit.",
                ResourceWarning,
                stacklevel=2,
            )

    @property
    def n_subjects(self):
        return len(self._data_slices)

    @property
    def n_params(self):
        return len(self.fit_param_names)

    @property
    def schema(self):
        """What every participant's model fits, and over what ranges."""
        if self._schema is None:
            self._build(0)
        return self._schema

    @property
    def fit_param_names(self):
        """Names of the fitted parameters, in the order the model reports them."""
        return self.schema.reported

    @property
    def bounds(self):
        """``(lower, upper)`` arrays of the search range, in `fit_param_names` order."""
        schema = self.schema
        return np.asarray(schema.lower, dtype=float), np.asarray(schema.upper, dtype=float)

    @property
    def transform(self):
        """Transform between the model's units and the unconstrained space of the group model."""
        lower, upper = self.bounds
        return BoundedTransform(lower=lower, upper=upper)

    def _build(self, subject_index):
        """Build and cache one participant's model, checking what it fits against the schema."""
        if subject_index in self._cache:
            return self._cache[subject_index]

        pec, inputs = self._factory(self._data_slices[subject_index], subject_index)
        schema = ParameterSchema.from_pec(
            pec, source=f"the model built for participant {subject_index}"
        )
        if self._schema is None:
            self._schema = schema
        else:
            self._schema.check_matches(schema)

        self._cache[subject_index] = (pec, inputs)
        return self._cache[subject_index]

    def log_likelihood(self, theta, subject_index):
        """Log-likelihood of one participant's data at parameters `theta`, in the model's units."""
        pec, inputs = self._build(subject_index)
        return float(pec.log_likelihood(*np.asarray(theta, dtype=float), inputs=inputs))

    def close(self):
        """Drop the cached models."""
        self._cache.clear()
