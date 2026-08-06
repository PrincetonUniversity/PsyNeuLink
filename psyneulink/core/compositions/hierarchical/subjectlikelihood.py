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
    "SubjectLikelihoodProvider",
    "SubjectSplit",
    "split_stacked_data",
]

#: Participant count above which building every model in one process is likely to exhaust memory.
#: Each participant's model is compiled separately, so they accumulate.
IN_PROCESS_SUBJECT_WARN_THRESHOLD = 32


def _parameter_name(qualified_name):
    """Strip the mechanism instance a parameter name is qualified by.

    A model reports its fitted parameters as ``<mechanism>.<parameter>``, and the mechanism part
    carries a number assigned in construction order.  Since each participant's model is built
    separately, the same parameter appears as ``DDM-6.rate`` for one participant and ``DDM-7.rate``
    for the next.  The parameter itself is what identifies it across participants.
    """
    return qualified_name.rsplit(".", 1)[-1]


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

    Notes
    -----

    Every participant's model must fit the same parameters, in the same order, over the same ranges.
    The group prior is defined in a space derived from those ranges, so ranges that differ between
    participants would silently mean different things for different people.  This is checked, not
    assumed.

    Models are compared by parameter name and bounds, but the comparison ignores the mechanism
    instance each name is qualified by.  A factory builds a fresh model per participant, so the same
    parameter is reported as ``DDM-6.rate`` for one and ``DDM-7.rate`` for the next; the instance
    number is an artefact of construction order and carries no meaning here.
    """

    def __init__(self, pec_factory, data_slices):
        self._factory = pec_factory
        self._data_slices = list(data_slices)
        self._cache = {}
        self._fit_param_names = None
        self._lower = None
        self._upper = None

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
    def fit_param_names(self):
        """Names of the fitted parameters, in the order the model reports them."""
        if self._fit_param_names is None:
            self._build(0)
        return self._fit_param_names

    @property
    def bounds(self):
        """``(lower, upper)`` arrays of the search range, in `fit_param_names` order."""
        if self._lower is None:
            self._build(0)
        return self._lower, self._upper

    @property
    def transform(self):
        """Transform between the model's units and the unconstrained space of the group model."""
        lower, upper = self.bounds
        return BoundedTransform(lower=lower, upper=upper)

    def _build(self, subject_index):
        """Build and cache one participant's model, checking it against the first one built."""
        if subject_index in self._cache:
            return self._cache[subject_index]

        pec, inputs = self._factory(self._data_slices[subject_index], subject_index)
        function = pec.controller.function
        qualified = tuple(function.fit_param_names)
        names = tuple(_parameter_name(n) for n in qualified)
        bound_map = function.fit_param_bounds
        lower = np.array([bound_map[n][0] for n in qualified], dtype=float)
        upper = np.array([bound_map[n][1] for n in qualified], dtype=float)

        if self._fit_param_names is None:
            self._fit_param_names, self._lower, self._upper = names, lower, upper
        else:
            if names != self._fit_param_names:
                raise ValueError(
                    f"participant {subject_index} fits {list(names)}, but participant 0 fits "
                    f"{list(self._fit_param_names)}; every participant must fit the same "
                    f"parameters in the same order"
                )
            if not (np.allclose(lower, self._lower) and np.allclose(upper, self._upper)):
                raise ValueError(
                    f"participant {subject_index} searches ranges "
                    f"{list(zip(lower, upper))}, but participant 0 searches "
                    f"{list(zip(self._lower, self._upper))}; the group prior is defined in terms "
                    f"of these ranges, so they must agree across participants"
                )

        self._cache[subject_index] = (pec, inputs)
        return self._cache[subject_index]

    def log_likelihood(self, theta, subject_index):
        """Log-likelihood of one participant's data at parameters `theta`, in the model's units."""
        pec, inputs = self._build(subject_index)
        return float(pec.log_likelihood(*np.asarray(theta, dtype=float), inputs=inputs))

    def close(self):
        """Drop the cached models."""
        self._cache.clear()
