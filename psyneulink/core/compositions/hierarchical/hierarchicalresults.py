# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ************************************  Hierarchical Results  **********************************************************

"""Results of a hierarchical fit, expressed in the model's own units.

Three group-level quantities are easy to conflate, so they are reported separately.
``group_parameters["value"]`` is the group mean mapped through the transform, which makes it the
**median** of the implied distribution of the parameter; ``subject_parameters.mean()`` is the mean of
the participants' own estimates; and ``mean_z`` is the estimate itself, unconstrained.

Spread is reported only as ``sd_z``.  A single standard deviation in the model's units would
misrepresent an interval that the transform makes asymmetric near a bound.  Per-participant spread is
carried into the model's units by the delta method, where it is local to that participant's estimate
and so does not have the same problem.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from psyneulink._typing import Any, Dict, Tuple

__all__ = ["HierarchicalPECResults"]


@dataclass
class HierarchicalPECResults:
    """What a hierarchical fit found.

    Attributes
    ----------

    fit_param_names : tuple
        Fitted parameters, in the order used by every array and frame here.

    subject_labels : tuple
        Participant identifiers, in the order used by every per-participant array and frame.  This is
        the order they first appeared in the data.

    group_parameters : pandas.DataFrame
        One row per parameter: the group estimate and its spread.  ``mean_z`` and ``sd_z`` are in
        unconstrained units; ``value`` is that mean mapped into the model's units, and is a median
        (see the module docstring).  ``lower`` and ``upper`` restate the search range.

    subject_parameters : pandas.DataFrame
        One row per participant, one column per parameter, in the model's units.

    subject_posteriors : pandas.DataFrame
        One row per participant and parameter, with the estimate and its uncertainty in both spaces,
        and whether that participant's fit converged.

    beta, sigma : numpy.ndarray
        Group means ``(n_predictors, n_params)`` and variances ``(n_params,)``, unconstrained.

    z_hat, posterior_variance : numpy.ndarray
        Per-participant estimates and variances ``(n_subjects, n_params)``, unconstrained.

    objective : float
        Laplace marginal log-likelihood at the group estimate reported here.

    converged : bool
        Whether the group estimate stopped moving before the iteration limit.  A fit that did not
        converge is still returned, so that it can be inspected.

    subject_converged : numpy.ndarray
        Per participant, whether their own fit converged.  Participants that did not still
        contributed to the group estimate.

    em_history : pandas.DataFrame
        One row per iteration.  Each pairs the objective with the group estimate that produced it,
        so the two can be read side by side.

    settings, transform_metadata : dict
        What the fit was asked to do, and the transform it used, recorded so a result can be
        interpreted without the code that produced it.
    """

    fit_param_names: Tuple[str, ...]
    subject_labels: Tuple[Any, ...]
    predictor_names: Tuple[str, ...]

    group_parameters: pd.DataFrame
    subject_parameters: pd.DataFrame
    subject_posteriors: pd.DataFrame

    beta: np.ndarray
    sigma: np.ndarray
    z_hat: np.ndarray
    posterior_variance: np.ndarray

    objective: float
    n_iter: int
    converged: bool
    subject_converged: np.ndarray
    em_history: pd.DataFrame

    transform_metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_em(
        cls,
        em_result,
        transform,
        fit_param_names,
        subject_labels,
        predictor_names=("intercept",),
        settings=None,
    ):
        """Build a result from an EM fit and the transform it ran in.

        Arguments
        ---------

        em_result : LaplaceEMResult
            Output of `fit_laplace_em <fit_laplace_em>`, in unconstrained units.

        transform : BoundedTransform or IdentityTransform
            The transform the fit ran in, used to express estimates in the model's units.

        fit_param_names, subject_labels : sequence
            Names for the parameter and participant axes.

        predictor_names : sequence : default ("intercept",)
            Names for the rows of `beta`.

        settings : dict : default None
            What the fit was asked to do.
        """
        names = tuple(fit_param_names)
        labels = tuple(subject_labels)
        z_hat = np.asarray(em_result.z_hat, dtype=float)
        variance = np.asarray(em_result.variance, dtype=float)
        beta = np.asarray(em_result.beta, dtype=float)
        sigma = np.asarray(em_result.sigma, dtype=float)

        theta_hat = np.vstack([transform.to_natural(z_hat[s]) for s in range(z_hat.shape[0])])
        # Delta method: a standard deviation in unconstrained units, scaled by the local slope of
        # the transform, approximates the same spread in the model's units.
        slope = np.vstack([transform.dtheta_dz(z_hat[s]) for s in range(z_hat.shape[0])])
        theta_sd = np.abs(slope) * np.sqrt(variance)

        group_value = transform.to_natural(beta[0])
        lower = getattr(transform, "lower", np.full(len(names), -np.inf))
        upper = getattr(transform, "upper", np.full(len(names), np.inf))
        group_parameters = pd.DataFrame(
            {
                "mean_z": beta[0],
                "sd_z": np.sqrt(sigma),
                "value": group_value,
                "lower": np.broadcast_to(lower, (len(names),)),
                "upper": np.broadcast_to(upper, (len(names),)),
            },
            index=pd.Index(names, name="parameter"),
        )

        subject_parameters = pd.DataFrame(
            theta_hat, index=pd.Index(labels, name="subject"), columns=list(names)
        )

        posteriors = pd.DataFrame({
            "subject": np.repeat(labels, len(names)),
            "parameter": list(names) * len(labels),
            "z_hat": z_hat.ravel(),
            "z_sd": np.sqrt(variance).ravel(),
            "theta_hat": theta_hat.ravel(),
            "theta_sd": theta_sd.ravel(),
            "converged": np.repeat(np.asarray(em_result.subject_converged, dtype=bool), len(names)),
        })

        history = pd.DataFrame([
            {
                "iter": h["iter"],
                "objective": h["objective"],
                "delta": h["delta"],
                "n_subject_failures": h["n_subject_failures"],
                **{f"beta_{n}": h["beta"][0][k] for k, n in enumerate(names)},
                **{f"sigma_{n}": h["sigma"][k] for k, n in enumerate(names)},
            }
            for h in em_result.history
        ])

        return cls(
            fit_param_names=names,
            subject_labels=labels,
            predictor_names=tuple(predictor_names),
            group_parameters=group_parameters,
            subject_parameters=subject_parameters,
            subject_posteriors=posteriors,
            beta=beta,
            sigma=sigma,
            z_hat=z_hat,
            posterior_variance=variance,
            objective=float(em_result.objective),
            n_iter=int(em_result.n_iter),
            converged=bool(em_result.converged),
            subject_converged=np.asarray(em_result.subject_converged, dtype=bool),
            em_history=history,
            transform_metadata={
                "kind": type(transform).__name__,
                "lower": np.asarray(lower).tolist(),
                "upper": np.asarray(upper).tolist(),
            },
            settings=dict(settings or {}),
        )

    def __repr__(self):
        status = "converged" if self.converged else f"did NOT converge in {self.n_iter}"
        failures = int(np.count_nonzero(~self.subject_converged))
        note = f", {failures} participant fit(s) did not converge" if failures else ""
        return (
            f"<HierarchicalPECResults: {len(self.subject_labels)} participants, "
            f"{len(self.fit_param_names)} parameters, {status} in {self.n_iter} iterations, "
            f"objective {self.objective:.4f}{note}>"
        )
