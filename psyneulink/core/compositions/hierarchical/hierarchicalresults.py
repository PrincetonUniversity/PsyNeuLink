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

``group_correlation`` says how the group's parameters vary together.  A fit run with a diagonal
covariance assumed they do not, so it reports the identity; only a full covariance measures it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from psyneulink._typing import Any, Dict, Tuple

__all__ = ["HierarchicalPECResults", "HierarchicalSamplingResults"]


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

    group_correlation : pandas.DataFrame
        Correlation between the group's parameters, implied by the group covariance.  The identity
        when the fit used a diagonal covariance; see the module docstring.

    beta, group_covariance : numpy.ndarray
        Group means ``(n_predictors, n_params)`` and covariance ``(n_params, n_params)``,
        unconstrained.

    z_hat : numpy.ndarray
        Per-participant estimates ``(n_subjects, n_params)``, unconstrained.

    posterior_covariance : numpy.ndarray
        Per-participant posterior covariance ``(n_subjects, n_params, n_params)``, unconstrained.

    sigma, posterior_variance : numpy.ndarray
        Diagonals of the two covariances: per-parameter group variance ``(n_params,)`` and
        per-participant variance ``(n_subjects, n_params)``.

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
    group_correlation: pd.DataFrame

    beta: np.ndarray
    group_covariance: np.ndarray
    z_hat: np.ndarray
    posterior_covariance: np.ndarray

    objective: float
    n_iter: int
    converged: bool
    subject_converged: np.ndarray
    em_history: pd.DataFrame

    transform_metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def sigma(self):
        """Per-parameter group variance: the diagonal of `group_covariance`."""
        return np.diag(self.group_covariance).copy()

    @property
    def posterior_variance(self):
        """Per-participant, per-parameter posterior variance."""
        return np.diagonal(self.posterior_covariance, axis1=1, axis2=2).copy()

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
        posterior_covariance = np.asarray(em_result.posterior_covariance, dtype=float)
        variance = np.diagonal(posterior_covariance, axis1=1, axis2=2)
        beta = np.asarray(em_result.beta, dtype=float)
        group_covariance = np.asarray(em_result.covariance, dtype=float)
        sigma = np.diag(group_covariance)

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

        # Correlation from the covariance. A diagonal fit gives the identity, which says the fit
        # assumed the parameters were independent rather than that it found them to be.
        scale = np.sqrt(np.outer(sigma, sigma))
        group_correlation = pd.DataFrame(
            group_covariance / scale,
            index=pd.Index(names, name="parameter"),
            columns=list(names),
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
                **{f"sigma_{n}": h["covariance"][k, k] for k, n in enumerate(names)},
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
            group_correlation=group_correlation,
            beta=beta,
            group_covariance=group_covariance,
            z_hat=z_hat,
            posterior_covariance=posterior_covariance,
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
        status = "converged" if self.converged else "stopped at the iteration limit"
        failures = int(np.count_nonzero(~self.subject_converged))
        note = f", {failures} participant fit(s) did not converge" if failures else ""
        return (
            f"<HierarchicalPECResults: {len(self.subject_labels)} participants, "
            f"{len(self.fit_param_names)} parameters, {status} after {self.n_iter} "
            f"iterations, objective {self.objective:.4f}{note}>"
        )


@dataclass
class HierarchicalSamplingResults:
    """What a sampled hierarchical fit found.

    Where `HierarchicalPECResults` reports an estimate and a Gaussian width around it, this
    reports the draws themselves.  Spread comes from the draws rather than from an assumed shape,
    so an interval near a bound, or one for a parameter that trades off against another, is as
    wide as the posterior actually is.

    Attributes
    ----------

    fit_param_names, subject_labels, predictor_names : tuple
        The parameter, participant and predictor axes, in the order every array here uses.

    group_parameters : pandas.DataFrame
        One row per parameter. ``mean_z`` and ``sd_z`` summarize the group mean in unconstrained
        units; ``value`` is the posterior median in the model's units, and ``lower_95`` and
        ``upper_95`` the interval the middle 95% of the draws fall in. ``lower`` and ``upper``
        restate the search range, as they do for an EM fit.

    subject_parameters : pandas.DataFrame
        One row per participant, one column per parameter: the posterior mean in the model's
        units.

    subject_posteriors : pandas.DataFrame
        One row per participant and parameter, with the estimate and its interval in both spaces.

    group_correlation : pandas.DataFrame
        Posterior mean correlation between the group's parameters. The identity when the fit used
        a diagonal covariance, which records the assumption rather than a measurement.

    convergence : pandas.DataFrame
        Per group-level quantity, ``r_hat`` and ``ess``: whether the chains agree, and how many
        independent draws they are worth. A fit whose ``r_hat`` exceeds about 1.01, or whose
        ``ess`` is in the low hundreds, has not yet described the posterior.

    diagnostics : NUTSDiagnostics
        What the sampler did. Read ``diagnostics.warnings`` before the estimates.

    group_draws, covariance_draws, subject_draws : numpy.ndarray
        The draws themselves: ``(chains, draws, n_predictors, n_params)`` group means in
        unconstrained units, ``(chains, draws, n_params, n_params)`` group covariances, and
        ``(chains, draws, n_subjects, n_params)`` participant parameters in the model's units.

    settings, transform_metadata : dict
        What the fit was asked to do, and the transform it used.
    """

    fit_param_names: Tuple[str, ...]
    subject_labels: Tuple[Any, ...]
    predictor_names: Tuple[str, ...]

    group_parameters: pd.DataFrame
    subject_parameters: pd.DataFrame
    subject_posteriors: pd.DataFrame
    group_correlation: pd.DataFrame
    convergence: pd.DataFrame

    diagnostics: Any
    group_draws: np.ndarray
    covariance_draws: np.ndarray
    subject_draws: np.ndarray

    transform_metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def group_covariance(self):
        """Posterior mean group covariance ``(n_params, n_params)``, unconstrained."""
        return self.covariance_draws.reshape(-1, *self.covariance_draws.shape[2:]).mean(axis=0)

    @property
    def converged(self):
        """Whether the chains agree and the sampler had nothing to report.

        A fit that is not converged is still returned, so that it can be inspected; its estimates
        should not be read as describing the posterior.
        """
        return bool(
            self.convergence["r_hat"].max() < 1.01
            and not self.diagnostics.warnings
        )

    @classmethod
    def from_draws(cls, draws, diagnostics, posterior, fit_param_names, subject_labels,
                   predictor_names=("intercept",), settings=None):
        """Build a result from the sampler's output and the posterior it sampled.

        Arguments
        ---------

        draws : numpy.ndarray
            ``(chains, draws, n_sampler_params)``, as `run_nuts` returns.

        diagnostics : NUTSDiagnostics
            What the sampler did, reported alongside the estimates.

        posterior : HierarchicalNeuralPosterior
            The posterior that was sampled, used to read the draws back into group and
            participant quantities.

        fit_param_names, subject_labels : sequence
            Names for the parameter and participant axes.

        predictor_names : sequence : default ("intercept",)
            Names for the rows of the group means.

        settings : dict : default None
            What the fit was asked to do.
        """
        import torch

        names = tuple(fit_param_names)
        labels = tuple(subject_labels)
        n_chains, n_draws = draws.shape[0], draws.shape[1]
        flat = draws.reshape(-1, draws.shape[-1])

        # Each draw is read back through the posterior that produced it, rather than by
        # re-deriving the layout here, so the two cannot drift apart.
        group, covariance, subject = [], [], []
        for row in flat:
            position = torch.as_tensor(row, dtype=torch.float64)
            beta, _, _, _ = posterior.unpack(position)
            group.append(beta.numpy().copy())
            covariance.append(posterior.group_covariance(position).numpy())
            subject.append(posterior.to_natural(posterior.subject_z(position)).numpy())

        shape = (n_chains, n_draws)
        group_draws = np.stack(group).reshape(*shape, posterior.n_predictors, posterior.n_params)
        covariance_draws = np.stack(covariance).reshape(*shape, len(names), len(names))
        subject_draws = np.stack(subject).reshape(*shape, len(labels), len(names))

        mean_z = group_draws[:, :, 0, :].reshape(-1, len(names))
        group_value = posterior.to_natural(torch.as_tensor(mean_z)).numpy()
        lower = np.asarray(posterior.lower.numpy())
        upper = lower + np.asarray(posterior.width.numpy())
        group_parameters = pd.DataFrame(
            {
                "mean_z": mean_z.mean(axis=0),
                "sd_z": mean_z.std(axis=0, ddof=1),
                "value": np.median(group_value, axis=0),
                "lower_95": np.quantile(group_value, 0.025, axis=0),
                "upper_95": np.quantile(group_value, 0.975, axis=0),
                "lower": lower,
                "upper": upper,
            },
            index=pd.Index(names, name="parameter"),
        )

        flat_subject = subject_draws.reshape(-1, len(labels), len(names))
        subject_mean = flat_subject.mean(axis=0)
        subject_parameters = pd.DataFrame(
            subject_mean, index=pd.Index(labels, name="subject"), columns=list(names)
        )

        z_draws = np.log((flat_subject - lower) / (upper - flat_subject))
        posteriors = pd.DataFrame({
            "subject": np.repeat(labels, len(names)),
            "parameter": list(names) * len(labels),
            "z_mean": z_draws.mean(axis=0).ravel(),
            "z_sd": z_draws.std(axis=0, ddof=1).ravel(),
            "theta_mean": subject_mean.ravel(),
            "theta_lower_95": np.quantile(flat_subject, 0.025, axis=0).ravel(),
            "theta_upper_95": np.quantile(flat_subject, 0.975, axis=0).ravel(),
        })

        flat_covariance = covariance_draws.reshape(-1, len(names), len(names))
        scale = np.sqrt(np.einsum("dii->di", flat_covariance))
        correlation = flat_covariance / (scale[:, :, None] * scale[:, None, :])
        group_correlation = pd.DataFrame(
            correlation.mean(axis=0), index=pd.Index(names, name="parameter"),
            columns=list(names),
        )

        convergence = _group_convergence(draws, posterior, names, predictor_names)

        return cls(
            fit_param_names=names,
            subject_labels=labels,
            predictor_names=tuple(predictor_names),
            group_parameters=group_parameters,
            subject_parameters=subject_parameters,
            subject_posteriors=posteriors,
            group_correlation=group_correlation,
            convergence=convergence,
            diagnostics=diagnostics,
            group_draws=group_draws,
            covariance_draws=covariance_draws,
            subject_draws=subject_draws,
            transform_metadata={
                "kind": "BoundedTransform",
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            },
            settings=dict(settings or {}),
        )

    def __repr__(self):
        status = "converged" if self.converged else "NOT converged"
        chains, draws = self.subject_draws.shape[0], self.subject_draws.shape[1]
        return (
            f"<HierarchicalSamplingResults: {len(self.subject_labels)} participants, "
            f"{len(self.fit_param_names)} parameters, {chains} chains of {draws} draws, "
            f"{status} (worst r_hat {self.convergence['r_hat'].max():.3f}, "
            f"{int(self.diagnostics.divergences.sum())} divergence(s))>"
        )


def _group_convergence(draws, posterior, names, predictor_names):
    """R-hat and effective sample size for each group-level quantity.

    Only the group quantities: there is one convergence row per parameter the group model has,
    not one per participant, since a participant's own parameters are read off the group draws
    rather than sampled separately in any meaningful sense.
    """
    from psyneulink.core.compositions.hierarchical.nuts import (
        effective_sample_size,
        potential_scale_reduction,
    )

    group_slice = draws[:, :, :posterior._off_end]
    r_hat = potential_scale_reduction(group_slice)
    ess = effective_sample_size(group_slice)

    labels = [f"{predictor}.{name}" for predictor in predictor_names for name in names]
    labels += [f"log_scale.{name}" for name in names]
    labels += [
        f"covariance_factor[{row},{col}]"
        for row in range(len(names)) for col in range(row)
    ]
    return pd.DataFrame(
        {"r_hat": r_hat, "ess": ess},
        index=pd.Index(labels[:len(r_hat)], name="quantity"),
    )
