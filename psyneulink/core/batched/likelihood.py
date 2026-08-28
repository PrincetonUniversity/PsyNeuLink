"""Histogram-based likelihood for batched stochastic simulations (GPU-friendly).

This is the batched-compiler counterpart to
:func:`psyneulink.core.components.functions.nonstateful.fitfunctions.simulation_likelihood`,
which estimates the likelihood of experimental data under a model by a *kernel*
density estimate (fastKDE) computed on CPU.  For the batched PEC path we instead
use a plain **histogram** density estimate that can run entirely in Torch on the
same device the simulation outcomes were produced on (no host round-trip, no
fastKDE dependency).  It is deliberately simpler than a KDE — the accuracy knob
is the number of ``bins`` — but it is fully vectorized over parameter sets,
subjects, and trials, so it scales with the batched simulator.

Data model
----------
The batched simulator returns outcomes shaped
``[parameter_set, subject, trial, estimate, outcome]``.  This module treats every
leading axis in front of ``(trial, estimate, outcome)`` as an independent
likelihood evaluation (a "lane"), so it accepts any ``[*lanes, trial, estimate,
outcome]`` tensor and returns one likelihood per lane.

Each ``outcome`` vector mixes *categorical* dimensions (e.g. a DDM decision, which
threshold was crossed) and *continuous* dimensions (e.g. response time).  For a
given experimental trial the likelihood of the observed point is

``density = (# sims matching the observed category AND landing in the observed
continuous bin) / (num_sims * bin_volume)``

which is exactly the histogram analogue of the KDE scaling used by
``simulation_likelihood`` (the per-category pdf, scaled by that category's share
of the simulations).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "histogram_likelihood",
    "histogram_log_likelihood",
]

# Densities of exactly zero are floored to this so ``log`` does not explode; this
# mirrors the ``ZERO_PROB`` used by ``simulation_likelihood``.
ZERO_PROB = 1e-10


def _as_categorical_mask(categorical_dims, num_outcome_dims: int) -> np.ndarray:
    """Normalize ``categorical_dims`` to a boolean mask of length ``num_outcome_dims``."""
    if categorical_dims is None:
        return np.zeros(num_outcome_dims, dtype=bool)
    arr = np.asarray(list(categorical_dims))
    if arr.dtype == bool:
        if arr.shape[0] != num_outcome_dims:
            raise ValueError(
                f"categorical_dims boolean mask has length {arr.shape[0]}, "
                f"expected {num_outcome_dims} (one per outcome dimension)."
            )
        return arr
    mask = np.zeros(num_outcome_dims, dtype=bool)
    mask[arr.astype(int)] = True
    return mask


def histogram_likelihood(
    sim_outcomes,
    exp_data,
    categorical_dims=None,
    *,
    bins: int = 100,
    bin_range: Sequence | None = None,
    smoothing_sigma: float = 0.0,
    pseudocount: float = 0.0,
    categorical_cardinalities: Sequence[int] | None = None,
    device: str | None = None,
):
    """Per-trial histogram likelihood of ``exp_data`` under ``sim_outcomes``.

    This is the direct analogue of
    :func:`~psyneulink.core.components.functions.nonstateful.fitfunctions.simulation_likelihood`
    (same return semantics), computed with a histogram instead of a KDE and run in
    Torch so it can execute on GPU.

    Parameters
    ----------
    sim_outcomes : array-like or ``torch.Tensor``
        Simulation outcomes shaped ``[*lanes, trial, estimate, outcome]``.  A
        bare ``[trial, estimate, outcome]`` array (one lane) is also accepted.
        If a ``torch.Tensor`` is passed it is used in place (so GPU tensors stay
        on the GPU); otherwise it is converted with ``torch.as_tensor``.
    exp_data : array-like
        Experimental data shaped ``[trial, outcome]``.
    categorical_dims : bool mask or index sequence, optional
        Which outcome dimensions are categorical.  A boolean mask must have one
        entry per outcome dimension; an integer sequence lists the categorical
        indices.  ``None`` means all dimensions are continuous.
    bins : int
        Number of histogram bins per continuous dimension.
    bin_range : sequence of (lo, hi), optional
        Explicit ``(lo, hi)`` range per continuous dimension.  If omitted, the
        range is anchored to the **experimental** data (min/max plus a small
        margin) along each dimension — not the simulated data — so the bins stay
        identical across parameter sets and ``num_estimates`` (see ``_bin_edges``).
    smoothing_sigma : float
        Standard deviation, in histogram-bin units, of a discrete Gaussian
        kernel applied independently along each continuous dimension.  The
        kernel is truncated at three standard deviations and renormalized at
        the histogram boundaries.  ``0`` (the default) retains exact-bin
        matching.
    pseudocount : float
        Symmetric Dirichlet pseudocount added to every joint categorical and
        continuous histogram cell.  ``0`` (the default) retains the original
        estimator.  A positive value prevents sparse but plausible cells from
        jumping directly to ``ZERO_PROB``.
    categorical_cardinalities : sequence of int, optional
        Number of possible values for each categorical outcome dimension, in
        categorical-dimension order.  This is used only to normalize a positive
        ``pseudocount``.  By default, cardinalities are inferred from the unique
        values present in ``exp_data``.  Specify them when the experimental data
        do not contain every possible category.
    device : str, optional
        Torch device to run on (e.g. ``"cuda"``).  Defaults to the device of
        ``sim_outcomes`` if it is already a tensor, else CPU.

    Returns
    -------
    numpy.ndarray
        Per-trial likelihood shaped ``[*lanes, trial]`` (densities, floored at
        ``ZERO_PROB``).  For a single-lane input the leading axis is dropped, so
        the shape is ``[trial]`` — matching ``simulation_likelihood``.
    """
    import torch

    if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f"bins must be a positive integer, got {bins!r}.")
    if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
        raise ValueError(
            f"smoothing_sigma must be finite and nonnegative, got {smoothing_sigma!r}."
        )
    if not np.isfinite(pseudocount) or pseudocount < 0:
        raise ValueError(
            f"pseudocount must be finite and nonnegative, got {pseudocount!r}."
        )

    sim, squeezed = _to_lane_tensor(sim_outcomes, torch, device)
    device = sim.device
    exp_array = np.asarray(exp_data, dtype=float)
    exp = torch.as_tensor(exp_array, dtype=sim.dtype, device=device)
    if exp.ndim != 2:
        raise ValueError(f"exp_data must be 2D [trial, outcome], got shape {tuple(exp.shape)}.")

    n_lanes, n_trials, n_sims, n_out = sim.shape
    if exp.shape[0] != n_trials:
        raise ValueError(
            f"exp_data has {exp.shape[0]} trials but sim_outcomes has {n_trials}."
        )
    if exp.shape[1] != n_out:
        raise ValueError(
            f"exp_data has {exp.shape[1]} outcome dims but sim_outcomes has {n_out}."
        )

    cat_mask = _as_categorical_mask(categorical_dims, n_out)
    cat_idx = torch.as_tensor(np.flatnonzero(cat_mask), dtype=torch.long, device=device)
    con_idx = torch.as_tensor(np.flatnonzero(~cat_mask), dtype=torch.long, device=device)

    # --- categorical match: sims whose categorical values equal the observed ones
    if cat_idx.numel() > 0:
        sim_cat = sim.index_select(-1, cat_idx)              # [L, T, S, Ccat]
        exp_cat = exp.index_select(-1, cat_idx)              # [T, Ccat]
        match_cat = torch.isclose(
            sim_cat, exp_cat[None, :, None, :], atol=1e-6, rtol=0.0
        ).all(dim=-1)                                         # [L, T, S]
    else:
        match_cat = torch.ones((n_lanes, n_trials, n_sims), dtype=torch.bool, device=device)

    # --- continuous match: sims landing in or near the observed histogram bin
    if con_idx.numel() > 0:
        sim_con = sim.index_select(-1, con_idx)              # [L, T, S, Ccon]
        exp_con = exp.index_select(-1, con_idx)              # [T, Ccon]
        edges = _bin_edges(sim_con, exp_con, bins, bin_range, torch)  # list of [bins+1]
        bin_volume = torch.tensor(1.0, dtype=sim.dtype, device=device)
        if smoothing_sigma == 0:
            match_con = torch.ones(
                (n_lanes, n_trials, n_sims), dtype=torch.bool, device=device
            )
            for d in range(con_idx.numel()):
                interior = edges[d][1:-1]
                sim_bin = torch.bucketize(sim_con[..., d], interior)
                exp_bin = torch.bucketize(exp_con[:, d], interior)
                match_con &= sim_bin == exp_bin[None, :, None]
                bin_volume = bin_volume * (edges[d][1] - edges[d][0])
            counts = (match_cat & match_con).sum(dim=2).to(sim.dtype)
        else:
            # Evaluate the separable, discrete Gaussian-smoothed histogram at
            # the observed bin without materializing a full (potentially
            # multidimensional) histogram.  Renormalizing the kernel for each
            # observed bin prevents probability mass from leaking through an
            # edge of the finite histogram range.
            radius = max(1, int(np.ceil(3.0 * smoothing_sigma)))
            offsets = torch.arange(-radius, radius + 1, device=device)
            kernel = torch.exp(
                -0.5 * (offsets.to(sim.dtype) / float(smoothing_sigma)) ** 2
            )
            weights = match_cat.to(sim.dtype)
            for d in range(con_idx.numel()):
                interior = edges[d][1:-1]
                sim_bin = torch.bucketize(sim_con[..., d], interior)
                exp_bin = torch.bucketize(exp_con[:, d], interior)
                delta = sim_bin - exp_bin[None, :, None]
                in_radius = delta.abs() <= radius

                valid_offsets = (
                    (exp_bin[:, None] + offsets[None, :] >= 0)
                    & (exp_bin[:, None] + offsets[None, :] < bins)
                )
                normalizer = (valid_offsets.to(sim.dtype) * kernel[None, :]).sum(dim=1)
                dim_weights = torch.exp(
                    -0.5 * (delta.to(sim.dtype) / float(smoothing_sigma)) ** 2
                ) / normalizer[None, :, None]
                weights = weights * torch.where(
                    in_radius, dim_weights, torch.zeros_like(dim_weights)
                )
                bin_volume = bin_volume * (edges[d][1] - edges[d][0])
            counts = weights.sum(dim=2)
    else:
        bin_volume = torch.tensor(1.0, dtype=sim.dtype, device=device)
        counts = match_cat.sum(dim=2).to(sim.dtype)

    if pseudocount > 0:
        cardinalities = _categorical_cardinalities(
            exp_array,
            cat_mask,
            categorical_cardinalities,
        )
        joint_bin_count = float(bins ** int(con_idx.numel()) * np.prod(cardinalities))
        density = (counts + pseudocount) / (
            (float(n_sims) + pseudocount * joint_bin_count) * bin_volume
        )
    else:
        density = counts / (float(n_sims) * bin_volume)
    density = torch.clamp(density, min=ZERO_PROB)

    like = density.cpu().numpy()
    if squeezed:
        return like[0]
    return like


def histogram_log_likelihood(
    sim_outcomes,
    exp_data,
    categorical_dims=None,
    *,
    bins: int = 100,
    bin_range: Sequence | None = None,
    smoothing_sigma: float = 0.0,
    pseudocount: float = 0.0,
    categorical_cardinalities: Sequence[int] | None = None,
    include_mask: np.ndarray | None = None,
    device: str | None = None,
) -> float | np.ndarray:
    """Total histogram log-likelihood per lane (``sum_t log p(exp_t)``).

    Convenience wrapper over :func:`histogram_likelihood` that applies the log,
    an optional per-trial ``include_mask``, and sums over trials.  This is the
    quantity a maximum-likelihood PEC objective maximizes.

    Returns a scalar ``float`` for a single-lane input, else a ``[*lanes]``
    array of per-lane total log-likelihoods.
    """
    like = histogram_likelihood(
        sim_outcomes,
        exp_data,
        categorical_dims,
        bins=bins,
        bin_range=bin_range,
        smoothing_sigma=smoothing_sigma,
        pseudocount=pseudocount,
        categorical_cardinalities=categorical_cardinalities,
        device=device,
    )
    log_like = np.log(like)
    if include_mask is not None:
        mask = np.asarray(include_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != log_like.shape[-1]:
            raise ValueError(
                f"include_mask has length {mask.shape[0]} but there are "
                f"{log_like.shape[-1]} trials."
            )
        log_like = log_like[..., mask]
    total = log_like.sum(axis=-1)
    if total.ndim == 0:
        return float(total)
    return total


def _categorical_cardinalities(exp_data, cat_mask, specified):
    """Validate or infer category counts used for Dirichlet normalization."""
    cat_indices = np.flatnonzero(cat_mask)
    if specified is None:
        cardinalities = [len(np.unique(exp_data[:, d])) for d in cat_indices]
    else:
        cardinalities = list(specified)
        if len(cardinalities) != len(cat_indices):
            raise ValueError(
                "categorical_cardinalities has length "
                f"{len(cardinalities)}, expected {len(cat_indices)} "
                "(one per categorical outcome dimension)."
            )
    for value in cardinalities:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < 1
        ):
            raise ValueError(
                "categorical_cardinalities values must be positive integers; "
                f"got {cardinalities!r}."
            )
    return cardinalities


def _to_lane_tensor(sim_outcomes, torch, device):
    """Return ``sim_outcomes`` as a 4D ``[lane, trial, estimate, outcome]`` tensor.

    A 3D ``[trial, estimate, outcome]`` input is treated as a single lane; the
    boolean second return value records whether a leading lane axis was added so
    the caller can squeeze it back out.
    """
    if isinstance(sim_outcomes, torch.Tensor):
        sim = sim_outcomes
        if device is not None:
            sim = sim.to(device)
        sim = sim.to(torch.float32) if not sim.is_floating_point() else sim
    else:
        sim = torch.as_tensor(np.asarray(sim_outcomes, dtype=float), dtype=torch.float32)
        if device is not None:
            sim = sim.to(device)

    if sim.ndim < 3:
        raise ValueError(
            "sim_outcomes must have at least 3 dims [trial, estimate, outcome], "
            f"got shape {tuple(sim.shape)}."
        )
    squeezed = False
    if sim.ndim == 3:
        sim = sim[None]
        squeezed = True
    elif sim.ndim > 4:
        # Flatten any number of leading lane axes into one.
        sim = sim.reshape(-1, *sim.shape[-3:])
    return sim, squeezed


def _bin_edges(sim_con, exp_con, bins: int, bin_range, torch):
    """Per-continuous-dimension histogram edges (``[bins + 1]`` each).

    When ``bin_range`` is not given, the range is anchored to the **experimental**
    data only (with a small margin), *not* the simulated data.  The likelihood is
    evaluated at the experimental points, so those are the only values that need to
    fall in interior bins; simulated values outside the range clamp into the empty
    edge bins.  Anchoring to the data keeps the bins identical across every
    parameter set and every ``num_estimates`` — otherwise the bins (and hence the
    likelihood surface) would drift with the simulated data's spread, adding noise
    to the optimizer's objective.
    """
    n_dims = sim_con.shape[-1]
    edges = []
    for d in range(n_dims):
        if bin_range is not None:
            lo, hi = bin_range[d]
        else:
            lo = float(exp_con[:, d].min().item())
            hi = float(exp_con[:, d].max().item())
            # Pad so the experimental extremes sit strictly inside interior bins
            # (out-of-range simulations pile into the edge bins, which must not
            # coincide with an evaluated data point).
            margin = (hi - lo) * 0.02 if hi > lo else 1.0
            lo -= margin
            hi += margin
        if not hi > lo:
            # Degenerate range (all values identical): make a unit-width bin around it.
            hi = lo + 1.0
        # Nudge the top edge up so the max value falls strictly inside the last bin.
        hi = hi + (hi - lo) * 1e-6
        edges.append(torch.linspace(lo, hi, bins + 1, dtype=sim_con.dtype, device=sim_con.device))
    return edges
