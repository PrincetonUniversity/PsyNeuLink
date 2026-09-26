"""Exact additive-output grid scoring from declared finite-support counts.

This does not infer that any model parameter is a passive readout. It scores an
explicit FP32 addition to the already reduced numeric output, preserving the
ordinary histogram's finite range, edge membership, smoothing and prior.
"""

import numpy as np

from psyneulink.core.batched.likelihood import (
    ZERO_PROB, _as_categorical_mask, _bin_edges, _categorical_cardinalities,
)


class ShiftedHistogramScorer:
    def __init__(self, support, shifts, data, categorical_dims, *, bins=100, bin_range=None,
                 smoothing_sigma=0., categorical_cardinalities=None, device="cuda"):
        import torch

        if isinstance(bins, bool) or not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError("bins must be a positive integer")
        if not np.isfinite(smoothing_sigma) or smoothing_sigma < 0:
            raise ValueError("smoothing_sigma must be finite and nonnegative")
        self.support = np.asarray(support, dtype=np.float32)
        shifts = np.asarray(shifts, dtype=np.float64)
        if self.support.ndim != 1 or not len(self.support) or not np.isfinite(self.support).all() or np.any(np.diff(self.support) <= 0):
            raise ValueError("support must be strictly increasing finite FP32 values")
        if shifts.ndim != 1 or not len(shifts) or not np.isfinite(shifts).all() or np.any(np.diff(shifts) <= 0):
            raise ValueError("shifts must be strictly increasing finite values")
        data = np.asarray(data, dtype=float)
        if data.ndim != 2 or not np.isfinite(data).all():
            raise ValueError("data must be a finite trial-by-outcome array")
        categorical = _as_categorical_mask(categorical_dims, data.shape[1])
        if np.count_nonzero(~categorical) != 1:
            raise ValueError("Shift scoring requires exactly one numeric output")
        observed = torch.tensor(data[:, ~categorical], dtype=torch.float32)
        edge = _bin_edges(observed, observed, bins, bin_range, torch)[0].numpy()
        self.volume = float(edge[1] - edge[0])
        self.joint_bins = bins * int(np.prod(_categorical_cardinalities(data, categorical, categorical_cardinalities)))
        rt = self.support[None, :] + shifts.astype(np.float32)[:, None]
        signature = np.searchsorted(edge[1:-1], rt, side="left").astype(np.int32)
        signature[(rt < edge[0]) | (rt > edge[-1])] = -1
        # Exact map equivalence over ALL declared values, including finite-range
        # exclusion. Keep the lowest original grid value as a deterministic tie.
        _, representatives = np.unique(signature, axis=0, return_index=True)
        representatives.sort()
        self.shifts = shifts[representatives]
        self.grid_size = len(shifts)
        signature = signature[representatives]
        radius = (min(bins - 1, max(1, int(np.ceil(3 * min(float(smoothing_sigma), bins)))))
                  if smoothing_sigma else 0)
        observed_values = observed.numpy()[:, 0]
        observed_bin = np.searchsorted(edge[1:-1], observed_values, side="left")
        offsets = np.arange(-radius, radius + 1)
        # Use Torch's FP32 weights to match the production scorer's arithmetic.
        weights = (torch.exp(-.5 * (torch.tensor(offsets).float() / smoothing_sigma) ** 2).numpy()
                   if radius else np.ones(1, dtype=np.float32))
        neighbors = np.arange(bins)[:, None] + offsets
        norms = (weights[None, :] * ((neighbors >= 0) & (neighbors < bins))).sum(-1)
        lookup = []
        maximum = 1
        for row in signature:
            by_bin = []
            for b in range(bins):
                indices = np.flatnonzero((row >= 0) & (np.abs(row - b) <= radius))
                w = weights[row[indices] - b + radius] / norms[b]
                by_bin.append((indices, w))
                maximum = max(maximum, len(indices))
            lookup.append(by_bin)
        index = np.zeros((len(data), len(signature), maximum), dtype=np.int64)
        weight = np.zeros(index.shape, dtype=np.float32)
        for t, b in enumerate(observed_bin):
            if not edge[0] <= observed_values[t] <= edge[-1]:
                continue
            for k, by_bin in enumerate(lookup):
                positions, w = by_bin[b]
                index[t, k, :len(positions)] = positions
                weight[t, k, :len(positions)] = w
        self.indices = torch.tensor(index, device=device)
        self.weights = torch.tensor(weight, device=device)
        self.support_tensor = torch.tensor(self.support, device=device)

    def densities(self, reduced, *, pseudocount=0.):
        """Return [candidate, subject, shift, trial] densities, including invalid NaNs."""
        import torch

        if not np.isfinite(pseudocount) or pseudocount < 0:
            raise ValueError("pseudocount must be finite and nonnegative")
        counts = reduced.counts
        if (counts.ndim != 4 or counts.shape[-2:] != (len(self.indices), len(self.support))
                or reduced.support.device != self.support_tensor.device
                or not torch.equal(reduced.support, self.support_tensor)):
            raise ValueError("Counts do not match the scorer's support, trials, or device")
        c, s, t, v = counts.shape
        k, w = self.indices.shape[1:]
        selected = torch.gather(counts[..., None, :].expand(c, s, t, k, v), -1,
                                self.indices[None, None].expand(c, s, t, k, w))
        weighted = (selected.float() * self.weights).sum(-1)
        density = (weighted + pseudocount) / ((reduced.num_estimates + pseudocount * self.joint_bins) * self.volume)
        result = torch.clamp(density, min=ZERO_PROB).permute(0, 1, 3, 2).cpu().numpy()
        result[~reduced.valid_candidates] = np.nan
        return result
