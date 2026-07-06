"""Bounded <-> unconstrained parameter transforms for hierarchical PEC.

Subject parameters ``theta`` live in a bounded box ``[lower, upper]`` (the PEC search
range). The hierarchical Gaussian random effects live in the unconstrained space ``z``.
The map is an element-wise scaled logit/sigmoid:

    theta = lower + (upper - lower) * sigmoid(z)
    z     = logit((theta - lower) / (upper - lower))

The prior is placed on ``z``; likelihoods are evaluated on ``theta``. The MAP objective is
a function of ``z`` with the prior native to ``z``, so no change-of-variables Jacobian enters
the optimization. ``dtheta_dz`` is provided for delta-method conversion of posterior variances
from ``z`` to natural units.
"""

import numpy as np


def _sigmoid(z):
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))


class BoundedTransform:
    """Element-wise scaled logit/sigmoid between a bounded box and R^d."""

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        if np.any(self.upper <= self.lower):
            raise ValueError("upper bounds must exceed lower bounds")
        self.width = self.upper - self.lower

    def to_natural(self, z):
        """z (unconstrained) -> theta (bounded)."""
        z = np.asarray(z, dtype=float)
        return self.lower + self.width * _sigmoid(z)

    def to_unconstrained(self, theta):
        """theta (bounded) -> z (unconstrained)."""
        theta = np.asarray(theta, dtype=float)
        u = (theta - self.lower) / self.width
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        return np.log(u) - np.log1p(-u)

    def dtheta_dz(self, z):
        """Element-wise derivative dtheta/dz = width * sigmoid'(z)."""
        s = _sigmoid(np.asarray(z, dtype=float))
        return self.width * s * (1.0 - s)

    def log_abs_det_jacobian(self, z):
        """log|dtheta/dz| summed over dimensions."""
        return float(np.sum(np.log(self.dtheta_dz(z))))


class IdentityTransform:
    """theta == z. Used by the unconstrained toy model."""

    def to_natural(self, z):
        return np.asarray(z, dtype=float)

    def to_unconstrained(self, theta):
        return np.asarray(theta, dtype=float)

    def dtheta_dz(self, z):
        return np.ones_like(np.asarray(z, dtype=float))

    def log_abs_det_jacobian(self, z):
        return 0.0
