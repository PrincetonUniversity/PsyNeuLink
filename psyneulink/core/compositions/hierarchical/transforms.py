# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.


# ****************************************  Hierarchical Transforms ****************************************************

"""Parameter transforms between a bounded search range and an unconstrained space.

The search range a user gives `ParameterEstimationComposition <ParameterEstimationComposition>` is
mapped to the whole real line by a scaled logit, and back by a scaled sigmoid::

    theta = lower + (upper - lower) * sigmoid(z)
    z     = logit((theta - lower) / (upper - lower))

The group random effect is defined natively on `z` rather than pushed forward from `theta`, so no
change-of-variables Jacobian enters the objective.  Likelihoods are always evaluated on `theta`.

The search range is hard support: values outside it are not representable, and the bounds themselves
are reached only in the limit.  `dtheta_dz` is provided for the delta method.
"""

import numpy as np
from scipy.special import expit

__all__ = ["BoundedTransform", "IdentityTransform"]


class BoundedTransform:
    """Element-wise scaled logit/sigmoid between a bounded box and R^d.

    Arguments
    ---------

    lower : array-like
        Lower bound of each parameter, in the model's units.

    upper : array-like
        Upper bound of each parameter.  Must exceed `lower` element-wise.
    """

    def __init__(self, lower, upper):
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        # Same shape, and a range with something in it, or the map below is not one-to-one.
        if self.lower.shape != self.upper.shape:
            raise ValueError(
                f"lower and upper bounds must have the same shape; "
                f"got {self.lower.shape} and {self.upper.shape}"
            )
        if np.any(self.upper <= self.lower):
            bad = np.flatnonzero(np.atleast_1d(self.upper <= self.lower))
            raise ValueError(
                f"upper bounds must exceed lower bounds; violated at index/indices {bad.tolist()}"
            )
        # Held rather than recomputed: every mapping in both directions needs it.
        self.width = self.upper - self.lower

    def to_natural(self, z):
        """Map unconstrained `z` to bounded parameters `theta`."""
        return self.lower + self.width * expit(np.asarray(z, dtype=float))

    def to_unconstrained(self, theta):
        """Map bounded parameters `theta` to unconstrained `z`."""
        theta = np.asarray(theta, dtype=float)
        # Where in the range each value sits, as a fraction.
        u = (theta - self.lower) / self.width
        # Held just inside it: the bounds themselves map to infinity.
        u = np.clip(u, 1e-12, 1.0 - 1e-12)
        return np.log(u) - np.log1p(-u)

    def dtheta_dz(self, z):
        """Element-wise derivative ``dtheta/dz = width * sigmoid'(z)``.

        Used by the delta method to express a standard deviation computed in `z` in the model's own
        units.
        """
        s = expit(np.asarray(z, dtype=float))
        return self.width * s * (1.0 - s)


class IdentityTransform:
    """A transform that does nothing: ``theta == z``.

    Its bounds are infinite, so that anything reading a range off a transform reads one here too.

    For a likelihood whose parameters are already unbounded.  A fit driven through
    `ParameterEstimationComposition <ParameterEstimationComposition>` never uses this, because
    a parameter is specified there as the values to search over and is bounded by construction;
    it is for `fit_laplace_em <laplaceem>` driven directly, against a likelihood of one's own.
    """

    lower = -np.inf
    upper = np.inf

    def to_natural(self, z):
        return np.asarray(z, dtype=float)

    def to_unconstrained(self, theta):
        return np.asarray(theta, dtype=float)

    def dtheta_dz(self, z):
        return np.ones_like(np.asarray(z, dtype=float))
