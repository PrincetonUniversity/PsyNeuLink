"""Tests for hierarchical parameter estimation.

Layered fastest first.  This layer covers the parameter transforms and needs neither PsyNeuLink nor
a cluster, so it runs under a plain ``[dev]`` install.
"""

import numpy as np
import pytest

from psyneulink.core.compositions.hierarchical.transforms import (
    BoundedTransform,
    IdentityTransform,
)


# ===========================================================================
# Transforms
# ===========================================================================
def test_bounded_transform_roundtrip():
    t = BoundedTransform(lower=[0.0, -2.0], upper=[1.0, 3.0])
    theta = np.array([0.2, 1.5])
    z = t.to_unconstrained(theta)
    assert np.allclose(t.to_natural(z), theta, atol=1e-10)


def test_bounded_transform_respects_bounds():
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    # Moderate z stays strictly interior; extreme z saturates within the closed box.
    interior = t.to_natural(np.array([-20.0, 0.0, 20.0]))
    assert np.all(interior > 0.0) and np.all(interior < 1.0)
    saturated = t.to_natural(np.array([-50.0, 50.0]))
    assert np.all(saturated >= 0.0) and np.all(saturated <= 1.0)
    assert saturated[0] < 1e-6 and saturated[1] > 1.0 - 1e-6


def test_bounded_transform_saturates_without_overflow():
    # Saturating inputs must reach the bounds without overflowing on the way.
    t = BoundedTransform(lower=[0.0], upper=[1.0])
    with np.errstate(over="raise", under="raise"):
        out = t.to_natural(np.array([-800.0, 800.0]))
    assert np.all(np.isfinite(out))
    assert out[0] == 0.0 and out[1] == 1.0


def test_dtheta_dz_matches_numerical():
    t = BoundedTransform(lower=[0.0, 1.0], upper=[2.0, 5.0])
    z = np.array([0.3, -0.7])
    h = 1e-6
    num = np.array([
        (t.to_natural(z + h * e)[k] - t.to_natural(z - h * e)[k]) / (2 * h)
        for k, e in enumerate(np.eye(2))
    ])
    assert np.allclose(t.dtheta_dz(z), num, rtol=1e-5)


def test_bounded_transform_rejects_bad_bounds():
    with pytest.raises(ValueError, match="upper bounds must exceed lower bounds"):
        BoundedTransform(lower=[0.0, 2.0], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="same shape"):
        BoundedTransform(lower=[0.0, 1.0], upper=[1.0])


def test_identity_transform_is_the_identity():
    t = IdentityTransform()
    theta = np.array([-3.0, 0.0, 2.5])
    assert np.allclose(t.to_natural(theta), theta)
    assert np.allclose(t.to_unconstrained(theta), theta)
    assert np.allclose(t.dtheta_dz(theta), np.ones_like(theta))
