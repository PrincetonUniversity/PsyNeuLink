"""Interpret PEC parameter specifications independently of the search algorithm.

Explicit specifications distinguish continuous bounds, regular grids, and exact
choices. Legacy arrays retain their historical interpretation: bounds for
continuous fitting, and a regular grid for Optuna.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from optuna.distributions import CategoricalDistribution, FloatDistribution

__all__ = []


@dataclass(frozen=True)
class _ParameterSpace:
    lower: float
    upper: float
    step: float | None = None
    values: tuple[float, ...] | None = None
    samples: np.ndarray | None = None

    @property
    def bounds(self):
        return self.lower, self.upper

    def continuous_bounds(self):
        if self.step is not None or self.values is not None:
            raise ValueError(
                "Explicit grids and choices require an Optuna optimizer; "
                "use {'bounds': (lower, upper)} for continuous fitting"
            )
        return self.bounds

    def optuna_distribution(self):
        if self.values is not None:
            return CategoricalDistribution(self.values)
        step = _legacy_step(self.samples) if self.samples is not None else self.step
        return FloatDistribution(self.lower, self.upper, step=step)


def _numeric_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Parameter values must be a nonempty, one-dimensional sequence of finite numbers")
    return values


def _parse_parameter_space(spec):
    if isinstance(spec, Mapping):
        if set(spec) == {'values'}:
            values = _numeric_values(spec['values'])
            return _ParameterSpace(float(min(values)), float(max(values)), values=tuple(values.tolist()))
        if set(spec) not in ({'bounds'}, {'bounds', 'step'}):
            raise ValueError("Specify either {'bounds': (lower, upper), 'step': optional_step} or {'values': [...]}")
        bounds = _numeric_values(spec['bounds'])
        if len(bounds) != 2 or bounds[0] > bounds[1]:
            raise ValueError("Parameter bounds must contain (lower, upper) with lower <= upper")
        step = spec.get('step')
        if step is not None:
            step = float(step)
            if not np.isfinite(step) or step <= 0:
                raise ValueError("Parameter step must be finite and positive")
        return _ParameterSpace(float(bounds[0]), float(bounds[1]), step=step)

    values = _numeric_values(spec)
    return _ParameterSpace(float(min(values)), float(max(values)), samples=np.asarray(spec))


def _legacy_step(samples):
    """Infer a regular grid only for the legacy Optuna array specification."""
    precision = np.finfo(samples.dtype).eps if np.issubdtype(samples.dtype, np.floating) else np.finfo(float).eps
    values = np.asarray(samples, dtype=float)
    if len(values) < 2:
        raise ValueError("Each parameter grid must contain at least two samples")
    differences = np.diff(values)
    if not np.all(differences > 0):
        raise ValueError("Parameter grid samples must be strictly increasing")

    # Match Optuna's decimal arithmetic without rounding away a small step or
    # rounding it upward enough to exclude the final sample.
    decimal_step = (Decimal(str(values[-1])) - Decimal(str(values[0]))) / (len(values) - 1)
    step = float(decimal_step)
    if Decimal(str(step)) > decimal_step:
        step = float(np.nextafter(step, 0.))
    if not np.allclose(differences, step, rtol=1e-9, atol=precision * np.max(np.abs(values))):
        raise ValueError("Step size for each parameter must be constant")
    return step
