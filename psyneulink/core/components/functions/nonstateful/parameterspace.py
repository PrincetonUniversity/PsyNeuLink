"""Interpret PEC parameter specifications independently of the search algorithm.

Explicit specifications distinguish continuous bounds, regular grids, and exact choices.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from optuna.distributions import CategoricalDistribution, FloatDistribution

__all__ = []


@dataclass(frozen=True)
class _ParameterSpace:
    lower: float
    upper: float
    step: float | None = None
    values: tuple[float, ...] | None = None

    @property
    def bounds(self):
        return self.lower, self.upper

    def continuous_bounds(self):
        if self.step is not None or self.values is not None:
            raise ValueError(
                "Continuous solvers do not support explicit grids or choices; "
                "use {'bounds': (lower, upper)} for continuous fitting"
            )
        return self.bounds

    def optuna_distribution(self):
        if self.values is not None:
            return CategoricalDistribution(self.values)
        return FloatDistribution(self.lower, self.upper, step=self.step)


def _numeric_values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("Parameter values must be a nonempty, one-dimensional sequence of finite numbers")
    return values


def _parse_parameter_space(spec):
    if not isinstance(spec, Mapping):
        raise ValueError(
            "PEC parameter spaces must be explicit dictionaries; bare sample lists, arrays, and tuples "
            "are no longer supported. Use {'bounds': (lower, upper)} for continuous fitting, "
            "{'bounds': (lower, upper), 'step': step} for a grid, "
            "or {'values': samples} for exact choices."
        )
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
