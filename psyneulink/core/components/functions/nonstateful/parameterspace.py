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

    def continuous_bounds(self, name):
        if self.step is not None or self.values is not None:
            raise ValueError(
                f"Invalid search space for {name}: continuous solvers do not support grids or choices. "
                "Use {'bounds': (lower, upper)} for continuous fitting."
            )
        return self.bounds

    def optuna_distribution(self):
        if self.values is not None:
            return CategoricalDistribution(self.values)
        return FloatDistribution(self.lower, self.upper, step=self.step)


def _numeric_values(values):
    values = np.asarray(values)
    if np.iscomplexobj(values) or (values.dtype == object and any(np.iscomplexobj(v) for v in values.flat)):
        raise ValueError("Parameter values must be real numbers")
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
    step = None
    if 'step' in spec:
        try:
            step = float(_numeric_values([spec['step']])[0])
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("Parameter step must be a finite, positive real number") from error
        if step <= 0:
            raise ValueError("Parameter step must be a finite, positive real number")
    return _ParameterSpace(float(bounds[0]), float(bounds[1]), step=step)


def _checked_optuna_params(params, distributions):
    """Validate sampler proposals and a resumed study's best trial against the current spaces."""
    checked = {}
    for name, dist in distributions.items():
        if name not in params:
            raise ValueError(f"Optuna trial is missing fitted parameter {name}")
        value = params[name]
        if isinstance(dist, CategoricalDistribution):
            valid = value in dist.choices
        else:
            value = float(value)
            valid = dist.low <= value <= dist.high
            if valid and dist.step is not None and value not in (dist.low, dist.high):
                # Compare in parameter units: subtracting a nonzero lower bound can lose
                # precision relative to a small step, even for the sampler's own grid points.
                index = (value - dist.low) / dist.step
                nearest = dist.low + round(index) * dist.step
                valid = abs(value - nearest) <= 1e-8 * dist.step
        if not valid:
            raise ValueError(f"Optuna trial value {value!r} for {name} is outside the PEC search space {dist}")
        checked[name] = value
    return checked
