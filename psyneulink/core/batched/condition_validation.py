"""Fail-closed validation of live PsyNeuLink scheduler conditions.

Condition instances are mutable.  Exact class checks alone are therefore not
enough before their semantics are snapshotted into object-free batched IR.
"""

from __future__ import annotations

import inspect

from psyneulink.core.scheduling.condition import (
    All,
    Always,
    AtPass,
    AtTrialStart,
    EveryNCalls,
    Never,
    WhenFinished,
)
from psyneulink.core.scheduling.time import TimeScale


def is_canonical_condition(condition) -> bool:
    """Whether ``condition`` still has its built-in class's exact semantics."""

    args = getattr(condition, "args", None)
    kwargs = getattr(condition, "kwargs", None)
    if type(args) is not tuple or type(kwargs) is not dict or kwargs:
        return False

    condition_type = type(condition)
    try:
        if condition_type is Always:
            canonical = Always()
            valid_args = args == ()
        elif condition_type is Never:
            canonical = Never()
            valid_args = args == ()
        elif condition_type is AtTrialStart:
            canonical = AtTrialStart()
            valid_args = args == (0,)
        elif condition_type is AtPass:
            if len(args) != 1:
                return False
            time_scale = _captured_time_scale(condition)
            if type(time_scale) is not TimeScale:
                return False
            canonical = AtPass(args[0], time_scale=time_scale)
            valid_args = True
        elif condition_type is WhenFinished:
            if len(args) != 1:
                return False
            canonical = WhenFinished(args[0])
            valid_args = True
        elif condition_type is EveryNCalls:
            if len(args) != 2:
                return False
            canonical = EveryNCalls(*args)
            valid_args = True
        elif condition_type is All:
            canonical = All(*args)
            valid_args = True
        else:
            return False
    except Exception:
        return False
    return valid_args and _callable_matches(condition, canonical)


def _captured_time_scale(condition):
    try:
        return inspect.getclosurevars(condition.func).nonlocals["time_scale"]
    except (AttributeError, KeyError, TypeError):
        return None


def _callable_matches(condition, canonical) -> bool:
    actual = getattr(condition, "func", None)
    expected = getattr(canonical, "func", None)
    if inspect.ismethod(expected):
        return (
            inspect.ismethod(actual)
            and actual.__func__ is expected.__func__
            and actual.__self__ is condition
        )
    if not inspect.isfunction(actual) or not inspect.isfunction(expected):
        return False
    if (
        actual.__code__ is not expected.__code__
        or actual.__defaults__ != expected.__defaults__
        or actual.__kwdefaults__ != expected.__kwdefaults__
    ):
        return False
    try:
        actual_nonlocals = inspect.getclosurevars(actual).nonlocals
        expected_nonlocals = inspect.getclosurevars(expected).nonlocals
    except (AttributeError, TypeError):
        return False
    if actual_nonlocals.keys() != expected_nonlocals.keys():
        return False
    for name, expected_value in expected_nonlocals.items():
        actual_value = actual_nonlocals[name]
        if name == "self":
            if actual_value is not condition:
                return False
        elif actual_value is not expected_value and actual_value != expected_value:
            return False
    return True
