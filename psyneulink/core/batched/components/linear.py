"""Batched op for the `Linear` transfer function."""

import sympy as sp

from psyneulink.core.batched.specs import LikelihoodEffectContract, batched_op
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear


_arguments = sp.symbols("x slope intercept scale offset", real=True)
with sp.evaluate(False):
    _x, _slope, _intercept, _scale, _offset = _arguments
    _value = sp.Lambda(_arguments, _scale * (_x * _slope + _intercept) + _offset)


@batched_op(Linear, likelihood_contract=LikelihoodEffectContract(value_rule="affine", symbolic_value=_value))
def linear(x, slope, intercept, scale, offset):
    return scale * (x * slope + intercept) + offset
