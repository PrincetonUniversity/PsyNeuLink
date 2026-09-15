"""Batched op for the `Logistic` transfer function."""

import sympy as sp

from psyneulink.core.batched.continuous_ir import Sigmoid

from psyneulink.core.batched.specs import LikelihoodEffectContract, batched_op
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic


_arguments = sp.symbols("x gain bias x_0 scale offset", real=True)
with sp.evaluate(False):
    _x, _gain, _bias, _x_0, _scale, _offset = _arguments
    _value = sp.Lambda(_arguments, _scale * Sigmoid(_gain * (_x + _bias - _x_0)) + _offset)


@batched_op(Logistic, likelihood_contract=LikelihoodEffectContract(symbolic_value=_value))
def logistic(x, gain, bias, x_0, scale, offset):
    return scale / (1.0 + tl.exp(-gain * (x + bias - x_0))) + offset
