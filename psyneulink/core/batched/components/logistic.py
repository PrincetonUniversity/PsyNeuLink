"""Batched op for the `Logistic` transfer function."""

from psyneulink.core.batched.specs import LikelihoodEffectContract, batched_op
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic


@batched_op(Logistic, likelihood_contract=LikelihoodEffectContract())
def logistic(x, gain, bias, x_0, scale, offset):
    return scale / (1.0 + tl.exp(-gain * (x + bias - x_0))) + offset
