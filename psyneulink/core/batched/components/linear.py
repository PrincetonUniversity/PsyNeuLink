"""Batched op for the `Linear` transfer function."""

from psyneulink.core.batched.specs import batched_op
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear


@batched_op(Linear)
def linear(x, slope, intercept):
    return slope * x + intercept
