"""Batched op for the `Logistic` transfer function.

Note: this is the gain-only logistic used by the batched compiler so far;
``bias``/``x_0``/``offset`` are not yet part of the supported subset.
"""

from psyneulink.core.batched.specs import batched_op
from psyneulink.core.components.functions.nonstateful.transferfunctions import Logistic


@batched_op(Logistic)
def logistic(x, gain):
    return 1.0 / (1.0 + tl.exp(-gain * x))
