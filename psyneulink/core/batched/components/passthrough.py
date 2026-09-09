"""Passthrough mechanism specs.

These mechanisms add no kernel semantics of their own: they combine their
inputs and apply their function.  A node with a passthrough mechanism class
is supported whenever its function class has a registered elementwise batched
op.
"""

from psyneulink.core.batched.specs import (
    LikelihoodEffectContract, PassthroughMechanismSpec, register_batched_op,
)
from psyneulink.core.components.mechanisms.processing.processingmechanism import (
    ProcessingMechanism,
)
from psyneulink.core.components.mechanisms.processing.transfermechanism import (
    TransferMechanism,
)

register_batched_op(PassthroughMechanismSpec(
    TransferMechanism, likelihood_contract=LikelihoodEffectContract(),
))
register_batched_op(PassthroughMechanismSpec(
    ProcessingMechanism, likelihood_contract=LikelihoodEffectContract(),
))
