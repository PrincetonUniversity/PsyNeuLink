"""Scalar NormalDist on ProcessingMechanism: forward draw and ideal-real law."""

from psyneulink.core.batched.backend.triton.api import pnl_triton_op
from psyneulink.core.batched.diagnostics import BatchedDiagnostic
from psyneulink.core.batched.likelihood_ir import GaussianReadout, LikelihoodEffectContract
from psyneulink.core.batched.specs import (
    ArgBinding, MechanismOpSpec, ParamBinding, RngDecl, register_batched_op,
)
from psyneulink.core.components.functions.nonstateful.distributionfunctions import NormalDist
from psyneulink.core.components.mechanisms.processing.processingmechanism import ProcessingMechanism


@pnl_triton_op
def normal_draw(mean, standard_deviation, seed, rng_base):
    return mean + standard_deviation * tl.randn(seed, rng_base)  # noqa: F821


def _supports(node):
    if len(node.input_ports) != 1 or node.input_ports[0].value.size != 1 or node.output_port.value.size != 1:
        return BatchedDiagnostic(node.name, "NormalDist batched support requires scalar input/output")


register_batched_op(MechanismOpSpec(
    mechanism_class=ProcessingMechanism,
    function_class=NormalDist,
    params=(ParamBinding("mean"), ParamBinding("standard_deviation", default=1., minimum=0.)),
    rng=(RngDecl("rng"),),
    triton_template=normal_draw,
    triton_bindings=(ArgBinding("param", "mean"), ArgBinding("param", "standard_deviation"),
                     ArgBinding("seed"), ArgBinding("rng_base")),
    supports=_supports,
    likelihood_contract=LikelihoodEffectContract(
        randomness="declared_streams",
        gaussian_readout=GaussianReadout(None, "mean", "standard_deviation"),
    ),
), function_specific=True)
