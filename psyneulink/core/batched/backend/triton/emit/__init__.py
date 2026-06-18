"""Triton kernel-source emission from backend-neutral KernelIR.

The emitter is split across modules for maintainability: `emitter.py` (class
assembly, orchestration, signature/module rendering), `lanes.py` (lane decode +
RNG-base layout + raw input), and `ops.py` (per-`KernelOp` emission + value
table).  New op kinds are added in `ops.py`.
"""

from psyneulink.core.batched.backend.triton.emit.emitter import (
    TritonGraphEmitter,
    triton_graph_kernel_source,
)

__all__ = ["TritonGraphEmitter", "triton_graph_kernel_source"]
