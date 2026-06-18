"""Backward-compatible re-export of the Triton emitter.

The emitter was split into the `emit/` package; this module preserves the
historical import path `backend.triton.graph_emit.triton_graph_kernel_source`.
"""

from psyneulink.core.batched.backend.triton.emit import (
    TritonGraphEmitter,
    triton_graph_kernel_source,
)

__all__ = ["TritonGraphEmitter", "triton_graph_kernel_source"]
