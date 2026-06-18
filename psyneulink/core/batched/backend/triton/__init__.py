"""Triton backend for batched PsyNeuLink execution."""

from psyneulink.core.batched.backend.triton.api import (
    TritonEmitContext,
    TritonOpCall,
    TritonOpError,
    TritonOpTemplate,
    pnl_triton_op,
)

__all__ = [
    "TritonEmitContext",
    "TritonOpCall",
    "TritonOpError",
    "TritonOpTemplate",
    "pnl_triton_op",
    "run_triton",
]


def __getattr__(name):
    # `runtime` pulls in the full lowering stack; import it lazily so that
    # importing the spec/registration API (which needs only `api`) does not
    # create an import cycle.
    if name == "run_triton":
        from psyneulink.core.batched.backend.triton.runtime import run_triton

        return run_triton
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
