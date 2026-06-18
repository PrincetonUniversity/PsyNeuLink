"""Backend-neutral math namespace for batched op bodies.

Elementwise batched op bodies are written once against ``bm`` (for example
``bm.exp``).  The same function object is executed directly with numpy by the
``ir_debug`` backend, while the Triton backend captures the body's source and
rewrites ``bm`` to ``triton.language`` (``tl``) at emission time.

Only names that exist with matching semantics in both numpy and
``triton.language`` belong here.
"""

from __future__ import annotations

import numpy as np


class _NeutralMath:
    """Numpy-backed implementations of the neutral ``bm`` namespace."""

    exp = staticmethod(np.exp)
    log = staticmethod(np.log)
    sqrt = staticmethod(np.sqrt)
    abs = staticmethod(np.abs)
    minimum = staticmethod(np.minimum)
    maximum = staticmethod(np.maximum)
    where = staticmethod(np.where)
    ceil = staticmethod(np.ceil)
    floor = staticmethod(np.floor)
    cos = staticmethod(np.cos)
    sin = staticmethod(np.sin)


#: Names allowed as ``bm.<name>`` in captured Triton sources.
NEUTRAL_MATH_NAMES = frozenset(
    name for name in vars(_NeutralMath) if not name.startswith("_")
)

bm = _NeutralMath()
