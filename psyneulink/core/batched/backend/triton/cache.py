from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load_triton_kernel_module(source: str, module_kind: str | None, model_kind: str, interpret: bool = False):
    """Write inspectable Triton source to the cache and import it as a module.

    Triton decides interpret-vs-compiled at ``@triton.jit`` decoration time (when
    this module is imported), reading ``knobs.runtime.interpret``.  The caller
    (:func:`interpret_scope`) holds that knob across both import *and* launch,
    because nested ``@triton.jit`` helpers re-check it at call time.  The module
    name is tagged by mode so a single process can hold both an interpreted (CPU)
    and a compiled (GPU) build of the same kernel without colliding in
    ``sys.modules``.
    """

    cache_dir = Path(os.environ.get("PNL_TRITON_CACHE_DIR", Path(tempfile.gettempdir()) / "psyneulink_triton_batch"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    mode = "interp" if interpret else "compiled"
    module_name = f"pnl_batched_{module_kind or model_kind}_{mode}_{digest}"
    module_path = cache_dir / f"{module_name}.py"
    if not module_path.exists():
        module_path.write_text(source, encoding="utf-8")

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


@contextlib.contextmanager
def interpret_scope(interpret: bool):
    """Set ``knobs.runtime.interpret`` for the duration of import + launch."""

    from triton import knobs

    previous = knobs.runtime.interpret
    knobs.runtime.interpret = interpret
    try:
        yield
    finally:
        knobs.runtime.interpret = previous
