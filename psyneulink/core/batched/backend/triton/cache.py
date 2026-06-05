from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path


def load_triton_kernel_module(source: str, module_kind: str | None, model_kind: str):
    """Write inspectable Triton source to the cache and import it as a module."""

    cache_dir = Path(os.environ.get("PNL_TRITON_CACHE_DIR", Path(tempfile.gettempdir()) / "psyneulink_triton_batch"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    module_path = cache_dir / f"pnl_batched_{module_kind or model_kind}_{digest}.py"
    if not module_path.exists():
        module_path.write_text(source, encoding="utf-8")

    module_name = f"pnl_batched_{module_kind or model_kind}_{digest}"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
