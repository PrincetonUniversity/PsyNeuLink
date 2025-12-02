# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""Global thread-control utilities for PsyNeuLink.

Provides a small API compatible with PyTorch-style set_num_threads / get_num_threads:

- set_num_threads(n: int) -> None
- get_num_threads() -> int
- reset_num_threads() -> None

Setting number of threads will attempt to set common environment variables used
by native/BLAS libraries and (if available) call library APIs such as
`torch.set_num_threads` and `threadpoolctl.threadpool_limits`.

This module intentionally keeps the implementation lightweight and best-effort
(does not force or guarantee thread-pools for all libraries).
"""

from __future__ import annotations

import logging
import os
import psutil

logger = logging.getLogger(__name__)

# Try to import optional libraries; failures are non-fatal and handled gracefully.
try:
    import torch  # type: ignore
except Exception:
    torch = None

# threadpoolctl is optional; used to configure thread pools for native libraries
# (e.g. OpenBLAS, MKL) if available at runtime.
try:
    import threadpoolctl  # type: ignore
except Exception:
    threadpoolctl = None

# Default number of threads: use psutil to get the number of CPUs available to the process (respects SLURM/CPU affinity)
# On MacOS, we need to use os.cpu_count() because psutil.Process().cpu_affinity() is not implemented
if os.name == "posix" and os.uname().sysname == "Darwin":
    _DEFAULT_NUM_THREADS: int = os.cpu_count() or 1
else:
    _DEFAULT_NUM_THREADS: int = len(psutil.Process().cpu_affinity())

# Current global setting (mutable)
_num_threads: int = _DEFAULT_NUM_THREADS

__all__ = [
    "set_num_threads",
    "get_num_threads",
    "reset_num_threads",
]


def _set_common_env_vars(n: int) -> None:
    """Set common environment variables that influence native thread pools.

    This is best-effort and will not raise on failures.
    """
    env_vars = [
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ]
    for v in env_vars:
        try:
            os.environ[v] = str(n)
        except Exception:
            logger.debug("Could not set environment variable %s", v)


def set_num_threads(n: int) -> None:
    """Set the global number of threads to use.

    Parameters
    ----------
    n : int
        Desired number of threads. Must be an integer >= 1.

    Notes
    -----
    - Updates an internal global value returned by `get_num_threads()`.
    - Attempts to set environment variables used by common native libraries.
    - If `torch` is available, calls `torch.set_num_threads(n)`.
    - If `threadpoolctl` is available, attempts to apply `threadpool_limits(n)`.

    Raises
    ------
    TypeError
        If `n` is not an integer.
    ValueError
        If `n` < 1.
    """
    global _num_threads
    if not isinstance(n, int):
        raise TypeError(f"n must be int, got {type(n).__name__}")
    if n < 1:
        raise ValueError("n must be >= 1")

    _num_threads = n

    # Set environment variables (best-effort)
    _set_common_env_vars(n)

    # Configure thread limits for libraries that expose an API
    if torch is not None:
        try:
            torch.set_num_threads(n)
        except Exception:
            logger.debug("torch.set_num_threads failed", exc_info=True)

    if threadpoolctl is not None:
        try:
            # threadpool_limits is available from threadpoolctl 2.x; use best-effort
            threadpoolctl.threadpool_limits(n)
        except Exception:
            logger.debug("threadpoolctl.threadpool_limits failed", exc_info=True)


def get_num_threads() -> int:
    """Return the current global number of threads (the value set with
    `set_num_threads`, or the platform default if never set).
    """
    return _num_threads


def reset_num_threads() -> None:
    """Reset global thread setting to the platform default (cpu_count()).

    This updates the internal setting and reapplies environment vars and library
    settings using `set_num_threads`.
    """
    global _num_threads
    _num_threads = _DEFAULT_NUM_THREADS
    set_num_threads(_num_threads)
