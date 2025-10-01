Thread configuration
====================

Control the global number of threads used by PsyNeuLink and common native/back-end
libraries at runtime.

Public API
----------

- ``psyneulink.core.globals.set_num_threads(n)`` — set the number of threads to use.
- ``psyneulink.core.globals.get_num_threads()`` — return the current global thread setting.
- ``psyneulink.core.globals.reset_num_threads()`` — reset to the platform default (``os.cpu_count()`` or 1).

Examples
--------

.. code-block:: python

    from psyneulink.core.globals import set_num_threads, get_num_threads, reset_num_threads

    # Query current value (platform default on first import)
    print(get_num_threads())  # e.g. 8

    # Request 2 threads for current process and supported libraries
    set_num_threads(2)
    assert get_num_threads() == 2

    # Reset to platform default
    reset_num_threads()

Semantics and behavior
----------------------

- The API provides a small, PyTorch-style control surface to set and query a global
  thread count for the current process.
- Calling ``set_num_threads(n)``:
  - Validates input: ``n`` must be an ``int`` >= 1; otherwise ``TypeError`` or ``ValueError`` is raised.
  - Updates an internal global value returned by ``get_num_threads()``.
  - Attempts to set common environment variables used by native thread pools:
    - ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``, ``OPENBLAS_NUM_THREADS``, ``VECLIB_MAXIMUM_THREADS``, ``NUMEXPR_NUM_THREADS``.
    - These environment changes affect the current Python process and any child processes spawned afterwards.
  - If optional libraries are available at runtime, PsyNeuLink will try to configure them:
    - If ``torch`` is importable, it calls ``torch.set_num_threads(n)``.
    - If ``threadpoolctl`` is importable, it attempts ``threadpoolctl.threadpool_limits(n)``.
  - All such library-specific calls are best-effort and failures are caught and logged; they will not raise to the caller.

Limitations and recommendations
-------------------------------

- Best-effort only: some libraries create thread pools at import/initialization time and may not honor changes made later in the process.
- If you rely on deterministic performance or want to avoid oversubscription (CI, multithreaded runners), set a fixed number of threads (for example 1 or the number of physical cores) early in test setup or before heavy computation.
- Use the API rather than mutating environment variables directly to keep a consistent internal record via ``get_num_threads()``.
- Invalid inputs raise:
  - ``TypeError`` if ``n`` is not an ``int``.
  - ``ValueError`` if ``n < 1``.



