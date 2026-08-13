"""Enable Triton interpret mode before anything imports Triton.

Triton bakes interpret-vs-compiled when its own ``@triton.jit`` library functions
(``tl.randn``, philox, ...) are first imported, so the ``triton_cpu`` backend
these tests use only works if ``TRITON_INTERPRET=1`` is set *before* the first
import of ``triton`` anywhere in the process.

That is not only this directory's concern.  On Linux ``torch`` ships Triton as a
dependency and imports it the first time an ``AutodiffComposition`` runs in
PyTorch mode.  Under ``-n auto`` a worker that happens to run an autodiff test
before these ones would import Triton *compiled*, and every batched test that
worker later picked up would fail.  That is exactly what CI hit, and it is why it
failed only on Linux: Triton is not installed on macOS or Windows, so the batched
tests skip there instead.

Setting the variable at conftest import means it is in place before any test
executes, whatever order the workers choose.  It applies to the whole process,
which is safe here: PsyNeuLink never compiles kernels through Triton, it only
imports it, and autodiff learning was verified to behave identically under
interpret mode.  The batched *GPU* backend cannot run in such a session, but no
test uses it -- GPU validation runs as its own process (see
``Scripts/Debug/pec_batch_compile``).

``setdefault`` so an explicit ``TRITON_INTERPRET`` in the environment still wins.
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")
