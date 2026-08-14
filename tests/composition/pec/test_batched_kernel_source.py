"""One representative golden-source snapshot for generated Triton kernels.

The stateless fixture pins exact source for mechanical emitter refactors. Broad
compiler coverage belongs in KernelIR structural assertions and Python/GPU
behavior tests; duplicating every fusion as a source snapshot made semantic
changes expensive without adding an independent correctness oracle.

Emission is deterministic. Both sides are re-rendered through the running
interpreter's unparser before comparison because `ast.unparse` formatting
differs between Python versions -- see `_normalized`.

To regenerate after an *intentional* emission change:

    PNL_UPDATE_KERNEL_GOLDENS=1 .venv/bin/python -m pytest \\
        tests/composition/pec/test_batched_kernel_source.py -q -n 0
"""

import ast
import os
from pathlib import Path

import pytest

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source


pytestmark = [pytest.mark.batched, pytest.mark.composition]

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_kernels"


def _source_for(plan):
    return triton_graph_kernel_source(lower_to_kernel_ir(plan.ir))


def _stateless_graph_plan():
    source = pnl.TransferMechanism(
        input_shapes=2, function=pnl.Linear(slope=1.0, intercept=0.0), name="source"
    )
    target = pnl.TransferMechanism(
        input_shapes=1, function=pnl.Linear(slope=3.0, intercept=1.0), name="target"
    )
    comp = pnl.Composition(pathways=[[source, pnl.MappingProjection(matrix=[[1.0], [2.0]]), target]])
    return BatchedCompositionCompiler.compile(comp, backend="triton_cpu")


def test_generated_kernel_source_matches_golden():
    """Keep one representative snapshot; semantic tests carry broad coverage."""

    _assert_matches_golden("stateless_graph", _source_for(_stateless_graph_plan()))


def _normalized(source, name):
    """Re-render `source` through this interpreter's own unparser.

    Component helper bodies are emitted with `ast.unparse`, whose formatting is
    not stable across Python versions -- 3.10 renders a tuple assignment target
    as `(a, b) = f()` where later versions emit `a, b = f()`. Comparing raw text
    therefore fails on whichever version did not write the golden (this showed up
    in CI as three golden mismatches on 3.10, on every platform).

    Round-tripping both sides through the *running* interpreter removes that
    difference without weakening the check: the comparison is still over complete
    source text, so any real emission change -- a renamed variable, a reordered
    op, a changed constant -- still fails.
    """

    return ast.unparse(ast.parse(source, filename=f"<{name}>"))


def _assert_matches_golden(name, source):
    golden_path = GOLDEN_DIR / f"{name}.py"

    if os.environ.get("PNL_UPDATE_KERNEL_GOLDENS"):
        golden_path.write_text(source, encoding="utf-8")

    golden = golden_path.read_text(encoding="utf-8")
    assert _normalized(source, name) == _normalized(golden, f"{name}-golden"), (
        f"Generated kernel source for '{name}' differs from the golden snapshot. "
        "If this change is intentional, regenerate with "
        "PNL_UPDATE_KERNEL_GOLDENS=1."
    )
    # The golden must also be valid Python (so the kernel module can be imported).
    compile(source, f"<{name}>", "exec")
