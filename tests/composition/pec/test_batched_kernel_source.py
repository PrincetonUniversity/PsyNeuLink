"""Golden-source snapshot tests for the generated Triton kernels.

These pin the *exact* emitted kernel source for representative fixtures. They are
the acceptance criterion for refactors that must not change emission (e.g.
splitting `graph_emit.py` into the `emit/` package): the generated source must
match the committed goldens in `golden_kernels/`.

Emission is deterministic (fixed op order, insertion-ordered template/param
dicts), so this is stable across runs. Both sides are re-rendered through the
running interpreter's unparser before comparison, because `ast.unparse`
formatting differs between Python versions -- see `_normalized`.

To regenerate after an *intentional* emission change:

    PNL_UPDATE_KERNEL_GOLDENS=1 .venv/bin/python -m pytest \\
        tests/composition/pec/test_batched_kernel_source.py -q -n 0
"""

import ast
import os
import sys
from pathlib import Path

import numpy as np
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


def _ddm_graph_plan():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0, rate=1.0, noise=0.0, threshold=0.05,
            non_decision_time=0.0, time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    stimulus = pnl.TransferMechanism(input_shapes=1, name="stimulus")
    comp = pnl.Composition(pathways=[[stimulus, decision]])
    return BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=64)


def _stateful_graph_plan():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01, ddm_time_step_size=0.01,
        threshold=0.05, ddm_noise=0.0, lca_noise=0.0,
    )
    return BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=256)


@pytest.mark.parametrize(
    "name, plan_factory",
    [
        ("stateless_graph", _stateless_graph_plan),
        ("ddm_graph", _ddm_graph_plan),
        ("stateful_graph", _stateful_graph_plan),
    ],
)
def test_generated_kernel_source_matches_golden(name, plan_factory):
    _assert_matches_golden(name, _source_for(plan_factory()))


def test_coevolving_graph_source_matches_golden():
    """The fused co-evolution loop (Always-LCA stepping with the DDM) emission."""

    from psyneulink.core.batched import batched_node_op, unregister_batched_instance_op

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Scripts" / "Debug" / "pec_batch_compile"))
    from csi_model_surrogate import make_stab_flex

    try:
        @batched_node_op("Drift Rate Value")
        def drift_rate(x0, x1, x2, x3, x4, x5, x6):
            a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
            b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
            c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
            d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
            pos = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
            neg = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
            return (pos - neg) * x6

        comp = make_stab_flex(iti=0, csi_repeat=0, csi_switch=0, threshold_collapse=-0.001,
                              ddm_noise=0.0, lca_noise=0.0)
        plan = BatchedCompositionCompiler.compile(comp, backend="triton_cpu", max_steps=4000)
        _assert_matches_golden("coevolving_graph", _source_for(plan))
    finally:
        unregister_batched_instance_op("Drift Rate Value")


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
