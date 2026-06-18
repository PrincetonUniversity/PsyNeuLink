"""Golden-source snapshot tests for the generated Triton kernels.

These pin the *exact* emitted kernel source for representative fixtures. They are
the acceptance criterion for refactors that must not change emission (e.g.
splitting `graph_emit.py` into the `emit/` package): the generated source must
stay byte-identical to the committed goldens in `golden_kernels/`.

Emission is deterministic (fixed op order, insertion-ordered template/param
dicts), so byte-equality is stable across runs.

To regenerate after an *intentional* emission change:

    PNL_UPDATE_KERNEL_GOLDENS=1 .venv/bin/python -m pytest \\
        tests/composition/pec/test_batched_kernel_source.py -q -n 0
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl

from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source


pytestmark = pytest.mark.composition

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
    source = _source_for(plan_factory())
    golden_path = GOLDEN_DIR / f"{name}.py"

    if os.environ.get("PNL_UPDATE_KERNEL_GOLDENS"):
        golden_path.write_text(source, encoding="utf-8")

    golden = golden_path.read_text(encoding="utf-8")
    assert source == golden, (
        f"Generated kernel source for '{name}' differs from the golden snapshot. "
        "If this change is intentional, regenerate with "
        "PNL_UPDATE_KERNEL_GOLDENS=1."
    )
    # The golden must also be valid Python (so the kernel module can be imported).
    compile(source, f"<{name}>", "exec")
