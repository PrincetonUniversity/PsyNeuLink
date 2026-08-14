"""Capability preflight and immutable-plan contracts for batched codegen."""

import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
    MechanismOpSpec,
    register_batched_instance_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import registry as batched_registry
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.fixture(autouse=True)
def _backend_capability_is_available(monkeypatch):
    """Keep source preflight independent of optional Triton installations."""

    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda _backend: (True, ()),
    )


def _registered_emitter_composition(node_name, emitter):
    node = pnl.TransferMechanism(input_shapes=1, name=node_name)
    composition = pnl.Composition(pathways=node)
    register_batched_instance_op(
        node.name,
        MechanismOpSpec(
            mechanism_class=type(node),
            function_class=type(node.function),
            display_name=node.name,
            triton_emit=emitter,
        ),
    )
    return composition, node


def test_raising_custom_emitter_is_a_codegen_diagnostic():
    def raising_emitter(_ctx, _node, _inputs, _outputs):
        raise RuntimeError("intentional source-emission failure")

    composition, node = _registered_emitter_composition(
        "raising source emitter",
        raising_emitter,
    )
    try:
        report = BatchedCompositionCompiler.diagnose(
            composition,
            backend="triton_cpu",
        )

        assert report.model_supported
        assert report.codegen_ready is False
        assert not report.can_execute
        assert len(report.codegen_diagnostics) == 1
        diagnostic = report.codegen_diagnostics[0]
        assert (
            diagnostic.code
            == BatchedDiagnosticCode.CODEGEN_SOURCE_EMISSION_FAILED
        )
        assert "intentional source-emission failure" in diagnostic.detail
        with pytest.raises(BatchedCompileError) as error:
            BatchedCompositionCompiler.compile(
                composition,
                backend="triton_cpu",
            )
        assert error.value.capability_report is not None
        assert (
            error.value.capability_report.codegen_diagnostics[0].code
            == BatchedDiagnosticCode.CODEGEN_SOURCE_EMISSION_FAILED
        )
    finally:
        unregister_batched_instance_op(node.name)


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_source_preflight_does_not_swallow_base_exceptions(exception_type):
    def interrupted_emitter(_ctx, _node, _inputs, _outputs):
        raise exception_type("intentional preflight interruption")

    composition, node = _registered_emitter_composition(
        f"{exception_type.__name__} source emitter",
        interrupted_emitter,
    )
    try:
        with pytest.raises(exception_type, match="intentional preflight interruption"):
            BatchedCompositionCompiler.diagnose(
                composition,
                backend="triton_cpu",
            )
    finally:
        unregister_batched_instance_op(node.name)


def test_successful_source_preflight_uses_executable_kernel(monkeypatch):
    emitted = []

    def record_source(kernel):
        source = triton_graph_kernel_source(kernel)
        emitted.append((kernel, source))
        return source

    monkeypatch.setattr(
        batched_registry,
        "triton_graph_kernel_source",
        record_source,
    )
    mechanism = pnl.TransferMechanism(input_shapes=1, name="preflight source")
    composition = pnl.Composition(pathways=mechanism)

    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
    )

    assert report.can_execute
    assert len(emitted) == 1
    kernel, source = emitted[0]
    assert kernel.executable
    assert "pnl_batched_stateless_graph_kernel" in source


def test_compile_reuses_exact_preflight_kernel_snapshot(monkeypatch):
    emitted_kernels = []
    node_name = "self replacing source emitter"

    def bad_replacement(_ctx, _node, _inputs, _outputs):
        raise RuntimeError("replacement emitter must not reach the plan")

    def replacing_emitter(_ctx, node, inputs, _outputs):
        register_batched_instance_op(
            node.name,
            MechanismOpSpec(
                mechanism_class=pnl.TransferMechanism,
                function_class=pnl.Linear,
                display_name=node.name,
                triton_emit=bad_replacement,
            ),
        )
        return inputs

    real_source_emitter = triton_graph_kernel_source

    def record_source(kernel):
        emitted_kernels.append(kernel)
        return real_source_emitter(kernel)

    monkeypatch.setattr(
        batched_registry,
        "triton_graph_kernel_source",
        record_source,
    )
    composition, node = _registered_emitter_composition(
        node_name,
        replacing_emitter,
    )
    try:
        plan = BatchedCompositionCompiler.compile(
            composition,
            backend="triton_cpu",
        )

        assert len(emitted_kernels) == 1
        assert plan.kernel_ir is emitted_kernels[0]
        assert plan.kernel_ir.executable
        assert "def pnl_batched_" in real_source_emitter(plan.kernel_ir)
    finally:
        unregister_batched_instance_op(node.name)
