import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnostic,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched import registry as batched_registry


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _supported_composition():
    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Linear(),
        name="linear",
    )
    return pnl.Composition(pathways=mechanism)


def _set_backend_available(monkeypatch):
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda backend: (True, []),
    )


def test_capability_report_distinguishes_all_ready_states(monkeypatch):
    _set_backend_available(monkeypatch)

    report = BatchedCompositionCompiler.diagnose(
        _supported_composition(),
        backend="triton_cpu",
    )

    assert report.model_supported
    assert report.codegen_ready is True
    assert report.codegen_status == "ready"
    assert report.backend_available
    assert report.can_execute
    assert report.is_supported

    serialized = report.to_dict()
    assert serialized["model_supported"] is True
    assert serialized["codegen_ready"] is True
    assert serialized["codegen_status"] == "ready"
    assert serialized["backend_available"] is True
    assert serialized["can_execute"] is True


def test_unsupported_model_leaves_codegen_not_checked(monkeypatch):
    _set_backend_available(monkeypatch)

    class CustomLinear(pnl.Linear):
        pass

    mechanism = pnl.TransferMechanism(
        input_shapes=1,
        function=CustomLinear(),
        name="custom_linear",
    )
    report = BatchedCompositionCompiler.diagnose(
        pnl.Composition(pathways=mechanism),
        backend="triton_cpu",
    )

    assert not report.model_supported
    assert report.codegen_ready is None
    assert report.codegen_status == "not_checked"
    assert report.backend_available
    assert not report.can_execute
    assert not report.is_supported

    assert len(report.rejected_nodes) == 1
    diagnostic = report.rejected_nodes[0]
    assert diagnostic.code == BatchedDiagnosticCode.MODEL_FUNCTION_UNSUPPORTED
    assert diagnostic.component == "custom_linear"
    assert diagnostic.component_id == "node:custom_linear"
    assert diagnostic.to_dict()["component_id"] == "node:custom_linear"


def test_codegen_failure_is_separate_from_model_support(monkeypatch):
    _set_backend_available(monkeypatch)
    codegen_diagnostic = BatchedDiagnostic(
        component="linear",
        reason="missing Triton implementation for batched op",
        detail="Linear",
        code=BatchedDiagnosticCode.CODEGEN_OP_MISSING,
        component_id="node:linear",
    )
    monkeypatch.setattr(
        batched_registry,
        "_triton_spec_diagnostics",
        lambda graph: [codegen_diagnostic],
    )

    report = BatchedCompositionCompiler.diagnose(
        _supported_composition(),
        backend="triton_cpu",
    )

    assert report.model_supported
    assert report.codegen_ready is False
    assert report.codegen_status == "not_ready"
    assert report.backend_available
    assert not report.can_execute
    assert not report.is_supported
    assert report.rejected_nodes == ()
    assert report.codegen_diagnostics == (codegen_diagnostic,)


def test_compile_rejects_unavailable_backend_with_report(monkeypatch):
    backend_diagnostic = BatchedDiagnostic(
        component="triton_cpu",
        reason="Triton is not installed.",
        code=BatchedDiagnosticCode.BACKEND_TRITON_MISSING,
        component_id="backend:triton_cpu",
    )
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda backend: (False, [backend_diagnostic]),
    )

    composition = _supported_composition()
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")
    assert report.model_supported
    assert report.codegen_ready is True
    assert not report.backend_available
    assert not report.can_execute
    # Compatibility: model + codegen support is independent of this machine.
    assert report.is_supported
    assert report.backend_diagnostics == (backend_diagnostic,)

    with pytest.raises(BatchedCompileError, match="Triton is not installed") as error:
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")

    compile_report = error.value.capability_report
    assert compile_report is not None
    assert not compile_report.can_execute
    assert compile_report.backend_diagnostics[0].code == (
        BatchedDiagnosticCode.BACKEND_TRITON_MISSING
    )


def test_backend_probe_emits_stable_dependency_codes(monkeypatch):
    monkeypatch.setattr(
        batched_registry.importlib.util,
        "find_spec",
        lambda name: None if name in {"torch", "triton"} else object(),
    )

    available, diagnostics = batched_registry._backend_availability("triton_cpu")

    assert not available
    assert {diagnostic.code for diagnostic in diagnostics} == {
        BatchedDiagnosticCode.BACKEND_TRITON_MISSING,
        BatchedDiagnosticCode.BACKEND_TORCH_MISSING,
    }
    assert {diagnostic.component_id for diagnostic in diagnostics} == {
        "backend:triton_cpu"
    }
