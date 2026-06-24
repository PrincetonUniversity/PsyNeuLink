from __future__ import annotations

from dataclasses import dataclass

from psyneulink.core.batched.bindings import (
    EMPTY_COMPONENT_BINDINGS,
    BatchedComponentBindings,
)
from psyneulink.core.batched.diagnostics import BatchedCapabilityReport
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.registry import analyze_composition


class BatchedCompileError(RuntimeError):
    pass


# "triton_cpu" runs the generated kernels through Triton's interpreter on CPU (no
# CUDA needed); "triton" compiles and runs them on the GPU.  Both execute the same
# kernel source, so the CPU path is a true stand-in for the GPU path in testing.
_BACKEND_DEVICES = {"triton_cpu": "cpu", "triton": "cuda"}
_SUPPORTED_BACKENDS = set(_BACKEND_DEVICES)


class BatchedCompositionCompiler:
    @staticmethod
    def diagnose(composition, backend: str = "triton_cpu", outputs=None, max_steps: int | None = None) -> BatchedCapabilityReport:
        _validate_backend(backend)
        report, _, _ = analyze_composition(
            composition,
            backend=backend,
            outputs=outputs,
            max_steps=max_steps,
        )
        return report

    @staticmethod
    def compile(composition, backend: str = "triton_cpu", outputs=None, max_steps: int | None = None) -> "BatchedSimulationPlan":
        _validate_backend(backend)
        report, ir, bindings = analyze_composition(
            composition,
            backend=backend,
            outputs=outputs,
            max_steps=max_steps,
        )
        if not report.is_supported or ir is None:
            raise BatchedCompileError(
                "Composition cannot be compiled for batched simulation: "
                + "; ".join(report.unsupported_reasons)
            )

        return BatchedSimulationPlan(
            ir=ir,
            backend=backend,
            capability_report=report,
            component_bindings=bindings,
        )


@dataclass(frozen=True)
class BatchedSimulationPlan:
    ir: BatchedCompositionIR
    backend: str
    capability_report: BatchedCapabilityReport
    component_bindings: BatchedComponentBindings = EMPTY_COMPONENT_BINDINGS

    def run(
        self,
        inputs,
        parameter_sets,
        num_estimates: int,
        subject_slices=None,
        seed=None,
        common_random_numbers: bool = True,
        strict_truncation: bool = False,
    ) -> BatchedSimulationResult:
        try:
            device = _BACKEND_DEVICES[self.backend]
        except KeyError:
            raise BatchedCompileError(f"Unknown batched backend '{self.backend}'.")

        from psyneulink.core.batched.backend.triton import run_triton

        return run_triton(
            self.ir,
            inputs,
            parameter_sets,
            num_estimates,
            subject_slices=subject_slices,
            seed=seed,
            common_random_numbers=common_random_numbers,
            device=device,
            strict_truncation=strict_truncation,
        )


def _validate_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise BatchedCompileError(f"Unknown batched backend '{backend}'.")
