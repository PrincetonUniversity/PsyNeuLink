from __future__ import annotations

from dataclasses import dataclass

from psyneulink.core.batched.bindings import (
    EMPTY_COMPONENT_BINDINGS,
    BatchedComponentBindings,
)
from psyneulink.core.batched.diagnostics import BatchedCapabilityReport
from psyneulink.core.batched.ir_debug import run_ir_debug
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.registry import analyze_composition


class BatchedCompileError(RuntimeError):
    pass


_SUPPORTED_BACKENDS = {"ir_debug", "triton"}


class BatchedCompositionCompiler:
    @staticmethod
    def diagnose(composition, backend: str = "ir_debug", outputs=None, max_steps: int | None = None) -> BatchedCapabilityReport:
        _validate_backend(backend)
        report, _, _ = analyze_composition(
            composition,
            backend=backend,
            outputs=outputs,
            max_steps=max_steps,
        )
        return report

    @staticmethod
    def compile(composition, backend: str = "ir_debug", outputs=None, max_steps: int | None = None) -> "BatchedSimulationPlan":
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
    ) -> BatchedSimulationResult:
        if self.backend == "ir_debug":
            return run_ir_debug(
                self.ir,
                inputs,
                parameter_sets,
                num_estimates,
                subject_slices=subject_slices,
                seed=seed,
                common_random_numbers=common_random_numbers,
        )

        if self.backend == "triton":
            from psyneulink.core.batched.backend.triton import run_triton

            return run_triton(
                self.ir,
                inputs,
                parameter_sets,
                num_estimates,
                subject_slices=subject_slices,
                seed=seed,
                common_random_numbers=common_random_numbers,
                component_bindings=self.component_bindings,
            )

        raise BatchedCompileError(f"Unknown batched backend '{self.backend}'.")


def _validate_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise BatchedCompileError(f"Unknown batched backend '{backend}'.")
