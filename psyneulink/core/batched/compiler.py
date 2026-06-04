from __future__ import annotations

from dataclasses import dataclass

from psyneulink.core.batched.diagnostics import BatchedCapabilityReport
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.reference import run_reference
from psyneulink.core.batched.registry import analyze_composition


class BatchedCompileError(RuntimeError):
    pass


class BatchedCompositionCompiler:
    @staticmethod
    def diagnose(composition, backend: str = "reference", outputs=None, max_steps: int | None = None) -> BatchedCapabilityReport:
        report, _ = analyze_composition(composition, backend=backend, outputs=outputs, max_steps=max_steps)
        return report

    @staticmethod
    def compile(composition, backend: str = "reference", outputs=None, max_steps: int | None = None) -> "BatchedSimulationPlan":
        report, ir = analyze_composition(composition, backend=backend, outputs=outputs, max_steps=max_steps)
        if not report.is_supported or ir is None:
            raise BatchedCompileError(
                "Composition cannot be compiled for batched simulation: "
                + "; ".join(report.unsupported_reasons)
            )

        if backend not in {"reference", "triton"}:
            raise BatchedCompileError(f"Unknown batched backend '{backend}'.")

        return BatchedSimulationPlan(ir=ir, backend=backend, capability_report=report)


@dataclass(frozen=True)
class BatchedSimulationPlan:
    ir: BatchedCompositionIR
    backend: str
    capability_report: BatchedCapabilityReport

    def run(
        self,
        inputs,
        parameter_sets,
        num_estimates: int,
        subject_slices=None,
        seed=None,
        common_random_numbers: bool = True,
    ) -> BatchedSimulationResult:
        if self.backend == "reference":
            return run_reference(
                self.ir,
                inputs,
                parameter_sets,
                num_estimates,
                subject_slices=subject_slices,
                seed=seed,
                common_random_numbers=common_random_numbers,
            )

        if self.backend == "triton":
            from psyneulink.core.batched.triton_backend import run_triton

            return run_triton(
                self.ir,
                inputs,
                parameter_sets,
                num_estimates,
                subject_slices=subject_slices,
                seed=seed,
                common_random_numbers=common_random_numbers,
            )

        raise BatchedCompileError(f"Unknown batched backend '{self.backend}'.")
