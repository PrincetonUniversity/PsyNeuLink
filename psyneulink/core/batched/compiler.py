from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from psyneulink.core.batched.bindings import (
    EMPTY_COMPONENT_BINDINGS,
    BatchedComponentBindings,
)
from psyneulink.core.batched.diagnostics import BatchedCapabilityReport
from psyneulink.core.batched.ir import BatchedCompositionIR, BatchedSimulationResult
from psyneulink.core.batched.kernel_ir import KernelIR
from psyneulink.core.batched.registry import analyze_composition


class BatchedCompileError(RuntimeError):
    def __init__(self, message, *, capability_report=None):
        super().__init__(message)
        self.capability_report = capability_report


# "triton_cpu" runs generated kernels through Triton's interpreter on CPU (no
# CUDA needed); "triton" compiles and runs them on the GPU.  The interpreter is
# useful for semantic coverage, but real-GPU tests remain necessary for launch,
# masking, device, and compiled-code behavior.
_BACKEND_DEVICES = {"triton_cpu": "cpu", "triton": "cuda"}
_SUPPORTED_BACKENDS = set(_BACKEND_DEVICES)


class BatchedCompositionCompiler:
    @staticmethod
    def diagnose(
        composition,
        backend: str = "triton_cpu",
        outputs=None,
        max_steps: int | None = None,
        *,
        ignored_control_nodes=(),
    ) -> BatchedCapabilityReport:
        _validate_backend(backend)
        report, _, _, _ = analyze_composition(
            composition,
            backend=backend,
            outputs=outputs,
            max_steps=max_steps,
            ignored_control_nodes=ignored_control_nodes,
        )
        return report

    @staticmethod
    def compile(
        composition,
        backend: str = "triton_cpu",
        outputs=None,
        max_steps: int | None = None,
        *,
        ignored_control_nodes=(),
    ) -> BatchedSimulationPlan:
        _validate_backend(backend)
        report, ir, bindings, kernel_ir = analyze_composition(
            composition,
            backend=backend,
            outputs=outputs,
            max_steps=max_steps,
            ignored_control_nodes=ignored_control_nodes,
        )
        if not report.can_execute or ir is None or kernel_ir is None:
            blockers = report.execution_blockers or (
                "compiler analysis did not produce an executable intermediate representation",
            )
            raise BatchedCompileError(
                "Composition cannot be compiled for batched simulation: "
                + "; ".join(blockers),
                capability_report=report,
            )

        return BatchedSimulationPlan(
            ir=ir,
            backend=backend,
            capability_report=report,
            kernel_ir=kernel_ir,
            component_bindings=bindings,
        )


@dataclass(frozen=True)
class BatchedSimulationPlan:
    ir: BatchedCompositionIR
    backend: str
    capability_report: BatchedCapabilityReport
    # This is the exact frozen snapshot that capability analysis successfully
    # emitted; recompiling it here would reopen the registry-mutation race.
    kernel_ir: KernelIR = field(repr=False)
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
        keep_device_values: bool = False,
        triton_launch_options: Mapping | None = None,
    ) -> BatchedSimulationResult:
        """Run one or more parameter sets over the supplied trial inputs.

        A parameter-set value is normally scalar.  Wrap a trial vector (or a
        ``[subject, trial]`` array) in :class:`BatchedTrialParameter` when that
        model parameter must change between trials while lane state persists.
        """
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
            keep_device_values=keep_device_values,
            kernel_ir=self.kernel_ir,
            component_bindings=self.component_bindings,
            launch_options=triton_launch_options,
        )

    def log_likelihood(
        self,
        inputs,
        parameter_sets,
        num_estimates: int,
        data,
        categorical_dims=None,
        *,
        outcome_indices=None,
        bins: int = 100,
        bin_range=None,
        smoothing_sigma: float = 0.0,
        pseudocount: float = 0.0,
        categorical_cardinalities=None,
        include_mask=None,
        subject_slices=None,
        seed=None,
        common_random_numbers: bool = True,
        strict_truncation: bool = False,
        triton_launch_options: Mapping | None = None,
    ):
        """Simulate and score experimental ``data`` with a histogram likelihood.

        Runs the batched simulation for ``parameter_sets`` and returns the total
        histogram log-likelihood of ``data`` per parameter set, aggregated over
        subjects and trials.  On the ``triton`` (GPU) backend the outcomes stay
        on the device and the likelihood is computed there (no host round-trip).

        Parameter values may use :class:`BatchedTrialParameter`, as for
        :meth:`run`.  ``data`` is the experimental data shaped ``[trial, outcome]``.
        ``outcome_indices`` selects/reorders the plan outputs to line up with the
        columns of ``data`` (defaults to all outputs, in order).  See
        :func:`psyneulink.core.batched.likelihood.histogram_log_likelihood` for
        ``categorical_dims`` / ``bins`` / ``bin_range`` / smoothing semantics.

        Returns a scalar for a single parameter set, else one log-likelihood per
        parameter set.
        """
        from psyneulink.core.batched.likelihood import histogram_log_likelihood

        device = _BACKEND_DEVICES.get(self.backend)
        keep_device = device == "cuda"

        result = self.run(
            inputs,
            parameter_sets,
            num_estimates,
            subject_slices=subject_slices,
            seed=seed,
            common_random_numbers=common_random_numbers,
            strict_truncation=strict_truncation,
            keep_device_values=keep_device,
            triton_launch_options=triton_launch_options,
        )

        # values: [parameter_set, subject, trial, estimate, outcome].  Collapse
        # (parameter_set, subject) into one leading "lane" axis for the likelihood
        # and select the outcome columns that align with ``data``.
        values = result.values
        if outcome_indices is not None:
            idx = list(outcome_indices)
            values = values[..., idx] if not keep_device else values.index_select(-1, _as_long(values, idx))

        n_param, n_subject = values.shape[0], values.shape[1]
        lanes = values.reshape(n_param * n_subject, *values.shape[2:])

        ll = histogram_log_likelihood(
            lanes,
            data,
            categorical_dims,
            bins=bins,
            bin_range=bin_range,
            smoothing_sigma=smoothing_sigma,
            pseudocount=pseudocount,
            categorical_cardinalities=categorical_cardinalities,
            include_mask=include_mask,
        )
        # ll is [n_param * n_subject] (or scalar for the 1x1 case); sum over
        # subjects to get one log-likelihood per parameter set.
        import numpy as np

        ll = np.asarray(ll).reshape(n_param, n_subject).sum(axis=1)
        if ll.shape[0] == 1:
            return float(ll[0])
        return ll


def _as_long(tensor_like, idx):
    import torch

    return torch.as_tensor(idx, dtype=torch.long, device=tensor_like.device)


def _validate_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise BatchedCompileError(f"Unknown batched backend '{backend}'.")
