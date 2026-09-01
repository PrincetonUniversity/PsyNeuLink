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
        initial_states=None,
        return_final_states: bool = False,
        rng_trial_offset: int = 0,
        rng_sequence_trials: int | None = None,
        triton_launch_options: Mapping | None = None,
        _defer_device_checks: bool = False,
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
            initial_states=initial_states,
            return_final_states=return_final_states,
            rng_trial_offset=rng_trial_offset,
            rng_sequence_trials=rng_sequence_trials,
            kernel_ir=self.kernel_ir,
            component_bindings=self.component_bindings,
            launch_options=triton_launch_options,
            defer_device_checks=_defer_device_checks,
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

    def conditioned_log_likelihood(
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
        """Sequential histogram likelihood with persistent-state conditioning.

        This experimental path launches one complete coupled trial at a time.
        It scores the observed choice/outcome, resamples the terminal retained
        states with those observation weights, and only then launches the next
        trial. Thus each state lane approximates
        ``p(state_t | observed outcomes before t)`` instead of carrying an
        unconditional simulated history through the whole sequence.

        It is currently intended for the CSI co-evolving LCA/DDM model. The
        state transport itself is general, but a general PEC contract must also
        specify how latent state, observation kernels, missing observations,
        and resampling interact for arbitrary models.
        """

        from collections.abc import Mapping as MappingABC

        import numpy as np
        import torch

        from psyneulink.core.batched.graph import COEVOLVING_GRAPH_FUSION
        from psyneulink.core.batched.ir import BatchedTrialParameter
        from psyneulink.core.batched.likelihood import histogram_observation_weights
        from psyneulink.core.batched.prep import normalize_parameter_sets

        if self.kernel_ir.fusion_kind != COEVOLVING_GRAPH_FUSION:
            raise BatchedCompileError(
                "conditioned_log_likelihood is currently enabled only for a "
                "co-evolving batched graph (the CSI LCA/DDM prototype)."
            )
        if pseudocount != 0.0:
            raise ValueError(
                "A histogram pseudocount has no particle ancestry to resample; "
                "conditioned_log_likelihood currently requires pseudocount=0."
            )
        if categorical_cardinalities is not None and pseudocount == 0.0:
            # It has no effect without pseudocounts, matching histogram_likelihood.
            categorical_cardinalities = None
        if subject_slices is not None:
            raise NotImplementedError(
                "The CSI conditioned-likelihood prototype currently accepts one "
                "contiguous subject sequence per call (subject_slices=None)."
            )
        if not isinstance(inputs, MappingABC):
            raise TypeError(
                "conditioned_log_likelihood requires node-keyed input sequences."
            )

        exp_data = np.asarray(data, dtype=float)
        if exp_data.ndim != 2:
            raise ValueError(
                f"data must be 2D [trial, outcome], got shape {exp_data.shape}."
            )
        num_trials = exp_data.shape[0]
        mask = (
            np.ones(num_trials, dtype=bool)
            if include_mask is None
            else np.asarray(include_mask, dtype=bool).reshape(-1)
        )
        if mask.shape[0] != num_trials:
            raise ValueError(
                f"include_mask has length {mask.shape[0]} but data has "
                f"{num_trials} trials."
            )
        if num_trials == 0:
            raise ValueError("conditioned_log_likelihood requires at least one trial.")

        parameter_rows = normalize_parameter_sets(parameter_sets, self.ir)
        state = None
        log_likelihood = None
        resampling_generator = None
        deferred_checks = None
        defer_device_checks = self.backend == "triton"

        for trial_index in range(num_trials):
            trial_inputs = {
                key: _slice_conditioned_trial(value, trial_index, num_trials)
                for key, value in inputs.items()
            }
            trial_parameter_rows = []
            for row in parameter_rows:
                trial_row = {}
                for name, value in row.items():
                    if isinstance(value, BatchedTrialParameter):
                        values = np.asarray(value.values)
                        if values.ndim == 1:
                            trial_row[name] = float(values[trial_index])
                        elif values.ndim == 2:
                            trial_row[name] = BatchedTrialParameter(
                                values[:, trial_index : trial_index + 1]
                            )
                        else:
                            raise ValueError(
                                f"Trial-varying parameter '{name}' must be a trial "
                                "vector or [subject, trial] array."
                            )
                    else:
                        trial_row[name] = value
                trial_parameter_rows.append(trial_row)

            result = self.run(
                trial_inputs,
                trial_parameter_rows,
                num_estimates,
                seed=seed,
                common_random_numbers=common_random_numbers,
                strict_truncation=strict_truncation,
                keep_device_values=True,
                initial_states=state,
                return_final_states=True,
                rng_trial_offset=trial_index,
                rng_sequence_trials=num_trials,
                triton_launch_options=triton_launch_options,
                _defer_device_checks=defer_device_checks,
            )
            if defer_device_checks:
                trial_checks = result.metadata["_deferred_device_checks"]
                if deferred_checks is None:
                    deferred_checks = trial_checks
                else:
                    deferred_checks["nonfinite_count"] = (
                        deferred_checks["nonfinite_count"]
                        + trial_checks["nonfinite_count"]
                    )
                    if deferred_checks["diagnostic_sums"] is not None:
                        deferred_checks["diagnostic_sums"] = (
                            deferred_checks["diagnostic_sums"]
                            + trial_checks["diagnostic_sums"]
                        )
                        deferred_checks["diagnostic_count"] += trial_checks[
                            "diagnostic_count"
                        ]
            outcomes = result.values[:, :, 0]
            if outcome_indices is not None:
                outcomes = outcomes.index_select(
                    -1,
                    torch.as_tensor(
                        list(outcome_indices), dtype=torch.long, device=outcomes.device
                    ),
                )
            weights, density = histogram_observation_weights(
                outcomes,
                exp_data,
                trial_index,
                categorical_dims,
                bins=bins,
                bin_range=bin_range,
                smoothing_sigma=smoothing_sigma,
            )
            if log_likelihood is None:
                log_likelihood = torch.zeros_like(density)
                resampling_generator = torch.Generator(device=outcomes.device)
                resampling_generator.manual_seed(
                    (0 if seed is None else int(seed)) ^ 0x5EED5EED
                )
            if mask[trial_index]:
                log_likelihood = log_likelihood + torch.log(density)

            terminal_state = result.metadata["final_states"]
            flat_weights = weights.reshape(-1, num_estimates)
            totals = flat_weights.sum(dim=-1, keepdim=True)
            normalized = torch.where(
                totals > 0,
                flat_weights / torch.clamp(totals, min=torch.finfo(weights.dtype).tiny),
                torch.full_like(flat_weights, 1.0 / float(num_estimates)),
            )
            ancestors = torch.multinomial(
                normalized,
                num_samples=num_estimates,
                replacement=True,
                generator=resampling_generator,
            )
            flat_state = terminal_state.reshape(
                -1, num_estimates, terminal_state.shape[-1]
            )
            state = torch.gather(
                flat_state,
                1,
                ancestors[..., None].expand(-1, -1, flat_state.shape[-1]),
            ).reshape_as(terminal_state)

        assert log_likelihood is not None
        if defer_device_checks:
            assert deferred_checks is not None
            from psyneulink.core.batched.backend.triton.runtime import (
                finalize_deferred_device_checks,
            )

            finalize_deferred_device_checks(
                deferred_checks,
                max_steps=self.ir.max_steps,
                strict_truncation=strict_truncation,
            )
        values = log_likelihood.detach().cpu().numpy().sum(axis=1)
        if values.shape[0] == 1:
            return float(values[0])
        return values

    def deterministic_history_log_likelihood(
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
        return_debug: bool = False,
    ):
        """CSI likelihood specialized for deterministic observed LCA history.

        The persistent LCA has no process noise in the fitted CSI model, so its
        state at each observed RT endpoint is a deterministic function of the
        parameters and preceding observations.  This path computes that history
        once per parameter set, stores the resulting within-trial DDM drift
        paths, and then simulates all DDM estimates in one parallel GPU launch.

        This is intentionally fail-closed to the authenticated CSI graph.  Use
        :meth:`conditioned_log_likelihood` for a stochastic persistent state or
        for a histogram-bin interpretation that marginalizes possible endpoint
        histories.
        """

        if self.backend != "triton":
            raise BatchedCompileError(
                "deterministic_history_log_likelihood currently requires the "
                "CUDA Triton backend."
            )
        if subject_slices is not None:
            raise NotImplementedError(
                "CSI deterministic-history likelihood currently accepts one "
                "contiguous subject sequence per call (subject_slices=None)."
            )
        if outcome_indices is not None and list(outcome_indices) != [0, 1]:
            raise ValueError(
                "CSI deterministic-history likelihood requires the plan's "
                "decision and response-time outputs in that order."
            )

        from psyneulink.core.batched.backend.triton.csi_deterministic import (
            run_csi_deterministic_history_likelihood,
        )

        return run_csi_deterministic_history_likelihood(
            self.ir,
            inputs,
            parameter_sets,
            num_estimates,
            data,
            categorical_dims,
            bins=bins,
            bin_range=bin_range,
            smoothing_sigma=smoothing_sigma,
            pseudocount=pseudocount,
            categorical_cardinalities=categorical_cardinalities,
            include_mask=include_mask,
            seed=seed,
            common_random_numbers=common_random_numbers,
            strict_truncation=strict_truncation,
            component_bindings=self.component_bindings,
            launch_options=triton_launch_options,
            return_debug=return_debug,
        )


def _as_long(tensor_like, idx):
    import torch

    return torch.as_tensor(idx, dtype=torch.long, device=tensor_like.device)


def _slice_conditioned_trial(value, trial_index: int, num_trials: int):
    """Slice one trial from the raw, pre-subject-splitting input sequence."""

    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0 or array.shape[0] != num_trials:
        raise ValueError(
            "Each conditioned-likelihood input must have the data trial axis "
            f"first (expected {num_trials}, got shape {array.shape})."
        )
    return array[trial_index : trial_index + 1]


def _validate_backend(backend: str) -> None:
    if backend not in _SUPPORTED_BACKENDS:
        raise BatchedCompileError(f"Unknown batched backend '{backend}'.")
