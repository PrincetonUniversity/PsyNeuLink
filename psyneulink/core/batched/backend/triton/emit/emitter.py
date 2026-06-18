"""The `TritonGraphEmitter` — lowers backend-neutral KernelIR ops to Triton source.

The emitter is intentionally Triton-specific; KernelIR stays free of `tl.*`
syntax and source fragments so another backend can lower the same ops.  The
class is split across mixins for maintainability (`LaneEmitMixin` in `lanes.py`,
`OpEmitMixin` in `ops.py`); they share this class's mutable state.  Component
implementations are resolved from the batched op spec registry via the
`spec_key` op attributes.
"""

from __future__ import annotations

from psyneulink.core.batched import specs
from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
)
from psyneulink.core.batched.kernel_ir import KernelIR
from psyneulink.core.batched.backend.triton.api import TritonOpTemplate
from psyneulink.core.batched.backend.triton.source_builder import (
    SourceBuilder,
    emit_triton_function_header,
    emit_triton_imports,
)
from psyneulink.core.batched.backend.triton.emit._helpers import float_literal, safe_ident
from psyneulink.core.batched.backend.triton.emit.lanes import LaneEmitMixin
from psyneulink.core.batched.backend.triton.emit.ops import OpEmitMixin


_KERNEL_NAMES = {
    STATELESS_GRAPH_FUSION: "pnl_batched_stateless_graph_kernel",
    DDM_GRAPH_FUSION: "pnl_batched_ddm_graph_kernel",
    STATEFUL_GRAPH_FUSION: "pnl_batched_stateful_graph_kernel",
}


class TritonGraphEmitter(LaneEmitMixin, OpEmitMixin):
    def __init__(self, kernel: KernelIR):
        self.kernel = kernel
        self.graph = kernel.graph
        self.builder = SourceBuilder()
        self.templates: dict[str, TritonOpTemplate] = {}
        self.input_index = {
            input_spec.node: idx
            for idx, input_spec in enumerate(kernel.inputs)
        }
        self.param_vars: dict[str, str] = {}
        self.state_vars: dict[tuple[str, int], str] = {}
        self.value_vars: dict[str, list[str]] = {}
        self.lca_stream_index: dict[str, int] = {}
        self.ddm_stream_index: dict[str, int] = {}
        self.lca_stream_count = 0
        self.ddm_stream_count = 0
        self.output_cursor = 0
        self.lane_out_emitted = False

    def emit(self) -> str:
        specs.ensure_builtin_specs()
        self._index_rng_streams()
        with self.builder.indent():
            self._emit_lane_decode()
            self._emit_params()
            self._emit_top_level_ops()
        body_source = self.builder.render()

        module_builder = SourceBuilder()
        emit_triton_imports(module_builder)
        for template in self.templates.values():
            module_builder.lines(template.source.splitlines())
            module_builder.line()
            module_builder.line()
        emit_triton_function_header(
            module_builder,
            self._kernel_name(),
            self._signature_args(),
        )
        module_builder.lines(body_source.splitlines())
        return module_builder.render()

    def register_template(self, template: TritonOpTemplate) -> str:
        existing = self.templates.get(template.name)
        if existing is not None and existing.source != template.source:
            raise ValueError(f"Conflicting Triton helper template '{template.name}'.")
        self.templates[template.name] = template
        return template.name

    def _kernel_name(self) -> str:
        try:
            return _KERNEL_NAMES[self.kernel.fusion_kind]
        except KeyError as error:
            raise ValueError(
                f"Unsupported Triton graph fusion kind '{self.kernel.fusion_kind}'."
            ) from error

    def _signature_args(self) -> tuple[str, ...]:
        args = [f"input_{idx}" for idx, _ in enumerate(self.kernel.inputs)]
        args.extend(f"param_{idx}" for idx, _ in enumerate(self.kernel.params))
        args.extend(
            [
                "out",
                "total_lanes: tl.constexpr",
                "num_subjects: tl.constexpr",
            ]
        )

        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            args.extend(
                [
                    "num_estimates: tl.constexpr",
                    "num_trials",
                    "LCA_MAX_STEPS: tl.constexpr",
                    "MAX_STEPS: tl.constexpr",
                    "COMMON_RANDOM: tl.constexpr",
                    "SEED: tl.constexpr",
                    "BLOCK: tl.constexpr",
                ]
            )
            return tuple(args)

        args.extend(
            [
                "num_trials: tl.constexpr",
                "num_estimates: tl.constexpr",
            ]
        )
        if self.kernel.fusion_kind == DDM_GRAPH_FUSION:
            args.extend(
                [
                    "MAX_STEPS: tl.constexpr",
                    "COMMON_RANDOM: tl.constexpr",
                    "SEED: tl.constexpr",
                ]
            )
        args.append("BLOCK: tl.constexpr")
        return tuple(args)

    def _emit_params(self) -> None:
        for idx, param_spec in enumerate(self.kernel.params):
            var = f"param_{idx}_value"
            self.param_vars[param_spec.name] = var
            default = float_literal(param_spec.default)
            self.builder.line(
                f"{var} = tl.load(param_{idx} + param_idx, mask=mask, other={default})"
            )
        if self.kernel.params:
            self.builder.line()

    def _emit_initialize_state(self) -> None:
        for state in self.kernel.states:
            safe_state = safe_ident(state.name)
            for idx, value in enumerate(state.initial_value):
                var = f"{safe_state}_{idx}"
                self.state_vars[(state.name, idx)] = var
                self.builder.line(
                    f"{var} = tl.full((BLOCK,), {float_literal(value)}, tl.float32)"
                )
        if self.kernel.states:
            self.builder.line()


def triton_graph_kernel_source(kernel: KernelIR) -> str:
    """Emit inspectable Triton source for a generated graph kernel."""

    return TritonGraphEmitter(kernel).emit()
