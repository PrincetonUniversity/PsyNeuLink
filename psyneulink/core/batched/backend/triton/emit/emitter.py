"""The `TritonGraphEmitter` — lowers backend-neutral KernelIR ops to Triton source.

The emitter is intentionally Triton-specific; KernelIR stays free of `tl.*`
syntax and source fragments so another backend can lower the same ops.  The
class is split across mixins for maintainability (`LaneEmitMixin` in `lanes.py`,
`OpEmitMixin` in `ops.py`); they share this class's mutable state.  Component
implementations are resolved from the immutable per-plan spec snapshot via the
`spec_key` op attributes.
"""

from __future__ import annotations

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
    STATELESS_GRAPH_FUSION,
)
from psyneulink.core.batched.kernel_ir import (
    KernelIR,
    component_symbol,
    diag_slots,
    validate_kernel_ir,
)
from psyneulink.core.batched.backend.triton.api import (
    TritonEmitContext,
    TritonOpCall,
    TritonOpTemplate,
)
from psyneulink.core.batched.backend.triton.source_builder import (
    SourceBuilder,
    emit_triton_function_header,
    emit_triton_imports,
)
from psyneulink.core.batched.backend.triton.emit._helpers import float_literal
from psyneulink.core.batched.backend.triton.emit.lanes import LaneEmitMixin
from psyneulink.core.batched.backend.triton.emit.ops import OpEmitMixin
from psyneulink.core.batched.specs import ElementwiseFunctionSpec


_KERNEL_NAMES = {
    STATELESS_GRAPH_FUSION: "pnl_batched_stateless_graph_kernel",
    DDM_GRAPH_FUSION: "pnl_batched_ddm_graph_kernel",
    STATEFUL_GRAPH_FUSION: "pnl_batched_stateful_graph_kernel",
    COEVOLVING_GRAPH_FUSION: "pnl_batched_coevolving_graph_kernel",
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
        self.effective_parameter_vars: dict[int, str] = {}
        self.value_vars: dict[str, list[str]] = {}
        # Dynamic consideration-set emission temporarily narrows the ordinary
        # lane mask and supplies a component-local execution clock while a
        # member body is emitted.  Outside that region both retain their
        # ordinary whole-kernel defaults.
        self.dynamic_active_mask = "mask"
        self.dynamic_execution_index: str | None = None
        # The typed program and its currently materialized scheduler slots are
        # available while a dynamic member body is emitted.  Stateful step
        # adapters use them to bind per-trial state and the exact pre-increment
        # RNG clock declared for that member.
        self.dynamic_program = None
        self.dynamic_slot_vars = None
        self.dynamic_single_execution_component_ids = frozenset()
        self.dynamic_fuel_bounded_component_ids = frozenset()
        self.dynamic_normal_cache_vars: dict[str, str] = {}
        # Explicit effective-parameter inputs sampled by the currently emitted
        # dynamic StepMechanism.  Component adapters resolve these by typed
        # target identity; the mapping is empty outside that one member body.
        self.dynamic_sampled_effective_parameters: dict[int, str] = {}
        self.dynamic_consumed_effective_parameter_ids: set[int] = set()
        self.rng_stream_slot: dict[str, int] = {}
        self.rng_stream_count = 0
        self.output_cursor = 0
        self.lane_out_emitted = False
        self.diag_slot_count = len(diag_slots(kernel))
        self.diag_lane_emitted = False

    def emit(self) -> str:
        # KernelIR attrs are mapping-valued for an extensible public schema.
        # Revalidate cross-op identity/effect invariants at the backend boundary
        # so post-construction mapping mutation cannot redirect retained state.
        validate_kernel_ir(self.kernel)
        if not self.kernel.executable:
            raise ValueError(
                "Cannot emit Triton source for declaration-only, non-executable "
                "KernelIR."
            )
        self._index_rng_streams()
        with self.builder.indent():
            self._emit_lane_decode()
            if self.kernel.fusion_kind in {
                STATEFUL_GRAPH_FUSION,
                COEVOLVING_GRAPH_FUSION,
            }:
                # Stateful lanes loop over trials internally.  Trial zero is
                # needed for parameter-dependent lane-state initialization;
                # the values are reloaded for each trial inside ForTrials.
                self.builder.line("trial_idx = 0")
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
        # Register dependencies first so their @triton.jit device functions are
        # emitted ahead of this template (which calls them by name).
        for dependency in template.dependencies:
            self.register_template(dependency)
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
        for idx, _ in enumerate(self.kernel.params):
            args.extend(
                (
                    f"param_{idx}_set_stride: tl.constexpr",
                    f"param_{idx}_trial_stride: tl.constexpr",
                )
            )
        args.append("out")
        # Per-lane diagnostic buffer (e.g. DDM truncation flags); only present
        # when the kernel emits StoreFlag ops, so diagnostic-free kernels (the
        # stateless graph) keep their original signature.
        if self.diag_slot_count:
            args.append("diag")
        args.extend(
            [
                "total_lanes: tl.constexpr",
                "num_subjects: tl.constexpr",
            ]
        )

        if self.kernel.fusion_kind in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION):
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

    def _emit_params(self, *, trial_varying_only: bool = False) -> None:
        for idx, param_spec in enumerate(self.kernel.params):
            var = f"param_{idx}_value"
            self.param_vars[param_spec.name] = var
            default = float_literal(param_spec.default)
            line = (
                f"{var} = tl.load(param_{idx} + "
                f"param_idx * param_{idx}_set_stride + "
                f"(subject_idx * num_trials + trial_idx) * "
                f"param_{idx}_trial_stride, mask=mask, other={default})"
            )
            if trial_varying_only:
                with self.builder.block(f"if param_{idx}_trial_stride"):
                    self.builder.line(line)
            else:
                self.builder.line(line)
        if self.kernel.params:
            self.builder.line()

    def _emit_initialize_state(self) -> None:
        state_slots: dict[int, int] = {}
        for state in self.kernel.states:
            node = self.graph.node(state.node)
            component_id = (
                state.component_id
                if state.component_id >= 0
                else node.component_id
            )
            if component_id < 0:
                component_id = self.graph.nodes.index(node)
            state_slot = state_slots.get(component_id, 0)
            state_slots[component_id] = state_slot + 1
            state_symbol = f"n{component_id}_state_{state_slot}"
            for idx, value in enumerate(state.initial_value):
                var = f"{state_symbol}_{idx}"
                self.state_vars[(state.name, idx)] = var
                self._emit_state_initializer_value(state, idx, value, var)
        if self.kernel.states:
            self.builder.line()

    def _emit_initialize_effective_parameter(self, op: KernelOp) -> None:
        """Declare one lane-persistent effective value outside ``ForTrials``.

        ``value_vars`` only records source-level value aliases.  Modulation needs
        an actual Triton variable declared before the trial loop so assigning the
        same identifier inside the loop creates loop-carried storage.
        """

        effective_parameter_id = op.attrs["effective_parameter_id"]
        if effective_parameter_id in self.effective_parameter_vars:
            raise ValueError(
                "Triton effective parameter "
                f"{effective_parameter_id} was initialized more than once."
            )
        output = op.outputs[0]
        storage_var = self._component_vars(output.name, output.width)[0]
        initial_value = op.attrs["initial_modulation_value"][0]
        self.builder.line(
            f"{storage_var} = tl.full((BLOCK,), "
            f"{float_literal(initial_value)}, tl.float32)"
        )
        self.effective_parameter_vars[effective_parameter_id] = storage_var
        self._set_value(output.name, [storage_var])
        self.builder.line()

    def _emit_reset_state(self, op: KernelOp) -> None:
        states_by_id = {state.state_id: state for state in self.kernel.states}
        self.builder.line(
            f"# reset component {op.attrs['component_id']} state at trial start"
        )
        for state_id, output in zip(op.attrs["state_ids"], op.outputs):
            state = states_by_id[state_id]
            state_vars = []
            for idx, value in enumerate(state.initial_value):
                var = self.state_vars[(state.name, idx)]
                self._emit_state_initializer_value(state, idx, value, var)
                state_vars.append(var)
            self._set_value(output.name, state_vars)
        self.builder.line()

    def _emit_state_initializer_value(self, state, index: int, value, output: str) -> None:
        initializer = state.function_initializer
        if initializer is None:
            self.builder.line(
                f"{output} = tl.full((BLOCK,), {float_literal(value)}, tl.float32)"
            )
            return
        self._emit_state_function_initializer(initializer, index, output)

    def _emit_state_function_initializer(self, initializer, index: int, output: str) -> None:
        spec = self.kernel.op_specs.lookup_spec(initializer.spec_key)
        if not isinstance(spec, ElementwiseFunctionSpec) or spec.triton_template is None:
            raise ValueError(
                "Batched state function initializer requires a registered "
                f"elementwise Triton implementation, got '{initializer.spec_key}'."
            )
        if len(initializer.input_value) <= index:
            raise ValueError(
                "Batched state function initializer input width does not match "
                "its state width."
            )
        args = [
            "tl.full((BLOCK,), "
            f"{float_literal(initializer.input_value[index])}, tl.float32)"
        ]
        for binding in spec.params:
            try:
                public_name = initializer.params[binding.arg]
                args.append(self.param_vars[public_name])
            except KeyError as error:
                raise ValueError(
                    "Batched state function initializer has no parameter binding "
                    f"for '{binding.arg}'."
                ) from error
        TritonEmitContext(self).emit_call(
            TritonOpCall(
                template=spec.triton_template,
                outputs=(output,),
                args=tuple(args),
            )
        )

    def component_symbol(self, node_spec) -> str:
        return component_symbol(self.graph, node_spec)

    def sampled_effective_parameter(
        self,
        node_spec,
        target_parameter: str,
    ) -> str | None:
        """Resolve one explicitly sampled dynamic effective parameter.

        The dynamic KernelIR carries effective values by numeric identity.  A
        component adapter names the parameter it implements (for example the
        DDM ``threshold``); this boundary cross-checks both before returning the
        currently bound Triton variable.
        """

        matches = []
        for parameter in self.kernel.effective_parameters:
            value = self.dynamic_sampled_effective_parameters.get(
                parameter.effective_parameter_id
            )
            if value is None:
                continue
            if (
                parameter.target_component_id == node_spec.component_id
                and parameter.target == node_spec.name
                and parameter.target_parameter == target_parameter
            ):
                matches.append((parameter.effective_parameter_id, value))
        if len(matches) > 1:
            raise ValueError(
                "Triton dynamic StepMechanism sampled duplicate effective "
                f"parameters for '{node_spec.name}.{target_parameter}'."
            )
        if not matches:
            return None
        parameter_id, value = matches[0]
        self.dynamic_consumed_effective_parameter_ids.add(parameter_id)
        return value


def triton_graph_kernel_source(kernel: KernelIR) -> str:
    """Emit inspectable Triton source for a generated graph kernel."""

    return TritonGraphEmitter(kernel).emit()
