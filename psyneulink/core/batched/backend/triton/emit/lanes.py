"""Lane decode, RNG-base layout, and raw-input emission for the Triton emitter.

`LaneEmitMixin` holds the lane/RNG/IO methods of `TritonGraphEmitter`; it shares
the emitter's mutable state (`self.builder`, stream indices, ...) and is mixed
into the concrete emitter in `emitter.py`.
"""

from __future__ import annotations

from psyneulink.core.batched.graph import (
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
)
from psyneulink.core.batched.backend.triton.emit._helpers import primary_output_port_name


class LaneEmitMixin:
    def _emit_lane_decode(self) -> None:
        self.builder.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = offsets < total_lanes")
        self.builder.line("estimate_idx = offsets % num_estimates")
        self.builder.line("tmp = offsets // num_estimates")

        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            self.builder.line("subject_idx = tmp % num_subjects")
            self.builder.line("param_idx = tmp // num_subjects")
            self.builder.line()
            return

        self.builder.line("trial_idx = tmp % num_trials")
        self.builder.line("tmp = tmp // num_trials")
        self.builder.line("subject_idx = tmp % num_subjects")
        self.builder.line("param_idx = tmp // num_subjects")
        self.builder.line()

    def _emit_stateful_random_base(self) -> None:
        random_stride = (
            f"({self.lca_stream_count}) * LCA_MAX_STEPS "
            f"+ ({self.ddm_stream_count}) * MAX_STEPS"
        )
        self.builder.line(f"random_stride = {random_stride}")
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                "* num_trials + trial_idx) * random_stride"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = (((param_idx * num_subjects + subject_idx) "
                "* num_estimates + estimate_idx) * num_trials + trial_idx) "
                "* random_stride"
            )
        self.builder.line()

    def _emit_trial_random_base_if_needed(self) -> None:
        if self.kernel.fusion_kind != DDM_GRAPH_FUSION:
            return
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                "* num_trials + trial_idx) * MAX_STEPS"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = (((param_idx * num_subjects + subject_idx) "
                "* num_estimates + estimate_idx) * num_trials + trial_idx) "
                "* MAX_STEPS"
            )

    def emit_trial_random_base_if_needed(self) -> None:
        self._emit_trial_random_base_if_needed()

    def _emit_lane_out(self) -> None:
        output_width = sum(output.width for output in self.kernel.outputs)
        self.builder.line(
            "lane_out = (((param_idx * num_subjects + subject_idx) "
            "* num_trials + trial_idx) * "
            f"num_estimates + estimate_idx) * {output_width}"
        )
        self.lane_out_emitted = True

    def _emit_diag_lane(self) -> None:
        self.builder.line(
            "diag_lane = (((param_idx * num_subjects + subject_idx) "
            "* num_trials + trial_idx) * "
            f"num_estimates + estimate_idx) * {self.diag_slot_count}"
        )
        self.diag_lane_emitted = True

    def _raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        if node_name in self.input_index:
            input_spec = self.kernel.inputs[self.input_index[node_name]]
            base = f"(subject_idx * num_trials + trial_idx) * {input_spec.width}"
            return (
                f"tl.load(input_{self.input_index[node_name]} + {base} + {component_idx}, "
                "mask=mask, other=0.0)"
            )
        node = self.graph.node(node_name)
        return self._get_value(f"{node_name}:{primary_output_port_name(node)}")[component_idx]

    def raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        return self._raw_input_value(node_name, component_idx)

    def _rng_base(self, node_name: str) -> str:
        if self.kernel.fusion_kind == STATEFUL_GRAPH_FUSION:
            return (
                f"random_base + ({self.lca_stream_count}) * LCA_MAX_STEPS "
                f"+ ({self.ddm_stream_index[node_name]}) * MAX_STEPS"
            )
        return "random_base"

    def rng_base(self, node_name: str) -> str:
        return self._rng_base(node_name)

    def _index_rng_streams(self) -> None:
        for stream in self.kernel.rng_streams:
            if stream.step_extent == "LCA_MAX_STEPS":
                self.lca_stream_index[stream.node] = self.lca_stream_count
                self.lca_stream_count += stream.width
            elif stream.step_extent == "MAX_STEPS":
                self.ddm_stream_index[stream.node] = self.ddm_stream_count
                self.ddm_stream_count += stream.width
