"""Lane decode, RNG-base layout, and raw-input emission for the Triton emitter.

`LaneEmitMixin` holds the lane/RNG/IO methods of `TritonGraphEmitter`; it shares
the emitter's mutable state (`self.builder`, stream indices, ...) and is mixed
into the concrete emitter in `emitter.py`.
"""

from __future__ import annotations

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
)
from psyneulink.core.batched.kernel_ir import node_output_value_name
from psyneulink.core.batched.backend.triton.emit._helpers import primary_output_port_name

# Philox counter space reserved per RNG stream.  Stream identity is packed into
# the high 32 bits of the offset and the step index into the low 32, which
# `randint4x` splits back into two counter words.  The point of a fixed stride
# is that offsets -- and so the draws -- do not depend on MAX_STEPS or
# LCA_MAX_STEPS: raising a step cap for safety no longer changes results.
#
# The base must be built inside the kernel from constants and lane arithmetic.
# Passing a precomputed 64-bit base as a runtime kernel argument silently drops
# the high word on the GPU (the offset stays 32-bit), which collapses every
# stream onto the same draws.
RNG_STREAM_STRIDE = 1 << 32


class LaneEmitMixin:
    def _emit_lane_decode(self) -> None:
        self.builder.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
        self.builder.line("mask = offsets < total_lanes")
        self.builder.line("estimate_idx = offsets % num_estimates")
        self.builder.line("tmp = offsets // num_estimates")

        # Lane-persistent fusions loop trials *inside* the lane, so their lane
        # space is (parameter_set, subject, estimate) -- no trial axis.  This
        # must match how the runtime sizes `total_lanes`, or the decode divides
        # by the wrong extents and folds several parameter sets onto param 0
        # (leaving the rest of the output buffer untouched).
        if self.kernel.fusion_kind in (STATEFUL_GRAPH_FUSION, COEVOLVING_GRAPH_FUSION):
            self.builder.line("subject_idx = tmp % num_subjects")
            self.builder.line("param_idx = tmp // num_subjects")
            self.builder.line()
            return

        self.builder.line("trial_idx = tmp % num_trials")
        self.builder.line("tmp = tmp // num_trials")
        self.builder.line("subject_idx = tmp % num_subjects")
        self.builder.line("param_idx = tmp // num_subjects")
        self.builder.line()

    @property
    def _lane_rng_stride(self) -> int:
        """Philox counter space reserved per lane: one stride per stream it owns."""

        return max(1, self.rng_stream_count) * RNG_STREAM_STRIDE

    def _emit_random_base(self) -> None:
        """Emit `random_base`, the lane's 64-bit Philox offset origin.

        The lane index is widened before scaling: it is int32 arithmetic, and
        the stride is far past int32.
        """

        stride = self._lane_rng_stride
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                f"* num_trials + trial_idx).to(tl.int64) * {stride}"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = (((param_idx * num_subjects + subject_idx) "
                "* num_estimates + estimate_idx) * num_trials + trial_idx)"
                f".to(tl.int64) * {stride}"
            )

    def _emit_stateful_random_base(self) -> None:
        self._emit_random_base()
        self.builder.line()

    def _emit_trial_random_base_if_needed(self) -> None:
        if self.kernel.fusion_kind != DDM_GRAPH_FUSION:
            return
        self._emit_random_base()

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
        return self._get_value(
            node_output_value_name(
                self.graph,
                node,
                primary_output_port_name(node),
            )
        )[component_idx]

    def raw_input_value(self, node_name: str, component_idx: int = 0) -> str:
        return self._raw_input_value(node_name, component_idx)

    def _rng_stream_offset(self, node_name: str, component_idx: int = 0) -> int:
        """Absolute Philox offset of one of a node's streams, from its lane base."""

        return (self.rng_stream_slot[node_name] + component_idx) * RNG_STREAM_STRIDE

    def rng_stream_offset(self, node_name: str, component_idx: int = 0) -> int:
        return self._rng_stream_offset(node_name, component_idx)

    def _rng_base(self, node_name: str) -> str:
        offset = self._rng_stream_offset(node_name)
        return "random_base" if offset == 0 else f"random_base + {offset}"

    def rng_base(self, node_name: str) -> str:
        return self._rng_base(node_name)

    def _index_rng_streams(self) -> None:
        # One flat pool: every stream gets the same stride, so which step cap
        # bounds a stream no longer affects where it lives.
        component_ids = tuple(
            stream.component_id for stream in self.kernel.rng_streams
        )
        node_names = tuple(stream.node for stream in self.kernel.rng_streams)
        if (
            len(set(component_ids)) != len(component_ids)
            or len(set(node_names)) != len(node_names)
        ):
            raise ValueError(
                "Triton RNG lowering supports at most one stream declaration "
                "per component."
            )
        stream_slot = {}
        stream_count = 0
        for stream in self.kernel.rng_streams:
            stream_slot[stream.node] = stream_count
            stream_count += stream.width
        self.rng_stream_slot = stream_slot
        self.rng_stream_count = stream_count
