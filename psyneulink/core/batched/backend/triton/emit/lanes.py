"""Lane decode, RNG-base layout, and raw-input emission for the Triton emitter.

`LaneEmitMixin` holds the lane/RNG/IO methods of `TritonGraphEmitter`; it shares
the emitter's mutable state (`self.builder`, stream indices, ...) and is mixed
into the concrete emitter in `emitter.py`.
"""

from __future__ import annotations

import ast
from dataclasses import replace

from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    DDM_GRAPH_FUSION,
    STATEFUL_GRAPH_FUSION,
)
from psyneulink.core.batched.kernel_ir import node_output_value_name
from psyneulink.core.batched.backend.triton.api import TritonOpTemplate
from psyneulink.core.batched.backend.triton.emit._helpers import primary_output_port_name

# Philox counter space reserved per RNG stream.  Stream identity is packed into
# the high 32 bits of the offset and a step-derived counter into the low 32,
# which `randint4x` splits back into two counter words.  Direct draws use the
# step itself; scalar scheduled normals pack an even/odd pair into one counter
# and retain the second Box-Muller result. Vector normals can instead group
# coordinates within an execution, keeping the original slot allocation.
# The point of a fixed stride
# is that offsets -- and so the draws -- do not depend on MAX_STEPS or
# LCA_MAX_STEPS: raising a step cap for safety no longer changes results.
#
# The base must be built inside the kernel from constants and lane arithmetic.
# Passing a precomputed 64-bit base as a runtime kernel argument silently drops
# the high word on the GPU (the offset stays 32-bit), which collapses every
# stream onto the same draws.
RNG_STREAM_STRIDE = 1 << 32
DEFAULT_NORMAL_RNG = "philox4x_v1"
FAST_NORMAL_RNG = "philox4x_fast_v1"

# The uniforms, clamp, log and sqrt match Triton's pair transform. Only the
# bounded-angle trigonometry changes. Keep this an explicit versioned mode:
# last-bit differences can alter a first-passage stopping step.
_FAST_NORMAL_PAIR = TritonOpTemplate(
    name="_pnl_fast_normal_pair_v1",
    arg_names=("u1", "u2"),
    source="""from triton.language.extra.cuda import libdevice as _pnl_rng_libdevice

@triton.jit
def _pnl_fast_normal_pair_v1(u1, u2):
    u1 = tl.maximum(1.0e-7, u1)
    theta = 6.283185307179586 * u2
    r = tl.sqrt(-2.0 * tl.log(u1))
    return r * _pnl_rng_libdevice.fast_cosf(theta), r * _pnl_rng_libdevice.fast_sinf(theta)
""",
)
_FAST_NORMAL_DRAW = TritonOpTemplate(
    name="_pnl_fast_normal_v1",
    arg_names=("seed", "offset", "n_rounds"),
    constexpr=("n_rounds",),
    source="""@triton.jit
def _pnl_fast_normal_v1(seed, offset, n_rounds: tl.constexpr = 10):
    u1, u2, _, _ = tl.rand4x(seed, offset, n_rounds)
    z, _ = _pnl_fast_normal_pair_v1(u1, u2)
    return z
""",
    dependencies=(_FAST_NORMAL_PAIR,),
)


def validate_normal_rng(mode):
    if not isinstance(mode, str) or mode not in ("legacy", DEFAULT_NORMAL_RNG, FAST_NORMAL_RNG):
        raise ValueError("Triton normal_rng must be 'legacy', 'philox4x_v1', or 'philox4x_fast_v1'.")
    return mode


class LaneEmitMixin:
    def _lower_normal_template(self, template):
        """Apply the selected transform to direct RNG calls in op templates.

        Standalone DDM/NormalDist bodies own their loops and call tl.randn
        directly. Rewrite only these exact calls in the backend's source copy;
        keep the registered implementation snapshot and uniform arguments intact.
        """
        if self.normal_rng != FAST_NORMAL_RNG or template.name in {
            _FAST_NORMAL_DRAW.name, _FAST_NORMAL_PAIR.name,
        }:
            return template
        tree = ast.parse(template.source)
        changed = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl"
                    and node.func.attr in {"randn", "pair_uniform_to_normal"}):
                helper = _FAST_NORMAL_DRAW if node.func.attr == "randn" else _FAST_NORMAL_PAIR
                name = self.register_template(helper)
                node.func = ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node.func)
                changed = True
        return replace(template, source=ast.unparse(tree)) if changed else template

    def _normal_pair_function(self):
        if self.normal_rng == FAST_NORMAL_RNG:
            return self.register_template(_FAST_NORMAL_PAIR)
        return "tl.pair_uniform_to_normal"

    def _normal_draw_expression(self, base):
        function = (self.register_template(_FAST_NORMAL_DRAW)
                    if self.normal_rng == FAST_NORMAL_RNG else "tl.randn")
        return f"{function}(SEED, {base})"

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
        stateful = self.kernel.fusion_kind in (
            STATEFUL_GRAPH_FUSION,
            COEVOLVING_GRAPH_FUSION,
        )
        trial_offset = "TRIAL_OFFSET" if stateful else "0"
        rng_num_trials = "RNG_NUM_TRIALS" if stateful else "num_trials"
        with self.builder.block("if COMMON_RANDOM"):
            self.builder.line(
                "random_base = ((subject_idx * num_estimates + estimate_idx) "
                f"* {rng_num_trials} + trial_idx + {trial_offset}).to(tl.int64) "
                f"* {stride}"
            )
        with self.builder.block("else"):
            self.builder.line(
                "random_base = ((((param_idx * num_subjects + subject_idx) "
                f"* num_estimates + estimate_idx) * {rng_num_trials} + trial_idx)"
                f".to(tl.int64) + {trial_offset}) "
                f"* {stride}"
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

    def normal_draw(self, node_name: str, step: str) -> str:
        """Return a standard-normal draw, reusing a Philox pair when legal.

        Pairing changes the exact seed-to-sample mapping relative to calling
        ``tl.randn`` for every step, but retains deterministic replay, stream
        separation, and common-random-number alignment across parameter lanes.
        """

        spare = self.dynamic_normal_cache_vars.get(node_name)
        rng_base = self._rng_base(node_name)
        if spare is None:
            return self._normal_draw_expression(f"{rng_base} + {step}")

        symbol = f"dynamic_rng_{self.rng_stream_slot[node_name]}"
        phase = f"{symbol}_phase"
        refresh = f"{symbol}_refresh"
        uniforms = tuple(f"{symbol}_uniform_{index}" for index in range(4))
        draws = tuple(f"{symbol}_draw_{index}" for index in range(2))
        draw = f"{symbol}_selected"
        self.builder.line(f"{phase} = {step} & 1")
        self.builder.line(
            f"{refresh} = {self.dynamic_active_mask} & ({phase} == 0)"
        )
        # Odd executions consume the spare normal retained from the preceding
        # even execution.  The selected value itself is iteration-local, so
        # only one extra vector must remain live across the scheduler loop.
        self.builder.line(f"{draw} = {spare}")
        with self.builder.block(
            f"if tl.max(tl.where({refresh}, 1, 0)) > 0"
        ):
            self.builder.line(
                f"{', '.join(uniforms)} = tl.rand4x("
                f"SEED, {rng_base} + ({step} // 2))"
            )
            self.builder.line(
                f"{', '.join(draws)} = {self._normal_pair_function()}("
                f"{uniforms[0]}, {uniforms[1]})"
            )
            self.builder.line(
                f"{spare} = tl.where({refresh}, {draws[1]}, {spare})"
            )
            self.builder.line(
                f"{draw} = tl.where({refresh}, {draws[0]}, {draw})"
            )
        return draw

    def normal_draws(self, node_name: str, step: str) -> tuple[str, ...]:
        """Draw the declared vector of independent normals at one RNG clock.

        philox4x_v1 can use both Box-Muller pairs from one Philox invocation.
        Groups belong to one component execution, never to different clocks or
        owners. Group j starts at the *original* stream slot 4*j; the other
        reserved slots remain unused. Keeping the stream inventory and step
        addressing unchanged prevents overlap with other components and keeps
        replay independent of launch geometry, step caps and scheduler masks.

        No spare values survive this execution. A width-one request uses the
        direct stream; scalar temporal pairing is a separate contract provided
        by normal_draw(). philox4x_fast_v1 preserves these uniform addresses,
        but uses the bounded-angle transform for all widths.
        """

        width = self.rng_stream_width[node_name]
        if self.normal_rng == "legacy":
            return tuple(
                f"tl.randn(SEED, random_base + {self._rng_stream_offset(node_name, i)} + {step})"
                for i in range(width)
            )
        result = []
        for start in range(0, width, 4):
            offset = self._rng_stream_offset(node_name, start)
            base = f"random_base + {offset} + {step}"
            count = min(4, width - start)
            if count == 1:
                result.append(self._normal_draw_expression(base))
                continue
            stem = f"vector_rng_{self.rng_stream_slot[node_name] + start}"
            uniforms = tuple(f"{stem}_u{i}" for i in range(4))
            draws = tuple(f"{stem}_z{i}" for i in range(4))
            self.builder.line(f"{', '.join(uniforms)} = tl.rand4x(SEED, {base})")
            for pair in range((count + 1) // 2):
                i = pair * 2
                self.builder.line(
                    f"{draws[i]}, {draws[i + 1]} = {self._normal_pair_function()}("
                    f"{uniforms[i]}, {uniforms[i + 1]})"
                )
            result.extend(draws[:count])
        return tuple(result)

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
        self.rng_stream_width = {stream.node: stream.width for stream in self.kernel.rng_streams}
