"""Vector Gaussian lowering: counter separation, distributions and replay."""

from types import SimpleNamespace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.backend.triton.cache import interpret_scope, load_triton_kernel_module
from psyneulink.core.batched.backend.triton.emit.lanes import LaneEmitMixin, RNG_STREAM_STRIDE
from psyneulink.core.batched.backend.triton.emit.emitter import TritonGraphEmitter
from psyneulink.core.batched.backend.triton.runtime import _import_torch_triton
from psyneulink.core.batched.backend.triton.source_builder import SourceBuilder


pytestmark = [pytest.mark.batched, pytest.mark.composition, pytest.mark.triton]


class _DrawEmitter(LaneEmitMixin):
    """Exercise the backend API without depending on an LCA adapter."""

    def __init__(self, widths, mode):
        self.templates = {}
        self.kernel = SimpleNamespace(rng_streams=tuple(
            SimpleNamespace(node=f"owner{i}", component_id=i, width=width)
            for i, width in enumerate(widths)
        ))
        self.normal_rng = mode
        self.builder = SourceBuilder()
        self._index_rng_streams()

    register_template = TritonGraphEmitter.register_template

    def source(self):
        b = self.builder
        b.lines(["import triton", "import triton.language as tl", "", "@triton.jit"])
        with b.block("def draws(out, old, N: tl.constexpr, STEPS: tl.constexpr, SEED: tl.constexpr, BLOCK: tl.constexpr)"):
            b.line("offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)")
            b.line("mask = offsets < N * STEPS")
            b.line("step = offsets % STEPS")
            b.line(f"random_base = (offsets // STEPS).to(tl.int64) * {self._lane_rng_stride}")
            for owner, slot in self.rng_stream_slot.items():
                for i, value in enumerate(self.normal_draws(owner, "step")):
                    index = slot + i
                    address = f"offsets * {self.rng_stream_count} + {index}"
                    b.line(f"tl.store(out + {address}, {value}, mask=mask)")
                    # The pre-optimization ABI: one randn per coordinate and
                    # execution, each coordinate owning a full 32-bit counter.
                    b.line(f"reference = tl.randn(SEED, random_base + {index * RNG_STREAM_STRIDE} + step)")
                    b.line(f"tl.store(old + {address}, reference, mask=mask)")
        return ("import triton\nimport triton.language as tl\n\n"
                + "\n\n".join(t.source for t in self.templates.values()) + "\n\n" + b.render())


@pytest.mark.parametrize("widths", [(1,), (2,), (3, 5), (4, 7), (32,)])
@pytest.mark.parametrize("mode", ["legacy", "philox4x_v1", "philox4x_fast_v1"])
def test_vector_normals_distribution_and_counter_abi(batched_backend, widths, mode):
    interpret = batched_backend == "triton_cpu"
    if interpret and mode == "philox4x_fast_v1":
        pytest.skip("CUDA bounded-angle intrinsics require the compiled GPU")
    torch, triton = _import_torch_triton(interpret)
    device = "cpu" if interpret else "cuda"
    # Multiple executions and owners expose reused counters/Box-Muller results.
    # An incomplete block also checks that launch padding cannot shift draws.
    n, steps = (1031 if interpret else 16387), 3
    shape = (n, steps, sum(widths))
    values = torch.empty(shape, device=device)
    old = torch.empty_like(values)
    source = _DrawEmitter(widths, mode).source()
    with interpret_scope(interpret):
        module = load_triton_kernel_module(source, "vector_normal_test", "test", interpret=interpret)
        module.draws[(triton.cdiv(n * steps, 128),)](
            values, old, n, steps, (1 << 40) + 29, 128,
        )
    actual, reference = values.cpu().numpy(), old.cpu().numpy()
    assert np.all(np.isfinite(actual))
    if mode == "legacy":
        np.testing.assert_array_equal(actual, reference)
    else:
        start = 0
        for width in widths:
            # Each four-coordinate group's first normal retains its old draw.
            if mode == "philox4x_fast_v1":
                np.testing.assert_allclose(actual[..., start:start + width:4],
                                           reference[..., start:start + width:4], rtol=0, atol=5e-6)
            else:
                np.testing.assert_array_equal(actual[..., start:start + width:4], reference[..., start:start + width:4])
            if width > 1:
                assert not np.array_equal(actual[..., start + 1], reference[..., start + 1])
            start += width
    if interpret:
        # The GPU exercises large populations; the interpreter checks the
        # counter ABI and awkward vector/block widths without costly sampling.
        return
    # Analytic standard-normal moments/tails, independent of legacy samples.
    np.testing.assert_allclose(actual.mean(axis=(0, 1)), 0., atol=.025)
    np.testing.assert_allclose(actual.var(axis=(0, 1)), 1., atol=.04)
    np.testing.assert_allclose((actual ** 4).mean(axis=(0, 1)), 3., atol=.22)
    np.testing.assert_allclose((np.abs(actual) > 2.).mean(axis=(0, 1)), .0455003, atol=.005)
    correlation = np.corrcoef(actual.reshape(n, -1).T)
    assert np.max(np.abs(correlation - np.eye(steps * sum(widths)))) < .055


def _persistent_model(backend, max_steps=64):
    lca = pnl.LCAMechanism(
        input_shapes=5, function=pnl.Logistic(gain=1.1),
        leak=.3, competition=.2, self_excitation=0.,
        noise=pnl.NormalDist(standard_deviation=.3, seed=11),
        time_step_size=.1, termination_threshold=.65,
        execute_until_finished=False, reset_stateful_function_when=pnl.Never(),
        output_ports=[pnl.RESULT, pnl.DECISION_TIME],
    )
    gate = pnl.ProcessingMechanism()
    comp = pnl.Composition()
    comp.add_nodes([lca, gate])
    comp.add_projection(sender=lca.output_ports[pnl.DECISION_TIME], receiver=gate)
    comp.scheduler.add_condition(lca, pnl.Always())
    comp.scheduler.add_condition(gate, pnl.WhenFinished(lca))
    return BatchedCompositionCompiler.compile(comp, backend=backend, max_steps=max_steps,
                                             outputs=list(lca.output_ports)), lca


@pytest.mark.parametrize("mode", ["legacy", "philox4x_v1", "philox4x_fast_v1"])
def test_vector_normals_stateful_replay_crn_and_resume(batched_backend, mode):
    if batched_backend == "triton_cpu" and mode == "philox4x_fast_v1":
        pytest.skip("CUDA bounded-angle intrinsics require the compiled GPU")
    plan, lca = _persistent_model(batched_backend)
    rows = np.array([[3., 1., 1., 1., 1.], [1., 3., 1., 1., 1.], [1., 1., 3., 1., 1.]])
    options = dict(seed=29, triton_launch_options={"normal_rng": mode}, strict_truncation=True)
    actual = plan.run({lca: rows}, [{}, {}], 37, return_final_states=True, **options)
    replay = plan.run({lca: rows}, [{}], 37, **options)
    np.testing.assert_array_equal(actual.values[:1], replay.values)
    np.testing.assert_array_equal(actual.values[0], actual.values[1])
    independent = plan.run({lca: rows}, [{}, {}], 37, common_random_numbers=False, **options)
    assert not np.array_equal(independent.values[0], independent.values[1])
    changed_seed = plan.run({lca: rows}, [{}], 37, **{**options, "seed": 30})
    assert not np.array_equal(replay.values, changed_seed.values)
    # The persistent stochastic state must survive a launch boundary and the
    # RNG must still address the same full-sequence trial/estimate coordinates.
    first = plan.run({lca: rows[:1]}, [{}, {}], 37, rng_sequence_trials=3,
                     return_final_states=True, **options)
    rest = plan.run({lca: rows[1:]}, [{}, {}], 37, rng_trial_offset=1, rng_sequence_trials=3,
                    initial_states=first.metadata["final_states"], return_final_states=True, **options)
    np.testing.assert_array_equal(np.concatenate((first.values, rest.values), axis=2), actual.values)
    np.testing.assert_array_equal(rest.metadata["final_states"], actual.metadata["final_states"])
    # Different subjects retain separate streams even with common randomness.
    subjects = plan.run({lca: np.concatenate((rows, rows))}, [{}], 37,
                        subject_slices=[slice(0, 3), slice(3, 6)], **options)
    np.testing.assert_array_equal(subjects.values[0, :1], replay.values[0])
    assert not np.array_equal(subjects.values[0, 0], subjects.values[0, 1])
    if batched_backend == "triton":
        tuned = plan.run({lca: rows}, [{}, {}], 37,
                         **{**options, "triton_launch_options": {
                             "normal_rng": mode, "block_size": 32, "num_warps": 1}})
        np.testing.assert_array_equal(tuned.values, actual.values)


@pytest.mark.triton_gpu
@pytest.mark.parametrize("seed", [17, (1 << 40) + 1234])
def test_fast_normal_numerics_moments_and_tails(seed):
    torch, triton = _import_torch_triton(False)
    n, steps = 2**20, 1
    values = torch.empty((n, steps, 4), device="cuda")
    old = torch.empty_like(values)
    module = load_triton_kernel_module(_DrawEmitter((4,), "philox4x_fast_v1").source(),
                                      "fast_normal_validation", "test")
    module.draws[(triton.cdiv(n, 128),)](values, old, n, steps, seed, 128)
    actual = values.cpu().numpy().reshape(n, 4)
    reference = old.cpu().numpy().reshape(n, 4)
    # Coordinate zero uses the exact same uniform pair as the old transform.
    np.testing.assert_allclose(actual[:, 0], reference[:, 0], rtol=0, atol=5e-6)
    np.testing.assert_allclose(actual.mean(axis=0), 0., atol=.004)
    np.testing.assert_allclose(actual.var(axis=0), 1., atol=.007)
    np.testing.assert_allclose((np.abs(actual) > 3.).mean(axis=0), .0026998, atol=.00025)
    np.testing.assert_allclose((np.abs(actual) > 4.).mean(axis=0), .0000633425, atol=.000035)
    assert np.max(np.abs(np.corrcoef(actual.T) - np.eye(4))) < .005


@pytest.mark.triton_gpu
def test_fast_scalar_pairing_and_independent_trial_replay():
    from test_batched_dynamic_terminator_acceptance import _build_scheduled_terminator

    model = _build_scheduled_terminator(noise=.25)
    plan = BatchedCompositionCompiler.compile(model.composition, backend="triton",
                                             outputs=model.outputs, max_steps=128)
    options = dict(seed=17, strict_truncation=True, return_final_states=True)
    synchronized = plan.run(model.inputs, [{}, {}], 37, **options,
                            triton_launch_options={"normal_rng": "philox4x_fast_v1"})
    independent = plan.run(model.inputs, [{}, {}], 37, **options,
                           triton_launch_options={"normal_rng": "philox4x_fast_v1",
                                                  "trial_schedule": "independent", "block_size": 32, "num_warps": 1})
    np.testing.assert_array_equal(synchronized.values, independent.values)
    np.testing.assert_array_equal(independent.values[0], independent.values[1])
    np.testing.assert_array_equal(synchronized.metadata['final_states'], independent.metadata['final_states'])


@pytest.mark.triton_gpu
@pytest.mark.parametrize("kind", ["normal", "ddm"])
def test_fast_mode_reaches_standalone_scalar_templates(kind):
    if kind == "normal":
        node = pnl.ProcessingMechanism(function=pnl.NormalDist(mean=.5, standard_deviation=.3))
    else:
        node = pnl.DDM(function=pnl.DriftDiffusionIntegrator(rate=.5, noise=.1, threshold=.1),
                       output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])
    plan = BatchedCompositionCompiler.compile(pnl.Composition(pathways=node), backend="triton", max_steps=100)
    from psyneulink.core.batched.backend.triton.graph_emit import triton_graph_kernel_source

    source = triton_graph_kernel_source(plan.kernel_ir, normal_rng="philox4x_fast_v1")
    assert "_pnl_fast_normal_v1" in source
    assert "tl.randn(" not in source
    options = dict(seed=17, strict_truncation=True, triton_launch_options={"normal_rng": "philox4x_fast_v1"})
    result = plan.run({node: [[1.], [2.]]}, [{}, {}], 1031, **options)
    replay = plan.run({node: [[1.], [2.]]}, [{}, {}], 1031, **options)
    np.testing.assert_array_equal(result.values, replay.values)
    np.testing.assert_array_equal(result.values[0], result.values[1])
