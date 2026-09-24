"""Parallel CPU and fused GPU parity, including carried trial history."""

from dataclasses import replace
import os
from pathlib import Path
import sys

import pytest
import torch

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_likelihood.continuous_flux import native_flux_block  # noqa: E402
from dawa_likelihood.continuous_likelihood import continuous_sequence_likelihood  # noqa: E402
from dawa_likelihood.continuous_solver import ContinuousConfig, ContinuousResponseSolver  # noqa: E402

pytestmark = [pytest.mark.composition]
GPU = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    yield
    torch.set_num_threads(old)


def compare_flux(run):
    generator = torch.Generator().manual_seed(381)
    for offset in (0., .2):  # Exercise reuse of graph buffers with different inputs.
        mass = (torch.rand(11, 11, generator=generator, dtype=torch.float64) + offset).requires_grad_()
        rates = torch.rand(5, 4, 11, 11, generator=generator, dtype=torch.float64).requires_grad_()
        w = torch.randn(mass.shape, generator=generator, dtype=mass.dtype)
        f = torch.randn(4, 3, generator=generator, dtype=mass.dtype)
        expected = native_flux_block(mass, rates, 3, .03)
        actual = run(mass, rates)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.cpu(), b, atol=2.e-13, rtol=2.e-13)
        ga = torch.autograd.grad((actual[0] * w.to(actual[0].device)).sum()
                                + (actual[1] * f.to(actual[1].device)).sum(), (mass, rates))
        gb = torch.autograd.grad((expected[0] * w).sum() + (expected[1] * f).sum(), (mass, rates))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b, atol=3.e-12, rtol=3.e-12)
        torch.testing.assert_close(actual[0].sum().cpu() + actual[1].sum().cpu(), mass.sum(), atol=1.e-12, rtol=1.e-12)


@pytest.mark.parametrize("threads", [2, 4])
def test_parallel_cpu_flux_and_adjoint(threads):
    compare_flux(lambda m, r: native_flux_block(m, r, 3, .03, threads))


@GPU
@pytest.mark.parametrize("graphs", [False, True])
def test_gpu_flux_and_adjoint(graphs):
    from dawa_likelihood.continuous_flux_gpu import gpu_flux_block
    compare_flux(lambda m, r: gpu_flux_block(m.cuda(), r.cuda(), 3, .03, graphs=graphs))


@pytest.mark.parametrize("backend", ["cpu_parallel", pytest.param("triton", marks=GPU)])
def test_parallel_coefficients_and_all_derivatives(backend):
    generator = torch.Generator().manual_seed(33)
    cfg = ContinuousConfig(points=11, flux_backend="native")
    values = [torch.randn(8, 2, generator=generator, dtype=torch.float64) * 3.,
              torch.tensor([.001, .1, 2., 5., 10., 20., 80., 150.], dtype=torch.float64),
              torch.tensor(-.4, dtype=torch.float64), torch.linspace(.04, .7, 8, dtype=torch.float64),
              torch.tensor([0., -10., 5., .1, 1., -1., .2, .7], dtype=torch.float64)]
    values = [v.requires_grad_() for v in values]
    expected = ContinuousResponseSolver(cfg)._rates(*values)
    device = "cuda" if backend == "triton" else "cpu"
    candidate = replace(cfg, flux_backend="triton" if backend == "triton" else "native", cpu_threads=4)
    actual = ContinuousResponseSolver(candidate)._rates(*(v.to(device) for v in values))
    torch.testing.assert_close(actual.cpu(), expected, atol=2.e-10, rtol=2.e-12)
    weights = torch.randn(expected.shape, generator=generator, dtype=expected.dtype)
    ga = torch.autograd.grad((actual * weights.to(device)).sum(), values)
    gb = torch.autograd.grad((expected * weights).sum(), values)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=2.e-9, rtol=2.e-11)


@pytest.mark.parametrize("backend", ["cpu_parallel", pytest.param("triton", marks=GPU)])
@pytest.mark.parametrize("retain", [False, True])
def test_sequential_values_and_all_gradients_preserve_history(backend, retain):
    cfg = ContinuousConfig(points=17, time_step=.002, checkpoint_steps=16, recompute_rates=not retain,
                           ode_backend="generated", flux_backend="native")
    tasks = [[1, 0], [0, 1], [1, 0]]
    stimuli = [[0, 1, 0, 1], [1, 0, 0, 1], [1, 0, 1, 0]]
    p = torch.tensor([[.18, .193, -.43, 11., .75, 1.3, 5.2]] * 3, dtype=torch.float64, requires_grad=True)

    def run(vector, config):
        return continuous_sequence_likelihood(vector, tasks, stimuli, [1, 0, 1], [.3437, .4117, .4531],
                                              include=[False, True, True], config=config)

    expected = run(p, cfg)
    device = "cuda" if backend == "triton" else "cpu"
    config = replace(cfg, flux_backend="triton" if backend == "triton" else "native", cpu_threads=4)
    actual = run(p.to(device), config)
    torch.testing.assert_close(actual.log_likelihood.cpu(), expected.log_likelihood, atol=2.e-10, rtol=2.e-10)
    torch.testing.assert_close(actual.control_after_trial.cpu(), expected.control_after_trial, atol=1.e-12, rtol=1.e-12)
    ga = torch.autograd.grad(actual.log_likelihood, p)[0]
    gb = torch.autograd.grad(expected.log_likelihood, p)[0]
    torch.testing.assert_close(ga, gb, atol=2.e-8, rtol=2.e-8)
    assert abs(float(ga[0, 1])) > 1.e-5  # Excluded trial NDT affects later scores through C.
    assert abs(float(ga[0, 3])) > 1.e-5
    for distribution in actual.distributions[1:]:
        assert float(distribution.mass_error) < 1.e-12
        assert float(distribution.minimum_mass) >= 0.
    if backend == "triton" and retain:
        with torch.no_grad():
            plus, minus = p.clone().cuda(), p.clone().cuda()
            plus[0, 1] += 1.e-6
            minus[0, 1] -= 1.e-6
            finite = (run(plus, config).log_likelihood - run(minus, config).log_likelihood) / 2.e-6
        torch.testing.assert_close(ga[0, 1], finite.cpu(), atol=2.e-6, rtol=2.e-6)


@GPU
def test_gpu_backend_rejects_reduced_precision_and_higher_order_gradients():
    from dawa_likelihood.continuous_flux_gpu import gpu_flux_block
    with pytest.raises(ValueError, match="float64"):
        gpu_flux_block(torch.ones(8, 8, device="cuda"), torch.ones(2, 4, 8, 8, device="cuda"), 1, .01)
    mass = torch.ones(8, 8, device="cuda", dtype=torch.float64, requires_grad=True)
    rates = torch.ones(2, 4, 8, 8, device="cuda", dtype=torch.float64)
    result = gpu_flux_block(mass, rates, 1, .01, graphs=False)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(result[0].sum(), mass, create_graph=True)


def test_thread_count_validation():
    for count in (0, -1, 1.5):
        with pytest.raises(ValueError, match="positive integer"):
            ContinuousConfig(cpu_threads=count)
