"""Model-independent numerical backend admission, adjoints, and probability checks."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.numerical import (
    FirstPassageMesh, FirstPassageProblem, compile_first_passage,
)
from psyneulink.core.batched.numerical.native import native_kernels_available
from psyneulink.core.batched.wiener import wiener_log_density


pytestmark = [pytest.mark.composition]
native = pytest.mark.skipif(not native_kernels_available(), reason="Requires Ninja and a C++ compiler.")
PROBLEM = FirstPassageProblem(
    process="continuous_time", stochastic_dimensions=1, drift_dependence="time_only",
    diffusion="constant_scalar", boundary="symmetric_linear", initial_state="point_center",
    coefficient_source="conditioned_deterministic", observation="choice_rt_interval",
)
MESH = FirstPassageMesh(time_step=.005, spatial_points=33)
GRADIENTS = ("drift", "threshold", "collapse_rate", "interval_low", "interval_high")


def _arguments(dtype=torch.float64, requires_grad=False):
    t = (torch.arange(43, dtype=dtype) + .5) * MESH.time_step
    values = dict(
        drift=torch.stack((.06 + .02 * torch.sin(5 * t), -.04 + .03 * torch.cos(7 * t))),
        threshold=torch.tensor([.065, .085], dtype=dtype),
        collapse_rate=torch.tensor([-.025, .015], dtype=dtype),
        interval_low=torch.tensor([.0731, .1247], dtype=dtype),
        interval_high=torch.tensor([.0783, .1312], dtype=dtype),
        choice=torch.tensor([0., 1.], dtype=dtype),
    )
    for name in GRADIENTS:
        values[name].requires_grad_(requires_grad)
    return values


def test_backend_contract_is_not_a_composition_recognizer():
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    description = plan.explain()
    assert description["problem"]["stochastic_dimensions"] == 1
    assert description["gradient_inputs"] == GRADIENTS
    assert description["gradient"] == "first_order_discrete_adjoint"
    assert "no graph-reduction proof" in description["assumptions"][0]


@pytest.mark.parametrize("changes", [
    {"stochastic_dimensions": 2}, {"stochastic_dimensions": True},
    {"coefficient_source": "latent_stochastic"}, {"process": "source"},
    {"diffusion": "state_dependent"}, {"drift_dependence": "state_and_time"},
    {"boundary": "nonlinear"}, {"initial_state": "arbitrary_density"},
    {"observation": "rt_density"},
])
def test_unsupported_mathematics_cannot_silently_select_scalar_solver(changes):
    with pytest.raises(LikelihoodPlanningError) as error:
        compile_first_passage(replace(PROBLEM, **changes), noise=.18)
    assert error.value.code == "numerical.unsupported_problem"


@pytest.mark.parametrize("name", ["noise", "initial_state", "time_step", "upstream_unknown"])
def test_unsupported_gradient_is_not_reported_as_zero(name):
    with pytest.raises(LikelihoodPlanningError) as error:
        compile_first_passage(PROBLEM, noise=.18, required_gradients=(name,))
    assert error.value.code == "numerical.gradient_unsupported"


@pytest.mark.parametrize("backend", ["cuda", "triton", "auto"], ids=["unsupported_device", "sampling_backend", "implicit_backend"])
def test_no_silent_device_fallback(backend):
    with pytest.raises(LikelihoodPlanningError, match="no device fallback"):
        compile_first_passage(PROBLEM, noise=.18, backend=backend)


@pytest.mark.parametrize("changes", [
    {"time_step": float("nan")}, {"time_step": 0.}, {"time_step": float("inf")},
    {"spatial_points": 8}, {"spatial_points": 9.5}, {"spatial_points": True},
    {"boundary_floor": -1.}, {"backward_euler_steps": -1}, {"backward_euler_steps": 1.5},
])
def test_invalid_mesh(changes):
    with pytest.raises(ValueError):
        FirstPassageMesh(**changes)


@pytest.mark.parametrize("noise", [0., -1., float("nan"), float("inf"), True, torch.tensor(.1, requires_grad=True)])
def test_noise_is_fixed_positive_and_not_silently_detached(noise):
    with pytest.raises(ValueError, match="fixed finite positive"):
        compile_first_passage(PROBLEM, noise=noise)


@native
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_matches_torch_forward_and_all_adjoint_inputs(dtype):
    args = _arguments(dtype, requires_grad=True)
    results, gradients = [], []
    for backend in ("cpp_cpu", "torch_cpu"):
        plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH, backend=backend)
        result = plan.solve_observation_batch(**args)
        results.append(result)
        gradients.append(torch.autograd.grad(result.probability.log().sum(), [args[name] for name in GRADIENTS]))
    for name in results[0].__dataclass_fields__:
        torch.testing.assert_close(getattr(results[0], name), getattr(results[1], name), rtol=3e-4, atol=2e-6)
    for actual, expected in zip(*gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-4 if dtype == torch.float32 else 1e-9,
                                   atol=3e-4 if dtype == torch.float32 else 1e-9)


@native
def test_native_all_five_adjoint_inputs_against_finite_differences():
    args = _arguments(requires_grad=True)
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH, required_gradients=GRADIENTS)

    def evaluate(*values):
        return plan.solve_observation_batch(**dict(zip(GRADIENTS, values, strict=True)), choice=args["choice"]).probability.log()

    assert torch.autograd.gradcheck(evaluate, tuple(args[name] for name in GRADIENTS), eps=1e-6, atol=2e-5, rtol=2e-4)


@native
def test_native_does_not_silently_return_partial_higher_derivatives():
    args = _arguments(requires_grad=True)
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    result = plan.solve_observation_batch(**args)
    with pytest.raises(RuntimeError, match="first-order derivatives only"):
        torch.autograd.grad(result.probability.log().sum(), args["drift"], create_graph=True)


@native
def test_non_csi_nonlinear_coefficients_receive_chain_rule_gradients():
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    args = _arguments()
    t = (torch.arange(43, dtype=torch.float64) + .5) * MESH.time_step

    def evaluate(parameters):
        # Arbitrary nonlinear deterministic upstream readout: no LCA/model names.
        trajectory = parameters[0] * torch.tanh(torch.sin(parameters[1] * t) + parameters[2])
        return plan.solve_observation_batch(**{**args, "drift": torch.stack((trajectory, -trajectory))}).probability.log()

    parameters = torch.tensor([.08, 7., .3], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate, (parameters,), eps=1e-6, atol=2e-5, rtol=2e-4)


@native
def test_constant_coefficients_match_integrated_analytic_density():
    # Compare interval masses, not a density sampled at the interval midpoint.
    mesh = FirstPassageMesh(time_step=.0005, spatial_points=129)
    plan = compile_first_passage(PROBLEM, noise=.1, mesh=mesh)
    args = dict(drift=torch.full((2, 820), .03, dtype=torch.float64),
                threshold=torch.full((2,), .12, dtype=torch.float64),
                collapse_rate=torch.zeros(2, dtype=torch.float64),
                interval_low=torch.full((2,), .3973, dtype=torch.float64),
                interval_high=torch.full((2,), .4027, dtype=torch.float64),
                choice=torch.tensor([0., 1.], dtype=torch.float64))
    nodes, weights = np.polynomial.legendre.leggauss(32)
    t = torch.tensor(.4 + .0027 * nodes, dtype=torch.float64)
    drift, bound, noise, start = (torch.tensor(value, dtype=torch.float64) for value in (.03, .12, .1, 0.))
    density = wiener_log_density(t[None, :], drift, bound, noise, start, args["choice"][:, None]).exp()
    expected = (density * torch.tensor(weights)[None, :]).sum(-1) * .0027
    with torch.no_grad():
        result = plan.solve_observation_batch(**args)
    torch.testing.assert_close(result.probability, expected, rtol=.001, atol=1e-10)
    assert float(result.mass_error.max()) < 1e-10


@pytest.mark.parametrize("change, message", [
    ({"drift": torch.zeros(2, 2, dtype=torch.float64)}, "cover"),
    ({"choice": torch.tensor([0., .5], dtype=torch.float64)}, "only 0 or 1"),
    ({"threshold": torch.tensor([float("nan"), .1], dtype=torch.float64)}, "finite"),
    ({"interval_high": torch.zeros(2, dtype=torch.float64)}, "positive width"),
    ({"collapse_rate": torch.zeros(2, dtype=torch.float32)}, "share dtype"),
    ({"threshold": torch.zeros(2, 1, dtype=torch.float64)}, "shape"),
])
def test_bad_inputs_fail_before_native_execution(change, message):
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    with pytest.raises(ValueError, match=message):
        plan.solve_observation_batch(**{**_arguments(), **change})


@native
def test_empty_batch_noncontiguous_coefficients_and_invalid_boundaries():
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    args = _arguments(requires_grad=True)
    args["drift"] = args["drift"].T.contiguous().T
    assert not args["drift"].is_contiguous()
    args["collapse_rate"] = torch.tensor([-2., .015], dtype=torch.float64, requires_grad=True)
    result = plan.solve_observation_batch(**args)
    assert result.invalid_boundary.tolist() == [True, False]
    assert result.probability[0] == 0
    gradients = torch.autograd.grad(result.probability.sum(), tuple(args[name] for name in GRADIENTS))
    assert all(bool((value[0] == 0).all()) for value in gradients)
    empty = {name: value[:0] for name, value in _arguments().items()}
    assert plan.solve_observation_batch(**empty).probability.shape == (0,)


@native
def test_score_only_does_not_store_a_density_tape(monkeypatch):
    from psyneulink.core.batched.numerical import native as native_module

    original = native_module.native_ddm_forward
    histories = []

    def record(*args, **kwargs):
        output = original(*args, **kwargs)
        histories.append((kwargs["store_history"], output[-1].numel()))
        return output

    monkeypatch.setattr(native_module, "native_ddm_forward", record)
    plan = compile_first_passage(PROBLEM, noise=.18, mesh=MESH)
    with torch.no_grad():
        plan.solve_observation_batch(**_arguments())
    plan.solve_observation_batch(**_arguments())
    assert histories == [(False, 0), (False, 0)]
    args = _arguments(requires_grad=True)
    result = plan.solve_observation_batch(**args)
    torch.autograd.grad(result.probability.sum(), args["drift"])
    assert histories[-1] == (True, 44 * 2 * 31)


@native
@pytest.mark.parametrize("name", ["history", "invalid", "gradient_probability"])
def test_cpp_adjoint_rejects_malformed_buffers(name):
    from psyneulink.core.batched.numerical.native import native_ddm_forward, native_ddm_backward

    args = _arguments()
    config = dict(time_step=MESH.time_step, spatial_points=MESH.spatial_points, noise=.18, rannacher_steps=2)
    values = native_ddm_forward(**args, **config, boundary_floor=MESH.boundary_floor, store_history=True)
    backward = dict(history=values[7], invalid=values[6], gradient_probability=torch.ones(2, dtype=torch.float64))
    backward[name] = backward[name][:0]
    with pytest.raises(RuntimeError, match="shape"):
        native_ddm_backward(**args, **config, **backward)
