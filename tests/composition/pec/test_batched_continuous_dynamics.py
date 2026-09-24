"""Explicit equation semantics, stochastic closure, generated RK4 and adjoints."""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import sympy as sp
import torch

from psyneulink.core.batched.continuous_ir import (
    ContinuousDynamics, DiffusionTerm, Sigmoid, analyze_continuous_dynamics, extract_deterministic_subsystem,
)
from psyneulink.core.batched.likelihood_planning import LikelihoodPlanningError
from psyneulink.core.batched.numerical.dynamics import compile_continuous_phase
from psyneulink.core.batched.numerical.dynamics_codegen import evaluate_equations, generate_dynamics_source
from psyneulink.core.batched.numerical.native import native_kernels_available


pytestmark = [pytest.mark.composition]
native = pytest.mark.skipif(not native_kernels_available(), reason="Requires Ninja and a C++ compiler.")


def _decay():
    x, k, t = sp.symbols("x k t", real=True)
    with sp.evaluate(False):
        return ContinuousDynamics(states=(x,), inputs=(), parameters=(k,), drift=(-k * x,),
                                  readouts=(("value", x + t),))


def _nonlinear():
    x, y, u, a, b, t = sp.symbols("x y u a b t", real=True)
    with sp.evaluate(False):
        return ContinuousDynamics(
            states=(x, y), inputs=(u,), parameters=(a, b),
            drift=(-a * x + Sigmoid(y * b) + u / (1 + t * t), sp.tanh(x) - b * y + sp.exp(-t)),
            readouts=(("signal", Sigmoid(x - y) + u * t), ("other", x * y + a / (1 + b))),
        )


def _arguments(requires_grad=False):
    values = dict(
        state=torch.tensor([[.2, -.1], [.3, .15]], dtype=torch.float64),
        inputs=torch.tensor([[.12], [-.2]], dtype=torch.float64),
        parameters=torch.tensor([[1.1, .7], [.8, .9]], dtype=torch.float64),
        duration=torch.tensor([.23, .31], dtype=torch.float64),
        start_time=torch.tensor([.13, .22], dtype=torch.float64),
    )
    for value in values.values():
        value.requires_grad_(requires_grad)
    return {**values, "steps": torch.tensor([3, 5])}


def _reference(dynamics, *, state, inputs, parameters, duration, start_time, steps):
    paths, finals = [], []
    maximum = int(steps.max()) if steps.numel() else 0
    for lane in range(state.shape[0]):
        x, u, p = state[lane], inputs[lane], parameters[lane]
        count = int(steps[lane])
        h = duration[lane] / count if count else duration[lane] * 0

        def rhs(x, t):
            return evaluate_equations(dynamics.drift, dynamics, x, u, p, t)

        def rk4(x, t, h):
            k1 = rhs(x, t)
            k2 = rhs(x + h / 2 * k1, t + h / 2)
            k3 = rhs(x + h / 2 * k2, t + h / 2)
            k4 = rhs(x + h * k3, t + h)
            return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        row = []
        for k in range(count):
            t = start_time[lane] + k * h
            mid = rk4(x, t, h / 2)
            row.append(evaluate_equations(tuple(e for _, e in dynamics.readouts), dynamics, mid, u, p, t + h / 2))
            x = rk4(mid, t + h / 2, h / 2)
        row += [state.new_zeros(len(dynamics.readouts)) for _ in range(maximum - count)]
        paths.append(torch.stack(row) if row else state.new_zeros((0, len(dynamics.readouts))))
        finals.append(x)
    return torch.stack(paths), torch.stack(finals)


def test_immutable_schema_bindings_and_source_safety():
    dynamics = _nonlinear()
    assert isinstance(dynamics.states, tuple)
    assert analyze_continuous_dynamics(dynamics).deterministic_states == dynamics.states
    with pytest.raises(ValueError, match="Unbound"):
        replace(dynamics, drift=(sp.Symbol("missing", real=True), sp.S.Zero))
    with pytest.raises(ValueError, match="physical seconds"):
        replace(dynamics, clock_unit="scheduler_pass")
    with pytest.raises(ValueError):
        replace(dynamics, drift=(sp.Function("python_callback")(dynamics.states[0]), sp.S.Zero))
    with pytest.raises(ValueError):
        replace(dynamics, drift=(sp.nan, sp.S.Zero))
    with pytest.raises(ValueError):
        replace(dynamics, states=(sp.Symbol("complex", real=False), dynamics.states[1]))
    with pytest.raises(ValueError):
        DiffusionTerm([], "driver", sp.Float(.1))
    with pytest.raises(TypeError):
        DiffusionTerm(dynamics.states[0], "driver", .1)
    # Names are bindings only, never source identifiers or interpolated code.
    unusual = 'x; throw "bad";'
    symbol = sp.Symbol(unusual, real=True)
    unusual_dynamics = ContinuousDynamics((symbol,), (), (), (-symbol,))
    assert unusual not in generate_dynamics_source(unusual_dynamics)
    assert generate_dynamics_source(dynamics) == generate_dynamics_source(dynamics)


def test_phase_plan_freezes_equations_and_integrator_template(monkeypatch):
    plan = compile_continuous_phase(_decay())
    source = plan.source
    original = Path.read_text

    def changed_header(path, *args, **kwargs):
        text = original(path, *args, **kwargs)
        return text + "\n// changed integrator revision\n" if path.name == "rk4_cpu.h" else text

    monkeypatch.setattr(Path, "read_text", changed_header)
    changed = compile_continuous_phase(_decay())
    assert plan.source == source
    assert changed.source != source
    assert changed.header_digest != plan.header_digest
    assert '#include "rk4_cpu.h"' not in source


def test_symbolic_admission_and_retained_structural_dependencies():
    x, y, k = sp.symbols("x y k", real=True)
    for bad in (sp.log(x), sp.sqrt(x), x ** k, sp.oo, sp.Integer(10) ** 1000):
        with pytest.raises(ValueError):
            ContinuousDynamics((x,), (), (k,), (bad,))
    with pytest.raises(ValueError, match="distinct real"):
        ContinuousDynamics((x,), (x,), (), (-x,))
    with sp.evaluate(False):
        zero = 0 * y
        canceled = y / y
    for expression in (zero, canceled):
        d = ContinuousDynamics((x, y), (), (), (expression, -y),
                              diffusion=(DiffusionTerm(y, "noise", sp.Float(.1)),))
        assert analyze_continuous_dynamics(d).stochastic_states == (x, y)
        with pytest.raises(LikelihoodPlanningError):
            compile_continuous_phase(d)
        assert d.drift[0] == expression  # Analysis did not simplify the declaration.
    # Distinct same-named bindings remain distinct; C++ uses array indices.
    a, b = sp.Dummy("same", real=True), sp.Dummy("same", real=True)
    d = ContinuousDynamics((a,), (), (b,), (-b * a,))
    source = generate_dynamics_source(d)
    assert "same" not in source
    assert source == generate_dynamics_source(d)


@native
@pytest.mark.parametrize("placement", ["drift", "readout", "intermediate_stage"])
def test_canceled_division_retains_domain_checks(placement):
    x = sp.Symbol("x", real=True)
    with sp.evaluate(False):
        quotient = x / x
        rhs = quotient if placement == "drift" else (-2 + quotient if placement == "intermediate_stage" else sp.S.Zero)
    d = ContinuousDynamics((x,), (), (), (rhs,), (("ratio", quotient),))
    plan = compile_continuous_phase(d)
    assert "_domain" in plan.source
    args = dict(state=torch.tensor([[1.]], dtype=torch.float64, requires_grad=True),
                inputs=torch.empty(1, 0, dtype=torch.float64), parameters=torch.empty(1, 0, dtype=torch.float64),
                duration=torch.tensor([.1], dtype=torch.float64), start_time=torch.zeros(1, dtype=torch.float64),
                steps=torch.ones(1, dtype=torch.int64))
    assert torch.autograd.gradcheck(lambda initial: plan.integrate(**{**args, "state": initial}).final_state,
                                   (args["state"],), atol=1e-7, rtol=1e-5)
    if placement == "intermediate_stage":
        args["duration"] = torch.tensor([2.], dtype=torch.float64)
    else:
        args["state"] = torch.zeros_like(args["state"], requires_grad=True)
    for enabled in (False, True):
        with torch.set_grad_enabled(enabled), pytest.raises(FloatingPointError, match="Nonfinite"):
            plan.integrate(**args)
    bad = evaluate_equations((quotient,), d, torch.zeros(1, dtype=torch.float64),
                             torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64), torch.tensor(0.))
    assert not bool(torch.isfinite(bad).all())


@native
def test_symbolic_sigmoid_saturation_and_safe_names():
    x = sp.Symbol('x; throw "bad";', real=True)
    d = ContinuousDynamics((x,), (), (), (sp.S.Zero,), (("logistic", Sigmoid(x)),))
    plan = compile_continuous_phase(d)
    initial = torch.tensor([[-1000.], [-20.], [0.], [20.], [1000.]], dtype=torch.float64, requires_grad=True)
    result = plan.integrate(state=initial, inputs=initial[:, :0], parameters=initial[:, :0],
                            duration=initial.new_ones(5), start_time=initial.new_zeros(5), steps=torch.ones(5, dtype=torch.int64))
    expected = initial.sigmoid()
    torch.testing.assert_close(result.readouts[:, 0], expected)
    torch.testing.assert_close(torch.autograd.grad(result.readouts.sum(), initial)[0], expected * (1 - expected))
    reference = evaluate_equations((Sigmoid(x),), d, initial, initial[:, :0], initial[:, :0], initial.new_zeros(5))
    torch.testing.assert_close(reference, expected)
    constant_readout = replace(d, readouts=(("constant", Sigmoid(sp.S.Zero)),))
    reference = evaluate_equations((constant_readout.readouts[0][1],), constant_readout,
                                   initial, initial[:, :0], initial[:, :0], initial.new_zeros(5))
    torch.testing.assert_close(reference, initial.new_full((5, 1), .5))


@pytest.mark.parametrize("feedback, expected", [(False, ("r0", "r1")), (True, ("h0", "h1", "r0", "r1"))])
def test_multidimensional_noise_and_feedback_closure(feedback, expected):
    h0, h1, r0, r1 = sp.symbols("h0 h1 r0 r1", real=True)
    dynamics = ContinuousDynamics(
        states=(h0, h1, r0, r1), inputs=(), parameters=(),
        drift=(-h0 + h1 + (r0 if feedback else sp.S.Zero), -h1 + Sigmoid(h0), h0 - r0 - r1, h1 - r1),
        readouts=(("control", h0), ("response", r0 - r1)),
        # One driver is enough to make multiple state coordinates stochastic.
        diffusion=(DiffusionTerm(r1, "shared_noise", sp.Float(.1)),),
    )
    report = analyze_continuous_dynamics(dynamics)
    assert tuple(map(str, report.stochastic_states)) == expected
    assert report.noise_drivers == ("shared_noise",)
    assert report.stochastic_readouts == (("control", "response") if feedback else ("response",))
    with pytest.raises(LikelihoodPlanningError, match="cannot ignore"):
        compile_continuous_phase(dynamics)
    with pytest.raises(ValueError, match="cannot discard"):
        generate_dynamics_source(dynamics)
    reduced = extract_deterministic_subsystem(dynamics)
    if feedback:
        assert reduced is None
    else:
        assert reduced.state_indices == (0, 1)
        assert reduced.readout_indices == (0,)
        assert reduced.dynamics.states == (h0, h1)
        assert not reduced.dynamics.diffusion
        assert compile_continuous_phase(reduced.dynamics).explain()["states"] == ("h0", "h1")


def test_deterministic_slice_preserves_bindings_and_drops_unused_latent_inputs():
    random, known, latent, drive, unused, rate = sp.symbols("random known latent drive unused rate", real=True)
    d = ContinuousDynamics(
        states=(random, known), inputs=(latent, drive), parameters=(unused, rate),
        drift=(latent + known, -rate * known + drive),
        readouts=(("observed", known), ("random_readout", random)),
        latent_inputs=(latent,), latent_initial_states=(random,),
    )
    reduced = extract_deterministic_subsystem(d)
    assert (reduced.state_indices, reduced.input_indices, reduced.parameter_indices, reduced.readout_indices) == ((1,), (1,), (1,), (0,))
    assert not reduced.dynamics.latent_inputs
    x = torch.tensor([.9, .3], dtype=torch.float64)
    u = torch.tensor([.7, .1], dtype=torch.float64)
    p = torch.tensor([4., 2.], dtype=torch.float64)
    full = evaluate_equations(d.drift, d, x, u, p, torch.tensor(0.))
    subset = evaluate_equations(reduced.dynamics.drift, reduced.dynamics, x[[1]], u[[1]], p[[1]], torch.tensor(0.))
    torch.testing.assert_close(subset, full[[1]])


def test_latent_initial_state_and_inputs_are_not_treated_as_deterministic():
    base = _nonlinear()
    for d in (replace(base, latent_initial_states=base.states[:1]), replace(base, latent_inputs=base.inputs)):
        assert analyze_continuous_dynamics(d).stochastic_states == base.states
        with pytest.raises(LikelihoodPlanningError):
            compile_continuous_phase(d)
    decay = _decay()
    zero_noise = replace(decay, diffusion=(DiffusionTerm(decay.states[0], "noise", sp.S.Zero),))
    assert not analyze_continuous_dynamics(zero_noise).stochastic_states
    # No algebraic simplifier may erase an unknown coefficient's dependence.
    with sp.evaluate(False):
        unknown = replace(decay, diffusion=(DiffusionTerm(decay.states[0], "noise", 0 * decay.parameters[0]),))
    assert analyze_continuous_dynamics(unknown).stochastic_states == decay.states


@native
def test_generated_decay_against_closed_form_and_clock_readout():
    plan = compile_continuous_phase(_decay())
    args = dict(state=torch.tensor([[.8]], dtype=torch.float64), inputs=torch.empty(1, 0, dtype=torch.float64),
                parameters=torch.tensor([[1.2]], dtype=torch.float64), duration=torch.tensor([.3], dtype=torch.float64),
                start_time=torch.tensor([.2], dtype=torch.float64), steps=torch.tensor([30]))
    result = plan.integrate(**args)
    torch.testing.assert_close(result.final_state, .8 * torch.exp(torch.tensor([[-.36]], dtype=torch.float64)), rtol=1e-10, atol=1e-12)
    midpoints = (torch.arange(30, dtype=torch.float64) + .5) * .01
    expected = .8 * torch.exp(-1.2 * midpoints) + .2 + midpoints
    torch.testing.assert_close(result.readouts[0, :, 0], expected, rtol=1e-10, atol=1e-12)


@native
def test_four_state_system_without_parameters_inputs_or_readouts():
    x, y, z, w = sp.symbols("x y z w", real=True)
    dynamics = ContinuousDynamics((x, y, z, w), (), (), (-x, x - y, y - z, z - w))
    plan = compile_continuous_phase(dynamics)
    durations = torch.tensor([.1, .4], dtype=torch.float64, requires_grad=True)
    args = dict(state=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]], dtype=torch.float64),
                inputs=torch.empty(2, 0, dtype=torch.float64), parameters=torch.empty(2, 0, dtype=torch.float64),
                duration=durations, start_time=torch.zeros(2, dtype=torch.float64), steps=torch.tensor([20, 40]))
    result = plan.integrate(**args)
    expected = torch.exp(-durations[:, None]) * torch.stack((torch.ones_like(durations), durations,
                                                            durations**2 / 2, durations**3 / 6), dim=-1)
    assert result.readouts.shape == (2, 40, 0)
    torch.testing.assert_close(result.final_state, expected, rtol=1e-8, atol=1e-11)
    actual_grad, = torch.autograd.grad(result.final_state.sum(), durations)
    expected_grad, = torch.autograd.grad(expected.sum(), durations)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-7, atol=1e-10)


@native
def test_generated_nonlinear_phase_and_all_gradients_match_torch():
    d = _nonlinear()
    args = _arguments(requires_grad=True)
    generated = compile_continuous_phase(d).integrate(**args)
    reference = _reference(d, **args)
    actual = (generated.readouts, generated.final_state)
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=1e-12)
    wr = torch.arange(actual[0].numel(), dtype=torch.float64).reshape_as(actual[0]) / 17
    wf = torch.tensor([[.3, -.2], [-.7, .8]], dtype=torch.float64)
    inputs = tuple(v for n, v in args.items() if n != "steps")
    ga = torch.autograd.grad((actual[0] * wr).sum() + (actual[1] * wf).sum(), inputs)
    gb = torch.autograd.grad((reference[0] * wr).sum() + (reference[1] * wf).sum(), inputs)
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-11)


@native
def test_generated_adjoint_finite_differences_include_duration_and_time_origin():
    args = _arguments(requires_grad=True)
    plan = compile_continuous_phase(_nonlinear())
    names = tuple(n for n in args if n != "steps")

    def evaluate(*values):
        result = plan.integrate(**dict(zip(names, values, strict=True)), steps=args["steps"])
        return result.readouts, result.final_state

    assert torch.autograd.gradcheck(evaluate, tuple(args[n] for n in names), atol=2e-7, rtol=2e-5)


@native
def test_zero_phase_and_padded_readouts_have_correct_gradients():
    plan = compile_continuous_phase(_nonlinear())
    args = _arguments(requires_grad=True)
    args["duration"] = torch.tensor([0., .31], dtype=torch.float64, requires_grad=True)
    args["steps"] = torch.tensor([0, 5])
    result = plan.integrate(**args)
    torch.testing.assert_close(result.final_state[0], args["state"][0])
    assert bool((result.readouts[0] == 0).all())
    gradients = torch.autograd.grad(result.readouts.sum() + result.final_state.sum(), tuple(v for n, v in args.items() if n != "steps"))
    torch.testing.assert_close(gradients[0][0], torch.ones(2, dtype=torch.float64))
    assert all(bool((g[0] == 0).all()) for g in gradients[1:])
    empty = {n: v[:0] for n, v in _arguments().items()}
    empty_result = plan.integrate(**empty)
    assert empty_result.readouts.shape == (0, 0, 2)
    assert empty_result.final_state.shape == (0, 2)


def test_runtime_shape_clock_and_dtype_checks_before_build(monkeypatch):
    plan = compile_continuous_phase(_nonlinear())
    monkeypatch.setattr(type(plan), "_module", lambda self: pytest.fail("invalid inputs reached compilation"))
    args = _arguments()
    for change in ({"duration": torch.tensor([-.1, .3], dtype=torch.float64)},
                   {"steps": torch.tensor([0, 5])}, {"steps": torch.tensor([3., 5.])},
                   {"state": args["state"].float()}, {"inputs": args["inputs"][:, :0]}):
        with pytest.raises(ValueError):
            plan.integrate(**{**args, **change})


@native
def test_multiple_phases_chain_with_parameter_dependent_timing():
    plan = compile_continuous_phase(_nonlinear())
    args = _arguments()
    theta = torch.tensor([.23, .31], dtype=torch.float64, requires_grad=True)

    def evaluate(durations):
        first = plan.integrate(**{**args, "duration": durations})
        second = plan.integrate(**{**args, "state": first.final_state, "duration": 1.3 * durations,
                                  "start_time": args["start_time"] + durations})
        return second.final_state, second.readouts

    assert torch.autograd.gradcheck(evaluate, (theta,), atol=2e-7, rtol=2e-5)


@native
def test_generated_dynamics_to_shared_pde_end_to_end_gradient():
    from psyneulink.core.batched.numerical import FirstPassageMesh, FirstPassageProblem, compile_first_passage

    phase = compile_continuous_phase(_nonlinear())
    problem = FirstPassageProblem(
        process="continuous_time", stochastic_dimensions=1, drift_dependence="time_only",
        diffusion="constant_scalar", boundary="symmetric_linear", initial_state="point_center",
        coefficient_source="conditioned_deterministic", observation="choice_rt_interval",
    )
    pde = compile_first_passage(problem, noise=.3, mesh=FirstPassageMesh(time_step=.005, spatial_points=33))
    args = {n: v[:1] for n, v in _arguments().items()}
    args.update(duration=torch.tensor([.2], dtype=torch.float64), start_time=torch.zeros(1, dtype=torch.float64),
                steps=torch.tensor([40]))

    def evaluate(parameters):
        path = phase.integrate(**{**args, "parameters": parameters})
        return pde.solve_observation_batch(
            drift=path.readouts[..., 0], threshold=torch.tensor([.2], dtype=torch.float64),
            collapse_rate=torch.tensor([-.03], dtype=torch.float64),
            interval_low=torch.tensor([.153], dtype=torch.float64), interval_high=torch.tensor([.161], dtype=torch.float64),
            choice=torch.tensor([1.], dtype=torch.float64),
        ).probability.log()

    parameters = torch.tensor([[1.1, .7]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(evaluate, (parameters,), atol=2e-6, rtol=3e-5)


@native
def test_no_tape_for_value_only_and_explicit_higher_derivative_rejection(monkeypatch):
    plan = compile_continuous_phase(_nonlinear())
    module = plan._module()
    original = module.forward
    tapes = []

    def record(*args):
        result = original(*args)
        tapes.append((args[-1], result[-1].numel()))
        return result

    monkeypatch.setattr(module, "forward", record)
    plan.integrate(**_arguments())
    assert tapes == [(False, 0)]
    args = _arguments(requires_grad=True)
    result = plan.integrate(**args)
    assert tapes[-1] == (True, 6 * 2 * 2)
    with pytest.raises(RuntimeError, match="first-order"):
        torch.autograd.grad(result.final_state.sum(), args["parameters"], create_graph=True)


def csi_equations():
    """Research model equation fixture, never a compiler recognizer."""
    x, y, gain = sp.symbols("x y gain", real=True)
    task0, task1, s0, s1, s2, s3, response = sp.symbols("task0 task1 stimulus0 stimulus1 stimulus2 stimulus3 response", real=True)
    with sp.evaluate(False):
        control0, control1 = Sigmoid(gain * x), Sigmoid(gain * y)
        a, b = Sigmoid(s0 - s1 + 4 * control0 - 4), Sigmoid(s1 - s0 + 4 * control0 - 4)
        c, d = Sigmoid(s2 - s3 + 4 * control1 - 4), Sigmoid(s3 - s2 + 4 * control1 - 4)
        contrast = a - b + c - d
        drift = (Sigmoid(contrast) - Sigmoid(-contrast)) * response
        return ContinuousDynamics(
            states=(x, y), inputs=(task0, task1, s0, s1, s2, s3, response), parameters=(gain,),
            drift=(-12 * x + task0 - 3 * control1, -12 * y + task1 - 3 * control0), readouts=(("drift", drift),),
        )


@native
def test_generated_csi_equations_match_existing_handwritten_drift_and_gradients():
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/csi/csi_fit"))
    from direct_likelihood.native import native_lca_drift_path

    plan = compile_continuous_phase(csi_equations())
    x = torch.tensor([[.03, -.02], [-.1, .02]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[1., 0., 1., 0., 0., 1., 1.], [0., 1., .2, .8, .9, .1, -1.]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[12.], [17.]], dtype=torch.float64, requires_grad=True)
    count, dt = 45, .001
    result = plan.integrate(state=x, inputs=u, parameters=p, duration=torch.full((2,), count * dt, dtype=torch.float64),
                            start_time=torch.zeros(2, dtype=torch.float64), steps=torch.full((2,), count, dtype=torch.int64))
    reference = native_lca_drift_path(x, u[:, :2], p[:, 0], u[:, 2:6], u[:, 6], steps=count, step_size=dt, leak=12., competition=3.)
    actual = result.readouts[..., 0], result.final_state
    for a, b in zip(actual, reference, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-12, atol=2e-13)
    ga = torch.autograd.grad(actual[0].square().sum() + actual[1].sum(), (x, u, p))
    gb = torch.autograd.grad(reference[0].square().sum() + reference[1].sum(), (x, u, p))
    for a, b in zip(ga, gb, strict=True):
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-11)
