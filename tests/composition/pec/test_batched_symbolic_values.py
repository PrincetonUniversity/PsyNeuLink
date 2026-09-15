"""Frozen PNL value algebra feeding the existing generated numerical backend."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import sympy as sp
import torch

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler as Compiler, LikelihoodEffectContract, LikelihoodPlanningError
from psyneulink.core.batched import specs, registry
from psyneulink.core.batched.continuous_ir import ContinuousDynamics
from psyneulink.core.batched.numerical import compile_continuous_phase, compile_first_passage, FirstPassageProblem, FirstPassageMesh
from psyneulink.core.batched.numerical.dynamics_codegen import evaluate_equations
from psyneulink.core.batched.numerical.native import native_kernels_available
from psyneulink.core.batched.symbolic_values import derive_symbolic_values


pytestmark = [pytest.mark.batched, pytest.mark.composition]
native = pytest.mark.skipif(not native_kernels_available(), reason="Requires Ninja and a C++ compiler.")


def _network(backend="triton_cpu"):
    origin = pnl.TransferMechanism(default_variable=[0., 0.], function=pnl.Linear(slope=1.3, intercept=-.1), name="value-input")
    middle = pnl.TransferMechanism(default_variable=[0., 0.], function=pnl.Linear(slope=-.7, offset=.2), name="value_middle")
    output = pnl.TransferMechanism(default_variable=[0., 0.], function=pnl.Logistic(gain=1.1, bias=.2, x_0=-.3, scale=.9, offset=.1), name="value.output")
    matrix = np.array([[.3, -.7], [.6, .2]])
    composition = pnl.Composition(pathways=[[origin, middle, output], [origin, matrix, output]])
    return composition, origin, middle, output, Compiler.compile(composition, backend=backend, outputs=[output.output_port])


def _phase(values):
    # A constant-state phase exposes the graph's instantaneous readout without
    # inventing any integration or trial-history semantics for the PNL graph.
    return ContinuousDynamics(values.input_symbols, (), values.parameter_symbols,
                              tuple(sp.S.Zero for _ in values.input_symbols),
                              tuple((f"value{i}", expr) for i, expr in enumerate(values.expressions)))


def _args(values, *, grad=False):
    x = torch.tensor([[-.3, .2], [.5, -.4], [.1, .7]], dtype=torch.float64, requires_grad=grad)
    p = torch.tensor([[p.default for p in values.parameters]], dtype=torch.float64).repeat(3, 1).requires_grad_(grad)
    return dict(state=x, inputs=x[:, :0], parameters=p, duration=x.new_full((3,), .01),
                start_time=x.new_zeros(3), steps=torch.ones(3, dtype=torch.int64))


@native
@pytest.mark.parametrize("backend", ["triton_cpu", "triton"])
def test_graph_values_match_pnl_and_batched_execution(backend):
    if backend == "triton" and not torch.cuda.is_available():
        pytest.skip("Requires CUDA.")
    if os.environ.get("PNL_SYMBOLIC_EXECUTION_CHILD") != backend:
        # Collection may import Triton before runtime selection. Isolate both
        # modes so neither depends on import order or changes other tests' mode.
        env = {**os.environ, "TRITON_INTERPRET": "1" if backend == "triton_cpu" else "0",
               "PNL_SYMBOLIC_EXECUTION_CHILD": backend}
        code = f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})['test_graph_values_match_pnl_and_batched_execution']({backend!r})"
        child = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
        assert child.returncode == 0, child.stdout + child.stderr
        return
    c, origin, _, _, source = _network(backend)
    values = source.derive_symbolic_values()
    json.dumps(values.explain())
    assert values.inputs == source.ir.graph.inputs
    assert values.parameters == source.ir.params
    args = _args(values)
    generated = compile_continuous_phase(_phase(values)).integrate(**args).readouts[:, 0].detach().numpy()
    inputs = {origin: args["state"].detach().numpy()}
    c.run(inputs=inputs)
    np.testing.assert_allclose(generated, np.asarray(c.results)[:, 0], rtol=2e-7, atol=2e-8)
    sampled = source.run(inputs, parameter_sets=[{}], num_estimates=1, seed=3)
    np.testing.assert_allclose(generated, sampled.values.reshape(3, 2), rtol=3e-6, atol=3e-7)


@native
def test_all_bound_parameters_and_inputs_have_checked_gradients():
    *_, source = _network()
    values = source.derive_symbolic_values()
    dynamics = _phase(values)
    plan = compile_continuous_phase(dynamics)
    args = _args(values, grad=True)

    def result(x, p):
        return plan.integrate(**{**args, "state": x, "inputs": x[:, :0], "parameters": p}).readouts[:, 0]

    assert torch.autograd.gradcheck(result, (args["state"], args["parameters"]), atol=2e-7, rtol=2e-5)
    actual = result(args["state"], args["parameters"])
    expected = evaluate_equations(values.expressions, dynamics, args["state"], args["inputs"], args["parameters"], args["start_time"])
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.sum(), (args["state"], args["parameters"]))
    ge = torch.autograd.grad(expected.sum(), (args["state"], args["parameters"]))
    for a, b in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-10)


def test_symbolic_rules_bind_registered_arguments_and_validate_domains():
    specs.ensure_builtin_specs()
    original = specs.function_spec_for(pnl.Linear())
    x, bad = sp.symbols("x bad", real=True)
    rule = sp.Lambda((x, bad), x + bad)
    with pytest.raises(specs.BatchedOpSpecError, match="every registered"):
        specs.register_batched_op(replace(original, likelihood_contract=LikelihoodEffectContract(symbolic_value=rule)))
    with pytest.raises(ValueError, match="replace randomness"):
        LikelihoodEffectContract(randomness="declared_streams", symbolic_value=rule)
    with pytest.raises(ValueError, match="Unbound"):
        LikelihoodEffectContract(symbolic_value=sp.Lambda(x, x + bad))
    with pytest.raises(ValueError, match="Unsupported"):
        LikelihoodEffectContract(symbolic_value=sp.Lambda(x, sp.log(x)))
    with pytest.raises(ValueError, match="Lambda"):
        LikelihoodEffectContract(symbolic_value=lambda x: x)


def test_snapshot_survives_live_nodes_and_registry_mutation(monkeypatch):
    c, _, _, output, source = _network()
    before = source.derive_symbolic_values()
    original = specs.function_spec_for(output.function)
    stripped = replace(original, likelihood_contract=LikelihoodEffectContract())
    monkeypatch.setitem(specs._FUNCTION_SPECS, type(output.function), stripped)
    monkeypatch.setitem(specs._SPECS_BY_KEY, original.key, stripped)
    output.function.parameters.gain.set(99.)
    after = source.derive_symbolic_values()
    assert before.expressions == after.expressions
    assert before.parameters == after.parameters
    with pytest.raises(LikelihoodPlanningError, match="registered symbolic rule"):
        Compiler.compile(c, outputs=[output.output_port]).derive_symbolic_values()


def test_symbolic_derivation_does_not_require_a_simulation_device(monkeypatch):
    c, _, _, out, source = _network()
    expected = source.derive_symbolic_values()
    monkeypatch.setattr(registry, "_backend_availability", lambda backend: (False, []))
    actual = Compiler.derive_symbolic_values(c, outputs=[out.output_port])
    assert actual.expressions == expected.expressions


@pytest.mark.parametrize("change", ["noise", "clip", "integration", "event"])
def test_state_noise_clipping_and_events_are_not_silently_removed(change):
    if change == "event":
        node = pnl.DDM(function=pnl.DriftDiffusionIntegrator(), output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])
    else:
        kwargs = dict(noise=.1) if change == "noise" else dict(clip=(.1, .9)) if change == "clip" else dict(
            integrator_mode=True, reset_stateful_function_when=pnl.AtTrialStart())
        node = pnl.TransferMechanism(function=pnl.Logistic(), **kwargs)
    c = pnl.Composition(pathways=[node])
    if change == "integration":
        c.scheduler.add_condition(node, pnl.AtPass(0))
    source = Compiler.compile(c)
    with pytest.raises(LikelihoodPlanningError):
        source.derive_symbolic_values()


@pytest.mark.parametrize("change", ["schedule", "termination"])
def test_unsupported_trial_semantics_are_not_reinterpreted(change):
    *_, source = _network()
    kernel = source.kernel_ir
    graph = kernel.graph
    if change == "schedule":
        graph = replace(graph, metadata={**graph.metadata, "schedule_kind": "dynamic_lane_local"})
    else:
        graph = replace(graph, termination=())
    with pytest.raises((LikelihoodPlanningError, ValueError)):
        derive_symbolic_values(replace(kernel, graph=graph))


@native
def test_graph_derived_readout_backpropagates_through_shared_pde():
    *_, source = _network()
    values = source.derive_symbolic_values()
    dynamics = _phase(values)
    phase = compile_continuous_phase(dynamics)
    problem = FirstPassageProblem("continuous_time", 1, "time_only", "constant_scalar", "symmetric_linear",
                                 "point_center", "conditioned_deterministic", "choice_rt_interval")
    pde = compile_first_passage(problem, noise=.3, mesh=FirstPassageMesh(time_step=.005, spatial_points=33))
    args = {k: v[:1] for k, v in _args(values).items()}
    args.update(duration=torch.tensor([.2], dtype=torch.float64), steps=torch.tensor([40]))
    parameters = args["parameters"].clone().requires_grad_()

    def score(p):
        drift = phase.integrate(**{**args, "parameters": p}).readouts[..., 0]
        return pde.solve_observation_batch(drift=drift, threshold=p.new_tensor([.2]), collapse_rate=p.new_tensor([-.03]),
                                          interval_low=p.new_tensor([.153]), interval_high=p.new_tensor([.161]),
                                          choice=p.new_tensor([1.])).probability.log()

    assert torch.autograd.gradcheck(score, (parameters,), atol=2e-6, rtol=3e-5)


@native
def test_csi_style_graph_readout_matches_explicit_equations_and_gradients(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile"))
    from benchmark_continuous_dynamics import csi_equations, csi_graph_equations

    plans = [compile_continuous_phase(d) for d in (csi_equations(), csi_graph_equations())]
    x = torch.tensor([[.03, -.02], [-.1, .02]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[1., 0., 1., 0., .2, .8, 1.], [0., 1., .2, .8, .9, .1, -1.]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[12.], [17.]], dtype=torch.float64, requires_grad=True)
    results, gradients = [], []
    for plan in plans:
        r = plan.integrate(state=x, inputs=u, parameters=p, duration=x.new_full((2,), .045),
                           start_time=x.new_zeros(2), steps=torch.full((2,), 45, dtype=torch.int64))
        results.append((r.readouts, r.final_state))
        gradients.append(torch.autograd.grad(r.readouts.square().sum() + r.final_state.sum(), (x, u, p)))
    for a, b in zip((*results[0], *gradients[0]), (*results[1], *gradients[1]), strict=True):
        torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-10)
