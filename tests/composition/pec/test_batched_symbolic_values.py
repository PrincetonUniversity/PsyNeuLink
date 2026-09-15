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
from psyneulink.core.batched import batched_node_op, unregister_batched_instance_op
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

    plans = [compile_continuous_phase(d) for d in (csi_equations(), csi_graph_equations(), csi_graph_equations(readout_region=True))]
    x = torch.tensor([[.03, -.02], [-.1, .02]], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([[1., 0., 1., 0., .2, .8, 1.], [0., 1., .2, .8, .9, .1, -1.]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[12.], [17.]], dtype=torch.float64, requires_grad=True)
    results, gradients = [], []
    for plan in plans:
        r = plan.integrate(state=x, inputs=u, parameters=p, duration=x.new_full((2,), .045),
                           start_time=x.new_zeros(2), steps=torch.full((2,), 45, dtype=torch.int64))
        results.append((r.readouts, r.final_state))
        gradients.append(torch.autograd.grad(r.readouts.square().sum() + r.final_state.sum(), (x, u, p)))
    for result, gradient in zip(results[1:], gradients[1:], strict=True):
        for a, b in zip((*results[0], *gradients[0]), (*result, *gradient), strict=True):
            torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-10)


def _boundary_phase(values):
    # A caller-supplied publication is a readout argument, not a claim that a
    # stochastic boundary has become an observed deterministic trajectory.
    states = values.input_symbols + values.boundary_symbols
    return ContinuousDynamics(states, (), values.parameter_symbols, (sp.S.Zero,) * len(states),
                              tuple((f"value{i}", e) for i, e in enumerate(values.expressions)))


@native
def test_stochastic_boundary_is_explicit_and_generated_derivatives_remain_correct():
    ddm = pnl.DDM(function=pnl.DriftDiffusionIntegrator(noise=.1),
                  output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME])
    out = pnl.TransferMechanism(function=pnl.Logistic(gain=1.2, bias=-.3))
    c = pnl.Composition(pathways=[[ddm, out]])
    source = Compiler.compile(c, outputs=[out.output_port])
    with pytest.raises(LikelihoodPlanningError):
        source.derive_symbolic_values()
    values = source.derive_symbolic_values(scope="readout")
    assert not values.inputs
    assert len(values.boundary_ports) == len(values.boundary_symbols) == 1
    assert source.component_bindings.ports_by_id[values.boundary_ports[0].port_id] is ddm.output_port
    assert "estimate" in values.boundary_dependencies[0].axes
    assert {p.owner_component_id for p in values.parameters} == set(values.component_ids)
    assert values.parameter_symbols == tuple(sp.Symbol(f"parameter_{p.parameter_id}", real=True) for p in values.parameters)
    assert values.explain()["guarantee"] == "conditional_readout_at_publication"
    json.dumps(values.explain())
    with pytest.raises(ValueError, match="Unbound"):
        ContinuousDynamics((sp.Symbol("unrelated", real=True),), (), values.parameter_symbols, (sp.S.Zero,),
                           (("value", values.expressions[0]),))  # Boundaries cannot be silently omitted.
    phase = compile_continuous_phase(_boundary_phase(values))
    x = torch.tensor([[-.2], [.4]], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([[p.default for p in values.parameters]] * 2, dtype=torch.float64, requires_grad=True)

    def evaluate(x, p):
        return phase.integrate(state=x, inputs=x[:, :0], parameters=p, duration=x.new_full((2,), .01),
                               start_time=x.new_zeros(2), steps=torch.ones(2, dtype=torch.int64)).readouts[:, 0]

    torch.testing.assert_close(evaluate(x, p), torch.sigmoid(1.2 * (x - .3)))
    assert torch.autograd.gradcheck(evaluate, (x, p), atol=2e-7, rtol=2e-5)


def test_zero_weight_does_not_erase_stochastic_boundary_dependency():
    from psyneulink.core.batched.continuous_ir import analyze_continuous_dynamics

    a = pnl.ProcessingMechanism(function=pnl.NormalDist())
    b = pnl.TransferMechanism(function=pnl.Linear())
    c = pnl.Composition(pathways=[[a, np.array([[0.]]), b]])
    values = Compiler.derive_symbolic_values(c, outputs=[b.output_port], scope="readout")
    assert len(values.boundary_symbols) == 1
    assert values.boundary_symbols[0] in values.expressions[0].free_symbols
    phase = replace(_boundary_phase(values), latent_initial_states=values.boundary_symbols)
    assert analyze_continuous_dynamics(phase).stochastic_readouts == ("value0",)


@pytest.mark.parametrize("effect", ["clip", "noise", "held"])
def test_readout_cuts_modified_and_conditionally_held_producers(effect):
    a = pnl.TransferMechanism(function=pnl.Linear(slope=2), **(dict(clip=(.1, .9)) if effect == "clip" else
                                                            dict(noise=.1) if effect == "noise" else {}))
    b = pnl.TransferMechanism(function=pnl.Logistic())
    c = pnl.Composition(pathways=[[a, b]])
    if effect == "held":
        c.scheduler.add_condition(a, pnl.AtPass(0))
        c.scheduler.add_condition(b, pnl.AtPass(1))
    source = Compiler.compile(c, outputs=[b.output_port])
    values = source.derive_symbolic_values(scope="readout")
    assert not values.inputs
    assert len(values.boundary_ports) == 1
    assert source.component_bindings.ports_by_id[values.boundary_ports[0].port_id] is a.output_port
    assert values.boundary_reasons[0][1] == ("held_conditional_publication" if effect == "held" else "state_noise_or_other_effect")
    assert set(values.boundary_symbols) <= values.expressions[0].free_symbols


@pytest.fixture
def csi_readout():
    from test_batched_csi_coevolving_acceptance import _model, _node, _csi_drift_rate

    c, inputs, _ = _model(ddm_noise=.1)
    drift = _node(c, "Drift Rate Value")
    batched_node_op(drift.name, likelihood_contract=LikelihoodEffectContract())(_csi_drift_rate)
    lca = _node(c, "Task Activations [C1, C2]")
    a = pnl.TransferMechanism(input_shapes=2, function=pnl.Logistic(gain=1.3, bias=-.4))
    b = pnl.TransferMechanism(input_shapes=2, function=pnl.Linear(slope=2))
    c.add_linear_processing_pathway([lca, a, np.array([[.3, -.7], [.6, .2]]), b])
    c.scheduler.add_condition(a, pnl.WhenFinished(lca))
    c.scheduler.add_condition(b, pnl.WhenFinished(lca))
    try:
        yield c, inputs, lca, a, b
    finally:
        unregister_batched_instance_op(drift.name)


@native
def test_region_in_real_csi_graph_matches_pnl_publications(csi_readout):
    c, inputs, lca, a, b = csi_readout
    source = Compiler.compile(c, outputs=[b.output_port])
    values = source.derive_symbolic_values(scope="readout")
    assert source.ir.graph.metadata["schedule_kind"] == "dynamic_lane_local"
    assert {source.component_bindings.nodes_by_id[i] for i in values.component_ids} == {a, b}
    assert all(s in source.ir.graph.scheduler for s in values.publication_schedule)
    assert source.component_bindings.ports_by_id[values.boundary_ports[0].port_id] is lca.output_port
    # Even the noise-free LCA has estimate-dependent trial-end state because
    # stochastic DDM termination determines its number of executions.
    assert "estimate" in values.boundary_dependencies[0].axes
    phase = compile_continuous_phase(_boundary_phase(values))
    actual, expected = [], []

    def capture():
        if b not in c.scheduler.execution_list[c.default_execution_id][-1]:
            return
        actual.append(np.asarray(lca.output_port.parameters.value.get(c)).reshape(-1).copy())
        expected.append(np.asarray(b.output_port.parameters.value.get(c)).reshape(-1).copy())

    c.run(inputs=inputs, call_after_time_step=capture)
    assert len(actual) > len(next(iter(inputs.values())))
    x = torch.tensor(np.asarray(actual), dtype=torch.float64)
    p = torch.tensor([[p.default for p in values.parameters]] * len(x), dtype=torch.float64)
    result = phase.integrate(state=x, inputs=x[:, :0], parameters=p, duration=x.new_full((len(x),), .01),
                             start_time=x.new_zeros(len(x)), steps=torch.ones(len(x), dtype=torch.int64))
    np.testing.assert_allclose(result.readouts[:, 0], expected, atol=2e-8, rtol=2e-7)
    # Reuse the frozen snapshot after live parameter changes.
    a.function.parameters.gain.set(77.)
    assert source.derive_symbolic_values(scope="readout") == values


@pytest.mark.parametrize("matching", [False, True])
def test_pass_gates_preserve_held_values_instead_of_recomputing_them(matching):
    a = pnl.TransferMechanism(function=pnl.Linear(slope=3))
    b = pnl.TransferMechanism(function=pnl.Logistic())
    out = pnl.TransferMechanism(function=pnl.Linear(slope=2))
    c = pnl.Composition(pathways=[[a, b, out]])
    c.scheduler.add_condition(a, pnl.AtPass(0))
    c.scheduler.add_condition(b, pnl.AtPass(1))
    c.scheduler.add_condition(out, pnl.AtPass(1 if matching else 2))
    source = Compiler.compile(c, outputs=[out.output_port])
    values = source.derive_symbolic_values(scope="readout")
    assert len(values.component_ids) == (2 if matching else 1)
    assert source.component_bindings.ports_by_id[values.boundary_ports[0].port_id] is (a if matching else b).output_port
    assert values.boundary_reasons[0][1] == "held_conditional_publication"


def test_readout_requires_supported_root_single_execution_and_valid_scope():
    c, a, _, b, _ = _network()
    with pytest.raises(ValueError, match="scope"):
        Compiler.derive_symbolic_values(c, outputs=[b.output_port], scope="guess")
    with pytest.raises(LikelihoodPlanningError, match="one execution"):
        Compiler.derive_symbolic_values(c, outputs=[a.output_port, b.output_port], scope="readout")
    b.clip = (.1, .9)
    with pytest.raises(LikelihoodPlanningError, match="Requested execution"):
        Compiler.derive_symbolic_values(c, outputs=[b.output_port], scope="readout")
