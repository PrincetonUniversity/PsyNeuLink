"""General component and scheduler features used by the DAWA LCA network."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _response(*, width=4, noise=0.0, matrix=None):
    return pnl.LCAMechanism(
        input_shapes=width, function=pnl.Logistic(gain=1.1, bias=-0.1),
        leak=0.3, competition=0.2, self_excitation=0.0,
        matrix=matrix, noise=pnl.NormalDist(standard_deviation=noise),
        time_step_size=0.1, termination_threshold=0.68,
        execute_until_finished=False, reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.RESULT, pnl.DECISION_INDEX, pnl.DECISION_TIME, pnl.ENERGY],
    )


def _network(**kwargs):
    response = _response(**kwargs)
    readout = pnl.ProcessingMechanism(input_shapes=1)
    composition = pnl.Composition()
    composition.add_nodes([response, readout])
    composition.add_projection(sender=response.output_ports[pnl.DECISION_TIME], receiver=readout)
    composition.scheduler.add_condition(readout, pnl.WhenFinished(response))
    return composition, response, readout


def _python_outputs(composition, inputs, ports):
    rows = []

    def collect():
        rows.append(np.concatenate([np.asarray(port.parameters.value.get(composition)).ravel() for port in ports]))

    composition.run(inputs, call_after_trial=collect)
    return np.asarray(rows)


@pytest.mark.parametrize("width", [2, 4])
def test_scheduled_lca_dense_recurrence_and_decision_readouts(batched_backend, width):
    matrix = np.zeros((width, width))
    matrix[0, 1], matrix[1, 0] = -0.7, -0.2
    if width == 4:
        matrix[2, 3], matrix[3, 2] = -0.8, -0.3
    c, response, _ = _network(width=width, matrix=matrix)
    inputs = {response: np.array([[3., 1., 0.2, -0.1], [0.2, 3., -0.1, 0.4]])[:, :width]}
    ports = tuple(response.output_ports)
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=32, outputs=ports)
    actual = plan.run(inputs, [{}], num_estimates=3, strict_truncation=True).values[0, 0]
    expected = _python_outputs(c, inputs, ports)
    np.testing.assert_allclose(actual, np.repeat(expected[:, None, :], 3, axis=1), atol=2e-6, rtol=2e-6)


def test_gaussian_lca_streams_are_independent_and_reproducible(batched_backend):
    c, response, _ = _network(noise=0.3)
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=64,
                                             outputs=[response.output_port])
    inputs = {response: [[2., 2., 2., 2.]]}
    first = plan.run(inputs, [{}, {}], num_estimates=19, seed=27, strict_truncation=True).values
    second = plan.run(inputs, [{}], num_estimates=19, seed=27, strict_truncation=True).values
    np.testing.assert_array_equal(first[:1], second)
    np.testing.assert_array_equal(first[0], first[1])
    assert np.std(first[0, 0, 0, :, 0]) > 0.01
    assert not np.array_equal(first[..., 0], first[..., 1])
    # Plans retain structural specs and numeric defaults independently of the
    # live objects and of later specializations registered for another width.
    response.parameters.noise.get().parameters.standard_deviation.set(0.)
    response.function.parameters.gain.set(3.)
    other, _, _ = _network(width=2)
    BatchedCompositionCompiler.compile(other)
    frozen = plan.run(inputs, [{}], num_estimates=19, seed=27, strict_truncation=True).values
    np.testing.assert_array_equal(frozen, second)


def test_scheduled_lca_reports_nonterminating_trials(batched_backend):
    c, response, _ = _network()
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=3)
    with pytest.raises(BatchedTruncationError):
        plan.run({response: [[0., 0., 0., 0.]]}, [{}], num_estimates=2, strict_truncation=True)


def test_fitzhugh_nagumo_euler_transfer_matches_python(batched_backend):
    fhn = pnl.TransferMechanism(
        input_shapes=1, integrator_mode=True,
        integrator_function=pnl.FitzHughNagumoIntegrator(
            integration_method="EULER", time_step_size=0.02, mode=0.9,
            time_constant_v=0.05, time_constant_w=5., uncorrelated_activity=0.5,
            a_v=-1., b_v=1., c_v=1., threshold=0.5,
        ), function=pnl.Linear(slope=1.5, intercept=5.),
        termination_measure=pnl.TimeScale.PASS, termination_threshold=10,
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[{pnl.NAME: "w", pnl.VARIABLE: (pnl.OWNER_VALUE, 1)}],
    )
    c = pnl.Composition(pathways=fhn)
    inputs = {fhn: [[0.25], [0.4]]}
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=32)
    actual = plan.run(inputs, [{}], num_estimates=1).values[0, 0, :, 0]
    expected = _python_outputs(c, inputs, [fhn.output_port])
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_scalar_override_fanout_and_elementwise_sampling(batched_backend):
    c, response, _ = _network()
    source = pnl.ProcessingMechanism(function=pnl.Linear(slope=0., intercept=1.3))
    stimulus = pnl.ProcessingMechanism(input_shapes=4)
    control = pnl.ControlMechanism(monitor_for_control=source, modulation=pnl.OVERRIDE,
                                   control_signals=[("gain", response), ("slope", stimulus)])
    c.add_nodes([source, control, stimulus])
    c.add_projection(sender=stimulus, receiver=response)
    for node in (source, control):
        c.scheduler.add_condition(node, pnl.AtPass(0))
    for node in (stimulus, response):
        c.scheduler.add_condition(node, pnl.Always())
    inputs = {source: [[0.], [0.]], stimulus: [[2., 0., 0., 0.], [0., 2., 0., 0.]]}
    ports = [response.output_port, response.output_ports[pnl.DECISION_TIME]]
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=32, outputs=ports)
    actual = plan.run(inputs, [{}], num_estimates=1, strict_truncation=True).values[0, 0, :, 0]
    expected = _python_outputs(c, inputs, ports)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    with pytest.raises(ValueError, match="held and sampled control values"):
        plan.run(inputs, [{}], 1, initial_states={})


def test_gaussian_lca_has_the_leaky_integrator_noise_scale(batched_backend):
    c, response, _ = _network(noise=0.3)
    response.parameters.termination_threshold.set(0.)
    response.parameters.noise.get().parameters.mean.set(0.2)
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, max_steps=2,
                                             outputs=[response.output_port])
    values = plan.run({response: [[1., 1., 1., 1.]]}, [{}], num_estimates=4096, seed=81,
                      strict_truncation=True).values[0, 0, 0]
    pre = np.log(values / (1. - values)) / 1.1 + 0.1
    initial_act = 1. / (1. + np.exp(.11))
    mean = (1. - 3. * .2 * initial_act) * .1 + .2 * np.sqrt(.1)
    np.testing.assert_allclose(pre.mean(axis=0), mean, atol=.005)
    np.testing.assert_allclose(pre.var(axis=0), .3 ** 2 * .1, rtol=.06)
    correlation = np.corrcoef(pre.T)
    assert np.max(np.abs(correlation - np.eye(4))) < .06


@pytest.mark.parametrize("change", ["selector", "noise", "clip", "matrix-parameter"])
def test_extended_lca_rejects_unrepresented_semantics(change):
    matrix = [[0., -.5, 0., 0.], [-.3, 0., 0., 0.], [0., 0., 0., -.7], [0., 0., -.2, 0.]]
    c, response, _ = _network(matrix=matrix)
    if change == "selector":
        response.output_ports[pnl.DECISION_TIME]._variable_spec = (pnl.OWNER_VALUE, 0)
    elif change == "noise":
        response.parameters.noise.set(lambda: 0.1)
    elif change == "clip":
        response.parameters.clip.set((0., 1.))
    else:
        plan = BatchedCompositionCompiler.compile(c)
        from psyneulink.core.batched.prep import normalize_parameter_sets
        with pytest.raises(ValueError, match="frozen"):
            normalize_parameter_sets([{f"{response.name}.competition": .9}], plan.ir)
        return
    assert not BatchedCompositionCompiler.diagnose(c).model_supported


@pytest.mark.parametrize("change", ["function", "method", "initializer", "selector", "noise"])
def test_fitzhugh_nagumo_rejects_unrepresented_semantics(change):
    fhn = pnl.TransferMechanism(
        input_shapes=1, integrator_mode=True,
        integrator_function=pnl.FitzHughNagumoIntegrator(
            integration_method="RK4" if change == "method" else "EULER",
            initial_v=1. if change == "initializer" else 0.,
        ),
        function=pnl.Logistic() if change == "function" else pnl.Linear(),
        noise=.1 if change == "noise" else 0.,
        termination_measure=pnl.TimeScale.PASS, termination_threshold=10,
        output_ports=[{pnl.NAME: "w", pnl.VARIABLE: (pnl.OWNER_VALUE, 1)}],
    )
    if change == "selector":
        fhn.output_port.function.parameters.slope.set(2.)
    assert not BatchedCompositionCompiler.diagnose(pnl.Composition(pathways=fhn)).model_supported


_DRIVER = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa/dawa_batched_simulation.py"


@pytest.mark.triton
@pytest.mark.triton_gpu
def test_local_dawa_model_and_full_conditional_pec_surface():
    spec = importlib.util.spec_from_file_location("dawa_acceptance_driver", _DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    c, inputs, outputs = driver.build_model(deterministic=True)
    plan = BatchedCompositionCompiler.compile(c, backend="triton", max_steps=500, outputs=outputs)
    result = plan.run(inputs, [{}], 3, seed=29, strict_truncation=True)
    expected = driver.reference_results(c, inputs, outputs, "llvm")
    np.testing.assert_allclose(result.values[0, 0, :, 0], expected, atol=2e-6, rtol=1e-5)
    c, inputs, outputs = driver.build_model(deterministic=True)
    ports = (*outputs, *(driver.node(c, name).output_port for name in (
        "Control Units\n[Color, Location]", "Stimulus Units\n[Red, Blue, Left, Right]",
        "Decision Units\n[Left, Right]", "Response Units\n[Left, Right]", "LC",
    )))
    plan = BatchedCompositionCompiler.compile(c, backend="triton", max_steps=500, outputs=ports)
    actual = plan.run(inputs, [{}], 1, strict_truncation=True).values[0, 0, :, 0]
    expected_states = _python_outputs(c, inputs, ports)
    # LC publishes gain after each target samples it. Later trial resets must
    # use the previous sampled gain, not the latest control publication.
    np.testing.assert_allclose(actual, expected_states, atol=4e-7, rtol=1e-6)
    c, inputs, outputs = driver.build_model()
    report = driver.pec_smoke(c, inputs, outputs, expected, backend="triton", max_steps=500, estimates=8, seed=29)
    assert len(report["candidate_scores"]) == 2
    assert len(report["fit_coordinates"]) == 12


def test_local_dawa_source_default_completes_multiple_trials():
    spec = importlib.util.spec_from_file_location("dawa_source_schedule_driver", _DRIVER)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    # Bypass the driver's scheduling overrides: the shared source builder used
    # by all three fitting scripts must support positive thresholds itself.
    c, inputs, outputs = driver.build_model(schedule="source", deterministic=True, trials=2)
    response = driver.node(c, "Response Units\n[Left, Right]")
    trial_executions = []

    def collect():
        trial_executions.append(response.parameters.num_executions.get(c).trial)
        assert all(port.owner.parameters.num_executions.get(c).trial == 1 for port in outputs)
        assert all(driver.node(c, name).parameters.num_executions.get(c).trial == 1
                   for name in ("Bias Mechanism", "w1 Mechanism", "w2 Mechanism"))

    # A bounded run makes the original hang fail promptly. Successful trials
    # end through AllHaveRun after their WhenFinished output gates execute.
    c.run(inputs, call_after_trial=collect, termination_processing={
        pnl.TimeScale.TRIAL: pnl.Any(pnl.AllHaveRun(), pnl.AfterNPasses(200)),
    })
    assert len(trial_executions) == 2
    assert all(1 < count < 200 for count in trial_executions)
