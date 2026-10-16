"""Python parity for persistent ACC state and scalar additive/LC-mode control."""

from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler, BatchedTrialParameter
from psyneulink.core.batched.kernel_ir import iter_kernel_ops, validate_kernel_ir


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _collect(composition, inputs, ports, before=None):
    rows = []
    composition.run(inputs, call_before_trial=before, call_after_trial=lambda: rows.append(
        np.concatenate([np.asarray(p.parameters.value.get(composition)).ravel() for p in ports])))
    return np.asarray(rows)


def _network(*, reset=False, once_per_trial=False):
    source = pnl.ProcessingMechanism()
    average = pnl.TransferMechanism(
        integrator_mode=True,
        integrator_function=pnl.AdaptiveIntegrator(rate=0.35, initializer=0.15, offset=0.02),
        function=pnl.Logistic(gain=1.3, bias=-0.4, scale=0.9, offset=0.1),
        reset_stateful_function_when=pnl.AtTrialStart() if reset else pnl.Never(),
    )
    lc = pnl.TransferMechanism(
        integrator_mode=True,
        integrator_function=pnl.FitzHughNagumoIntegrator(
            integration_method='EULER', time_step_size=0.02, mode=0.2,
            time_constant_v=0.05, time_constant_w=5., uncorrelated_activity=0.5,
        ),
        function=pnl.Linear(slope=1.5, intercept=0.2),
        termination_measure=pnl.TimeScale.PASS, termination_threshold=3,
        output_ports=[{pnl.NAME: 'w', pnl.VARIABLE: (pnl.OWNER_VALUE, 1)}],
    )
    mode = pnl.ControlMechanism(monitor_for_control=average, modulation=pnl.OVERRIDE,
                                control_signals=[(pnl.MODE, lc)])
    layers = [pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=1. + 0.2 * i, bias=-0.1),
        leak=0.2, competition=0.1, noise=0., time_step_size=0.1,
        termination_measure=max if i == 3 else pnl.TimeScale.TRIAL,
        termination_threshold=0.75 if i == 3 else 0., execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
        output_ports=[pnl.RESULT, pnl.MAX_VAL, pnl.DECISION_TIME],
    ) for i in range(4)]
    gain = pnl.ControlMechanism(monitor_for_control=lc, modulation=pnl.ADDITIVE,
                                control_signals=[(pnl.GAIN, layer) for layer in layers])
    readout = pnl.ProcessingMechanism()
    c = pnl.Composition()
    c.add_nodes([source, average, mode, lc, gain, *layers, readout])
    for layer in layers:
        c.add_node(layer, required_roles=pnl.NodeRole.INPUT)
    c.add_projection(sender=source, receiver=average)
    c.add_projection(sender=layers[-1].output_ports[pnl.DECISION_TIME], receiver=readout)
    for node in [source, average, mode, lc, gain, *layers]:
        c.scheduler.add_condition(node, pnl.Always())
    if once_per_trial:
        c.scheduler.add_condition(gain, pnl.AtPass(0))
    c.scheduler.add_condition(readout, pnl.WhenFinished(layers[-1]))
    inputs = {source: [[0.2], [1.], [-0.5]], lc: [[0.25], [0.3], [0.2]]}
    inputs.update({layer: [[2., 1.], [1., 2.], [2., 0.5]] for layer in layers})
    ports = [average.output_port, lc.output_port, *(p for layer in layers for p in layer.output_ports)]
    return c, inputs, ports, layers


@pytest.mark.parametrize('reset,once_per_trial', [(False, False), (True, True)])
def test_acc_state_mode_and_four_additive_gains_match_python(batched_backend, reset, once_per_trial):
    c, inputs, ports, layers = _network(reset=reset, once_per_trial=once_per_trial)
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend, outputs=ports, max_steps=64)
    # The held ADDITIVE signal must be combined with each trial's runtime base,
    # including when the controller has not yet executed on that trial.
    gains = [1.0, 1.8, 1.3]
    parameters = {f'{layers[0].name}.gain': BatchedTrialParameter(gains)}
    result = plan.run(inputs, [parameters], num_estimates=3, strict_truncation=True).values[0, 0]
    trial = iter(gains)
    expected = _collect(c, inputs, ports, lambda: layers[0].function.parameters.gain.set(next(trial), context=c))
    np.testing.assert_allclose(result, np.repeat(expected[:, None, :], 3, axis=1), atol=2e-6, rtol=2e-6)

    first = plan.run({n: v[:1] for n, v in inputs.items()},
                     [{f'{layers[0].name}.gain': gains[0]}], num_estimates=3,
                     return_final_states=True, strict_truncation=True)
    rest = plan.run({n: v[1:] for n, v in inputs.items()},
                    [{f'{layers[0].name}.gain': BatchedTrialParameter(gains[1:])}], num_estimates=3,
                    initial_states=first.metadata['final_states'], strict_truncation=True)
    np.testing.assert_array_equal(rest.values[0, 0], result[1:])


@pytest.mark.parametrize('reset', [False, True])
def test_adaptive_logistic_single_update_and_reset(batched_backend, reset):
    node = pnl.TransferMechanism(
        integrator_mode=True,
        integrator_function=pnl.AdaptiveIntegrator(rate=0.3, initializer=0.2, offset=-0.04),
        function=pnl.Logistic(gain=-1.5, bias=0.1),
        reset_stateful_function_when=pnl.AtTrialStart() if reset else pnl.Never(),
    )
    c = pnl.Composition(pathways=node)
    inputs = {node: [[0.5], [-0.2], [1.2]]}
    plan = BatchedCompositionCompiler.compile(c, backend=batched_backend)
    result = plan.run(inputs, [{}], num_estimates=1).values[0, 0, :, 0]
    expected = _collect(c, inputs, [node.output_port])
    np.testing.assert_allclose(result, expected, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('change', ['noise', 'clip', 'threshold', 'output'])
def test_adaptive_transfer_rejects_unmodeled_semantics(change):
    node = pnl.TransferMechanism(integrator_mode=True, function=pnl.Logistic())
    if change == 'noise':
        node.integrator_function.parameters.noise.set(0.1)
    elif change == 'clip':
        node.parameters.clip.set((0.2, 0.8))
    elif change == 'threshold':
        node.parameters.termination_threshold.set(0.5)
    else:
        node.output_port.function.parameters.slope.set(2.)
    assert not BatchedCompositionCompiler.diagnose(pnl.Composition(pathways=node)).model_supported


@pytest.mark.usefixtures("batched_runtime")
def test_additive_sampling_cannot_be_retargeted_in_kernel_ir():
    c, _, ports, _ = _network()
    plan = BatchedCompositionCompiler.compile(c, outputs=ports, max_steps=64)
    kernel = plan.kernel_ir
    region = next(op for op in iter_kernel_ops(kernel) if op.kind == 'ForPasses')
    program = region.attrs['program']
    sets = list(program.consideration_sets)
    for index, item in enumerate(sets):
        members = list(item.members)
        for member_index, member in enumerate(members):
            samples = [op for op in member.body if op.kind == 'SampleModulation']
            if not samples:
                continue
            original = samples[0]
            corrupted = replace(original, attrs={**original.attrs, 'base_parameter_id': 0})
            members[member_index] = replace(member, body=tuple(corrupted if op is original else op for op in member.body))
            sets[index] = replace(item, members=tuple(members))
            bad_region = replace(region, attrs={**region.attrs, 'program': replace(program, consideration_sets=tuple(sets))})
            trial = kernel.ops[-1]
            bad_trial = replace(trial, attrs={**trial.attrs, 'body': tuple(
                bad_region if op is region else op for op in trial.attrs['body'])})
            with pytest.raises(ValueError):
                validate_kernel_ir(replace(kernel, ops=(*kernel.ops[:-1], bad_trial)))
            return
    pytest.fail('Expected additive parameter sampling')


@pytest.mark.parametrize('change', ['max_selector', 'max_function', 'parameter_function', 'multiplicative'])
def test_control_and_max_readout_reject_unmodeled_semantics(change):
    c, _, _, layers = _network()
    layer = layers[0]
    if change == 'max_selector':
        layer.output_ports[pnl.MAX_VAL]._variable_spec = (pnl.OWNER_VALUE, 1)
    elif change == 'max_function':
        layer.output_ports[pnl.MAX_VAL].function = pnl.Linear()
    elif change == 'parameter_function':
        layer.parameter_ports[pnl.GAIN].function.parameters.slope.set(2.)
    else:
        layer.parameter_ports[pnl.GAIN].mod_afferents[0].sender.modulation = pnl.MULTIPLICATIVE
    assert not BatchedCompositionCompiler.diagnose(c).model_supported


def _local_source(*, noise):
    path = Path(__file__).resolve().parents[3] / 'Scripts/Debug/pec_batch_compile/lc_acc/full_lca_model_lc_acc.py'
    if not path.exists():
        pytest.skip('Optional local LC/ACC research model is not in the repository')
    spec = importlib.util.spec_from_file_location('lc_acc_acceptance', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    c = module.make_lca_model(sdrc_noise=noise)
    tasks, stimuli = module.conflict_task_sequence(n=6, seed=17)
    inputs = {c.nodes['Task Input']: tasks, c.nodes['Stimulus Input']: stimuli}
    inputs.update({c.nodes[name]: np.zeros((6, 1)) for name in ('Bias Mechanism', 'w1 Mechanism', 'w2 Mechanism')})
    outputs = [c.nodes[name].output_port for name in ('DECISION_GATE', 'RT_GATE')]
    return c, inputs, outputs


@pytest.mark.triton
@pytest.mark.triton_gpu
def test_local_lc_acc_source_matches_python():
    c, inputs, outputs = _local_source(noise=0.)
    ports = outputs + [c.nodes[name].output_port for name in ('ACC Short', 'ACC Long', 'ACC', 'LC')]
    ports += [c.nodes['Decision Units\n[Left, Right]'].output_ports[pnl.MAX_VAL]]
    plan = BatchedCompositionCompiler.compile(c, backend='triton', outputs=ports, max_steps=1000)
    actual = plan.run(inputs, [{}], 3, strict_truncation=True).values[0, 0]
    expected = _collect(c, inputs, ports)
    np.testing.assert_array_equal(actual[:, 0, 0], expected[:, 0])
    np.testing.assert_allclose(actual[:, 0, 1], expected[:, 1], atol=1e-7, rtol=1e-7)
    # FP32 recurrent FHN/ACC state accumulates rounding across the six trials.
    np.testing.assert_allclose(actual, np.repeat(expected[:, None, :], 3, axis=1), atol=5e-6, rtol=1e-5)


@pytest.mark.triton
@pytest.mark.triton_gpu
def test_local_lc_acc_conditional_pec_scores():
    import pandas as pd

    c, inputs, outputs = _local_source(noise=0.035)
    response, lc, gate, bias = (c.nodes[name] for name in (
        'Response Units\n[Left, Right]', 'LC', 'RT_GATE', 'Bias Mechanism'))
    data = pd.DataFrame({'decision': pd.Categorical([0., 1., 0., 1., 0., 1.]),
                         'response_time': [.313, .353, .423, .443, .403, .433],
                         'subject_nr': pd.Categorical([1, 1, 1, 2, 2, 2])})
    pec = pnl.ParameterEstimationComposition(
        nodes=c,
        parameters={('termination_threshold', response): np.linspace(.25, .7, 5),
                    ('intercept', gate): np.linspace(.1, .3, 5),
                    ('intercept', bias): np.linspace(-.5, 0., 5),
                    ('slope', lc): np.linspace(1., 4., 5),
                    ('intercept', lc): np.linspace(0., .3, 4)},
        depends_on={key: 'subject_nr' for key in (
            ('termination_threshold', response), ('intercept', gate), ('slope', lc))},
        outcome_variables=outputs, data=data,
        likelihood_include_mask=np.array([True, False, True, True, False, True]),
        optimization_function=pnl.PECOptimizationFunction(
            method='differential_evolution', max_iterations=1, batched_backend='triton',
            batched_max_steps=1000, batched_seed=17, batched_bins=20,
            batched_bin_range=[(0., 10.3)], batched_pseudocount=1.),
        num_estimates=16, initial_seed=17,
    )
    pec.controller._pec_input_values_by_node = inputs
    values = [.35, .35, .193, .193, -.48, 1.2, 1.2, .1]
    candidates = np.array([values, values])
    candidates[1, 0] = .4
    objective = pec.controller.function._make_objective_func()
    scores = objective._batched_parameter_sets(candidates)
    assert len(pec.controller.function.fit_param_names) == 8
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
