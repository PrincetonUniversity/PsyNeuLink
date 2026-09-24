import numpy as np
import optuna
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core import llvm as pnlvm
from psyneulink.core.globals.context import Context
from psyneulink.core.globals.parameters import ParameterError


pytestmark = pytest.mark.usefixtures('set_threads_to_one')
MODES = ['Python', pytest.param('LLVM', marks=pytest.mark.llvm)]


def _make_pec(*, policy=None, estimates=8, seed=29, same_seed=True, nested=False, lca=False, streams=2):
    source = pnl.ProcessingMechanism(name='Input')
    if lca:
        nodes = [pnl.LCAMechanism(
            name=f'Noise-{i}', input_shapes=1, function=pnl.Linear,
            noise=pnl.NormalDist(seed=10), leak=0., competition=0., self_excitation=0.,
            time_step_size=1., execute_until_finished=False,
            reset_stateful_function_when=pnl.AtTrialStart(),
        ) for i in range(streams)]
    else:
        nodes = [pnl.ProcessingMechanism(name=f'Noise-{i}', function=pnl.NormalDist(seed=10))
                 for i in range(streams)]
    if nested:
        inner = pnl.Composition(nodes=nodes)
        readouts = [pnl.ProcessingMechanism(name=f'Result-{i}') for i in range(streams)]
        model = pnl.Composition(nodes=[source, inner, *readouts])
        for node, readout in zip(nodes, readouts):
            model.add_projection(sender=source, receiver=node)
            model.add_projection(sender=node, receiver=readout)
        model._analyze_graph()
    else:
        model = pnl.Composition(pathways=[[source, node] for node in nodes])
    captured = []

    def objective(samples):
        captured.append(samples.copy())
        return float(np.mean(samples))

    options = {} if policy is None else {'noise_stream_policy': policy}
    pec = pnl.ParameterEstimationComposition(
        model=model, parameters={('slope', source): np.array([0., 1.])},
        outcome_variables=readouts if nested else nodes,
        optimization_function=pnl.PECOptimizationFunction(
            method=optuna.samplers.GridSampler({f'{source.name}.slope': [0., 1.]}, seed=0),
            max_iterations=max(2, estimates),
        ),
        data=pd.DataFrame(np.zeros((2, streams))),
        num_estimates=estimates, initial_seed=seed,
        same_seed_for_all_parameter_combinations=same_seed, **options,
    )
    pec.controller.function.set_pec_objective_function(objective)
    return pec, {source: np.zeros((2, 1))}, captured


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('policy', [None, 'shared_seed'])
@pytest.mark.parametrize('nested', [False, True])
def test_component_noise_streams_and_candidate_replay(mode, policy, nested):
    """Check actual simulated draws, including routing through nested compositions."""
    pec, inputs, captured = _make_pec(policy=policy, nested=nested)
    pec.controller.parameters.comp_execution_mode.set(mode)
    pec.run(inputs=inputs)
    assert len(captured) == 2
    np.testing.assert_array_equal(captured[0], captured[1])
    assert pec.noise_stream_policy == (policy or 'independent')
    assert captured[0].shape == (2, 8, 2)
    if policy == 'shared_seed':
        np.testing.assert_array_equal(captured[0][..., 0], captured[0][..., 1])
    else:
        assert not np.array_equal(captured[0][..., 0], captured[0][..., 1])
    assert np.std(captured[0], axis=1).min() > 0.

    # PNL's MT19937 initialization uses a one-element seed array. Comparing to
    # NumPy also catches offsets being applied twice at nested boundaries.
    if mode == 'Python' or pnlvm.LLVMBuilderContext.default_float_ty == pnlvm.ir.DoubleType():
        expected = np.array([
            [np.random.RandomState([29 + j if policy == 'shared_seed' else 2 * (29 + j) + i]).normal(size=2)
             for i in range(2)] for j in range(8)
        ]).transpose(2, 0, 1)
        np.testing.assert_allclose(captured[0], expected, atol=1e-14)


@pytest.mark.parametrize('mode', MODES)
def test_fresh_streams_across_candidates(mode):
    pec, inputs, captured = _make_pec(same_seed=False)
    pec.controller.parameters.comp_execution_mode.set(mode)
    pec.run(inputs=inputs)
    assert len(captured) == 2
    assert not np.array_equal(captured[0], captured[1])
    for samples in captured:
        assert not np.array_equal(samples[..., 0], samples[..., 1])


@pytest.mark.llvm
@pytest.mark.parametrize('lca', [False, True])
def test_independent_noise_statistics_and_thread_replay(lca):
    pec, inputs, _ = _make_pec(estimates=1024, lca=lca)
    _, first = pec.log_likelihood(0., inputs=inputs, return_sim_data=True)
    try:
        pnl.set_num_threads(2)
        _, second = pec.log_likelihood(1., inputs=inputs, return_sim_data=True)
    finally:
        pnl.set_num_threads(1)
    np.testing.assert_array_equal(first, second)
    draws = first.reshape(-1, 2)
    assert abs(np.corrcoef(draws.T)[0, 1]) < .1
    np.testing.assert_allclose(draws.mean(axis=0), 0., atol=.1)
    np.testing.assert_allclose(draws.var(axis=0), 1., atol=.15)


@pytest.mark.parametrize('float_type, seed_limit', [(pnlvm.ir.FloatType, 2**24), (pnlvm.ir.DoubleType, 2**32)])
def test_seed_blocks_wrap_without_collisions_or_rounding(monkeypatch, float_type, seed_limit):
    monkeypatch.setattr(pnlvm.LLVMBuilderContext, 'default_float_ty', float_type())
    pec, _, _ = _make_pec(streams=3, seed=2**32 - 1)
    controller = pec.controller
    controller._seed_counter = seed_limit // 3 - 2
    context = Context(execution_id=None)
    bases = controller.gen_new_seed_sequence(context)
    seeds = np.array(bases)[:, None] + np.arange(3)
    assert seeds.min() >= 0
    assert seeds.max() < seed_limit
    assert len(np.unique(seeds)) == seeds.size
    dtype = np.float32 if float_type is pnlvm.ir.FloatType else np.float64
    np.testing.assert_array_equal(seeds.astype(dtype), seeds)
    following = np.array(controller.gen_new_seed_sequence(context))[:, None] + np.arange(3)
    assert not np.intersect1d(seeds, following).size

    controller.parameters.num_estimates.set(seed_limit // 3 + 1)
    with pytest.raises(pnl.OptimizationControlMechanismError, match='at most .* estimates have distinct seeds'):
        controller.gen_new_seed_sequence(context)


@pytest.mark.llvm
def test_large_initial_seed_replays_without_collapsing_streams():
    pec, inputs, _ = _make_pec(seed=2**32 - 1, streams=3)
    _, first = pec.log_likelihood(0., inputs=inputs, return_sim_data=True)
    _, second = pec.log_likelihood(0., inputs=inputs, return_sim_data=True)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (2, 8, 3)
    # These would coincide if a seed were rounded on the float32 control path.
    assert len(np.unique(first[0])) == 24


def test_policy_validation_and_legacy_ocm_default():
    with pytest.raises(ParameterError, match="must be 'independent' or 'shared_seed'"):
        _make_pec(policy='invalid')
    pec, _, _ = _make_pec()
    with pytest.raises(ParameterError, match='read.only'):
        pec.controller.parameters.noise_stream_policy.set('shared_seed')

    source = pnl.ProcessingMechanism(function=pnl.NormalDist())
    model = pnl.Composition(nodes=source)
    controller = pnl.OptimizationControlMechanism(
        agent_rep=model, monitor_for_control=source, num_estimates=2, initial_seed=29,
        control_signals=[pnl.ControlSignal(modulates=('mean', source), allocation_samples=[0., 1.])],
    )
    assert controller.noise_stream_policy == 'shared_seed'
    assert controller.gen_new_seed_sequence(Context(execution_id=None)) == [31, 32]


@pytest.mark.llvm
def test_single_stream_preserves_existing_sequence():
    # Changing PEC's default must not change seeded results for one random component.
    independent, inputs, _ = _make_pec(streams=1)
    _, actual = independent.log_likelihood(0., inputs=inputs, return_sim_data=True)
    shared, inputs, _ = _make_pec(streams=1, policy='shared_seed')
    _, expected = shared.log_likelihood(0., inputs=inputs, return_sim_data=True)
    np.testing.assert_array_equal(actual, expected)
