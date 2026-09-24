"""Gaussian scheduled LCAs retain PNL construction state and trial history."""

from dataclasses import replace

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _model(width=2, competition=.2):
    node = pnl.LCAMechanism(
        input_shapes=width, function=pnl.Logistic(gain=1.1, bias=-.1),
        leak=.4, competition=competition, self_excitation=0.,
        noise=pnl.NormalDist(mean=.2, standard_deviation=.3, seed=11),
        time_step_size=.1, execute_until_finished=False,
        termination_measure=max, termination_threshold=0.,
        reset_stateful_function_when=pnl.Never(),
    )
    follower = pnl.TransferMechanism(input_shapes=width)
    comp = pnl.Composition(pathways=[node, follower])
    comp.scheduler.add_condition(node, pnl.Always())
    comp.scheduler.add_condition(follower, pnl.WhenFinished(node))
    return comp, node


@pytest.mark.parametrize("width", [2, 4])
def test_constructed_result_is_preserved_with_runtime_parameters(batched_backend, width):
    comp, node = _model(width)
    initial = node.output_port.defaults.value.copy()
    # PNL constructs the mechanism and RESULT port separately, using different
    # noise draws. Only the latter is the recurrent sender.
    assert not np.array_equal(initial, node.defaults.value.ravel())
    noise = node.parameters.noise.get()
    noise.parameters.mean.set(0.)
    noise.parameters.standard_deviation.set(0.)
    plan = BatchedCompositionCompiler.compile(comp, backend=batched_backend, max_steps=4)
    state = next(s for s in plan.kernel_ir.states if s.name == f"{node.name}.act")
    np.testing.assert_array_equal(state.initial_value, initial)
    assert state.function_initializer is None
    inputs = {node: np.arange(3 * width).reshape(3, width) / 10.}
    values = plan.run(inputs, [{f"{node.name}.gain": 1.8}], 3, seed=29).values[0, 0]
    # Changing gain after construction must not transform the initial sender.
    node.function.parameters.gain.set(1.8)
    comp.run(inputs)
    expected = np.array(comp.results)[:, 0, :]
    np.testing.assert_allclose(values, np.repeat(expected[:, None, :], 3, axis=1), atol=2e-6)
    # Execution and subsequent edits to the live graph cannot change a plan.
    node.output_port.defaults.value[:] = 0.
    noise.parameters.standard_deviation.set(2.)
    np.testing.assert_array_equal(
        values, plan.run(inputs, [{f"{node.name}.gain": 1.8}], 3, seed=29).values[0, 0])


def test_persistent_noise_moments_streams_and_replay(batched_backend):
    comp, node = _model(competition=0.)
    inputs = {node: np.zeros((4, 2))}
    plan = BatchedCompositionCompiler.compile(comp, backend=batched_backend, max_steps=4)
    values = plan.run(inputs, [{}, {}], 4096, seed=29).values[:, 0]
    np.testing.assert_array_equal(values[0], values[1])
    replay = plan.run(inputs, [{}], 4096, seed=29).values[0, 0]
    np.testing.assert_array_equal(values[0], replay)
    larger_cap = BatchedCompositionCompiler.compile(comp, backend=batched_backend, max_steps=8)
    np.testing.assert_array_equal(replay, larger_cap.run(inputs, [{}], 4096, seed=29).values[0, 0])

    # With no recurrence the pre-Logistic state is an exactly soluble AR(1).
    # These moments detect accidental trial resets; innovations detect reused
    # draws between trials and between accumulators.
    pre = np.log(replay / (1. - replay)) / 1.1 + .1
    a, mean, variance = .96, 0., 0.
    for row in pre:
        mean = a * mean + .2 * np.sqrt(.1)
        variance = a * a * variance + .3 ** 2 * .1
        np.testing.assert_allclose(row.mean(axis=0), mean, atol=.009)
        np.testing.assert_allclose(row.var(axis=0), variance, rtol=.08)
    previous = np.concatenate((np.zeros_like(pre[:1]), pre[:-1]), axis=0)
    innovations = pre - a * previous
    correlation = np.corrcoef(innovations.transpose(1, 0, 2).reshape(4096, -1).T)
    assert np.max(np.abs(correlation - np.eye(8))) < .06


def test_persistent_noise_initial_state_is_authenticated():
    comp, node = _model()
    kernel = BatchedCompositionCompiler.compile(comp, max_steps=4).kernel_ir
    states = tuple(replace(s, initial_value=(0., 0.)) if s.name == f"{node.name}.act" else s
                   for s in kernel.states)
    with pytest.raises(ValueError, match="retained state"):
        replace(kernel, states=states)
