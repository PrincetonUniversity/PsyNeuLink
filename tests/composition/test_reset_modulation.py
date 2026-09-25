import numpy as np
import pytest

import psyneulink as pnl


@pytest.mark.composition
@pytest.mark.control
# Per-node execution does not synchronize the recurrent projection's old_val
# after Python resets between trials; exercise full composition execution.
@pytest.mark.usefixtures("comp_mode_no_per_node")
@pytest.mark.parametrize("control_mode", [pnl.BEFORE, pnl.AFTER])
@pytest.mark.parametrize("modulation", [pnl.OVERRIDE, pnl.MULTIPLICATIVE, pnl.ADDITIVE])
@pytest.mark.parametrize("warm_start", [False, True])
def test_lca_reset_uses_last_sampled_gain(comp_mode, control_mode, modulation, warm_start):
    """Reset must not sample a control signal published since the LCA last ran."""
    lca = pnl.LCAMechanism(
        input_shapes=2,
        function=pnl.Logistic(gain=2., bias=-.45),
        competition=2., self_excitation=0., leak=1., noise=0.,
        time_step_size=.1, execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
    )
    allocation = pnl.ProcessingMechanism()
    controller = pnl.ControlMechanism(
        monitor_for_control=allocation,
        control_signals=[("gain", lca)], modulation=modulation,
    )
    comp = pnl.Composition(
        nodes=[lca, allocation], controller=controller,
        enable_controller=True, controller_mode=control_mode,
    )
    comp.require_node_roles(lca, pnl.NodeRole.OUTPUT)

    allocations = [3., 4., 6.]
    if warm_start:
        # Compilation must initialize the cache from this context's sampled
        # values, rather than construction defaults or the latest publication.
        comp.run({lca: [[.5, .5]], allocation: allocations[:1]})
        comp.run({lca: [[.5, .5]] * 2, allocation: allocations[1:]}, execution_mode=comp_mode)
    else:
        comp.run({lca: [[.5, .5]] * 3, allocation: allocations}, execution_mode=comp_mode)

    def logistic(x, gain):
        return 1. / (1. + np.exp(-gain * (x - .45)))

    expected = []
    sampled_gain = 2.
    # A BEFORE controller first observes the allocation node's initial value;
    # an AFTER controller first leaves its default allocation in place. Later
    # executions use the allocation node's preceding trial value in both cases.
    signals = [0. if control_mode == pnl.BEFORE else 1., *allocations[:-1]]
    for signal in signals:
        # Python reset uses the existing ParameterPort value; normal execution
        # samples the currently published signal, including its modulation rule.
        reset_activity = logistic(0., sampled_gain)
        sampled_gain = {pnl.OVERRIDE: signal, pnl.MULTIPLICATIVE: 2. * signal,
                        pnl.ADDITIVE: 2. + signal}[modulation]
        integrated = .1 * (.5 - 2. * reset_activity)
        expected.append([logistic(integrated, sampled_gain)] * 2)

    output_index = comp.get_nodes_by_role(pnl.NodeRole.OUTPUT).index(lca)
    actual = [trial[output_index] for trial in comp.results]
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


@pytest.mark.composition
@pytest.mark.control
def test_lca_first_reset_uses_initial_gain_and_bias(comp_mode):
    """Initial controller allocations must not override an LCA's reset state."""
    lca = pnl.LCAMechanism(
        input_shapes=2, function=pnl.Logistic(gain=5., bias=-.45),
        competition=8., self_excitation=0., leak=8., noise=0.,
        time_step_size=.01, execute_until_finished=False,
        reset_stateful_function_when=pnl.AtTrialStart(),
    )
    gain = pnl.ProcessingMechanism(function=pnl.Linear(slope=0., intercept=5.))
    bias = pnl.ProcessingMechanism(function=pnl.Linear(slope=0., intercept=-.45))
    gain_controller = pnl.ControlMechanism(
        monitor_for_control=gain, control_signals=[("gain", lca)], modulation=pnl.OVERRIDE,
    )
    bias_controller = pnl.ControlMechanism(
        monitor_for_control=bias, control_signals=[("bias", lca)], modulation=pnl.OVERRIDE,
    )
    comp = pnl.Composition(nodes=[lca, gain, bias, gain_controller, bias_controller])
    comp.require_node_roles(lca, pnl.NodeRole.INPUT)
    comp.run({lca: [[0., 0.]], gain: [[0.]], bias: [[0.]]}, execution_mode=comp_mode)

    # Reset: sigmoid(5 * -.45), then one step of recurrent inhibition.
    reset_activity = 1. / (1. + np.exp(2.25))
    integrated = -.08 * reset_activity
    expected = 1. / (1. + np.exp(-5. * (integrated - .45)))
    np.testing.assert_allclose(comp.results[0][0], [expected, expected], rtol=1e-6)
