import pytest

import doctest
import psyneulink as pnl

def clear_registry():
    from psyneulink.core.components.component import DeferredInitRegistry
    from psyneulink.core.components.system import SystemRegistry
    from psyneulink.core.components.process import ProcessRegistry
    from psyneulink.core.components.mechanisms.mechanism import MechanismRegistry
    from psyneulink.core.components.projections.projection import ProjectionRegistry
    # Clear Registry to have a stable reference for indexed suffixes of default names
    pnl.clear_registry(DeferredInitRegistry)
    pnl.clear_registry(SystemRegistry)
    pnl.clear_registry(ProcessRegistry)
    pnl.clear_registry(MechanismRegistry)
    pnl.clear_registry(ProjectionRegistry)

def test_state_docs():
    # get examples of mechanisms that can be used with GatingSignals/Mechanisms
    pass


def test_parameter_state_docs():
    clear_registry()
    fail, total = doctest.testmod(pnl.core.components.states.parameterstate, globs={})

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total),
                    pytrace=False)


def test_output_state_docs():
    fail, total = doctest.testmod(pnl.core.components.states.outputstate)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_control_signal_docs():
    fail, total = doctest.testmod(pnl.core.components.states.modulatorysignals.controlsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))


def test_gating_signal_docs():
    fail, total = doctest.testmod(pnl.core.components.states.modulatorysignals.gatingsignal)

    if fail > 0:
        pytest.fail("{} out of {} examples failed".format(fail, total))
