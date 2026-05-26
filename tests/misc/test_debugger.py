"""Tests for the psyneulink._debugger hook (Phase 1 of stepping/breakpoint work).

The hook is intentionally minimal: a single global listener, lazy locals,
and emission at well-defined call sites. These tests verify (a) no-listener
is a true no-op, (b) registering a listener receives the expected categories
when running a small composition, and (c) the locals provider produces the
documented keys per category on demand.
"""

import pytest

import psyneulink as pnl
from psyneulink import _debugger
from psyneulink._debugger import BreakpointCategory


@pytest.fixture(autouse=True)
def _clear_listener():
    """Each test starts and ends with no listener registered."""
    _debugger.set_listener(None)
    yield
    _debugger.set_listener(None)


def test_no_listener_is_noop():
    """``step()`` with no listener registered must do nothing and not raise.

    The hot path is a single ``is not None`` comparison; this exercises it.
    """
    assert _debugger.get_listener() is None
    _debugger.step(
        BreakpointCategory.NODE_EXECUTION,
        lambda: {"node": "would build this if asked"},
    )


def test_set_and_clear_listener():
    calls = []

    def listener(category, locals_provider):
        calls.append(category)

    _debugger.set_listener(listener)
    assert _debugger.get_listener() is listener

    _debugger.step(BreakpointCategory.EXECUTION_SET, lambda: {})
    assert calls == [BreakpointCategory.EXECUTION_SET]

    _debugger.set_listener(None)
    _debugger.step(BreakpointCategory.EXECUTION_SET, lambda: {})
    assert calls == [BreakpointCategory.EXECUTION_SET], \
        "step() after clearing listener must not fire"


def test_locals_provider_is_lazy():
    """The locals dict is only built when the listener calls the provider."""
    built = []

    def builder():
        built.append(True)
        return {"key": "value"}

    def ignoring_listener(category, locals_provider):
        # Does NOT call locals_provider — provider must not have run yet.
        pass

    _debugger.set_listener(ignoring_listener)
    _debugger.step(BreakpointCategory.EXECUTION_SET, builder)
    assert built == [], "locals provider should not run when listener ignores it"

    def consuming_listener(category, locals_provider):
        locals_provider()

    _debugger.set_listener(consuming_listener)
    _debugger.step(BreakpointCategory.EXECUTION_SET, builder)
    assert built == [True], "locals provider should run when listener consumes it"


def test_categories_fire_during_composition_run():
    """Run a one-trial 1-mechanism composition and check the core categories fire."""
    fired = []

    def listener(category, locals_provider):
        fired.append((category, locals_provider()))

    mech = pnl.TransferMechanism(name="T")
    comp = pnl.Composition(name="comp")
    comp.add_node(mech)

    _debugger.set_listener(listener)
    try:
        comp.run(inputs={mech: [[1.0]]}, num_trials=1)
    finally:
        _debugger.set_listener(None)

    fired_categories = {cat for cat, _ in fired}
    # These four must fire at least once for any non-trivial composition run.
    for required in (
        BreakpointCategory.BEGINNING_OF_TRIAL,
        BreakpointCategory.EXECUTION_SET,
        BreakpointCategory.INPUTS_TO_NODE,
        BreakpointCategory.NODE_EXECUTION,
        BreakpointCategory.END_OF_TRIAL,
    ):
        assert required in fired_categories, f"expected {required} to fire, got {fired_categories}"

    # Locals shape per category — informal schema check.
    for cat, locs in fired:
        if cat is BreakpointCategory.BEGINNING_OF_TRIAL:
            assert {"trial_num", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.END_OF_TRIAL:
            assert {"trial_num", "scheduler", "context", "outputs"} <= locs.keys()
        elif cat is BreakpointCategory.EXECUTION_SET:
            assert {"execution_set", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.INPUTS_TO_NODE:
            assert {"node", "execution_set", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.NODE_EXECUTION:
            assert {"node", "execution_set", "scheduler", "context"} <= locs.keys()


def test_end_of_init_fires_on_component_creation():
    fired = []
    _debugger.set_listener(lambda c, lp: fired.append((c, lp())))
    pnl.TransferMechanism(name="end_of_init_probe")
    cats = [c for c, _ in fired]
    assert BreakpointCategory.END_OF_INIT in cats


def test_run_results_unchanged_with_recording_listener():
    """Listener attachment must not perturb composition behavior."""
    mech = pnl.TransferMechanism(name="T_regression")
    comp = pnl.Composition(name="comp_regression")
    comp.add_node(mech)
    inputs = {mech: [[1.0], [2.0], [3.0]]}

    baseline = comp.run(inputs=inputs, num_trials=3)

    _debugger.set_listener(lambda c, lp: None)
    try:
        with_listener = comp.run(inputs=inputs, num_trials=3)
    finally:
        _debugger.set_listener(None)

    assert baseline == with_listener
