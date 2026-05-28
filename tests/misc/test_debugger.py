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
    # These must fire at least once for any non-trivial composition run.
    for required in (
        BreakpointCategory.BEGINNING_OF_RUN,
        BreakpointCategory.BEGINNING_OF_TRIAL,
        BreakpointCategory.EXECUTION_SET,
        BreakpointCategory.END_OF_EXECUTION_SET,
        BreakpointCategory.INPUTS_TO_NODE,
        BreakpointCategory.NODE_EXECUTION,
        BreakpointCategory.END_OF_TRIAL,
        BreakpointCategory.END_OF_RUN,
    ):
        assert required in fired_categories, f"expected {required} to fire, got {fired_categories}"

    # Locals shape per category — informal schema check.
    for cat, locs in fired:
        if cat is BreakpointCategory.BEGINNING_OF_RUN:
            assert {"scheduler", "context", "num_trials"} <= locs.keys()
        elif cat is BreakpointCategory.END_OF_RUN:
            assert {"scheduler", "context", "results"} <= locs.keys()
        elif cat is BreakpointCategory.BEGINNING_OF_TRIAL:
            assert {"trial_num", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.END_OF_TRIAL:
            assert {"trial_num", "scheduler", "context", "outputs"} <= locs.keys()
        elif cat is BreakpointCategory.EXECUTION_SET:
            assert {"execution_set", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.END_OF_EXECUTION_SET:
            assert {"execution_set", "scheduler", "context", "outputs"} <= locs.keys()
        elif cat is BreakpointCategory.INPUTS_TO_NODE:
            assert {"node", "execution_set", "scheduler", "context"} <= locs.keys()
        elif cat is BreakpointCategory.NODE_EXECUTION:
            assert {"node", "execution_set", "scheduler", "context"} <= locs.keys()


def test_run_level_categories_fire_once_per_run():
    """BEGINNING_OF_RUN and END_OF_RUN bracket all trial-level events, once each."""
    fired = []
    _debugger.set_listener(lambda c, lp: fired.append(c))

    mech = pnl.TransferMechanism(name="T_run_bookends")
    comp = pnl.Composition(name="comp_run_bookends")
    comp.add_node(mech)
    try:
        comp.run(inputs={mech: [[1.0], [2.0], [3.0]]}, num_trials=3)
    finally:
        _debugger.set_listener(None)

    assert fired.count(BreakpointCategory.BEGINNING_OF_RUN) == 1
    assert fired.count(BreakpointCategory.END_OF_RUN) == 1

    # All trial-level events must fall between the run-level bookends.
    # (PARAMETER_SETTING can fire outside — composition setup/teardown
    # mutates parameters before/after the run proper.)
    begin_idx = fired.index(BreakpointCategory.BEGINNING_OF_RUN)
    end_idx = fired.index(BreakpointCategory.END_OF_RUN)
    assert begin_idx < end_idx

    trial_level = {
        BreakpointCategory.BEGINNING_OF_TRIAL,
        BreakpointCategory.EXECUTION_SET,
        BreakpointCategory.END_OF_EXECUTION_SET,
        BreakpointCategory.INPUTS_TO_NODE,
        BreakpointCategory.NODE_EXECUTION,
        BreakpointCategory.END_OF_TRIAL,
    }
    for i, cat in enumerate(fired):
        if cat in trial_level:
            assert begin_idx < i < end_idx, \
                f"{cat} at index {i} fell outside [{begin_idx}, {end_idx}]"


def test_end_of_execution_set_reports_outputs():
    """END_OF_EXECUTION_SET fires per set with freshly-computed output values."""
    captured = []

    def listener(category, locals_provider):
        if category is BreakpointCategory.END_OF_EXECUTION_SET:
            captured.append(locals_provider())

    mech = pnl.TransferMechanism(name="T_end_of_set")
    comp = pnl.Composition(name="comp_end_of_set")
    comp.add_node(mech)

    _debugger.set_listener(listener)
    try:
        comp.run(inputs={mech: [[2.0]]}, num_trials=1)
    finally:
        _debugger.set_listener(None)

    assert captured, "END_OF_EXECUTION_SET should fire at least once"
    locs = captured[0]
    assert mech in locs["execution_set"]
    assert mech in locs["outputs"]
    # TransferMechanism with input 2.0 and default linear function returns [[2.0]].
    assert locs["outputs"][mech][0][0] == 2.0


def test_exception_category_fires_then_reraises():
    """A failing node execution fires EXCEPTION and re-raises the original error."""
    fired = []

    def listener(category, locals_provider):
        if category is BreakpointCategory.EXCEPTION:
            fired.append(locals_provider())

    class _ExplodingError(RuntimeError):
        pass

    # The custom function is also invoked during _instantiate_value at __init__,
    # so we gate the explosion behind a flag flipped only just before run().
    state = {"armed": False}

    def maybe_explode(variable, context=None):
        if state["armed"]:
            raise _ExplodingError("boom")
        return variable

    mech = pnl.ProcessingMechanism(name="T_explode", function=maybe_explode)
    comp = pnl.Composition(name="comp_explode")
    comp.add_node(mech)

    _debugger.set_listener(listener)
    try:
        state["armed"] = True
        with pytest.raises(_ExplodingError, match="boom"):
            comp.run(inputs={mech: [[1.0]]}, num_trials=1)
    finally:
        state["armed"] = False
        _debugger.set_listener(None)

    assert fired, "EXCEPTION category should have fired"
    locs = fired[0]
    assert isinstance(locs["exception"], _ExplodingError)
    assert locs["trial_num"] == 0
    assert {"scheduler", "context"} <= locs.keys()


def test_no_exception_means_no_exception_category():
    """A clean run must not fire EXCEPTION."""
    fired = []
    _debugger.set_listener(lambda c, lp: fired.append(c))

    mech = pnl.TransferMechanism(name="T_clean")
    comp = pnl.Composition(name="comp_clean")
    comp.add_node(mech)
    try:
        comp.run(inputs={mech: [[1.0]]}, num_trials=1)
    finally:
        _debugger.set_listener(None)

    assert BreakpointCategory.EXCEPTION not in fired


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
