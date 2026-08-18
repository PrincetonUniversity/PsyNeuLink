"""Tests for the ``element_names`` per-port label feature (#11).

The feature is purely additive: ``Port.element_names`` is ``None`` by default,
gets stored verbatim when explicitly passed, and the ``Mechanism`` constructor
shorthand applies the same list to the default input and output ports when
those don't already carry their own labels. No execution-time behavior
changes; the labels are static construction metadata consumed by downstream
tools (PsyNeuView's pill / Run State / Inspector via the ``_debugger``
snapshot path, plus future MDF serialization).
"""

import pytest

import psyneulink as pnl
from psyneulink.core.components.ports.port import PortError


def test_unset_element_names_default_to_none():
    """No element_names argument anywhere → both ports report None."""
    m = pnl.TransferMechanism(input_shapes=[2], name='m')
    assert m.input_ports[0].element_names is None
    assert m.output_ports[0].element_names is None


def test_mechanism_shorthand_applies_to_default_input_and_output():
    """Mechanism-level ``element_names`` lands on the default input + output port."""
    m = pnl.TransferMechanism(
        input_shapes=[2],
        name='hidden',
        element_names=['red', 'green'],
    )
    assert m.input_ports[0].element_names == ['red', 'green']
    assert m.output_ports[0].element_names == ['red', 'green']


def test_explicit_outputport_element_names_persist():
    """An OutputPort constructed with element_names carries them once owned.

    element_names is a structural Parameter, so (like every other port
    Parameter, e.g. default_input) it is populated when the port is
    instantiated into a Mechanism, not while standalone/deferred.
    """
    m = pnl.TransferMechanism(
        input_shapes=[3],
        output_ports=[pnl.OutputPort(name='RESULT', element_names=['a', 'b', 'c'])],
    )
    assert m.output_ports[0].element_names == ['a', 'b', 'c']


def test_explicit_inputport_element_names_persist():
    """An InputPort constructed with element_names carries them once owned."""
    m = pnl.TransferMechanism(
        default_variable=[[0, 0]],
        input_ports=[pnl.InputPort(name='SRC', element_names=['x', 'y'])],
    )
    assert m.input_ports[0].element_names == ['x', 'y']


def test_explicit_port_element_names_override_shorthand():
    """Per-port element_names win over the Mechanism-level shorthand.

    Important for multi-port architectures where the default input port
    and default output port carry different element-level semantics.
    """
    m = pnl.TransferMechanism(
        input_shapes=[2],
        name='m3',
        element_names=['shorthand_a', 'shorthand_b'],
        output_ports=[pnl.OutputPort(name='RESULT', element_names=['explicit_x', 'explicit_y'])],
    )
    assert m.output_ports[0].element_names == ['explicit_x', 'explicit_y']


def test_element_names_stored_as_list_not_aliased():
    """A list passed in is copied — caller mutations don't bleed in."""
    names = ['a', 'b']
    m = pnl.TransferMechanism(
        default_variable=[[0, 0]],
        input_ports=[pnl.InputPort(name='SRC', element_names=names)],
    )
    names.append('c')
    assert m.input_ports[0].element_names == ['a', 'b']


def test_falsy_element_names_treated_as_unset():
    """Empty list / None → unset (None), consistent with the
    "labels are optional" principle. Avoids surfacing zero-length arrays
    to downstream tools as if labels were intentionally provided."""
    m_none = pnl.TransferMechanism(
        default_variable=[[0]],
        output_ports=[pnl.OutputPort(name='O', element_names=None)],
    )
    assert m_none.output_ports[0].element_names is None
    m_empty = pnl.TransferMechanism(
        default_variable=[[0]],
        output_ports=[pnl.OutputPort(name='O', element_names=[])],
    )
    assert m_empty.output_ports[0].element_names is None


# ---------------------------------------------------------------------------
# Length validation (Phase 1 polish)
# ---------------------------------------------------------------------------


def test_mechanism_shorthand_length_match_ok():
    """Length matches the port's value size → no error."""
    m = pnl.TransferMechanism(
        input_shapes=[3],
        name='ok',
        element_names=['a', 'b', 'c'],
    )
    assert m.input_ports[0].element_names == ['a', 'b', 'c']
    assert m.output_ports[0].element_names == ['a', 'b', 'c']


def test_mechanism_shorthand_length_mismatch_raises():
    """Mechanism shorthand with wrong number of labels → PortError."""
    with pytest.raises(PortError, match='element_names'):
        pnl.TransferMechanism(
            input_shapes=[3],
            name='bad',
            element_names=['only', 'two'],
        )


def test_explicit_outputport_length_mismatch_raises():
    """Explicit OutputPort with mismatched element_names → PortError.

    Must attach to an owner so the deferred-init path resolves and the
    port's value shape is known.
    """
    with pytest.raises(PortError, match='element_names'):
        pnl.TransferMechanism(
            input_shapes=[2],
            name='bad_out',
            output_ports=[pnl.OutputPort(name='RESULT', element_names=['a', 'b', 'c'])],
        )


def test_explicit_inputport_length_mismatch_raises():
    """Explicit InputPort with mismatched element_names → PortError."""
    with pytest.raises(PortError, match='element_names'):
        pnl.TransferMechanism(
            default_variable=[[0, 0]],
            name='bad_in',
            input_ports=[pnl.InputPort(name='InputPort-0',
                                       element_names=['a', 'b', 'c'])],
        )
