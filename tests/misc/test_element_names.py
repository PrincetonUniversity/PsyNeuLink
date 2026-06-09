"""Tests for the ``element_names`` per-port label feature (#11).

The feature is purely additive: ``Port.element_names`` is ``None`` by default,
gets stored verbatim when explicitly passed, and the ``Mechanism`` constructor
shorthand applies the same list to the default input and output ports when
those don't already carry their own labels. No execution-time behavior
changes; the labels are static construction metadata consumed by downstream
tools (PsyNeuView's pill / Run State / Inspector via the ``_debugger``
snapshot path, plus future MDF serialization).
"""

import psyneulink as pnl


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
    """Constructing an OutputPort with element_names stores them."""
    op = pnl.OutputPort(element_names=['a', 'b', 'c'])
    assert op.element_names == ['a', 'b', 'c']


def test_explicit_inputport_element_names_persist():
    """Constructing an InputPort with element_names stores them."""
    ip = pnl.InputPort(element_names=['x', 'y'])
    assert ip.element_names == ['x', 'y']


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
    op = pnl.OutputPort(element_names=names)
    names.append('c')
    assert op.element_names == ['a', 'b']


def test_falsy_element_names_treated_as_unset():
    """Empty list / None / "" → unset (None), consistent with the
    "labels are optional" principle. Avoids surfacing zero-length arrays
    to downstream tools as if labels were intentionally provided."""
    assert pnl.OutputPort(element_names=None).element_names is None
    assert pnl.OutputPort(element_names=[]).element_names is None
