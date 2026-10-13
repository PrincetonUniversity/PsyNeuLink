"""Tests for MDF round-trip of ``element_names`` (slice 4 of #11).

Surface tested:

  - Export side: ``InputPort.as_mdf_model`` / ``OutputPort.as_mdf_model``
    inject ``element_names`` into the MDF ``metadata`` block when set.
  - Export side: ``Mechanism_Base.as_mdf_model`` surfaces the shorthand
    ``element_names`` to the mechanism's own metadata when both default
    ports carry identical labels.
  - Import / script-generation side: ``generate_script_from_mdf`` emits
    the ``element_names=[...]`` kwarg back onto the mechanism so the
    re-imported model has labels intact.
  - Default case: ports without ``element_names`` don't accumulate a
    stray ``element_names: None`` in their metadata.
"""

import psyneulink as pnl
from psyneulink.core.globals.mdf import generate_json, generate_script_from_mdf


def _mech_with_shorthand():
    m = pnl.TransferMechanism(
        input_shapes=[2],
        element_names=['red_word', 'green_word'],
        name='word_hidden',
    )
    pnl.Composition(name='c', nodes=[m])
    return m


def test_input_port_export_includes_element_names():
    m = _mech_with_shorthand()
    mdf = m.input_ports[0].as_mdf_model()
    assert mdf.metadata.get('element_names') == ['red_word', 'green_word']


def test_output_port_export_includes_element_names():
    m = _mech_with_shorthand()
    mdf = m.output_ports[0].as_mdf_model()
    assert mdf.metadata.get('element_names') == ['red_word', 'green_word']


def test_mechanism_export_surfaces_shorthand_element_names():
    """When both default ports carry the same labels, the mechanism's
    own metadata gets the shorthand so the script generator can emit
    ``element_names=...`` on the mechanism constructor."""
    m = _mech_with_shorthand()
    mdf = m.as_mdf_model()
    assert mdf.metadata.get('element_names') == ['red_word', 'green_word']


def test_unset_port_has_no_element_names_in_metadata():
    """A port without element_names doesn't accumulate a stray entry."""
    m = pnl.TransferMechanism(input_shapes=[2], name='plain')
    pnl.Composition(name='c2', nodes=[m])
    ip = m.input_ports[0].as_mdf_model()
    op = m.output_ports[0].as_mdf_model()
    assert 'element_names' not in ip.metadata
    assert 'element_names' not in op.metadata


def test_unset_mechanism_has_no_shorthand_in_metadata():
    m = pnl.TransferMechanism(input_shapes=[2], name='plain')
    pnl.Composition(name='c3', nodes=[m])
    assert 'element_names' not in m.as_mdf_model().metadata


def test_script_generator_emits_element_names_kwarg():
    """End-to-end round-trip: export → script → exec → port labels intact."""
    m = _mech_with_shorthand()
    comp = pnl.Composition(name='roundtrip', nodes=[m])
    # ``comp`` is collected by exporting the active composition list
    mdf_json = generate_json(comp)
    script = generate_script_from_mdf(mdf_json)
    assert "element_names=['red_word', 'green_word']" in script
    ns = {}
    exec(script, ns)
    m2 = ns['word_hidden']
    assert m2.input_ports[0].element_names == ['red_word', 'green_word']
    assert m2.output_ports[0].element_names == ['red_word', 'green_word']


def test_divergent_port_labels_not_surfaced_as_shorthand():
    """If input and output port carry different labels, the mechanism's
    metadata shorthand isn't emitted (the labels still live on each
    port's own metadata for tools that want them)."""
    m = pnl.TransferMechanism(
        default_variable=[[0, 0]],
        name='divergent',
        input_ports=[pnl.InputPort(name='InputPort-0', element_names=['in_a', 'in_b'])],
        output_ports=[pnl.OutputPort(name='RESULT', element_names=['out_x', 'out_y'])],
    )
    pnl.Composition(name='c4', nodes=[m])
    mdf = m.as_mdf_model()
    assert 'element_names' not in mdf.metadata
