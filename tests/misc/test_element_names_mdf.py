"""MDF export coverage for ``element_names``.

Because ``element_names`` is a registered structural Parameter, it is
carried into a port's MDF model automatically -- no metadata-channel
plumbing required. These tests lock in that export behavior.

Scope note: this covers the *export* side (PNL -> MDF). The full
PNL -> MDF -> generated-script -> PNL round-trip does not yet re-emit
``element_names`` (the script generator serializes ports by name only);
that is tracked as a follow-up.
"""
import pytest

pytest.importorskip(
    'modeci_mdf',
    reason='MDF methods require modeci_mdf package',
)

import psyneulink as pnl  # noqa: E402


def _port_md(port):
    return port.as_mdf_model().metadata or {}


def test_explicit_input_port_element_names_export():
    m = pnl.TransferMechanism(
        default_variable=[[0, 0]],
        input_ports=[pnl.InputPort(name='I', element_names=['red', 'green'])],
    )
    assert _port_md(m.input_ports[0]).get('element_names') == ['red', 'green']


def test_explicit_output_port_element_names_export():
    m = pnl.TransferMechanism(
        input_shapes=[2],
        output_ports=[pnl.OutputPort(name='O', element_names=['a', 'b'])],
    )
    assert _port_md(m.output_ports[0]).get('element_names') == ['a', 'b']


def test_mechanism_shorthand_element_names_export():
    """Shorthand labels serialize on both default ports (they set the
    Parameter default, which MDF reads), matching explicit per-port."""
    m = pnl.TransferMechanism(input_shapes=2, name='S', element_names=['x', 'y'])
    assert _port_md(m.input_ports[0]).get('element_names') == ['x', 'y']
    assert _port_md(m.output_ports[0]).get('element_names') == ['x', 'y']


def test_unset_element_names_serialize_as_none():
    """An unlabeled port carries ``element_names: None`` in its MDF metadata,
    consistent with the sibling structural Parameters (default_input,
    shadow_inputs). Consumers treat None as 'no labels'."""
    m = pnl.TransferMechanism(default_variable=[[0]], name='P')
    assert _port_md(m.input_ports[0]).get('element_names') is None
