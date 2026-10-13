"""Tests for ``Composition._propagate_element_names`` (#14, slice 3 of #11).

The propagation rules (conservative v1):

  - Identity ``MappingProjection`` with no learning → propagates labels from
    sender ``OutputPort.element_names`` to receiver ``InputPort.element_names``.
  - Non-identity matrix → no propagation (transformation changes meaning).
  - Learnable projection → no propagation (weights may move).
  - ``ControlProjection`` / ``LearningProjection`` → no propagation
    (modulators, not data flow).
  - Multi-source fan-in: only when every contributing safe projection
    presents the same labels.
  - Explicit downstream ``element_names`` always wins over propagation.

Each test uses the smallest composition that exercises one rule.
"""

import numpy as np
import psyneulink as pnl


def _run_propagation(comp):
    """Force an _analyze_graph pass so the propagation runs without execution."""
    comp._analyze_graph()


def test_identity_projection_propagates():
    src = pnl.TransferMechanism(
        input_shapes=[2], name="src", element_names=["red", "green"],
    )
    dst = pnl.TransferMechanism(input_shapes=[2], name="dst")
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(pnl.MappingProjection(sender=src, receiver=dst))
    _run_propagation(comp)
    assert dst.input_ports[0].element_names == ["red", "green"]


def test_explicit_eye_matrix_propagates():
    src = pnl.TransferMechanism(
        input_shapes=[2], name="src", element_names=["a", "b"],
    )
    dst = pnl.TransferMechanism(input_shapes=[2], name="dst")
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(
        pnl.MappingProjection(sender=src, receiver=dst, matrix=np.eye(2))
    )
    _run_propagation(comp)
    assert dst.input_ports[0].element_names == ["a", "b"]


def test_non_identity_matrix_does_not_propagate():
    """Permutation matrix is unitary but reorders elements — meanings shift."""
    src = pnl.TransferMechanism(
        input_shapes=[2], name="src", element_names=["a", "b"],
    )
    dst = pnl.TransferMechanism(input_shapes=[2], name="dst")
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(
        pnl.MappingProjection(sender=src, receiver=dst, matrix=np.array([[0, 1], [1, 0]]))
    )
    _run_propagation(comp)
    assert dst.input_ports[0].element_names is None


def test_scaled_identity_does_not_propagate():
    """2 * I still mixes magnitudes; conservative rule rejects."""
    src = pnl.TransferMechanism(
        input_shapes=[2], name="src", element_names=["a", "b"],
    )
    dst = pnl.TransferMechanism(input_shapes=[2], name="dst")
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(
        pnl.MappingProjection(sender=src, receiver=dst, matrix=2 * np.eye(2))
    )
    _run_propagation(comp)
    assert dst.input_ports[0].element_names is None


def test_explicit_downstream_element_names_wins():
    src = pnl.TransferMechanism(
        input_shapes=[2], name="src", element_names=["shorthand_a", "shorthand_b"],
    )
    dst = pnl.TransferMechanism(
        input_shapes=[2], name="dst", element_names=["explicit_x", "explicit_y"],
    )
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(pnl.MappingProjection(sender=src, receiver=dst))
    _run_propagation(comp)
    assert dst.input_ports[0].element_names == ["explicit_x", "explicit_y"]


def test_unlabeled_upstream_leaves_downstream_unlabeled():
    src = pnl.TransferMechanism(input_shapes=[2], name="src")
    dst = pnl.TransferMechanism(input_shapes=[2], name="dst")
    comp = pnl.Composition(name="comp")
    comp.add_node(src)
    comp.add_node(dst)
    comp.add_projection(pnl.MappingProjection(sender=src, receiver=dst))
    _run_propagation(comp)
    assert dst.input_ports[0].element_names is None


# ---------------------------------------------------------------------------
# CIM / nested Composition propagation (#18, Phase 2 of #11)
# ---------------------------------------------------------------------------

def test_input_cim_mirrors_input_node_labels():
    """input_CIM's port_map pair inherits labels from the INPUT-node port."""
    inp = pnl.TransferMechanism(
        input_shapes=[2], name="inp", element_names=["a", "b"],
    )
    comp = pnl.Composition(name="cim_in", nodes=[inp])
    _run_propagation(comp)
    pair = comp.input_CIM.port_map[inp.input_ports[0]]
    cim_in, cim_out = pair
    assert cim_in.element_names == ["a", "b"]
    assert cim_out.element_names == ["a", "b"]


def test_output_cim_mirrors_output_node_labels():
    """output_CIM's port_map pair inherits labels from the OUTPUT-node port."""
    out = pnl.TransferMechanism(
        input_shapes=[2], name="out", element_names=["x", "y"],
    )
    comp = pnl.Composition(name="cim_out", nodes=[out])
    _run_propagation(comp)
    pair = comp.output_CIM.port_map[out.output_ports[0]]
    cim_in, cim_out = pair
    assert cim_in.element_names == ["x", "y"]
    assert cim_out.element_names == ["x", "y"]


def test_cims_left_alone_when_inner_unlabeled():
    """No inner labels → no mirroring; CIMs stay element_names=None."""
    m = pnl.TransferMechanism(input_shapes=[2], name="m")
    comp = pnl.Composition(name="no_labels", nodes=[m])
    _run_propagation(comp)
    cim_in, cim_out = comp.input_CIM.port_map[m.input_ports[0]]
    assert cim_in.element_names is None
    assert cim_out.element_names is None


def test_explicit_cim_labels_preserved():
    """A user-set element_names on a CIM port is not overwritten."""
    inp = pnl.TransferMechanism(
        input_shapes=[2], name="inp", element_names=["a", "b"],
    )
    comp = pnl.Composition(name="explicit_cim", nodes=[inp])
    # Set explicit labels on the CIM's outer-facing input port before
    # the propagation pass runs.
    comp.input_CIM.port_map[inp.input_ports[0]][0].element_names = ["keep_me_x", "keep_me_y"]
    _run_propagation(comp)
    cim_in, cim_out = comp.input_CIM.port_map[inp.input_ports[0]]
    assert cim_in.element_names == ["keep_me_x", "keep_me_y"]
    # The other half of the pair was unset, so it does inherit:
    assert cim_out.element_names == ["a", "b"]


def test_nested_composition_propagates_through_cims():
    """A nested Composition's labeled inner ports surface on its CIMs,
    and the outer Composition's per-node pass picks them up via the
    nested→outer identity projection."""
    inner_in = pnl.TransferMechanism(
        input_shapes=[2], name="inner_in", element_names=["red", "green"],
    )
    inner_out = pnl.TransferMechanism(
        input_shapes=[2], name="inner_out", element_names=["red", "green"],
    )
    nested = pnl.Composition(name="nested", pathways=[inner_in, inner_out])
    outer_sink = pnl.TransferMechanism(input_shapes=[2], name="outer_sink")
    outer = pnl.Composition(name="outer", pathways=[nested, outer_sink])
    _run_propagation(outer)
    # The nested composition's output_CIM's outer-facing OutputPort
    # carries the labels from inner_out.
    nested_out_cim_in, nested_out_cim_out = nested.output_CIM.port_map[
        inner_out.output_ports[0]
    ]
    assert nested_out_cim_out.element_names == ["red", "green"]
    # The outer sink, which receives via an identity projection from
    # nested.output_CIM, inherits the labels.
    assert outer_sink.input_ports[0].element_names == ["red", "green"]
