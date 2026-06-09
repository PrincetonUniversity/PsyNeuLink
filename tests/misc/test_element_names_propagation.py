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
