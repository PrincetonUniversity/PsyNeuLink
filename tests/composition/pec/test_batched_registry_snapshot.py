import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched import registry as batched_registry
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def _make_reducer_composition(node_name):
    def product(variable):
        values = np.asarray(variable, dtype=float).reshape(-1)
        return values[0] * values[1] if values.size >= 2 else 0.0

    left = pnl.ProcessingMechanism(input_shapes=1, name="snapshot-left")
    right = pnl.ProcessingMechanism(input_shapes=1, name="snapshot-right")
    reducer = pnl.ProcessingMechanism(
        name=node_name,
        input_ports=[
            {pnl.NAME: "in", pnl.INPUT_SHAPES: 2, pnl.COMBINE: pnl.SUM},
        ],
        function=pnl.UserDefinedFunction(custom_function=product),
    )
    composition = pnl.Composition()
    composition.add_node(left)
    composition.add_node(right)
    composition.add_node(reducer)
    composition.add_projection(
        sender=left,
        receiver=reducer,
        projection=pnl.MappingProjection(matrix=np.array([[1.0, 0.0]])),
    )
    composition.add_projection(
        sender=right,
        receiver=reducer,
        projection=pnl.MappingProjection(matrix=np.array([[0.0, 1.0]])),
    )
    return composition


def test_compiled_plan_freezes_registered_op_specs(monkeypatch):
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda backend: (True, []),
    )
    node_name = "Registry Snapshot Reducer"
    composition = _make_reducer_composition(node_name)

    try:
        @batched_node_op(node_name)
        def add_reducer(x0, x1):
            return x0 + x1

        add_plan = BatchedCompositionCompiler.compile(composition)
        add_source = triton_graph_kernel_source(add_plan.kernel_ir)

        @batched_node_op(node_name)
        def subtract_reducer(x0, x1):
            return x0 - x1

        subtract_plan = BatchedCompositionCompiler.compile(composition)
        subtract_source = triton_graph_kernel_source(subtract_plan.kernel_ir)

        unregister_batched_instance_op(node_name)

        assert add_source != subtract_source
        assert triton_graph_kernel_source(add_plan.kernel_ir) == add_source
        assert triton_graph_kernel_source(subtract_plan.kernel_ir) == subtract_source
        with pytest.raises(BatchedCompileError):
            BatchedCompositionCompiler.compile(composition)
    finally:
        unregister_batched_instance_op(node_name)
