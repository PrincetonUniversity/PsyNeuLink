from dataclasses import replace

import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched import registry as batched_registry
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.kernel_ir import (
    KernelValue,
    add_constant_op,
    clamp_op,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


@pytest.fixture
def linear_kernel(monkeypatch):
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda backend: (True, []),
    )
    mechanism = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=2.0, intercept=0.5),
        name="constant-transform",
    )
    composition = pnl.Composition(pathways=mechanism)
    return BatchedCompositionCompiler.compile(composition).kernel_ir


def _with_constant_transforms(kernel, *, add, lower, upper):
    load = next(op for op in kernel.ops if op.kind == "LoadInput")
    call = next(op for op in kernel.ops if op.kind == "CallFunction")
    store = next(op for op in kernel.ops if op.kind == "StoreOutput")

    added = KernelValue("test:add", load.outputs[0].width)
    function_output = KernelValue("test:function", added.width)
    clamped = KernelValue("test:clamp", function_output.width)
    ops = (
        load,
        add_constant_op(
            target=call.target,
            input_value=load.outputs[0],
            output_value=added,
            value=add,
        ),
        replace(call, inputs=(added,), outputs=(function_output,)),
        clamp_op(
            target=call.target,
            input_value=function_output,
            output_value=clamped,
            lower=lower,
            upper=upper,
        ),
        replace(store, inputs=(clamped,)),
    )
    return replace(kernel, ops=ops)


@pytest.mark.parametrize(
    "add, lower, upper, expected_lines",
    [
        (
            0.25,
            -1.0,
            1.0,
            (
                "n_test_add_0 = tl.load(input_0 + "
                "(subject_idx * num_trials + trial_idx) * 2 + 0, "
                "mask=mask, other=0.0) + (0.25)",
                "n_test_add_1 = tl.load(input_0 + "
                "(subject_idx * num_trials + trial_idx) * 2 + 1, "
                "mask=mask, other=0.0) + (0.25)",
                "n_test_clamp_0 = tl.minimum(tl.maximum(n_test_function_0, -1.0), 1.0)",
                "n_test_clamp_1 = tl.minimum(tl.maximum(n_test_function_1, -1.0), 1.0)",
            ),
        ),
        (
            (0.25, -0.5),
            (-1.0, 0.0),
            (1.0, 2.0),
            (
                "n_test_add_0 = tl.load(input_0 + "
                "(subject_idx * num_trials + trial_idx) * 2 + 0, "
                "mask=mask, other=0.0) + (0.25)",
                "n_test_add_1 = tl.load(input_0 + "
                "(subject_idx * num_trials + trial_idx) * 2 + 1, "
                "mask=mask, other=0.0) + (-0.5)",
                "n_test_clamp_0 = tl.minimum(tl.maximum(n_test_function_0, -1.0), 1.0)",
                "n_test_clamp_1 = tl.minimum(tl.maximum(n_test_function_1, 0.0), 2.0)",
            ),
        ),
    ],
    ids=("scalar-broadcast", "exact-width-vector"),
)
def test_constant_add_and_clamp_emit_typed_components(
    linear_kernel,
    add,
    lower,
    upper,
    expected_lines,
):
    kernel = _with_constant_transforms(
        linear_kernel,
        add=add,
        lower=lower,
        upper=upper,
    )

    assert tuple(op.kind for op in kernel.ops) == (
        "LoadInput",
        "AddConstant",
        "CallFunction",
        "Clamp",
        "StoreOutput",
    )
    source = triton_graph_kernel_source(kernel)
    for line in expected_lines:
        assert line in source
    assert source.index("n_test_add_0 =") < source.index("n_test_function_0 =")
    assert source.index("n_test_function_0 =") < source.index("n_test_clamp_0 =")


def test_constant_ops_require_scalar_or_exact_width_vectors():
    input_value = KernelValue("input", 2)
    output_value = KernelValue("output", 2)

    with pytest.raises(ValueError, match="must be scalar or have width 2, got width 3"):
        add_constant_op(
            target="node",
            input_value=input_value,
            output_value=output_value,
            value=(1.0, 2.0, 3.0),
        )

    with pytest.raises(ValueError, match="must be scalar or have width 2, got width 3"):
        clamp_op(
            target="node",
            input_value=input_value,
            output_value=output_value,
            lower=(-1.0, -2.0, -3.0),
            upper=1.0,
        )


def test_constant_ops_validate_value_type_and_clamp_order():
    input_value = KernelValue("input", 2)

    with pytest.raises(ValueError, match="input/output dtypes must match"):
        add_constant_op(
            target="node",
            input_value=input_value,
            output_value=KernelValue("output", 2, dtype="float64"),
            value=1.0,
        )

    with pytest.raises(
        ValueError,
        match=r"lower bound exceeds upper bound at component 1: 3.0 > 2.0",
    ):
        clamp_op(
            target="node",
            input_value=input_value,
            output_value=KernelValue("output", 2),
            lower=(-1.0, 3.0),
            upper=(1.0, 2.0),
        )
