"""Backend-neutral state initialization through a registered function op."""

import pytest

import psyneulink as pnl
from psyneulink.core.batched import specs
from psyneulink.core.batched.backend.triton.emit import triton_graph_kernel_source
from psyneulink.core.batched.graph import STATEFUL_GRAPH_FUSION
from psyneulink.core.batched.ir import (
    BatchedGraphIR,
    BatchedNodeSpec,
    BatchedParamSpec,
    BatchedResetSpec,
    BatchedStateFunctionInitializer,
    BatchedStateSpec,
)
from psyneulink.core.batched.kernel_ir import (
    KernelIR,
    KernelLaneLayout,
    KernelOp,
    STATEFUL_LANE_LAYOUT,
)


pytestmark = pytest.mark.batched


def _state_initializer_kernel(*, omit_parameter=None):
    specs.ensure_builtin_specs()
    logistic_spec = specs.function_spec_for(pnl.Logistic())
    assert logistic_spec is not None
    parameter_defaults = {
        "gain": 2.0,
        "bias": 0.25,
        "x_0": -0.5,
        "scale": 1.5,
        "offset": 0.125,
    }
    parameters = tuple(
        BatchedParamSpec(
            name,
            default,
            parameter_id=index,
            owner_component_id=0,
            owner_scope="function",
        )
        for index, (name, default) in enumerate(parameter_defaults.items())
    )
    initializer_params = {
        name: name
        for name in parameter_defaults
        if name != omit_parameter
    }
    initializer = BatchedStateFunctionInitializer(
        spec_key=logistic_spec.key,
        input_value=(0.0, 0.0),
        params=initializer_params,
    )
    states = (
        BatchedStateSpec(
            "lca.pre",
            "lca",
            2,
            (0.0, 0.0),
            component_id=0,
            state_id=0,
        ),
        BatchedStateSpec(
            "lca.act",
            "lca",
            2,
            (0.0, 0.0),
            component_id=0,
            state_id=1,
            function_initializer=initializer,
        ),
    )
    graph = BatchedGraphIR(
        nodes=(
            BatchedNodeSpec(
                "lca",
                "LCAMechanism",
                "Logistic",
                2,
                2,
                component_id=0,
            ),
        ),
        inputs=(),
        projections=(),
        outputs=(),
        states=states,
        resets=(
            BatchedResetSpec(
                node="lca",
                condition_type="Never",
                state_ids=(0, 1),
                component_id=0,
            ),
        ),
        scheduler=(),
        ops=(),
        execution_order=("lca",),
        fusion_kind=STATEFUL_GRAPH_FUSION,
    )
    kernel = KernelIR(
        model_kind="graph",
        fusion_kind=STATEFUL_GRAPH_FUSION,
        lane_layout=KernelLaneLayout(
            STATEFUL_LANE_LAYOUT,
            ("parameter_set", "subject", "estimate"),
        ),
        inputs=(),
        params=parameters,
        states=states,
        outputs=(),
        rng_streams=(),
        ops=(
            KernelOp(kind="InitializeState", target="lane"),
            KernelOp(kind="ForTrials", target="trials", attrs={"body": ()}),
        ),
        output_names=(),
        max_steps=1,
        graph=graph,
        op_specs=specs.snapshot_batched_op_specs((logistic_spec.key,)),
        resets=graph.resets,
    )
    return kernel, logistic_spec.triton_template.name


def test_state_initializer_calls_registered_logistic_with_lane_parameters():
    kernel, helper_name = _state_initializer_kernel()

    source = triton_graph_kernel_source(kernel)

    assert source.count(f" = {helper_name}(") == 2
    assert source.count("tl.full((BLOCK,), 0.0, tl.float32)") == 4
    expected_args = ", ".join(
        ["tl.full((BLOCK,), 0.0, tl.float32)"]
        + [f"param_{index}_value" for index in range(5)]
    )
    assert source.count(f"{helper_name}({expected_args})") == 2


def test_state_initializer_rejects_missing_function_parameter_binding():
    kernel, _ = _state_initializer_kernel(omit_parameter="bias")

    with pytest.raises(ValueError, match="no parameter binding for 'bias'"):
        triton_graph_kernel_source(kernel)
