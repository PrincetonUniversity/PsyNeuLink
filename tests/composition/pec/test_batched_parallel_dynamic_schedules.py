"""Compositional acceptance for the generic lane-local scheduler.

The two controlled chains share consideration sets but have independent state,
finished values, modulation effects, and lane-local execution counts.  Adding
the second chain must not require another topology-specific KernelIR or Triton
loop.
"""

from dataclasses import dataclass
import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.kernel_ir import iter_kernel_ops


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_BUILD_NUMBERS = itertools.count()
_EXPECTED = np.asarray(
    [
        [1.0202979047745746, -1.0704821522954266],
        [-0.0769758015573534, 0.9176584685525668],
    ],
    dtype=float,
)


@dataclass(frozen=True)
class _ParallelModel:
    composition: pnl.Composition
    inputs: dict
    outputs: tuple
    producers: tuple
    roles: dict


def _build_parallel_controlled_chains(*, cue_values=None) -> _ParallelModel:
    build_number = next(_BUILD_NUMBERS)
    composition = pnl.Composition(
        name=f"parallel controlled chains {build_number}"
    )
    inputs = {}
    outputs = []
    producers = []
    roles = {}
    if cue_values is None:
        cue_values = (
            np.asarray([[1.0], [3.0]], dtype=float),
            np.asarray([[3.0], [1.0]], dtype=float),
        )
    cue_values = tuple(np.asarray(values, dtype=float) for values in cue_values)
    assert len(cue_values) == 2
    assert cue_values[0].shape == cue_values[1].shape
    task_values = (
        np.asarray([[1.0, -1.0], [0.5, 0.25]], dtype=float),
        np.asarray([[-0.25, 1.0], [1.5, -0.5]], dtype=float),
    )

    for chain_index in range(2):
        stem = f"parallel {build_number} chain {chain_index}"
        cue = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(),
            name=f"{stem} cue",
        )
        task = pnl.TransferMechanism(
            input_shapes=2,
            function=pnl.Linear(),
            name=f"{stem} task",
        )
        producer = pnl.LCAMechanism(
            input_shapes=2,
            function=pnl.Logistic(gain=1.0),
            leak=0.0,
            competition=0.0,
            self_excitation=0.0,
            noise=0.0,
            termination_measure=pnl.TimeScale.TRIAL,
            termination_threshold=9.0,
            time_step_size=0.5,
            execute_until_finished=False,
            reset_stateful_function_when=pnl.AtTrialStart(),
            name=f"{stem} producer",
        )
        controller = pnl.ControlMechanism(
            function=pnl.Linear(),
            control_signals=[(pnl.TERMINATION_THRESHOLD, producer)],
            monitor_for_control=cue,
            modulation=pnl.OVERRIDE,
            name=f"{stem} controller",
        )
        follower = pnl.TransferMechanism(
            input_shapes=1,
            function=pnl.Linear(slope=2.0, intercept=-0.25),
            name=f"{stem} follower",
        )

        composition.add_nodes([cue, task, controller, producer, follower])
        composition.add_projection(sender=task, receiver=producer)
        composition.add_projection(
            sender=producer,
            receiver=follower,
            projection=pnl.MappingProjection(matrix=[[1.0], [-1.0]]),
        )
        for node, condition in (
            (cue, pnl.AtPass(0)),
            (task, pnl.AtPass(0)),
            (controller, pnl.AtPass(0)),
            (producer, pnl.Always()),
            (follower, pnl.WhenFinished(producer)),
        ):
            composition.scheduler.add_condition(node, condition)

        inputs[cue] = cue_values[chain_index]
        inputs[task] = np.resize(
            task_values[chain_index],
            (cue_values[chain_index].shape[0], 2),
        )
        outputs.append(follower.output_port)
        producers.append(producer)
        roles.update(
            {
                cue: f"cue{chain_index}",
                task: f"task{chain_index}",
                controller: f"controller{chain_index}",
                producer: f"producer{chain_index}",
                follower: f"follower{chain_index}",
            }
        )

    return _ParallelModel(
        composition,
        inputs,
        tuple(outputs),
        tuple(producers),
        roles,
    )


def _selected_python_results(model: _ParallelModel) -> np.ndarray:
    indices = []
    for output in model.outputs:
        matches = tuple(
            index
            for index, cim_input in enumerate(
                model.composition.output_CIM.input_ports
            )
            if any(
                projection.sender is output
                for projection in cim_input.path_afferents
            )
        )
        assert len(matches) == 1
        indices.append(matches[0])
    return np.asarray(
        [
            [
                float(np.asarray(trial[index]).reshape(-1)[0])
                for index in indices
            ]
            for trial in model.composition.results
        ],
        dtype=float,
    )


def _python_role_trace(model: _ParallelModel):
    execution_list = model.composition.scheduler.execution_list[
        model.composition.default_execution_id
    ]
    return tuple(
        frozenset(model.roles[node] for node in execution_set)
        for execution_set in execution_list
    )


def _expected_python_role_trace():
    traces = []
    for fast_follower in ("follower0", "follower1"):
        traces.extend(
            (
                frozenset({"cue0", "cue1", "task0", "task1"}),
                frozenset({"controller0", "controller1"}),
                frozenset({"producer0", "producer1"}),
                frozenset({fast_follower}),
                frozenset({"producer0", "producer1"}),
                frozenset({fast_follower}),
                frozenset({"producer0", "producer1"}),
                frozenset({"follower0", "follower1"}),
            )
        )
    return tuple(traces)


def test_parallel_controlled_chains_use_one_generic_dynamic_region():
    model = _build_parallel_controlled_chains()
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend="triton_cpu",
        outputs=model.outputs,
        max_steps=8,
    )

    assert plan.kernel_ir.executable
    assert len(plan.kernel_ir.modulations) == 2
    assert len(plan.kernel_ir.effective_parameters) == 2
    assert len(plan.kernel_ir.finished_values) == 2
    regions = tuple(
        op
        for op in iter_kernel_ops(plan.kernel_ir)
        if op.kind == "ForPasses"
    )
    assert len(regions) == 1
    assert regions[0].attrs["trace_kind"] == "lane_local_dynamic"
    program = regions[0].attrs["program"]
    assert tuple(
        tuple(member.component_id for member in item.members)
        for item in program.consideration_sets
    ) == tuple(
        item.component_ids for item in plan.kernel_ir.consideration_sets
    )


def test_parallel_controlled_chains_match_fresh_python(batched_backend):
    python_model = _build_parallel_controlled_chains()
    python_model.composition.run(
        inputs=python_model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_model)
    np.testing.assert_allclose(expected, _EXPECTED, rtol=1.0e-12, atol=1.0e-12)
    assert _python_role_trace(python_model) == _expected_python_role_trace()

    compiled_model = _build_parallel_controlled_chains()
    plan = BatchedCompositionCompiler.compile(
        compiled_model.composition,
        backend=batched_backend,
        outputs=compiled_model.outputs,
        max_steps=8,
    )
    result = plan.run(
        inputs=compiled_model.inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=17,
    )

    np.testing.assert_allclose(
        result.values[0, 0, :, 0, :],
        expected,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    assert result.metadata["truncation"]
    assert set(result.metadata["truncation"].values()) == {0.0}


def test_parallel_controlled_chains_report_independent_truncation(
    batched_backend,
):
    model = _build_parallel_controlled_chains(
        cue_values=(
            np.asarray([[3.0], [3.0], [1.0]]),
            np.asarray([[3.0], [1.0], [1.0]]),
        )
    )
    plan = BatchedCompositionCompiler.compile(
        model.composition,
        backend=batched_backend,
        outputs=model.outputs,
        max_steps=2,
    )

    with pytest.warns(UserWarning, match="truncated bounded loops"):
        result = plan.run(
            inputs=model.inputs,
            parameter_sets=[{}],
            num_estimates=1,
            seed=17,
        )

    assert set(result.metadata["truncation"]) == {
        producer.name for producer in model.producers
    }
    expected_fractions = (2.0 / 3.0, 1.0 / 3.0)
    for producer, expected_fraction in zip(
        model.producers,
        expected_fractions,
    ):
        assert result.metadata["truncation"][producer.name] == pytest.approx(
            expected_fraction
        )
