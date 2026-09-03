"""Acceptance contract for the executable CSI research-model boundary.

This uses the real CSI surrogate rather than a compiler-shaped stand-in.  It
covers affine cue timing, delayed ITI, numeric LCA noise, stochastic DDM
execution, folded threshold control, persistent state, runtime fit lanes, and
the two finished-gated output mechanisms.
"""

from dataclasses import replace
import importlib.util
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompositionCompiler,
    batched_node_op,
    unregister_batched_instance_op,
)
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.backend.triton.runtime import BatchedTruncationError
from psyneulink.core.batched.graph import (
    COEVOLVING_GRAPH_FUSION,
    lower_composition,
)
from psyneulink.core.batched.kernel_ir import (
    KernelDynamicScheduleProgram,
    iter_kernel_ops,
)
from psyneulink.core.batched.prep import normalize_parameter_sets


pytestmark = [pytest.mark.batched, pytest.mark.composition]


_DRIFT_NODE_NAME = "Drift Rate Value"
_CSI_PATH = (
    Path(__file__).resolve().parents[3]
    / "Scripts"
    / "Debug"
    / "pec_batch_compile"
    / "csi_model_surrogate.py"
)
_EXPECTED = np.asarray(
    [
        [1.0, 0.54],
        [1.0, 0.59],
    ],
    dtype=float,
)


def _csi_drift_rate(x0, x1, x2, x3, x4, x5, x6):
    """Inspectable Triton transcription of the CSI nested-logistic UDF."""

    a = 1.0 / (1.0 + tl.exp(-((x0 - x1) + 4.0 * x4 - 4.0)))
    b = 1.0 / (1.0 + tl.exp(-((x1 - x0) + 4.0 * x4 - 4.0)))
    c = 1.0 / (1.0 + tl.exp(-((x2 - x3) + 4.0 * x5 - 4.0)))
    d = 1.0 / (1.0 + tl.exp(-((x3 - x2) + 4.0 * x5 - 4.0)))
    positive = 1.0 / (1.0 + tl.exp(-(a - b + c - d)))
    negative = 1.0 / (1.0 + tl.exp(-(-a + b - c + d)))
    return (positive - negative) * x6


@pytest.fixture
def registered_csi_drift_rate():
    batched_node_op(_DRIFT_NODE_NAME)(_csi_drift_rate)
    try:
        yield
    finally:
        unregister_batched_instance_op(_DRIFT_NODE_NAME)


def _make_stab_flex(**overrides):
    module_spec = importlib.util.spec_from_file_location(
        "_batched_csi_model_surrogate",
        _CSI_PATH,
    )
    assert module_spec is not None
    assert module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    options = {
        "iti": 0,
        "csi_repeat": 0,
        "csi_switch": 1,
        "threshold_collapse": -0.001,
        "ddm_noise": 0.0,
        "lca_noise": 0.0,
    }
    options.update(overrides)
    return module.make_stab_flex(
        **options,
    )


def _node(composition, base_name):
    matches = tuple(
        node
        for node in composition.nodes
        if re.sub(r"-\d+$", "", node.name) == base_name
    )
    assert len(matches) == 1
    return matches[0]


def _model(
    *,
    correct_values=None,
    csi_repeat=0,
    csi_switch=1,
    cue_values=None,
    ddm_rate=None,
    ddm_noise=0.0,
    lca_noise=0.0,
    iti=0,
    one_trial=False,
):
    composition = _make_stab_flex(
        csi_repeat=csi_repeat,
        csi_switch=csi_switch,
        ddm_noise=ddm_noise,
        lca_noise=lca_noise,
        iti=iti,
    )
    stimulus = _node(composition, "Stimulus Input")
    task = _node(composition, "Task Input")
    correct = _node(composition, "Correct Response")
    cue = _node(composition, "Cue Stimulus Interval")
    decision_gate = _node(composition, "DECISION_GATE")
    response_gate = _node(composition, "RESPONSE_GATE")
    if ddm_rate is not None:
        ddm = _node(composition, "DDM")
        ddm.function.parameters.rate.set(float(ddm_rate))
    if cue_values is None:
        cue_values = [[1.0], [3.0]]
    cue_values = np.asarray(cue_values, dtype=float).reshape(-1, 1)
    num_trials = cue_values.shape[0]
    trial_slice = slice(None, 1) if one_trial else slice(None)
    if correct_values is None:
        correct_values = np.resize(
            np.asarray([[1.0], [-1.0]], dtype=float),
            (num_trials, 1),
        )
    inputs = {
        stimulus: np.resize(
            np.asarray(
                [[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]],
                dtype=float,
            ),
            (num_trials, 4),
        )[trial_slice],
        task: np.resize(
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float),
            (num_trials, 2),
        )[trial_slice],
        correct: np.asarray(correct_values, dtype=float)[trial_slice],
        cue: cue_values[trial_slice],
    }
    outputs = (decision_gate.output_port, response_gate.output_port)
    return composition, inputs, outputs


def _selected_python_results(composition, outputs):
    result_indices = []
    for output in outputs:
        matches = tuple(
            index
            for index, cim_input in enumerate(composition.output_CIM.input_ports)
            if any(
                projection.sender is output
                for projection in cim_input.path_afferents
            )
        )
        assert len(matches) == 1
        result_indices.append(matches[0])
    return np.asarray(
        [
            [
                float(np.asarray(trial[index]).reshape(-1)[0])
                for index in result_indices
            ]
            for trial in composition.results
        ],
        dtype=float,
    )


def _run_compiled_csi(backend, **model_options):
    composition, inputs, outputs = _model(**model_options)
    plan = _compile_csi(
        composition,
        backend=backend,
        outputs=outputs,
        max_steps=128,
    )
    result = plan.run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=0,
    )
    return result.values[0, 0, :, 0, :]


def _assert_generic_csi_schedule(kernel):
    """Require every CSI runtime case to use the compositional executor."""

    regions = tuple(
        op
        for op in iter_kernel_ops(kernel)
        if op.kind == "ForPasses"
    )
    dynamic = tuple(
        op
        for op in regions
        if op.attrs.get("trace_kind") == "lane_local_dynamic"
    )
    assert len(dynamic) == 1
    assert not any(
        op.attrs.get("trace_kind") == "lane_local_coevolving"
        for op in regions
    )
    assert type(dynamic[0].attrs.get("program")) is KernelDynamicScheduleProgram
    return dynamic[0]


def _compile_csi(composition, *, backend, outputs, max_steps):
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend=backend,
        outputs=outputs,
        max_steps=max_steps,
    )
    _assert_generic_csi_schedule(plan.kernel_ir)
    return plan


def _replace_csi_dynamic_program(kernel, program):
    """Rebuild one CSI kernel around a deliberately forged schedule program."""

    region = _assert_generic_csi_schedule(kernel)
    trial = kernel.ops[-1]
    body = tuple(
        replace(op, attrs={**op.attrs, "program": program}) if op is region else op
        for op in trial.attrs["body"]
    )
    return replace(
        kernel,
        ops=(*kernel.ops[:-1], replace(trial, attrs={**trial.attrs, "body": body})),
    )


def _recovery_surface_model():
    """Build the reduced three-parameter subset of the CSI fit surface."""

    composition = _make_stab_flex(
        gain=10.0,
        csi_switch=10.0,
        non_decision_time=0.2,
        iti=0,
        csi_repeat=0,
        leak=7.0,
        competition=3.0,
        threshold=0.12,
        threshold_collapse=-0.001,
        ddm_noise=0.1,
        lca_noise=0.0,
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
    )
    stimulus = _node(composition, "Stimulus Input")
    task = _node(composition, "Task Input")
    correct = _node(composition, "Correct Response")
    cue = _node(composition, "Cue Stimulus Interval")
    threshold_source = _node(composition, "Threshold Mechanism")
    lca = _node(composition, "Task Activations [C1, C2]")
    ddm = _node(composition, "DDM")
    decision_gate = _node(composition, "DECISION_GATE")
    response_gate = _node(composition, "RESPONSE_GATE")
    inputs = {
        stimulus: np.asarray(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        task: np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]],
            dtype=float,
        ),
        correct: np.asarray([[1.0], [-1.0], [-1.0], [1.0]], dtype=float),
        cue: np.asarray([[0.0], [1.0], [0.0], [1.0]], dtype=float),
        # PEC supplies every live origin.  The batched compiler absorbs this
        # source into the collapsing-threshold DDM boundary, so this entry is
        # intentionally extra from the plan's point of view.
        threshold_source: np.zeros((4, 1), dtype=float),
    }
    common = {
        f"{lca.name}.gain": 10.0,
        f"{cue.name}.slope": 10.0,
        f"{ddm.name}.non_decision_time": 0.2,
    }
    parameter_sets = (
        common,
        {**common, f"{lca.name}.gain": 5.0},
        {**common, f"{cue.name}.slope": 0.0},
        {**common, f"{ddm.name}.non_decision_time": 0.5},
    )
    outputs = (decision_gate.output_port, response_gate.output_port)
    return composition, inputs, parameter_sets, outputs


def _recovery_pec(
    composition,
    inputs,
    outputs,
    *,
    backend,
    include_historical_threshold_parameters,
    deterministic_history_likelihood=False,
):
    """Build the real PEC wrapper used by the CSI recovery workflows."""

    cue = _node(composition, "Cue Stimulus Interval")
    lca = _node(composition, "Task Activations [C1, C2]")
    ddm = _node(composition, "DDM")
    fit_parameters = {
        ("gain", lca): np.asarray([5.0, 20.0]),
        ("slope", cue): np.asarray([0.0, 50.0]),
    }
    if include_historical_threshold_parameters:
        threshold_source = _node(composition, "Threshold Mechanism")
        fit_parameters.update(
            {
                ("intercept", threshold_source): np.asarray([0.08, 0.25]),
                ("offset-integrator_function", threshold_source): np.asarray(
                    [-0.003, 0.0]
                ),
            }
        )
    fit_parameters[("non_decision_time", ddm)] = np.asarray([0.0, 0.6])

    data = pd.DataFrame(
        {
            "decision": [1.0, 0.0, 1.0, 0.0],
            "response_time": [0.55, 0.70, 0.65, 0.80],
        }
    )
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec_csi_recovery_acceptance",
        nodes=composition,
        parameters=fit_parameters,
        outcome_variables=list(outputs),
        data=data,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
            batched_backend=backend,
            batched_max_steps=600,
            batched_bins=20,
            batched_seed=29,
            deterministic_history_likelihood=deterministic_history_likelihood,
        ),
        num_estimates=8,
        initial_seed=29,
    )
    # ``PEC.run`` normally populates this cache before constructing the batched
    # objective.  Setting it directly keeps this a focused one-evaluation smoke
    # test rather than running an optimizer.
    pec.controller._pec_input_values_by_node = inputs
    return pec


def test_csi_compiles_to_one_generic_lane_local_dynamic_region(
    registered_csi_drift_rate,
):
    composition, _, outputs = _model()
    plan = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    )
    kernel = plan.kernel_ir
    graph = kernel.graph

    assert kernel.executable
    assert graph.executable
    assert kernel.fusion_kind == COEVOLVING_GRAPH_FUSION
    assert len(kernel.modulations) == 1
    assert len(kernel.effective_parameters) == 2
    assert len(kernel.folded_affine_controls) == 1
    assert len(kernel.finished_values) == 2
    assert tuple(output.node for output in graph.outputs) == tuple(
        output.owner.name for output in outputs
    )

    all_ops = iter_kernel_ops(kernel)
    region = _assert_generic_csi_schedule(kernel)
    program = region.attrs["program"]
    assert region.attrs["declaration_only"] is False

    stepper = graph.node(_node(composition, "Task Activations [C1, C2]").name)
    terminator = graph.node(_node(composition, "DDM").name)
    members = tuple(
        member
        for consideration_set in program.consideration_sets
        for member in consideration_set.members
    )
    assert tuple(sorted(member.component_id for member in members)) == tuple(
        node.component_id for node in graph.nodes
    )

    terminator_member = next(
        member for member in members if member.component_id == terminator.component_id
    )
    terminator_step = next(
        op for op in terminator_member.body if op.kind == "StepMechanism"
    )
    trial_states = tuple(
        carry
        for carry in program.loop_carries
        if carry.kind == "trial_state"
        and carry.owner_component_id == terminator.component_id
    )
    assert tuple(
        carry.value.name.rsplit(".", 1)[-1] for carry in trial_states
    ) == ("value", "steps", "finished")
    assert terminator_step.attrs["trial_state_ids"] == tuple(
        carry.value_id for carry in trial_states
    )
    assert terminator_step.attrs["finished_trial_state_id"] == (
        trial_states[-1].value_id
    )
    assert len(
        tuple(
            publication
            for publication in terminator_member.publications
            if publication.kind == "finished"
        )
    ) == 1

    modulation = kernel.modulations[0]
    controller_member = next(
        member
        for member in members
        if member.component_id == modulation.controller_component_id
    )
    assert tuple(effect.kind for effect in controller_member.effects) == (
        "ApplyModulation",
    )
    assert any(
        carry.kind == "effective_parameter"
        and carry.value_id == modulation.effective_parameter_id
        for carry in program.loop_carries
    )
    folded = kernel.folded_affine_controls[0]
    assert folded.clock_component_id == folded.controller_component_id
    folded_member = next(
        member
        for member in members
        if member.component_id == folded.controller_component_id
    )
    affine = next(
        op for op in folded_member.body if op.kind == "AffineSchedulerValue"
    )
    assert affine.attrs == {
        "folded_control_id": folded.folded_control_id,
        "base_parameter_id": folded.base_parameter_id,
        "delta_parameter_id": folded.delta_parameter_id,
    }
    assert affine.inputs[0].name == (
        f"schedule:execution-count:{folded.clock_component_id}"
    )
    assert tuple(effect.kind for effect in folded_member.effects) == (
        "ApplyModulation",
    )
    assert terminator_step.attrs["sampled_effective_parameter_ids"] == (
        folded.effective_parameter_id,
    )
    assert any(
        carry.kind == "effective_parameter"
        and carry.value_id == folded.effective_parameter_id
        and carry.owner_component_id == folded.target_component_id
        for carry in program.loop_carries
    )
    stepper_finished = next(
        value
        for value in kernel.finished_values
        if value.component_id == stepper.component_id
    )
    stepper_finished_slot = next(
        slot
        for slot in program.scheduler_state_slots
        if slot.kind == "finished"
        and slot.owner_component_id == stepper.component_id
    )
    assert stepper_finished_slot.initialization == (
        "count_zero_vs_effective_parameter"
    )
    assert stepper_finished_slot.initial_effective_parameter_id == (
        stepper_finished.attrs["effective_parameter_id"]
    )
    assert sum(op.kind == "InitializeEffectiveParameter" for op in kernel.ops) == 2
    assert sum(op.kind == "StoreOutput" for op in all_ops) == 2

    source = triton_graph_kernel_source(kernel)
    assert "initial_state" in source
    assert "final_state" in source
    assert "if USE_INITIAL_STATE:" in source
    assert "if STORE_FINAL_STATE:" in source
    assert "SEED: tl.constexpr" not in source
    assert "TRIAL_OFFSET: tl.constexpr" not in source
    assert "RNG_NUM_TRIALS: tl.constexpr" not in source
    assert "LCA_MAX_STEPS: tl.constexpr" not in source
    assert "do_not_specialize=['num_trials', 'LCA_MAX_STEPS', 'SEED', " in source
    assert "tl.store(final_state + offsets" in source
    assert "lane_local_coevolving" not in source
    assert "coevolving_required_passes" not in source
    assert "tl.rand4x(SEED, random_base + (n_schedule_rng_clock_8_0_0 // 2))" in source
    assert "tl.pair_uniform_to_normal(" in source
    assert "dynamic_rng_spare_0" in source
    # Backend scheduler state is packed without changing the explicit typed
    # KernelIR slots.  CSI has eleven scheduled components, so their AllHaveRun
    # state occupies one word with the low eleven bits set (2**11 - 1).
    assert source.count(
        "dynamic_has_run_word_0 = tl.zeros((BLOCK,), dtype=tl.int32)"
    ) == 1
    termination_check = "(dynamic_has_run_word_0 & 2047) == 2047"
    assert source.count(termination_check) == len(program.consideration_sets)
    assert "schedule_has_run" not in source
    assert "n_schedule_pass_index_0" not in source
    for component_id in (1, 2, 3, 4, 5):
        bit = 1 << component_id
        assert f"(dynamic_has_run_word_0 & {bit}) == 0" not in source

    # The controller consumes its execution ordinal and the two stateful
    # mechanisms use it as their step index.  Every other CSI count is proven
    # redundant by either a maximum-one budget or the outer schedule fuel.
    retained_count_ids = {
        folded.clock_component_id,
        stepper.component_id,
        terminator.component_id,
    }
    for node in graph.nodes:
        count_name = f"n_schedule_execution_count_{node.component_id}_0"
        assert (count_name in source) is (
            node.component_id in retained_count_ids
        )


def test_csi_axis_analysis_accounts_for_stochastic_schedule_duration(
    registered_csi_drift_rate,
):
    composition, _, outputs = _model(ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    )
    graph = plan.kernel_ir.graph
    metadata = plan.kernel_ir.metadata["axis_dependencies"]
    axes_by_component = {
        component_id: axes
        for component_id, _, _, axes in metadata["nodes"]
    }
    lca = graph.node(_node(composition, "Task Activations [C1, C2]").name)
    ddm = graph.node(_node(composition, "DDM").name)

    # The LCA has no random input, but it runs under Always until the dynamic
    # trial region ends.  Stochastic DDM completion therefore changes how many
    # LCA steps a lane executes and must make its retained state estimate-local.
    assert "estimate" in axes_by_component[lca.component_id]
    assert "estimate" in axes_by_component[ddm.component_id]
    assert metadata["stochastic_root_component_ids"] == (ddm.component_id,)
    assert any(
        producer in metadata["estimate_invariant_component_ids"]
        and consumer in metadata["estimate_dependent_component_ids"]
        for producer, consumer, _ in metadata["estimate_frontier_edges"]
    )


@pytest.mark.parametrize(
    "forgery",
    (
        "folded-base",
        "folded-effective",
        "affine-clock",
        "affine-base",
        "folded-effect",
        "missing-sample",
        "finished-initializer",
    ),
)
def test_generic_csi_kernel_ir_rejects_folded_control_forgery(
    registered_csi_drift_rate, forgery
):
    """Authenticate every CSI-only identity at the generic KIR boundary."""

    composition, _, outputs = _model()
    kernel = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    ).kernel_ir
    region = _assert_generic_csi_schedule(kernel)
    program = region.attrs["program"]
    folded = kernel.folded_affine_controls[0]
    foreign_parameter_id = next(
        parameter.parameter_id
        for parameter in kernel.params
        if parameter.owner_component_id != folded.target_component_id
        and parameter.parameter_id != folded.delta_parameter_id
    )

    if forgery in {"folded-base", "folded-effective"}:
        changes = (
            {"base_parameter_id": foreign_parameter_id}
            if forgery == "folded-base"
            else {
                "effective_parameter_id": max(
                    parameter.effective_parameter_id
                    for parameter in kernel.effective_parameters
                )
                + 1
            }
        )
        message = "folded affine control|exact bijection"
        forged = replace(folded, **changes)
        with pytest.raises(ValueError, match=message):
            replace(
                kernel,
                graph=replace(kernel.graph, folded_affine_controls=(forged,)),
                folded_affine_controls=(forged,),
            )
        return

    if forgery == "finished-initializer":
        slot = next(
            slot
            for slot in program.scheduler_state_slots
            if slot.initialization == "count_zero_vs_effective_parameter"
        )
        foreign_effective_id = next(
            parameter.effective_parameter_id
            for parameter in kernel.effective_parameters
            if parameter.effective_parameter_id
            != slot.initial_effective_parameter_id
        )
        program = replace(
            program,
            scheduler_state_slots=tuple(
                replace(
                    candidate,
                    initial_effective_parameter_id=foreign_effective_id,
                )
                if candidate is slot
                else candidate
                for candidate in program.scheduler_state_slots
            ),
        )
        with pytest.raises(ValueError, match="compiler-derived"):
            _replace_csi_dynamic_program(kernel, program)
        return

    members = tuple(
        member
        for consideration_set in program.consideration_sets
        for member in consideration_set.members
    )
    member = next(
        item
        for item in members
        if item.component_id
        == (
            folded.target_component_id
            if forgery == "missing-sample"
            else folded.controller_component_id
        )
    )
    if forgery == "folded-effect":
        effect = member.effects[0]
        member = replace(
            member,
            effects=(
                replace(
                    effect,
                    attrs={
                        **effect.attrs,
                        "folded_control_id": effect.attrs["folded_control_id"] + 1,
                    },
                ),
            ),
        )
    else:
        kind = (
            "StepMechanism" if forgery == "missing-sample" else "AffineSchedulerValue"
        )
        op = next(item for item in member.body if item.kind == kind)
        if forgery == "affine-clock":
            foreign_clock = next(
                slot.value
                for slot in program.scheduler_state_slots
                if slot.kind == "execution_count"
                and slot.owner_component_id != folded.clock_component_id
            )
            forged_op = replace(op, inputs=(foreign_clock,))
        elif forgery == "affine-base":
            forged_op = replace(
                op,
                attrs={**op.attrs, "base_parameter_id": foreign_parameter_id},
            )
        else:
            forged_op = replace(
                op,
                inputs=(op.inputs[0], *op.inputs[2:]),
                attrs={**op.attrs, "sampled_effective_parameter_ids": ()},
            )
        member = replace(
            member,
            body=tuple(forged_op if item is op else item for item in member.body),
        )
    program = replace(
        program,
        consideration_sets=tuple(
            replace(
                item,
                members=tuple(
                    member
                    if candidate.component_id == member.component_id
                    else candidate
                    for candidate in item.members
                ),
            )
            for item in program.consideration_sets
        ),
    )
    with pytest.raises(ValueError, match="compiler-derived|sampled effective"):
        _replace_csi_dynamic_program(kernel, program)


def test_generic_csi_kernel_ir_rejects_coupled_folded_parameter_forgeries(
    registered_csi_drift_rate,
):
    """The KIR boundary reauthenticates mutable and frozen folded lanes."""

    composition, _, outputs = _model()
    kernel = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    ).kernel_ir
    folded = kernel.folded_affine_controls[0]
    params = {parameter.parameter_id: parameter for parameter in kernel.params}
    base = params[folded.base_parameter_id]
    delta = params[folded.delta_parameter_id]
    target = kernel.graph.node(folded.target)
    frozen = tuple(
        next(parameter for parameter in kernel.params if parameter.name == target.params[arg])
        for arg in ("noise", "starting_value", "offset")
    )
    frozen_reason = "dynamic scheduled terminator parameter is frozen in KernelIR"

    assert all(
        parameter.runtime_mutable is False
        and parameter.runtime_constraint == frozen_reason
        for parameter in frozen
    )

    cases = (
        ({base.parameter_id: {"default": -0.01}}, (-0.01,)),
        ({delta.parameter_id: {"default": 0.01}}, None),
        (
            {
                base.parameter_id: {
                    "minimum": None,
                    "minimum_inclusive": False,
                    "maximum": 1.0,
                    "maximum_inclusive": False,
                }
            },
            None,
        ),
        (
            {
                delta.parameter_id: {
                    "minimum": -1.0,
                    "minimum_inclusive": False,
                    "maximum": None,
                    "maximum_inclusive": False,
                }
            },
            None,
        ),
        ({base.parameter_id: {"aliases": base.aliases[1:]}}, None),
        ({delta.parameter_id: {"aliases": delta.aliases[:-1]}}, None),
        (
            {
                parameter.parameter_id: {
                    "runtime_mutable": True,
                    "runtime_constraint": "",
                }
                for parameter in frozen
            },
            None,
        ),
    )
    for parameter_changes, effective_base in cases:
        forged_params = tuple(
            replace(parameter, **parameter_changes[parameter.parameter_id])
            if parameter.parameter_id in parameter_changes
            else parameter
            for parameter in kernel.params
        )
        forged_effective = kernel.effective_parameters
        if effective_base is not None:
            forged_effective = tuple(
                replace(parameter, base_value=effective_base)
                if parameter.effective_parameter_id
                == folded.effective_parameter_id
                else parameter
                for parameter in kernel.effective_parameters
            )
        with pytest.raises(ValueError, match="folded affine control"):
            replace(
                kernel,
                params=forged_params,
                effective_parameters=forged_effective,
                graph=replace(
                    kernel.graph,
                    effective_parameters=forged_effective,
                ),
            )


def test_csi_recovery_parameter_surface_runs_multiple_parameter_lanes(
    registered_csi_drift_rate,
    batched_backend,
):
    """Exercise the stochastic three-parameter subset of the CSI fit surface.

    This is deliberately an end-to-end gate: admission alone is insufficient
    because the cue slope must remain lane-mutable and the co-evolving DDM must
    consume its trial-local random stream during execution.
    """

    composition, inputs, parameter_sets, outputs = _recovery_surface_model()
    plan = _compile_csi(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=600,
    )
    normalized = normalize_parameter_sets(parameter_sets, plan.ir)
    cue = _node(composition, "Cue Stimulus Interval")
    lca = _node(composition, "Task Activations [C1, C2]")
    ddm = _node(composition, "DDM")
    gain_name = f"{lca.name}.gain"
    switch_name = f"{cue.name}.slope"
    non_decision_name = f"{ddm.name}.non_decision_time"
    parameters_by_name = {parameter.name: parameter for parameter in plan.ir.params}

    assert parameters_by_name[gain_name].runtime_mutable
    assert parameters_by_name[switch_name].runtime_mutable
    assert parameters_by_name[non_decision_name].runtime_mutable
    assert [row[gain_name] for row in normalized] == [10.0, 5.0, 10.0, 10.0]
    assert [row[switch_name] for row in normalized] == [10.0, 10.0, 0.0, 10.0]
    assert [row[non_decision_name] for row in normalized] == [0.2, 0.2, 0.2, 0.5]

    first = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=64,
        seed=11,
    )
    replay = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=64,
        seed=11,
    )

    assert first.values.shape == (4, 1, 4, 64, 2)
    assert np.all(np.isfinite(first.values))
    assert set(np.unique(first.values[..., 0])) <= {0.0, 1.0}
    np.testing.assert_array_equal(first.values, replay.values)
    # Rows 0 and 3 differ only in non-decision time. Common random numbers
    # therefore make the modeled response-time shift exact lane by lane.
    np.testing.assert_allclose(
        first.values[3, ..., 1] - first.values[0, ..., 1],
        0.3,
        rtol=1e-5,
        atol=1e-6,
    )
    # The switch-CSI row must not collapse onto the base row; cue=1 trials
    # change both the controlled LCA count and the CSI contribution to RT.
    assert not np.array_equal(first.values[2], first.values[0])


def test_csi_three_parameter_pec_objective_compiles_and_scores(
    registered_csi_drift_rate,
    batched_backend,
):
    """The reduced fit surface must work through PEC, not just a bare plan.

    PEC injects fitting ControlMechanisms into its model.  The batched objective
    passes the candidate values as parameter rows, so compilation must recover
    the pristine CSI simulation semantics rather than attempting to execute
    those host-side fitting controls in the device graph.
    """

    composition, inputs, _, outputs = _recovery_surface_model()
    cue = _node(composition, "Cue Stimulus Interval")
    lca = _node(composition, "Task Activations [C1, C2]")
    ddm = _node(composition, "DDM")
    pec = _recovery_pec(
        composition,
        inputs,
        outputs,
        backend=batched_backend,
        include_historical_threshold_parameters=False,
    )
    optimization = pec.controller.function

    assert optimization.fit_param_names == [
        f"{lca.name}.gain",
        f"{cue.name}.slope",
        f"{ddm.name}.non_decision_time",
    ]
    report = pec.can_compile_batched(backend=batched_backend)
    assert report.can_execute
    objective = optimization._make_objective_func()
    score = objective(10.0, 10.0, 0.2)

    assert np.asarray(score).size == 1
    assert np.isfinite(float(np.asarray(score).reshape(-1)[0]))
    assert optimization._batched_plan is not None
    assert optimization._batched_plan.kernel_ir.executable


@pytest.mark.triton_gpu
def test_csi_deterministic_history_routes_through_pec_objective(
    registered_csi_drift_rate,
):
    composition, inputs, _, outputs = _recovery_surface_model()
    pec = _recovery_pec(
        composition,
        inputs,
        outputs,
        backend="triton",
        include_historical_threshold_parameters=False,
        deterministic_history_likelihood=True,
    )
    optimization = pec.controller.function
    objective = optimization._make_objective_func()

    first = objective(10.0, 10.0, 0.2)
    replay = objective(10.0, 10.0, 0.2)

    assert np.isfinite(first)
    assert replay == first
    assert optimization._batched_plan is not None


@pytest.mark.parametrize(
    "ignored_factory",
    (
        pytest.param(tuple, id="reusable-tuple"),
        pytest.param(iter, id="one-shot-iterator"),
    ),
)
def test_csi_intrinsic_control_cannot_be_ignored_as_a_parameter_lane(
    registered_csi_drift_rate,
    ignored_factory,
):
    """Only PEC's external fit controls may be erased during lowering."""

    composition, _, outputs = _model()
    intrinsic_control = _node(composition, "CSI Override")

    with pytest.raises(
        ValueError,
        match="may ignore only external parameter controls",
    ):
        lower_composition(
            composition,
            outputs=outputs,
            ignored_control_nodes=ignored_factory((intrinsic_control,)),
        )


def test_csi_historical_threshold_fit_names_bind_to_folded_ddm_lanes(
    registered_csi_drift_rate,
    batched_backend,
):
    """Retain the five-parameter surface used by the original CSI recovery.

    The Threshold Mechanism is deliberately absent from GraphIR because its
    integrating output is folded into the DDM boundary.  Its two public PEC
    parameter names must nevertheless remain live aliases for the folded
    starting threshold and per-step collapse parameters.
    """

    # The historical recovery builds a data-generating model and then a second
    # fit model in the same process.  Keep the former alive so PNL assigns the
    # latter its real ``-N`` rebuild suffix and the test exercises both aliases.
    reference_composition, _, _, _ = _recovery_surface_model()
    composition, inputs, _, outputs = _recovery_surface_model()
    plan = _compile_csi(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=600,
    )
    lowering = lower_composition(composition, outputs=outputs)
    cue = _node(composition, "Cue Stimulus Interval")
    lca = _node(composition, "Task Activations [C1, C2]")
    threshold_source = _node(composition, "Threshold Mechanism")
    ddm = _node(composition, "DDM")
    pec = _recovery_pec(
        composition,
        inputs,
        outputs,
        backend=batched_backend,
        include_historical_threshold_parameters=True,
    )
    pec_names = pec.controller.function.fit_param_names
    gain_name = f"{lca.name}.gain"
    switch_name = f"{cue.name}.slope"
    threshold_pec_name = f"{threshold_source.name}.intercept"
    collapse_pec_name = (
        f"{threshold_source.name}.offset-integrator_function"
    )
    non_decision_name = f"{ddm.name}.non_decision_time"

    assert pec_names == [
        gain_name,
        switch_name,
        threshold_pec_name,
        collapse_pec_name,
        non_decision_name,
    ]
    assert reference_composition is not composition
    assert threshold_source.name.startswith("Threshold Mechanism-")
    terminator = next(
        node for node in lowering.graph.nodes if node.component_type == "DDM"
    )
    parameters_by_name = {parameter.name: parameter for parameter in lowering.params}
    threshold_parameter = parameters_by_name[terminator.params["threshold"]]
    collapse_parameter = parameters_by_name[
        terminator.params["threshold_collapse"]
    ]
    assert threshold_parameter.runtime_mutable
    assert collapse_parameter.runtime_mutable
    assert threshold_pec_name in (
        threshold_parameter.name,
        *threshold_parameter.aliases,
    )
    assert "Threshold Mechanism.intercept" in (
        threshold_parameter.name,
        *threshold_parameter.aliases,
    )
    assert collapse_pec_name in (
        collapse_parameter.name,
        *collapse_parameter.aliases,
    )
    assert "Threshold Mechanism.offset-integrator_function" in (
        collapse_parameter.name,
        *collapse_parameter.aliases,
    )
    assert threshold_parameter.minimum == 0.0
    assert threshold_parameter.minimum_inclusive
    assert threshold_parameter.maximum is None
    assert collapse_parameter.minimum is None
    assert collapse_parameter.maximum == 0.0
    assert collapse_parameter.maximum_inclusive
    assert lowering.bindings.parameter_by_id(
        threshold_parameter.parameter_id
    ) is threshold_source.function.parameters.intercept
    assert lowering.bindings.parameter_by_id(
        collapse_parameter.parameter_id
    ) is threshold_source.integrator_function.parameters.offset

    value_rows = (
        (10.0, 10.0, 0.12, -0.001, 0.2),
        (5.0, 10.0, 0.12, -0.001, 0.2),
        (10.0, 0.0, 0.12, -0.001, 0.2),
        (10.0, 10.0, 0.20, -0.001, 0.2),
        (10.0, 10.0, 0.12, 0.0, 0.2),
        (10.0, 10.0, 0.12, -0.001, 0.5),
        (10.0, 10.0, 0.12, -0.001, 0.2),
    )
    parameter_sets = tuple(
        dict(zip(pec_names, values)) for values in value_rows
    )
    normalized = normalize_parameter_sets(parameter_sets, plan.ir)
    assert [row[threshold_parameter.name] for row in normalized] == [
        0.12,
        0.12,
        0.12,
        0.20,
        0.12,
        0.12,
        0.12,
    ]
    assert [row[collapse_parameter.name] for row in normalized] == [
        -0.001,
        -0.001,
        -0.001,
        -0.001,
        0.0,
        -0.001,
        -0.001,
    ]
    with pytest.raises(ValueError, match="must be >= 0.0"):
        normalize_parameter_sets(
            [{threshold_pec_name: -0.01}],
            plan.ir,
        )
    with pytest.raises(ValueError, match="must be <= 0.0"):
        normalize_parameter_sets(
            [{collapse_pec_name: 0.001}],
            plan.ir,
        )

    result = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=64,
        seed=31,
        common_random_numbers=True,
    )

    assert result.values.shape == (7, 1, 4, 64, 2)
    assert np.all(np.isfinite(result.values))
    assert not np.array_equal(result.values[3], result.values[0])
    assert not np.array_equal(result.values[4], result.values[0])
    np.testing.assert_array_equal(result.values[6], result.values[0])
    np.testing.assert_allclose(
        result.values[5, ..., 1] - result.values[0, ..., 1],
        0.3,
        rtol=1e-5,
        atol=1e-6,
    )
    report = pec.can_compile_batched(backend=batched_backend)
    assert report.can_execute
    objective = pec.controller.function._make_objective_func()
    score = objective(*value_rows[0])
    assert np.asarray(score).size == 1
    assert np.isfinite(float(np.asarray(score).reshape(-1)[0]))
    assert pec.controller.function._batched_plan is not None
    assert pec.controller.function._batched_plan.kernel_ir.executable


@pytest.mark.parametrize(
    "argument, replacement",
    (
        ("noise", 0.1),
        ("starting_value", 0.01),
        ("offset", 0.01),
    ),
)
def test_csi_folded_ddm_boundary_parameters_are_fixed(
    registered_csi_drift_rate,
    argument,
    replacement,
):
    composition, _, outputs = _model()
    plan = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=128,
    )
    terminator = next(
        node for node in plan.kernel_ir.graph.nodes if node.component_type == "DDM"
    )
    parameter_name = terminator.params[argument]
    parameter = next(
        item for item in plan.ir.params if item.name == parameter_name
    )

    assert not parameter.runtime_mutable
    assert parameter.runtime_constraint == (
        "dynamic scheduled terminator parameter is frozen in KernelIR"
    )
    with pytest.raises(ValueError, match="is fixed at"):
        normalize_parameter_sets(
            [{parameter_name: replacement}],
            plan.ir,
        )


def test_csi_fixed_nonzero_ddm_noise_is_admitted_and_frozen(
    registered_csi_drift_rate,
):
    composition, _, outputs = _model(ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=64,
    )
    terminator = next(
        node for node in plan.kernel_ir.graph.nodes if node.component_type == "DDM"
    )
    noise_name = terminator.params["noise"]
    noise = next(parameter for parameter in plan.ir.params if parameter.name == noise_name)

    assert noise.default == 0.1
    assert not noise.runtime_mutable
    assert normalize_parameter_sets([{noise_name: 0.1}], plan.ir)[0][noise_name] == 0.1
    with pytest.raises(ValueError, match="is fixed at 0.1"):
        normalize_parameter_sets([{noise_name: 0.2}], plan.ir)


@pytest.mark.parametrize(
    "noise",
    (
        pytest.param(-0.1, id="negative"),
        pytest.param(float("inf"), id="positive_infinity"),
        pytest.param(float("nan"), id="nan"),
    ),
)
def test_csi_invalid_ddm_noise_remains_fail_closed(
    registered_csi_drift_rate,
    noise,
):
    composition, _, outputs = _model(ddm_noise=noise)
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
        outputs=outputs,
        max_steps=64,
    )

    assert not report.can_execute
    assert report.model_diagnostics


def test_csi_stochastic_ddm_replays_and_uses_common_random_numbers(
    registered_csi_drift_rate,
    batched_backend,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    ddm = _node(composition, "DDM")
    parameter_sets = (
        {f"{ddm.name}.non_decision_time": 0.2},
        {f"{ddm.name}.non_decision_time": 0.4},
    )
    plan = _compile_csi(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=64,
    )

    first = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=16,
        seed=17,
        common_random_numbers=True,
        strict_truncation=True,
    )
    replay = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=16,
        seed=17,
        common_random_numbers=True,
        strict_truncation=True,
    )
    changed_seed = plan.run(
        inputs=inputs,
        parameter_sets=parameter_sets,
        num_estimates=16,
        seed=18,
        common_random_numbers=True,
        strict_truncation=True,
    )

    np.testing.assert_array_equal(first.values, replay.values)
    assert not np.array_equal(first.values, changed_seed.values)
    np.testing.assert_array_equal(
        first.values[0, ..., 0],
        first.values[1, ..., 0],
    )
    np.testing.assert_allclose(
        first.values[1, ..., 1] - first.values[0, ..., 1],
        0.2,
        rtol=1e-5,
        atol=1e-6,
    )
    assert np.unique(first.values[0, ..., 1]).size > 2
    assert all(fraction == 0.0 for fraction in first.metadata["truncation"].values())


def test_csi_one_trial_state_transport_matches_unsplit_sequence(
    registered_csi_drift_rate,
    batched_backend,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=64,
    )
    estimates = 8
    seed = 29
    full = plan.run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=estimates,
        seed=seed,
        strict_truncation=True,
    )

    state = None
    split_trials = []
    for trial in range(2):
        one_trial_inputs = {
            node: np.asarray(values)[trial : trial + 1]
            for node, values in inputs.items()
        }
        result = plan.run(
            inputs=one_trial_inputs,
            parameter_sets=[{}],
            num_estimates=estimates,
            seed=seed,
            strict_truncation=True,
            initial_states=state,
            return_final_states=True,
            rng_trial_offset=trial,
            rng_sequence_trials=2,
        )
        split_trials.append(result.values)
        state = result.metadata["final_states"]

    split = np.concatenate(split_trials, axis=2)
    np.testing.assert_array_equal(split, full.values)


def test_csi_conditioned_likelihood_runs_and_replays(
    registered_csi_drift_rate,
    batched_backend,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend=batched_backend,
        outputs=outputs,
        max_steps=64,
    )
    observed = plan.run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=41,
        strict_truncation=True,
    ).values[0, 0, :, 0]

    kwargs = dict(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=32,
        data=observed,
        categorical_dims=[True, False],
        bins=8,
        bin_range=[(0.0, 2.0)],
        smoothing_sigma=1.0,
        seed=43,
        strict_truncation=True,
    )
    first = plan.conditioned_log_likelihood(**kwargs)
    replay = plan.conditioned_log_likelihood(**kwargs)

    assert np.isfinite(first)
    assert replay == first


@pytest.mark.triton_gpu
def test_csi_deterministic_history_matches_lca_endpoint_recurrence(
    registered_csi_drift_rate,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=64,
    )
    observed = np.asarray(
        [[1.0, 0.62], [1.0, 0.73]],
        dtype=np.float32,
    )
    kwargs = dict(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=32,
        data=observed,
        categorical_dims=[0],
        bins=8,
        bin_range=[(0.0, 2.0)],
        smoothing_sigma=1.0,
        seed=43,
        strict_truncation=True,
        return_debug=True,
    )
    first, debug = plan.deterministic_history_log_likelihood(**kwargs)
    replay, replay_debug = plan.deterministic_history_log_likelihood(**kwargs)

    assert np.isfinite(first)
    assert replay == first
    np.testing.assert_array_equal(
        replay_debug["values"].detach().cpu(),
        debug["values"].detach().cpu(),
    )

    # Independent scalar recurrence for the persistent zero-noise LCA.  The
    # CSI model uses gain=10, leak=7, competition=3, self-excitation=0, and
    # 10 ms Euler steps.  Its observed state advances by the cue-controlled
    # onset steps and then by the participant's inferred DDM endpoint steps.
    pre = np.zeros(2, dtype=np.float64)
    activity = np.zeros(2, dtype=np.float64)
    initialized = False
    expected = []
    observed_steps = debug["observed_steps"][0]
    effective_steps = np.asarray([1, 3], dtype=int)
    task_values = np.asarray(inputs[_node(composition, "Task Input")])
    for trial, task_value in enumerate(task_values):
        total_steps = max(effective_steps[trial] - 1, 0) + observed_steps[trial]
        for _ in range(total_steps):
            if not initialized:
                activity[:] = 0.5
            recurrence = np.asarray(
                [
                    -3.0 * activity[1],
                    -3.0 * activity[0],
                ]
            )
            pre += (task_value + recurrence - 7.0 * pre) * 0.01
            activity = 1.0 / (1.0 + np.exp(-10.0 * pre))
            initialized = True
        expected.append([*pre, *activity, 1.0])

    np.testing.assert_allclose(
        debug["history_states"].detach().cpu().numpy()[0],
        expected,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.triton_gpu
@pytest.mark.parametrize("smoothing_sigma", (0.0, 0.5, 1.0))
def test_csi_deterministic_history_fused_histogram_matches_materialized_outcomes(
    registered_csi_drift_rate,
    smoothing_sigma,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=64,
    )
    observed = np.asarray(
        [[1.0, 0.62], [0.0, 0.73]],
        dtype=np.float32,
    )
    kwargs = dict(
        inputs=inputs,
        parameter_sets=[{}, {}],
        num_estimates=257,
        data=observed,
        categorical_dims=[0],
        bins=8,
        bin_range=[(0.0, 2.0)],
        smoothing_sigma=smoothing_sigma,
        pseudocount=0.25,
        categorical_cardinalities=[2],
        include_mask=[True, False],
        seed=47,
        strict_truncation=False,
    )

    fused = plan.deterministic_history_log_likelihood(**kwargs)
    replay = plan.deterministic_history_log_likelihood(**kwargs)
    materialized, _ = plan.deterministic_history_log_likelihood(
        **kwargs,
        return_debug=True,
    )

    np.testing.assert_array_equal(fused, replay)
    np.testing.assert_allclose(fused, materialized, rtol=0.0, atol=1e-6)


@pytest.mark.triton_gpu
def test_csi_deterministic_history_fused_histogram_reports_truncation(
    registered_csi_drift_rate,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=1,
    )

    with pytest.raises(BatchedTruncationError):
        plan.deterministic_history_log_likelihood(
            inputs=inputs,
            parameter_sets=[{}],
            num_estimates=32,
            data=np.asarray([[1.0, 0.31], [1.0, 0.33]]),
            categorical_dims=[0],
            bins=8,
            bin_range=[(0.0, 2.0)],
            seed=53,
            strict_truncation=True,
        )


@pytest.mark.parametrize(
    "model_options",
    (
        pytest.param({}, id="positive-csi"),
        pytest.param(
            {
                "iti": 10,
                "csi_repeat": 0,
                "csi_switch": 0,
                "cue_values": [[0.0], [1.0]],
            },
            id="zero-csi-overlapped-onset",
        ),
        pytest.param(
            {
                "iti": 2,
                "csi_repeat": 3,
                "csi_switch": 4,
                "cue_values": [[0.0], [1.0]],
            },
            id="iti-plus-positive-csi",
        ),
    ),
)
@pytest.mark.triton_gpu
def test_csi_deterministic_history_matches_coupled_endpoint_execution(
    registered_csi_drift_rate,
    model_options,
):
    composition, inputs, outputs = _model(ddm_noise=0.0, **model_options)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=64,
    )
    coupled_result = plan.run(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=1,
        seed=17,
        strict_truncation=True,
        return_final_states=True,
    )
    coupled = coupled_result.values[0, 0, :, 0]

    _, debug = plan.deterministic_history_log_likelihood(
        inputs=inputs,
        parameter_sets=[{}],
        num_estimates=8,
        data=coupled,
        categorical_dims=[0],
        bins=8,
        bin_range=[(0.0, 2.0)],
        seed=19,
        strict_truncation=True,
        return_debug=True,
    )

    specialized = debug["values"].detach().cpu().numpy()[0]
    np.testing.assert_allclose(
        specialized,
        np.broadcast_to(coupled[:, None, :], specialized.shape),
        rtol=1e-6,
        atol=1e-6,
    )
    coupled_final_state = np.asarray(
        coupled_result.metadata["final_states"]
    )[0, 0, 0]
    np.testing.assert_allclose(
        debug["history_states"].detach().cpu().numpy()[0, -1],
        coupled_final_state,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.triton_gpu
def test_csi_deterministic_history_rejects_stochastic_lca(
    registered_csi_drift_rate,
):
    composition, inputs, outputs = _model(lca_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=64,
    )

    with pytest.raises(ValueError, match="deterministic LCA noise=0"):
        plan.deterministic_history_log_likelihood(
            inputs=inputs,
            parameter_sets=[{}],
            num_estimates=8,
            data=np.asarray([[1.0, 0.5], [1.0, 0.5]]),
            categorical_dims=[0],
            bins=8,
            bin_range=[(0.0, 2.0)],
        )


@pytest.mark.triton_gpu
def test_csi_conditioned_likelihood_preserves_strict_truncation_check(
    registered_csi_drift_rate,
):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    plan = _compile_csi(
        composition,
        backend="triton",
        outputs=outputs,
        max_steps=1,
    )

    with pytest.raises(BatchedTruncationError):
        plan.conditioned_log_likelihood(
            inputs=inputs,
            parameter_sets=[{}],
            num_estimates=32,
            data=np.asarray([[1.0, 0.5], [1.0, 0.5]]),
            categorical_dims=[True, False],
            bins=8,
            bin_range=[(0.0, 2.0)],
            seed=43,
            strict_truncation=True,
        )


def test_csi_conditioned_llvm_pec_runs_and_replays(registered_csi_drift_rate):
    composition, inputs, outputs = _model(ddm_rate=0.0, ddm_noise=0.1)
    ddm = _node(composition, "DDM")
    threshold = _node(composition, "Threshold Mechanism")
    inputs[threshold] = np.zeros((2, 1), dtype=float)
    observed = pd.DataFrame(
        {
            "decision": pd.Categorical([1.0, -1.0]),
            "response_time": [0.50, 0.55],
        }
    )
    pec = pnl.ParameterEstimationComposition(
        nodes=composition,
        parameters={
            ("non_decision_time", ddm): np.linspace(0.1, 0.3, 3),
        },
        outcome_variables=list(outputs),
        data=observed,
        optimization_function=pnl.PECOptimizationFunction(
            method="differential_evolution",
            max_iterations=1,
            conditioned_likelihood=True,
            batched_bins=8,
            batched_bin_range=[(0.0, 2.0)],
            batched_smoothing_sigma=1.0,
            batched_seed=47,
        ),
        num_estimates=8,
        initial_seed=47,
    )
    pec.controller.parameters.comp_execution_mode.set("LLVM")

    first = pec.log_likelihood(0.2, inputs=inputs)
    replay = pec.log_likelihood(0.2, inputs=inputs)

    assert np.isfinite(first)
    assert replay == first


def test_csi_stochastic_ddm_draws_are_cap_and_onset_independent(
    registered_csi_drift_rate,
    batched_backend,
):
    base, base_inputs, base_outputs = _model(
        ddm_rate=0.0,
        ddm_noise=0.1,
        iti=0,
    )
    delayed, delayed_inputs, delayed_outputs = _model(
        ddm_rate=0.0,
        ddm_noise=0.1,
        iti=5,
    )
    plans = (
        _compile_csi(
            base,
            backend=batched_backend,
            outputs=base_outputs,
            max_steps=64,
        ),
        _compile_csi(
            base,
            backend=batched_backend,
            outputs=base_outputs,
            max_steps=128,
        ),
        _compile_csi(
            delayed,
            backend=batched_backend,
            outputs=delayed_outputs,
            max_steps=64,
        ),
    )
    runs = (
        plans[0].run(
            inputs=base_inputs,
            parameter_sets=[{}],
            num_estimates=16,
            seed=23,
            strict_truncation=True,
        ),
        plans[1].run(
            inputs=base_inputs,
            parameter_sets=[{}],
            num_estimates=16,
            seed=23,
            strict_truncation=True,
        ),
        plans[2].run(
            inputs=delayed_inputs,
            parameter_sets=[{}],
            num_estimates=16,
            seed=23,
            strict_truncation=True,
        ),
    )

    np.testing.assert_array_equal(runs[0].values, runs[1].values)
    np.testing.assert_array_equal(runs[0].values, runs[2].values)


@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_deterministic_interpreter_matches_fresh_python(
    registered_csi_drift_rate,
):
    python_composition, python_inputs, python_outputs = _model()
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_composition, python_outputs)
    np.testing.assert_allclose(expected, _EXPECTED, rtol=0.0, atol=1e-12)

    actual = _run_compiled_csi("triton_cpu")

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual, _EXPECTED, rtol=1e-5, atol=1e-6)


@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_count_zero_trial_entry_folded_threshold_matches_fresh_python(
    registered_csi_drift_rate,
):
    """A raw zero LCA count makes WhenFinished true before trial execution."""

    options = {
        "csi_switch": 0,
        "csi_repeat": 0,
        "iti": 0,
        "cue_values": [[0.0], [0.0], [0.0]],
    }
    python_composition, python_inputs, python_outputs = _model(**options)
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_composition, python_outputs)
    np.testing.assert_allclose(
        expected,
        [[1.0, 0.53], [1.0, 0.55], [1.0, 0.55]],
        rtol=0.0,
        atol=1e-12,
    )

    actual = _run_compiled_csi("triton_cpu", **options)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


_AFFINE_TIMING_CASES = (
    pytest.param(
        {
            "iti": 10,
            "csi_repeat": 0,
            "csi_switch": 0,
            "cue_values": [[0.0], [1.0]],
        },
        True,
        id="zero-csi-ddm-starts-one-pass-before-task",
    ),
    pytest.param(
        {
            "iti": 2,
            "csi_repeat": 3,
            "csi_switch": 4,
            "cue_values": [[0.0], [1.0], [0.0], [1.0]],
        },
        False,
        id="combined-iti-repeat-switch-mixed-cues",
    ),
)


@pytest.mark.parametrize("model_options,pin_zero_csi_timing", _AFFINE_TIMING_CASES)
@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_affine_timing_interpreter_matches_fresh_python(
    registered_csi_drift_rate,
    model_options,
    pin_zero_csi_timing,
):
    python_composition, python_inputs, python_outputs = _model(**model_options)
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_composition, python_outputs)

    actual = _run_compiled_csi("triton_cpu", **model_options)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    if pin_zero_csi_timing:
        # This is the model's historical scheduler behavior, not an intended
        # correction: a threshold of ITI + CSI == 10 becomes finished on the
        # tenth LCA execution (global pass 9), while Task Input fires at
        # AtPass(10). The DDM therefore executes once before task onset.
        task = _node(python_composition, "Task Input")
        stepper = _node(python_composition, "Task Activations [C1, C2]")
        terminator = _node(python_composition, "DDM")
        execution = python_composition.scheduler.execution_list[
            python_composition.default_execution_id
        ]
        first_ddm = next(
            index
            for index, execution_set in enumerate(execution)
            if terminator in execution_set
        )
        first_task = next(
            index
            for index, execution_set in enumerate(execution)
            if task in execution_set
        )
        assert first_ddm < first_task
        assert sum(
            stepper in execution_set
            for execution_set in execution[:first_ddm]
        ) == 10


@pytest.mark.parametrize(
    "model_options, expected",
    (
        (
            {"correct_values": [[0.0]], "one_trial": True},
            np.asarray([[0.0, 0.92]]),
        ),
        (
            {"ddm_rate": 40.0},
            np.asarray([[1.0, 0.33], [1.0, 0.34]]),
        ),
        (
            {"ddm_rate": 500.0},
            np.asarray([[1.0, 0.32], [1.0, 0.34]]),
        ),
    ),
    ids=(
        "boundary-crosses-zero",
        "persistent-threshold-control-value",
        "one-step-threshold-cleanup",
    ),
)
@pytest.mark.triton
@pytest.mark.triton_interpreter
def test_csi_interpreter_matches_ddm_boundary_transition_oracle(
    registered_csi_drift_rate,
    model_options,
    expected,
):
    python_composition, python_inputs, python_outputs = _model(**model_options)
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    python_result = _selected_python_results(
        python_composition,
        python_outputs,
    )
    np.testing.assert_allclose(python_result, expected, rtol=0.0, atol=1e-12)

    actual = _run_compiled_csi("triton_cpu", **model_options)

    np.testing.assert_allclose(actual, python_result, rtol=1e-5, atol=1e-6)


@pytest.mark.triton
@pytest.mark.triton_gpu
def test_csi_deterministic_gpu_matches_oracle(registered_csi_drift_rate):
    actual = _run_compiled_csi("triton")

    np.testing.assert_allclose(actual, _EXPECTED, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        _run_compiled_csi(
            "triton",
            correct_values=[[0.0]],
            one_trial=True,
        ),
        [[0.0, 0.92]],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _run_compiled_csi("triton", ddm_rate=40.0),
        [[1.0, 0.33], [1.0, 0.34]],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _run_compiled_csi("triton", ddm_rate=500.0),
        [[1.0, 0.32], [1.0, 0.34]],
        rtol=1e-5,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        _run_compiled_csi(
            "triton",
            csi_switch=0,
            csi_repeat=0,
            iti=0,
            cue_values=[[0.0], [0.0], [0.0]],
        ),
        [[1.0, 0.53], [1.0, 0.55], [1.0, 0.55]],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("model_options,_", _AFFINE_TIMING_CASES)
@pytest.mark.triton
@pytest.mark.triton_gpu
def test_csi_affine_timing_gpu_matches_fresh_python(
    registered_csi_drift_rate,
    model_options,
    _,
):
    python_composition, python_inputs, python_outputs = _model(**model_options)
    python_composition.run(
        inputs=python_inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )
    expected = _selected_python_results(python_composition, python_outputs)

    actual = _run_compiled_csi("triton", **model_options)

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
