"""Exact semantic boundary for the batched width-two LCA subset.

The cases are informed by PsyNeuLink's LLVM-enabled LCA, Logistic, and
LeakyCompetingIntegrator tests.  Runtime cases always build independent Python
and batched models; capability cases pin configurations that must fail closed
until their behavior is represented explicitly by the batched IR.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import itertools
import sys

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
    BatchedDiagnosticCode,
)
from psyneulink.core.batched import specs as batched_specs
from psyneulink.core.batched.backend.triton.graph_emit import (
    triton_graph_kernel_source,
)
from psyneulink.core.batched.graph import lower_composition
from psyneulink.core.batched.ir import FP32_EXACT_INTEGER_LIMIT, BatchedCompositionIR
from psyneulink.core.batched.kernel_ir import lower_to_kernel_ir
from psyneulink.core.batched.prep import (
    lca_max_steps,
    normalize_parameter_sets,
    prepare_inputs,
)

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition]


LCA_RECURRENCE_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCA::test_LCAMechanism_length_2"
)
LCA_THRESHOLD_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCA::test_LCAMechanism_threshold"
)
LOGISTIC_LLVM_PROVENANCE = "tests/functions/test_transfer.py::test_execute[LOGISTIC]"
LCI_LLVM_PROVENANCE = (
    "tests/functions/test_integrator.py::test_execute[LeakyCompetingIntegrator]"
)
LCA_RESET_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCAReset::test_reset_run"
)
LCA_MATRIX_PROVENANCE = (
    "tests/mechanisms/test_lca.py::TestLCA::test_LCAMechanism_matrix"
)
LCA_CLIP_PROVENANCE = "tests/mechanisms/test_lca.py::TestClip::test_clip_array"


def _lca_case(
    name,
    *,
    inputs,
    threshold=1,
    function_kwargs=None,
    leak=0.0,
    competition=0.0,
    self_excitation=0.0,
    time_step_size=0.4,
    max_executions_before_finished=None,
    provenance,
):
    input_values = np.asarray(inputs, dtype=float)
    assert input_values.ndim == 2 and input_values.shape[1] == 2
    build_number = itertools.count()

    def build():
        lca = pnl.LCAMechanism(
            input_shapes=2,
            function=pnl.Logistic(**(function_kwargs or {})),
            leak=leak,
            competition=competition,
            self_excitation=self_excitation,
            noise=0.0,
            time_step_size=time_step_size,
            termination_measure=pnl.TimeScale.TRIAL,
            termination_threshold=threshold,
            reset_stateful_function_when=pnl.Never(),
            name=f"{name}-{next(build_number)}",
        )
        if max_executions_before_finished is not None:
            lca.max_executions_before_finished = max_executions_before_finished
        return SemanticModel(
            composition=pnl.Composition(pathways=lca),
            inputs={lca: input_values.copy()},
            outputs=(lca.output_port,),
        )

    return SemanticCase(
        name=name,
        build=build,
        provenance=provenance,
        atol=1e-6,
        rtol=1e-5,
        max_steps=16,
    )


def _composition_ir(lowering):
    graph = lowering.graph
    assert graph is not None
    return BatchedCompositionIR(
        model_kind=lowering.model_kind,
        node_names=tuple(node.name for node in graph.nodes),
        params=lowering.params,
        output_names=tuple(output.name for output in graph.outputs),
        graph=graph,
    )


FIRST_STEP_CASE = _lca_case(
    "lca_first_step_uses_logistic_zero_activity",
    inputs=[[0.0, 0.0]],
    leak=0.0,
    competition=0.3,
    self_excitation=1.2,
    time_step_size=0.2,
    provenance=LCA_RECURRENCE_PROVENANCE,
)


PERSISTENCE_CASE = _lca_case(
    "lca_asymmetric_recurrence_persists_across_trials",
    inputs=[
        [1.0, -0.25],
        [-0.6, 0.85],
        [0.35, -1.1],
    ],
    leak=0.3,
    # Equal self-excitation and competition cancel the initial [0.5, 0.5]
    # activity on trial one.  Later trials therefore isolate persistence of the
    # now-asymmetric recurrent activity rather than the separate initialization
    # contract pinned by FIRST_STEP_CASE.
    competition=0.4,
    self_excitation=0.4,
    time_step_size=0.2,
    provenance=LCA_RECURRENCE_PROVENANCE,
)


_LOGISTIC_INPUTS = [[1.25, -0.75], [-0.4, 1.6]]
LOGISTIC_CASES = (
    _lca_case(
        "lca_logistic_gain",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={"gain": 1.7},
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
    _lca_case(
        "lca_logistic_bias",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={"bias": 0.35},
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
    _lca_case(
        "lca_logistic_x_0",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={"x_0": -0.4},
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
    _lca_case(
        "lca_logistic_scale",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={"scale": 1.6},
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
    _lca_case(
        "lca_logistic_offset",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={"offset": -0.2},
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
    _lca_case(
        "lca_logistic_all_parameters",
        inputs=_LOGISTIC_INPUTS,
        function_kwargs={
            "gain": 1.3,
            "bias": 0.4,
            "x_0": -0.35,
            "scale": 1.7,
            "offset": -0.25,
        },
        provenance=LOGISTIC_LLVM_PROVENANCE,
    ),
)


THRESHOLD_COUNTS = ((0, 1), (1, 1), (1.00000001, 2), (1.1, 2), (3, 3))
THRESHOLD_CASES = tuple(
    (
        _lca_case(
            f"lca_trial_threshold_{threshold}",
            inputs=[[1.25, -0.75]],
            threshold=threshold,
            provenance=LCA_THRESHOLD_PROVENANCE,
        ),
        expected_count,
    )
    for threshold, expected_count in THRESHOLD_COUNTS
)


CAPPED_EXECUTION_CASE = _lca_case(
    "lca_maximum_execution_cap",
    inputs=[[1.25, -0.75]],
    threshold=3,
    max_executions_before_finished=1,
    provenance=LCA_THRESHOLD_PROVENANCE,
)


def _controlled_lca_caps_case():
    build_number = itertools.count()

    def build():
        index = next(build_number)
        composition = pnl.Composition()
        inputs = {}
        outputs = []
        for slot, execution_cap in enumerate((1, 3)):
            task = pnl.TransferMechanism(
                input_shapes=2,
                name=f"controlled task {slot}-{index}",
            )
            cue = pnl.TransferMechanism(
                input_shapes=1,
                name=f"controlled cue {slot}-{index}",
            )
            lca = pnl.LCAMechanism(
                input_shapes=2,
                function=pnl.Logistic(),
                leak=0.0,
                competition=0.0,
                self_excitation=0.0,
                noise=0.0,
                time_step_size=0.4,
                termination_measure=pnl.TimeScale.TRIAL,
                termination_threshold=8,
                reset_stateful_function_when=pnl.Never(),
                name=f"controlled lca {slot}-{index}",
            )
            lca.max_executions_before_finished = execution_cap
            controller = pnl.ControlMechanism(
                monitor_for_control=cue,
                control_signals=[(pnl.TERMINATION_THRESHOLD, lca)],
                modulation=pnl.OVERRIDE,
                name=f"termination controller {slot}-{index}",
            )
            composition.add_nodes([task, cue, lca, controller])
            composition.add_projection(sender=task, receiver=lca)
            inputs[task] = np.asarray([[1.25, -0.75]])
            inputs[cue] = np.asarray([[3.0]])
            outputs.append(lca.output_port)
        return SemanticModel(
            composition=composition,
            inputs=inputs,
            outputs=tuple(outputs),
        )

    return SemanticCase(
        name="controlled_lca_distinct_execution_caps",
        build=build,
        provenance=(
            "tests/composition/test_control.py OVERRIDE control; each LCA keeps "
            "its own max_executions_before_finished"
        ),
        atol=1e-6,
        rtol=1e-5,
        max_steps=16,
    )


CONTROLLED_CAPS_CASE = _controlled_lca_caps_case()


PARITY_CASES = (
    FIRST_STEP_CASE,
    PERSISTENCE_CASE,
    *LOGISTIC_CASES,
    *(case for case, _ in THRESHOLD_CASES),
    CAPPED_EXECUTION_CASE,
    CONTROLLED_CAPS_CASE,
)


def test_python_first_step_recurrence_starts_from_logistic_zero_activity():
    model = FIRST_STEP_CASE.build()
    result = model.composition.run(
        inputs=model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )

    # Logistic(0) initializes both activities to 0.5.  With auto=1.2,
    # competition=0.3, and dt=0.2, the first pre-function value is 0.09.
    expected = 1.0 / (1.0 + np.exp(-0.09))
    np.testing.assert_allclose(result, [[expected, expected]])


@pytest.mark.parametrize(
    "case, expected_count",
    THRESHOLD_CASES,
    ids=lambda value: value.name if isinstance(value, SemanticCase) else None,
)
def test_python_trial_threshold_execution_count(case, expected_count):
    model = case.build()
    lca = next(
        node for node in model.composition.nodes if isinstance(node, pnl.LCAMechanism)
    )
    model.composition.run(
        inputs=model.inputs,
        execution_mode=pnl.ExecutionMode.Python,
    )

    # A mechanism executes at least once; positive fractional trial thresholds
    # take effect on the first integer execution count at or above the value.
    assert lca.num_executions_before_finished == expected_count


def test_lca_activation_state_initializer_binds_frozen_logistic_spec():
    model = FIRST_STEP_CASE.build()
    lca = next(
        node for node in model.composition.nodes if isinstance(node, pnl.LCAMechanism)
    )
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions
    assert lowering.graph is not None

    graph = lowering.graph
    activation_state = next(
        state for state in graph.states if state.name == f"{lca.name}.act"
    )
    initializer = activation_state.function_initializer
    assert initializer is not None
    logistic_spec = batched_specs.function_spec_for(lca.function)
    assert logistic_spec is not None
    assert initializer.spec_key == logistic_spec.key
    assert initializer.input_value == (0.0, 0.0)

    function_arguments = ("gain", "bias", "x_0", "scale", "offset")
    node_spec = graph.node(lca.name)
    assert initializer.params == {
        argument: node_spec.params[argument] for argument in function_arguments
    }

    kernel = lower_to_kernel_ir(_composition_ir(lowering))
    assert kernel.op_specs.lookup_spec(initializer.spec_key) is logistic_spec


def test_lca_time_step_binding_declares_positive_runtime_domain():
    model = FIRST_STEP_CASE.build()
    lca = next(
        node for node in model.composition.nodes if isinstance(node, pnl.LCAMechanism)
    )
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None
    parameter_name = lowering.graph.node(lca.name).params["time_step_size"]
    parameter = next(spec for spec in lowering.params if spec.name == parameter_name)

    assert parameter.minimum == 0.0
    assert not parameter.minimum_inclusive


@pytest.mark.parametrize(
    "case, expected_count",
    THRESHOLD_CASES,
    ids=lambda value: value.name if isinstance(value, SemanticCase) else None,
)
def test_static_trial_threshold_sizes_lca_step_cap(case, expected_count):
    model = case.build()
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions

    assert lca_max_steps(_composition_ir(lowering), inputs={}) == expected_count


def test_maximum_execution_count_limits_static_step_cap():
    model = CAPPED_EXECUTION_CASE.build()
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert not lowering.rejected_nodes
    assert not lowering.rejected_conditions

    assert lca_max_steps(_composition_ir(lowering), inputs={}) == 1


def test_default_maximum_execution_count_is_preserved_as_an_integer():
    model = FIRST_STEP_CASE.build()
    lca = next(
        node for node in model.composition.nodes if isinstance(node, pnl.LCAMechanism)
    )
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None

    assert (
        lowering.graph.node(lca.name).attrs["max_executions_before_finished"]
        == sys.maxsize
    )


def test_static_lca_step_literal_is_floating_point_for_compiled_triton():
    model = FIRST_STEP_CASE.build()
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None

    source = triton_graph_kernel_source(lower_to_kernel_ir(_composition_ir(lowering)))

    # The interpreter accepts tl.ceil(int), but compiled Triton requires a
    # floating-point operand.  This source assertion complements the GPU gate.
    assert "tl.ceil(1.0)" in source
    assert "tl.ceil(1)" not in source


@pytest.mark.parametrize(
    "invalid_value",
    (1.00000001, -1, np.nan, FP32_EXACT_INTEGER_LIMIT + 1),
    ids=("fractional", "negative", "nonfinite", "outside-exact-fp32-range"),
)
def test_invalid_controlled_step_count_rejects_before_fp32_rounding(invalid_value):
    model = CONTROLLED_CAPS_CASE.build()
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None
    cue = next(
        node
        for node in model.composition.nodes
        if node.name.startswith("controlled cue 0-")
    )
    invalid_inputs = dict(model.inputs)
    invalid_inputs[cue] = np.asarray([[invalid_value]])

    with pytest.raises(ValueError, match="finite, nonnegative integer values"):
        prepare_inputs(
            _composition_ir(lowering),
            invalid_inputs,
            component_bindings=lowering.bindings,
        )


@pytest.mark.parametrize(
    "argument, value",
    (("slope", 2.0), ("intercept", 1.0), ("scale", 2.0), ("offset", 1.0)),
)
def test_absorbed_termination_source_parameters_are_default_only(argument, value):
    model = CONTROLLED_CAPS_CASE.build()
    lowering = lower_composition(model.composition, outputs=model.outputs)
    assert lowering.graph is not None
    cue = next(
        node
        for node in lowering.graph.nodes
        if node.name.startswith("controlled cue 0-")
    )
    parameter_name = cue.params[argument]
    parameter = next(spec for spec in lowering.params if spec.name == parameter_name)

    assert not parameter.runtime_mutable
    assert parameter.runtime_constraint == (
        "absorbed identity termination-threshold source for "
        f"{next(node.name for node in lowering.graph.nodes if node.name.startswith('controlled lca 0-'))}"
    )
    normalized_default = normalize_parameter_sets(
        [{parameter_name: parameter.default}],
        _composition_ir(lowering),
    )
    assert normalized_default[0][parameter_name] == parameter.default
    with pytest.raises(ValueError, match="is fixed at"):
        normalize_parameter_sets([{parameter_name: value}], _composition_ir(lowering))
    with pytest.raises(ValueError, match="is fixed at"):
        normalize_parameter_sets(
            {parameter_name: np.asarray([parameter.default, value])},
            _composition_ir(lowering),
        )

    positional_row = np.asarray(
        [spec.default for spec in lowering.params],
        dtype=float,
    )
    parameter_index = next(
        index
        for index, spec in enumerate(lowering.params)
        if spec.name == parameter_name
    )
    positional_row[parameter_index] = value
    with pytest.raises(ValueError, match="is fixed at"):
        normalize_parameter_sets([positional_row], _composition_ir(lowering))


@pytest.mark.parametrize("case", PARITY_CASES, ids=lambda case: case.name)
def test_exact_lca_subset_matches_python(case, batched_backend):
    assert_matches_python(case, backend=batched_backend)


def test_lca_activation_initializer_uses_each_parameter_lane(batched_backend):
    biases = (-1.0, 1.0)
    batched_case = _lca_case(
        "lca_lane_local_initializer",
        inputs=[[0.0, 0.0]],
        function_kwargs={"bias": 0.0},
        leak=0.0,
        competition=0.0,
        self_excitation=2.0,
        time_step_size=0.5,
        provenance=LOGISTIC_LLVM_PROVENANCE,
    )
    batched_model = batched_case.build()
    lca = next(
        node
        for node in batched_model.composition.nodes
        if isinstance(node, pnl.LCAMechanism)
    )
    plan = BatchedCompositionCompiler.compile(
        batched_model.composition,
        backend=batched_backend,
        outputs=batched_model.outputs,
        max_steps=batched_case.max_steps,
    )
    bias_parameter = plan.ir.graph.node(lca.name).params["bias"]
    result = plan.run(
        inputs=batched_model.inputs,
        parameter_sets=[{bias_parameter: bias} for bias in biases],
        num_estimates=1,
        seed=0,
    )

    expected = []
    for index, bias in enumerate(biases):
        python_case = _lca_case(
            f"lca_lane_local_initializer_python_{index}",
            inputs=[[0.0, 0.0]],
            function_kwargs={"bias": bias},
            leak=0.0,
            competition=0.0,
            self_excitation=2.0,
            time_step_size=0.5,
            provenance=LOGISTIC_LLVM_PROVENANCE,
        )
        python_model = python_case.build()
        python_model.composition.run(
            inputs=python_model.inputs,
            execution_mode=pnl.ExecutionMode.Python,
        )
        expected.append(
            np.concatenate([
                np.asarray(value, dtype=float).reshape(-1)
                for value in python_model.composition.results[-1]
            ])
        )

    got = np.asarray(result.values)[:, 0, 0, 0, :]
    expected = np.asarray(expected)
    assert not np.allclose(got[0], got[1])
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)


@dataclass(frozen=True)
class _RejectionCase:
    name: str
    build: Callable[[], tuple[pnl.Composition, object]]
    reason: str
    detail: str
    provenance: str
    code: str | None = None


def _base_lca(name, **overrides):
    kwargs = {
        "input_shapes": 2,
        "function": pnl.Logistic(),
        "leak": 0.4,
        "competition": 0.3,
        "self_excitation": 0.2,
        "noise": 0.0,
        "time_step_size": 0.1,
        "termination_measure": pnl.TimeScale.TRIAL,
        "termination_threshold": 2,
        "reset_stateful_function_when": pnl.Never(),
        "name": name,
    }
    if "matrix" in overrides:
        # A custom matrix is itself the feature under test; do not also specify
        # the mutually exclusive auto/hetero shorthand.
        kwargs.pop("competition")
        kwargs.pop("self_excitation")
    kwargs.update(overrides)
    return pnl.LCAMechanism(**kwargs)


def _lca_rejection_builder(name, overrides: Callable[[], Mapping]):
    def build():
        lca = _base_lca(name, **overrides())
        return pnl.Composition(pathways=lca), lca

    return build


def _constant_noise():
    return 0.125


def _lci_initializer_model():
    # LCAMechanism construction synchronizes a supplied integrator's initializer
    # from its own initial_value.  Set the live LCI parameter after composition
    # construction to exercise a genuinely deferred initializer configuration.
    lca = _base_lca("initializer lca")
    composition = pnl.Composition(pathways=lca)
    lca.integrator_function.parameters.initializer.set([[0.1, -0.2]], None)
    return composition, lca


def _nonfinite_logistic_model():
    lca = _base_lca("nonfinite Logistic lca")
    composition = pnl.Composition(pathways=lca)
    lca.function.parameters.bias.set(float("nan"), None)
    return composition, lca


def _controlled_gain_model():
    source = pnl.TransferMechanism(input_shapes=1, name="gain source")
    lca = _base_lca("controlled gain lca")
    controller = pnl.ControlMechanism(
        monitor_for_control=source,
        control_signals=[("gain", lca)],
        modulation=pnl.OVERRIDE,
        name="gain controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([source, lca, controller])
    return composition, controller


def _controlled_lca_schedule_model(role, condition):
    cue = pnl.TransferMechanism(input_shapes=1, name="scheduled source")
    lca = _base_lca("scheduled target")
    controller = pnl.ControlMechanism(
        monitor_for_control=cue,
        control_signals=[(pnl.TERMINATION_THRESHOLD, lca)],
        modulation=pnl.OVERRIDE,
        name="scheduled controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([cue, lca, controller])
    component = {"source": cue, "controller": controller, "target": lca}[role]
    composition.scheduler.add_condition(component, condition)
    return composition, controller


def _unsupported_termination_source_model():
    cue = pnl.TransferMechanism(
        input_shapes=1,
        function=pnl.Exponential(),
        name="unsupported termination source",
    )
    lca = _base_lca("unsupported source target")
    controller = pnl.ControlMechanism(
        monitor_for_control=cue,
        control_signals=[(pnl.TERMINATION_THRESHOLD, lca)],
        modulation=pnl.OVERRIDE,
        name="unsupported source controller",
    )
    composition = pnl.Composition()
    composition.add_nodes([cue, lca, controller])
    return composition, cue


def _invalid_maximum_execution_model(value):
    lca = _base_lca(f"invalid maximum executions {value!r}")
    composition = pnl.Composition(pathways=lca)
    lca.max_executions_before_finished = value
    return composition, lca


def _recurrent_projection_model(*, function_parameter=None, projection_parameter=None):
    lca = _base_lca("recurrent projection lca")
    composition = pnl.Composition(pathways=lca)
    projection = lca.recurrent_projection
    if function_parameter is not None:
        name, value = function_parameter
        getattr(projection.function.parameters, name).set(value, None)
    if projection_parameter is not None:
        name, value = projection_parameter
        getattr(projection.parameters, name).set(value, None)
    return composition, lca


REJECTION_CASES = (
    _RejectionCase(
        "complex_logistic_parameter",
        _lca_rejection_builder(
            "complex Logistic lca",
            lambda: {"function": pnl.Logistic(gain=1.0 + 1.0j)},
        ),
        "unsupported complex Logistic parameter for batched v2",
        "gain dtype=complex128",
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "float32_overflow_logistic_parameter",
        _lca_rejection_builder(
            "out-of-range Logistic lca",
            lambda: {"function": pnl.Logistic(gain=1e40)},
        ),
        "unsupported out-of-range Logistic parameter for batched v2",
        "gain is not representable as float32",
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "float32_overflow_lca_parameter",
        _lca_rejection_builder(
            "out-of-range leak lca",
            lambda: {"leak": 1e40},
        ),
        "unsupported out-of-range LCA parameter for batched v2",
        "leak is not representable as float32",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "near_zero_initial_value",
        _lca_rejection_builder(
            "near-zero initial value lca",
            lambda: {"initial_value": [1e-9, -1e-9]},
        ),
        "unsupported LCA initial_value for batched v2",
        "requires zero",
        LCA_RESET_PROVENANCE,
    ),
    _RejectionCase(
        "near_zero_noise",
        _lca_rejection_builder("near-zero noise lca", lambda: {"noise": 1e-9}),
        "unsupported LCA noise for batched v2",
        "requires numeric zero",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "nearly_scalar_leak",
        _lca_rejection_builder(
            "nearly scalar leak lca",
            lambda: {"leak": [0.4, 0.4000001]},
        ),
        "unsupported non-scalar LCA parameter for batched v2",
        "leak",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "near_canonical_recurrent_matrix",
        _lca_rejection_builder(
            "near-canonical matrix lca",
            lambda: {"matrix": [[0.0, -0.9999999], [-1.0, 0.0]]},
        ),
        "unsupported LCA recurrent matrix for batched v2",
        "requires canonical self-excitation/competition matrix",
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "near_broadcast_vector_logistic_parameter",
        _lca_rejection_builder(
            "vector Logistic lca",
            lambda: {"function": pnl.Logistic(bias=[0.1, 0.1000001])},
        ),
        "unsupported non-scalar Logistic parameter for batched v2",
        "bias shape=(2,)",
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "nonfinite_logistic_parameter",
        _nonfinite_logistic_model,
        "unsupported non-finite Logistic parameter for batched v2",
        "bias contains non-finite values",
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "nonzero_numeric_noise",
        _lca_rejection_builder("numeric noise lca", lambda: {"noise": 0.125}),
        "unsupported LCA noise for batched v2",
        "requires numeric zero",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "callable_noise",
        _lca_rejection_builder("callable noise lca", lambda: {"noise": _constant_noise}),
        "unsupported LCA noise for batched v2",
        "requires numeric zero",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "lci_offset",
        _lca_rejection_builder(
            "offset lca",
            lambda: {"integrator_function": pnl.LeakyCompetingIntegrator(offset=0.2)},
        ),
        "unsupported LCA integrator offset for batched v2",
        "requires zero",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "lci_initializer",
        _lci_initializer_model,
        "unsupported LCA integrator initializer for batched v2",
        "requires zero",
        LCI_LLVM_PROVENANCE,
    ),
    _RejectionCase(
        "mechanism_initial_value",
        _lca_rejection_builder(
            "initial value lca",
            lambda: {"initial_value": [0.1, -0.2]},
        ),
        "unsupported LCA initial_value for batched v2",
        "requires zero",
        LCA_RESET_PROVENANCE,
    ),
    _RejectionCase(
        "trial_reset",
        _lca_rejection_builder(
            "reset lca",
            lambda: {"reset_stateful_function_when": pnl.AtTrialStart()},
        ),
        "unsupported LCA reset policy for batched v2",
        "AtTrialStart",
        LCA_RESET_PROVENANCE,
    ),
    _RejectionCase(
        "clip",
        _lca_rejection_builder("clip lca", lambda: {"clip": (0.2, 0.8)}),
        "unsupported LCA clip for batched v2",
        "",
        LCA_CLIP_PROVENANCE,
    ),
    _RejectionCase(
        "custom_recurrent_matrix",
        _lca_rejection_builder(
            "matrix lca",
            lambda: {"matrix": [[0.2, -0.1], [-0.4, 0.2]]},
        ),
        "unsupported LCA recurrent matrix for batched v2",
        "requires canonical self-excitation/competition matrix",
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "normalized_recurrent_projection",
        lambda: _recurrent_projection_model(
            function_parameter=("normalize", True),
        ),
        "unsupported LCA recurrent projection for batched v2",
        (
            "requires MatrixTransform(operation=DOT_PRODUCT, normalize=False); "
            "got MatrixTransform(operation='dot_product', normalize=True, "
            "weight=None, exponent=None)"
        ),
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "non_dot_recurrent_projection",
        lambda: _recurrent_projection_model(
            function_parameter=("operation", pnl.L0),
        ),
        "unsupported LCA recurrent projection for batched v2",
        (
            "requires MatrixTransform(operation=DOT_PRODUCT, normalize=False); "
            "got MatrixTransform(operation='difference', normalize=False, "
            "weight=None, exponent=None)"
        ),
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "weighted_recurrent_projection",
        lambda: _recurrent_projection_model(
            projection_parameter=("weight", 2.0),
        ),
        "unsupported LCA recurrent projection for batched v2",
        (
            "requires MatrixTransform(operation=DOT_PRODUCT, normalize=False); "
            "got MatrixTransform(operation='dot_product', normalize=False, "
            "weight=2.0, exponent=None)"
        ),
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "exponentiated_recurrent_projection",
        lambda: _recurrent_projection_model(
            projection_parameter=("exponent", 2.0),
        ),
        "unsupported LCA recurrent projection for batched v2",
        (
            "requires MatrixTransform(operation=DOT_PRODUCT, normalize=False); "
            "got MatrixTransform(operation='dot_product', normalize=False, "
            "weight=None, exponent=2.0)"
        ),
        LCA_MATRIX_PROVENANCE,
    ),
    _RejectionCase(
        "non_trial_termination_measure",
        _lca_rejection_builder(
            "termination measure lca",
            lambda: {"termination_measure": max},
        ),
        "unsupported LCA termination measure for batched v2",
        "requires TimeScale.TRIAL step-count semantics",
        LCA_THRESHOLD_PROVENANCE,
    ),
    _RejectionCase(
        "negative_trial_threshold",
        _lca_rejection_builder(
            "negative threshold lca",
            lambda: {"termination_threshold": -0.5},
        ),
        "unsupported LCA termination_threshold for batched v2",
        "requires a finite nonnegative scalar",
        LCA_THRESHOLD_PROVENANCE,
    ),
    _RejectionCase(
        "static_step_count_exceeds_fp32_exact_range",
        _lca_rejection_builder(
            "oversized threshold lca",
            lambda: {"termination_threshold": FP32_EXACT_INTEGER_LIMIT + 1},
        ),
        "unsupported LCA termination step count for batched v2",
        f"requires no more than {FP32_EXACT_INTEGER_LIMIT} executions",
        LCA_THRESHOLD_PROVENANCE,
    ),
    *(
        _RejectionCase(
            f"invalid_maximum_execution_count_{name}",
            lambda value=value: _invalid_maximum_execution_model(value),
            "unsupported LCA maximum execution count for batched v2",
            "requires a positive integer",
            LCA_THRESHOLD_PROVENANCE,
        )
        for name, value in (("zero", 0), ("fractional", 1.5), ("nonfinite", np.nan))
    ),
    _RejectionCase(
        "isolated_stepwise_execution",
        _lca_rejection_builder(
            "stepwise lca",
            lambda: {"execute_until_finished": False},
        ),
        "unsupported LCA execution mode for batched v2",
        "execute_until_finished must be False only for an Always/WhenFinished stepwise pair",
        LCA_THRESHOLD_PROVENANCE,
    ),
    _RejectionCase(
        "controlled_logistic_gain",
        _controlled_gain_model,
        "unsupported generic ControlMechanism for batched v2",
        "gain source->controlled gain lca.gain",
        LOGISTIC_LLVM_PROVENANCE,
    ),
    *(
        _RejectionCase(
            f"absorbed_control_{role}_{type(condition).__name__.lower()}_schedule",
            lambda role=role, condition=condition: _controlled_lca_schedule_model(
                role,
                condition,
            ),
            "unsupported absorbed control scheduler condition for batched v2",
            f"{role} scheduled {role} uses {condition_label}",
            "tests/scheduling/test_scheduler.py condition matrix",
            BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED,
        )
        for role, condition, condition_label in (
            ("controller", pnl.Never(), "Never"),
            (
                "target",
                pnl.AtPass(0),
                "AtPass(0, time_scale=ENVIRONMENT_STATE_UPDATE)",
            ),
            (
                "source",
                pnl.AtPass(
                    0,
                    time_scale=pnl.TimeScale.ENVIRONMENT_SEQUENCE,
                ),
                "AtPass(0, time_scale=ENVIRONMENT_SEQUENCE)",
            ),
        )
    ),
)


def test_absorbed_control_source_explicit_always_is_the_implicit_default():
    composition, controller = _controlled_lca_schedule_model("source", pnl.Always())
    source = controller.input_ports[0].path_afferents[0].sender.owner

    lowering = lower_composition(composition)
    report = BatchedCompositionCompiler.diagnose(
        composition,
        backend="triton_cpu",
    )

    assert lowering.graph is not None
    source_component_id = next(
        component_id
        for component_id, component in lowering.bindings.nodes_by_id.items()
        if component is source
    )
    source_condition = next(
        condition
        for condition in lowering.graph.scheduler
        if condition.component_id == source_component_id
    )
    assert source_condition.condition_type == "Always"
    assert source_condition.attrs == {"implicit": True}
    assert not lowering.rejected_conditions
    assert report.model_supported


@pytest.mark.parametrize("case", REJECTION_CASES, ids=lambda case: case.name)
def test_unmodeled_lca_configuration_has_structured_rejection(case):
    composition, component = case.build()
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")

    matches = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.component == component.name
    ]
    assert not report.model_supported, case.provenance
    assert report.codegen_ready is None
    assert len(matches) == 1
    diagnostic = matches[0]
    assert diagnostic.reason == case.reason
    assert diagnostic.detail == case.detail
    assert diagnostic.code.startswith("model.")
    if case.code is not None:
        assert diagnostic.code == case.code
    assert diagnostic.component_id.endswith(f":{component.name}")
    assert diagnostic.to_dict() == {
        "code": diagnostic.code,
        "component": component.name,
        "component_id": diagnostic.component_id,
        "reason": case.reason,
        "detail": case.detail,
    }

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")
    assert error.value.capability_report == report
    assert diagnostic.formatted_reason in str(error.value)


def test_unsupported_absorbed_source_returns_diagnostics_instead_of_crashing():
    composition, cue = _unsupported_termination_source_model()
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")

    cue_diagnostics = [
        diagnostic
        for diagnostic in report.model_diagnostics
        if diagnostic.component == cue.name
    ]
    assert not report.model_supported
    assert any(
        diagnostic.code == BatchedDiagnosticCode.MODEL_FUNCTION_UNSUPPORTED
        for diagnostic in cue_diagnostics
    )

    with pytest.raises(BatchedCompileError) as error:
        BatchedCompositionCompiler.compile(composition, backend="triton_cpu")
    assert error.value.capability_report == report


def test_absorbed_source_at_pass_zero_remains_supported():
    composition, _ = _controlled_lca_schedule_model("source", pnl.AtPass(0))
    report = BatchedCompositionCompiler.diagnose(composition, backend="triton_cpu")

    assert report.model_supported, report.to_dict()
    assert not any(
        diagnostic.code
        == BatchedDiagnosticCode.MODEL_SCHEDULER_CONDITION_UNSUPPORTED
        for diagnostic in report.model_diagnostics
    )
