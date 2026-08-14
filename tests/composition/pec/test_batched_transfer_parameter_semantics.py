import itertools

import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched import registry as batched_registry

from batched_semantic_test_support import (
    SemanticCase,
    SemanticModel,
    assert_matches_python,
)


pytestmark = [
    pytest.mark.batched,
    pytest.mark.composition,
]


LINEAR_LLVM_PROVENANCE = "tests/functions/test_transfer.py::test_execute[LINEAR]"
LOGISTIC_LLVM_PROVENANCE = "tests/functions/test_transfer.py::test_execute[LOGISTIC]"
NOISE_LLVM_PROVENANCE = (
    "tests/mechanisms/test_transfer_mechanism.py::"
    "TestTransferMechanismNoise::test_transfer_mech_array_var_float_noise"
)
CLIP_LLVM_PROVENANCE = (
    "tests/mechanisms/test_transfer_mechanism.py::TestClip::test_clip_array"
)

_UNSPECIFIED = object()


def _transfer_case(
    name,
    function_class,
    function_kwargs,
    inputs,
    provenance,
    *,
    noise=_UNSPECIFIED,
    clip=_UNSPECIFIED,
):
    input_values = np.asarray(inputs, dtype=float)
    assert input_values.ndim == 2
    build_number = itertools.count()

    def build():
        mechanism_kwargs = {}
        if noise is not _UNSPECIFIED:
            mechanism_kwargs["noise"] = noise
        if clip is not _UNSPECIFIED:
            mechanism_kwargs["clip"] = clip

        mechanism = pnl.TransferMechanism(
            input_shapes=input_values.shape[1],
            function=function_class(**function_kwargs),
            name=f"{name}-{next(build_number)}",
            **mechanism_kwargs,
        )
        composition = pnl.Composition(pathways=mechanism)
        return SemanticModel(
            composition=composition,
            inputs={mechanism: input_values.copy()},
            outputs=(mechanism.output_port,),
        )

    return SemanticCase(
        name=name,
        build=build,
        provenance=provenance,
    )


LINEAR_INPUTS = (
    (-2.0, -0.25, 1.5),
    (0.0, 2.25, -4.0),
)

LINEAR_PARAMETER_CASES = (
    _transfer_case(
        "linear_slope_only",
        pnl.Linear,
        {"slope": -1.75},
        LINEAR_INPUTS,
        LINEAR_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "linear_intercept_only",
        pnl.Linear,
        {"intercept": 0.625},
        LINEAR_INPUTS,
        LINEAR_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "linear_scale_only",
        pnl.Linear,
        {"scale": 2.5},
        LINEAR_INPUTS,
        LINEAR_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "linear_offset_only",
        pnl.Linear,
        {"offset": -0.75},
        LINEAR_INPUTS,
        LINEAR_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "linear_all_parameters_order",
        pnl.Linear,
        {
            "slope": -1.5,
            "intercept": 0.75,
            "scale": 2.25,
            "offset": -0.5,
        },
        (
            (-2.0, -0.25, 0.5, 1.75),
            (3.0, 0.0, -1.25, 4.5),
        ),
        LINEAR_LLVM_PROVENANCE,
    ),
)


LOGISTIC_INPUTS = (
    (-3.0, -0.5, 0.25, 2.0),
    (1.0, -1.25, 4.0, 0.0),
)

LOGISTIC_PARAMETER_CASES = (
    _transfer_case(
        "logistic_gain_only",
        pnl.Logistic,
        {"gain": 2.5},
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "logistic_bias_only",
        pnl.Logistic,
        {"bias": 0.75},
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "logistic_x_0_only",
        pnl.Logistic,
        {"x_0": 1.25},
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "logistic_scale_only",
        pnl.Logistic,
        {"scale": 1.7},
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "logistic_offset_only",
        pnl.Logistic,
        {"offset": -0.3},
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
    _transfer_case(
        "logistic_all_parameters_order",
        pnl.Logistic,
        {
            "gain": 1.6,
            "bias": 0.45,
            "x_0": 1.1,
            "scale": 1.8,
            "offset": -0.25,
        },
        LOGISTIC_INPUTS,
        LOGISTIC_LLVM_PROVENANCE,
    ),
)


TRANSFER_PIPELINE_CASES = (
    _transfer_case(
        "constant_noise_before_linear",
        pnl.Linear,
        {"slope": -2.0, "intercept": 0.4},
        (
            (-2.0, -0.75, 0.0, 1.25, 3.0),
            (4.0, -3.5, 0.5, 2.0, -0.25),
        ),
        NOISE_LLVM_PROVENANCE,
        noise=0.75,
    ),
    _transfer_case(
        "clip_after_linear",
        pnl.Linear,
        {"slope": -1.75, "intercept": 0.6},
        (
            (-2.0, -0.5, 0.0, 0.75, 2.0),
            (3.0, -4.0, 0.25, 1.0, -1.25),
        ),
        CLIP_LLVM_PROVENANCE,
        clip=(-1.2, 1.4),
    ),
    _transfer_case(
        "linear_noise_function_clip_order",
        pnl.Linear,
        {
            "slope": -1.6,
            "intercept": 0.7,
            "scale": 2.3,
            "offset": -0.4,
        },
        (
            (-3.0, -0.5, 0.0, 1.25, 4.0),
            (2.0, -2.0, 0.75, 3.0, -0.25),
        ),
        f"{LINEAR_LLVM_PROVENANCE}; {NOISE_LLVM_PROVENANCE}; {CLIP_LLVM_PROVENANCE}",
        noise=0.55,
        clip=(-2.1, 2.4),
    ),
    _transfer_case(
        "logistic_noise_function_clip_order",
        pnl.Logistic,
        {
            "gain": 1.7,
            "bias": 0.45,
            "x_0": -0.25,
            "scale": 2.1,
            "offset": -0.4,
        },
        (
            (-3.0, -1.0, 0.0, 1.0, 3.0),
            (2.0, -2.5, 0.5, 4.0, -0.25),
        ),
        f"{LOGISTIC_LLVM_PROVENANCE}; {NOISE_LLVM_PROVENANCE}; {CLIP_LLVM_PROVENANCE}",
        noise=0.6,
        clip=(0.05, 1.25),
    ),
)


CASES = LINEAR_PARAMETER_CASES + LOGISTIC_PARAMETER_CASES + TRANSFER_PIPELINE_CASES


def test_transfer_pipeline_lowers_as_explicit_chained_ops(monkeypatch):
    monkeypatch.setattr(
        batched_registry,
        "_backend_availability",
        lambda backend: (True, []),
    )
    mechanism = pnl.TransferMechanism(
        input_shapes=2,
        function=pnl.Linear(slope=-2.0, intercept=0.4),
        noise=[0.75, 0.75],
        clip=(-1.2, 1.4),
        name="pipeline",
    )
    plan = BatchedCompositionCompiler.compile(pnl.Composition(pathways=mechanism))
    ops = plan.kernel_ir.ops

    assert tuple(op.kind for op in ops) == (
        "LoadInput",
        "AddConstant",
        "CallFunction",
        "Clamp",
        "StoreOutput",
    )
    load, add, call, clamp, store = ops
    assert load.outputs == add.inputs
    assert add.outputs == call.inputs
    assert call.outputs == clamp.inputs
    assert clamp.outputs == store.inputs


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_transfer_semantics_match_python(case, batched_backend):
    assert_matches_python(case, backend=batched_backend)
