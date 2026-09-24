import numpy as np
import pytest

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler
from psyneulink.core.batched.backend.triton.runtime import (
    _normalize_launch_options,
)


pytestmark = [pytest.mark.batched, pytest.mark.composition, pytest.mark.triton]


def test_triton_launch_option_defaults_are_stable():
    assert _normalize_launch_options(None, interpret=False) == {
        "block_size": 128,
        "num_warps": 4,
        "maxnreg": None,
        "normal_rng": "philox4x_v1",
    }


@pytest.mark.parametrize(
    "options",
    [
        {"block_size": 64},
        {"num_warps": 2},
        {"maxnreg": 128},
        {"normal_rng": "legacy"},
        {"block_size": 256, "num_warps": 8, "maxnreg": 96},
    ],
)
def test_triton_launch_options_accept_benchmark_configurations(options):
    normalized = _normalize_launch_options(options, interpret=False)
    assert normalized == {
        "block_size": 128,
        "num_warps": 4,
        "maxnreg": None,
        "normal_rng": "philox4x_v1",
        **options,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"unknown": 1}, "Unknown Triton launch option"),
        ({"block_size": 96}, "power-of-two"),
        ({"block_size": True}, "block_size"),
        ({"normal_rng": "fast"}, "normal_rng"),
        ({"normal_rng": None}, "normal_rng"),
        ({"num_warps": 3}, "num_warps"),
        ({"num_warps": 4.0}, "num_warps"),
        ({"maxnreg": 256}, "maxnreg"),
        ({"maxnreg": True}, "maxnreg"),
    ],
)
def test_triton_launch_options_reject_invalid_values(options, message):
    with pytest.raises(ValueError, match=message):
        _normalize_launch_options(options, interpret=False)


def test_custom_triton_launch_options_reject_cpu_interpreter():
    with pytest.raises(ValueError, match="compiled GPU backend"):
        _normalize_launch_options({"block_size": 64}, interpret=True)


def test_rng_selection_is_available_in_interpreter():
    assert _normalize_launch_options({"normal_rng": "legacy"}, interpret=True)["normal_rng"] == "legacy"


@pytest.mark.triton_gpu
def test_custom_triton_launch_options_match_default_gpu_result():
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            rate=0.2,
            noise=0.1,
            threshold=0.1,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
    )
    composition = pnl.Composition(pathways=decision)
    plan = BatchedCompositionCompiler.compile(
        composition,
        backend="triton",
        max_steps=100,
    )
    run_args = {
        "inputs": {decision: np.array([[1.0], [-1.0]])},
        "parameter_sets": [{"rate": 0.2}, {"rate": 0.3}],
        "num_estimates": 32,
        "seed": 7,
    }

    default = plan.run(**run_args)
    tuned = plan.run(
        **run_args,
        triton_launch_options={
            "block_size": 64,
            "num_warps": 2,
            "maxnreg": 96,
        },
    )

    np.testing.assert_array_equal(tuned.values, default.values)
    assert tuned.metadata["triton_launch_options"] == {
        "block_size": 64,
        "num_warps": 2,
        "maxnreg": 96,
        "normal_rng": "philox4x_v1",
    }
