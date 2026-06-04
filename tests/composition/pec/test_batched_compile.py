import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import psyneulink as pnl

from psyneulink.core.batched import (
    BatchedCompileError,
    BatchedCompositionCompiler,
)
from psyneulink.core.components.functions.nonstateful.fitfunctions import PECOptimizationFunction


pytestmark = pytest.mark.usefixtures("set_threads_to_one")


def _make_ddm_comp(noise=0.0):
    decision = pnl.DDM(
        function=pnl.DriftDiffusionIntegrator(
            starting_value=0.0,
            rate=1.0,
            noise=noise,
            threshold=0.05,
            non_decision_time=0.0,
            time_step_size=0.01,
        ),
        output_ports=[pnl.DECISION_OUTCOME, pnl.RESPONSE_TIME],
        name="DDM",
    )
    return pnl.Composition(pathways=decision), decision


def _triton_available():
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


@pytest.mark.composition
def test_batched_reference_ddm_deterministic():
    comp, decision = _make_ddm_comp(noise=0.0)
    plan = BatchedCompositionCompiler.compile(comp, backend="reference")
    result = plan.run(
        inputs={decision: np.array([[1.0], [-1.0]], dtype=float)},
        parameter_sets=[{"rate": 1.0, "threshold": 0.05, "time_step_size": 0.01}],
        num_estimates=2,
        seed=1,
    )

    assert result.values.shape == (1, 1, 2, 2, 2)
    np.testing.assert_allclose(result.values[0, 0, :, :, 0], [[1.0, 1.0], [0.0, 0.0]])
    np.testing.assert_allclose(result.values[0, 0, :, :, 1], [[0.05, 0.05], [0.05, 0.05]])


@pytest.mark.composition
def test_batched_reference_ddm_common_random_numbers():
    comp, decision = _make_ddm_comp(noise=0.2)
    plan = BatchedCompositionCompiler.compile(comp, backend="reference")
    params = [
        {"rate": 0.0, "threshold": 0.05, "noise": 0.2, "time_step_size": 0.01},
        {"rate": 0.0, "threshold": 0.05, "noise": 0.2, "time_step_size": 0.01},
    ]
    result = plan.run(
        inputs={decision: np.array([[0.0], [0.0]], dtype=float)},
        parameter_sets=params,
        num_estimates=4,
        seed=7,
        common_random_numbers=True,
    )

    np.testing.assert_array_equal(result.values[0], result.values[1])


@pytest.mark.composition
def test_batched_compiler_rejects_unsupported_composition():
    mech = pnl.ProcessingMechanism(input_shapes=1, name="plain")
    comp = pnl.Composition(pathways=mech)
    report = BatchedCompositionCompiler.diagnose(comp)

    assert not report.is_supported
    assert report.unsupported_reasons
    with pytest.raises(BatchedCompileError):
        BatchedCompositionCompiler.compile(comp)


@pytest.mark.composition
def test_pec_can_compile_batched_diagnostic():
    comp, decision = _make_ddm_comp(noise=0.0)
    data = pd.DataFrame({"decision": [1.0], "response_time": [0.05]})
    data["decision"] = data["decision"].astype("category")
    pec = pnl.ParameterEstimationComposition(
        name="pec",
        nodes=[comp],
        parameters={("threshold", decision): [0.05, 0.1]},
        outcome_variables=[
            decision.output_ports[pnl.DECISION_OUTCOME],
            decision.output_ports[pnl.RESPONSE_TIME],
        ],
        data=data,
        optimization_function=PECOptimizationFunction(method="differential_evolution"),
        num_estimates=1,
        initial_seed=1,
    )

    report = pec.can_compile_batched(backend="reference")
    assert report.is_supported
    assert report.model_kind == "ddm"


@pytest.mark.composition
def test_stability_flexibility_reference_smoke():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=3)
    inputs = make_input_dict(comp, task[:2], stimulus[:2], cue[:2], correct[:2])
    plan = BatchedCompositionCompiler.compile(comp, backend="reference")
    result = plan.run(
        inputs=inputs,
        parameter_sets=[{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}],
        num_estimates=1,
        seed=3,
    )

    assert result.values.shape == (1, 1, 2, 1, 2)
    assert np.all(np.isfinite(result.values))


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_ddm_matches_reference_deterministic():
    comp, decision = _make_ddm_comp(noise=0.0)
    inputs = {decision: np.array([[1.0], [-1.0]], dtype=float)}
    params = [{"rate": 1.0, "threshold": 0.05, "noise": 0.0, "time_step_size": 0.01}]
    reference = BatchedCompositionCompiler.compile(comp, backend="reference", max_steps=64).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )
    triton = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=64).run(
        inputs=inputs,
        parameter_sets=params,
        num_estimates=2,
        seed=11,
    )

    np.testing.assert_allclose(triton.values, reference.values, rtol=1e-6, atol=1e-6)


@pytest.mark.triton
@pytest.mark.skipif(not _triton_available(), reason="Triton CUDA backend is not available")
def test_triton_stability_flexibility_smoke():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_stab_flex_pec_fit import generate_trial_sequence, make_input_dict, make_stab_flex

    comp = make_stab_flex(
        lca_time_step_size=0.01,
        ddm_time_step_size=0.01,
        threshold=0.05,
        ddm_noise=0.0,
        lca_noise=0.0,
    )
    task, stimulus, cue, correct = generate_trial_sequence(16, 0.5, seed=4)
    inputs = make_input_dict(comp, task[:1], stimulus[:1], cue[:1], correct[:1])
    result = BatchedCompositionCompiler.compile(comp, backend="triton", max_steps=256).run(
        inputs=inputs,
        parameter_sets=[{"threshold": 0.05, "ddm_noise": 0.0, "lca_noise": 0.0}],
        num_estimates=1,
        seed=4,
    )

    assert result.values.shape == (1, 1, 1, 1, 2)
    assert np.all(np.isfinite(result.values))
