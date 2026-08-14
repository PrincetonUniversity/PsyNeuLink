"""Reusable Python-oracle support for deterministic batched compiler tests."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

import psyneulink as pnl
from psyneulink.core.batched import BatchedCompositionCompiler


@dataclass(frozen=True)
class SemanticModel:
    """One freshly built composition and the objects needed to execute it."""

    composition: pnl.Composition
    inputs: Mapping
    outputs: Sequence


@dataclass(frozen=True)
class SemanticCase:
    """A deterministic semantic case that can be rebuilt for each execution path."""

    name: str
    build: Callable[[], SemanticModel]
    provenance: str
    parameter_sets: Sequence[Mapping] = field(default_factory=lambda: ({},))
    num_estimates: int = 1
    atol: float = 1e-5
    rtol: float = 1e-4
    max_steps: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class SemanticComparison:
    python_values: np.ndarray
    batched_values: np.ndarray


def assert_matches_python(case: SemanticCase, *, backend: str) -> SemanticComparison:
    """Run ``case`` on fresh Python and batched models and compare every output."""

    python_model = case.build()
    python_values = _run_python_model(python_model)

    if not case.parameter_sets:
        raise AssertionError(f"SemanticCase {case.name!r} must define at least one parameter set")
    if any(parameter_set for parameter_set in case.parameter_sets):
        raise NotImplementedError(
            "Non-empty runtime parameter sets need a Python parameter applicator; "
            f"case provenance: {case.provenance}"
        )
    if case.num_estimates < 1:
        raise AssertionError(f"SemanticCase {case.name!r} num_estimates must be positive")

    batched_model = case.build()
    if batched_model.composition is python_model.composition:
        raise AssertionError(
            f"SemanticCase {case.name!r} reused a Composition; build() must return a fresh model"
        )

    compile_kwargs = {}
    if case.max_steps is not None:
        compile_kwargs["max_steps"] = case.max_steps
    plan = BatchedCompositionCompiler.compile(
        batched_model.composition,
        backend=backend,
        outputs=tuple(batched_model.outputs),
        **compile_kwargs,
    )
    result = plan.run(
        inputs=batched_model.inputs,
        parameter_sets=case.parameter_sets,
        num_estimates=case.num_estimates,
        seed=case.seed,
    )

    expected_shape = (
        len(case.parameter_sets),
        1,
        python_values.shape[0],
        case.num_estimates,
        python_values.shape[1],
    )
    assert result.values.shape == expected_shape
    batched_values = np.asarray(result.values)
    assert np.all(np.isfinite(batched_values))
    expected = np.broadcast_to(
        python_values[None, None, :, None, :],
        expected_shape,
    )
    np.testing.assert_allclose(
        batched_values,
        expected,
        atol=case.atol,
        rtol=case.rtol,
        err_msg=f"batched backend {backend!r} disagrees with Python for {case.name!r}",
    )
    return SemanticComparison(python_values=python_values, batched_values=batched_values)


def _run_python_model(model: SemanticModel) -> np.ndarray:
    model.composition.run(inputs=model.inputs, execution_mode=pnl.ExecutionMode.Python)
    result_indices = [_result_index(model.composition, output) for output in model.outputs]

    rows = []
    for trial in model.composition.results:
        rows.append(np.concatenate([
            np.asarray(trial[index], dtype=float).reshape(-1)
            for index in result_indices
        ]))
    return np.asarray(rows, dtype=float)


def _result_index(composition, output_port) -> int:
    matches = []
    for index, cim_input in enumerate(composition.output_CIM.input_ports):
        if any(projection.sender is output_port for projection in cim_input.path_afferents):
            matches.append(index)
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one output-CIM mapping for {output_port.full_name}; found {len(matches)}"
        )
    return matches[0]
