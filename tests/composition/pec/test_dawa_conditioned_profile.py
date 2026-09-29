"""Nuisance optimization must actually compensate while the profiled mode stays fixed."""

from pathlib import Path
import sys

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/dawa"
sys.path.insert(0, str(DIRECTORY))
from dawa_conditioned_profile import canonical_profile_values, optimize_point  # noqa: E402


def test_profile_reoptimizes_nuisance_coordinates_on_an_exact_ridge():
    names = ["mode", "gain", "other"]
    bounds = {name: (0.0, 1.0, 0.01) for name in names}
    initial = [0.5, 0.1, 0.9]
    seen = []

    def evaluate(rows):
        values = np.asarray(rows)
        seen.extend(values.tolist())
        # All fixed mode values have the same optimum if gain is reoptimized.
        return -100 * (
            (values[:, 0] + values[:, 1] - 1) ** 2 + (values[:, 2] - 0.3) ** 2
        )

    results = []
    for value in (0.2, 0.8):
        seen.clear()
        result = optimize_point(
            evaluate,
            names,
            bounds,
            [initial],
            0,
            value,
            evaluations=401,
            population=10,
            optimizer_seed=121,
            sigma=0.2,
        )
        assert len(seen) == result["evaluations"] == 401
        assert all(row[0] == value for row in seen)
        assert result["parameters"][1] == pytest.approx(1 - value, abs=0.02)
        assert result["parameters"][2] == pytest.approx(0.3, abs=0.02)
        assert result["training_score"] > -0.05
        assert result["training_score"] > result["best_initial_score"] + 10
        results.append(result)
    np.testing.assert_array_equal(initial, [0.5, 0.1, 0.9])
    assert abs(results[0]["training_score"] - results[1]["training_score"]) < 0.05


def test_profile_requires_grid_aligned_fixed_value_and_sufficient_budget():
    names = ["a", "b", "c"]
    bounds = {name: (0.0, 1.0, 0.1) for name in names}
    common = dict(
        evaluate=lambda rows: np.zeros(len(rows)),
        names=names,
        bounds=bounds,
        initials=[[0.5, 0.5, 0.5]],
        fixed_index=0,
        population=4,
        optimizer_seed=1,
    )
    with pytest.raises(ValueError, match="parameter grid"):
        optimize_point(**common, fixed_value=0.55, evaluations=10)
    with pytest.raises(ValueError, match="budget"):
        optimize_point(**common, fixed_value=0.5, evaluations=0)


def test_profile_grid_aliases_do_not_create_duplicate_validation_labels():
    raw = [0.3, 0.1 + 200 * 0.001, 0.65, 0.9]
    assert len(set(raw)) == 4
    values = canonical_profile_values(raw, (0.1, 0.9, 0.001))
    assert len(values) == 3
    assert [f"mode0={value:.6g}" for value in values] == [
        "mode0=0.3",
        "mode0=0.65",
        "mode0=0.9",
    ]
    with pytest.raises(ValueError, match="parameter grid"):
        canonical_profile_values([0.1005], (0.1, 0.9, 0.001))
