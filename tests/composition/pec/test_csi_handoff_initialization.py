"""Keep direct seconds and GPU step coordinates distinct when starting fits."""

import importlib.util
import json
from pathlib import Path

import pytest


path = (Path(__file__).resolve().parents[3] / "Scripts/Debug/pec_batch_compile/csi_fit"
        / "data fitting/csi_warm_start.py")
spec = importlib.util.spec_from_file_location("csi_warm_start", path)
warm_start = importlib.util.module_from_spec(spec)
spec.loader.exec_module(warm_start)


@pytest.fixture
def initial_fit(tmp_path):
    payload = {
        "subject_nr": 1, "parameter_names": warm_start.PARAMETER_NAMES,
        "parameter_vector": [23.985, 38.614, 22.326, .113429, .274977, .068305, .083005,
                             -.3, -.025358, -.045534, .150499, .438501, .398071],
    }
    source = tmp_path / "fit.json"
    source.write_text(json.dumps(payload))
    return source, payload


def bounds(dt):
    result = {"Cue Stimulus Interval.slope": (0., round(.3 / dt), 1.)}
    for condition in ("RealRare", "RealFrequent", "NoInstruction"):
        for name, limits in (
            ("Task Activations [C1, C2].gain", (5., 120., .1)),
            ("Threshold Mechanism.intercept", (.05, .3, .0005)),
            ("Threshold Mechanism.offset-integrator_function", (-.3 * dt, 0., .001 * dt)),
            ("DDM.non_decision_time", (.1, .5, .001)),
        ):
            result[f"{name}[{condition}]"] = limits
    return result


@pytest.mark.parametrize("dt, csi_steps", [(.001, 113.), (.01, 11.)])
def test_direct_start_preserves_physical_units_and_condition_labels(initial_fit, dt, csi_steps):
    source, _ = initial_fit
    initial, record = warm_start.load_direct_start(source, 1, dt, bounds(dt))
    assert initial["Cue Stimulus Interval.slope"] == csi_steps
    assert initial["Task Activations [C1, C2].gain[RealRare]"] == pytest.approx(38.6)
    assert initial["Threshold Mechanism.intercept[NoInstruction]"] == pytest.approx(.275)
    assert initial["DDM.non_decision_time[RealRare]"] == pytest.approx(.439)
    assert initial["Threshold Mechanism.offset-integrator_function[RealRare]"] / dt == pytest.approx(-.025)
    assert initial["Threshold Mechanism.offset-integrator_function[NoInstruction]"] / dt == pytest.approx(-.3)
    assert len(initial) == 13
    assert record["gpu_parameters"] == initial
    assert record["gpu_parameters_before_rounding"]["Cue Stimulus Interval.slope"] * dt == pytest.approx(.113429)


@pytest.mark.parametrize("invalid", ["subject", "order", "nonfinite", "bounds"])
def test_incompatible_direct_starts_are_rejected(initial_fit, invalid):
    source, payload = initial_fit
    if invalid == "subject":
        payload["subject_nr"] = 2
    elif invalid == "order":
        payload["parameter_names"] = list(reversed(payload["parameter_names"]))
    elif invalid == "nonfinite":
        payload["parameter_vector"][0] = float("nan")
    else:
        payload["parameter_vector"][0] = 121.
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        warm_start.load_direct_start(source, 1, .001, bounds(.001))
