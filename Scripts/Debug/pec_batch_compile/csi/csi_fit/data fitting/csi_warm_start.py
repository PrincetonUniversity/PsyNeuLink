"""Convert a direct CSI fit into a feasible GPU optimizer starting point."""

import hashlib
import json
import math
from pathlib import Path
import re


CONDITIONS = ("NoInstruction", "RealRare", "RealFrequent")
PARAMETER_NAMES = (
    [f"gain[{c}]" for c in CONDITIONS] + ["csi_duration"]
    + [f"threshold[{c}]" for c in CONDITIONS]
    + [f"collapse_rate[{c}]" for c in CONDITIONS]
    + [f"non_decision_time[{c}]" for c in CONDITIONS]
)


def load_direct_start(path, subject, time_step, bounds):
    """Return PNL parameter values snapped to the actual optimizer grids.

    Direct JSON values use seconds and collapse per second. GPU parameters
    use CSI steps and collapse per step. Reject incompatible or out-of-bound
    starts rather than silently changing the intended initialization.
    """
    if not math.isfinite(time_step) or time_step <= 0:
        raise ValueError("The GPU time step must be finite and positive.")
    path = Path(path).expanduser().resolve(strict=True)
    source = path.read_bytes()
    fit = json.loads(source)
    if fit.get("subject_nr") != subject:
        raise ValueError(f"Initial fit subject_nr must match the selected subject {subject}.")
    if fit.get("parameter_names") != PARAMETER_NAMES:
        raise ValueError("Initial parameters must be a direct CSI fit JSON with the expected parameter names/order.")
    vector = fit.get("parameter_vector", [])
    if len(vector) != 13 or not all(math.isfinite(float(x)) for x in vector):
        raise ValueError("Initial parameters must contain 13 finite values.")
    vector = list(map(float, vector))
    raw = {"Cue Stimulus Interval.slope": vector[3] / time_step}
    for index, condition in enumerate(CONDITIONS):
        for name, value in (
            ("Task Activations [C1, C2].gain", vector[index]),
            ("Threshold Mechanism.intercept", vector[4 + index]),
            ("Threshold Mechanism.offset-integrator_function", vector[7 + index] * time_step),
            ("DDM.non_decision_time", vector[10 + index]),
        ):
            raw[f"{name}[{condition}]"] = value

    initial = {}
    original = {}
    for name, (lower, upper, step) in bounds.items():
        mechanism, parameter = name.split(".", 1)
        mechanism = re.sub(r"-\d+$", "", mechanism)
        canonical = f"{mechanism}.{parameter}"
        if canonical not in raw:
            raise ValueError(f"Unexpected GPU fit parameter: {name}")
        value = raw[canonical]
        if not lower - 1e-12 <= value <= upper + 1e-12:
            raise ValueError(f"Initial {name}={value} is outside [{lower}, {upper}].")
        initial[name] = float(min(upper, max(lower, lower + round((value - lower) / step) * step)))
        original[name] = value
    if len(initial) != len(raw):
        raise ValueError("The GPU fit must contain all 13 CSI parameters.")
    return initial, {
        "source": str(path), "source_sha256": hashlib.sha256(source).hexdigest(),
        "subject_nr": subject, "model_time_step": time_step,
        "direct_parameter_names": PARAMETER_NAMES, "direct_parameter_vector": vector,
        "gpu_parameters_before_rounding": original, "gpu_parameters": initial,
        "gpu_parameter_bounds": {name: list(values) for name, values in bounds.items()},
    }
