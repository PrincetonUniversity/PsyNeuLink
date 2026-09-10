"""Device paths retain the inspection contract without full host transfers."""

from dataclasses import replace

import numpy as np
import pytest

from psyneulink.core.batched import BoundaryTrajectoryError
from test_batched_reduced_scoring import scoring_case


pytestmark = [pytest.mark.batched, pytest.mark.composition]


def test_device_paths_match_inspection_without_large_host_copies(scoring_case, monkeypatch):
    import torch

    observation, inputs, data, rows = scoring_case
    plan = observation.sampler.path_plan
    reference = plan.generate(inputs, data, rows)
    original = torch.Tensor.cpu

    def small_only(tensor, *args, **kwargs):
        assert tensor.numel() < 512, "A full trajectory/diagnostic array was downloaded"
        return original(tensor, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", small_only)
        device = plan.generate_device(inputs, data, rows)
    np.testing.assert_array_equal(device.values.cpu().numpy(), reference.values)
    for name in ("start_states", "end_states", "start_effective_parameters", "end_effective_parameters", "event_counts"):
        np.testing.assert_array_equal(getattr(device.history, name), getattr(reference.history, name))
    assert device.fields == reference.fields
    assert device.values.device.type == ("cuda" if plan.history_plan.simulation_plan.backend == "triton" else "cpu")
    assert not hasattr(device, "pass_indices")


@pytest.mark.parametrize("fault,code", [("nonfinite", "boundary.nonfinite"),
                                       ("prefix", "boundary.prefix_missing"),
                                       ("start", "boundary.start_mismatch")])
def test_device_and_host_paths_reject_same_faults(scoring_case, monkeypatch, fault, code):
    from psyneulink.core.batched.backend.triton import trajectories

    observation, inputs, data, rows = scoring_case
    plan = observation.sampler.path_plan
    original = trajectories._run_history_trace

    def inject(*args, **kwargs):
        result = original(*args, **kwargs)
        values, valid, _, _, _ = kwargs["extra_kernel_args"]
        if fault == "nonfinite":
            values[0, 0, 0, 0] = float("nan")
        elif fault == "prefix":
            valid[0, 0, 0] = False
        else:
            starts = result.start_states.copy()
            starts[0, 0, 0] += 1.
            result = replace(result, start_states=starts)
        return result

    monkeypatch.setattr(trajectories, "_run_history_trace", inject)
    for operation in (plan.generate, plan.generate_device):
        with pytest.raises(BoundaryTrajectoryError) as error:
            operation(inputs, data, rows, horizon=7)
        assert error.value.code == code


def test_device_validator_matches_masked_finite_and_prefix_rules(scoring_case):
    import torch
    from psyneulink.core.batched.backend.triton.trajectories import _validate_device_paths

    observation, _, _, _ = scoring_case
    interpret = observation.sampler.path_plan.history_plan.simulation_plan.backend == "triton_cpu"
    device = "cpu" if interpret else "cuda"
    # Non-power-of-two length exercises the final scan tile's mask.
    values = torch.ones((1, 2, 517, 3), device=device)
    valid = torch.zeros((1, 2, 517), dtype=torch.bool, device=device)
    valid[:, 0, :333] = True
    valid[:, 1, :500] = True
    expected = np.array([[333, 500]])
    values[:, 0, 333:] = float("nan")  # Invalid suffix is not consumed.
    _validate_device_paths(values, valid, expected, interpret=interpret)
    for bad in (float("nan"), float("inf"), -float("inf")):
        values[0, 1, 499, 2] = bad
        with pytest.raises(BoundaryTrajectoryError) as error:
            _validate_device_paths(values, valid, expected, interpret=interpret)
        assert error.value.code == "boundary.nonfinite"
    values[0, 1, 499, 2] = 1.
    valid[0, 1, 500] = True  # Extra valid step, not just a missing prefix step.
    with pytest.raises(BoundaryTrajectoryError) as error:
        _validate_device_paths(values, valid, expected, interpret=interpret)
    assert error.value.code == "boundary.prefix_missing"


def test_device_paths_preserve_resource_and_witness_guards(scoring_case):
    observation, inputs, data, rows = scoring_case
    plan = observation.sampler.path_plan
    with pytest.raises(BoundaryTrajectoryError) as error:
        plan.generate_device(inputs, data, rows, max_buffer_bytes=1)
    assert error.value.code == "boundary.memory_budget"
    forged = replace(plan, witness=replace(plan.witness, fields=()))
    with pytest.raises(BoundaryTrajectoryError) as error:
        forged.generate_device(inputs, data, rows)
    assert error.value.code == "boundary.witness_mismatch"
