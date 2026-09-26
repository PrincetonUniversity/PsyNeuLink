"""Source reuse must not bypass validation after nested IR mutations."""

import numpy as np
import pytest

from psyneulink.core.batched.backend.triton.cache import cached_kernel_source
from psyneulink.core.batched.backend.triton.free_running_score import HistogramEmitter
from tests.composition.pec.test_batched_reset_state_kernel_ir import _counted_lca_kernel, _trial_body

pytestmark = [pytest.mark.batched, pytest.mark.composition]


def test_cached_source_rechecks_mutable_ir_and_emission_variant():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    calls = []

    def emit():
        calls.append(True)
        return HistogramEmitter(kernel, [0], [False]).emit()

    source = cached_kernel_source(kernel, ("histogram", 0), emit)
    assert cached_kernel_source(kernel, ("histogram", 0), emit) == source
    assert len(calls) == 1
    # Distinct lowering options cannot share a cached source.
    assert cached_kernel_source(kernel, ("histogram", 1), emit) == source
    assert len(calls) == 2
    # Array contents nested inside metadata are part of the snapshot too.
    kernel.metadata["probe"] = {"matrix": np.zeros((2, 2))}
    cached_kernel_source(kernel, ("histogram", 0), emit)
    kernel.metadata["probe"]["matrix"][0, 0] = 1.
    cached_kernel_source(kernel, ("histogram", 0), emit)
    assert len(calls) == 4
    reset = _trial_body(kernel)[0]
    reset.attrs["state_ids"] = (reset.attrs["state_ids"][0],)
    for _ in range(2):
        with pytest.raises(ValueError, match="ResetState"):
            cached_kernel_source(kernel, ("histogram", 0), emit)
    assert len(calls) == 6  # Failed validation was never cached.


def test_nonserializable_extension_metadata_uses_validated_uncached_path():
    kernel, _ = _counted_lca_kernel(reset_at_trial_start=True)
    kernel.metadata["extension"] = lambda: None
    calls = []

    def emit():
        calls.append(True)
        return HistogramEmitter(kernel, [0], [False]).emit()

    assert cached_kernel_source(kernel, (), emit) == cached_kernel_source(kernel, (), emit)
    assert len(calls) == 2
