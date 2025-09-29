import os
import pytest
from typing import cast

from psyneulink.core.globals import get_num_threads, set_num_threads, reset_num_threads

# Use the restore_num_threads fixture for these tests to ensure thread globals are restored
pytestmark = pytest.mark.usefixtures("restore_num_threads")


def test_set_get_reset_roundtrip():
    """Round-trip set/get/reset and verify common env vars are updated."""
    original = get_num_threads()
    try:
        # Set to 2 and verify
        set_num_threads(2)
        assert get_num_threads() == 2
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(v) == "2"

        # Set to 1 and verify
        set_num_threads(1)
        assert get_num_threads() == 1
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(v) == "1"
    finally:
        # Restore original setting to avoid side-effects
        set_num_threads(original)
        assert get_num_threads() == original


def test_set_invalid_types_and_values():
    """Invalid inputs should raise appropriate exceptions."""
    # Non-int types (use cast to avoid static type-checker warnings while still passing non-int at runtime)
    with pytest.raises(TypeError):
        set_num_threads(cast(int, 2.5))
    with pytest.raises(TypeError):
        set_num_threads(cast(int, "4"))

    # Invalid ints
    with pytest.raises(ValueError):
        set_num_threads(0)
    with pytest.raises(ValueError):
        set_num_threads(-1)


def test_reset_num_threads():
    """Verify that reset_num_threads restores the platform default and env vars."""
    original = get_num_threads()
    try:
        # Ensure we change to a non-default value first
        set_num_threads(2 if (os.cpu_count() or 1) != 2 else 1)
        assert get_num_threads() != (os.cpu_count() or 1)

        # Now reset and verify it equals platform default
        reset_num_threads()
        default = os.cpu_count() or 1
        assert get_num_threads() == default
        for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert os.environ.get(v) == str(default)
    finally:
        # Restore original to avoid affecting other tests
        set_num_threads(original)
        assert get_num_threads() == original
