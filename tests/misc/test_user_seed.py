import subprocess
import sys

def test_user_seed():
    """Check that different invocations use different seed at startup"""

    cmd = 'from psyneulink.core.globals.utilities import _get_global_seed; print(_get_global_seed())'

    seed1 = subprocess.check_output((sys.executable, "-c", cmd))
    seed2 = subprocess.check_output((sys.executable, "-c", cmd))

    assert seed1 != seed2

def test_seed_type():
    """Check that updating the global seed maintains its type"""

    cmd_vanilla = 'from psyneulink.core.globals.utilities import _get_global_seed; print(_get_global_seed().dtype)'
    cmd_updated = 'from psyneulink.core.globals.utilities import _get_global_seed, set_global_seed; set_global_seed(5); print(_get_global_seed().dtype)'

    seed1_dtype = subprocess.check_output((sys.executable, "-c", cmd_vanilla))
    seed2_dtype = subprocess.check_output((sys.executable, "-c", cmd_updated))

    assert seed1_dtype == seed2_dtype
