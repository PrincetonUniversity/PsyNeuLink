import subprocess
import sys

def test_user_seed():
    seed1 = subprocess.check_output((sys.executable, "-c" ,"from psyneulink.core.globals.utilities import get_global_seed; print(get_global_seed())"))
    seed2 = subprocess.check_output((sys.executable, "-c" ,"from psyneulink.core.globals.utilities import get_global_seed; print(get_global_seed())"))
    assert seed1 != seed2
