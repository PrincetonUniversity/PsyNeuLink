import numpy as np
import os
import pytest
from itertools import combinations

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition

debug_options=["const_input=[[[7]]]", "const_input", "const_data", "const_params", "const_data", "const_state"]
options_combinations = (";".join(("debug_info", *c)) for i in range(len(debug_options) + 1) for c in combinations(debug_options, i))

@pytest.mark.composition
@pytest.mark.parametrize("mode", [
                                  pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                                  ])
@pytest.mark.parametrize("debug_env", [comb for comb in options_combinations if comb.count("const_input") < 2])
def test_debug_comp(mode, debug_env):
    # save old debug env var
    old_env = os.environ.get("PNL_LLVM_DEBUG")
    if debug_env is not None:
        os.environ["PNL_LLVM_DEBUG"] = debug_env
        pnlvm.debug._update()

    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0), integrator_mode=True)
    comp.add_linear_processing_pathway([A, B])

    inputs_dict = {A: [5]}
    output1 = comp.run(inputs=inputs_dict, bin_execute=mode)
    output2 = comp.run(inputs=inputs_dict, bin_execute=mode)
    # restore old debug env var and cleanup the debug configuration
    if old_env is None:
        del os.environ["PNL_LLVM_DEBUG"]
    else:
        os.environ["PNL_LLVM_DEBUG"] = old_env
    pnlvm.debug._update()

    assert len(comp.results) == 2

    if "const_input=" in debug_env:
        expected1 = 87.5
        expected2 = 131.25
    elif "const_input" in debug_env:
        expected1 = 12.5
        expected2 = 18.75
    else:
        expected1 = 62.5
        expected2 = 93.75

    if "const_state" in debug_env:
        expected2 = expected1


    assert np.allclose(expected1, output1[0][0])
    assert np.allclose(expected2, output2[0][0])
