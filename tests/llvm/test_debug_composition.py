import numpy as np
import os
import pytest

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.nonstateful.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.compositions.composition import Composition


@pytest.fixture(autouse=True)
def preserve_env():

    # Save old debug env var
    old_env = os.environ.get("PNL_LLVM_DEBUG")

    yield

    # Restore old debug env var and reset the debug configuration
    if old_env is None:
        del os.environ["PNL_LLVM_DEBUG"]
    else:
        os.environ["PNL_LLVM_DEBUG"] = old_env
    pnlvm.debug._update()


debug_options = ["const_input=[[[7]]]", "const_input", "const_params", "const_data", "const_state",
                 "stat", "time_stat", "unaligned_copy", "printf_tags={'always'}"]
options_combinations = (";".join(c) for c in pytest.helpers.power_set(debug_options))

@pytest.mark.composition
@pytest.mark.parametrize("mode", [pytest.param(pnlvm.ExecutionMode.LLVMRun, marks=pytest.mark.llvm),
                                  pytest.helpers.cuda_param(pnlvm.ExecutionMode.PTXRun)
                                 ])
@pytest.mark.parametrize("debug_env", [comb for comb in options_combinations if comb.count("const_input") < 2])
def test_debug_comp(mode, debug_env):
    if debug_env is not None:
        os.environ["PNL_LLVM_DEBUG"] = debug_env
        pnlvm.debug._update()

    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0), integrator_mode=True)
    comp.add_linear_processing_pathway([A, B])

    inputs_dict = {A: [5]}
    output1 = comp.run(inputs=inputs_dict, execution_mode=mode)
    output2 = comp.run(inputs=inputs_dict, execution_mode=mode)

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

    np.testing.assert_allclose(expected1, output1[0][0])
    np.testing.assert_allclose(expected2, output2[0][0])
