import numpy as np
import os
import pytest
from itertools import combinations

from psyneulink.core import llvm as pnlvm
from psyneulink.core.components.functions.transferfunctions import Linear
from psyneulink.core.components.mechanisms.processing.integratormechanism import IntegratorMechanism
from psyneulink.core.components.mechanisms.processing.transfermechanism import TransferMechanism
from psyneulink.core.components.projections.pathway.mappingprojection import MappingProjection
from psyneulink.core.compositions.composition import Composition
from psyneulink.core.scheduling.scheduler import Scheduler

debug_options=["const_input=[[[7]]]", "const_data", "const_params", "const_data", "const_state"]

@pytest.mark.composition
@pytest.mark.parametrize("mode", [
                                  pytest.param('LLVMRun', marks=pytest.mark.llvm),
                                  pytest.param('PTXRun', marks=[pytest.mark.llvm, pytest.mark.cuda])
                                  ])
@pytest.mark.parametrize("debug_env", [";".join(list(comb) + ["debug_info"]) for i in range(len(debug_options) + 1) for comb in combinations(debug_options, i)])
def test_debug_comp(mode, debug_env):
    # save old debug env var
    old_env = os.environ.get("PNL_LLVM_DEBUG")
    if debug_env is not None:
        os.environ["PNL_LLVM_DEBUG"] = debug_env
        pnlvm.debug._update()

    comp = Composition()
    A = IntegratorMechanism(default_variable=1.0, function=Linear(slope=5.0))
    B = TransferMechanism(function=Linear(slope=5.0), integrator_mode=True)
    comp.add_node(A)
    comp.add_node(B)
    comp.add_projection(MappingProjection(sender=A, receiver=B), A, B)
    sched = Scheduler(composition=comp)

    inputs_dict = {A: [5]}
    output1 = comp.run(inputs=inputs_dict, scheduler=sched, bin_execute=mode)
    output2 = comp.run(inputs=inputs_dict, scheduler=sched, bin_execute=mode)
    # restore old debug env var and cleanup the debug configuration
    if old_env is None:
        del os.environ["PNL_LLVM_DEBUG"]
    else:
        os.environ["PNL_LLVM_DEBUG"] = old_env
    pnlvm.debug._update()

    assert len(comp.results) == 2

    if "const_input" in debug_env:
        expected1 = 87.5
        expected2 = 131.25
    else:
        expected1 = 62.5
        expected2 = 93.75

    if "const_state" in debug_env:
        expected2 = expected1


    assert np.allclose(expected1, output1[0][0])
    assert np.allclose(expected2, output2[0][0])
