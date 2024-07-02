import gc
import numpy as np
import pytest
import weakref

import graph_scheduler as gs
import psyneulink as pnl


@pytest.mark.composition
@pytest.mark.parametrize("run", ["not_run", "run"])
def test_composition_leak(comp_mode, run):

    c = pnl.Composition()
    t = pnl.TransferMechanism()
    c.add_node(t)

    if run == "run":
        res = c.run([5], execution_mode=comp_mode)
        np.testing.assert_array_equal(res, [[5]])

    weak_c = weakref.ref(c)
    weak_t = weakref.ref(t)

    # Clear all known global references
    for registry in pnl.primary_registries:
        pnl.clear_registry(registry)

    pnl.core.llvm.LLVMBinaryFunction.get.cache_clear()
    pnl.core.llvm.LLVMBinaryFunction.from_obj.cache_clear()

    gs.utilities.cached_hashable_graph_function.cache_clear()

    # Remove the original references
    del t
    del c

    gc.collect()

    def print_ref(r, depth=0):
        if depth == 3:
            return

        if isinstance(r, (dict, set, list, tuple)):
            for r1 in gc.get_referrers(r):
                print_ref(r1, depth + 1)

    if weak_t() is not None:
        for r in gc.get_referrers(weak_t()):
            print_ref(r)

    if weak_c() is not None:
        for r in gc.get_referrers(weak_c()):
            print_ref(r)

    assert weak_c() is None
    assert weak_t() is None
