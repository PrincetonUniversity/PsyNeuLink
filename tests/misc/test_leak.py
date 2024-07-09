import gc
import numpy as np
import pytest
import weakref

import graph_scheduler as gs
import psyneulink as pnl

show_graph_args = ["show_all", "show_node_structure", "show_cim", "show_learning", "show_types", "show_dimensions",
                   "show_projection_labels", "show_projections_not_in_composition"]

@pytest.mark.composition
@pytest.mark.parametrize("show_graph_args", [pytest.param(None, id="show_graph_disabled"),
                                             pytest.param({}, id="show_graph_default"),
                                             *(pytest.param({arg: True}, id=arg) for arg in show_graph_args),
                                            ])
@pytest.mark.parametrize("op", ["construct", "run"])
def test_composition_leak(comp_mode, op, show_graph_args):

    c = pnl.Composition()
    t = pnl.TransferMechanism()
    c.add_node(t)

    if op == "run":
        res = c.run([5], execution_mode=comp_mode)
        np.testing.assert_array_equal(res, [[5]])

    if show_graph_args is not None:
        c.show_graph(**show_graph_args, output_fmt=None)

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

    assert weak_c() is None, gc.get_referrers(weak_c())
    assert weak_t() is None, gc.get_referrers(weak_t())
